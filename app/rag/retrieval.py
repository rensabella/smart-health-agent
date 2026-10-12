"""Immutable offline-built Qdrant index with verified payloads and local semantic queries."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from pydantic import Field
from qdrant_client import QdrantClient, models

from app.rag.corpus import CHUNKER_VERSION, MAX_CHUNKS, chunk_corpus, load_corpus
from app.rag.embeddings import MAX_BATCH, E5ONNXEmbedder, normalized_vectors
from app.rag.files import checked_path, harden_directory, read_json, unavailable
from app.rag.schemas import (
    Digest,
    EmbeddingId,
    Evidence,
    FrozenRagModel,
    KnowledgeChunk,
    RetrievalResult,
    Version,
    canonical_hash,
)

COLLECTION = "smart_health_knowledge_v1"


class IndexManifest(FrozenRagModel):
    schema_version: str
    knowledge_version: Version
    embedding_id: EmbeddingId
    dimensions: int = Field(ge=1, le=4096)
    chunker_version: str
    corpus_hash: Digest
    chunks_hash: Digest
    vectors_hash: Digest
    chunk_count: int = Field(ge=0, le=MAX_CHUNKS)


def _point_id(chunk):
    return str(uuid5(NAMESPACE_URL, chunk.evidence_id))


def build_index(corpus_path: Path, directory: Path, embedder) -> IndexManifest:
    """Create a new index only. No replacing a live database or partially updating its release."""
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise unavailable("rag_index_exists")
    corpus = load_corpus(corpus_path)
    chunks = chunk_corpus(corpus, embedder)
    # Exact duplicate contents are indexed once; corpus provenance stays in the manifest hash.
    unique = {c.content_hash: c for c in reversed(chunks)}
    chunks = tuple(sorted(unique.values(), key=lambda c: c.evidence_id))
    vectors = []
    for start in range(0, len(chunks), MAX_BATCH):
        batch = chunks[start : start + MAX_BATCH]
        vectors.extend(
            normalized_vectors(
                embedder.encode_passages([c.content for c in batch]),
                len(batch),
                embedder.dimensions,
            ).tolist()
        )
    manifest = IndexManifest(
        schema_version="1.0",
        knowledge_version=corpus.knowledge_version,
        embedding_id=embedder.embedding_id,
        dimensions=embedder.dimensions,
        chunker_version=CHUNKER_VERSION,
        corpus_hash=canonical_hash(corpus.model_dump(mode="json")),
        chunks_hash=canonical_hash([c.model_dump(mode="json") for c in chunks]),
        vectors_hash=canonical_hash(vectors),
        chunk_count=len(chunks),
    )
    directory = checked_path(directory, directory=True, create=True)
    client = QdrantClient(path=str(directory / "qdrant"), force_disable_check_same_thread=True)
    try:
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=embedder.dimensions, distance=models.Distance.COSINE
            ),
        )
        if chunks:
            client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(id=_point_id(c), vector=v, payload=c.model_dump(mode="json"))
                    for c, v in zip(chunks, vectors)
                ],
                wait=True,
            )
        # Qdrant local mode normalizes vectors again when loading persisted points.
        # Fingerprint the reopened representation that runtime actually queries.
        client.close()
        client = QdrantClient(path=str(directory / "qdrant"), force_disable_check_same_thread=True)
        stored_points, _ = client.scroll(COLLECTION, limit=MAX_CHUNKS + 1, with_vectors=True)
        manifest = manifest.model_copy(
            update={
                "vectors_hash": canonical_hash(
                    [p.vector for p in sorted(stored_points, key=lambda p: str(p.id))]
                )
            }
        )
        # Completion marker written last; readers refuse a directory without it.
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], ensure_ascii=False)
        )
        (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    finally:
        client.close()
        harden_directory(directory)
    return manifest


class RetrievalService:
    def __init__(self, directory: Path, embedder, *, min_score=0.78):
        self._lock = threading.RLock()
        self._client = None
        self._embedder = embedder
        self._closed = False
        if (
            type(min_score) not in (int, float)
            or not math.isfinite(min_score)
            or not -1 <= min_score <= 1
        ):
            raise ValueError("Retrieval threshold must be a finite cosine score")
        self.min_score = float(min_score)
        try:
            directory = checked_path(directory, directory=True)
            harden_directory(directory)
            self.manifest = IndexManifest.model_validate(
                read_json(directory / "manifest.json", max_bytes=8192), strict=True
            )
            m = self.manifest
            if (m.schema_version, m.chunker_version, m.embedding_id, m.dimensions) != (
                "1.0",
                CHUNKER_VERSION,
                embedder.embedding_id,
                embedder.dimensions,
            ):
                raise unavailable("rag_index_incompatible")
            data = read_json(directory / "chunks.json", max_bytes=2 * 1024 * 1024)
            if not isinstance(data, list) or len(data) != m.chunk_count:
                raise unavailable("rag_index_invalid")
            chunks = tuple(KnowledgeChunk.model_validate(c, strict=True) for c in data)
            if canonical_hash(data) != m.chunks_hash or len({c.evidence_id for c in chunks}) != len(
                chunks
            ):
                raise unavailable("rag_index_invalid")
            if any(c.knowledge_version != m.knowledge_version for c in chunks):
                raise unavailable("rag_index_invalid")
            self._chunks = {_point_id(c): c for c in chunks}
            self._client = QdrantClient(
                path=str(directory / "qdrant"), force_disable_check_same_thread=True
            )
            info = self._client.get_collection(COLLECTION)
            params = info.config.params.vectors
            if (
                isinstance(params, dict)
                or params.size != m.dimensions
                or params.distance != models.Distance.COSINE
                or info.points_count != len(chunks)
            ):
                raise unavailable("rag_index_incompatible")
            points, cursor = self._client.scroll(
                COLLECTION, limit=MAX_CHUNKS + 1, with_payload=True, with_vectors=True
            )
            if cursor is not None or len(points) != len(chunks):
                raise unavailable("rag_index_invalid")
            for p in points:
                if str(p.id) not in self._chunks or p.payload != self._chunks[str(p.id)].model_dump(
                    mode="json"
                ):
                    raise unavailable("rag_index_invalid")
                vector = np.asarray(p.vector)
                if (
                    vector.shape != (m.dimensions,)
                    or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)
                ):
                    raise unavailable("rag_index_invalid")
            if (
                canonical_hash([p.vector for p in sorted(points, key=lambda p: str(p.id))])
                != m.vectors_hash
            ):
                raise unavailable("rag_index_invalid")
            # Store keeps immutable, validated payloads; queries recheck every returned point.
        except Exception:
            if self._client is not None:
                self._client.close()
            raise unavailable("rag_index_invalid") from None

    @property
    def knowledge_version(self):
        return self.manifest.knowledge_version

    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 1000
            or type(top_k) is not int
            or not 1 <= top_k <= 6
        ):
            raise unavailable("rag_invalid_query")
        with self._lock:
            if self._closed:
                raise unavailable()
            try:
                vectors = normalized_vectors(
                    self._embedder.encode_queries([query]), 1, self.manifest.dimensions
                )
                points = self._client.query_points(
                    COLLECTION,
                    query=vectors[0].tolist(),
                    limit=top_k,
                    score_threshold=self.min_score,
                    with_payload=True,
                ).points
                hits = []
                for p in points:
                    chunk = self._chunks.get(str(p.id))
                    if chunk is None or p.payload != chunk.model_dump(mode="json"):
                        raise unavailable("rag_index_invalid")
                    score = float(p.score)
                    if not math.isfinite(score) or score < -1.000001 or score > 1.000001:
                        raise unavailable("rag_index_invalid")
                    hits.append(Evidence(**chunk.model_dump(), score=max(-1.0, min(1.0, score))))
                return RetrievalResult(
                    status="success" if hits else "insufficient_evidence",
                    hits=tuple(hits),
                    knowledge_version=self.knowledge_version,
                    embedding_id=self.manifest.embedding_id,
                )
            except Exception as exc:
                from app.core.errors import SkillError

                if isinstance(exc, SkillError):
                    raise
                raise unavailable() from None

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._client.close()
                self._embedder.close()


def load_retrieval_from_env():
    mode = os.getenv("HEALTH_RAG_MODE", "disabled")
    if mode == "disabled":
        return None
    if mode != "local":
        raise ValueError("HEALTH_RAG_MODE must be disabled or local")
    model_dir, index_dir = os.getenv("HEALTH_RAG_MODEL_DIR"), os.getenv("HEALTH_RAG_INDEX_DIR")
    if not model_dir or not index_dir:
        raise ValueError("Local RAG requires explicit model and index directories")
    embedder = E5ONNXEmbedder(Path(model_dir))
    try:
        return RetrievalService(
            Path(index_dir), embedder, min_score=float(os.getenv("HEALTH_RAG_MIN_SCORE", "0.78"))
        )
    except Exception:
        embedder.close()
        raise
