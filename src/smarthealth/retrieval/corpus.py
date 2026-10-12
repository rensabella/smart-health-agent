"""Reviewed corpus loading and token-bounded overlapping chunks with exact locators."""

from pathlib import Path

from smarthealth.retrieval.files import read_json, unavailable
from smarthealth.retrieval.schemas import KnowledgeChunk, KnowledgeCorpus, evidence_id, sha256
from smarthealth.runtime.errors import SkillError

CHUNKER_VERSION = "char_token_bound_v1"
MAX_CHUNK_TOKENS = 384
MAX_CHUNK_CHARS = 768
OVERLAP_CHARS = 80
MAX_CHUNKS = 256


def load_corpus(path: Path) -> KnowledgeCorpus:
    try:
        return KnowledgeCorpus.model_validate(read_json(path, max_bytes=512 * 1024), strict=True)
    except Exception:
        raise unavailable("rag_invalid_corpus") from None


def chunk_corpus(corpus: KnowledgeCorpus, embedder) -> tuple[KnowledgeChunk, ...]:
    chunks = []
    for document in corpus.documents:
        start = 0
        per_document = 0
        while start < len(document.content):
            while start < len(document.content) and document.content[start].isspace():
                start += 1
            if start == len(document.content):
                break
            low, high = start + 1, min(start + MAX_CHUNK_CHARS, len(document.content))
            end = start
            while low <= high:
                mid = (low + high) // 2
                count = embedder.token_count(document.content[start:mid], prefix="passage: ")
                if type(count) is not int or count < 1:
                    raise unavailable("rag_embedding_invalid")
                if count <= MAX_CHUNK_TOKENS:
                    end, low = mid, mid + 1
                else:
                    high = mid - 1
            if end <= start:
                raise unavailable("rag_document_too_long")
            # Prefer a sentence boundary near the end, without dropping any text.
            if end < len(document.content):
                floor = start + (end - start) * 3 // 4
                boundaries = [
                    i + 1 for i in range(floor, end) if document.content[i] in "。！？.!?\n"
                ]
                if boundaries:
                    end = boundaries[-1]
            while end > start and document.content[end - 1].isspace():
                end -= 1
            content = document.content[start:end]
            digest = sha256(content)
            locator = f"{document.locator} | chars {start}:{end}"
            chunks.append(
                KnowledgeChunk(
                    evidence_id=evidence_id(
                        document.source.id, locator, digest, corpus.knowledge_version
                    ),
                    source=document.source,
                    content=content,
                    locator=locator,
                    content_hash=digest,
                    knowledge_version=corpus.knowledge_version,
                )
            )
            per_document += 1
            if per_document > 32 or len(chunks) > MAX_CHUNKS:
                raise SkillError(
                    "rag_corpus_too_large", "Knowledge corpus exceeds its chunk budget."
                )
            if end == len(document.content) or not document.content[end:].strip():
                break
            start = max(start + 1, end - OVERLAP_CHARS)
    return tuple(chunks)
