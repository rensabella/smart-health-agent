"""Scoped summary storage in real local Qdrant with deterministic lexical vectors.

``char_hash_256_v1`` is a character n-gram hashing baseline, not a pretrained
semantic embedding. Only allowlisted facts are embedded; search text is transient.
The local client is serialized for worker-thread use and supports one process per
persistent directory. An unavailable or incompatible database fails closed.
"""

from __future__ import annotations

import math
import os
import stat
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, TypeAdapter, ValidationError, model_validator
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import HashingVectorizer

from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.memory.schemas import (
    LongTermEntry,
    MemoryHit,
    MemoryIdentifier,
    Preferences,
    render_entry,
)
from smarthealth.skills.base_skill import SkillModel

EMBEDDING_VERSION = "char_hash_256_v1"
COLLECTION_NAME = "smart_health_memory_char_hash_256_v1"
_OWNER = TypeAdapter(MemoryIdentifier)


class _Payload(SkillModel):
    owner_id: MemoryIdentifier
    embedding_version: Literal["char_hash_256_v1"] = EMBEDDING_VERSION
    kind: Literal["analysis", "preferences"]
    expires_at: Annotated[float, Field(gt=0)]
    entry: LongTermEntry

    @model_validator(mode="after")
    def consistent(self):
        if self.kind != self.entry.kind or self.expires_at != self.entry.expires_at:
            raise ValueError("Memory metadata disagrees with entry")
        return self


def _unavailable() -> SkillError:
    return SkillError("memory_unavailable", "Long-term memory storage is unavailable")


def _invalid() -> SkillError:
    return SkillError("invalid_memory", "Memory input is invalid")


class QdrantLongTermStore:
    """Owner-filtered, expiring summaries; callers cannot supply raw narratives."""

    def __init__(
        self,
        path: Path | None = None,
        retention_seconds: float = 2592000,
        max_entries_per_owner: int = 512,
        clock: Callable[[], float] = time.time,
    ):
        if (
            type(retention_seconds) not in (int, float)
            or not math.isfinite(retention_seconds)
            or retention_seconds <= 0
            or type(max_entries_per_owner) is not int
            or not 1 <= max_entries_per_owner <= 512
            or not callable(clock)
        ):
            raise ValueError("Memory limits and clock must be valid")
        self.path = Path(path).absolute() if path is not None else None
        self.retention_seconds = float(retention_seconds)
        self.max_entries_per_owner = max_entries_per_owner
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._client: QdrantClient | None = None
        self._vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=256,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        try:
            self._secure_storage(create=True)
            self._client = (
                QdrantClient(":memory:", force_disable_check_same_thread=True)
                if self.path is None
                else QdrantClient(path=str(self.path), force_disable_check_same_thread=True)
            )
            if not self._client.collection_exists(COLLECTION_NAME):
                self._client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=256, distance=models.Distance.COSINE),
                )
            vectors = self._client.get_collection(COLLECTION_NAME).config.params.vectors
            if (
                not isinstance(vectors, models.VectorParams)
                or vectors.size != 256
                or vectors.distance != models.Distance.COSINE
            ):
                raise _unavailable()
            self._secure_storage()
        except Exception:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._closed = True
            raise _unavailable() from None

    def put_analysis(self, owner_id: str, entry: LongTermEntry) -> bool:
        with self._operation():
            owner_id = self._owner(owner_id)
            now = self._now()
            entry = self._entry(entry, "analysis", now)
            self._cleanup(now)
            previous = self._get_entry(owner_id, entry.kind, entry.source_run_id, now)
            if previous is not None:
                if previous != entry:
                    raise SkillError("memory_conflict", "This run already has a different summary")
                return False
            self._check_capacity(owner_id, now)
            self._put(owner_id, entry)
            return True

    def set_preferences(self, owner_id: str, entry: LongTermEntry) -> None:
        with self._operation():
            owner_id = self._owner(owner_id)
            now = self._now()
            entry = self._entry(entry, "preferences", now)
            self._cleanup(now)
            if self._get_entry(owner_id, "preferences", None, now) is None:
                self._check_capacity(owner_id, now)
            self._put(owner_id, entry)

    def search(self, owner_id: str, query: str, limit: int = 3) -> list[MemoryHit]:
        with self._operation():
            owner_id = self._owner(owner_id)
            self._limit(limit, 10)
            if not isinstance(query, str) or not query.strip() or len(query) > 1000:
                raise _invalid()
            now = self._now()
            result = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=self._embed(query),
                query_filter=self._filter(owner_id, now, "analysis"),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            hits = []
            for point in result.points:
                entry = self._decode(point, owner_id, now, "analysis")
                if not math.isfinite(point.score):
                    raise _unavailable()
                hits.append(MemoryHit(entry=entry, score=max(-1.0, min(1.0, point.score))))
            return hits

    def list_analysis(self, owner_id: str, limit: int = 100) -> list[LongTermEntry]:
        with self._operation():
            owner_id = self._owner(owner_id)
            self._limit(limit, 512)
            now = self._now()
            entries = list(self._scroll(owner_id, now, "analysis"))
            entries.sort(key=lambda entry: (entry.created_at, entry.source_run_id))
            return entries[-limit:]

    def get_preferences(self, owner_id: str) -> Preferences | None:
        with self._operation():
            owner_id = self._owner(owner_id)
            entry = self._get_entry(owner_id, "preferences", None, self._now())
            return entry.preferences.model_copy(deep=True) if entry is not None else None

    def delete_owner(self, owner_id: str) -> None:
        with self._operation():
            owner_id = self._owner(owner_id)
            now = self._now()
            # Clear expired points first so owner deletion also removes old data.
            self._cleanup(now)
            self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(filter=self._filter(owner_id, now)),
                wait=True,
            )

    def cleanup(self) -> None:
        with self._operation():
            self._cleanup(self._now())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._secure_storage()
                self._client.close()
                self._secure_storage()
            except Exception:
                raise _unavailable() from None
            finally:
                self._closed = True

    @contextmanager
    def _operation(self):
        with self._lock:
            if self._closed:
                raise _unavailable()
            try:
                self._secure_storage()
                yield
                self._secure_storage()
            except SkillError:
                raise
            except Exception:
                raise _unavailable() from None

    @staticmethod
    def _owner(owner_id: str) -> str:
        try:
            return _OWNER.validate_python(owner_id)
        except (TypeError, ValueError, ValidationError):
            raise _invalid() from None

    @staticmethod
    def _limit(limit: int, maximum: int) -> None:
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise _invalid()

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise _unavailable()
        return float(value)

    def _entry(self, entry: LongTermEntry, kind: str, now: float) -> LongTermEntry:
        try:
            if not isinstance(entry, LongTermEntry):
                raise ValueError("Expected typed memory")
            validated = LongTermEntry.model_validate(entry.model_dump())
            if (
                validated.kind != kind
                or validated.created_at > now
                or validated.expires_at <= now
                or validated.expires_at - validated.created_at > self.retention_seconds + 1e-6
            ):
                raise ValueError("Invalid memory retention")
            return validated
        except (TypeError, ValueError, ValidationError):
            raise _invalid() from None

    @staticmethod
    def _point_id(owner_id: str, kind: str, source_run_id: str | None) -> str:
        return str(
            uuid5(
                NAMESPACE_URL, f"smarthealth-memory:{owner_id}:{kind}:{source_run_id or 'current'}"
            )
        )

    @staticmethod
    def _filter(owner_id: str, now: float, kind: str | None = None) -> models.Filter:
        conditions = [
            models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id)),
            models.FieldCondition(key="expires_at", range=models.Range(gt=now)),
        ]
        if kind is not None:
            conditions.append(
                models.FieldCondition(key="kind", match=models.MatchValue(value=kind))
            )
        return models.Filter(must=conditions)

    def _embed(self, text: str) -> list[float]:
        return self._vectorizer.transform([text]).toarray()[0].tolist()

    def _put(self, owner_id: str, entry: LongTermEntry) -> None:
        payload = _Payload(
            owner_id=owner_id, kind=entry.kind, expires_at=entry.expires_at, entry=entry
        )
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=self._point_id(owner_id, entry.kind, entry.source_run_id),
                    vector=self._embed(render_entry(entry)),
                    payload=payload.model_dump(mode="json"),
                )
            ],
            wait=True,
        )

    def _decode(self, point, owner_id: str, now: float, kind: str | None = None) -> LongTermEntry:
        try:
            payload = _Payload.model_validate(point.payload)
            if (
                payload.owner_id != owner_id
                or payload.expires_at <= now
                or payload.entry.created_at > now
                or payload.expires_at - payload.entry.created_at > self.retention_seconds + 1e-6
                or kind is not None
                and payload.kind != kind
                or str(point.id)
                != self._point_id(owner_id, payload.kind, payload.entry.source_run_id)
            ):
                raise ValueError("Invalid stored memory")
            return payload.entry
        except (TypeError, ValueError, ValidationError):
            raise _unavailable() from None

    def _get_entry(
        self, owner_id: str, kind: str, run_id: str | None, now: float
    ) -> LongTermEntry | None:
        point_id = self._point_id(owner_id, kind, run_id)
        query_filter = self._filter(owner_id, now, kind)
        query_filter.must.append(models.HasIdCondition(has_id=[point_id]))
        points, _ = self._client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return self._decode(points[0], owner_id, now, kind) if points else None

    def _scroll(self, owner_id: str, now: float, kind: str) -> Iterator[LongTermEntry]:
        offset = None
        seen = 0
        while True:
            points, offset = self._client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=self._filter(owner_id, now, kind),
                limit=64,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                seen += 1
                if seen > self.max_entries_per_owner:
                    raise _unavailable()
                yield self._decode(point, owner_id, now, kind)
            if offset is None:
                return

    def _check_capacity(self, owner_id: str, now: float) -> None:
        count = self._client.count(
            collection_name=COLLECTION_NAME,
            count_filter=self._filter(owner_id, now),
            exact=True,
        ).count
        if count >= self.max_entries_per_owner:
            raise SkillError("memory_capacity_exceeded", "Owner memory capacity is full")

    def _cleanup(self, now: float) -> None:
        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="expires_at", range=models.Range(lte=now))]
                )
            ),
            wait=True,
        )

    def _secure_storage(self, *, create: bool = False) -> None:
        if self.path is None:
            return
        for parent in (self.path, *self.path.parents):
            if parent.is_symlink():
                raise _unavailable()
        if create:
            self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.path.is_dir():
            raise _unavailable()
        self.path.chmod(0o700)
        for directory, directories, files in os.walk(self.path, followlinks=False):
            for name in directories + files:
                target = Path(directory) / name
                mode = target.lstat().st_mode
                if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                    raise _unavailable()
                target.chmod(0o700 if stat.S_ISDIR(mode) else 0o600)
