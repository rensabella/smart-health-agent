"""Bounded, copy-isolated temporary artifacts. No raw data is written to disk."""

from __future__ import annotations

import copy
import pickle
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.errors import SkillError

_REF = re.compile(r"^art_[0-9a-f]{32}$")


@dataclass(frozen=True)
class _Artifact:
    value: Any
    kind: str
    owner_id: str
    session_id: str
    input_version: str
    expires_at: float
    size_bytes: int
    source_ref: str | None
    schema_version: str


class ArtifactStore:
    """Internal values are trusted Python objects; public callers only receive opaque refs."""

    def __init__(self, ttl_seconds=3600, max_bytes=67108864, max_items=256):
        if ttl_seconds <= 0 or max_bytes <= 0 or max_items <= 0:
            raise ValueError("Artifact bounds must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.max_items = max_items
        self._items: dict[str, _Artifact] = {}
        self._bytes = 0
        self._lock = threading.RLock()

    def _remove(self, ref: str) -> None:
        item = self._items.pop(ref)
        self._bytes -= item.size_bytes

    def cleanup(self) -> int:
        with self._lock:
            expired = [r for r, a in self._items.items() if a.expires_at <= time.monotonic()]
            for ref in expired:
                self._remove(ref)
            return len(expired)

    def _get(self, ref: str, context: SkillContext, kind: str | None = None) -> _Artifact:
        if not isinstance(ref, str) or not _REF.fullmatch(ref):
            raise SkillError("invalid_input", "An opaque artifact reference is required.")
        item = self._items.get(ref)
        if item is None:
            raise SkillError("artifact_not_found", "The artifact is unavailable or expired.")
        if (item.owner_id, item.session_id) != (context.owner_id, context.session_id):
            raise SkillError("permission_denied", "The artifact is outside this session.")
        if item.expires_at <= time.monotonic():
            self._remove(ref)
            raise SkillError("artifact_expired", "The artifact has expired; upload again.")
        if item.input_version != context.input_version:
            raise SkillError("stale_input", "The artifact belongs to another input version.")
        if kind is not None and item.kind != kind:
            raise SkillError("invalid_artifact_type", "The artifact has an incompatible type.")
        return item

    def put(self, value, *, kind, context, source_ref=None, schema_version="1.0") -> str:
        context.check_active()
        snapshot = copy.deepcopy(value)
        # Serialize only internal trusted objects to estimate capacity; never load client pickle.
        size_bytes = len(pickle.dumps(snapshot, protocol=5)) + 512
        if size_bytes > self.max_bytes:
            raise SkillError("artifact_capacity_exceeded", "The artifact exceeds store capacity.")
        with self._lock:
            self.cleanup()
            context.check_active()
            if source_ref is not None:
                self._get(source_ref, context)
            if len(self._items) >= self.max_items or self._bytes + size_bytes > self.max_bytes:
                raise SkillError("artifact_capacity_exceeded", "Temporary storage is full.")
            ref = "art_" + uuid4().hex
            self._items[ref] = _Artifact(
                snapshot,
                kind,
                context.owner_id,
                context.session_id,
                context.input_version,
                time.monotonic() + self.ttl_seconds,
                size_bytes,
                source_ref,
                schema_version,
            )
            self._bytes += size_bytes
            return ref

    def get(self, ref, *, kind, context):
        context.check_active()
        with self._lock:
            return copy.deepcopy(self._get(ref, context, kind).value)

    def check_source(self, ref, *, source_ref, kind, context):
        """Verify lineage metadata without exposing private stored values or trusting payloads."""
        context.check_active()
        with self._lock:
            item = self._get(ref, context, kind)
            self._get(source_ref, context)
            if item.source_ref != source_ref:
                raise SkillError("harness_invalid_output", "Artifact source does not match input.")

    def delete(self, ref, *, context) -> None:
        with self._lock:
            self._get(ref, context)
            self._remove(ref)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    @property
    def item_count(self) -> int:
        with self._lock:
            return len(self._items)
