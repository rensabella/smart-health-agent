"""Bounded in-process session memory with expiring, revocable execution leases."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Callable
from uuid import uuid4

from pydantic import Field, TypeAdapter

from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.memory.schemas import (
    ExecutionSummary,
    MemoryIdentifier,
    MemoryRef,
    SessionMemory,
    TurnSummary,
)

_IDENTIFIER = TypeAdapter(MemoryIdentifier)
_UPLOADS = TypeAdapter(Annotated[list[MemoryRef], Field(max_length=1)])
_TERMINAL = frozenset({"completed", "awaiting_input", "blocked", "failed", "cancelled"})


@dataclass
class _Entry:
    memory: SessionMemory
    updated_at: float
    revision: int
    lease: str | None = None
    active_run_id: str | None = None


class LocalShortTermStore:
    """Store summaries only; returned references do not grant artifact access.

    All methods are synchronous and guarded by one reentrant lock. A lease is
    valid for exactly one active run and cannot recreate an expired or deleted
    session. Reads do not renew retention, including reads of active sessions.
    """

    def __init__(
        self,
        ttl_seconds: float = 3600,
        max_sessions: int = 128,
        clock: Callable[[], float] = time.time,
    ):
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("Memory TTL must be a positive finite number.")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("Memory capacity must be a positive integer.")
        self._ttl_seconds = float(ttl_seconds)
        self._max_sessions = max_sessions
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._revision = 0

    def begin(
        self,
        owner_id: str,
        session_id: str,
        input_version: str,
        run_id: str,
        upload_refs: list[str],
    ) -> str:
        key = self._key(owner_id, session_id)
        version = _IDENTIFIER.validate_python(input_version, strict=True)
        run = _IDENTIFIER.validate_python(run_id, strict=True)
        refs = _UPLOADS.validate_python(upload_refs, strict=True)
        with self._lock:
            now = self._now()
            self._cleanup(now)
            entry = self._entries.get(key)
            if entry is not None and entry.active_run_id is not None:
                if entry.active_run_id != run:
                    raise SkillError("session_busy", "The session already has an active run.")
                if entry.memory.input_version != version or entry.memory.upload_refs != refs:
                    raise SkillError(
                        "memory_state_conflict", "An active run's input cannot change."
                    )
                assert entry.lease is not None
                return entry.lease
            if entry is None:
                entry = self._new_entry(key, version, now)
            self._set_input(entry, version, refs)
            entry.memory.execution = ExecutionSummary(run_id=run, status="queued")
            entry.active_run_id = run
            entry.lease = uuid4().hex
            self._touch(entry, now)
            return entry.lease

    def attach_upload(
        self, owner_id: str, session_id: str, input_version: str, upload_ref: str
    ) -> None:
        key = self._key(owner_id, session_id)
        version = _IDENTIFIER.validate_python(input_version, strict=True)
        refs = _UPLOADS.validate_python([upload_ref], strict=True)
        with self._lock:
            now = self._now()
            self._cleanup(now)
            entry = self._entries.get(key)
            if entry is not None and entry.active_run_id is not None:
                raise SkillError("session_busy", "The session already has an active run.")
            if entry is None:
                entry = self._new_entry(key, version, now)
            self._set_input(entry, version, refs)
            self._touch(entry, now)

    def update(
        self, owner_id: str, session_id: str, lease: str, execution: ExecutionSummary
    ) -> bool:
        key = self._key(owner_id, session_id)
        value = ExecutionSummary.model_validate(execution.model_dump())
        with self._lock:
            now = self._now()
            entry = self._leased_entry(key, lease, now)
            if entry is None or entry.active_run_id != value.run_id:
                return False
            previous = entry.memory.execution
            if previous is not None and (
                value.last_event_id < previous.last_event_id
                or value.decision_count < previous.decision_count
                or value.tool_calls < previous.tool_calls
                or (previous.status in _TERMINAL and value.status != previous.status)
            ):
                return False
            entry.memory.execution = value
            self._touch(entry, now)
            return True

    def finish(
        self,
        owner_id: str,
        session_id: str,
        lease: str,
        turn: TurnSummary,
        result_run_id: str | None = None,
    ) -> bool:
        key = self._key(owner_id, session_id)
        value = TurnSummary.model_validate(turn.model_dump())
        result_id = (
            None
            if result_run_id is None
            else _IDENTIFIER.validate_python(result_run_id, strict=True)
        )
        with self._lock:
            now = self._now()
            entry = self._leased_entry(key, lease, now)
            if (
                entry is None
                or entry.active_run_id != value.run_id
                or entry.memory.input_version != value.input_version
                or value.assistant_status not in _TERMINAL
                or (value.analysis is not None and value.assistant_status != "completed")
                or (result_id is not None and result_id != value.run_id)
            ):
                return False
            if (
                result_id is not None
                and value.assistant_status == "completed"
                and value.user_intent == "data_analysis"
                and value.analysis is not None
            ):
                entry.memory.current_result_run_id = result_id
                entry.memory.prediction = value.analysis.model_copy(deep=True)
            entry.memory.turns = [*entry.memory.turns, value][-5:]
            previous = entry.memory.execution
            entry.memory.execution = ExecutionSummary(
                run_id=value.run_id,
                status=value.assistant_status,
                last_event_id=previous.last_event_id if previous is not None else 0,
                decision_count=previous.decision_count if previous is not None else 0,
                tool_calls=previous.tool_calls if previous is not None else 0,
            )
            entry.lease = None
            entry.active_run_id = None
            self._touch(entry, now)
            return True

    def get(self, owner_id: str, session_id: str) -> SessionMemory | None:
        key = self._key(owner_id, session_id)
        with self._lock:
            self._cleanup(self._now())
            entry = self._entries.get(key)
            return None if entry is None else entry.memory.model_copy(deep=True)

    def delete_session(self, owner_id: str, session_id: str) -> None:
        key = self._key(owner_id, session_id)
        with self._lock:
            self._entries.pop(key, None)

    def delete_owner(self, owner_id: str) -> None:
        owner = _IDENTIFIER.validate_python(owner_id, strict=True)
        with self._lock:
            for key in [key for key in self._entries if key[0] == owner]:
                del self._entries[key]

    def cleanup(self) -> int:
        with self._lock:
            return self._cleanup(self._now())

    @staticmethod
    def _key(owner_id: str, session_id: str) -> tuple[str, str]:
        return (
            _IDENTIFIER.validate_python(owner_id, strict=True),
            _IDENTIFIER.validate_python(session_id, strict=True),
        )

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now) or now < 0 or not math.isfinite(now + self._ttl_seconds):
            raise ValueError("The memory clock must return a finite nonnegative timestamp.")
        return now

    def _cleanup(self, now: float) -> int:
        expired = [key for key, entry in self._entries.items() if entry.memory.expires_at <= now]
        for key in expired:
            del self._entries[key]
        return len(expired)

    def _leased_entry(self, key: tuple[str, str], lease: str, now: float) -> _Entry | None:
        self._cleanup(now)
        entry = self._entries.get(key)
        if entry is None or entry.lease is None or entry.lease != lease:
            return None
        return entry

    def _new_entry(self, key: tuple[str, str], version: str, now: float) -> _Entry:
        if len(self._entries) >= self._max_sessions:
            inactive = [
                (candidate_key, entry)
                for candidate_key, entry in self._entries.items()
                if entry.active_run_id is None
            ]
            if not inactive:
                raise SkillError(
                    "memory_capacity_exceeded", "All retained sessions have active runs."
                )
            oldest_key, _ = min(inactive, key=lambda item: (item[1].updated_at, item[1].revision))
            del self._entries[oldest_key]
        entry = _Entry(
            memory=SessionMemory(
                session_id=key[1], input_version=version, expires_at=now + self._ttl_seconds
            ),
            updated_at=now,
            revision=self._revision,
        )
        self._entries[key] = entry
        return entry

    @staticmethod
    def _set_input(entry: _Entry, version: str, refs: list[str]) -> None:
        if entry.memory.input_version != version:
            entry.memory.current_result_run_id = None
            entry.memory.prediction = None
            entry.memory.execution = None
        entry.memory.input_version = version
        entry.memory.upload_refs = list(refs)

    def _touch(self, entry: _Entry, now: float) -> None:
        self._revision += 1
        entry.updated_at = now
        entry.revision = self._revision
        entry.memory.expires_at = now + self._ttl_seconds
