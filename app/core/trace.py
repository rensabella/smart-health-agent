"""Bounded audit events containing no arguments, physiological values or raw outputs."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

from app.core.context import SkillContext
from skills.base_skill import SkillModel


class TraceEvent(SkillModel):
    event: str
    timestamp: str
    run_id: str
    call_id: str
    owner_id: str
    session_id: str
    skill: str
    version: str | None = None
    input_version: str
    input_fields: list[str]
    output_fields: list[str]
    budget_used: int
    latency_ms: float
    error_code: str | None = None
    harness_version: str | None = None
    harness_hash: str | None = None


class InMemoryTraceSink:
    def __init__(self, max_events: int = 5000):
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._events: deque[TraceEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event.model_copy(deep=True))

    def events_for(self, context: SkillContext) -> list[dict]:
        with self._lock:
            return [
                e.model_dump(mode="json")
                for e in self._events
                if (e.run_id, e.owner_id, e.session_id)
                == (context.run_id, context.owner_id, context.session_id)
            ]

    def export(self, context: SkillContext, directory: Path) -> Path:
        """Explicit local export of one authorized run; no raw data is retained."""
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination = directory / f"{context.run_id}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for event in self.events_for(context):
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        destination.chmod(0o600)
        return destination
