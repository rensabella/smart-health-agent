"""Server-owned identities, per-run budgets and cooperative cancellation."""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from uuid import uuid4

from app.core.errors import SkillError

IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class CallBudget:
    """A budget shared by a run's calls, including failures and retries."""

    def __init__(self, max_calls: int = 18):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a nonnegative integer")
        self.max_calls = max_calls
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def reserve(self) -> int:
        with self._lock:
            if self._used >= self.max_calls:
                raise SkillError("budget_exceeded", "The run's Skill call budget is exhausted.")
            self._used += 1
            return self._used


@dataclass(frozen=True)
class SkillContext:
    """Injected by the service, never parsed from LLM-provided tool arguments."""

    owner_id: str
    session_id: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    input_version: str = field(default_factory=lambda: uuid4().hex)
    allowed_skills: frozenset[str] = field(default_factory=frozenset)
    deadline: float = field(default_factory=lambda: time.monotonic() + 180.0)
    budget: CallBudget = field(default_factory=CallBudget)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    _ancestor_cancellations: tuple[threading.Event, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        for value in (self.owner_id, self.session_id, self.run_id, self.input_version):
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                raise ValueError("Context identifiers must be bounded opaque identifiers")
        if not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite")
        object.__setattr__(self, "allowed_skills", frozenset(self.allowed_skills))

    def check_active(self) -> None:
        if self.cancelled.is_set() or any(e.is_set() for e in self._ancestor_cancellations):
            raise SkillError("cancelled", "This Skill execution has been cancelled.")
        if time.monotonic() >= self.deadline:
            raise SkillError("timeout", "The execution deadline has elapsed.")

    def child(self, timeout_seconds: float) -> SkillContext:
        return replace(
            self,
            deadline=min(self.deadline, time.monotonic() + timeout_seconds),
            cancelled=threading.Event(),
            _ancestor_cancellations=(*self._ancestor_cancellations, self.cancelled),
        )
