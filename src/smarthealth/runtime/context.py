"""Server-owned identities, per-run budgets and cooperative cancellation."""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from uuid import uuid4

from smarthealth.runtime.guard import CallBudget as CallBudget
from smarthealth.runtime.guard import ensure_active

IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


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

    agent_name: str = "tool_api"

    def __post_init__(self) -> None:
        for value in (
            self.owner_id,
            self.session_id,
            self.run_id,
            self.input_version,
            self.agent_name,
        ):
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                raise ValueError("Context identifiers must be bounded opaque identifiers")
        if not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite")
        object.__setattr__(self, "allowed_skills", frozenset(self.allowed_skills))

    def check_active(self) -> None:
        ensure_active(self.deadline, self.cancelled, self._ancestor_cancellations)

    def child(self, timeout_seconds: float) -> SkillContext:
        return replace(
            self,
            deadline=min(self.deadline, time.monotonic() + timeout_seconds),
            cancelled=threading.Event(),
            _ancestor_cancellations=(*self._ancestor_cancellations, self.cancelled),
        )
