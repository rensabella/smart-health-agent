"""Atomic execution budgets and cooperative deadline/cancellation checks."""

import threading
import time

from smarthealth.runtime.errors import SkillError


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


def ensure_active(deadline, cancelled, ancestor_cancellations=()):
    if cancelled.is_set() or any(event.is_set() for event in ancestor_cancellations):
        raise SkillError("cancelled", "This Skill execution has been cancelled.")
    if time.monotonic() >= deadline:
        raise SkillError("timeout", "The execution deadline has elapsed.")
