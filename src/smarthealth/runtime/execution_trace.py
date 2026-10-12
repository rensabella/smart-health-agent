"""Bounded execution spans containing identifiers and safe outcomes, never task text."""

from __future__ import annotations

import asyncio
import re
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from functools import wraps
from typing import Literal
from uuid import uuid4

from pydantic import Field

from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.logging import log_event
from smarthealth.skills.base_skill import SkillModel


def now():
    return datetime.now(timezone.utc).isoformat()


class ExecutionStep(SkillModel):
    task_id: str
    step_id: str
    agent_name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    skill_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    operation: Literal["task", "agent", "skill"]
    start_time: str
    end_time: str | None = None
    status: Literal["running", "completed", "failed", "cancelled", "blocked", "awaiting_input"]
    error_code: str | None = None
    error_message: str | None = None


class ExecutionTrace:
    def __init__(self, task_id: str, max_steps: int = 128):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
            raise ValueError("task_id must be a bounded opaque identifier")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.task_id, self.max_steps = task_id, max_steps
        self._steps: OrderedDict[str, ExecutionStep] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, agent_name, *, skill_name=None, operation="agent") -> str:
        with self._lock:
            if len(self._steps) >= self.max_steps:
                expired = next((key for key, step in self._steps.items() if step.end_time), None)
                if expired is None:
                    raise SkillError(
                        "trace_capacity_exceeded", "Execution trace capacity exhausted."
                    )
                del self._steps[expired]
            step = ExecutionStep(
                task_id=self.task_id,
                step_id=uuid4().hex,
                agent_name=agent_name,
                skill_name=skill_name,
                operation=operation,
                start_time=now(),
                status="running",
            )
            self._steps[step.step_id] = step
            log_event("execution.started", **step.model_dump())
            return step.step_id

    def finish(self, step_id, *, status="completed", error_code=None):
        with self._lock:
            step = self._steps[step_id]
            if step.end_time is not None:
                return step.model_copy(deep=True)
            if error_code is not None and not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code):
                error_code = "internal_error"
            payload = step.model_dump()
            payload.update(
                end_time=now(),
                status=status,
                error_code=error_code,
                error_message="Execution did not complete successfully." if error_code else None,
            )
            completed = ExecutionStep.model_validate(payload)
            if completed.status == "running":
                raise ValueError("Cannot finish a step with running status")
            self._steps[step_id] = completed
            log_event("execution.finished", **completed.model_dump())
            return completed.model_copy(deep=True)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [step.model_dump(mode="json") for step in self._steps.values()]


def traced_execution(operation):
    """Record actual task/role invocations, including raised errors and cancellation."""

    def decorate(method):
        @wraps(method)
        async def wrapped(self, record, *args, **kwargs):
            role = args[0] if operation == "agent" else record.request.entry_agent
            step_id = record.execution_trace.start(role, operation=operation)
            status, code = "completed", None
            try:
                result = await method(self, record, *args, **kwargs)
                if record.status in {"failed", "cancelled", "blocked", "awaiting_input"}:
                    status, code = record.status, record.error_code
                return result
            except asyncio.CancelledError:
                status, code = "cancelled", "cancelled"
                raise
            except Exception as exc:
                status = "failed"
                code = exc.code if isinstance(exc, SkillError) else "internal_error"
                raise
            finally:
                record.execution_trace.finish(step_id, status=status, error_code=code)

        return wrapped

    return decorate
