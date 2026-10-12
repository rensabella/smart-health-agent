"""Typed dispatch with authorization, atomic budgets, bounded workers and audit events."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.trace import InMemoryTraceSink, TraceEvent
from smarthealth.skills.base_skill import BaseSkill, SkillFailure, SkillModel, SkillResult
from smarthealth.skills.validation import validate_input, validate_output

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SkillRegistry:
    """Instances belong to one application/event loop. Registration is explicit and trusted."""

    def __init__(self, max_concurrency: int = 4, trace_sink=None):
        if isinstance(max_concurrency, bool) or max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._skills: dict[str, BaseSkill] = {}
        self.trace_sink = trace_sink if trace_sink is not None else InMemoryTraceSink()
        self._slots = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="skill")
        self._tasks: set[asyncio.Task] = set()
        self._closed = False
        self._harness = None

    @property
    def harness(self):
        return self._harness

    def bind_harness(self, harness):
        """Bind once before dispatch. A request cannot replace policy or register more code."""
        if self._harness is not None:
            if self._harness.fingerprint != harness.fingerprint:
                raise ValueError("Registry already belongs to another Harness snapshot")
            return
        harness.validate_catalogue(self.describe())
        self._harness = harness

    def register(self, skill: BaseSkill) -> None:
        if self._harness is not None:
            raise ValueError("The configured Skill catalogue is frozen")
        if self._closed:
            raise ValueError("The Registry is closed")
        if not isinstance(skill, BaseSkill) or not _NAME.fullmatch(skill.name):
            raise ValueError("A Skill must have a valid explicit name")
        if skill.name in self._skills:
            raise ValueError(f"Duplicate Skill registration: {skill.name}")
        for model in (skill.input_schema, skill.output_schema):
            if not isinstance(model, type) or not issubclass(model, SkillModel):
                raise ValueError("Skill input/output schemas must inherit SkillModel")
            model.model_json_schema()
        if not skill.description or not skill.version:
            raise ValueError("A Skill requires description and version")
        if not math.isfinite(skill.timeout_seconds) or skill.timeout_seconds <= 0:
            raise ValueError("Skill timeout must be finite and positive")
        self._skills[skill.name] = skill

    def describe(self, allowed_skills=None) -> list[dict]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "version": skill.version,
                "input_schema": skill.input_schema.model_json_schema(),
                "output_schema": skill.output_schema.model_json_schema(),
                "timeout_seconds": self._harness.skill_timeout(
                    name, skill.version, skill.timeout_seconds
                )
                if self._harness
                else skill.timeout_seconds,
            }
            for name, skill in sorted(self._skills.items())
            if allowed_skills is None or name in allowed_skills
            if self._harness is None or name in self._harness.enabled_skills
        ]

    def tool_definitions(self, allowed_skills) -> list[dict]:
        """Export callable schemas; availability never replaces execution-time authorization."""
        return [
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item["description"],
                    "parameters": item["input_schema"],
                },
            }
            for item in self.describe(allowed_skills)
        ]

    def require_available(self, names):
        """Preflight a deterministic API operation before creating any derived artifacts."""
        for name in names:
            skill = self._skills.get(name)
            if skill is None:
                raise SkillError("skill_not_found", "A required Skill is not registered.")
            if self._harness is not None:
                self._harness.skill_timeout(name, skill.version, skill.timeout_seconds)

    async def execute(self, name: str, payload: dict, context: SkillContext) -> SkillResult:
        start = time.monotonic()
        start_time = datetime.now(timezone.utc).isoformat()
        call_id = uuid4().hex
        skill = self._skills.get(name) if isinstance(name, str) else None
        safe_name = skill.name if skill else "unknown"
        child = None
        started = False
        input_fields: list[str] = []

        def elapsed() -> float:
            return (time.monotonic() - start) * 1000

        def emit(event: str, *, error_code=None, output_fields=None) -> None:
            try:
                self.trace_sink.emit(
                    TraceEvent(
                        event=event,
                        task_id=context.run_id,
                        agent_name=context.agent_name,
                        skill_name=safe_name,
                        start_time=start_time,
                        end_time=None
                        if event == "skill.started"
                        else datetime.now(timezone.utc).isoformat(),
                        status={
                            "skill.started": "running",
                            "skill.completed": "completed",
                            "skill.cancelled": "cancelled",
                            "skill.rejected": "rejected",
                            "skill.failed": "failed",
                        }[event],
                        error_message="Skill execution did not complete successfully."
                        if error_code
                        else None,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        run_id=context.run_id,
                        call_id=call_id,
                        owner_id=context.owner_id,
                        session_id=context.session_id,
                        skill=safe_name,
                        version=skill.version if skill else None,
                        input_version=context.input_version,
                        input_fields=input_fields,
                        output_fields=output_fields or [],
                        budget_used=context.budget.used,
                        latency_ms=elapsed(),
                        error_code=error_code,
                        harness_version=self._harness.version if self._harness else None,
                        harness_hash=self._harness.fingerprint if self._harness else None,
                    )
                )
            except Exception:
                raise SkillError(
                    "trace_unavailable", "The audit event could not be recorded."
                ) from None

        def result(*, data=None, error=None) -> SkillResult:
            warnings = data.get("warnings", []) if data else []
            if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
                warnings = []
            return SkillResult(
                status="error" if error else "success",
                data=data,
                error=error,
                warnings=warnings,
                skill=safe_name,
                version=skill.version if skill else None,
                run_id=context.run_id,
                call_id=call_id,
                latency_ms=elapsed(),
            )

        try:
            if self._closed:
                raise SkillError("registry_closed", "The Skill Registry is closed.")
            context.check_active()
            if skill is None:
                raise SkillError("skill_not_found", "The requested Skill is not registered.")
            if name not in context.allowed_skills:
                raise SkillError(
                    "permission_denied", "This Agent is not allowed to call this Skill."
                )
            timeout = skill.timeout_seconds
            if self._harness is not None:
                timeout = self._harness.skill_timeout(name, skill.version, timeout)
            if not isinstance(payload, dict):
                raise SkillError("invalid_input", "Skill input must be a JSON object.")
            try:
                validated = validate_input(skill.input_schema, payload)
            except ValidationError:
                raise SkillError(
                    "invalid_input", "Input does not match the Skill schema."
                ) from None
            input_fields = sorted(skill.input_schema.model_fields)
            child = context.child(timeout)

            async def dispatch():
                nonlocal started
                async with self._slots:
                    child.check_active()
                    context.budget.reserve()
                    emit("skill.started")
                    started = True
                    if inspect.iscoroutinefunction(skill.execute):
                        output = await skill.execute(validated, child)
                    else:
                        output = await asyncio.get_running_loop().run_in_executor(
                            self._executor, skill.execute, validated, child
                        )
                    child.check_active()
                    try:
                        data = validate_output(skill.output_schema, output)
                        if self._harness is not None:
                            self._harness.check_skill_output(
                                name,
                                validated.model_dump(),
                                data,
                                context=child,
                                service=skill.service,
                            )
                        return data
                    except (ValidationError, ValueError, TypeError):
                        raise SkillError(
                            "invalid_output", "The Skill returned an invalid output contract."
                        ) from None

            task = asyncio.create_task(dispatch())
            self._tasks.add(task)
            task.add_done_callback(self._drained)
            remaining = max(0.0, child.deadline - time.monotonic())
            try:
                # Shield retains the worker slot until the underlying computation truly exits.
                data = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except asyncio.TimeoutError:
                child.cancelled.set()
                raise SkillError("timeout", "The Skill exceeded its execution deadline.") from None
            context.check_active()
            emit("skill.completed", output_fields=sorted(skill.output_schema.model_fields))
            return result(data=data)
        except asyncio.CancelledError:
            if child is not None:
                child.cancelled.set()
            try:
                emit("skill.cancelled", error_code="cancelled")
            except SkillError:
                pass
            raise
        except Exception as exc:
            if child is not None:
                child.cancelled.set()
            error = (
                exc
                if isinstance(exc, SkillError)
                else SkillError(
                    "internal_error", "The Skill could not complete the requested operation."
                )
            )
            try:
                emit("skill.failed" if started else "skill.rejected", error_code=error.code)
            except SkillError as trace_error:
                error = trace_error
            return result(error=SkillFailure(code=error.code, message=error.message))

    def _drained(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()  # Consume errors from work that finished after the caller timed out.

    async def aclose(self) -> None:
        self._closed = True
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._executor.shutdown(wait=True)
