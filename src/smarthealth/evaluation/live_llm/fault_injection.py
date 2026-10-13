"""One-shot faults for freshly constructed evaluation fixtures only.

No application factory imports this module. ``wrap_registry`` replaces a target
instance in the *per-task* Registry's private catalogue, the only private seam
used here. The real Registry still performs schema, permission, budget, timeout,
lineage and audit checks. Never wrap a shared application Registry.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading
from dataclasses import replace

from smarthealth.retrieval.schemas import RetrievalResult
from smarthealth.skills.base_skill import BaseSkill

TOOL_FAULTS = frozenset(
    {
        "none",
        "tool_timeout",
        "invalid_output",
        "invalid_arguments",
        "stale_artifact",
        "duplicate_execution",
        "empty_retrieval",
    }
)


class FaultInjector:
    """State belongs to one task repetition; faults cannot silently become success.

    Injection is recorded separately from the observed Registry outcome. Duplicate
    execution deliberately makes two real calls and consumes both call budgets;
    it does not claim that existing Skills provide exactly-once semantics.
    """

    def __init__(self, kind: str = "none", target_skill: str = "csv_analysis"):
        if kind not in TOOL_FAULTS:
            raise ValueError("Unsupported evaluation tool fault")
        self.kind = kind
        self.target_skill = target_skill
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._triggered = False
        self._wrapped = False

    @property
    def triggered(self) -> bool:
        with self._lock:
            return self._triggered

    @property
    def fired(self) -> int:
        return int(self.triggered)

    @property
    def events(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._events)

    def _claim(self, target: str) -> bool:
        with self._lock:
            if self._triggered or self.kind == "none":
                return False
            self._triggered = True
            self._events.append({"phase": "injected", "kind": self.kind, "target": target})
            return True

    def _observed(self, target, result, *, attempt=1):
        with self._lock:
            self._events.append(
                {
                    "phase": "observed",
                    "kind": self.kind,
                    "target": target,
                    "attempt": attempt,
                    "status": result.status,
                    "error_code": result.error.code if result.error else None,
                    "call_id": result.call_id,
                }
            )

    def wrap_registry(self, registry):
        """Wrap a new, unused task Registry; its lifecycle remains caller-owned."""
        if self._wrapped:
            raise ValueError("An injector may wrap only one task Registry")
        if self.kind not in {"none", "empty_retrieval"}:
            if self.target_skill not in registry._skills:
                raise ValueError("Fault target must be an existing registered Skill")
            if self.kind in {"tool_timeout", "invalid_output"}:
                # Evaluation-only instance replacement. The schemas, version,
                # service and active Harness are preserved, including checks.
                registry._skills[self.target_skill] = _FaultSkill(
                    registry._skills[self.target_skill], self
                )
        self._wrapped = True
        return _FaultRegistry(registry, self)

    def wrap_retrieval(self, retrieval):
        if self.kind != "empty_retrieval":
            return retrieval
        if retrieval is None:
            raise ValueError("Empty-result injection requires an actual retrieval service")
        return _EmptyRetrieval(retrieval, self)


class _FaultSkill(BaseSkill):
    def __init__(self, inner, injector):
        self.inner, self.injector = inner, injector
        for field in ("name", "description", "version", "input_schema", "output_schema"):
            setattr(self, field, getattr(inner, field))
        self.timeout_seconds = inner.timeout_seconds
        if hasattr(inner, "service"):
            self.service = inner.service

    async def execute(self, input, context):
        if self.injector._claim(self.name):
            if self.injector.kind == "tool_timeout":
                # The proxy shortens only this invocation's context. The real
                # Registry timer fires; no synthetic SkillError is returned.
                await asyncio.sleep(0.05)
                context.check_active()
            elif self.injector.kind == "invalid_output":
                # This does not execute a real side effect then hide its result.
                # The Registry's actual output validator rejects this object.
                return {"evaluation_invalid_output": True}
        if inspect.iscoroutinefunction(self.inner.execute):
            return await self.inner.execute(input, context)
        return await asyncio.to_thread(self.inner.execute, input, context)


class _FaultRegistry:
    def __init__(self, inner, injector):
        self.inner, self.injector = inner, injector

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def execute(self, name, payload, context):
        state = self.injector
        if name != state.target_skill or state.kind in {"none", "empty_retrieval"}:
            return await self.inner.execute(name, payload, context)
        was_triggered = state.triggered
        if state.kind == "tool_timeout" and not was_triggered:
            context = context.child(0.02)
        elif state.kind in {"invalid_arguments", "stale_artifact", "duplicate_execution"}:
            if state._claim(name):
                if state.kind == "invalid_arguments":
                    payload = dict(payload, evaluation_invalid_argument=True)
                elif state.kind == "stale_artifact":
                    version = "evaluation_stale_v2"
                    if context.input_version == version:
                        version = "evaluation_stale_v1"
                    context = replace(context, input_version=version)
                else:
                    first = await self.inner.execute(name, payload, context)
                    state._observed(name, first, attempt=1)
                    second = await self.inner.execute(name, payload, context)
                    state._observed(name, second, attempt=2)
                    return second
        result = await self.inner.execute(name, payload, context)
        if not was_triggered and state.triggered:
            state._observed(name, result)
        return result


class _EmptyRetrieval:
    def __init__(self, inner, injector):
        self.inner, self.injector = inner, injector

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def retrieve(self, query, top_k=4):
        # Query validation and the actual local index still execute. Suppress
        # one result set, preserving its real corpus/embedding identity.
        result = self.inner.retrieve(query, top_k=top_k)
        if not self.injector._claim("rag_retrieval"):
            return result
        return RetrievalResult(
            status="insufficient_evidence",
            hits=(),
            knowledge_version=result.knowledge_version,
            embedding_id=result.embedding_id,
        )
