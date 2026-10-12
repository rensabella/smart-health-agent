"""A decision-driven ReAct runtime with a single authoritative coordinator per run."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from app.agents.contracts import (
    ACTION_ADAPTER,
    TERMINAL_STATUSES,
    CallSkillAction,
    CreateRunRequest,
    DecisionReply,
    DecisionRequest,
    DelegateAction,
    FinishAction,
    Observation,
    RequestInputAction,
    RunSnapshot,
    UpgradeWorkflowAction,
)
from app.agents.limits import AgentLimits as AgentLimits
from app.agents.router import RouteDecision
from app.core.context import CallBudget, SkillContext
from app.core.errors import SkillError
from app.harness.config import ALL_SKILLS, REPORT_SKILLS
from app.harness.harness_validator import HarnessValidator, analysis_summary, score_definition
from app.memory.memory_manager import score_summary
from app.memory.schemas import ExecutionSummary, MemoryContext, TurnSummary
from app.schemas.response import AnalyzeResponse, Explanation, Prediction

logger = logging.getLogger("smart_health")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _Run:
    request: CreateRunRequest
    context: SkillContext
    request_id: str
    backend_mode: str
    decision_model: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    status: str = "queued"
    decision_count: int = 0
    llm_requests: int = 0
    delegation_count: int = 0
    data_tool_calls: int = 0
    role_tool_calls: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    token_usage_available: bool = False
    final_result: dict | None = None
    error_code: str | None = None
    message: str | None = None
    finished_at: float | None = None
    refs: dict[str, list[str]] = field(default_factory=dict)
    observations: deque = field(default_factory=deque)
    plots: dict[str, list[dict]] = field(default_factory=dict)
    quality_reports: dict[str, dict] = field(default_factory=dict)
    historical_tool_calls: int | None = None
    usage_missing: bool = False
    restored: bool = False
    route: RouteDecision | None = None
    initial_workflow_type: str | None = None
    workflow_upgrades: int = 0
    started_at: float = field(default_factory=time.monotonic)
    result_expires_at: float | None = None
    memory_lease: str | None = None
    memory_context: MemoryContext = field(default_factory=MemoryContext)
    memory_status: str = "disabled"
    harness_version: str | None = None
    harness_hash: str | None = None


_REF_INPUTS = {
    "csv_analysis": ("upload_ref", "upload"),
    "signal_processing": ("dataset_ref", "dataset"),
    "feature_extraction": ("window_set_ref", "window_set"),
    "anomaly_prediction": ("feature_set_ref", "feature_set"),
    "feature_explanation": ("prediction_ref", "prediction"),
    "visualization": ("analysis_ref", "analysis"),
    "rag_retrieval": ("analysis_ref", "analysis"),
    "report_generation": ("evidence_ref", "evidence"),
    "safety_verification": ("draft_ref", "draft"),
}
_REF_OUTPUTS = {
    "csv_analysis": ("dataset_ref", "dataset"),
    "signal_processing": ("window_set_ref", "window_set"),
    "feature_extraction": ("feature_set_ref", "feature_set"),
    "anomaly_prediction": ("prediction_ref", "prediction"),
    "feature_explanation": ("analysis_ref", "analysis"),
    "rag_retrieval": ("evidence_ref", "evidence"),
    "report_generation": ("draft_ref", "draft"),
    "safety_verification": ("review_ref", "review"),
}
_INPUT_MESSAGES = {
    "missing_upload": "请上传符合数据协议的 CSV，再使用 parent_run_id 创建新任务。",
    "invalid_data": "数据未通过检查，请修正 CSV 后重新上传并创建新任务。",
    "clarify_task": "请补充需要分析的数据或明确的数据分析目标，再创建新任务。",
    "missing_prior_result": "请提供当前 session 和 input_version 下已完成分析的 source_run_id；系统不会猜测历史结果。",
}


class AgentRuntime:
    def __init__(
        self,
        *,
        registry,
        service,
        backend,
        events,
        limits=None,
        router=None,
        memory=None,
        harness=None,
    ):
        self.registry, self.service, self.backend, self.events = registry, service, backend, events
        base_harness = harness or registry.harness or HarnessValidator()
        self.harness = HarnessValidator(base_harness.config, limits=limits)
        self.limits = self.harness.limits
        registry.bind_harness(self.harness)
        # The HTTP application always supplies a router. None supports isolated engine tests.
        self.router = router
        self.memory = memory
        self._runs: dict[str, _Run] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._create_lock = asyncio.Lock()
        self._closed = False

    def recover(self):
        """Restore only safe task metadata and audit history; never resume model execution."""
        for metadata in self.events.recover():
            try:
                request = CreateRunRequest(
                    query="Recovered task; original query was not persisted.",
                    session_id=metadata["session_id"],
                    input_version=metadata["input_version"],
                    objective=metadata["objective"],
                    entry_agent=metadata["entry_agent"],
                    parent_run_id=metadata.get("parent_run_id"),
                    source_run_id=metadata.get("source_run_id"),
                )
                context = self._context(metadata["owner_id"], request, metadata["run_id"])
                record = _Run(
                    request=request,
                    context=context,
                    request_id=metadata["request_id"],
                    backend_mode=metadata["backend_mode"],
                    decision_model=metadata.get("decision_model"),
                    created_at=metadata["created_at"],
                    updated_at=metadata.get("updated_at", metadata["created_at"]),
                    status=metadata["status"],
                    decision_count=metadata.get("decision_count", 0),
                    llm_requests=metadata.get("llm_requests", 0),
                    delegation_count=metadata.get("delegation_count", 0),
                    input_tokens=metadata.get("input_tokens", 0),
                    output_tokens=metadata.get("output_tokens", 0),
                    token_usage_available=metadata.get("token_usage_available", False),
                    historical_tool_calls=metadata.get("tool_calls", 0),
                    error_code=metadata.get("error_code"),
                    message="进程已重启；保留执行记录，原始输入和分析结果不在持久化记录中。",
                    finished_at=time.monotonic(),
                    restored=True,
                    memory_status="unknown" if self.memory is not None else "disabled",
                    route=RouteDecision.model_validate(metadata["route"])
                    if metadata.get("route")
                    else None,
                    initial_workflow_type=metadata.get("initial_workflow_type"),
                    workflow_upgrades=metadata.get("workflow_upgrades", 0),
                    harness_version=metadata.get("harness_version"),
                    harness_hash=metadata.get("harness_hash"),
                )
                self._runs[context.run_id] = record
            except (KeyError, ValueError):
                raise SkillError(
                    "trace_unavailable", "Stored run metadata is incompatible."
                ) from None

    def _context(self, owner_id, request, run_id=None):
        return SkillContext(
            owner_id=owner_id,
            session_id=request.session_id,
            input_version=request.input_version,
            run_id=run_id or uuid4().hex,
            allowed_skills=self.harness.allowed_skills("data_analysis"),
            deadline=time.monotonic() + self.limits.run_timeout_seconds,
            budget=CallBudget(self.limits.max_tool_calls),
        )

    def cleanup(self):
        now = time.monotonic()
        for record in self._runs.values():
            if self._result_expired(record, now):
                self._expire_result(record)
        expired = {
            run_id
            for run_id, record in self._runs.items()
            if record.finished_at is not None
            and now - record.finished_at >= self.limits.retention_seconds
        }
        for run_id in expired:
            self._runs.pop(run_id)
            self.events.forget(run_id)
        self._idempotency = {
            key: value for key, value in self._idempotency.items() if value[1] not in expired
        }

    @staticmethod
    def _expire_result(record):
        if record.final_result is not None:
            record.final_result = None
            record.message = "分析结果已超过保留期，请重新上传分析；执行记录仍可查询。"

    def _result_expired(self, record, now):
        return (
            record.finished_at is not None
            and now - record.finished_at >= self.limits.result_ttl_seconds
        ) or (record.result_expires_at is not None and now >= record.result_expires_at)

    def _make_room(self, exclude_id=None):
        if len(self._runs) < self.limits.max_retained_runs:
            return
        terminal = [
            r
            for r in self._runs.values()
            if r.status in TERMINAL_STATUSES
            and r.memory_status != "pending"
            and r.context.run_id != exclude_id
        ]
        if not terminal:
            raise SkillError("run_capacity_exceeded", "Agent task capacity is exhausted.")
        oldest = min(terminal, key=lambda r: r.updated_at)
        self.events.forget(oldest.context.run_id)
        self._runs.pop(oldest.context.run_id)
        self._idempotency = {
            key: value
            for key, value in self._idempotency.items()
            if value[1] != oldest.context.run_id
        }

    async def create(self, request, *, owner_id, request_id, idempotency_key=None):
        request = CreateRunRequest.model_validate(request.model_dump(), strict=True)
        digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not 1 <= len(idempotency_key) <= 128
            or any(ord(c) < 33 or ord(c) > 126 for c in idempotency_key)
        ):
            raise SkillError("invalid_input", "Idempotency-Key must be bounded printable ASCII.")
        async with self._create_lock:
            if self._closed:
                raise SkillError("runtime_unavailable", "Agent runtime is shutting down.")
            self.cleanup()
            key = (owner_id, idempotency_key)
            if idempotency_key and key in self._idempotency:
                previous_digest, previous_run = self._idempotency[key]
                if previous_digest != digest:
                    raise SkillError(
                        "idempotency_conflict", "The key already identifies different input."
                    )
                return self.snapshot(previous_run, owner_id=owner_id, session_id=request.session_id)
            if self.backend.mode == "disabled":
                raise SkillError(
                    "backend_unavailable",
                    "Configure a real decision backend before creating Agent tasks.",
                )
            active = [
                r
                for r in self._runs.values()
                if r.status not in TERMINAL_STATUSES or r.memory_status == "pending"
            ]
            if any(
                (r.context.owner_id, r.context.session_id) == (owner_id, request.session_id)
                for r in active
            ):
                raise SkillError("session_busy", "This session already has an active task.")
            if len(active) >= self.limits.max_active_runs:
                raise SkillError("run_capacity_exceeded", "Agent task capacity is exhausted.")
            if request.parent_run_id:
                parent = self._owned(request.parent_run_id, owner_id, request.session_id)
                if parent.status != "awaiting_input":
                    raise SkillError(
                        "invalid_parent_run", "Only a task awaiting input may be continued."
                    )
            context = self._context(owner_id, request)
            for ref in request.upload_refs:
                # Validate ownership, kind, version and expiration before starting the task.
                self.service.store.get(ref, kind="upload", context=context)
            route = (
                self._route(request, owner_id=owner_id, context=context) if self.router else None
            )
            if route:
                request = request.model_copy(
                    update={
                        "objective": route.objective,
                        "entry_agent": route.entry_agent,
                        "include_visualizations": "visualization" in route.required_capabilities,
                    }
                )
            self._make_room(exclude_id=request.source_run_id)
            record = _Run(
                request=request,
                context=context,
                request_id=request_id,
                backend_mode=self.backend.mode,
                decision_model=getattr(self.backend, "model_name", None),
                refs={"upload": list(request.upload_refs)},
                observations=deque(maxlen=self.limits.max_observations),
                route=route,
                initial_workflow_type=route.workflow_type if route else None,
                harness_version=self.harness.version,
                harness_hash=self.harness.fingerprint,
            )
            if route and route.reason_code == "existing_result":
                source = self._source_result(request, owner_id, context)
                record.refs["published_analysis"] = [source["analysis"]["analysis_ref"]]
            self.events.register(
                context.run_id,
                owner_id=owner_id,
                session_id=context.session_id,
                metadata={
                    "input_version": context.input_version,
                    "request_id": request_id,
                    "backend_mode": record.backend_mode,
                    "decision_model": record.decision_model,
                    "objective": request.objective,
                    "entry_agent": request.entry_agent,
                    "created_at": record.created_at,
                    "parent_run_id": request.parent_run_id,
                    "source_run_id": request.source_run_id,
                    "route": route.model_dump(mode="json") if route else None,
                    "harness_version": record.harness_version,
                    "harness_hash": record.harness_hash,
                },
            )
            self._emit(record, "run.queued", data={"backend_mode": record.backend_mode})
            if route:
                self._emit_route(record, "route.selected")
            self._emit(
                record,
                "harness.loaded",
                data={
                    "harness_version": record.harness_version,
                    "harness_hash": record.harness_hash,
                },
            )
            if self.memory is not None:
                try:
                    record.memory_lease = self.memory.begin(
                        owner_id=owner_id,
                        session_id=request.session_id,
                        input_version=request.input_version,
                        run_id=context.run_id,
                        upload_refs=list(request.upload_refs),
                    )
                    record.memory_status = "active"
                except SkillError:
                    self._memory_unavailable(record)
            self._runs[context.run_id] = record
            if idempotency_key:
                self._idempotency[key] = (digest, context.run_id)
            task = asyncio.create_task(self._run(record), name=f"health-run-{context.run_id}")
            self._tasks[context.run_id] = task
            task.add_done_callback(lambda finished: self._task_done(context.run_id, finished))
            return self.snapshot(context.run_id, owner_id=owner_id, session_id=request.session_id)

    def _task_done(self, run_id, task):
        self._tasks.pop(run_id, None)
        if not task.cancelled():
            task.exception()

    def preview_route(self, request, *, owner_id):
        """Read-only rule evaluation with the same reference checks as task creation."""
        request = CreateRunRequest.model_validate(request.model_dump(), strict=True)
        if self.router is None:
            raise SkillError("router_unavailable", "No task router is configured.")
        context = self._context(owner_id, request)
        for ref in request.upload_refs:
            self.service.store.get(ref, kind="upload", context=context)
        return self._route(request, owner_id=owner_id, context=context)

    def _route(self, request, *, owner_id, context):
        existing = False
        if request.source_run_id:
            self._source_result(request, owner_id, context)
            existing = True
        route = self.router.route(
            query=request.query,
            has_upload=bool(request.upload_refs),
            has_existing_result=existing,
            objective_hint=request.objective,
            include_visualizations=request.include_visualizations,
        )
        if (
            route.reason_code == "existing_result"
            and "visualization" in route.required_capabilities
        ):
            self._source_result(
                request.model_copy(update={"include_visualizations": True}), owner_id, context
            )
        return route

    def _source_result(self, request, owner_id, context):
        source = self._owned(request.source_run_id, owner_id, request.session_id)
        if source.context.input_version != request.input_version:
            raise SkillError("stale_input", "The prior result belongs to another input version.")
        if self._result_expired(source, time.monotonic()):
            self._expire_result(source)
        if (
            source.status != "completed"
            or source.final_result is None
            or source.final_result.get("kind") != "data_analysis"
        ):
            raise SkillError(
                "source_result_unavailable", "A completed, retained analysis is required."
            )
        result = deepcopy(source.final_result)
        analysis = AnalyzeResponse.model_validate(result["analysis"])
        store = self.service.store
        # Reuse must not outlive any part of its original, authorized artifact lineage.
        evidence = store.get(analysis.analysis_ref, kind="analysis", context=context)
        prediction = store.get(evidence["prediction_ref"], kind="prediction", context=context)
        features = store.get(prediction["feature_set_ref"], kind="feature_set", context=context)
        windows = store.get(features["window_set_ref"], kind="window_set", context=context)
        dataset = store.get(windows["dataset_ref"], kind="dataset", context=context)
        store.get(dataset["upload_ref"], kind="upload", context=context)
        if request.upload_refs and dataset["upload_ref"] not in request.upload_refs:
            raise SkillError("stale_input", "The prior analysis does not match this upload.")
        for plot in analysis.visualization:
            store.get(plot.artifact_ref, kind="image", context=context)
        if request.include_visualizations and not analysis.visualization:
            raise SkillError(
                "source_visualization_unavailable",
                "This analysis has no existing plots; request a new data analysis to create them.",
            )
        return result

    def _existing_result(self, record, ref):
        if not ref or ref not in record.refs.get("published_analysis", []):
            raise SkillError(
                "unobserved_reference", "Only the selected prior result may be displayed."
            )
        result = self._source_result(record.request, record.context.owner_id, record.context)
        if result["analysis"]["analysis_ref"] != ref:
            raise SkillError("stale_input", "The selected result is no longer available.")
        source = self._owned(
            record.request.source_run_id, record.context.owner_id, record.context.session_id
        )
        # Displaying a result does not refresh the original analysis retention clock.
        record.result_expires_at = min(
            source.finished_at + self.limits.result_ttl_seconds,
            source.result_expires_at or float("inf"),
        )
        return {**result, "source_run_id": record.request.source_run_id}

    def _emit_route(self, record, event):
        self._emit(
            record,
            event,
            data={
                **record.route.model_dump(mode="json"),
                "workflow_upgrades": record.workflow_upgrades,
            },
        )

    def _route_gate(self, record):
        """Missing inputs and unavailable report capabilities need no model invocation."""
        if record.route is None and record.request.objective != "health_report":
            return False
        if record.route is not None and record.route.needs_input:
            self._transition(
                record,
                "awaiting_input",
                error_code=record.route.needs_input,
                message=_INPUT_MESSAGES[record.route.needs_input],
            )
            return True
        if record.request.objective == "health_report" and not self._reports_available():
            self._transition(
                record,
                "blocked",
                error_code="safety_unavailable",
                message="该任务需要启用报告策略、已构建的知识索引及独立审核能力；依赖未就绪，当前不发布报告。",
            )
            return True
        if record.request.objective == "health_report" and not record.request.upload_refs:
            self._transition(
                record,
                "awaiting_input",
                error_code="missing_upload",
                message=_INPUT_MESSAGES["missing_upload"],
            )
            return True
        return False

    def _reports_available(self):
        return (
            self.harness.config.safety.medical_report_enabled
            and REPORT_SKILLS <= self.harness.enabled_skills
            and self.service.retrieval is not None
            and self.service.reports is not None
            and self.service.reports.policy_hash == self.harness.fingerprint
            and all(
                skill in self.harness.allowed_skills(role)
                for role, skill in (
                    ("rag", "rag_retrieval"),
                    ("report_writer", "report_generation"),
                    ("safety_verification", "safety_verification"),
                )
            )
        )

    @staticmethod
    def _latest(record, kind):
        values = record.refs.get(kind, [])
        return values[-1] if values else None

    def _report_context(self, record, role):
        if record.request.objective != "health_report" or not self._reports_available():
            return {}
        analysis, evidence, draft, review = (
            self._latest(record, k) for k in ("analysis", "evidence", "draft", "review")
        )
        status = {
            "analysis_ref": analysis,
            "evidence_ref": evidence,
            "draft_ref": draft,
            "review_ref": review,
            "scope": "synthetic_analysis_with_educational_context",
        }
        if review:
            checked = self.service.store.get(review, kind="report_review", context=record.context)
            # Service context has only bounded, schema-validated fields; raw review is never journaled.
            status["review"] = {
                key: checked[key]
                for key in ("safe", "warnings", "draft_ref", "draft_version", "draft_hash")
            }
            status["review_safe"] = checked.get("safe", False)
            status["review_current"] = checked["draft_ref"] == draft
        if draft:
            status["draft_version"] = self.service.store.get(
                draft, kind="report_draft", context=record.context
            )["draft_version"]
        if role == "report_writer" and evidence:
            status.update(self.service.reports.writer_context(analysis, evidence, record.context))
        elif role == "safety_verification" and draft:
            status.update(self.service.reports.review_context(draft, record.context))
        elif role == "rag" and analysis:
            result = self._analysis_result(record, analysis)["analysis"]
            status["analysis_summary"] = result["summary"]
            status["feature_names"] = [f["name"] for f in result["explanation"]["local_evidence"]]
        return status

    def _report_prerequisite(self, record, role):
        if role not in {"rag", "report_writer", "safety_verification"}:
            return
        if record.request.objective != "health_report" or not self._reports_available():
            raise SkillError("permission_denied", "Report roles require an enabled report task.")
        if not self._latest(record, "analysis"):
            raise SkillError("incomplete_analysis", "Analyze the current upload before retrieval.")
        self._analysis_result(record, self._latest(record, "analysis"))
        if role in {"report_writer", "safety_verification"}:
            evidence = self._latest(record, "evidence")
            if not evidence:
                raise SkillError("insufficient_evidence", "Retrieve supporting evidence first.")
            stored = self.service.store.get(evidence, kind="evidence", context=record.context)
            if stored["result"]["status"] != "success" or not stored["result"]["hits"]:
                raise SkillError("insufficient_evidence", "No supporting evidence is available.")
        if role == "safety_verification" and not self._latest(record, "draft"):
            raise SkillError("missing_draft", "Write a versioned draft before review.")

    def _is_simple(self, record):
        return record.route is not None and record.route.workflow_type == "simple"

    def _decision_cap(self, record):
        return (
            min(self.limits.max_decisions, self.limits.simple_max_decisions)
            if self._is_simple(record)
            else self.limits.max_decisions
        )

    def _llm_cap(self, record):
        return (
            min(self.limits.max_llm_requests, self.limits.simple_max_llm_requests)
            if self._is_simple(record)
            else self.limits.max_llm_requests
        )

    def _deadline(self, record):
        return (
            min(record.context.deadline, record.started_at + self.limits.simple_timeout_seconds)
            if self._is_simple(record)
            else record.context.deadline
        )

    def _check_active(self, record):
        record.context.check_active()
        if time.monotonic() >= self._deadline(record):
            raise SkillError("timeout", "The workflow deadline has elapsed.")

    def _owned(self, run_id, owner_id, session_id):
        record = self._runs.get(run_id)
        if record is None:
            raise SkillError("run_not_found", "The Agent task is unavailable or expired.")
        if (record.context.owner_id, record.context.session_id) != (owner_id, session_id):
            raise SkillError("permission_denied", "The task is outside this session.")
        return record

    def snapshot(self, run_id, *, owner_id, session_id):
        record = self._owned(run_id, owner_id, session_id)
        if self._result_expired(record, time.monotonic()):
            self._expire_result(record)
        return RunSnapshot(
            run_id=run_id,
            request_id=record.request_id,
            session_id=record.context.session_id,
            input_version=record.context.input_version,
            parent_run_id=record.request.parent_run_id,
            status=record.status,
            backend_mode=record.backend_mode,
            decision_model=record.decision_model,
            objective=record.request.objective,
            entry_agent=record.request.entry_agent,
            created_at=record.created_at,
            updated_at=record.updated_at,
            decision_count=record.decision_count,
            llm_requests=record.llm_requests,
            tool_calls=record.historical_tool_calls
            if record.restored
            else record.context.budget.used,
            delegation_count=record.delegation_count,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            token_usage_available=record.token_usage_available,
            last_event_id=self.events.latest_id(run_id),
            final_result=deepcopy(record.final_result),
            result_available=record.final_result is not None,
            error_code=record.error_code,
            message=record.message,
            events_url=f"/agent/runs/{run_id}/events?session_id={record.context.session_id}",
            route=record.route.model_copy(deep=True) if record.route else None,
            initial_workflow_type=record.initial_workflow_type,
            workflow_upgrades=record.workflow_upgrades,
            source_run_id=record.request.source_run_id,
            memory_status=record.memory_status,
            harness_version=record.harness_version,
            harness_hash=record.harness_hash,
        )

    def get_events(self, run_id, *, owner_id, session_id, after_id=0):
        self._owned(run_id, owner_id, session_id)
        return self.events.get_events(
            run_id, owner_id=owner_id, session_id=session_id, after_id=after_id
        )

    async def wait(self, run_id, *, owner_id, session_id):
        self._owned(run_id, owner_id, session_id)
        task = self._tasks.get(run_id)
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        return self.snapshot(run_id, owner_id=owner_id, session_id=session_id)

    def cancel(self, run_id, *, owner_id, session_id):
        record = self._owned(run_id, owner_id, session_id)
        if record.status not in TERMINAL_STATUSES:
            record.context.cancelled.set()
            task = self._tasks.get(run_id)
            if task:
                task.cancel()
            self._transition(record, "cancelled", error_code="cancelled")
            record.request = record.request.model_copy(update={"query": "Task input released."})
        return self.snapshot(run_id, owner_id=owner_id, session_id=session_id)

    def _emit(self, record, event, *, agent=None, step_id=None, parent_step_id=None, data=None):
        emitted = self.events.append(
            record.context.run_id,
            event,
            agent=agent,
            step_id=step_id,
            parent_step_id=parent_step_id,
            data=data or {},
        )
        record.updated_at = emitted.timestamp
        if record.memory_lease and self.memory is not None:
            self.memory.update(
                owner_id=record.context.owner_id,
                session_id=record.context.session_id,
                lease=record.memory_lease,
                execution=ExecutionSummary(
                    run_id=record.context.run_id,
                    status=record.status,
                    last_event_id=emitted.event_id,
                    decision_count=record.decision_count,
                    tool_calls=record.context.budget.used,
                ),
            )
        return emitted

    def _transition(self, record, status, *, error_code=None, message=None):
        self._emit(
            record,
            f"run.{status}",
            data={
                "status": status,
                "error_code": error_code,
                "decision_count": record.decision_count,
                "llm_requests": record.llm_requests,
                "tool_calls": record.context.budget.used,
                "delegation_count": record.delegation_count,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "token_usage_available": record.token_usage_available,
            },
        )
        record.status, record.error_code, record.message = status, error_code, message
        record.updated_at = utcnow()
        if status in TERMINAL_STATUSES:
            record.finished_at = time.monotonic()
            if status != "completed":
                self._finish_short_memory(record)
            elif record.memory_lease:
                record.memory_status = "pending"

    def _memory_turn(self, record, analysis=None):
        return TurnSummary(
            run_id=record.context.run_id,
            input_version=record.context.input_version,
            user_intent=record.request.objective,
            assistant_status=record.status,
            completed_at=self.memory.clock(),
            analysis=analysis,
        )

    def _finish_short_memory(self, record):
        if self.memory is not None and record.memory_lease:
            try:
                accepted = self.memory.finish_short(
                    owner_id=record.context.owner_id,
                    session_id=record.context.session_id,
                    lease=record.memory_lease,
                    turn=self._memory_turn(record),
                )
                record.memory_status = "stored" if accepted else "skipped"
            except SkillError:
                self._memory_unavailable(record)
            finally:
                record.memory_lease = None

    def _memory_unavailable(self, record):
        record.memory_status = "unavailable"
        logger.warning(
            json.dumps(
                {
                    "event": "memory.unavailable",
                    "run_id": record.context.run_id,
                    "request_id": record.request_id,
                    "error_code": "memory_unavailable",
                }
            )
        )

    async def _finish_memory(self, record):
        if self.memory is None or not record.memory_lease:
            return
        try:
            summary = None
            if (
                record.status == "completed"
                and record.final_result
                and record.final_result.get("kind") in {"data_analysis", "health_report"}
            ):
                summary = score_summary(
                    record.final_result["analysis"], self.service.predictor.metadata
                )
            record.memory_status = await self.memory.finish(
                owner_id=record.context.owner_id,
                session_id=record.context.session_id,
                lease=record.memory_lease,
                turn=self._memory_turn(record, summary),
                fresh_analysis=summary is not None
                and record.final_result.get("source_run_id") is None,
            )
            logger.info(
                json.dumps(
                    {
                        "event": "memory.finalized",
                        "run_id": record.context.run_id,
                        "request_id": record.request_id,
                        "status": record.memory_status,
                    }
                )
            )
        except asyncio.CancelledError:
            self._memory_unavailable(record)
            raise
        except Exception:
            self._memory_unavailable(record)
            # Also release the active session if summary projection failed before finish().
            self._finish_short_memory(record)
            record.memory_status = "unavailable"
        finally:
            record.memory_lease = None
            record.updated_at = utcnow()

    def _observe(self, record, role, step_id, action, *, skill=None, data=None, error_code=None):
        record.observations.append(
            Observation(
                step_id=step_id,
                agent=role,
                action=action,
                skill=skill,
                status="error" if error_code else "success",
                data=data or {},
                error_code=error_code,
            )
        )

    async def _run(self, record):
        try:
            self._transition(record, "running")
            if self._route_gate(record):
                return
            if self.memory is not None and record.memory_lease:
                try:
                    record.memory_context = await self.memory.context(
                        owner_id=record.context.owner_id,
                        session_id=record.context.session_id,
                        query=record.request.query,
                    )
                    self._emit(
                        record,
                        "memory.loaded",
                        data={
                            "memory_recent_turns": len(record.memory_context.recent_turns),
                            "memory_history_hits": len(record.memory_context.historical_summaries),
                            "preferences_available": record.memory_context.preferences is not None,
                        },
                    )
                except SkillError:
                    self._memory_unavailable(record)
            await asyncio.wait_for(
                self._agent_loop(record, record.request.entry_agent, record.request.query),
                timeout=max(0.001, record.context.deadline - time.monotonic()),
            )
        except asyncio.CancelledError:
            record.context.cancelled.set()
            if record.status not in TERMINAL_STATUSES:
                self._fail(record, "cancelled", status="cancelled")
        except asyncio.TimeoutError:
            record.context.cancelled.set()
            self._fail(record, "timeout")
        except SkillError as exc:
            record.context.cancelled.set()
            status = (
                "cancelled"
                if exc.code == "cancelled"
                else (
                    "blocked"
                    if exc.code
                    in {
                        "budget_exceeded",
                        "decision_budget_exceeded",
                        "llm_budget_exceeded",
                        "token_budget_exceeded",
                        "harness_publication_denied",
                        "harness_invalid_output",
                        "skill_disabled",
                        "skill_version_mismatch",
                    }
                    else "failed"
                )
            )
            self._fail(record, exc.code, status=status)
        except Exception:
            record.context.cancelled.set()
            self._fail(record, "internal_error")
        finally:
            # Task/query/observations are ephemeral; terminal snapshots contain only approved data.
            if record.status in TERMINAL_STATUSES:
                await self._finish_memory(record)
                record.request = record.request.model_copy(update={"query": "Task input released."})
                record.memory_context = MemoryContext()
                record.observations.clear()
                record.quality_reports.clear()
                record.plots.clear()

    def _fail(self, record, code, status="failed"):
        if record.status in TERMINAL_STATUSES:
            return
        record.final_result = None
        try:
            self._transition(record, status, error_code=code)
        except SkillError:
            # Publication stays closed even if the journal itself is unavailable.
            record.status, record.error_code = "failed", "trace_unavailable"
            record.finished_at, record.updated_at = time.monotonic(), utcnow()

    async def _decision(self, record, role, task):
        for attempt in range(2):
            self._check_active(record)
            if record.input_tokens + record.output_tokens >= self.limits.max_total_tokens:
                raise SkillError("token_budget_exceeded", "Reported token budget exhausted.")
            if record.decision_count >= self._decision_cap(record):
                raise SkillError("decision_budget_exceeded", "Decision budget exhausted.")
            if self.backend.mode == "live" and record.llm_requests >= self._llm_cap(record):
                raise SkillError("llm_budget_exceeded", "LLM request budget exhausted.")
            record.decision_count += 1
            if self.backend.mode == "live":
                record.llm_requests += 1
            step_id = uuid4().hex
            self._emit(
                record,
                "agent.thinking",
                agent=role,
                step_id=step_id,
                data={
                    "decision_count": record.decision_count,
                    "llm_requests": record.llm_requests,
                    "attempt": attempt + 1,
                },
            )
            request = DecisionRequest(
                run_id=record.context.run_id,
                agent=role,
                task=task,
                objective=record.request.objective,
                include_visualizations=record.request.include_visualizations,
                available_refs=deepcopy(record.refs),
                observations=[o.model_copy(deep=True) for o in record.observations],
                tools=self.registry.tool_definitions(
                    self.harness.allowed_skills(role, simple=self._is_simple(record))
                ),
                allowed_delegates=sorted(self.harness.allowed_delegates(role))
                if not self._is_simple(record)
                else [],
                remaining_tool_calls=0
                if self._is_simple(record)
                else min(
                    self.limits.max_tool_calls - record.context.budget.used,
                    self.harness.role_tool_limit(role) - record.role_tool_calls.get(role, 0),
                ),
                remaining_decisions=self._decision_cap(record) - record.decision_count,
                workflow_type=record.route.workflow_type if record.route else None,
                can_upgrade_workflow=self._is_simple(record) and record.workflow_upgrades == 0,
                memory=record.memory_context.model_copy(deep=True)
                if self.memory is not None
                and self.memory.is_current(
                    owner_id=record.context.owner_id,
                    session_id=record.context.session_id,
                    run_id=record.context.run_id,
                )
                else MemoryContext(),
                report_context=self._report_context(record, role),
            )
            try:
                reply = await asyncio.wait_for(
                    self.backend.decide(request),
                    timeout=min(
                        self.limits.decision_timeout_seconds,
                        max(0.001, self._deadline(record) - time.monotonic()),
                    ),
                )
                self._check_active(record)
                reply = DecisionReply.model_validate(reply.model_dump(warnings=False), strict=True)
                decision = ACTION_ADAPTER.validate_python(reply.decision.model_dump(), strict=True)
                if reply.input_tokens is not None and reply.output_tokens is not None:
                    record.input_tokens += reply.input_tokens
                    record.output_tokens += reply.output_tokens
                else:
                    record.usage_missing = True
                record.token_usage_available = not record.usage_missing
                if record.input_tokens + record.output_tokens > self.limits.max_total_tokens:
                    raise SkillError(
                        "token_budget_exceeded", "Reported token usage exceeded the run allowance."
                    )
                return step_id, decision
            except (ValidationError, AttributeError):
                error = SkillError(
                    "invalid_decision", "Decision did not match the action contract."
                )
            except asyncio.TimeoutError:
                error = SkillError("llm_timeout", "The decision backend timed out.")
            except SkillError as exc:
                error = exc
            self._emit(
                record,
                "agent.decision_failed",
                agent=role,
                step_id=step_id,
                data={"error_code": error.code, "attempt": attempt + 1},
            )
            if self.backend.mode == "live":
                record.usage_missing = True
                record.token_usage_available = False
            if attempt or error.code not in {
                "llm_timeout",
                "backend_timeout",
                "backend_transport_error",
                "backend_rate_limited",
                "backend_unavailable",
            }:
                raise error
        raise RuntimeError("Unreachable decision retry state")

    async def _agent_loop(self, record, role, task, *, depth=0, parent_step_id=None):
        self._emit(record, "agent.started", agent=role, parent_step_id=parent_step_id)
        while record.status == "running":
            try:
                step_id, action = await self._decision(record, role, task)
            except SkillError as exc:
                if exc.code not in {
                    "invalid_decision",
                    "backend_invalid_decision",
                    "backend_invalid_response",
                }:
                    raise
                self._observe(record, role, uuid4().hex, "decision", error_code=exc.code)
                continue
            public_action = {
                "action": action.action,
                "decision_count": record.decision_count,
                "llm_requests": record.llm_requests,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "token_usage_available": record.token_usage_available,
            }
            if isinstance(action, CallSkillAction):
                public_action["skill"] = action.skill if action.skill in ALL_SKILLS else "unknown"
            self._emit(
                record,
                "agent.decision",
                agent=role,
                step_id=step_id,
                parent_step_id=parent_step_id,
                data=public_action,
            )
            if self._is_simple(record) and isinstance(action, (CallSkillAction, DelegateAction)):
                self._reject(record, role, step_id, action.action, "workflow_upgrade_required")
                continue
            if isinstance(action, CallSkillAction):
                await self._call_skill(record, role, step_id, action, parent_step_id)
            elif isinstance(action, DelegateAction):
                if not self.harness.delegation_allowed(role, action.agent, depth=depth):
                    self._reject(record, role, step_id, action.action, "delegation_denied")
                    continue
                try:
                    self._report_prerequisite(record, action.agent)
                except SkillError as exc:
                    self._reject(record, role, step_id, action.action, exc.code)
                    continue
                if record.delegation_count >= self.limits.max_delegations:
                    raise SkillError("budget_exceeded", "Delegation budget exhausted.")
                record.delegation_count += 1
                result = await self._agent_loop(
                    record, action.agent, action.task, depth=depth + 1, parent_step_id=step_id
                )
                if record.status == "running":
                    self._observe(record, role, step_id, "delegate", data=result)
                    self._emit(
                        record,
                        "agent.delegation_completed",
                        agent=role,
                        step_id=step_id,
                        data={
                            "agent": action.agent,
                            "delegation_count": record.delegation_count,
                        },
                    )
            elif isinstance(action, UpgradeWorkflowAction):
                if depth or not self._is_simple(record) or record.workflow_upgrades >= 1:
                    self._reject(record, role, step_id, action.action, "workflow_upgrade_denied")
                    continue
                route = self.router.upgrade(
                    record.route,
                    objective=action.objective,
                    has_upload=bool(record.request.upload_refs),
                )
                # Persist the decision before allowing any work under expanded capabilities.
                self._emit(
                    record,
                    "route.upgraded",
                    data={**route.model_dump(mode="json"), "workflow_upgrades": 1},
                )
                record.route = route
                record.workflow_upgrades = 1
                record.request = record.request.model_copy(
                    update={"objective": route.objective, "entry_agent": route.entry_agent}
                )
                self._observe(
                    record, role, step_id, action.action, data={"workflow_type": "complex"}
                )
                if self._route_gate(record):
                    return None
                return await self._agent_loop(record, "planner", task)
            elif isinstance(action, RequestInputAction):
                self._transition(record, "awaiting_input", message=_INPUT_MESSAGES[action.reason])
                return None
            elif isinstance(action, FinishAction):
                if depth:
                    try:
                        expected = {
                            "data_analysis": "data_analysis",
                            "rag": "evidence",
                            "report_writer": "draft",
                            "safety_verification": "review",
                        }[role]
                        if action.result_kind != expected:
                            raise SkillError(
                                "objective_mismatch", "Delegate may only return data analysis."
                            )
                        if role == "data_analysis":
                            self._analysis_result(record, action.result_ref)
                        else:
                            if action.result_ref != self._latest(record, expected):
                                raise SkillError(
                                    "unobserved_reference", "Return the current delegated result."
                                )
                            kind = {
                                "evidence": "evidence",
                                "draft": "report_draft",
                                "review": "report_review",
                            }[expected]
                            self.service.store.get(
                                action.result_ref, kind=kind, context=record.context
                            )
                    except SkillError as exc:
                        self._reject(record, role, step_id, "finish", exc.code)
                        continue
                    self._emit(
                        record,
                        "agent.completed",
                        agent=role,
                        step_id=step_id,
                        parent_step_id=parent_step_id,
                        data={"result_ref": action.result_ref},
                    )
                    return {
                        f"{'analysis' if role == 'data_analysis' else expected}_ref": action.result_ref
                    }
                if action.result_kind != record.request.objective:
                    self._reject(record, role, step_id, "finish", "objective_mismatch")
                    continue
                if action.result_kind == "health_report" and not self._reports_available():
                    self._transition(
                        record,
                        "blocked",
                        error_code="safety_unavailable",
                        message="完整健康报告需要后续 RAG、报告写作与安全审核能力，当前不发布。",
                    )
                    return None
                try:
                    candidate = (
                        self._report_result(record, action.result_ref)
                        if action.result_kind == "health_report"
                        else (
                            (
                                self._existing_result(record, action.result_ref)
                                if self._is_simple(record)
                                else self._analysis_result(record, action.result_ref)
                            )
                            if action.result_kind == "data_analysis"
                            else self._score_definition()
                        )
                    )
                except SkillError as exc:
                    self._reject(record, role, step_id, "finish", exc.code)
                    if self._is_simple(record) and exc.code in {
                        "source_result_unavailable",
                        "source_visualization_unavailable",
                        "run_not_found",
                        "artifact_not_found",
                        "artifact_expired",
                        "stale_input",
                    }:
                        self._transition(
                            record,
                            "awaiting_input",
                            error_code=exc.code,
                            message=_INPUT_MESSAGES["missing_prior_result"],
                        )
                        return None
                    continue
                self._transition(record, "verifying")
                self._check_active(record)
                try:
                    candidate = self.harness.validate_publication(
                        candidate,
                        objective=action.result_kind,
                        context=record.context,
                        service=self.service,
                    )
                except SkillError as exc:
                    self._emit(
                        record,
                        "publication.rejected",
                        data={
                            "error_code": exc.code,
                            "harness_hash": record.harness_hash,
                        },
                    )
                    self._fail(record, exc.code, status="blocked")
                    return None
                self._emit(
                    record,
                    "publication.checked",
                    data={
                        "policy": self.harness.config.safety.publication_policy,
                        "result_kind": action.result_kind,
                        "harness_hash": record.harness_hash,
                    },
                )
                self._check_active(record)
                if action.result_kind == "health_report":
                    body = candidate["report"]["body"]
                    report_hash = hashlib.sha256(body.encode()).hexdigest()
                    for index in range((len(body) + 399) // 400):
                        self._emit(
                            record,
                            "report.chunk",
                            data={"chunk_index": index, "report_hash": report_hash},
                        )
                self._transition(record, "completed")
                record.final_result = candidate
                return candidate

    def _reject(self, record, role, step_id, action, code):
        self._observe(record, role, step_id, action, error_code=code)
        self._emit(
            record,
            "agent.action_rejected",
            agent=role,
            step_id=step_id,
            data={"action": action, "error_code": code},
        )

    def _check_report_action(self, record, action):
        args, context = action.arguments, record.context
        analysis, evidence, draft, review = (
            self._latest(record, k) for k in ("analysis", "evidence", "draft", "review")
        )
        if action.skill == "rag_retrieval":
            if args.get("analysis_ref") != analysis or draft:
                raise SkillError(
                    "stale_input", "Retrieve for the latest analysis before writing a draft."
                )
        elif action.skill == "report_generation":
            if args.get("analysis_ref") != analysis or args.get("evidence_ref") != evidence:
                raise SkillError("stale_input", "Draft must use the latest analysis and evidence.")
            if draft and (
                args.get("previous_draft_ref") != draft
                or args.get("previous_review_ref") != review
                or not review
            ):
                raise SkillError(
                    "revision_required", "Revise the current draft using its failed review."
                )
        elif action.skill == "safety_verification":
            if args.get("draft_ref") != draft:
                raise SkillError("stale_input", "Review only the current draft.")
            if review:
                prior = self.service.store.get(review, kind="report_review", context=context)
                if prior.get("draft_ref") == draft:
                    raise SkillError(
                        "revision_required", "An unchanged draft cannot be reviewed repeatedly."
                    )

    def _report_result(self, record, ref):
        draft, review = self._latest(record, "draft"), self._latest(record, "review")
        if not draft or not review or ref != review:
            raise SkillError(
                "unreviewed_report", "Finish requires the current independent review reference."
            )
        report = self.service.reports.publication(draft, review, record.context)
        if report.analysis_ref != self._latest(
            record, "analysis"
        ) or report.evidence_ref != self._latest(record, "evidence"):
            raise SkillError("stale_input", "Report evidence does not match the current run.")
        analysis = self._analysis_result(record, report.analysis_ref)["analysis"]
        return {
            "kind": "health_report",
            "analysis": analysis,
            "report": report.model_dump(mode="json"),
            "review_mode": "llm_assisted" if self.backend.mode == "live" else "scripted_test",
        }

    def report_chunk(self, run_id, *, owner_id, session_id, index, report_hash):
        snapshot = self.snapshot(run_id, owner_id=owner_id, session_id=session_id)
        if (
            snapshot.status != "completed"
            or not snapshot.final_result
            or snapshot.final_result.get("kind") != "health_report"
        ):
            return None
        body = snapshot.final_result["report"]["body"]
        if hashlib.sha256(body.encode()).hexdigest() != report_hash:
            raise SkillError(
                "harness_publication_denied", "Stored report no longer matches its publication."
            )
        return body[index * 400 : (index + 1) * 400]

    async def _call_skill(self, record, role, step_id, action, parent_step_id):
        if self._is_simple(record):
            self._reject(record, role, step_id, "call_skill", "workflow_upgrade_required")
            return
        try:
            self.harness.check_call(
                role,
                action.skill,
                simple=self._is_simple(record),
                used=record.context.budget.used,
                role_used=record.role_tool_calls.get(role, 0),
                context=record.context,
            )
        except SkillError as exc:
            if exc.code != "permission_denied":
                raise
            self._reject(record, role, step_id, "call_skill", exc.code)
            return
        if (
            record.request.objective == "health_report"
            and role != "safety_verification"
            and record.context.budget.used >= self.limits.max_tool_calls - 1
        ):
            raise SkillError(
                "budget_exceeded", "One tool call is reserved for final Safety verification."
            )
        field_name, kind = _REF_INPUTS[action.skill]
        ref = action.arguments.get(field_name)
        if ref not in record.refs.get(kind, []):
            self._reject(record, role, step_id, "call_skill", "unobserved_reference")
            return
        if action.skill in REPORT_SKILLS:
            try:
                self._report_prerequisite(record, role)
                self._check_report_action(record, action)
            except SkillError as exc:
                self._reject(record, role, step_id, "call_skill", exc.code)
                return
        self._emit(
            record,
            "skill.dispatched",
            agent=role,
            step_id=step_id,
            parent_step_id=parent_step_id,
            data={"skill": action.skill, "input_ref": ref},
        )
        before = record.context.budget.used
        role_context = replace(record.context, allowed_skills=self.harness.allowed_skills(role))
        result = await self.registry.execute(action.skill, action.arguments, role_context)
        consumed = record.context.budget.used - before
        record.role_tool_calls[role] = record.role_tool_calls.get(role, 0) + consumed
        if role == "data_analysis":
            record.data_tool_calls += consumed
        record.context.check_active()
        safe_data = {}
        if not result.error:
            safe_data = self._commit_skill(record, action, result.data)
            if action.skill == "safety_verification":
                self._emit(
                    record,
                    "safety.reviewed",
                    agent=role,
                    data={
                        "safe": result.data["safe"],
                        "revision_needed": result.data["revision_needed"],
                        "draft_hash": result.data["draft_hash"],
                        "review_ref": result.data["review_ref"],
                    },
                )
        self._observe(
            record,
            role,
            step_id,
            "call_skill",
            skill=action.skill,
            data=safe_data,
            error_code=result.error.code if result.error else None,
        )
        self._emit(
            record,
            "skill.failed" if result.error else "skill.completed",
            agent=role,
            step_id=step_id,
            parent_step_id=parent_step_id,
            data={
                "skill": action.skill,
                "call_id": result.call_id,
                "version": result.version,
                "latency_ms": result.latency_ms,
                "tool_calls": record.context.budget.used,
                "error_code": result.error.code if result.error else None,
                "output_refs": [v for k, v in safe_data.items() if k.endswith("_ref")],
            },
        )
        if not result.error and action.skill == "safety_verification" and not result.data["safe"]:
            draft = self.service.store.get(
                self._latest(record, "draft"), kind="report_draft", context=record.context
            )
            if draft["draft_version"] >= 3:
                self._fail(record, "safety_review_failed", status="blocked")
        if result.error and result.error.code in {
            "budget_exceeded",
            "trace_unavailable",
            "model_unavailable",
            "dependency_unavailable",
            "harness_invalid_output",
            "skill_disabled",
            "skill_version_mismatch",
            "retrieval_unavailable",
            "safety_unavailable",
            "revision_limit_exceeded",
            "report_revision_limit",
            "rag_unavailable",
            "rag_index_invalid",
            "rag_embedding_unavailable",
        }:
            raise SkillError(result.error.code, "Required Skill execution is unavailable.")

    def _commit_skill(self, record, action, data):
        """Only bounded derived facts and refs enter LLM observations; never sensor rows."""
        facts = {}
        if action.skill in _REF_OUTPUTS:
            field_name, kind = _REF_OUTPUTS[action.skill]
            ref = data[field_name]
            record.refs.setdefault(kind, []).append(ref)
            facts[field_name] = ref
        for name in (
            "row_count",
            "window_count",
            "prediction",
            "risk_score",
            "risk_level",
            "highest_risk_window",
            "anomalous_ratio",
            "method",
            "warnings",
            "limitations",
            "hit_count",
            "knowledge_version",
            "status",
            "draft_version",
            "draft_hash",
            "evidence_hash",
            "claim_count",
            "safe",
            "revision_needed",
        ):
            if name in data:
                facts[name] = deepcopy(data[name])
        if action.skill == "signal_processing":
            record.quality_reports[data["window_set_ref"]] = deepcopy(data["cleaning_report"])
            facts["quality"] = {
                k: data["cleaning_report"][k] for k in ("imputed_values", "dropped_tail_rows")
            }
        if action.skill == "feature_explanation":
            facts["important_features"] = [
                {"name": e["name"], "direction": e["direction"]} for e in data["local_evidence"]
            ]
        if action.skill == "visualization":
            record.plots[action.arguments["analysis_ref"]] = deepcopy(data["artifacts"])
            facts["plot_count"] = len(data["artifacts"])
        return facts

    def _analysis_result(self, record, ref):
        if not ref or ref not in record.refs.get("analysis", []):
            raise SkillError(
                "incomplete_analysis", "Finish requires an analysis produced in this run."
            )
        if record.request.include_visualizations and not record.plots.get(ref):
            raise SkillError(
                "missing_visualization", "The requested visualization is not complete."
            )
        context, store = record.context, self.service.store
        analysis = store.get(ref, kind="analysis", context=context)
        prediction = store.get(analysis["prediction_ref"], kind="prediction", context=context)
        features = store.get(prediction["feature_set_ref"], kind="feature_set", context=context)
        windows = store.get(features["window_set_ref"], kind="window_set", context=context)
        dataset = store.get(windows["dataset_ref"], kind="dataset", context=context)
        if dataset["upload_ref"] not in record.request.upload_refs:
            raise SkillError("stale_input", "Analysis does not match this task's input.")
        quality = record.quality_reports.get(features["window_set_ref"])
        if quality is None:
            raise SkillError("incomplete_analysis", "Verified cleaning provenance is required.")
        predicted = Prediction.model_validate(prediction["result"])
        explanation = Explanation.model_validate(analysis["explanation"])
        images = deepcopy(record.plots.get(ref, []))
        for image in images:
            store.get(image["artifact_ref"], kind="image", context=context)
            image["url"] += (
                f"?session_id={context.session_id}&input_version={context.input_version}"
            )
        summary = analysis_summary(predicted)
        response = AnalyzeResponse(
            request_id=record.request_id,
            session_id=context.session_id,
            input_version=context.input_version,
            analysis_ref=ref,
            summary=summary,
            risk_level=predicted.risk_level,
            prediction=predicted,
            explanation=explanation,
            visualization=images,
            quality_report=quality,
            warnings=quality.get("warnings", []),
        )
        return {"kind": "data_analysis", "analysis": response.model_dump(mode="json")}

    def _score_definition(self):
        predictor = self.service.predictor
        if predictor is None:
            raise SkillError("model_unavailable", "A loaded model card is required.")
        return score_definition(predictor.metadata)

    async def aclose(self):
        self._closed = True
        for record in list(self._runs.values()):
            if record.status not in TERMINAL_STATUSES:
                record.context.cancelled.set()
                task = self._tasks.get(record.context.run_id)
                if task:
                    task.cancel()
        await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)
        await self.backend.aclose()
