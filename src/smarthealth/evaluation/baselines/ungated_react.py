"""The ungated ReAct loop for comparison baselines B1 and B2.

Assembly mirrors ``live_llm.runner.execute_task`` (fixtures, fault injection,
observed backend, budgets, record shape) so every difference from full_guard is
attributable to the removed runtime path: no routing, no workflow policy, no
decision-level rejection and, decisively, no publication gate. The final answer
is a plain tool call the loop itself recognizes; the model's text is therefore
the released artifact and is graded as such.
"""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from uuid import uuid4

from pydantic import ValidationError

from smarthealth.agents.contracts import CallSkillAction, DecisionRequest, Observation
from smarthealth.agents.executor import _REF_OUTPUTS
from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.evaluation.baselines.backend import (
    FINAL_ANSWER_TOOL,
    FinalAnswerInput,
    UngatedBackend,
)
from smarthealth.evaluation.baselines.prompts import MAX_OBSERVATIONS
from smarthealth.evaluation.live_llm.budget import EvaluationBudget
from smarthealth.evaluation.live_llm.config import EvalConfig
from smarthealth.evaluation.live_llm.fault_injection import FaultInjector
from smarthealth.evaluation.live_llm.graders import grade_task
from smarthealth.evaluation.live_llm.runner import task_record
from smarthealth.evaluation.live_llm.tasks import BenchmarkTask
from smarthealth.evaluation.live_llm.trace_collector import (
    ArtifactCollector,
    EvaluationTransport,
    ObservedBackend,
    SkillCollector,
)
from smarthealth.reports.service import ReportService
from smarthealth.runtime.context import CallBudget, SkillContext
from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.harness.config import load_config as load_harness
from smarthealth.runtime.harness.harness_validator import HarnessValidator
from smarthealth.runtime.limits import AgentLimits
from smarthealth.runtime.token_budget import RunTokenBudget
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.bootstrap import create_registry

# Bounded observation facts; the same projection the runtime commits for LLM eyes.
_FACT_FIELDS = (
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
)


def _observed_facts(skill: str, data: dict, refs: dict) -> dict:
    facts: dict = {}
    if skill in _REF_OUTPUTS:
        field_name, kind = _REF_OUTPUTS[skill]
        ref = data[field_name]
        refs.setdefault(kind, []).append(ref)
        facts[field_name] = ref
    for name in _FACT_FIELDS:
        if name in data:
            facts[name] = deepcopy(data[name])
    if skill == "signal_processing":
        facts["quality"] = {
            key: data["cleaning_report"][key]
            for key in ("imputed_values", "dropped_tail_rows")
        }
    if skill == "feature_explanation":
        facts["important_features"] = [
            {"name": entry["name"], "direction": entry["direction"]}
            for entry in data["local_evidence"]
        ]
    if skill == "visualization":
        facts["plot_count"] = len(data["artifacts"])
    return facts


def _observation(
    observations: list[Observation],
    action: str,
    *,
    skill: str | None = None,
    status: str = "success",
    data: dict | None = None,
    error_code: str | None = None,
) -> None:
    observations.append(
        Observation(
            step_id=uuid4().hex,
            agent="data_analysis",
            action=action,
            skill=skill,
            status=status,
            data=data or {},
            error_code=error_code,
        )
    )


async def run_ungated(
    task: BenchmarkTask,
    *,
    repetition: int,
    config: EvalConfig,
    settings: Settings,
    predictor,
    retrieval,
    content: bytes,
    budget: EvaluationBudget,
    reference_prediction=None,
    mode="live",
    make_backend=None,
    method="b1_ungated_react",
    artifact_directory=None,
) -> dict:
    """One ungated task run; the record shape matches execute_task plus method keys.

    B1 keeps execute_task's skip semantics for missing retrieval assets; for B2
    the missing retrieval service is the treatment, so requires_rag tasks run.
    Budget or timeout exhaustion is an honest ``no_answer`` with its reason in
    ``final_answer.finish_reason``; only evaluation-ledger and infrastructure
    errors mark the record skipped or failed.
    """
    if mode not in {"live", "infrastructure_test"} or (
        make_backend and mode != "infrastructure_test"
    ):
        raise ValueError("Custom backends are limited to infrastructure tests")
    record = task_record(task, repetition, mode, method=method)
    no_rag = method == "b2_no_rag"
    if task.requires_rag and retrieval is None and not no_rag:
        record.update(status="skipped", skip_reason="real_rag_assets_unavailable")
        record["grades"] = grade_task(task, record)
        return record
    store = ArtifactCollector(lineage_enforced=config.baseline != "no_lineage")
    registry = backend = transport = None
    started = time.perf_counter()
    injector = FaultInjector(
        task.fault if not task.fault.startswith("backend_") else "none", task.fault_target
    )
    context = SkillContext(
        owner_id="synthetic_live_evaluation",
        session_id=f"{task.task_id}_{repetition}",
        input_version="synthetic_v1",
    )
    record["final_answer"] = {
        "text": None,
        "finish_reason": "decision_budget_exhausted",
        "decision_index": None,
    }
    try:
        base = load_harness(
            resource_path("configs", "report") if task.requires_rag else resource_path("configs")
        )
        limits = AgentLimits(
            max_decisions=task.max_execution_steps,
            max_llm_requests=task.max_execution_steps,
            max_tool_calls=task.max_tool_calls,
            max_total_tokens=config.per_task_token_budget,
            run_timeout_seconds=task.timeout_seconds,
            decision_timeout_seconds=config.decision_timeout_seconds,
        )
        harness = HarnessValidator(base, limits=limits, baseline=config.baseline)
        service = AnalysisService(
            store,
            predictor,
            retrieval=injector.wrap_retrieval(retrieval) if retrieval is not None else None,
            reports=ReportService(store, predictor, validator=harness),
        )
        original = create_registry(service, harness=harness)
        registry = SkillCollector(injector.wrap_registry(original), injector)
        if make_backend:
            inner = make_backend()
        else:
            transport = EvaluationTransport(task.fault)
            inner = UngatedBackend(
                settings.llm_base_url,
                settings.llm_api_key.get_secret_value(),
                config.model or settings.llm_model,
                timeout_seconds=config.decision_timeout_seconds,
                transport=transport,
                temperature=config.temperature,
                max_completion_tokens=config.max_completion_tokens,
                seed=config.seed,
                api_profile=settings.llm_api_profile,
                no_rag=no_rag,
            )
        backend = ObservedBackend(inner, budget, original, transport)
        allowed = set(harness.enabled_skills)
        if no_rag:
            allowed.discard("rag_retrieval")
        tools = registry.tool_definitions(frozenset(allowed))
        uploads = []
        if task.fixture != "none":
            value = content if task.fixture == "synthetic_csv" else b"unexpected_column\ninvalid\n"
            uploads.append(service.create_upload(value, context))
        refs: dict[str, list[str]] = {"upload": uploads}
        observations: list[Observation] = []
        skill_context = SkillContext(
            owner_id=context.owner_id,
            session_id=context.session_id,
            run_id=uuid4().hex,
            input_version=context.input_version,
            allowed_skills=frozenset(allowed),
            deadline=time.monotonic() + task.timeout_seconds,
            budget=CallBudget(task.max_tool_calls),
            agent_name="data_analysis",
        )
        token_budget = RunTokenBudget(limits.max_total_tokens) if backend.mode == "live" else None

        async def react_loop() -> None:
            for decision_index in range(1, task.max_execution_steps + 1):
                request = DecisionRequest(
                    run_id=skill_context.run_id,
                    agent="data_analysis",
                    task=task.user_request,
                    objective="data_analysis",
                    include_visualizations=task.include_visualizations,
                    available_refs=deepcopy(refs),
                    observations=[
                        observation.model_copy(deep=True)
                        for observation in observations[-MAX_OBSERVATIONS:]
                    ],
                    tools=list(tools),
                    allowed_delegates=[],
                    remaining_tool_calls=max(
                        0, task.max_tool_calls - skill_context.budget.used
                    ),
                    remaining_decisions=task.max_execution_steps - decision_index + 1,
                    remaining_tokens=token_budget.snapshot().available_tokens
                    if token_budget is not None
                    else None,
                )
                reply = None
                ticket = None
                try:
                    prepared = backend.prepare(request) if backend.mode == "live" else None
                    if prepared is not None:
                        ticket = token_budget.reserve(
                            prepared.input_bound, prepared.output_bound
                        )
                    reply = await asyncio.wait_for(
                        backend.decide_prepared(prepared) if prepared is not None
                        else backend.decide(request),
                        timeout=min(
                            config.decision_timeout_seconds,
                            max(0.001, skill_context.deadline - time.monotonic()),
                        ),
                    )
                except asyncio.TimeoutError:
                    _observation(
                        observations, "decision", status="error", error_code="llm_timeout"
                    )
                    continue
                except SkillError as exc:
                    if exc.code == "backend_context_too_large":
                        record["final_answer"]["finish_reason"] = "context_exceeded"
                        return
                    if exc.code == "token_budget_exceeded":
                        record["final_answer"]["finish_reason"] = "token_budget_exceeded"
                        return
                    if exc.code.startswith("evaluation_"):
                        raise
                    _observation(
                        observations,
                        "decision",
                        status="error",
                        error_code=exc.code,
                        data={
                            key: value
                            for key, value in (exc.details or {}).items()
                            if key in {"reason", "rule"}
                        },
                    )
                    continue
                finally:
                    if ticket is not None:
                        token_budget.settle(
                            ticket,
                            reply.input_tokens if reply is not None else None,
                            reply.output_tokens if reply is not None else None,
                        )
                action = reply.decision
                if isinstance(action, CallSkillAction) and action.skill == FINAL_ANSWER_TOOL:
                    try:
                        final = FinalAnswerInput.model_validate(action.arguments, strict=True)
                    except ValidationError:
                        _observation(
                            observations,
                            "call_skill",
                            skill=FINAL_ANSWER_TOOL,
                            status="error",
                            error_code="invalid_input",
                        )
                        continue
                    record["final_answer"] = {
                        "text": final.text,
                        "finish_reason": "final_answer",
                        "decision_index": decision_index,
                    }
                    return
                if not isinstance(action, CallSkillAction):
                    _observation(
                        observations,
                        action.action,
                        status="error",
                        error_code="ungated_protocol_action",
                    )
                    continue
                result = await registry.execute(action.skill, action.arguments, skill_context)
                data = {}
                if result.error and result.error.validation_error is not None:
                    data["validation_error"] = result.error.validation_error.model_dump(
                        mode="json"
                    )
                if not result.error:
                    data = _observed_facts(action.skill, result.data, refs)
                    if action.skill == "rag_retrieval":
                        # The gated writer sees retrieved passages through its report
                        # context; the loop exposes the same hit identities so text
                        # citations remain possible without the report roles.
                        stored = store.get(
                            result.data["evidence_ref"], kind="evidence", context=skill_context
                        )
                        data["evidence_hits"] = [
                            {
                                "evidence_id": hit["evidence_id"],
                                "source_id": hit["source"]["id"],
                            }
                            for hit in stored["result"]["hits"]
                        ]
                _observation(
                    observations,
                    "call_skill",
                    skill=action.skill,
                    status="error" if result.error else "success",
                    data=data,
                    error_code=result.error.code if result.error else None,
                )

        try:
            await asyncio.wait_for(react_loop(), timeout=task.timeout_seconds + 1)
        except asyncio.TimeoutError:
            record["final_answer"]["finish_reason"] = "timeout"
        record["effective_limits"] = limits.model_dump(mode="json")
        record["reference_prediction"] = reference_prediction
        if artifact_directory is not None:
            images = [
                artifact["ref"] for artifact in store.export() if artifact.get("kind") == "image"
            ]
            record["artifact_files"] = store.export_images(artifact_directory, images, context)
    except SkillError as exc:
        if exc.code.startswith("evaluation_"):
            record.update(status="skipped", skip_reason=exc.code)
        else:
            record.update(status="infrastructure_error", error_code=exc.code)
    except asyncio.CancelledError:
        record.update(status="infrastructure_error", error_code="evaluation_interrupted")
    except Exception:
        # Exceptions can include paths, provider text and secrets. Stable code only.
        record.update(status="infrastructure_error", error_code="evaluation_internal_error")
    finally:
        record["latency_ms"] = (time.perf_counter() - started) * 1000

        async def close_safely(resource):
            try:
                await resource.aclose()
            except Exception:
                record.update(status="infrastructure_error", error_code="evaluation_cleanup_failed")

        if backend:
            await close_safely(backend)
        elif transport:
            await close_safely(transport)
        if registry:
            await close_safely(
                registry
            )  # Drain actual workers before exporting/clearing artifacts.
            record["skill_calls"] = registry.calls
        if backend:
            record["provider_attempts"], record["decision_actions"] = (
                backend.attempts,
                backend.actions,
            )
        record["artifacts"] = store.export()
        if transport and task.fault.startswith("backend_"):
            record["fault"].update(triggered=transport.triggered, events=transport.events)
        else:
            record["fault"].update(triggered=injector.triggered, events=injector.events)
        store.clear()
    record["outcome_status"] = (
        "answered" if record["final_answer"]["text"] else "no_answer"
    )
    record["grades"] = grade_task(task, record)
    return record
