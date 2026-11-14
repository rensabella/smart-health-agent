"""Offline regressions for the real pilot's actions; no external provider calls."""

import asyncio
import json

import httpx
import pytest
from test_agent_backends import response, state
from test_agent_runtime import close, complete, make_runtime, request_for
from test_report_runtime import close as close_report
from test_report_runtime import execute as execute_report
from test_report_runtime import make as make_report
from test_report_runtime import retrieval as retrieval

from smarthealth.agents.backends import DecisionBackend, OpenAICompatibleBackend
from smarthealth.agents.contracts import (
    CallSkillAction,
    DelegateAction,
    FinishAction,
    TerminalResult,
    UpgradeWorkflowAction,
)
from smarthealth.agents.router import HealthRouter
from smarthealth.examples.report_policy import scripted_report
from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.events import RunEventStore
from smarthealth.runtime.limits import AgentLimits


def live_backend(handler, **kwargs):
    return OpenAICompatibleBackend(
        "https://llm.example/v1",
        "test-secret",
        "fixture-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def events(runtime, result):
    return runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)


async def run_score(runtime, query="解释risk score"):
    runtime.router = HealthRouter()
    queued = await runtime.create(
        request_for(runtime).model_copy(update={"query": query}),
        owner_id="owner",
        request_id="guard-v2",
    )
    return await complete(runtime, queued)


@pytest.mark.parametrize(
    "query", ["解释risk score", "风险分数是什么意思", "What does risk score mean?"]
)
@pytest.mark.parametrize("profile", ["standard", "dashscope"])
async def test_definition_has_no_upgrade_and_recovers_from_provider_ignoring_catalogue(
    predictor, query, profile
):
    calls = []

    def handle(request):
        wire = json.loads(request.content)
        calls.append(wire)
        assert "upgrade_workflow" not in {t["function"]["name"] for t in wire["tools"]}
        finish = next(t["function"] for t in wire["tools"] if t["function"]["name"] == "finish")
        assert finish["parameters"]["properties"]["result_kind"]["enum"] == ["score_definition"]
        value = (
            response("upgrade_workflow", {"objective": "health_report"})
            if len(calls) == 1
            else response()
        )
        return httpx.Response(200, json=value)

    runtime = make_runtime(predictor, backend=live_backend(handle, api_profile=profile))
    try:
        result = await run_score(runtime, query)
        assert result.status == "completed" and result.workflow_upgrades == 0
        assert result.tool_calls == result.delegation_count == 0 and len(calls) == 2
        assert result.final_result["kind"] == "score_definition"
        assert not any(e.event == "route.upgraded" for e in events(runtime, result))
        assert any(
            e.data.get("error_code") == "backend_invalid_decision" for e in events(runtime, result)
        )
    finally:
        await close(runtime)


async def test_runtime_rechecks_upgrade_even_when_backend_ignores_tool_catalogue(predictor):
    seen = []

    def policy(request):
        seen.append(request)
        return (
            UpgradeWorkflowAction(objective="health_report")
            if len(seen) == 1
            else FinishAction(result_kind="score_definition")
        )

    runtime = make_runtime(predictor, policy)
    try:
        result = await run_score(runtime)
        assert result.status == "completed" and result.workflow_upgrades == 0
        assert seen[1].observations[-1].error_code == "workflow_upgrade_denied"
        assert not seen[0].can_upgrade_workflow and seen[0].allowed_upgrade_objectives == []
    finally:
        await close(runtime)


async def test_enabling_report_skills_does_not_authorize_definition_upgrade(
    predictor, csv_bytes, retrieval
):
    def policy(request):
        assert not request.can_upgrade_workflow
        if not request.observations:
            return UpgradeWorkflowAction(objective="health_report")
        return FinishAction(result_kind="score_definition")

    runtime = make_report(predictor, retrieval, policy)
    try:
        result = await execute_report(runtime, csv_bytes, query="解释risk score")
        assert result.status == "completed" and result.workflow_upgrades == 0
        assert result.tool_calls == result.delegation_count == 0
    finally:
        await close_report(runtime)


@pytest.mark.parametrize("initial_review_fails", [False, True])
async def test_review_terminal_reference_blocks_reexecution_and_allows_revision(
    predictor, csv_bytes, retrieval, initial_review_fails
):
    terminal_views, previous_calls, injected = [], {}, set()

    def policy(request):
        if request.terminal_result and request.agent == "safety_verification":
            terminal_views.append(request)
            ref = request.terminal_result.result_ref
            if ref not in injected:
                injected.add(ref)
                # Deliberately ignore the catalogue, as in the live failure.
                return previous_calls[request.report_context["draft_ref"]]
        action = scripted_report(request)
        if isinstance(action, CallSkillAction) and action.skill == "safety_verification":
            if initial_review_fails and request.report_context["draft_version"] == 1:
                for review in action.arguments["reviews"]:
                    review.update(supported=False, reason="insufficient_support")
            previous_calls[action.arguments["draft_ref"]] = action
        return action

    runtime = make_report(predictor, retrieval, policy)
    try:
        result = await execute_report(runtime, csv_bytes)
        assert result.status == "completed", result.model_dump()
        review_count = 2 if initial_review_fails else 1
        assert result.final_result["report"]["safety"]["draft_version"] == review_count
        journal = events(runtime, result)
        assert sum(e.event == "safety.reviewed" for e in journal) == review_count
        rejected = [e for e in journal if e.data.get("error_code") == "review_already_exists"]
        assert len(rejected) == review_count
        assert all(e.data["result_kind"] == "review" and e.data["result_ref"] for e in rejected)
        assert terminal_views and all(
            v.tools == [] and v.remaining_tool_calls == 0 for v in terminal_views
        )
        assert all(v.allowed_finish_kinds == ["review"] for v in terminal_views)
        assert all(
            "claims" not in v.report_context and "retrieved_evidence" not in v.report_context
            for v in terminal_views
        )
        assert sum(e.event == "publication.checked" for e in journal) == 1
        assert next(i for i, e in enumerate(journal) if e.event == "publication.checked") < next(
            i for i, e in enumerate(journal) if e.event == "report.chunk"
        )
    finally:
        await close_report(runtime)


async def test_ready_planner_cannot_delegate_again_or_skip_publication_checks(
    predictor, csv_bytes, retrieval
):
    attempted = False

    def policy(request):
        nonlocal attempted
        if request.agent == "planner" and request.terminal_result and not attempted:
            attempted = True
            assert request.allowed_delegates == []
            return DelegateAction(agent="safety_verification", task="Review the same draft again")
        return scripted_report(request)

    runtime = make_report(predictor, retrieval, policy)
    try:
        result = await execute_report(runtime, csv_bytes)
        assert result.status == "completed" and attempted
        assert result.delegation_count == 4 and result.tool_calls == 8
        assert any(e.data.get("error_code") == "role_complete" for e in events(runtime, result))
        assert sum(e.event == "publication.checked" for e in events(runtime, result)) == 1
    finally:
        await close_report(runtime)


@pytest.mark.parametrize("profile", ["standard", "dashscope"])
async def test_terminal_wire_contract_and_prepared_payload_are_exact(profile):
    captured = []
    reference = "art_" + "a" * 32
    request = state("planner").model_copy(
        update={
            "objective": "health_report",
            "allowed_delegates": [],
            "allowed_finish_kinds": ["health_report"],
            "terminal_result": TerminalResult(
                reason="report_ready", result_kind="health_report", result_ref=reference
            ),
        }
    )

    def handle(sent):
        captured.append(sent.content)
        return httpx.Response(
            200, json=response("finish", {"result_kind": "health_report", "result_ref": reference})
        )

    backend = live_backend(handle, api_profile=profile)
    try:
        prepared = backend.prepare(request)
        request.task = "Mutated after admission; must never replace prepared bytes"
        result = await backend.decide_prepared(prepared)
        assert result.decision.result_ref == reference
        assert captured == [prepared.payload] and prepared.input_bound == len(captured[0])
        wire = json.loads(captured[0])
        assert [t["function"]["name"] for t in wire["tools"]] == ["finish"]
        assert wire["tools"][0]["function"]["parameters"]["properties"]["result_ref"]["enum"] == [
            reference
        ]
        assert "Mutated after admission" not in captured[0].decode()
        assert "test-secret" not in repr(prepared)
    finally:
        await backend.aclose()


async def test_admission_rejection_happens_before_any_http_request(predictor):
    calls = []

    def handle(request):
        calls.append(request)
        pytest.fail("Budget must reject before networking")

    runtime = make_runtime(
        predictor, backend=live_backend(handle), limits=AgentLimits(max_total_tokens=1)
    )
    try:
        result = await run_score(runtime)
        assert result.status == "blocked" and result.error_code == "token_budget_exceeded"
        assert result.llm_requests == result.token_budget.admitted_requests == 0 and not calls
        assert result.token_budget.charged_tokens == 0 and result.token_budget.limit == 1
        assert any(e.event == "budget.rejected" for e in events(runtime, result))
        rejected = next(e for e in events(runtime, result) if e.event == "budget.rejected")
        demand = rejected.data["admission_demand"]
        assert demand["request_bytes"] == demand["input_bound"] > 0
        assert demand["requested_tokens"] == demand["request_bytes"] + demand["output_cap"]
        assert demand["output_cap"] == 2048 and demand["available_tokens"] == 1
        assert demand["shortfall_tokens"] == demand["requested_tokens"] - 1
        assert all(type(v) is int for v in demand.values())
        assert not any(e.event == "budget.admitted" for e in events(runtime, result))
    finally:
        await close(runtime)


async def test_admission_demand_matches_exact_sent_bytes_and_durable_rejected_metadata(
    predictor, tmp_path
):
    calls = []

    def handle(request):
        calls.append(request.content)
        value = response("finish", {"result_kind": "data_analysis", "result_ref": None})
        value["usage"] = {"prompt_tokens": len(request.content), "completion_tokens": 1}
        return httpx.Response(200, json=value)

    runtime = make_runtime(
        predictor,
        backend=live_backend(handle),
        limits=AgentLimits(max_total_tokens=10000),
        events=RunEventStore(tmp_path),
    )
    try:
        result = await run_score(runtime)
        journal = events(runtime, result)
        admitted = next(e for e in journal if e.event == "budget.admitted")
        rejected = next(e for e in journal if e.event == "budget.rejected")
        assert len(calls) == result.llm_requests == 1
        assert admitted.data["admission_demand"]["request_bytes"] == len(calls[0])
        assert admitted.data["admission_demand"]["available_tokens"] == 10000
        demand = rejected.data["admission_demand"]
        assert demand["available_tokens"] == 10000 - len(calls[0]) - 1
        assert demand["shortfall_tokens"] > 0
        assert demand["requested_tokens"] == demand["input_bound"] + 2048
        assert result.token_budget.charged_tokens == len(calls[0]) + 1
        assert all(
            e.event not in {"agent.thinking", "budget.admitted", "skill.dispatched"}
            for e in journal
            if e.event_id > rejected.event_id
        )
        on_disk = "".join(p.read_text() for p in tmp_path.rglob("*.jsonl"))
        assert '"admission_demand"' in on_disk
        assert "test-secret" not in on_disk and "messages" not in on_disk
    finally:
        await close(runtime)


@pytest.mark.parametrize("outcome", ["known", "unknown", "timeout"])
async def test_next_request_or_retry_cannot_spend_unreserved_tokens(predictor, outcome):
    calls = []

    def handle(request):
        calls.append(request)
        if outcome == "timeout":
            raise httpx.ReadTimeout("Controlled timeout", request=request)
        value = response("finish", {"result_kind": "data_analysis", "result_ref": None})
        value["usage"] = (
            {"prompt_tokens": len(request.content), "completion_tokens": 1}
            if outcome == "known"
            else None
        )
        return httpx.Response(200, json=value)

    runtime = make_runtime(
        predictor, backend=live_backend(handle), limits=AgentLimits(max_total_tokens=10000)
    )
    try:
        result = await run_score(runtime)
        assert result.status == "blocked" and result.error_code == "token_budget_exceeded"
        assert len(calls) == result.llm_requests == result.token_budget.settled_requests == 1
        assert result.final_result is None and result.token_budget.in_flight_tokens == 0
        assert result.token_usage_available == (outcome == "known")
        expected = len(calls[0].content) + (1 if outcome == "known" else 2048)
        assert result.token_budget.charged_tokens == expected
    finally:
        await close(runtime)


async def test_known_usage_overrun_is_not_mislabeled_as_missing_usage(predictor):
    calls = []

    def handle(request):
        calls.append(request)
        value = response()
        value["usage"]["prompt_tokens"] = 70000
        return httpx.Response(200, json=value)

    runtime = make_runtime(predictor, backend=live_backend(handle))
    try:
        result = await run_score(runtime)
        assert result.status == "blocked" and result.error_code == "token_budget_exceeded"
        assert result.token_budget.bound_exceeded and result.token_usage_available
        assert result.input_tokens == 70000 and result.output_tokens == 9
        assert result.token_budget.unknown_usage_attempts == 0 and len(calls) == 1
        assert result.final_result is None
    finally:
        await close(runtime)


async def test_cancelled_request_retains_reservation_and_recovers_terminal_audit(
    predictor, tmp_path
):
    entered = asyncio.Event()
    calls = []

    async def handle(request):
        calls.append(request)
        entered.set()
        await asyncio.Event().wait()

    runtime = make_runtime(predictor, backend=live_backend(handle), events=RunEventStore(tmp_path))
    try:
        queued = await runtime.create(
            request_for(runtime, objective="score_definition"),
            owner_id="owner",
            request_id="cancel-budget",
        )
        await asyncio.wait_for(entered.wait(), 2)
        runtime.cancel(queued.run_id, owner_id="owner", session_id=queued.session_id)
        result = await complete(runtime, queued)
        assert result.status == "cancelled" and result.error_code == "cancelled"
        assert result.token_budget.charged_tokens == len(calls[0].content) + 2048
        assert (
            result.token_budget.in_flight_tokens == 0
            and result.token_budget.unknown_usage_attempts == 1
        )
        assert result.final_result is None
    finally:
        await close(runtime)
    restored = make_runtime(predictor, events=RunEventStore(tmp_path))
    try:
        restored.recover()
        audit = restored.snapshot(result.run_id, owner_id="owner", session_id=result.session_id)
        assert audit.status == audit.error_code == "cancelled"
        assert audit.token_budget == result.token_budget and audit.guard_version == "2.0"
        assert not restored._tasks
    finally:
        await close(restored)


async def test_completed_run_cannot_resume_decisions_or_transition_back_to_running(predictor):
    runtime = make_runtime(predictor)
    try:
        result = await run_score(runtime)
        record = runtime._runs[result.run_id]
        before = len(events(runtime, result))
        with pytest.raises(SkillError, match="terminal run"):
            await runtime._decision(record, "data_analysis", "Try again")
        with pytest.raises(SkillError, match="transition"):
            runtime._transition(record, "running")
        assert len(events(runtime, result)) == before and record.status == "completed"
    finally:
        await close(runtime)


async def test_live_adapter_without_admission_contract_fails_before_decide(predictor):
    class UnsupportedBackend(DecisionBackend):
        mode = "live"

        async def decide(self, request):
            pytest.fail("Unsupported live adapter cannot bypass admission")

        async def aclose(self):
            pass

    runtime = make_runtime(predictor, backend=UnsupportedBackend())
    try:
        result = await run_score(runtime)
        assert result.error_code == "backend_budget_unavailable" and result.llm_requests == 0
    finally:
        await close(runtime)
