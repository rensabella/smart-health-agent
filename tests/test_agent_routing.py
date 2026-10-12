"""Router decisions must constrain execution, publication, references and budgets."""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from test_agent_runtime import close, complete, make_runtime, request_for

from app.agents.backends import OpenAICompatibleBackend, ScriptedDecisionBackend
from app.agents.contracts import (
    CallSkillAction,
    UpgradeWorkflowAction,
)
from app.agents.events import RunEventStore
from app.agents.router import HealthRouter
from app.agents.runtime import AgentLimits
from app.core.errors import SkillError
from app.main import create_app
from examples.demo_policy import scripted_analysis


def routed_runtime(predictor, policy=scripted_analysis, **kwargs):
    runtime = make_runtime(predictor, policy, **kwargs)
    runtime.router = HealthRouter()
    return runtime


def request(runtime, query, csv_bytes=None, **kwargs):
    return request_for(runtime, csv_bytes, **kwargs).model_copy(update={"query": query})


async def execute(runtime, body):
    return await complete(
        runtime, await runtime.create(body, owner_id="owner", request_id="routing-test")
    )


def trace(runtime, snapshot):
    return runtime.get_events(snapshot.run_id, owner_id="owner", session_id=snapshot.session_id)


async def test_automatic_entry_and_scope_override_client_hints(predictor, csv_bytes):
    observed = []

    def policy(state):
        observed.append(state)
        return scripted_analysis(state)

    runtime = routed_runtime(predictor, policy)
    try:
        simple = await execute(runtime, request(runtime, "解释risk score", entry_agent="planner"))
        assert simple.status == "completed" and simple.entry_agent == "data_analysis"
        assert simple.route.workflow_type == "simple" and simple.decision_count == 1
        assert simple.tool_calls == simple.delegation_count == 0
        assert observed[0].tools == [] and observed[0].remaining_tool_calls == 0
        assert observed[0].can_upgrade_workflow
        assert simple.final_result["kind"] == "score_definition"
        observed.clear()
        complex_run = await execute(
            runtime, request(runtime, "分析CSV数据", csv_bytes, objective="score_definition")
        )
        assert complex_run.status == "completed" and complex_run.entry_agent == "planner"
        assert complex_run.objective == "data_analysis" and complex_run.tool_calls == 5
        assert observed[0].agent == "planner" and complex_run.delegation_count == 1
        assert trace(runtime, complex_run)[1].event == "route.selected"
    finally:
        await close(runtime)


@pytest.mark.parametrize(
    "query,status,code",
    [
        ("这个用户为什么风险较高", "blocked", "safety_unavailable"),
        ("上传数据并生成完整健康分析报告", "blocked", "safety_unavailable"),
        ("分析CSV数据", "awaiting_input", "missing_upload"),
        ("显示上次结果", "awaiting_input", "missing_prior_result"),
        ("继续", "awaiting_input", "clarify_task"),
    ],
)
async def test_route_gates_do_not_spend_decisions_or_publish(query, status, code, predictor):
    def must_not_call(state):
        pytest.fail("A gated route must not invoke a decision backend")

    runtime = routed_runtime(predictor, must_not_call)
    try:
        result = await execute(runtime, request(runtime, query, objective="score_definition"))
        assert result.status == status and result.error_code == code
        assert result.decision_count == result.llm_requests == result.tool_calls == 0
        assert result.final_result is None
    finally:
        await close(runtime)


async def test_simple_cannot_dispatch_skills_without_upgrade(predictor, csv_bytes):
    def bypass(state):
        return CallSkillAction(
            skill="csv_analysis", arguments={"upload_ref": state.available_refs["upload"][0]}
        )

    runtime = routed_runtime(predictor, bypass, limits=AgentLimits(simple_max_decisions=2))
    try:
        result = await execute(runtime, request(runtime, "解释risk score", csv_bytes))
        assert result.status == "blocked" and result.error_code == "decision_budget_exceeded"
        assert result.decision_count == 2 and result.tool_calls == 0
        assert not any(e.event == "skill.dispatched" for e in trace(runtime, result))
        assert any(
            e.data.get("error_code") == "workflow_upgrade_required" for e in trace(runtime, result)
        )
    finally:
        await close(runtime)


async def test_upgrade_is_one_way_and_preserves_run_and_spent_budget(predictor, csv_bytes):
    seen = []

    def policy(state):
        seen.append(state)
        if state.can_upgrade_workflow:
            return UpgradeWorkflowAction(objective="data_analysis")
        if state.agent == "planner" and not any(
            o.error_code == "workflow_upgrade_denied" for o in state.observations
        ):
            return UpgradeWorkflowAction(objective="data_analysis")
        return scripted_analysis(state)

    runtime = routed_runtime(predictor, policy)
    try:
        result = await execute(runtime, request(runtime, "解释risk score", csv_bytes))
        assert result.status == "completed", result.model_dump()
        assert result.initial_workflow_type == "simple" and result.route.workflow_type == "complex"
        assert result.workflow_upgrades == 1 and result.entry_agent == "planner"
        assert result.decision_count == 10 and result.tool_calls == 5
        assert {s.run_id for s in seen} == {result.run_id}
        assert sum(e.event == "route.upgraded" for e in trace(runtime, result)) == 1
        assert seen[1].remaining_decisions == 30  # Both the upgrade and this decision count.
        assert all(not s.can_upgrade_workflow for s in seen[1:])
    finally:
        await close(runtime)


@pytest.mark.parametrize(
    "objective,code", [("data_analysis", "missing_upload"), ("health_report", "safety_unavailable")]
)
async def test_upgrade_respects_missing_input_and_safety_gate(predictor, objective, code):
    runtime = routed_runtime(predictor, lambda state: UpgradeWorkflowAction(objective=objective))
    try:
        result = await execute(runtime, request(runtime, "解释risk score"))
        assert result.error_code == code and result.workflow_upgrades == 1
        assert result.decision_count == 1 and result.final_result is None
    finally:
        await close(runtime)


async def test_upgrade_does_not_reset_global_budget(predictor, csv_bytes):
    runtime = routed_runtime(
        predictor,
        lambda state: UpgradeWorkflowAction(objective="data_analysis"),
        limits=AgentLimits(max_decisions=1),
    )
    try:
        result = await execute(runtime, request(runtime, "解释risk score", csv_bytes))
        assert result.workflow_upgrades == 1 and result.decision_count == 1
        assert result.error_code == "decision_budget_exceeded" and result.final_result is None
    finally:
        await close(runtime)


async def test_simple_deadline_is_enforced_before_late_upgrade(predictor, csv_bytes):
    async def late(state):
        await asyncio.sleep(0.05)
        return UpgradeWorkflowAction(objective="data_analysis")

    runtime = routed_runtime(predictor, late, limits=AgentLimits(simple_timeout_seconds=0.01))
    try:
        result = await execute(runtime, request(runtime, "解释risk score", csv_bytes))
        assert result.status == "failed" and result.error_code == "timeout"
        assert result.workflow_upgrades == 0 and result.decision_count == 1
    finally:
        await close(runtime)


async def test_existing_verified_result_reused_without_new_tools_or_ttl_refresh(
    predictor, csv_bytes
):
    runtime = routed_runtime(predictor)
    try:
        original = await execute(
            runtime, request(runtime, "分析CSV数据", csv_bytes, include_visualizations=True)
        )
        assert original.status == "completed"
        body = request(
            runtime,
            "显示上次结果和已有图表",
            session_id=original.session_id,
            input_version=original.input_version,
            source_run_id=original.run_id,
            include_visualizations=True,
        )
        reused = await execute(runtime, body)
        assert reused.status == "completed", reused.model_dump()
        assert reused.route.reason_code == "existing_result"
        assert reused.tool_calls == reused.delegation_count == 0 and reused.decision_count == 1
        assert reused.final_result["analysis"] == original.final_result["analysis"]
        assert reused.final_result["source_run_id"] == original.run_id
        source_record = runtime._runs[original.run_id]
        assert (
            runtime._runs[reused.run_id].result_expires_at
            == source_record.finished_at + runtime.limits.result_ttl_seconds
        )
        runtime._runs[reused.run_id].result_expires_at = time.monotonic() - 1
        assert (
            runtime.snapshot(
                reused.run_id, owner_id="owner", session_id=reused.session_id
            ).final_result
            is None
        )
    finally:
        await close(runtime)


@pytest.mark.parametrize(
    "case,code",
    [
        ("owner", "permission_denied"),
        ("session", "permission_denied"),
        ("version", "stale_input"),
        ("expired", "source_result_unavailable"),
        ("artifact", "artifact_not_found"),
    ],
)
async def test_source_reuse_revalidates_scope_and_retention(predictor, csv_bytes, case, code):
    runtime = routed_runtime(predictor)
    try:
        original = await execute(runtime, request(runtime, "分析CSV数据", csv_bytes))
        body = request(
            runtime,
            "显示上次结果",
            session_id=original.session_id,
            input_version=original.input_version,
            source_run_id=original.run_id,
        )
        owner = "someone_else" if case == "owner" else "owner"
        if case == "session":
            body = body.model_copy(update={"session_id": "another_session"})
        elif case == "version":
            body = body.model_copy(update={"input_version": "new_version"})
        elif case == "expired":
            runtime._runs[original.run_id].finished_at -= runtime.limits.result_ttl_seconds + 1
        elif case == "artifact":
            runtime.service.store.clear()
        with pytest.raises(SkillError) as error:
            await runtime.create(body, owner_id=owner, request_id="reuse-rejected")
        assert error.value.code == code
    finally:
        await close(runtime)


async def test_source_expiration_between_routing_and_publication_fails_closed(predictor, csv_bytes):
    entered, release = asyncio.Event(), asyncio.Event()

    async def policy(state):
        if state.available_refs.get("published_analysis"):
            entered.set()
            await release.wait()
        return scripted_analysis(state)

    runtime = routed_runtime(predictor, policy, limits=AgentLimits(simple_max_decisions=1))
    try:
        original = await execute(runtime, request(runtime, "分析CSV数据", csv_bytes))
        body = request(
            runtime,
            "显示上次结果",
            session_id=original.session_id,
            input_version=original.input_version,
            source_run_id=original.run_id,
        )
        initial = await runtime.create(body, owner_id="owner", request_id="reuse-expired")
        await asyncio.wait_for(entered.wait(), 2)
        runtime._runs[original.run_id].finished_at -= runtime.limits.result_ttl_seconds + 1
        release.set()
        result = await complete(runtime, initial)
        assert result.final_result is None
        assert any(
            e.data.get("error_code") == "source_result_unavailable" for e in trace(runtime, result)
        )
    finally:
        release.set()
        await close(runtime)


def test_preview_needs_no_backend_and_cannot_bypass_auth_or_reference_checks(predictor):
    app = create_app(predictor=predictor, api_token="test-token", agent_event_store=RunEventStore())
    with TestClient(app) as client:
        body = {"query": "解释risk score", "session_id": "session", "input_version": "v1"}
        headers = {"Authorization": "Bearer test-token"}
        assert client.post("/agent/route", json=body).status_code == 401
        preview = client.post("/agent/route", json=body, headers=headers)
        assert preview.status_code == 200 and preview.json()["workflow_type"] == "simple"
        assert app.state.agent_runtime._runs == {}
        invalid = client.post(
            "/agent/route", json={**body, "upload_refs": ["art_" + "0" * 32]}, headers=headers
        )
        assert invalid.status_code == 404
        assert client.post("/agent/route", content=b"x" * 32769, headers=headers).status_code == 413


def test_api_automatically_routes_and_journals_without_persisting_query(predictor):
    app = create_app(
        predictor=predictor,
        decision_backend=ScriptedDecisionBackend(scripted_analysis),
        agent_event_store=RunEventStore(),
    )
    with TestClient(app) as client:
        body = {"query": "解释risk score", "session_id": "session", "input_version": "v1"}
        preview = client.post("/agent/route", json=body).json()
        initial = client.post(
            "/agent/runs", json=body, headers={"Idempotency-Key": "route-once"}
        ).json()
        assert initial["route"] == preview
        stream = client.get(initial["events_url"]).text
        assert "route.selected" in stream and body["query"] not in stream
        result = client.get(f"/agent/runs/{initial['run_id']}?session_id=session").json()
        assert result["status"] == "completed" and result["decision_count"] == 1
        assert (
            client.post("/agent/runs", json=body, headers={"Idempotency-Key": "route-once"}).json()[
                "run_id"
            ]
            == initial["run_id"]
        )
        blocked = client.post(
            "/agent/runs",
            json={**body, "query": "为什么我的风险高", "objective": "score_definition"},
        ).json()
        assert blocked["route"]["objective"] == "health_report"
        assert "run.blocked" in client.get(blocked["events_url"]).text


async def test_live_protocol_exposes_upgrade_only_on_simple_path_and_counts_it(
    predictor, csv_bytes
):
    payloads = []

    def handle(req):
        payload = json.loads(req.content)
        payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "type": "function",
                                    "function": {
                                        "name": "upgrade_workflow",
                                        "arguments": '{"objective":"data_analysis"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    backend = OpenAICompatibleBackend(
        "https://llm.example/v1", "secret", "model", transport=httpx.MockTransport(handle)
    )
    runtime = routed_runtime(predictor, backend=backend, limits=AgentLimits(max_llm_requests=2))
    try:
        result = await execute(runtime, request(runtime, "解释risk score", csv_bytes))
        assert result.workflow_upgrades == 1 and result.llm_requests == 2
        assert result.error_code == "llm_budget_exceeded" and result.final_result is None
        assert {t["function"]["name"] for t in payloads[0]["tools"]} == {
            "finish",
            "request_input",
            "upgrade_workflow",
        }
        assert {t["function"]["name"] for t in payloads[1]["tools"]} == {
            "finish",
            "request_input",
            "delegate",
        }
        assert result.input_tokens == 11 and result.output_tokens == 3
        assert (
            not result.token_usage_available
        )  # Rejected second reply cannot supply trusted usage.
    finally:
        await close(runtime)


async def test_route_and_upgrade_survive_restart_without_result_restore(predictor, tmp_path):
    runtime = routed_runtime(
        predictor,
        lambda state: UpgradeWorkflowAction(objective="health_report"),
        events=RunEventStore(tmp_path),
    )
    try:
        result = await execute(runtime, request(runtime, "解释risk score"))
    finally:
        await close(runtime)
    restored = routed_runtime(predictor, events=RunEventStore(tmp_path))
    try:
        restored.recover()
        snapshot = restored.snapshot(result.run_id, owner_id="owner", session_id=result.session_id)
        assert snapshot.route == result.route and snapshot.objective == "health_report"
        assert snapshot.entry_agent == "planner" and snapshot.workflow_upgrades == 1
        assert snapshot.initial_workflow_type == "simple" and snapshot.final_result is None
    finally:
        await close(restored)


async def test_cancelled_display_cannot_publish_a_late_backend_result(predictor, csv_bytes):
    entered = asyncio.Event()

    async def policy(state):
        if state.available_refs.get("published_analysis"):
            entered.set()
            try:
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                pass  # Deliberately uncooperative backend still cannot publish.
        return scripted_analysis(state)

    runtime = routed_runtime(predictor, policy)
    try:
        original = await execute(runtime, request(runtime, "分析CSV数据", csv_bytes))
        body = request(
            runtime,
            "显示上次结果",
            session_id=original.session_id,
            input_version=original.input_version,
            source_run_id=original.run_id,
        )
        initial = await runtime.create(body, owner_id="owner", request_id="cancel-reuse")
        await asyncio.wait_for(entered.wait(), 2)
        runtime.cancel(initial.run_id, owner_id="owner", session_id=initial.session_id)
        final = await complete(runtime, initial)
        assert final.status == "cancelled" and final.final_result is None
        assert not any(e.event == "publication.checked" for e in trace(runtime, final))
    finally:
        await close(runtime)


async def test_query_plot_requirement_is_enforced_for_new_and_existing_results(
    predictor, csv_bytes
):
    runtime = routed_runtime(predictor)
    try:
        original = await execute(runtime, request(runtime, "分析CSV数据", csv_bytes))
        body = request(
            runtime,
            "显示上次结果和已有图表",
            session_id=original.session_id,
            input_version=original.input_version,
            source_run_id=original.run_id,
        )
        with pytest.raises(SkillError) as error:
            runtime.preview_route(body, owner_id="owner")
        assert error.value.code == "source_visualization_unavailable"
        new = await execute(runtime, request(runtime, "分析CSV数据并生成图表", csv_bytes))
        assert new.status == "completed" and new.tool_calls == 6
        assert new.final_result["analysis"]["visualization"]
        assert "visualization" in new.route.required_capabilities
    finally:
        await close(runtime)


async def test_existing_source_does_not_match_another_upload_in_same_version(predictor, csv_bytes):
    runtime = routed_runtime(predictor)
    try:
        original = await execute(runtime, request(runtime, "分析CSV数据", csv_bytes))
        body = request(
            runtime,
            "显示上次结果",
            csv_bytes,
            session_id=original.session_id,
            input_version=original.input_version,
            source_run_id=original.run_id,
        )
        with pytest.raises(SkillError) as error:
            await runtime.create(body, owner_id="owner", request_id="mismatched-upload")
        assert error.value.code == "stale_input"
    finally:
        await close(runtime)
