"""Reproduce the unnecessary upload failure using the real policy/runtime, offline."""

import json

import httpx
import pytest
from test_agent_backends import response
from test_agent_runtime import close, complete, make_runtime, request_for
from test_runtime_guard_v2 import events, live_backend, run_score

from smarthealth.agents.backends import _tools_for
from smarthealth.agents.contracts import (
    CallSkillAction,
    FinishAction,
    RequestInputAction,
    UpgradeWorkflowAction,
)
from smarthealth.agents.router import HealthRouter
from smarthealth.examples.demo_policy import scripted_analysis


@pytest.mark.parametrize(
    "action",
    [
        RequestInputAction(reason="missing_upload"),
        RequestInputAction(reason="invalid_data"),
        RequestInputAction(reason="clarify_task"),
        UpgradeWorkflowAction(objective="health_report"),
        CallSkillAction(skill="csv_analysis", arguments={}),
    ],
)
async def test_score_rejects_unnecessary_actions_and_finishes_without_upload(predictor, action):
    seen = []

    def policy(request):
        seen.append(request)
        assert request.allowed_input_reasons == []
        assert not request.available_refs.get("upload")
        assert request.tools == [] and not request.can_upgrade_workflow
        return action if len(seen) == 1 else FinishAction(result_kind="score_definition")

    runtime = make_runtime(predictor, policy)
    try:
        result = await run_score(runtime)
        assert result.status == "completed" and result.final_result["kind"] == "score_definition"
        assert result.final_result["source"]["score_semantics"] == "synthetic_anomaly_model_score"
        assert result.tool_calls == result.delegation_count == result.workflow_upgrades == 0
        journal = events(runtime, result)
        assert journal[-1].event == "run.completed"
        assert any(e.event == "publication.checked" for e in journal)
        assert not any(
            e.event in {"skill.dispatched", "route.upgraded", "run.awaiting_input"} for e in journal
        )
        if isinstance(action, RequestInputAction):
            assert seen[1].observations[-1].error_code == "input_not_required"
    finally:
        await close(runtime)


@pytest.mark.parametrize("profile", ["standard", "dashscope"])
async def test_score_wire_excludes_input_and_completes_with_model_card(predictor, profile):
    calls = []

    def handle(request):
        wire = json.loads(request.content)
        calls.append(wire)
        assert [t["function"]["name"] for t in wire["tools"]] == ["finish"]
        state = json.loads(wire["messages"][1]["content"])
        assert state["allowed_input_reasons"] == []
        return httpx.Response(200, json=response())

    runtime = make_runtime(predictor, backend=live_backend(handle, api_profile=profile))
    try:
        result = await run_score(runtime)
        assert result.status == "completed" and len(calls) == 1
        assert result.final_result["source"]["type"] == "model_card"
    finally:
        await close(runtime)


async def test_missing_upload_remains_allowed_without_router(predictor):
    seen = []

    def policy(request):
        seen.append(request)
        assert request.allowed_input_reasons == ["missing_upload"]
        tools, _ = _tools_for(request)
        schema = next(
            t["function"]["parameters"] for t in tools if t["function"]["name"] == "request_input"
        )
        assert schema["properties"]["reason"]["enum"] == ["missing_upload"]
        return RequestInputAction(reason="missing_upload")

    runtime = make_runtime(predictor, policy)
    try:
        queued = await runtime.create(request_for(runtime), owner_id="owner", request_id="missing")
        result = await complete(runtime, queued)
        assert seen and result.status == "awaiting_input" and result.final_result is None
    finally:
        await close(runtime)


async def test_valid_upload_cannot_be_requested_again(predictor, csv_bytes):
    seen = []

    def policy(request):
        seen.append(request)
        if len(seen) == 1:
            assert request.allowed_input_reasons == []
            return RequestInputAction(reason="missing_upload")
        return scripted_analysis(request)

    runtime = make_runtime(predictor, policy)
    try:
        queued = await runtime.create(
            request_for(runtime, csv_bytes), owner_id="owner", request_id="present"
        )
        result = await complete(runtime, queued)
        assert result.status == "completed" and result.tool_calls == 5
        assert seen[1].observations[-1].error_code == "input_not_required"
    finally:
        await close(runtime)


async def test_score_objective_cannot_dispatch_even_without_router(predictor, csv_bytes):
    seen = []

    def policy(request):
        seen.append(request)
        assert request.tools == [] and request.allowed_delegates == []
        if len(seen) == 1:
            return CallSkillAction(
                skill="csv_analysis", arguments={"upload_ref": request.available_refs["upload"][0]}
            )
        return FinishAction(result_kind="score_definition")

    runtime = make_runtime(predictor, policy)
    try:
        queued = await runtime.create(
            request_for(runtime, csv_bytes, objective="score_definition"),
            owner_id="owner",
            request_id="unrouted-score",
        )
        result = await complete(runtime, queued)
        assert result.status == "completed" and result.tool_calls == 0
        assert seen[1].observations[-1].error_code == "permission_denied"
    finally:
        await close(runtime)


async def test_data_error_allows_replacement_but_schema_error_does_not(predictor, csv_bytes):
    for content, extra in ((b"bad,column\n1,2\n", False), (csv_bytes, True)):
        seen = []

        def policy(request):
            seen.append(request)
            if len(seen) == 1:
                args = {"upload_ref": request.available_refs["upload"][0]}
                if extra:
                    args["unexpected"] = "private"
                return CallSkillAction(skill="csv_analysis", arguments=args)
            if len(seen) == 2:
                assert request.allowed_input_reasons == ([] if extra else ["invalid_data"])
                return RequestInputAction(reason="invalid_data")
            return scripted_analysis(request)

        runtime = make_runtime(predictor, policy)
        try:
            queued = await runtime.create(
                request_for(runtime, content), owner_id="owner", request_id="data"
            )
            result = await complete(runtime, queued)
            assert result.status == ("completed" if extra else "awaiting_input")
            if extra:
                assert seen[2].observations[-1].error_code == "input_not_required"
        finally:
            await close(runtime)


async def test_real_ambiguous_input_still_asks_for_clarification_without_model(predictor):
    runtime = make_runtime(predictor, lambda _: pytest.fail("Router handles real ambiguity"))
    runtime.router = HealthRouter()
    try:
        queued = await runtime.create(
            request_for(runtime).model_copy(update={"query": "继续"}),
            owner_id="owner",
            request_id="ambiguous",
        )
        result = await complete(runtime, queued)
        assert result.status == "awaiting_input" and result.error_code == "clarify_task"
        assert result.decision_count == 0
    finally:
        await close(runtime)
