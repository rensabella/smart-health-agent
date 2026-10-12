import asyncio
import json
import threading
from uuid import uuid4

import pytest
from agent_helpers import scripted_analysis

from app.agents.backends import DecisionBackend, ScriptedDecisionBackend
from app.agents.contracts import CallSkillAction, CreateRunRequest, DecisionReply, FinishAction
from app.agents.events import RunEventStore
from app.agents.runtime import AgentLimits, AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.context import SkillContext
from app.core.errors import SkillError
from app.services.analysis_service import AnalysisService
from skills.bootstrap import create_registry


def make_runtime(predictor, policy=scripted_analysis, *, limits=None, events=None, backend=None):
    service = AnalysisService(ArtifactStore(), predictor)
    registry = create_registry(service)
    runtime = AgentRuntime(
        registry=registry,
        service=service,
        backend=backend or ScriptedDecisionBackend(policy),
        events=events or RunEventStore(),
        limits=limits,
    )
    return runtime


def request_for(runtime, csv_bytes=None, *, session_id=None, input_version=None, **kwargs):
    session_id, input_version = session_id or uuid4().hex, input_version or uuid4().hex
    refs = []
    if csv_bytes:
        context = SkillContext(owner_id="owner", session_id=session_id, input_version=input_version)
        refs.append(runtime.service.create_upload(csv_bytes, context))
    return CreateRunRequest(
        query="Analyze synthetic input",
        session_id=session_id,
        input_version=input_version,
        upload_refs=refs,
        **kwargs,
    )


async def complete(runtime, snapshot):
    task = runtime._tasks.get(snapshot.run_id)
    if task:
        await asyncio.wait_for(asyncio.shield(task), 30)
    return runtime.snapshot(snapshot.run_id, owner_id="owner", session_id=snapshot.session_id)


async def close(runtime):
    await runtime.aclose()
    await runtime.registry.aclose()


@pytest.mark.parametrize("entry", ["data_analysis", "planner"])
async def test_real_skills_react_and_planner_returns_to_parent(predictor, csv_bytes, entry):
    runtime = make_runtime(predictor)
    try:
        request = request_for(runtime, csv_bytes, entry_agent=entry, include_visualizations=True)
        initial = await runtime.create(request, owner_id="owner", request_id="react-test")
        result = await complete(runtime, initial)
        assert result.status == "completed", result.model_dump()
        assert result.tool_calls == 6 and result.llm_requests == 0
        assert result.backend_mode == "scripted"
        assert result.final_result["analysis"]["prediction"]["window_count"] == 4
        events = runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)
        assert sum(e.event == "skill.completed" for e in events) == 6
        if entry == "planner":
            assert result.delegation_count == 1
            child_done = next(
                i
                for i, e in enumerate(events)
                if e.event == "agent.completed" and e.agent == "data_analysis"
            )
            assert any(
                e.agent == "planner" and e.event == "agent.decision"
                for e in events[child_done + 1 :]
            )
        assert events[-1].event == "run.completed"
        assert any(e.event == "publication.checked" for e in events)
    finally:
        await close(runtime)


async def test_observation_changes_next_action_after_bad_input(predictor):
    runtime = make_runtime(predictor)
    try:
        request = request_for(runtime, b"bad,column\n1,2\n")
        result = await complete(
            runtime, await runtime.create(request, owner_id="owner", request_id="bad-data")
        )
        assert result.status == "awaiting_input" and result.tool_calls == 1
        assert result.decision_count == 2 and result.final_result is None
        assert "CSV" in result.message
    finally:
        await close(runtime)


async def test_recovery_from_invalid_arguments_and_reference_rejection(predictor, csv_bytes):
    seen = []

    def policy(state):
        seen.append(state)
        if not state.observations:
            return CallSkillAction(
                skill="csv_analysis", arguments={"upload_ref": "art_" + "f" * 32}
            )
        if len(state.observations) == 1:
            return CallSkillAction(
                skill="csv_analysis",
                arguments={
                    "upload_ref": state.available_refs["upload"][0],
                    "file_path": "SECRET.csv",
                },
            )
        return scripted_analysis(state)

    runtime = make_runtime(predictor, policy)
    try:
        initial = await runtime.create(
            request_for(runtime, csv_bytes), owner_id="owner", request_id="recover"
        )
        result = await complete(runtime, initial)
        assert result.status == "completed" and result.tool_calls == 5
        assert seen[1].observations[-1].error_code == "unobserved_reference"
        assert seen[2].observations[-1].error_code == "invalid_input"
        trace = runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)
        assert "SECRET" not in json.dumps([e.model_dump() for e in trace])
    finally:
        await close(runtime)


async def test_foreign_run_finish_ref_and_unauthorized_tools_cannot_publish(predictor):
    def policy(state):
        if not state.observations:
            return CallSkillAction(skill="SECRET_TOOL", arguments={"query": "SECRET_PAYLOAD"})
        return FinishAction(result_kind="data_analysis", result_ref="art_" + "a" * 32)

    runtime = make_runtime(predictor, policy, limits=AgentLimits(max_decisions=3))
    try:
        result = await complete(
            runtime, await runtime.create(request_for(runtime), owner_id="owner", request_id="refs")
        )
        assert result.status == "blocked" and result.error_code == "decision_budget_exceeded"
        assert result.final_result is None and result.tool_calls == 0
        serialized = json.dumps(
            [
                e.model_dump()
                for e in runtime.get_events(
                    result.run_id, owner_id="owner", session_id=result.session_id
                )
            ]
        )
        assert "SECRET" not in serialized
    finally:
        await close(runtime)


@pytest.mark.parametrize("finish_kind", ["data_analysis", "health_report", "score_definition"])
async def test_health_report_cannot_be_downgraded_or_published(predictor, finish_kind):
    runtime = make_runtime(
        predictor,
        lambda r: FinishAction(result_kind=finish_kind),
        limits=AgentLimits(max_decisions=2),
    )
    try:
        result = await complete(
            runtime,
            await runtime.create(
                request_for(runtime, objective="health_report"),
                owner_id="owner",
                request_id="safety",
            ),
        )
        assert result.status == "blocked" and result.final_result is None
    finally:
        await close(runtime)


async def test_idempotency_session_scope_and_cancel_waiting_model(predictor):
    entered = asyncio.Event()

    async def wait(state):
        entered.set()
        await asyncio.Event().wait()

    runtime = make_runtime(predictor, wait)
    try:
        request = request_for(runtime)
        first, second = await asyncio.gather(
            *[
                runtime.create(
                    request, owner_id="owner", request_id="idempotent", idempotency_key="same"
                )
                for _ in range(2)
            ]
        )
        assert first.run_id == second.run_id and len(runtime._tasks) == 1
        await entered.wait()
        with pytest.raises(SkillError) as error:
            await runtime.create(
                request.model_copy(update={"query": "new query"}),
                owner_id="owner",
                request_id="changed",
                idempotency_key="same",
            )
        assert error.value.code == "idempotency_conflict"
        with pytest.raises(SkillError) as error:
            await runtime.create(request, owner_id="owner", request_id="busy")
        assert error.value.code == "session_busy"
        with pytest.raises(SkillError):
            runtime.snapshot(first.run_id, owner_id="attacker", session_id=first.session_id)
        result = runtime.cancel(first.run_id, owner_id="owner", session_id=first.session_id)
        assert result.status == "cancelled" and result.final_result is None
        assert (await complete(runtime, result)).status == "cancelled"
        assert (
            runtime.cancel(first.run_id, owner_id="owner", session_id=first.session_id).status
            == "cancelled"
        )
    finally:
        await close(runtime)


async def test_cancel_during_native_skill_prevents_late_write(predictor, csv_bytes, monkeypatch):
    runtime = make_runtime(predictor)
    entered, release = threading.Event(), threading.Event()
    original = runtime.service.inspect_csv

    def slow(ref, context):
        entered.set()
        release.wait(3)
        return original(ref, context)

    monkeypatch.setattr(runtime.service, "inspect_csv", slow)
    try:
        first = await runtime.create(
            request_for(runtime, csv_bytes), owner_id="owner", request_id="cancel-skill"
        )
        assert await asyncio.to_thread(entered.wait, 2)
        runtime.cancel(first.run_id, owner_id="owner", session_id=first.session_id)
        release.set()
        result = await complete(runtime, first)
        await runtime.registry.aclose()
        assert result.status == "cancelled" and result.final_result is None
        assert runtime.service.store.item_count == 1
        assert not runtime._runs[first.run_id].refs.get("dataset")
    finally:
        release.set()
        await close(runtime)


async def test_shared_tool_budget_terminates_delegated_work(predictor, csv_bytes):
    runtime = make_runtime(predictor, limits=AgentLimits(max_tool_calls=2))
    try:
        initial = await runtime.create(
            request_for(runtime, csv_bytes, entry_agent="planner"),
            owner_id="owner",
            request_id="budget",
        )
        result = await complete(runtime, initial)
        assert result.status == "blocked" and result.error_code == "budget_exceeded"
        assert result.tool_calls == 2 and result.delegation_count == 1
    finally:
        await close(runtime)


async def test_live_retry_is_accounted_and_bounded(predictor):
    class FakeLive(DecisionBackend):
        mode = "live"

        def __init__(self):
            self.calls = 0

        async def decide(self, state):
            self.calls += 1
            if self.calls == 1:
                raise SkillError("backend_rate_limited", "Controlled failure")
            return DecisionReply(
                decision=FinishAction(result_kind="score_definition"),
                input_tokens=20,
                output_tokens=5,
            )

        async def aclose(self):
            pass

    backend = FakeLive()
    runtime = make_runtime(predictor, backend=backend)
    try:
        initial = await runtime.create(
            request_for(runtime, objective="score_definition"), owner_id="owner", request_id="retry"
        )
        result = await complete(runtime, initial)
        assert result.status == "completed" and result.llm_requests == backend.calls == 2
        assert result.input_tokens == 20 and result.output_tokens == 5
    finally:
        await close(runtime)


async def test_run_deadline_blocks_publication(predictor):
    async def wait(state):
        await asyncio.sleep(10)

    runtime = make_runtime(predictor, wait, limits=AgentLimits(run_timeout_seconds=0.03))
    try:
        initial = await runtime.create(request_for(runtime), owner_id="owner", request_id="timeout")
        result = await complete(runtime, initial)
        assert result.status == "failed" and result.error_code == "timeout"
    finally:
        await close(runtime)


async def test_same_session_valid_artifact_from_another_run_is_rejected(predictor, csv_bytes):
    runtime = make_runtime(predictor)
    try:
        request = request_for(runtime, csv_bytes)
        first = await complete(
            runtime, await runtime.create(request, owner_id="owner", request_id="first")
        )
        ref = first.final_result["analysis"]["analysis_ref"]
        runtime.backend = ScriptedDecisionBackend(
            lambda r: FinishAction(result_kind="data_analysis", result_ref=ref)
        )
        runtime.limits = AgentLimits(max_decisions=2)
        second = await complete(
            runtime, await runtime.create(request, owner_id="owner", request_id="second")
        )
        assert second.status == "blocked" and second.tool_calls == 0 and second.final_result is None
        # The artifact still exists; it is the run boundary that rejects it.
        assert runtime.service.store.get(
            ref, kind="analysis", context=runtime._runs[first.run_id].context
        )
    finally:
        await close(runtime)


async def test_missing_required_plot_prevents_finish(predictor, csv_bytes):
    def skip_plots(state):
        if state.available_refs.get("analysis"):
            return FinishAction(
                result_kind="data_analysis", result_ref=state.available_refs["analysis"][-1]
            )
        return scripted_analysis(state)

    runtime = make_runtime(predictor, skip_plots, limits=AgentLimits(max_decisions=7))
    try:
        result = await complete(
            runtime,
            await runtime.create(
                request_for(runtime, csv_bytes, include_visualizations=True),
                owner_id="owner",
                request_id="plot",
            ),
        )
        assert result.status == "blocked" and result.final_result is None
        events = runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)
        assert any(e.data.get("error_code") == "missing_visualization" for e in events)
    finally:
        await close(runtime)


@pytest.mark.parametrize("failure", ["cancel", "audit"])
async def test_publication_gate_fails_closed(predictor, failure):
    class GateStore(RunEventStore):
        runtime = None

        def append(self, run_id, event, **kwargs):
            if event == "run.completed" and failure == "audit":
                raise SkillError("trace_unavailable", "Controlled audit failure")
            result = super().append(run_id, event, **kwargs)
            if event == "publication.checked" and failure == "cancel":
                self.runtime._runs[run_id].context.cancelled.set()
            return result

    events = GateStore()
    runtime = make_runtime(predictor, events=events)
    events.runtime = runtime
    try:
        result = await complete(
            runtime,
            await runtime.create(
                request_for(runtime, objective="score_definition"),
                owner_id="owner",
                request_id="gate",
            ),
        )
        assert result.final_result is None
        assert result.status == ("cancelled" if failure == "cancel" else "failed")
        assert not any(
            e.event == "run.completed"
            for e in runtime.get_events(
                result.run_id, owner_id="owner", session_id=result.session_id
            )
        )
    finally:
        await close(runtime)


async def test_result_expiry_and_retained_capacity_are_bounded(predictor):
    runtime = make_runtime(
        predictor, limits=AgentLimits(max_retained_runs=1, result_ttl_seconds=1.0)
    )
    try:
        first = await complete(
            runtime,
            await runtime.create(
                request_for(runtime, objective="score_definition"),
                owner_id="owner",
                request_id="old",
            ),
        )
        runtime._runs[first.run_id].finished_at -= 2
        expired = runtime.snapshot(first.run_id, owner_id="owner", session_id=first.session_id)
        assert expired.status == "completed" and expired.final_result is None
        second = await complete(
            runtime,
            await runtime.create(
                request_for(runtime, objective="score_definition"),
                owner_id="owner",
                request_id="new",
            ),
        )
        assert second.status == "completed" and len(runtime._runs) == 1
        with pytest.raises(SkillError) as error:
            runtime.snapshot(first.run_id, owner_id="owner", session_id=first.session_id)
        assert error.value.code == "run_not_found"
    finally:
        await close(runtime)


async def test_queued_cancel_and_wait_do_not_cancel_the_caller(predictor):
    runtime = make_runtime(predictor)
    try:
        first = await runtime.create(
            request_for(runtime, objective="score_definition"),
            owner_id="owner",
            request_id="queued",
        )
        runtime.cancel(first.run_id, owner_id="owner", session_id=first.session_id)
        result = await runtime.wait(first.run_id, owner_id="owner", session_id=first.session_id)
        assert result.status == "cancelled" and result.decision_count == 0
    finally:
        await close(runtime)


async def test_real_adapter_with_mock_http_runs_actual_skills(predictor, csv_bytes):
    import httpx

    from app.agents.backends import OpenAICompatibleBackend
    from app.agents.contracts import DecisionRequest

    calls = []

    def transport(request):
        payload = json.loads(request.content)
        state = DecisionRequest.model_validate(
            {
                "run_id": "protocol_fixture",
                "tools": [],
                **json.loads(payload["messages"][1]["content"]),
            }
        )
        action = scripted_analysis(state)
        name = action.skill if isinstance(action, CallSkillAction) else action.action
        arguments = (
            action.arguments
            if isinstance(action, CallSkillAction)
            else action.model_dump(exclude={"action"})
        )
        calls.append(name)
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
                                    "type": "function",
                                    "id": "mock_call",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    backend = OpenAICompatibleBackend(
        "https://protocol.example/v1",
        "test-key",
        "protocol-fixture-model",
        transport=httpx.MockTransport(transport),
    )
    runtime = make_runtime(predictor, backend=backend)
    try:
        result = await complete(
            runtime,
            await runtime.create(
                request_for(runtime, csv_bytes, entry_agent="planner"),
                owner_id="owner",
                request_id="protocol",
            ),
        )
        assert result.status == "completed" and result.tool_calls == 5
        assert result.llm_requests == len(calls) == 8
        assert result.decision_model == "protocol-fixture-model"
        assert result.input_tokens == 160 and result.output_tokens == 80
    finally:
        await close(runtime)


async def test_runtime_recovery_is_read_only_and_does_not_resume_llm(predictor, tmp_path):
    events = RunEventStore(tmp_path)
    events.register(
        "interrupted",
        owner_id="owner",
        session_id="session",
        metadata={
            "request_id": "request1",
            "input_version": "v1",
            "backend_mode": "scripted",
            "objective": "data_analysis",
            "entry_agent": "data_analysis",
        },
    )
    events.append("interrupted", "run.running", data={"status": "running"})
    runtime = make_runtime(predictor, events=RunEventStore(tmp_path))
    try:
        runtime.recover()
        result = runtime.snapshot("interrupted", owner_id="owner", session_id="session")
        assert result.status == "failed" and result.error_code == "process_restarted"
        assert result.final_result is None and not runtime._tasks
    finally:
        await close(runtime)
