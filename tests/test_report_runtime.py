import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rag_helpers import ScriptedEmbedding
from test_agent_runtime import request_for

from app.agents.backends import ScriptedDecisionBackend
from app.agents.contracts import CallSkillAction, FinishAction
from app.agents.events import RunEventStore
from app.agents.router import HealthRouter
from app.agents.runtime import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.harness.config import HarnessConfig, load_config
from app.harness.harness_validator import HarnessValidator
from app.main import create_app
from app.memory.long_term import QdrantLongTermStore
from app.memory.memory_manager import MemoryManager
from app.rag.retrieval import RetrievalService, build_index
from app.reports.service import ReportService
from app.services.analysis_service import AnalysisService
from examples.report_policy import scripted_report
from skills.bootstrap import create_registry


@pytest.fixture
def retrieval(tmp_path):
    path = tmp_path / "index"
    build_index(Path("knowledge/corpus.json"), path, ScriptedEmbedding())
    service = RetrievalService(path, ScriptedEmbedding(), min_score=0.2)
    yield service
    service.close()


def make(predictor, retrieval, policy=scripted_report, config=None):
    harness = HarnessValidator(config if config else load_config("configs/report"))
    store = ArtifactStore()
    service = AnalysisService(
        store,
        predictor,
        retrieval=retrieval,
        reports=ReportService(store, predictor, validator=harness),
    )
    return AgentRuntime(
        service=service,
        registry=create_registry(service, harness=harness),
        backend=ScriptedDecisionBackend(policy),
        events=RunEventStore(),
        router=HealthRouter(),
        harness=harness,
        memory=MemoryManager(long_term=QdrantLongTermStore()),
    )


async def execute(runtime, content, **changes):
    request = request_for(runtime, content).model_copy(
        update={"query": "分析CSV并生成有来源的健康报告", **changes}
    )
    queued = await runtime.create(request, owner_id="owner", request_id="report-test")
    return await runtime.wait(queued.run_id, owner_id="owner", session_id=queued.session_id)


async def close(runtime):
    await runtime.aclose()
    await runtime.registry.aclose()
    await runtime.memory.aclose()


async def test_five_roles_produce_reviewed_report_and_metadata_only_journal(
    predictor, csv_bytes, retrieval
):
    runtime = make(predictor, retrieval)
    try:
        result = await execute(runtime, csv_bytes)
        assert result.status == "completed", result.model_dump()
        assert result.tool_calls == 8 and result.delegation_count == 4
        assert result.llm_requests == 0 and result.final_result["review_mode"] == "scripted_test"
        assert result.memory_status == "stored"
        assert (
            runtime.memory.long_term.list_analysis("owner")[0].analysis.risk_score
            == result.final_result["analysis"]["prediction"]["risk_score"]
        )
        events = runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)
        assert {e.agent for e in events if e.event == "agent.started"} == {
            "planner",
            "data_analysis",
            "rag",
            "report_writer",
            "safety_verification",
        }
        verified = next(i for i, e in enumerate(events) if e.event == "publication.checked")
        chunks = [e for e in events if e.event == "report.chunk"]
        assert chunks and all(events.index(e) > verified for e in chunks)
        body = "".join(
            runtime.report_chunk(
                result.run_id,
                owner_id="owner",
                session_id=result.session_id,
                index=e.data["chunk_index"],
                report_hash=e.data["report_hash"],
            )
            for e in chunks
        )
        assert body == result.final_result["report"]["body"]
        assert "身体活动和兴奋" not in json.dumps(
            [e.model_dump() for e in events], ensure_ascii=False
        )
    finally:
        await close(runtime)


@pytest.mark.parametrize("reject_count", [1, 3])
async def test_failed_review_revises_bounded_draft_then_blocks_or_publishes(
    predictor, csv_bytes, retrieval, reject_count
):
    def policy(state):
        action = scripted_report(state)
        if (
            isinstance(action, CallSkillAction)
            and action.skill == "safety_verification"
            and state.report_context["draft_version"] <= reject_count
        ):
            for review in action.arguments["reviews"]:
                review.update(supported=False, reason="insufficient_support")
        return action

    runtime = make(predictor, retrieval, policy)
    try:
        result = await execute(runtime, csv_bytes)
        if reject_count == 1:
            assert result.status == "completed", result.model_dump()
            assert result.final_result["report"]["safety"]["draft_version"] == 2
        else:
            assert result.status == "blocked", result.model_dump()
            assert result.error_code == "safety_review_failed"
            assert result.final_result is None
            assert runtime.memory.long_term.list_analysis("owner") == []
            assert not any(
                e.event == "report.chunk"
                for e in runtime.get_events(
                    result.run_id, owner_id="owner", session_id=result.session_id
                )
            )
    finally:
        await close(runtime)


async def test_writer_cannot_review_and_planner_cannot_publish_early(
    predictor, csv_bytes, retrieval
):
    attempted = set()

    def policy(state):
        if state.agent == "planner" and "early" not in attempted:
            attempted.add("early")
            return FinishAction(result_kind="health_report", result_ref="art_" + "a" * 32)
        if state.agent == "report_writer" and "review" not in attempted:
            attempted.add("review")
            return CallSkillAction(skill="safety_verification", arguments={"safe": True})
        return scripted_report(state)

    runtime = make(predictor, retrieval, policy)
    try:
        result = await execute(runtime, csv_bytes)
        assert result.status == "completed", result.model_dump()
        assert result.tool_calls == 8
        events = runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)
        codes = [e.data.get("error_code") for e in events if e.event == "agent.action_rejected"]
        assert "permission_denied" in codes and "unreviewed_report" in codes
        assert (
            next(e for e in events if e.event == "safety.reviewed").agent == "safety_verification"
        )
    finally:
        await close(runtime)


async def test_cancellation_during_safety_never_streams_or_stores_report(
    predictor, csv_bytes, retrieval
):
    entered = asyncio.Event()

    async def policy(state):
        if state.agent == "safety_verification":
            entered.set()
            await asyncio.Event().wait()
        return scripted_report(state)

    runtime = make(predictor, retrieval, policy)
    try:
        request = request_for(runtime, csv_bytes).model_copy(
            update={"query": "分析CSV并生成健康报告"}
        )
        queued = await runtime.create(request, owner_id="owner", request_id="cancel-review")
        await asyncio.wait_for(entered.wait(), 10)
        runtime.cancel(queued.run_id, owner_id="owner", session_id=queued.session_id)
        result = await runtime.wait(queued.run_id, owner_id="owner", session_id=queued.session_id)
        assert result.status == "cancelled" and result.final_result is None
        assert runtime.memory.long_term.list_analysis("owner") == []
        events = runtime.get_events(result.run_id, owner_id="owner", session_id=result.session_id)
        assert not any(e.event in {"report.chunk", "publication.checked"} for e in events)
    finally:
        await close(runtime)


async def test_report_reserves_last_tool_for_safety(predictor, csv_bytes, retrieval):
    config = load_config("configs/report").model_dump()
    config["agent"]["limits"]["max_tool_calls"] = 7
    runtime = make(predictor, retrieval, config=HarnessConfig.model_validate(config))
    try:
        result = await execute(runtime, csv_bytes)
        assert result.status == "blocked" and result.error_code == "budget_exceeded"
        assert result.tool_calls == 6 and result.final_result is None
        assert not runtime._runs[result.run_id].refs.get("draft")
    finally:
        await close(runtime)


async def test_insufficient_evidence_cannot_be_downgraded_to_success(
    predictor, csv_bytes, retrieval
):
    retrieval.min_score = 1.0

    # Query never exactly equals a multi-signal document in this test vectorizer.
    def policy(state):
        action = scripted_report(state)
        if isinstance(action, CallSkillAction) and action.skill == "rag_retrieval":
            action.arguments["query"] = "unrelated galaxy orbit"
        return action

    runtime = make(predictor, retrieval, policy)
    try:
        result = await execute(runtime, csv_bytes)
        assert result.status == "awaiting_input", result.model_dump()
        assert result.final_result is None and result.tool_calls == 6
        assert runtime.memory.long_term.list_analysis("owner") == []
    finally:
        await close(runtime)


async def test_missing_or_disabled_report_dependencies_do_no_work(predictor, csv_bytes, retrieval):
    runtime = make(predictor, retrieval)
    runtime.service.retrieval = None
    try:
        result = await execute(runtime, csv_bytes)
        assert result.status == "blocked" and result.error_code == "safety_unavailable"
        assert result.decision_count == result.tool_calls == 0
    finally:
        await close(runtime)


def test_http_sse_only_streams_approved_body_and_replays(predictor, csv_bytes, retrieval):
    app = create_app(
        predictor=predictor,
        retrieval_service=retrieval,
        harness_config=load_config("configs/report"),
        decision_backend=ScriptedDecisionBackend(scripted_report),
        agent_event_store=RunEventStore(),
    )
    with TestClient(app) as client:
        upload = client.post("/uploads", files={"file": ("demo.csv", csv_bytes)}).json()
        created = client.post(
            "/agent/runs",
            json={
                "query": "分析并生成健康报告",
                "session_id": upload["session_id"],
                "input_version": upload["input_version"],
                "upload_refs": [upload["upload_ref"]],
            },
        ).json()
        response = client.get(created["events_url"])
        assert response.status_code == 200
        events = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        chunks = [e for e in events if e["event"] == "report.chunk"]
        final = client.get(
            f"/agent/runs/{created['run_id']}?session_id={created['session_id']}"
        ).json()
        assert final["status"] == "completed", final
        assert all(c["data"]["content_available"] for c in chunks)
        assert "".join(c["data"]["text"] for c in chunks) == final["final_result"]["report"]["body"]
        replay = client.get(
            created["events_url"], headers={"Last-Event-ID": str(chunks[0]["event_id"] - 1)}
        )
        assert chunks[0]["data"]["text"] in "".join(
            json.loads(line[6:])["data"].get("text", "")
            for line in replay.text.splitlines()
            if line.startswith("data: ")
        )
