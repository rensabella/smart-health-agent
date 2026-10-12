"""Memory is background context, never a source of current measurements or authority."""

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_agent_runtime import complete, make_runtime, request_for
from test_long_term_memory import Clock, analysis

from app.agents.backends import ScriptedDecisionBackend
from app.agents.contracts import FinishAction
from app.agents.events import RunEventStore
from app.agents.router import HealthRouter
from app.agents.runtime import AgentLimits
from app.core.errors import SkillError
from app.main import create_app
from app.memory.long_term import COLLECTION_NAME, QdrantLongTermStore
from app.memory.memory_manager import MemoryManager, score_summary
from app.memory.schemas import Preferences, TurnSummary
from examples.demo_policy import scripted_analysis


def make(predictor, policy=scripted_analysis, *, store=None, clock=None):
    long = store if store is not None else QdrantLongTermStore()
    manager = MemoryManager(long_term=long, **({"clock": clock} if clock else {}))
    runtime = make_runtime(predictor, policy)
    runtime.router = HealthRouter()
    runtime.memory = manager
    return runtime


def body(runtime, csv_bytes=None, query="分析CSV数据", **kwargs):
    return request_for(runtime, csv_bytes, **kwargs).model_copy(update={"query": query})


async def run(runtime, request):
    initial = await runtime.create(request, owner_id="owner", request_id="memory-test")
    return await complete(runtime, initial)


async def shutdown(runtime):
    await runtime.aclose()
    await runtime.registry.aclose()
    await runtime.memory.aclose()


async def test_completed_analysis_stores_only_whitelisted_summary(predictor, csv_bytes):
    runtime = make(predictor)
    try:
        request = body(runtime, csv_bytes, query="分析CSV数据 PRIVATE_PATIENT_TEXT")
        result = await run(runtime, request)
        assert result.status == "completed" and result.memory_status == "stored", (
            result.model_dump()
        )
        entries = runtime.memory.long_term.list_analysis("owner")
        assert len(entries) == 1 and entries[0].source_run_id == result.run_id
        assert (
            entries[0].analysis.risk_score
            == result.final_result["analysis"]["prediction"]["risk_score"]
        )
        assert entries[0].analysis.task_scope == "synthetic_anomaly_only"
        points, _ = runtime.memory.long_term._client.scroll(COLLECTION_NAME, with_vectors=True)
        serialized = json.dumps([p.model_dump() for p in points])
        for forbidden in (
            "PRIVATE_PATIENT_TEXT",
            "heart_rate_mean",
            "observed_value",
            "local_evidence",
            "window_predictions",
            "upload_ref",
            "analysis_ref",
            "session_id",
        ):
            assert forbidden not in serialized
        assert len(points[0].vector) == 256
        short = runtime.memory.short_term.get("owner", result.session_id)
        assert len(short.turns) == 1 and short.current_result_run_id == result.run_id
        assert short.execution.last_event_id == result.last_event_id
        assert short.execution.tool_calls == 5
        # Model/provider text cannot become a summary by adding fields to a typed object.
        with pytest.raises(ValidationError):
            type(entries[0]).model_validate(
                {**entries[0].model_dump(), "raw_csv": csv_bytes.decode()}
            )
    finally:
        await shutdown(runtime)


async def test_latest_five_turns_and_scoped_history_are_injected_as_background(
    predictor, csv_bytes
):
    observed = []

    def policy(state):
        observed.append(state)
        return scripted_analysis(state)

    runtime = make(predictor, policy)
    try:
        original = await run(runtime, body(runtime, csv_bytes))
        await runtime.memory.set_preferences("owner", Preferences(language="en", detail="brief"))
        await runtime.memory.set_preferences("outsider", Preferences(language="zh-CN"))
        for _ in range(6):
            result = await run(
                runtime,
                body(
                    runtime,
                    query="解释risk score",
                    session_id=original.session_id,
                    input_version=original.input_version,
                ),
            )
            assert result.memory_status == "stored"
        state = observed[-1]
        assert len(state.memory.recent_turns) == 5 and state.memory.historical_only
        assert state.memory.preferences == Preferences(language="en", detail="brief")
        assert len(state.memory.historical_summaries) == 1
        assert state.memory.historical_summaries[0].entry.source_run_id == original.run_id
        assert state.available_refs.get("analysis", []) == []
        assert state.available_refs.get("published_analysis", []) == []
        session = runtime.memory.short_term.get("owner", original.session_id)
        assert len(session.turns) == 5
        assert original.run_id not in [turn.run_id for turn in session.turns]
        assert len(runtime.memory.long_term.list_analysis("owner")) == 1
        unrelated = await runtime.memory.context(
            owner_id="outsider", session_id=original.session_id, query="risk score"
        )
        assert not unrelated.recent_turns and not unrelated.historical_summaries
        assert unrelated.preferences.language == "zh-CN"
    finally:
        await shutdown(runtime)


async def test_history_cannot_satisfy_new_analysis_or_missing_upload(predictor, csv_bytes):
    runtime = make(
        predictor,
        lambda state: FinishAction(result_kind="data_analysis", result_ref="art_" + "0" * 32),
    )
    try:
        runtime.memory.long_term.put_analysis(
            "owner",
            analysis(created=runtime.memory.clock() - 2, expiry=runtime.memory.clock() + 60),
        )
        result = await run(runtime, body(runtime))
        assert result.status == "awaiting_input" and result.error_code == "missing_upload"
        assert result.final_result is None and result.decision_count == 0
        result = await run(runtime, body(runtime, csv_bytes))
        assert result.status == "blocked" and result.final_result is None
        assert len(runtime.memory.long_term.list_analysis("owner")) == 1
    finally:
        await shutdown(runtime)


async def test_failed_cancelled_blocked_and_reused_runs_do_not_add_long_term_rows(
    predictor, csv_bytes
):
    runtime = make(predictor)
    try:
        original = await run(runtime, body(runtime, csv_bytes))
        reused = await run(
            runtime,
            body(
                runtime,
                query="显示上次结果",
                session_id=original.session_id,
                input_version=original.input_version,
                source_run_id=original.run_id,
            ),
        )
        assert reused.status == "completed" and reused.memory_status == "stored"
        blocked = await run(runtime, body(runtime, query="这个用户为什么风险高"))
        invalid = await run(runtime, body(runtime, b"bad,column\n1,2\n"))
        assert blocked.status == "blocked" and invalid.status == "awaiting_input"
        initial = await runtime.create(
            body(runtime, query="解释risk score"),
            owner_id="owner",
            request_id="cancel-before-start",
        )
        runtime.cancel(initial.run_id, owner_id="owner", session_id=initial.session_id)
        cancelled = await runtime.wait(
            initial.run_id, owner_id="owner", session_id=initial.session_id
        )
        assert cancelled.status == "cancelled"
        assert (
            runtime.memory.short_term.get("owner", initial.session_id).turns[0].assistant_status
            == "cancelled"
        )
        assert [e.source_run_id for e in runtime.memory.long_term.list_analysis("owner")] == [
            original.run_id
        ]
    finally:
        await shutdown(runtime)


async def test_delete_during_decision_revokes_lease_and_removes_future_context(
    predictor, csv_bytes
):
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []

    async def policy(state):
        observed.append(state)
        if state.agent == "planner" and not entered.is_set():
            entered.set()
            await release.wait()
        return scripted_analysis(state)

    runtime = make(predictor, policy)
    try:
        await runtime.memory.set_preferences("owner", Preferences(language="en"))
        request = body(runtime, csv_bytes)
        initial = await runtime.create(request, owner_id="owner", request_id="delete-in-flight")
        await asyncio.wait_for(entered.wait(), 2)
        assert observed[0].memory.preferences.language == "en"
        await runtime.memory.delete_owner("owner")
        release.set()
        result = await complete(runtime, initial)
        assert result.status == "completed" and result.memory_status == "skipped"
        assert runtime.memory.short_term.get("owner", request.session_id) is None
        assert runtime.memory.long_term.list_analysis("owner") == []
        assert all(s.memory.preferences is None for s in observed[1:])
    finally:
        release.set()
        await shutdown(runtime)


async def test_memory_write_failure_does_not_relabel_valid_analysis_or_claim_persistence(
    predictor, csv_bytes, monkeypatch, caplog
):
    runtime = make(predictor)

    def unavailable(*args, **kwargs):
        raise OSError("SECRET database details")

    monkeypatch.setattr(runtime.memory.long_term, "put_analysis", unavailable)
    try:
        result = await run(runtime, body(runtime, csv_bytes))
        assert result.status == "completed" and result.final_result is not None
        assert result.memory_status == "unavailable"
        assert runtime.memory.long_term.list_analysis("owner") == []
        assert "SECRET" not in caplog.text
        assert (
            runtime.memory.short_term.get("owner", result.session_id).execution.status
            == "completed"
        )
    finally:
        await shutdown(runtime)


async def test_delete_serializes_with_native_write_even_when_waiter_is_cancelled(monkeypatch):
    clock = Clock()
    store = QdrantLongTermStore(clock=clock, retention_seconds=200)
    manager = MemoryManager(long_term=store, clock=clock)
    started, release = threading.Event(), threading.Event()
    original_put = store.put_analysis

    def slow_put(*args, **kwargs):
        started.set()
        release.wait(5)
        return original_put(*args, **kwargs)

    monkeypatch.setattr(store, "put_analysis", slow_put)
    lease = manager.begin(
        owner_id="owner", session_id="session", input_version="v1", run_id="run1", upload_refs=[]
    )
    turn = TurnSummary(
        run_id="run1",
        input_version="v1",
        user_intent="data_analysis",
        assistant_status="completed",
        completed_at=clock(),
        analysis=analysis().analysis,
    )
    finishing = asyncio.create_task(
        manager.finish(
            owner_id="owner", session_id="session", lease=lease, turn=turn, fresh_analysis=True
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        finishing.cancel()
        deleting = asyncio.create_task(manager.delete_owner("owner"))
        await asyncio.sleep(0)
        assert not deleting.done()
        release.set()
        await asyncio.gather(finishing, return_exceptions=True)
        await deleting
        assert store.list_analysis("owner") == []
        assert manager.short_term.get("owner", "session") is None
    finally:
        release.set()
        await manager.aclose()


async def test_trends_are_chronological_and_separate_model_fingerprints():
    clock = Clock()
    store = QdrantLongTermStore(clock=clock, retention_seconds=200)
    manager = MemoryManager(long_term=store, clock=clock)
    try:
        for name, created, fingerprint in [
            ("late", 998.0, "a"),
            ("early", 991.0, "a"),
            ("other_model", 995.0, "b"),
        ]:
            item = analysis(name, created=created)
            item.analysis.model_fingerprint = fingerprint * 64
            store.put_analysis("owner", item)
        result = await manager.trends("owner")
        assert len(result["groups"]) == 2 and result["historical_only"]
        group = next(g for g in result["groups"] if g["protocol"]["model_fingerprint"] == "a" * 64)
        assert [p["source_run_id"] for p in group["points"]] == ["early", "late"]
        assert (await manager.trends("outsider"))["groups"] == []
    finally:
        await manager.aclose()


def test_api_memory_auth_preference_schema_and_body_limits(predictor):
    app = create_app(predictor=predictor, api_token="token", agent_event_store=RunEventStore())
    with TestClient(app) as client:
        for method, path in [
            ("GET", "/memory/preferences"),
            ("GET", "/memory/trends"),
            ("GET", "/memory/sessions/x"),
            ("DELETE", "/memory"),
            ("DELETE", "/memory/sessions/x"),
        ]:
            assert client.request(method, path).status_code == 401
        h = {"Authorization": "Bearer token"}
        prefs = client.put(
            "/memory/preferences", json={"language": "en", "detail": "brief"}, headers=h
        )
        assert prefs.status_code == 200 and prefs.json()["language"] == "en"
        assert client.get("/memory/preferences", headers=h).json() == prefs.json()
        assert (
            client.put(
                "/memory/preferences", json={"disease": "private", "owner_id": "other"}, headers=h
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/memory/search", json={"query": "risk", "owner_id": "other"}, headers=h
            ).status_code
            == 422
        )
        assert client.post("/memory/search", json={"query": " "}, headers=h).status_code == 422
        for method, path in [("PUT", "/memory/preferences"), ("POST", "/memory/search")]:
            assert client.request(method, path, content=b"x" * 32769, headers=h).status_code == 413
        assert client.delete("/memory", headers=h).status_code == 204
        assert client.get("/memory/preferences", headers=h).json() is None


def test_api_session_upload_execution_and_expired_reference_projection(predictor, csv_bytes):
    app = create_app(
        predictor=predictor,
        decision_backend=ScriptedDecisionBackend(scripted_analysis),
        agent_event_store=RunEventStore(),
    )
    with TestClient(app) as client:
        uploaded = client.post("/uploads", files={"file": ("demo.csv", csv_bytes)}).json()
        path = f"/memory/sessions/{uploaded['session_id']}"
        session = client.get(path).json()
        assert session["upload_refs"] == [uploaded["upload_ref"]] and session["turns"] == []
        created = client.post(
            "/agent/runs",
            json={
                "query": "分析CSV数据",
                "session_id": uploaded["session_id"],
                "input_version": uploaded["input_version"],
                "upload_refs": [uploaded["upload_ref"]],
            },
        ).json()
        client.get(created["events_url"])
        client.portal.call(
            lambda: app.state.agent_runtime.wait(
                created["run_id"], owner_id="local_demo", session_id=uploaded["session_id"]
            )
        )
        session = client.get(path).json()
        assert session["current_result_run_id"] == created["run_id"]
        assert session["prediction"]["model_name"] == "RandomForest"
        assert len(client.post("/memory/search", json={"query": "risk 风险"}).json()) == 1
        assert len(client.get("/memory/trends").json()["groups"]) == 1
        app.state.store.clear()
        stale = client.get(path).json()
        assert (
            stale["upload_refs"] == []
            and stale["current_result_run_id"] is None
            and stale["prediction"] is None
        )
        assert len(stale["turns"]) == 1  # Historical summary survives; it is not a live artifact.
        assert client.delete(path).status_code == 204
        assert client.get(path).status_code == 404


async def test_summary_projection_rejects_wrong_model_and_never_copies_free_text(
    predictor, csv_bytes
):
    runtime = make(predictor)
    try:
        result = await run(runtime, body(runtime, csv_bytes))
        approved = result.final_result["analysis"]
        approved["summary"] = "PRIVATE_RAW_NARRATIVE"
        projected = score_summary(approved, predictor.metadata)
        assert "PRIVATE" not in projected.model_dump_json()
        metadata = predictor.metadata
        metadata["name"] = "different"
        with pytest.raises(SkillError):
            score_summary(approved, metadata)
    finally:
        await shutdown(runtime)


async def test_reanalysis_with_context_source_is_new_memory_and_restart_status_is_unknown(
    predictor, csv_bytes, tmp_path
):
    database = tmp_path / "qdrant"
    runtime = make(predictor, store=QdrantLongTermStore(path=database))
    runtime.events = RunEventStore(tmp_path / "events")
    request = body(runtime, csv_bytes)
    try:
        first = await run(runtime, request)
        second = await run(runtime, request.model_copy(update={"source_run_id": first.run_id}))
        assert second.tool_calls == 5 and second.status == "completed"
        assert second.final_result.get("source_run_id") is None
        assert len(runtime.memory.long_term.list_analysis("owner")) == 2
    finally:
        await shutdown(runtime)
    restored = make(predictor, store=QdrantLongTermStore(path=database))
    restored.events = RunEventStore(tmp_path / "events")
    try:
        restored.recover()
        snapshot = restored.snapshot(second.run_id, owner_id="owner", session_id=request.session_id)
        assert snapshot.memory_status == "unknown" and snapshot.final_result is None
        assert len(restored.memory.long_term.list_analysis("owner")) == 2
        assert restored.memory.short_term.get("owner", request.session_id) is None
    finally:
        await shutdown(restored)


async def test_pending_memory_finalization_is_not_evicted(predictor, monkeypatch):
    runtime = make(predictor)
    runtime.limits = AgentLimits(max_retained_runs=1)
    started, release = asyncio.Event(), asyncio.Event()
    original_finish = runtime.memory.finish

    async def pending(**kwargs):
        started.set()
        await release.wait()
        return await original_finish(**kwargs)

    monkeypatch.setattr(runtime.memory, "finish", pending)
    try:
        initial = await runtime.create(
            body(runtime, query="解释risk score"), owner_id="owner", request_id="pending-memory"
        )
        await asyncio.wait_for(started.wait(), 2)
        with pytest.raises(SkillError) as error:
            await runtime.create(
                body(runtime, query="解释risk score"), owner_id="owner", request_id="capacity-test"
            )
        assert error.value.code == "run_capacity_exceeded"
        assert initial.run_id in runtime._runs
        release.set()
        assert (await complete(runtime, initial)).memory_status == "stored"
    finally:
        release.set()
        await shutdown(runtime)
