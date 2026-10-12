import asyncio
import json

from agent_helpers import scripted_analysis
from fastapi.testclient import TestClient

from app.agents.backends import ScriptedDecisionBackend
from app.agents.contracts import FinishAction
from app.agents.events import RunEventStore
from app.main import create_app


def app_for(predictor, policy=scripted_analysis, **kwargs):
    return create_app(
        predictor=predictor,
        decision_backend=ScriptedDecisionBackend(policy),
        agent_event_store=RunEventStore(),
        **kwargs,
    )


def sse_events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def create(client, csv_bytes, headers=None, **kwargs):
    uploaded = client.post(
        "/uploads", files={"file": ("demo.csv", csv_bytes)}, headers=headers
    ).json()
    request = {
        "query": "Analyze synthetic data",
        "session_id": uploaded["session_id"],
        "input_version": uploaded["input_version"],
        "upload_refs": [uploaded["upload_ref"]],
        **kwargs,
    }
    response = client.post("/agent/runs", json=request, headers=headers)
    assert response.status_code == 202, response.text
    return response.json(), request


def test_agent_api_sse_replay_auth_scope_and_final_result(predictor, csv_bytes, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="smart_health")
    with TestClient(app_for(predictor, api_token="test-token")) as client:
        headers = {"Authorization": "Bearer test-token", "Idempotency-Key": "api-demo"}
        initial, body = create(
            client, csv_bytes, headers, entry_agent="planner", include_visualizations=True
        )
        events_url = initial["events_url"]
        assert client.get(events_url).status_code == 401
        stream = client.get(events_url, headers=headers)
        assert stream.status_code == 200 and stream.headers["content-type"].startswith(
            "text/event-stream"
        )
        events = sse_events(stream.text)
        assert events[0]["event"] == "run.queued" and events[-1]["event"] == "run.completed"
        assert [e["event_id"] for e in events] == list(range(1, len(events) + 1))
        result = client.get(
            f"/agent/runs/{initial['run_id']}?session_id={initial['session_id']}", headers=headers
        ).json()
        assert result["status"] == "completed" and result["tool_calls"] == 6
        assert result["final_result"]["analysis"]["prediction"]["model"]["name"] == "RandomForest"
        replay = client.get(events_url, headers={**headers, "Last-Event-ID": "3"})
        assert sse_events(replay.text)[0]["event_id"] == 4
        exhausted = client.get(
            events_url, headers={**headers, "Last-Event-ID": str(events[-1]["event_id"])}
        )
        assert exhausted.status_code == 200 and not exhausted.text
        for cursor in ("-1", "abc", "999999999999999999999"):
            bad = client.get(events_url, headers={**headers, "Last-Event-ID": cursor})
            assert bad.status_code == 422 and bad.headers["content-type"].startswith(
                "application/json"
            )
        assert (
            client.get(
                events_url.replace(initial["session_id"], "another"), headers=headers
            ).status_code
            == 403
        )
        trace = client.get(
            f"/agent/runs/{initial['run_id']}/trace?session_id={initial['session_id']}",
            headers=headers,
        ).json()
        assert trace["events"] == events
        same = client.post("/agent/runs", json=body, headers=headers)
        assert same.json()["run_id"] == initial["run_id"]
        assert (
            client.post(
                "/agent/runs", json={**body, "query": "different"}, headers=headers
            ).status_code
            == 409
        )
        image_url = result["final_result"]["analysis"]["visualization"][0]["url"]
        assert client.get(image_url, headers=headers).content.startswith(b"\x89PNG")
        assert "Analyze synthetic data" not in json.dumps(trace)
    assert "Analyze synthetic data" not in caplog.text


def test_agent_disabled_does_not_break_tool_api(predictor, csv_bytes, monkeypatch):
    monkeypatch.delenv("HEALTH_AGENT_BACKEND", raising=False)
    with TestClient(create_app(predictor=predictor, agent_event_store=RunEventStore())) as client:
        body = {"query": "analyze", "session_id": "session", "input_version": "v1"}
        response = client.post("/agent/runs", json=body)
        assert (
            response.status_code == 503
            and response.json()["error"]["code"] == "backend_unavailable"
        )
        assert client.post("/predict", files={"file": ("demo.csv", csv_bytes)}).status_code == 200


def test_cancel_endpoint_and_session_busy(predictor):
    async def slow(state):
        await asyncio.sleep(10)
        return FinishAction(result_kind="score_definition")

    with TestClient(app_for(predictor, slow)) as client:
        body = {
            "query": "What does risk_score mean?",
            "session_id": "session",
            "input_version": "v1",
            "objective": "score_definition",
        }
        initial = client.post("/agent/runs", json=body).json()
        assert client.post("/agent/runs", json=body).status_code == 409
        cancelled = client.post(f"/agent/runs/{initial['run_id']}/cancel?session_id=session")
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
        events = sse_events(client.get(initial["events_url"]).text)
        assert events[-1]["event"] == "run.cancelled"


def test_resume_awaiting_input_uses_new_upload_version(predictor, csv_bytes):
    with TestClient(app_for(predictor)) as client:
        body = {"query": "Analyze data", "session_id": "session", "input_version": "v1"}
        initial = client.post("/agent/runs", json=body).json()
        events = sse_events(client.get(initial["events_url"]).text)
        assert events[-1]["event"] == "run.awaiting_input"
        uploaded = client.post(
            "/uploads?session_id=session", files={"file": ("demo.csv", csv_bytes)}
        ).json()
        assert uploaded["session_id"] == "session" and uploaded["input_version"] != "v1"
        second = client.post(
            "/agent/runs",
            json={
                **body,
                "input_version": uploaded["input_version"],
                "upload_refs": [uploaded["upload_ref"]],
                "parent_run_id": initial["run_id"],
            },
        ).json()
        assert second["run_id"] != initial["run_id"]
        assert sse_events(client.get(second["events_url"]).text)[-1]["event"] == "run.completed"


def test_agent_json_body_limit_precedes_schema_parse(predictor):
    with TestClient(app_for(predictor)) as client:
        response = client.post(
            "/agent/runs", content=b"x" * 32769, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "file_too_large"


def test_replay_gap_returns_json_before_sse_headers(predictor):
    app = create_app(
        predictor=predictor,
        decision_backend=ScriptedDecisionBackend(scripted_analysis),
        agent_event_store=RunEventStore(max_events=2),
    )
    with TestClient(app) as client:
        body = {
            "query": "What does risk_score mean?",
            "session_id": "session",
            "input_version": "v1",
            "objective": "score_definition",
        }
        initial = client.post("/agent/runs", json=body).json()
        # A finished definition has more than two events, regardless of connection timing.
        for _ in range(50):
            result = client.get(f"/agent/runs/{initial['run_id']}?session_id=session").json()
            if result["status"] == "completed":
                break
        assert result["status"] == "completed"
        response = client.get(initial["events_url"])
        assert (
            response.status_code == 409 and response.json()["error"]["code"] == "replay_unavailable"
        )
