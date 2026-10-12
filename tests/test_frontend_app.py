"""Streamlit widget smoke tests; optional UI extra is required to execute these."""

import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from app.agents.contracts import RunSnapshot
from frontend.client import EventBatch, ProgressTracker

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"


def test_home_page_and_credential_reset_do_not_call_backend():
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception and app.title[0].value == "SmartHealth Agent"
    assert [tab.label for tab in app.tabs] == ["数据工具", "Agent 工作台"]
    assert next(b for b in app.button if b.label == "快速预测").disabled
    app.text_input(key="health_api_token").set_value("private-test-token").run()
    assert not app.exception
    assert app.session_state["health_api_token"] == "private-test-token"
    next(b for b in app.button if b.label == "清除凭据与页面会话").click().run()
    assert not app.exception and app.session_state["health_api_token"] == ""


def test_reviewed_report_renders_as_text_and_shows_scripted_mode(monkeypatch):
    from frontend import client

    class ImageClient:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def image(self, *_):
            return Path("docs/harness_example/risk.png").read_bytes()

    monkeypatch.setattr(client, "HealthApiClient", ImageClient)
    snapshot = RunSnapshot.model_validate_json(Path("docs/report_example/run.json").read_text())
    app = AppTest.from_file(APP_PATH, default_timeout=10)
    app.session_state["health_run"] = snapshot
    app.session_state["health_tracker"] = ProgressTracker(
        snapshot.run_id, cursor=snapshot.last_event_id
    )
    app.session_state["health_last_poll"] = time.monotonic()
    app.run()
    assert not app.exception
    assert any(snapshot.final_result["report"]["body"] == text.value for text in app.text)
    assert any("脚本决策" in warning.value for warning in app.warning)


def test_active_run_does_not_mark_future_report_chunks_expired(monkeypatch):
    from frontend import client

    snapshot = RunSnapshot.model_validate_json(Path("docs/report_example/run.json").read_text())
    snapshot = snapshot.model_copy(
        update={
            "status": "running",
            "final_result": None,
            "result_available": False,
            "last_event_id": 0,
        }
    )

    class RunningClient:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def snapshot(self, *_):
            return snapshot

        def events(self, *_, **kwargs):
            return EventBatch()

    monkeypatch.setattr(client, "HealthApiClient", RunningClient)
    app = AppTest.from_file(APP_PATH, default_timeout=10)
    app.session_state["health_run"] = snapshot
    app.session_state["health_tracker"] = ProgressTracker(snapshot.run_id)
    app.run()
    assert not app.exception
    assert app.session_state["health_tracker"].content_available
