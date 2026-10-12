"""Trace lifecycle, real role propagation, privacy and bounded retention."""

import asyncio
import json
import logging
from datetime import datetime

import pytest
from test_agent_runtime import close, complete, make_runtime, request_for

from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.execution_trace import ExecutionTrace
from smarthealth.runtime.logging import JsonFormatter


def test_span_lifecycle_is_bounded_and_snapshot_does_not_mutate_state():
    trace = ExecutionTrace("task_1", max_steps=1)
    first = trace.start("planner")
    with pytest.raises(SkillError, match="capacity"):
        trace.start("data_analysis")
    result = trace.finish(first)
    assert datetime.fromisoformat(result.end_time) >= datetime.fromisoformat(result.start_time)
    trace.snapshot()[0]["status"] = "failed"
    assert trace.snapshot()[0]["status"] == "completed"
    second = trace.start("data_analysis")
    trace.finish(second, status="failed", error_code="timeout")
    assert len(trace.snapshot()) == 1 and trace.snapshot()[0]["step_id"] == second


def test_errors_are_sanitized_and_logs_have_json_envelope(caplog):
    caplog.set_level(logging.INFO, logger="smart_health")
    trace = ExecutionTrace("task_1")
    step = trace.start("planner")
    trace.finish(step, status="failed", error_code="provider returned private-token")
    record = caplog.records[-1]
    payload = json.loads(JsonFormatter().format(record))
    assert payload["error_code"] == "internal_error"
    assert payload["level"] == "INFO" and payload["timestamp"]
    assert "private-token" not in record.getMessage()


@pytest.mark.parametrize("valid_csv", [True, False])
async def test_real_runtime_role_and_skill_spans(predictor, csv_bytes, caplog, valid_csv):
    caplog.set_level(logging.INFO, logger="smart_health")
    runtime = make_runtime(predictor)
    try:
        request = request_for(
            runtime, csv_bytes if valid_csv else b"bad,column\n1,2\n", entry_agent="planner"
        )
        result = await complete(
            runtime, await runtime.create(request, owner_id="owner", request_id="trace")
        )
        assert result.status == ("completed" if valid_csv else "awaiting_input")
        spans = runtime._runs[result.run_id].execution_trace.snapshot()
        assert {span["agent_name"] for span in spans} >= {"planner", "data_analysis"}
        assert all(
            span["task_id"] == result.run_id and span["start_time"] and span["end_time"]
            for span in spans
        )
        logs = [
            json.loads(record.getMessage())
            for record in caplog.records
            if record.name == "smart_health"
        ]
        skills = [
            event for event in logs if event.get("event") in {"skill.completed", "skill.failed"}
        ]
        assert skills
        assert all(
            event["agent_name"] == "data_analysis" and event["skill_name"] == event["skill"]
            for event in skills
        )
        assert all(event["task_id"] == result.run_id and event["end_time"] for event in skills)
        if not valid_csv:
            assert any(event["status"] == "failed" and event["error_message"] for event in skills)
    finally:
        await close(runtime)


async def test_cancelled_runtime_closes_all_started_spans(predictor):
    started = asyncio.Event()

    async def waiting_policy(request):
        started.set()
        await asyncio.Event().wait()

    runtime = make_runtime(predictor, policy=waiting_policy)
    try:
        request = request_for(runtime)
        initial = await runtime.create(request, owner_id="owner", request_id="cancel-trace")
        await asyncio.wait_for(started.wait(), 5)
        runtime.cancel(initial.run_id, owner_id="owner", session_id=initial.session_id)
        result = await complete(runtime, initial)
        assert result.status == "cancelled"
        spans = runtime._runs[result.run_id].execution_trace.snapshot()
        assert len(spans) == 2
        assert all(span["status"] == "cancelled" and span["end_time"] for span in spans)
        assert all(span["error_code"] == "cancelled" for span in spans)
    finally:
        await close(runtime)
