import json
import os

import pytest

from app.agents.events import RunEventStore
from app.agents.router import RouteDecision
from app.core.errors import SkillError


def register(store, run_id="run1", **metadata):
    store.register(
        run_id,
        owner_id="owner",
        session_id="session",
        metadata={
            "input_version": "v1",
            "request_id": "req1",
            "backend_mode": "scripted",
            "objective": "data_analysis",
            "entry_agent": "data_analysis",
            **metadata,
        },
    )


def read(store, **kw):
    return store.get_events("run1", owner_id="owner", session_id="session", **kw)


def test_owned_replay_and_copy_isolation():
    store = RunEventStore(max_events=3)
    register(store)
    event = store.append("run1", "run.running", data={"status": "running"})
    event.data["status"] = "SECRET"
    assert read(store)[0].data["status"] == "running"
    with pytest.raises(SkillError, match="another session"):
        store.get_events("run1", owner_id="attacker", session_id="session")
    for _ in range(3):
        store.append("run1", "agent.thinking", data={"decision_count": 1})
    with pytest.raises(SkillError) as error:
        read(store, after_id=0)
    assert error.value.code == "replay_unavailable"
    assert [e.event_id for e in read(store, after_id=2)] == [3, 4]
    for cursor in (-1, 5, True):
        with pytest.raises(SkillError) as error:
            read(store, after_id=cursor)
        assert error.value.code == "invalid_event_cursor"


def test_journal_recovery_marks_interrupted_and_keeps_only_metadata(tmp_path):
    store = RunEventStore(tmp_path)
    register(store)
    store.append("run1", "run.running", data={"status": "running"})
    store.append("run1", "agent.thinking", data={"decision_count": 2})
    with (tmp_path / "run1.jsonl").open("ab") as stream:
        stream.write(b'{"partial":')
    recovered = RunEventStore(tmp_path)
    metadata = recovered.recover()[0]
    assert metadata["status"] == "failed" and metadata["error_code"] == "process_restarted"
    assert metadata["decision_count"] == 2 and metadata["result_available"] is False
    assert read(recovered)[-1].event == "run.failed"
    again = RunEventStore(tmp_path)
    assert again.recover()[0]["last_event_id"] == 3
    assert os.stat(tmp_path / "run1.jsonl").st_mode & 0o777 == 0o600
    assert all(json.loads(line) for line in (tmp_path / "run1.jsonl").read_text().splitlines())


@pytest.mark.parametrize(
    "data", [{"query": "private"}, {"content": "private"}, {"input_ref": "/secret.csv"}]
)
def test_trace_privacy_rejects_uncontrolled_fields(tmp_path, data):
    store = RunEventStore(tmp_path)
    register(store)
    with pytest.raises(SkillError):
        store.append("run1", "agent.decision", data=data)
    assert "private" not in (tmp_path / "run1.jsonl").read_text()
    assert store.latest_id("run1") == 0


def test_audit_write_failure_does_not_publish_in_memory(tmp_path, monkeypatch):
    store = RunEventStore(tmp_path)
    register(store)

    def fail(*args):
        raise OSError("SECRET_PATH")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(SkillError) as error:
        store.append("run1", "run.running")
    assert error.value.code == "trace_unavailable" and "SECRET" not in str(error.value)
    assert read(store) == []
    assert len((tmp_path / "run1.jsonl").read_text().splitlines()) == 1


def test_capacity_preserves_active_and_unknown_files(tmp_path):
    foreign = tmp_path / "other.jsonl"
    foreign.write_text('{"ordinary":"file"}\n')
    store = RunEventStore(tmp_path, max_runs=1)
    assert store.recover() == []
    register(store)
    with pytest.raises(SkillError) as error:
        register(store, "run2")
    assert error.value.code == "run_capacity_exceeded"
    store.append("run1", "run.completed", data={"status": "completed"})
    register(store, "run2")
    assert not (tmp_path / "run1.jsonl").exists()
    assert foreign.read_text() == '{"ordinary":"file"}\n'


def simple_route():
    return RouteDecision(
        workflow_type="simple",
        reason_code="score_definition",
        objective="score_definition",
        entry_agent="data_analysis",
        required_capabilities=["model_card"],
        needs_input=None,
    )


def complex_route():
    return RouteDecision(
        workflow_type="complex",
        reason_code="data_analysis",
        objective="data_analysis",
        entry_agent="planner",
        required_capabilities=["csv_analysis", "anomaly_prediction"],
        needs_input=None,
    )


def test_selected_route_recovery_preserves_metadata_without_result(tmp_path):
    route = simple_route()
    store = RunEventStore(tmp_path)
    register(
        store,
        route=route.model_dump(),
        objective=route.objective,
        source_run_id="prior_run",
    )
    data = {**route.model_dump(), "workflow_upgrades": 0}
    store.append("run1", "route.selected", data=data)
    data["required_capabilities"].append("injected")
    store.append("run1", "run.completed", data={"status": "completed"})

    restored = RunEventStore(tmp_path)
    metadata = restored.recover()[0]
    assert metadata["route"] == route.model_dump()
    assert metadata["initial_workflow_type"] == "simple"
    assert metadata["workflow_upgrades"] == 0
    assert metadata["source_run_id"] == "prior_run"
    assert metadata["result_available"] is False
    assert read(restored)[0].data["required_capabilities"] == ["model_card"]


def test_upgraded_route_recovery_keeps_latest_objective_and_initial_workflow(tmp_path):
    initial, upgraded = simple_route(), complex_route()
    store = RunEventStore(tmp_path, max_events=2)
    register(store, route=initial.model_dump(), objective=initial.objective)
    store.append("run1", "route.selected", data={**initial.model_dump(), "workflow_upgrades": 0})
    store.append("run1", "route.upgraded", data={**upgraded.model_dump(), "workflow_upgrades": 1})
    store.append("run1", "run.completed", data={"status": "completed", "tool_calls": 5})

    metadata = RunEventStore(tmp_path, max_events=1).recover()[0]
    assert metadata["route"] == upgraded.model_dump()
    assert metadata["objective"] == "data_analysis"
    assert metadata["entry_agent"] == "planner"
    assert metadata["initial_workflow_type"] == "simple"
    assert metadata["workflow_upgrades"] == 1
    assert metadata["tool_calls"] == 5
    assert metadata["source_run_id"] is None


def test_pre_router_journal_is_backward_compatible(tmp_path):
    store = RunEventStore(tmp_path)
    register(store)
    store.append("run1", "run.completed", data={"status": "completed"})
    path = tmp_path / "run1.jsonl"
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header.pop("route")
    header.pop("source_run_id")
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n")

    metadata = RunEventStore(tmp_path).recover()[0]
    assert metadata["status"] == "completed"
    assert metadata["objective"] == "data_analysis"
    assert metadata["route"] is None
    assert metadata["initial_workflow_type"] is None
    assert metadata["source_run_id"] is None
    assert metadata["workflow_upgrades"] == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"reason_code": "private_patient_detail"},
        {"reason": "private patient detail"},
        {"required_capabilities": ["private patient detail"]},
        {"required_capabilities": ["model_card"] * 17},
        {"router_version": "private patient detail"},
        {"workflow_upgrades": True},
        {"workflow_upgrades": 2},
        {"workflow_upgrades": 1},
    ],
)
def test_invalid_route_event_is_rejected_before_journal_write(tmp_path, updates):
    route = simple_route()
    store = RunEventStore(tmp_path)
    register(store, route=route.model_dump())
    path = tmp_path / "run1.jsonl"
    before = path.read_bytes()
    with pytest.raises(SkillError) as error:
        store.append(
            "run1", "route.selected", data={**route.model_dump(), "workflow_upgrades": 0, **updates}
        )
    assert error.value.code == "invalid_trace_event"
    assert path.read_bytes() == before
    assert store.latest_id("run1") == 0


def test_partial_route_and_repeated_upgrade_are_rejected(tmp_path):
    initial, upgraded = simple_route(), complex_route()
    store = RunEventStore(tmp_path)
    register(store, route=initial.model_dump())
    incomplete = {**initial.model_dump(), "workflow_upgrades": 0}
    del incomplete["objective"]
    with pytest.raises(SkillError):
        store.append("run1", "route.selected", data=incomplete)
    store.append("run1", "route.upgraded", data={**upgraded.model_dump(), "workflow_upgrades": 1})
    path = tmp_path / "run1.jsonl"
    before = path.read_bytes()
    for event, route, count in (
        ("route.upgraded", upgraded, 1),
        ("route.upgraded", initial, 1),
        ("route.selected", initial, 0),
    ):
        with pytest.raises(SkillError):
            store.append("run1", event, data={**route.model_dump(), "workflow_upgrades": count})
    assert path.read_bytes() == before
    assert store.latest_id("run1") == 1


def test_corrupted_route_event_fails_closed_on_recovery(tmp_path):
    route = simple_route()
    store = RunEventStore(tmp_path)
    register(store, route=route.model_dump())
    store.append("run1", "route.selected", data={**route.model_dump(), "workflow_upgrades": 0})
    path = tmp_path / "run1.jsonl"
    lines = path.read_text().splitlines()
    event = json.loads(lines[1])
    event["data"]["objective"] = "health_report"
    lines[1] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(SkillError):
        RunEventStore(tmp_path).recover()


def test_harness_metadata_and_event_survive_journal_recovery(tmp_path):
    metadata = {"harness_version": "1.0-rc.2", "harness_hash": "abcdef0123456789" * 4}
    store = RunEventStore(tmp_path)
    register(store, **metadata)
    store.append("run1", "harness.validated", data=metadata)
    store.append("run1", "run.completed", data={"status": "completed"})

    recovered = RunEventStore(tmp_path)
    restored = recovered.recover()[0]
    assert {key: restored[key] for key in metadata} == metadata
    assert read(recovered)[0].data == metadata
    assert restored["status"] == "completed"
    assert restored["result_available"] is False


@pytest.mark.parametrize(
    "metadata",
    [
        {"harness_version": "1.0"},
        {"harness_hash": "a" * 64},
        {"harness_version": "1.0", "harness_hash": None},
        {"harness_version": None, "harness_hash": "a" * 64},
        {"harness_version": "private patient detail", "harness_hash": "a" * 64},
        {"harness_version": "", "harness_hash": "a" * 64},
        {"harness_version": "a" * 65, "harness_hash": "a" * 64},
        {"harness_version": 1, "harness_hash": "a" * 64},
        {"harness_version": "1.0", "harness_hash": "A" * 64},
        {"harness_version": "1.0", "harness_hash": "a" * 63},
        {"harness_version": "1.0", "harness_hash": "g" * 64},
        {"harness_version": "1.0", "harness_hash": 123},
    ],
)
def test_invalid_or_partial_harness_header_is_rejected_before_write(tmp_path, metadata):
    store = RunEventStore(tmp_path)
    with pytest.raises(SkillError) as error:
        register(store, **metadata)
    assert error.value.code == "invalid_run_metadata"
    assert not (tmp_path / "run1.jsonl").exists()


@pytest.mark.parametrize(
    "data",
    [
        {"harness_version": "private patient detail"},
        {"harness_version": "a" * 65},
        {"harness_version": ""},
        {"harness_version": "1.0\n"},
        {"harness_version": None},
        {"harness_version": True},
        {"harness_hash": "A" * 64},
        {"harness_hash": "g" * 64},
        {"harness_hash": "a" * 65},
        {"harness_hash": "a" * 63},
        {"harness_hash": "a" * 64 + "\n"},
        {"harness_hash": None},
        {"harness_hash": 123},
    ],
)
def test_invalid_harness_event_metadata_is_never_persisted(tmp_path, data):
    store = RunEventStore(tmp_path)
    register(store)
    path = tmp_path / "run1.jsonl"
    before = path.read_bytes()
    with pytest.raises(SkillError) as error:
        store.append("run1", "harness.validated", data=data)
    assert error.value.code == "invalid_trace_event"
    assert path.read_bytes() == before
    assert store.latest_id("run1") == 0


@pytest.mark.parametrize("omit_fields", [False, True])
def test_journals_without_harness_metadata_remain_readable(tmp_path, omit_fields):
    store = RunEventStore(tmp_path)
    register(store, harness_version=None, harness_hash=None)
    store.append("run1", "run.completed", data={"status": "completed"})
    if omit_fields:
        path = tmp_path / "run1.jsonl"
        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        del header["harness_version"]
        del header["harness_hash"]
        lines[0] = json.dumps(header)
        path.write_text("\n".join(lines) + "\n")
    restored = RunEventStore(tmp_path).recover()[0]
    assert restored["harness_version"] is None
    assert restored["harness_hash"] is None
    assert restored["status"] == "completed"


def test_partial_harness_header_fails_closed_on_recovery(tmp_path):
    store = RunEventStore(tmp_path)
    register(store, harness_version="1.0", harness_hash="a" * 64)
    path = tmp_path / "run1.jsonl"
    header = json.loads(path.read_text())
    del header["harness_hash"]
    path.write_text(json.dumps(header) + "\n")
    with pytest.raises(SkillError) as error:
        RunEventStore(tmp_path).recover()
    assert error.value.code == "trace_unavailable"
