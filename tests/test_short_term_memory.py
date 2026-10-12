from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from app.core.errors import SkillError
from app.memory.schemas import ExecutionSummary, ScoreSummary, TurnSummary
from app.memory.short_term import LocalShortTermStore

UPLOAD = "art_" + "a" * 32
NEW_UPLOAD = "art_" + "b" * 32


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def score():
    return ScoreSummary(
        prediction="anomaly",
        risk_score=0.82,
        risk_level="high",
        window_count=10,
        model_name="LightGBM",
        model_version="1.0",
        model_fingerprint="c" * 64,
        input_profile="synthetic_physiology_v1",
        feature_schema_version="1.0",
    )


def turn(run="run1", version="v1", status="completed", analysis=None, intent="data_analysis"):
    return TurnSummary(
        run_id=run,
        input_version=version,
        user_intent=intent,
        assistant_status=status,
        completed_at=1000.0,
        analysis=analysis,
    )


def finish_run(store, run="run1", version="v1", session="session1", owner="owner1", analysis=None):
    lease = store.begin(owner, session, version, run, [UPLOAD])
    assert store.finish(
        owner,
        session,
        lease,
        turn(run=run, version=version, analysis=analysis),
        result_run_id=run if analysis is not None else None,
    )
    return lease


def test_recent_five_finished_interactions_are_separate_from_active_execution():
    store = LocalShortTermStore()
    for index in range(7):
        finish_run(store, run=f"run{index}")
    lease = store.begin("owner1", "session1", "v1", "active", [UPLOAD])
    for event in range(1, 101):
        assert store.update(
            "owner1",
            "session1",
            lease,
            ExecutionSummary(run_id="active", status="running", last_event_id=event),
        )
    snapshot = store.get("owner1", "session1")
    assert [item.run_id for item in snapshot.turns] == [f"run{i}" for i in range(2, 7)]
    assert snapshot.execution.last_event_id == 100
    assert store.finish("owner1", "session1", lease, turn(run="active", status="awaiting_input"))
    assert [item.run_id for item in store.get("owner1", "session1").turns] == [
        "run3",
        "run4",
        "run5",
        "run6",
        "active",
    ]


def test_read_polling_does_not_extend_idle_retention():
    clock = Clock()
    store = LocalShortTermStore(ttl_seconds=10, clock=clock)
    finish_run(store)
    for elapsed in (1, 5, 9):
        clock.now = 1000 + elapsed
        assert store.get("owner1", "session1").expires_at == 1010
    clock.now = 1010
    assert store.get("owner1", "session1") is None


def test_runtime_updates_renew_ttl_but_expired_leases_cannot_recreate_state():
    clock = Clock()
    store = LocalShortTermStore(ttl_seconds=10, clock=clock)
    lease = store.begin("owner1", "session1", "v1", "run1", [UPLOAD])
    clock.now = 1009
    execution = ExecutionSummary(run_id="run1", status="running")
    assert store.update("owner1", "session1", lease, execution)
    assert store.get("owner1", "session1").expires_at == 1019
    clock.now = 1019
    assert not store.update("owner1", "session1", lease, execution)
    assert not store.finish("owner1", "session1", lease, turn())
    assert store.get("owner1", "session1") is None
    new_lease = store.begin("owner1", "session1", "v1", "run1", [UPLOAD])
    assert new_lease != lease
    assert not store.finish("owner1", "session1", lease, turn())


def test_new_input_clears_current_facts_but_preserves_completed_turns():
    store = LocalShortTermStore()
    finish_run(store, analysis=score())
    initial = store.get("owner1", "session1")
    assert initial.current_result_run_id == "run1"
    store.begin("owner1", "session1", "v2", "run2", [NEW_UPLOAD])
    current = store.get("owner1", "session1")
    assert current.input_version == "v2" and current.upload_refs == [NEW_UPLOAD]
    assert current.current_result_run_id is None and current.prediction is None
    assert current.turns[0].analysis == score()


@pytest.mark.parametrize("status", ["failed", "blocked", "cancelled", "awaiting_input"])
def test_unsuccessful_turn_does_not_replace_same_version_approved_result(status):
    store = LocalShortTermStore()
    finish_run(store, analysis=score())
    lease = store.begin("owner1", "session1", "v1", "run2", [])
    assert store.finish("owner1", "session1", lease, turn(run="run2", status=status))
    current = store.get("owner1", "session1")
    assert current.current_result_run_id == "run1" and current.prediction == score()
    assert current.execution.status == status and current.turns[-1].assistant_status == status


def test_owner_scope_includes_reads_deletions_and_leases():
    store = LocalShortTermStore()
    lease_a = store.begin("owner_a", "shared", "v1", "run1", [UPLOAD])
    lease_b = store.begin("owner_b", "shared", "v1", "run1", [NEW_UPLOAD])
    assert store.get("unknown", "shared") is None
    assert not store.finish("owner_b", "shared", lease_a, turn())
    store.delete_owner("owner_a")
    assert store.get("owner_a", "shared") is None
    assert store.get("owner_b", "shared").upload_refs == [NEW_UPLOAD]
    assert store.finish("owner_b", "shared", lease_b, turn())


@pytest.mark.parametrize("deletion", ["session", "owner"])
def test_delete_invalidates_active_lease_and_late_results(deletion):
    store = LocalShortTermStore()
    lease = store.begin("owner1", "session1", "v1", "run1", [UPLOAD])
    if deletion == "session":
        store.delete_session("owner1", "session1")
    else:
        store.delete_owner("owner1")
    assert not store.finish("owner1", "session1", lease, turn(analysis=score()), "run1")
    assert not store.update(
        "owner1", "session1", lease, ExecutionSummary(run_id="run1", status="completed")
    )
    assert store.get("owner1", "session1") is None
    new_lease = store.begin("owner1", "session1", "v1", "run1", [UPLOAD])
    assert new_lease != lease
    assert not store.finish("owner1", "session1", lease, turn())


def test_capacity_evicts_oldest_inactive_and_never_active_session():
    clock = Clock()
    store = LocalShortTermStore(max_sessions=2, clock=clock)
    active = store.begin("owner1", "active", "v1", "active_run", [UPLOAD])
    clock.now += 1
    finish_run(store, session="older")
    clock.now += 1
    store.begin("owner1", "newer", "v1", "new_run", [])
    assert store.get("owner1", "older") is None
    assert store.get("owner1", "active") is not None
    with pytest.raises(SkillError) as error:
        store.begin("owner1", "overflow", "v1", "overflow_run", [])
    assert error.value.code == "memory_capacity_exceeded"
    assert store.finish("owner1", "active", active, turn(run="active_run"))


def test_reads_do_not_change_capacity_eviction_order():
    clock = Clock()
    store = LocalShortTermStore(max_sessions=2, clock=clock)
    finish_run(store, session="old")
    clock.now += 1
    finish_run(store, session="new")
    clock.now += 1
    assert store.get("owner1", "old") is not None
    store.attach_upload("owner1", "third", "v1", UPLOAD)
    assert store.get("owner1", "old") is None
    assert store.get("owner1", "new") is not None


def test_returned_and_input_models_cannot_mutate_retained_state():
    store = LocalShortTermStore()
    refs = [UPLOAD]
    lease = store.begin("owner1", "session1", "v1", "run1", refs)
    refs.clear()
    execution = ExecutionSummary(run_id="run1", status="running", last_event_id=3)
    assert store.update("owner1", "session1", lease, execution)
    execution.last_event_id = 999
    value = turn(analysis=score())
    assert store.finish("owner1", "session1", lease, value, "run1")
    value.analysis.risk_score = 0.0
    current = store.get("owner1", "session1")
    current.upload_refs.clear()
    current.prediction.risk_score = 0.1
    current.turns.clear()
    final = store.get("owner1", "session1")
    assert final.upload_refs == [UPLOAD]
    assert final.prediction.risk_score == final.turns[0].analysis.risk_score == 0.82
    assert final.execution.last_event_id == 3


def test_begin_idempotency_and_busy_input_protection():
    store = LocalShortTermStore()
    lease = store.begin("owner1", "session1", "v1", "run1", [UPLOAD])
    assert store.begin("owner1", "session1", "v1", "run1", [UPLOAD]) == lease
    with pytest.raises(SkillError) as error:
        store.begin("owner1", "session1", "v2", "run2", [NEW_UPLOAD])
    assert error.value.code == "session_busy"
    with pytest.raises(SkillError) as error:
        store.begin("owner1", "session1", "v2", "run1", [NEW_UPLOAD])
    assert error.value.code == "memory_state_conflict"
    assert store.get("owner1", "session1").upload_refs == [UPLOAD]


def test_invalid_and_out_of_order_updates_do_not_write_or_extend_ttl():
    clock = Clock()
    store = LocalShortTermStore(ttl_seconds=10, clock=clock)
    lease = store.begin("owner1", "session1", "v1", "run1", [UPLOAD])
    execution = ExecutionSummary(
        run_id="run1", status="running", last_event_id=5, decision_count=2, tool_calls=1
    )
    assert store.update("owner1", "session1", lease, execution)
    clock.now += 3
    assert not store.update("owner1", "session1", "invalid", execution)
    assert not store.update(
        "owner1", "session1", lease, execution.model_copy(update={"run_id": "wrong"})
    )
    assert not store.update(
        "owner1", "session1", lease, execution.model_copy(update={"last_event_id": 4})
    )
    assert not store.finish("owner1", "session1", lease, turn(version="wrong"))
    assert not store.finish("owner1", "session1", lease, turn(status="running"))
    assert not store.finish("owner1", "session1", lease, turn(), "other_run")
    assert store.get("owner1", "session1").expires_at == 1010


def test_typed_instances_and_identifiers_revalidated_at_boundary():
    store = LocalShortTermStore()
    with pytest.raises(ValidationError):
        store.begin("../owner", "session1", "v1", "run1", [])
    with pytest.raises(ValidationError):
        store.attach_upload("owner1", "session1", "v1", "/private/data.csv")
    with pytest.raises(ValidationError):
        store.begin("owner1", "session1", "v1", "run1", [UPLOAD, NEW_UPLOAD])
    lease = store.begin("owner1", "session1", "v1", "run1", [])
    with pytest.raises(ValidationError):
        store.update(
            "owner1",
            "session1",
            lease,
            ExecutionSummary.model_construct(run_id="run1", status="running", last_event_id=-1),
        )
    with pytest.raises(ValidationError):
        store.finish("owner1", "session1", lease, turn().model_copy(update={"completed_at": -1}))


def test_upload_attachment_obeys_active_protection_and_version_rules():
    store = LocalShortTermStore()
    store.attach_upload("owner1", "session1", "v1", UPLOAD)
    memory = store.get("owner1", "session1")
    assert memory.execution is None and memory.turns == []
    finish_run(store, analysis=score())
    store.attach_upload("owner1", "session1", "v2", NEW_UPLOAD)
    memory = store.get("owner1", "session1")
    assert memory.current_result_run_id is None and memory.prediction is None
    assert memory.execution is None and len(memory.turns) == 1
    assert memory.upload_refs == [NEW_UPLOAD] and memory.input_version == "v2"
    store.begin("owner1", "session1", "v2", "run2", [NEW_UPLOAD])
    with pytest.raises(SkillError) as error:
        store.attach_upload("owner1", "session1", "v3", UPLOAD)
    assert error.value.code == "session_busy"
    assert store.get("owner1", "session1").input_version == "v2"


def test_concurrent_updates_and_finishes_are_atomic():
    store = LocalShortTermStore()
    lease = store.begin("owner1", "session1", "v1", "run1", [])

    def update(event):
        return store.update(
            "owner1",
            "session1",
            lease,
            ExecutionSummary(run_id="run1", status="running", last_event_id=event),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(update, range(1, 101)))
        finished = list(
            pool.map(lambda _: store.finish("owner1", "session1", lease, turn()), range(8))
        )
    assert sum(finished) == 1
    memory = store.get("owner1", "session1")
    assert memory.execution.last_event_id == 100 and len(memory.turns) == 1


def test_concurrent_begin_allows_only_one_active_run():
    store = LocalShortTermStore()

    def begin(index):
        try:
            return store.begin("owner1", "session1", "v1", f"run{index}", [])
        except SkillError as exc:
            assert exc.code == "session_busy"
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        leases = list(pool.map(begin, range(16)))
    assert sum(lease is not None for lease in leases) == 1


@pytest.mark.parametrize("ttl", [0, -1, float("nan"), float("inf"), True])
def test_invalid_ttl_rejected(ttl):
    with pytest.raises(ValueError):
        LocalShortTermStore(ttl_seconds=ttl)


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ValueError):
        LocalShortTermStore(max_sessions=capacity)


def test_cleanup_expires_all_entries_including_abandoned_active_run():
    clock = Clock()
    store = LocalShortTermStore(ttl_seconds=10, clock=clock)
    store.begin("owner1", "session1", "v1", "run1", [])
    store.attach_upload("owner2", "session2", "v1", UPLOAD)
    clock.now += 10
    assert store.cleanup() == 2
    assert store.cleanup() == 0
