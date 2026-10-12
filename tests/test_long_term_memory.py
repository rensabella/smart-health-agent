import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from qdrant_client import QdrantClient, models

from app.core.errors import SkillError
from app.memory.long_term import COLLECTION_NAME, QdrantLongTermStore
from app.memory.schemas import LongTermEntry, Preferences, ScoreSummary


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def analysis(run="run1", created=990.0, expiry=1100.0, risk=0.8):
    return LongTermEntry(
        kind="analysis",
        source_run_id=run,
        created_at=created,
        expires_at=expiry,
        analysis=ScoreSummary(
            prediction="anomaly" if risk >= 0.5 else "normal",
            risk_score=risk,
            risk_level="high" if risk >= 0.7 else "low",
            window_count=10,
            model_name="LightGBM",
            model_version="1.0",
            model_fingerprint="a" * 64,
            input_profile="synthetic_1hz_v1",
            feature_schema_version="1.0",
        ),
    )


def preferences(created=990.0, expiry=1100.0, language="zh-CN"):
    return LongTermEntry(
        kind="preferences",
        created_at=created,
        expires_at=expiry,
        preferences=Preferences(language=language),
    )


@pytest.fixture
def store():
    instance = QdrantLongTermStore(clock=Clock(), retention_seconds=200)
    yield instance
    instance.close()


def test_real_local_qdrant_persistence_and_permissions(tmp_path):
    path = tmp_path / "memory"
    first = QdrantLongTermStore(path=path, clock=Clock(), retention_seconds=200)
    entry = analysis()
    assert first.put_analysis(owner_id="alice", entry=entry) is True
    first.set_preferences("alice", preferences(language="en"))
    first.close()
    first.close()

    second = QdrantLongTermStore(path=path, clock=Clock(), retention_seconds=200)
    try:
        assert second.list_analysis("alice") == [entry]
        assert second.get_preferences("alice").language == "en"
        hit = second.search("alice", "high anomaly 风险 高")[0]
        assert hit.entry == entry
        assert hit.score_type == "cosine_lexical_hash"
        assert 0 < hit.score <= 1
        assert second.search("bob", "high anomaly") == []
    finally:
        second.close()
    for directory, _, files in os.walk(path):
        assert os.stat(directory).st_mode & 0o777 == 0o700
        for name in files:
            assert os.stat(os.path.join(directory, name)).st_mode & 0o777 == 0o600


def test_owner_filters_apply_to_search_list_preferences_and_delete(store):
    first, other = analysis(risk=0.8), analysis(risk=0.2)
    store.put_analysis("alice", first)
    store.put_analysis("bob", other)
    store.set_preferences("alice", preferences(language="en"))
    store.set_preferences("bob", preferences())
    assert [hit.entry for hit in store.search("alice", "normal low 风险 低")] == [first]
    assert store.list_analysis("bob") == [other]
    assert store.get_preferences("alice").language == "en"
    assert store.get_preferences("nobody") is None
    store.delete_owner("alice")
    assert store.search("alice", "risk score") == []
    assert store.list_analysis("alice") == []
    assert store.get_preferences("alice") is None
    assert store.list_analysis("bob") == [other]
    assert store.get_preferences("bob").language == "zh-CN"


def test_expired_entries_are_invisible_before_cleanup_and_removed_after():
    clock = Clock()
    store = QdrantLongTermStore(clock=clock, retention_seconds=200)
    try:
        store.put_analysis("alice", analysis(expiry=1001.0))
        store.set_preferences("alice", preferences(expiry=1001.0))
        store.put_analysis("bob", analysis())
        clock.now = 1001.0
        assert store.search("alice", "risk") == []
        assert store.list_analysis("alice") == []
        assert store.get_preferences("alice") is None
        assert store._client.count(COLLECTION_NAME).count == 3
        store.cleanup()
        assert store._client.count(COLLECTION_NAME).count == 1
        assert len(store.list_analysis("bob")) == 1
    finally:
        store.close()


def test_run_idempotency_never_refreshes_retention_or_overwrites(store):
    entry = analysis()
    assert store.put_analysis("alice", entry) is True
    assert store.put_analysis("alice", entry.model_copy(deep=True)) is False
    for changed in (analysis(risk=0.1), analysis(expiry=1101.0), analysis(created=991.0)):
        with pytest.raises(SkillError) as error:
            store.put_analysis("alice", changed)
        assert error.value.code == "memory_conflict"
    assert store.list_analysis("alice") == [entry]
    assert store._client.count(COLLECTION_NAME).count == 1


def test_preferences_are_explicit_and_replace_one_entry(store):
    assert store.get_preferences("alice") is None
    store.set_preferences("alice", preferences())
    store.set_preferences("alice", preferences(created=995.0, language="en"))
    assert store.get_preferences("alice") == Preferences(language="en")
    assert store._client.count(COLLECTION_NAME).count == 1
    assert store.search("alice", "language en") == []
    with pytest.raises(SkillError) as error:
        store.set_preferences("alice", analysis())
    assert error.value.code == "invalid_memory"
    with pytest.raises(SkillError):
        store.put_analysis("alice", preferences())


def test_only_typed_facts_are_stored_and_queries_are_not_persisted(store):
    entry = analysis()
    store.put_analysis("alice", entry)
    query = "SECRET PATIENT QUERY RAW HEART RATE 123"
    store.search("alice", query)
    points, _ = store._client.scroll(COLLECTION_NAME, with_vectors=True)
    payload = points[0].payload
    assert set(payload) == {"owner_id", "embedding_version", "kind", "expires_at", "entry"}
    assert payload["embedding_version"] == "char_hash_256_v1"
    assert payload["entry"] == entry.model_dump()
    assert "SECRET" not in json.dumps(payload)
    assert len(points[0].vector) == 256
    first = store.search("alice", "high risk")[0]
    first.entry.analysis.risk_score = 0.0
    assert store.list_analysis("alice")[0].analysis.risk_score == 0.8


def test_chronological_history_uses_all_pages_then_takes_most_recent(store):
    # Insert out of order; >64 entries forces actual owner-filtered pagination.
    for i in reversed(range(70)):
        store.put_analysis("alice", analysis(run=f"run{i:03}", created=900.0 + i))
    store.put_analysis("bob", analysis(run="other", created=999.0))
    assert [entry.source_run_id for entry in store.list_analysis("alice", limit=3)] == [
        "run067",
        "run068",
        "run069",
    ]
    assert len(store.list_analysis("alice", limit=100)) == 70


def test_owner_capacity_is_enforced_and_expiry_releases_space():
    clock = Clock()
    store = QdrantLongTermStore(clock=clock, max_entries_per_owner=2, retention_seconds=200)
    try:
        store.put_analysis("alice", analysis(expiry=1001.0))
        store.set_preferences("alice", preferences())
        store.set_preferences("alice", preferences(language="en"))
        assert store.put_analysis("alice", analysis(expiry=1001.0)) is False
        with pytest.raises(SkillError) as error:
            store.put_analysis("alice", analysis(run="run2"))
        assert error.value.code == "memory_capacity_exceeded"
        store.put_analysis("bob", analysis())
        clock.now = 1001.0
        assert store.put_analysis("alice", analysis(run="run2")) is True
        assert store.get_preferences("alice").language == "en"
    finally:
        store.close()


@pytest.mark.parametrize(
    "entry",
    [analysis(created=1001.0), analysis(expiry=1000.0), analysis(expiry=1200.0)],
)
def test_invalid_retention_is_rejected(store, entry):
    with pytest.raises(SkillError) as error:
        store.put_analysis("alice", entry)
    assert error.value.code == "invalid_memory"
    assert store.list_analysis("alice") == []


@pytest.mark.parametrize("owner", ["../private", "", "alice bob", 7])
def test_invalid_owner_never_reaches_storage(store, owner):
    with pytest.raises(SkillError) as error:
        store.put_analysis(owner, analysis())
    assert error.value.code == "invalid_memory"


@pytest.mark.parametrize(
    "query,limit", [("", 3), ("  ", 3), ("x" * 1001, 3), ("risk", 0), ("risk", True), ("risk", 11)]
)
def test_search_input_is_bounded(store, query, limit):
    with pytest.raises(SkillError) as error:
        store.search("alice", query, limit)
    assert error.value.code == "invalid_memory"


def test_incompatible_collection_and_symlink_paths_fail_closed(tmp_path):
    path = tmp_path / "wrong_vectors"
    client = QdrantClient(path=str(path))
    client.create_collection(
        COLLECTION_NAME, vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT)
    )
    client.close()
    with pytest.raises(SkillError) as error:
        QdrantLongTermStore(path=path)
    assert error.value.code == "memory_unavailable"
    target = tmp_path / "actual"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(SkillError):
        QdrantLongTermStore(path=link)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "private_link").symlink_to(target, target_is_directory=True)
    with pytest.raises(SkillError):
        QdrantLongTermStore(path=nested)


@pytest.mark.parametrize("corruption", ["extra", "version", "retention"])
def test_corrupt_stored_payload_is_rejected_on_read(store, corruption):
    store.put_analysis("alice", analysis())
    point_id = store._point_id("alice", "analysis", "run1")
    if corruption == "extra":
        change = {"query": "SECRET RAW DATA"}
    elif corruption == "version":
        change = {"embedding_version": "untrusted"}
    else:
        change = {"expires_at": 1200.0}
    store._client.set_payload(COLLECTION_NAME, change, points=[point_id])
    for operation in (
        lambda: store.list_analysis("alice"),
        lambda: store.search("alice", "risk"),
    ):
        with pytest.raises(SkillError) as error:
            operation()
        assert error.value.code == "memory_unavailable"
        assert "SECRET" not in str(error.value)


def test_worker_threads_use_one_serialized_local_client(tmp_path):
    store = QdrantLongTermStore(path=tmp_path / "workers", clock=Clock(), retention_seconds=200)
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures = [
                workers.submit(store.put_analysis, "alice", analysis(run=f"run{i}"))
                for i in range(12)
            ]
            assert all(future.result() for future in futures)
            assert workers.submit(store.list_analysis, "alice").result() == store.list_analysis(
                "alice"
            )
        assert len(store.list_analysis("alice")) == 12
    finally:
        store.close()
    with pytest.raises(SkillError) as error:
        store.search("alice", "risk")
    assert error.value.code == "memory_unavailable"
