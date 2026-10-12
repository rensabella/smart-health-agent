from dataclasses import replace

import pytest

from app.core.artifacts import ArtifactStore
from app.core.context import SkillContext
from app.core.errors import SkillError


def test_copy_isolation_and_scope():
    store = ArtifactStore()
    ctx = SkillContext(owner_id="owner", session_id="session")
    value = {"data": [1, 2]}
    ref = store.put(value, kind="dataset", context=ctx)
    value["data"].append(3)
    result = store.get(ref, kind="dataset", context=ctx)
    result["data"].append(4)
    assert store.get(ref, kind="dataset", context=ctx) == {"data": [1, 2]}
    for changes, code in [
        ({"owner_id": "another"}, "permission_denied"),
        ({"session_id": "another"}, "permission_denied"),
        ({"input_version": "another"}, "stale_input"),
    ]:
        with pytest.raises(SkillError) as error:
            store.get(ref, kind="dataset", context=replace(ctx, **changes))
        assert error.value.code == code
    with pytest.raises(SkillError) as error:
        store.get(ref, kind="image", context=ctx)
    assert error.value.code == "invalid_artifact_type"


def test_expiry_capacity_and_cancelled_write(monkeypatch):
    import time

    now = time.monotonic()
    ctx = SkillContext(owner_id="owner", session_id="session")
    store = ArtifactStore(ttl_seconds=1, max_items=1)
    ref = store.put(b"abc", kind="upload", context=ctx)
    with pytest.raises(SkillError) as error:
        store.put(b"def", kind="upload", context=ctx)
    assert error.value.code == "artifact_capacity_exceeded"
    monkeypatch.setattr("app.core.artifacts.time.monotonic", lambda: now + 2)
    with pytest.raises(SkillError) as error:
        store.get(ref, kind="upload", context=ctx)
    assert error.value.code == "artifact_expired"
    assert store.item_count == 0
    ctx.cancelled.set()
    with pytest.raises(SkillError) as error:
        store.put(b"late", kind="upload", context=ctx)
    assert error.value.code == "cancelled"


def test_paths_cannot_be_used_as_artifact_refs():
    ctx = SkillContext(owner_id="owner", session_id="session")
    with pytest.raises(SkillError) as error:
        ArtifactStore().get("/etc/passwd", kind="upload", context=ctx)
    assert error.value.code == "invalid_input"
