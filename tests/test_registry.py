import asyncio
import json
import threading

import pytest
from pydantic import Field

from app.core.artifacts import ArtifactStore
from app.core.context import CallBudget, SkillContext
from skills.base_skill import BaseSkill, SkillModel
from skills.registry import SkillRegistry


class Input(SkillModel):
    value: int = Field(ge=0)


class Output(SkillModel):
    result: int


class Echo(BaseSkill[Input, Output]):
    name = "echo"
    description = "Test-only deterministic capability"
    input_schema = Input
    output_schema = Output

    def __init__(self):
        self.calls = []

    def execute(self, input, context):
        self.calls.append(input.value)
        return Output(result=input.value + 1)


def context(**kwargs):
    return SkillContext(
        owner_id="owner", session_id="session", allowed_skills=frozenset({"echo"}), **kwargs
    )


@pytest.fixture
async def registry():
    registry = SkillRegistry()
    yield registry
    await registry.aclose()


def test_registration_and_real_nested_schema():
    registry = SkillRegistry()
    registry.register(Echo())
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(Echo())
    assert registry.tool_definitions(set()) == []
    schema = registry.tool_definitions({"echo"})[0]["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["value"]["type"] == "integer"
    assert "output_schema" in registry.describe()[0]
    asyncio.run(registry.aclose())


@pytest.mark.parametrize(
    "payload",
    [
        {"value": "2"},
        {"value": True},
        {"value": -1},
        {"value": 2, "owner_id": "victim"},
        {"value": float("nan")},
        [],
    ],
)
async def test_invalid_inputs_have_no_side_effects(registry, payload):
    skill = Echo()
    registry.register(skill)
    ctx = context()
    result = await registry.execute("echo", payload, ctx)
    assert result.error.code == "invalid_input"
    assert not skill.calls and ctx.budget.used == 0


async def test_unknown_and_empty_allowlist(registry):
    skill = Echo()
    registry.register(skill)
    denied = SkillContext(owner_id="owner", session_id="session")
    result = await registry.execute("echo", {"value": 1}, denied)
    assert result.error.code == "permission_denied"
    assert not skill.calls
    unknown = await registry.execute("missing", {}, context())
    assert unknown.error.code == "skill_not_found"


async def test_concurrent_budget_is_atomic(registry):
    skill = Echo()
    registry.register(skill)
    ctx = context(budget=CallBudget(3))
    results = await asyncio.gather(
        *(registry.execute("echo", {"value": i}, ctx) for i in range(20))
    )
    assert sum(r.status == "success" for r in results) == 3
    assert len(skill.calls) == ctx.budget.used == 3
    assert all(r.error.code == "budget_exceeded" for r in results if r.error)


async def test_output_is_revalidated_even_if_constructed(registry):
    class Bad(Echo):
        def execute(self, input, context):
            return Output.model_construct(result="invalid")

    registry.register(Bad())
    result = await registry.execute("echo", {"value": 1}, context())
    assert result.error.code == "invalid_output" and result.data is None


async def test_trace_and_errors_do_not_leak_values(registry):
    class Explodes(Echo):
        def execute(self, input, context):
            raise RuntimeError("PRIVATE-HEALTH-DATA /private/patient.csv")

    registry.register(Explodes())
    ctx = context()
    result = await registry.execute("echo", {"value": 987654321}, ctx)
    serialized = json.dumps([result.model_dump(), registry.trace_sink.events_for(ctx)])
    assert result.error.code == "internal_error"
    for private in ("PRIVATE-HEALTH-DATA", "/private/patient.csv", "987654321"):
        assert private not in serialized
    assert [e["event"] for e in registry.trace_sink.events_for(ctx)] == [
        "skill.started",
        "skill.failed",
    ]


async def test_timeout_keeps_slot_and_blocks_late_commit():
    release = threading.Event()
    attempted = threading.Event()
    store = ArtifactStore()

    class Slow(Echo):
        timeout_seconds = 0.04

        def execute(self, input, context):
            release.wait(2)
            attempted.set()
            store.put({"value": input.value}, kind="result", context=context)
            return Output(result=1)

    registry = SkillRegistry(max_concurrency=1)
    registry.register(Slow())
    ctx = context()
    try:
        first = await registry.execute("echo", {"value": 1}, ctx)
        assert first.error.code == "timeout"
        second = await registry.execute("echo", {"value": 2}, ctx)
        assert second.error.code == "timeout"
        assert ctx.budget.used == 1  # Queued work never acquired a worker slot.
    finally:
        release.set()
        await registry.aclose()
    assert attempted.is_set() and store.item_count == 0


async def test_async_skill_and_expired_context(registry):
    class AsyncEcho(Echo):
        async def execute(self, input, context):
            await asyncio.sleep(0)
            return Output(result=input.value)

    registry.register(AsyncEcho())
    success = await registry.execute("echo", {"value": 3}, context())
    assert success.data == {"result": 3}
    expired = await registry.execute("echo", {"value": 3}, context(deadline=0))
    assert expired.error.code == "timeout"


async def test_audit_failure_prevents_execution():
    class BrokenSink:
        def emit(self, event):
            raise OSError("private-log-path")

    registry = SkillRegistry(trace_sink=BrokenSink())
    skill = Echo()
    registry.register(skill)
    result = await registry.execute("echo", {"value": 1}, context())
    assert result.error.code == "trace_unavailable" and not skill.calls
    await registry.aclose()
