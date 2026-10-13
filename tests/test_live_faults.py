"""Local evaluation-infrastructure tests; these are not live LLM results."""

from pathlib import Path

import pytest
from rag_helpers import ScriptedEmbedding

from smarthealth.evaluation.live_llm.fault_injection import FaultInjector
from smarthealth.retrieval.retrieval import RetrievalService, build_index
from smarthealth.runtime.artifacts import ArtifactStore
from smarthealth.runtime.context import CallBudget, SkillContext
from smarthealth.runtime.harness.harness_validator import HarnessValidator
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.bootstrap import create_registry


def fixture(csv_bytes, fault, *, budget=5):
    context = SkillContext(
        owner_id="evaluation",
        session_id="task",
        input_version="v1",
        allowed_skills={"csv_analysis"},
        budget=CallBudget(budget),
    )
    service = AnalysisService(ArtifactStore())
    upload = service.create_upload(csv_bytes, context)
    inner = create_registry(service, harness=HarnessValidator())
    injector = FaultInjector(fault)
    registry = injector.wrap_registry(inner)
    return service, context, upload, registry, injector


@pytest.mark.parametrize(
    "fault,error,first_budget",
    [
        ("tool_timeout", "timeout", 1),
        ("invalid_output", "invalid_output", 1),
        ("invalid_arguments", "invalid_input", 0),
        ("stale_artifact", "stale_input", 1),
    ],
)
async def test_faults_reach_actual_registry_guards_and_recover(
    csv_bytes, fault, error, first_budget
):
    service, context, upload, registry, injector = fixture(csv_bytes, fault)
    payload = {"upload_ref": upload}
    try:
        first = await registry.execute("csv_analysis", payload, context)
        assert first.error.code == error
        assert context.budget.used == first_budget
        assert injector.triggered
        assert injector.events[-1]["error_code"] == error
        assert service.store.item_count == 1  # Only original upload; no phantom output.
        second = await registry.execute("csv_analysis", payload, context)
        assert second.status == "success"
        dataset = service.store.get(second.data["dataset_ref"], kind="dataset", context=context)
        assert len(dataset["frame"]) > 0
        assert context.budget.used == first_budget + 1
        assert payload == {"upload_ref": upload}
        assert context.input_version == "v1"
        assert sum(e["phase"] == "injected" for e in injector.events) == 1
    finally:
        await registry.aclose()
    assert service.store.item_count == 2  # Timed-out worker cannot write after return.


async def test_duplicate_dispatch_is_real_counted_and_leaves_two_audited_artifacts(csv_bytes):
    service, context, upload, registry, injector = fixture(csv_bytes, "duplicate_execution")
    try:
        result = await registry.execute("csv_analysis", {"upload_ref": upload}, context)
        assert result.status == "success" and context.budget.used == 2
        assert service.store.item_count == 3
        attempts = [e for e in injector.events if e["phase"] == "observed"]
        assert [e["attempt"] for e in attempts] == [1, 2]
        assert len({e["call_id"] for e in attempts}) == 2
        trace = registry.trace_sink.events_for(context)
        assert sum(e["event"] == "skill.completed" for e in trace) == 2
        service.store.get(result.data["dataset_ref"], kind="dataset", context=context)
    finally:
        await registry.aclose()


async def test_duplicate_cannot_bypass_shared_budget(csv_bytes):
    service, context, upload, registry, injector = fixture(
        csv_bytes, "duplicate_execution", budget=1
    )
    try:
        result = await registry.execute("csv_analysis", {"upload_ref": upload}, context)
        assert result.error.code == "budget_exceeded"
        assert context.budget.used == 1 and service.store.item_count == 2
        assert injector.events[-1]["error_code"] == "budget_exceeded"
    finally:
        await registry.aclose()


async def test_wrapping_one_registry_does_not_change_fresh_registry(csv_bytes):
    first_service, context, upload, injected, injector = fixture(csv_bytes, "invalid_output")
    clean = create_registry(first_service, harness=HarnessValidator())
    try:
        assert (await injected.execute("csv_analysis", {"upload_ref": upload}, context)).error
        result = await clean.execute("csv_analysis", {"upload_ref": upload}, context)
        assert result.status == "success"
        with pytest.raises(ValueError, match="only one"):
            injector.wrap_registry(clean)
        snapshot = injector.events
        snapshot[0]["kind"] = "tampered"
        assert injector.events[0]["kind"] == "invalid_output"
    finally:
        await injected.aclose()
        await clean.aclose()


def test_empty_retrieval_suppresses_one_real_index_result_and_keeps_provenance(tmp_path):
    path = tmp_path / "index"
    build_index(Path("knowledge/corpus.json"), path, ScriptedEmbedding())
    original = RetrievalService(path, ScriptedEmbedding(), min_score=0.2)
    injector = FaultInjector("empty_retrieval")
    retrieval = injector.wrap_retrieval(original)
    try:
        empty = retrieval.retrieve("EDA 皮电", top_k=3)
        assert empty.status == "insufficient_evidence" and empty.hits == ()
        assert empty.knowledge_version == original.manifest.knowledge_version
        assert empty.embedding_id == original.manifest.embedding_id
        assert retrieval.manifest is original.manifest
        recovered = retrieval.retrieve("EDA 皮电", top_k=3)
        assert recovered.status == "success" and recovered.hits
        assert len(injector.events) == 1
    finally:
        retrieval.close()


def test_invalid_injection_configuration_fails_before_work():
    with pytest.raises(ValueError, match="Unsupported"):
        FaultInjector("disable_safety")
    with pytest.raises(ValueError, match="actual retrieval"):
        FaultInjector("empty_retrieval").wrap_retrieval(None)
