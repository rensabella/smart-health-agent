"""Evaluation infrastructure tests with mocked HTTP, never live benchmark results."""

import hashlib
import json

import httpx
import pandas as pd
import pytest

from smarthealth.agents.backends import OpenAICompatibleBackend
from smarthealth.agents.contracts import CreateRunRequest
from smarthealth.agents.executor import AgentRuntime
from smarthealth.agents.router import HealthRouter
from smarthealth.evaluation.live_llm.budget import EvaluationBudget
from smarthealth.evaluation.live_llm.config import EvalConfig
from smarthealth.evaluation.live_llm.fault_injection import FaultInjector
from smarthealth.evaluation.live_llm.graders import physical_calls
from smarthealth.evaluation.live_llm.trace_collector import (
    ArtifactCollector,
    EvaluationTransport,
    ObservedBackend,
    SkillCollector,
)
from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.events import RunEventStore
from smarthealth.runtime.harness.harness_validator import HarnessValidator
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.bootstrap import create_registry


def test_artifact_export_preserves_lineage_without_csv_or_dataframe_values():
    store = ArtifactCollector()
    context = SkillContext(owner_id="synthetic_test", session_id="test", input_version="v1")
    csv = b"synthetic_private_marker,heart_rate\nnever_export_this,83.421934\n"
    upload = store.put(csv, kind="upload", context=context)
    frame = pd.DataFrame(
        {"synthetic_private_marker": ["never_export_this"], "heart_rate": [83.421934]}
    )
    dataset = store.put({"frame": frame}, kind="dataset", context=context, source_ref=upload)
    exported = store.export()
    serialized = json.dumps(exported)
    for marker in (
        "synthetic_private_marker",
        "never_export_this",
        "83.421934",
        "heart_rate",
        "frame",
    ):
        assert marker not in serialized
    assert {m["ref"] for m in exported} == {upload, dataset}
    assert exported[1]["source_ref"] == upload
    assert all(len(m["sha256"]) == hashlib.sha256().digest_size * 2 for m in exported)
    exported[0]["kind"] = "tampered"
    assert store.export()[0]["kind"] == "upload"
    assert store.get(upload, kind="upload", context=context) == csv
    assert store.get(dataset, kind="dataset", context=context)["frame"].equals(frame)


@pytest.mark.parametrize("fault", ["invalid_output", "duplicate_execution"])
async def test_skill_collector_does_not_hide_failures_or_physical_duplicate_calls(csv_bytes, fault):
    store = ArtifactCollector()
    context = SkillContext(
        owner_id="synthetic_test", session_id="test", allowed_skills={"csv_analysis"}
    )
    service = AnalysisService(store)
    upload = service.create_upload(csv_bytes, context)
    injector = FaultInjector(fault)
    original = create_registry(service, harness=HarnessValidator())
    collector = SkillCollector(injector.wrap_registry(original), injector)
    try:
        result = await collector.execute("csv_analysis", {"upload_ref": upload}, context)
        calls = collector.calls
        assert len(calls) == 1 and calls[0]["injected"]
        assert calls[0]["arguments_valid"] and calls[0]["allowed"]
        assert calls[0]["end_time"] >= calls[0]["start_time"]
        assert calls[0]["latency_ms"] >= 0
        physical = physical_calls({"registry_events": collector.trace_sink.events_for(context)})
        if fault == "invalid_output":
            assert result.error.code == calls[0]["error_code"] == "invalid_output"
            assert len(physical) == 1 and physical[0]["status"] == "error"
            assert calls[0]["output_refs"] == []
        else:
            assert result.status == calls[0]["status"] == "success"
            assert calls[0]["output_refs"] == [result.data["dataset_ref"]]
            assert len(physical) == context.budget.used == 2
            assert len({call["call_id"] for call in physical}) == 2
            assert all(call["status"] == "success" for call in physical)
    finally:
        await collector.aclose()


def finish_response():
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "infrastructure_test_call",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps(
                                    {"result_kind": "score_definition", "result_ref": None}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 12},
    }


@pytest.mark.parametrize(
    "fault,error",
    [
        ("backend_transient", "backend_unavailable"),
        ("backend_invalid_json", "backend_invalid_response"),
    ],
)
async def test_mock_http_fault_recovery_uses_real_adapter_and_existing_runtime(
    predictor, fault, error
):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=finish_response())

    transport = EvaluationTransport(fault, inner=httpx.MockTransport(respond))
    service = AnalysisService(ArtifactCollector(), predictor)
    registry = create_registry(service)
    budget = EvaluationBudget(EvalConfig(max_total_tokens=200000))
    adapter = OpenAICompatibleBackend(
        "https://provider.invalid/v1", "test-only-secret", "test-model", transport=transport
    )
    backend = ObservedBackend(adapter, budget, registry, transport)
    runtime = AgentRuntime(
        service=service,
        registry=registry,
        backend=backend,
        events=RunEventStore(),
        router=HealthRouter(),
    )
    try:
        request = CreateRunRequest(
            query="Explain the risk score.", session_id="mock_http", input_version="v1"
        )
        initial = await runtime.create(request, owner_id="synthetic_test", request_id="mock_http")
        final = await runtime.wait(
            initial.run_id, owner_id="synthetic_test", session_id=request.session_id
        )
        assert final.status == "completed"
        assert final.llm_requests == final.decision_count == 2
        assert final.tool_calls == 0
        assert len(requests) == transport.network_dispatches == 1
        assert backend.attempts[0]["error_code"] == error
        assert backend.attempts[0]["injected"] and not backend.attempts[0]["network_dispatched"]
        assert backend.attempts[1]["status"] == "success"
        assert not backend.attempts[1]["injected"] and backend.attempts[1]["network_dispatched"]
        assert backend.attempts[1]["input_tokens"] == 42
        assert backend.attempts[1]["output_tokens"] == 12
        snapshot = budget.snapshot()
        assert snapshot["settled_requests"] == 2
        assert snapshot["known_usage_attempts"] == snapshot["unknown_usage_attempts"] == 1
        assert snapshot["known_total_tokens"] == 54
        assert snapshot["retained_unknown_tokens"] > 0
        # This transport call is mocked infrastructure evidence, not a live score.
        assert "test-only-secret" not in json.dumps(
            backend.attempts + backend.actions + transport.events
        )
        assert "arguments" not in json.dumps(backend.actions)
    finally:
        await runtime.aclose()
        await registry.aclose()
