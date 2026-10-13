"""Metric arithmetic fixtures only, not measured model performance."""

import copy

import pytest

from smarthealth.evaluation.live_llm.graders import grade_task
from smarthealth.evaluation.live_llm.metrics import aggregate
from smarthealth.evaluation.live_llm.runner import task_record
from smarthealth.evaluation.live_llm.tasks import load_tasks


def row(*, mode="infrastructure_test", status="executed", success=True, latency=100, fault=False):
    return {
        "mode": mode,
        "status": status,
        "category": "workflow",
        "latency_ms": latency,
        "provider_attempts": [],
        "grades": {
            "eligible": status == "executed",
            "success": success,
            "checks": {},
            "failures": [],
            "invariant_violations": [],
            "counts": {},
            "fault": {
                "configured": fault,
                "triggered": fault,
                "recovered": success if fault else None,
                "safe_termination": False if fault else None,
            },
            "live_exercised": False,
            "router_only": False,
        },
    }


def test_no_observations_produce_unavailable_metrics_and_percentiles():
    metrics = aggregate([])
    assert metrics["task_completion_rate"]["value"] is None
    assert metrics["citation_identity_precision"]["value"] is None
    assert metrics["recovery_success_rate"]["value"] is None
    assert metrics["latency_ms"] == {"samples": 0, "p50": None, "p95": None}


def test_completion_includes_infrastructure_failure_but_quality_denominator_is_explicit():
    good, failure = row(), row(status="infrastructure_error", success=None, latency=None)
    skipped, dry = (
        row(status="skipped", success=None),
        row(mode="dry_run", status="dry_run", success=None),
    )
    metrics = aggregate([good, failure, skipped, dry])
    assert metrics["task_completion_rate"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert metrics["agent_quality_rate"]["denominator"] == 1
    assert metrics["coverage"]["denominator"] == 4
    assert metrics["counts"]["infrastructure_errors"] == 1
    assert metrics["counts"]["live_exercised"] == 0
    assert metrics["by_mode"]["dry_run"]["task_completion_rate"]["value"] is None


def test_percentiles_include_executed_failures_and_cannot_infer_throughput():
    metrics = aggregate([row(latency=100), row(success=False, latency=300)])
    assert metrics["latency_ms"]["p50"] == 200
    assert metrics["latency_ms"]["p95"] == 290
    assert metrics["throughput_tasks_per_second"] is None
    assert metrics["first_token_latency_ms"] is None


def test_unknown_tokens_are_not_zero_and_injected_failures_are_not_network_usage():
    case = row()
    case["provider_attempts"] = [
        {"network_dispatched": True, "input_tokens": 12, "output_tokens": 4},
        {"network_dispatched": True, "input_tokens": None, "output_tokens": None},
        {"network_dispatched": False, "input_tokens": None, "output_tokens": None},
    ]
    tokens = aggregate([case])["tokens"]
    assert tokens["network_requests"] == 2 and not tokens["complete"]
    assert tokens["known_input_tokens"] == 12
    assert tokens["input_usage_coverage"]["value"] == 0.5


def test_untriggered_fault_is_never_recovery_and_infrastructure_fault_is_in_denominator():
    good = row(fault=True)
    missing = row(fault=True)
    missing["grades"]["fault"].update(triggered=False, recovered=None)
    broken = row(fault=True, status="infrastructure_error", success=None)
    broken["grades"]["fault"]["recovered"] = False
    metrics = aggregate([good, missing, broken])
    assert metrics["recovery_success_rate"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert metrics["fault_exercise_rate"]["denominator"] == 3


@pytest.mark.parametrize("response_status,expected", [("success", True), ("error", False)])
def test_live_mode_requires_verified_provider_decision_not_just_connection(
    response_status, expected
):
    task = load_tasks()[0].tasks[0]
    record = task_record(task, 1, "live")
    record.update(
        status="infrastructure_error",
        provider_attempts=[{"network_dispatched": True, "status": response_status}],
    )
    grade = grade_task(task, record)
    assert grade["live_exercised"] is expected
    assert grade["live_attempted"]
    mocked = copy.deepcopy(record)
    mocked["mode"] = "infrastructure_test"
    assert not grade_task(task, mocked)["live_exercised"]
