"""Observed numerators and denominators; unknown values remain unavailable."""

from __future__ import annotations

import math
from collections import Counter

from smarthealth.evaluation.live_llm.graders import physical_calls


def ratio(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(records):
    executed = [r for r in records if r.get("status") == "executed"]
    graded = [
        r
        for r in executed
        if r.get("grades", {}).get("eligible") and type(r["grades"].get("success")) is bool
    ]
    workflow = [r for r in graded if r.get("category") in {"workflow", "rag"}]
    unsupported = [r for r in graded if r.get("category") == "unsupported"]
    fault_configured = [
        r
        for r in records
        if r.get("mode") != "dry_run" and r.get("grades", {}).get("fault", {}).get("configured")
    ]
    fault_exercised = [r for r in fault_configured if r["grades"]["fault"]["triggered"]]
    live = [r for r in records if r.get("grades", {}).get("live_exercised")]
    attempted = [
        r
        for r in records
        if r.get("status") in {"executed", "infrastructure_error"} or r.get("provider_attempts")
    ]
    citation_runs = [r for r in graded if "citation_identity" in r["grades"]["checks"]]

    def total(key):
        return sum(r["grades"].get("counts", {}).get(key, 0) for r in graded)

    def success(rows):
        return sum(r["grades"]["success"] is True for r in rows)

    latencies = [
        r["latency_ms"]
        for r in executed
        if type(r.get("latency_ms")) in (int, float)
        and math.isfinite(r["latency_ms"])
        and r["latency_ms"] >= 0
    ]
    requests = [
        a for r in records for a in r.get("provider_attempts", []) if a.get("network_dispatched")
    ]
    known_inputs = [a["input_tokens"] for a in requests if type(a.get("input_tokens")) is int]
    known_outputs = [a["output_tokens"] for a in requests if type(a.get("output_tokens")) is int]
    errors = Counter()
    for record in records:
        for code in {record.get("error_code"), (record.get("snapshot") or {}).get("error_code")}:
            if code:
                errors[code] += 1
        for failure in record.get("grades", {}).get("failures", []):
            errors[f"grade:{failure}"] += 1
        for violation in record.get("grades", {}).get("invariant_violations", []):
            errors[f"invariant:{violation}"] += 1
        for attempt in record.get("provider_attempts", []):
            if attempt.get("error_code"):
                errors[f"provider:{attempt['error_code']}"] += 1
        for call in physical_calls(record):
            if call.get("error_code"):
                errors[f"tool:{call['error_code']}"] += 1
    return {
        "counts": {
            "selected": len(records),
            "executed": len(executed),
            "attempted": len(attempted),
            "graded": len(graded),
            "skipped": sum(r.get("status") == "skipped" for r in records),
            "dry_run": sum(r.get("status") == "dry_run" for r in records),
            "infrastructure_errors": sum(
                r.get("status") == "infrastructure_error" for r in records
            ),
            "live_exercised": len(live),
            "live_attempted": sum(
                r.get("grades", {}).get("live_attempted", False) for r in records
            ),
            "router_only": sum(r["grades"].get("router_only", False) for r in graded),
            "faults_configured": len(fault_configured),
            "faults_triggered": len(fault_exercised),
        },
        "coverage": ratio(len(graded), len(records)),
        "task_completion_rate": ratio(success(attempted), len(attempted)),
        "agent_quality_rate": ratio(success(graded), len(graded)),
        "live_exercised_task_completion_rate": ratio(success(live), len(live)),
        "workflow_completion_rate": ratio(success(workflow), len(workflow)),
        "unsupported_request_handling_accuracy": ratio(success(unsupported), len(unsupported)),
        "tool_selection_correctness": ratio(
            total("permitted_tool_calls"), total("model_tool_calls")
        ),
        "tool_argument_validity": ratio(total("valid_arguments"), total("model_tool_calls")),
        "tool_success_rate": ratio(
            total("model_physical_tool_successes"), total("model_physical_tool_calls")
        ),
        "physical_tool_success_rate": ratio(
            total("physical_tool_successes"), total("physical_tool_calls")
        ),
        "citation_identity_precision": ratio(total("citations_valid"), total("citations")),
        "citation_complete_report_rate": ratio(
            sum(r["grades"]["checks"]["citation_identity"] for r in citation_runs),
            len(citation_runs),
        ),
        "runtime_constraint_violation_rate": ratio(
            sum(bool(r["grades"]["invariant_violations"]) for r in graded), len(graded)
        ),
        "attempted_policy_violations": sum(
            r["grades"].get("attempted_violations", 0) for r in graded
        ),
        "prevented_policy_violations": sum(
            r["grades"].get("prevented_violations", 0) for r in graded
        ),
        "recovery_success_rate": ratio(
            sum(r["grades"]["fault"]["recovered"] is True for r in fault_exercised),
            len(fault_exercised),
        ),
        "safe_termination_rate": ratio(
            sum(r["grades"]["fault"]["safe_termination"] is True for r in fault_exercised),
            len(fault_exercised),
        ),
        "fault_exercise_rate": ratio(len(fault_exercised), len(fault_configured)),
        "average_tool_calls": ratio(total("physical_tool_calls"), len(graded)),
        "latency_ms": {
            "samples": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "tokens": {
            "network_requests": len(requests),
            "known_input_tokens": sum(known_inputs) if known_inputs else None,
            "known_output_tokens": sum(known_outputs) if known_outputs else None,
            "input_usage_coverage": ratio(len(known_inputs), len(requests)),
            "output_usage_coverage": ratio(len(known_outputs), len(requests)),
            "complete": bool(requests) and len(known_inputs) == len(known_outputs) == len(requests),
        },
        "semantic_grounding": None,
        "first_token_latency_ms": None,
        "throughput_tasks_per_second": None,
        "skip_reasons": dict(
            Counter(
                r.get("skip_reason", "unspecified") for r in records if r.get("status") == "skipped"
            )
        ),
        "failure_categories": dict(sorted(errors.items())),
    }


def aggregate(records: list[dict]) -> dict:
    """Disjoint mode summaries prevent scripted outcomes becoming live evidence.

    Latency percentiles include executed failures but exclude setup failures,
    skipped cases and plans. Unknown token usage is never silently counted as 0.
    """
    result = _summary(records)
    result["by_mode"] = {
        mode: _summary([r for r in records if r.get("mode") == mode])
        for mode in sorted({r.get("mode", "unknown") for r in records})
    }
    return result
