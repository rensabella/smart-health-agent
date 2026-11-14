"""Observed numerators and denominators; unknown values remain unavailable."""

from __future__ import annotations

import math
from collections import Counter

from smarthealth.evaluation.baselines.outcome import SAFETY_TASK_IDS
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


def _contract_scope(record):
    """Baseline grade records are outcome-scoped; contract keys are absent, not passed."""
    return (record.get("grades") or {}).get("check_scope", "contract") == "contract"


def _summary(records):
    executed = [r for r in records if r.get("status") == "executed"]
    graded = [
        r
        for r in executed
        if r.get("grades", {}).get("eligible") and type(r["grades"].get("success")) is bool
    ]
    contract = [r for r in graded if _contract_scope(r)]
    workflow = [r for r in contract if r.get("category") in {"workflow", "rag"}]
    unsupported = [r for r in contract if r.get("category") == "unsupported"]
    fault_configured = [
        r
        for r in records
        if r.get("mode") != "dry_run" and r.get("grades", {}).get("fault", {}).get("configured")
    ]
    fault_exercised = [r for r in fault_configured if r["grades"]["fault"]["triggered"]]
    live = [r for r in records if r.get("grades", {}).get("live_exercised") and _contract_scope(r)]
    attempted = [
        r
        for r in records
        if r.get("status") in {"executed", "infrastructure_error"} or r.get("provider_attempts")
    ]
    attempted_contract = [r for r in attempted if _contract_scope(r)]
    citation_runs = [r for r in contract if "citation_identity" in r["grades"]["checks"]]

    def total(key, rows):
        return sum(r["grades"].get("counts", {}).get(key, 0) for r in rows)

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
    # Method-agnostic outcome partition: identical detectors for every method,
    # computed from the released text and the shared trace fields.
    outcome_attempted = [
        r for r in records if r.get("status") in {"executed", "infrastructure_error"}
    ]

    def components(r):
        return (r.get("grades") or {}).get("outcome_components") or {}

    def outcome_success(r):
        grades = r.get("grades") or {}
        value = grades.get("outcome_success")
        if value is None and grades.get("check_scope") == "outcome":
            value = grades.get("success")
        return value is True

    safety = [r for r in outcome_attempted if r.get("task_id") in SAFETY_TASK_IDS]
    safety_released = [r for r in safety if components(r).get("released")]

    def unsafe(r):
        c = components(r)
        return bool(
            c.get("guard_violations") or c.get("label_violation") or c.get("fabricated_quantitative")
        )

    released = [r for r in graded if components(r).get("released")]
    mention_total = sum((components(r).get("text_citations") or {}).get("total", 0) for r in released)
    mention_valid = sum((components(r).get("text_citations") or {}).get("valid", 0) for r in released)
    citation_required = [r for r in outcome_attempted if r.get("require_citations")]

    def coverage_gap(r):
        grades = r.get("grades") or {}
        if grades.get("check_scope") == "outcome":
            return (components(r).get("text_citations") or {}).get("valid", 0) < 1
        return not (grades.get("checks") or {}).get("citation_identity")

    rag_required = [r for r in outcome_attempted if r.get("requires_rag")]

    def retrieval_exercised(r):
        return any(
            artifact.get("kind") == "evidence" and artifact.get("evidence_ids")
            for artifact in r.get("artifacts") or []
        )

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
        "task_completion_rate": ratio(success(attempted_contract), len(attempted_contract)),
        "agent_quality_rate": ratio(success(contract), len(contract)),
        "live_exercised_task_completion_rate": ratio(success(live), len(live)),
        "workflow_completion_rate": ratio(success(workflow), len(workflow)),
        "unsupported_request_handling_accuracy": ratio(success(unsupported), len(unsupported)),
        "tool_selection_correctness": ratio(
            total("permitted_tool_calls", graded), total("model_tool_calls", graded)
        ),
        "tool_argument_validity": ratio(
            total("valid_arguments", graded), total("model_tool_calls", graded)
        ),
        "tool_success_rate": ratio(
            total("model_physical_tool_successes", graded),
            total("model_physical_tool_calls", graded),
        ),
        "physical_tool_success_rate": ratio(
            total("physical_tool_successes", graded), total("physical_tool_calls", graded)
        ),
        "citation_identity_precision": ratio(
            total("citations_valid", contract), total("citations", contract)
        ),
        "citation_complete_report_rate": ratio(
            sum(r["grades"]["checks"]["citation_identity"] for r in citation_runs),
            len(citation_runs),
        ),
        "runtime_constraint_violation_rate": ratio(
            sum(bool(r["grades"]["invariant_violations"]) for r in graded), len(graded)
        ),
        "outcome": {
            "task_completion_rate": ratio(
                sum(outcome_success(r) for r in outcome_attempted), len(outcome_attempted)
            ),
            # Wide safety denominator: every safety-contract task that was
            # attempted, released text scanned with the production detectors.
            "unsafe_release_on_safety_tasks": {
                **ratio(sum(unsafe(r) for r in safety_released), len(safety)),
                "released": len(safety_released),
                "guard_violations": sum(
                    bool(components(r).get("guard_violations")) for r in safety_released
                ),
                "label_violations": sum(
                    bool(components(r).get("label_violation")) for r in safety_released
                ),
                "fabricated_quantitative": sum(
                    bool(components(r).get("fabricated_quantitative")) for r in safety_released
                ),
                "text_citation_fabrication": sum(
                    bool(components(r).get("text_citation_fabrication")) for r in safety_released
                ),
            },
            "text_citation_identity_precision": ratio(mention_valid, mention_total),
            "citation_coverage_gap_rate": ratio(
                sum(coverage_gap(r) for r in citation_required), len(citation_required)
            ),
            "retrieval_exercised_rate": ratio(
                sum(retrieval_exercised(r) for r in rag_required), len(rag_required)
            ),
        },
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
        "average_tool_calls": ratio(total("physical_tool_calls", graded), len(graded)),
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
    # Records from before Phase 3 carry no method key; they are full_guard runs.
    result["by_method"] = {
        method: _summary([r for r in records if r.get("method", "full_guard") == method])
        for method in sorted({r.get("method", "full_guard") for r in records})
    }
    return result
