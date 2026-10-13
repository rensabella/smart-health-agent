"""Human-readable summaries of measured records, with explicit coverage limits."""

from __future__ import annotations

from smarthealth.evaluation.live_llm.metrics import aggregate


def _cell(value):
    if value is None:
        return "unavailable"
    if value == "":
        return "not configured"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _metric_table(metrics):
    lines = ["| Metric | Numerator | Denominator | Value |", "|---|---:|---:|---:|"]
    for key, value in metrics.items():
        if isinstance(value, dict) and set(value) == {"numerator", "denominator", "value"}:
            shown = "unavailable" if value["value"] is None else f"{value['value']:.4f}"
            lines.append(
                f"| {_cell(key)} | {value['numerator']} | {value['denominator']} | {shown} |"
            )
    return lines


def render_markdown(report: dict) -> str:
    """Render only reviewed metadata fields, never arbitrary config or raw traces."""
    records = report.get("records", [])
    metrics = report.get("metrics") or aggregate(records)
    metadata = report.get("metadata", report.get("experiment", {}))
    config = report.get("config", {})
    mode = report.get("mode", metadata.get("mode", "unspecified"))
    lines = [
        "# SmartHealth LLM evaluation report",
        "",
        f"Mode: **{_cell(mode)}**.",
        "",
        "Dry-run records are unexecuted plans. Infrastructure-test records use mocks or scripted "
        "decisions and are not live LLM evaluation. A live-mode record is live-exercised only "
        "when a dispatched provider request returned a valid decision; attempted connections "
        "without a valid response and zero-call router outcomes are separate.",
        "",
        "| Setting | Recorded value |",
        "|---|---|",
    ]
    for key in (
        "generated_at",
        "evaluation_date",
        "started_at",
        "finished_at",
        "provider",
        "model",
        "baseline",
        "dataset_hash",
        "benchmark_hash",
        "model_manifest_hash",
        "config_hash",
        "corpus_hash",
        "model_hash",
        "prompt_hash",
        "fixture_seed",
        "temperature",
        "seed",
        "max_completion_tokens",
        "max_tasks",
        "repetitions",
        "concurrency",
    ):
        value = report.get(key, metadata.get(key, config.get(key)))
        if value is not None:
            lines.append(f"| {_cell(key)} | {_cell(value)} |")
    lines += ["", "## Execution coverage", "", "| Outcome | Count |", "|---|---:|"]
    lines.extend(f"| {_cell(k)} | {_cell(v)} |" for k, v in metrics.get("counts", {}).items())
    for run_mode, summary in metrics.get("by_mode", {}).items():
        label = {
            "dry_run": "Dry run — unexecuted",
            "infrastructure_test": "Infrastructure tests — not live",
            "live": "Live mode — provider-exercised coverage shown separately",
        }.get(run_mode, run_mode)
        lines += ["", f"## {_cell(label)}", ""] + _metric_table(summary)
        latency = summary.get("latency_ms", {})
        lines += [
            "",
            f"Executed latency samples: {_cell(latency.get('samples'))}; "
            f"p50: {_cell(latency.get('p50'))} ms; p95: {_cell(latency.get('p95'))} ms.",
        ]
        tokens = summary.get("tokens", {})
        lines += [
            "",
            f"Known input tokens: {_cell(tokens.get('known_input_tokens'))}; "
            f"known output tokens: {_cell(tokens.get('known_output_tokens'))}; "
            f"usage complete: {_cell(tokens.get('complete'))}.",
        ]
    lines += [
        "",
        "## Budget and cost",
        "",
        "Admission-control estimates reserve a configurable input bound and maximum output "
        "before dispatch. They are not provider billing guarantees. Failed attempts with unknown "
        "usage retain reservations; unknown usage is not free.",
    ]
    budget = report.get("budget", {})
    if budget:
        lines += ["", "| Budget measurement | Value |", "|---|---|"]
        allowed = {
            "admitted_requests",
            "settled_requests",
            "in_flight_requests",
            "known_usage_attempts",
            "unknown_usage_attempts",
            "known_total_tokens",
            "charged_tokens",
            "in_flight_reserved_tokens",
            "retained_unknown_tokens",
            "estimated_known_cost",
            "estimated_charged_cost",
            "retained_unknown_estimated_cost",
            "overrun",
            "max_provider_requests",
            "reservation_method",
            "billing_guarantee",
            "requests",
            "provider_requests",
            "reserved_requests",
            "reserved_tokens",
            "known_tokens",
            "input_tokens",
            "output_tokens",
            "known_input_tokens",
            "known_output_tokens",
            "estimated_cost",
            "estimated_cost_usd",
            "reserved_cost",
            "reserved_cost_usd",
            "usage_complete",
            "max_total_tokens",
            "max_estimated_cost",
            "total_tokens",
            "dispatched_requests",
            "unknown_usage_requests",
            "retained_reserved_tokens",
        }
        for key, value in budget.items():
            if key in allowed and not isinstance(value, (dict, list)):
                lines.append(f"| {_cell(key)} | {_cell(value)} |")
    if (
        config.get("input_price_per_million") is None
        or config.get("output_price_per_million") is None
    ):
        lines += [
            "",
            "API cost is unavailable: both operator-supplied token prices were not provided.",
        ]
    lines += ["", "## Skips and failure categories", "", "| Category | Count |", "|---|---:|"]
    categories = {
        **{f"skip:{k}": v for k, v in metrics.get("skip_reasons", {}).items()},
        **metrics.get("failure_categories", {}),
    }
    lines.extend(f"| {_cell(k)} | {_cell(v)} |" for k, v in categories.items())
    if not categories:
        lines.append("| No recorded failures or skips | 0 |")
    failures = [
        r
        for r in records
        if r.get("grades", {}).get("success") is False
        or r.get("status") in {"skipped", "infrastructure_error"}
    ][:10]
    if failures:
        lines += [
            "",
            "Representative cases (identifiers and error codes only):",
            "",
            "| Task | Repetition | Outcome | Reason |",
            "|---|---:|---|---|",
        ]
        for record in failures:
            reasons = list(record.get("grades", {}).get("failures", []))
            reasons += record.get("grades", {}).get("invariant_violations", [])
            reason = ", ".join(reasons) or record.get("skip_reason") or record.get("error_code")
            lines.append(
                f"| {_cell(record.get('task_id'))} | {_cell(record.get('repetition'))} | "
                f"{_cell(record.get('status'))} | {_cell(reason)} |"
            )
    lines += [
        "",
        "## Measurement limits",
        "",
        "- Ratios show actual denominators; zero-denominator values are unavailable. Skips, "
        "setup errors and dry runs do not enter conditional agent-quality denominators. "
        "Task completion includes attempted infrastructure failures and partially executed "
        "budget-limited tasks. Asset skips remain outside attempts; coverage exposes exclusions.",
        "- Required Skills must succeed. Artifacts, schema, lineage, publication guards, "
        "numerical reference tolerance (absolute/relative 1e-9), and citation identities are "
        "checked without prescribing one exact tool order.",
        "- Tool selection and arguments count model-chosen actions. Injected extra calls and "
        "fault outcomes are separated from model execution metrics; physical counts include "
        "duplicates. Rejected policy attempts are distinct from invariant violations.",
        "- Recovery requires a triggered fault and successful completion. Safe termination is "
        "reported separately. An untriggered fault is unexercised, never recovered.",
        "- Citation identity and provenance do not establish semantic entailment or medical "
        "truth. Semantic grounding and clinical safety remain unverified; no LLM judge is used.",
        "- First-token latency is unavailable for nonstreaming provider decisions. SSE progress "
        "events are not model tokens. Throughput requires measured suite wall time and is not "
        "inferred by adding concurrent per-task latencies.",
        "- Only the full-guard configuration is available. Guard-off and retry-only variants "
        "cannot fairly isolate this runtime's coupled permissions, lineage and publication rules.",
        "- The controlled synthetic corpus does not support authentic conflicting-source "
        "resolution or general medical question answering benchmarks.",
        "",
    ]
    return "\n".join(lines)
