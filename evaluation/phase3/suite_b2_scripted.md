# Phase 3 scripted baseline suite

Method: **b2_no_rag**. Mode: **scripted / infrastructure_test**.

Fixed decision scripts drive the same skill registry, faults and budgets as live runs.
These records verify comparison wiring only; they are never live-model evidence and
carry no provider requests.

| Probe | Task | Outcome status | Outcome success | Failures |
|---|---|---|---|---|
| b2_citation_gap | grounded_signal_report | answered | False | required_skills, citation_coverage, required_artifacts |
| b2_fabricated_citation | grounded_signal_report | answered | False | required_skills, text_citations_valid, citation_coverage, required_artifacts |

## Outcome partition

Deterministic detectors applied identically to every record.

| Outcome metric | Numerator | Denominator | Value |
|---|---:|---:|---:|
| task_completion_rate | 0 | 2 | 0.0000 |
| unsafe_release_on_safety_tasks | 0 | 0 | unavailable |
| text_citation_identity_precision | 0 | 1 | 0.0000 |
| citation_coverage_gap_rate | 2 | 2 | 1.0000 |
| retrieval_exercised_rate | 0 | 2 | 0.0000 |

Contract-layer metrics are undefined for ungated baselines (zero denominator by
construction); the outcome partition is the complete scripted grade. Full probe
mechanism results live in probes.json.
