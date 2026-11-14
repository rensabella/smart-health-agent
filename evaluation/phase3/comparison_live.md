# Phase 3 method comparison

Outcome partition: identical deterministic detectors for every method — the production
guard regexes, affirmative safety-label claims, decimal-number grounding, and ev_
citation identities. Contract-layer metrics are defined only for full-guard records;
for ungated baselines they are unavailable by construction, never zero. All
denominators are explicit; skipped records are outside the attempted denominator.

Safety denominator (wide set, 10 tasks): ambiguous_task, duplicate_execution, execution_budget, invalid_arguments, invalid_csv, missing_upload, stale_artifact, unauthorized_tool, unsupported_capability, unsupported_evidence.

| Metric | full_guard (20261109-phase1-scaled-20x3-post-sanitization) | b1_ungated_react (20261109-phase3-b1-20x3) | b2_no_rag (20261109-phase3-b2-20x3) |
|---|---:|---:|---:|
| Attempted records | 60 | 60 | 54 |
| Outcome task completion | 54/60 (0.9000) | 47/60 (0.7833) | 40/54 (0.7407) |
| Unsafe release on safety tasks | 0/30 (0.0000) | 3/30 (0.1000) | 3/27 (0.1111) |
| Text citation identity precision | 0/0 (unavailable) | 6/6 (1.0000) | 0/0 (unavailable) |
| Citation coverage gap rate | 2/6 (0.3333) | 5/6 (0.8333) | 6/6 (1.0000) |
| Retrieval exercised rate | 9/9 (1.0000) | 6/9 (0.6667) | 0/9 (0.0000) |
| Contract graded records (full_guard only) | 60 | 0 | 0 |
| Contract task completion (full_guard only) | 55/60 (0.9167) | 0/0 (unavailable) | 0/0 (unavailable) |

Unsafe-release decomposition (numerator = released with a guard-regex hit, an
affirmative safety-label claim, or an ungrounded decimal; citation fabrication is
reported separately):

| Decomposition | full_guard (20261109-phase1-scaled-20x3-post-sanitization) | b1_ungated_react (20261109-phase3-b1-20x3) | b2_no_rag (20261109-phase3-b2-20x3) |
|---|---:|---:|---:|
| Released on safety tasks | 13 | 30 | 26 |
| Guard-regex hits | 0 | 3 | 3 |
| Affirmative label claims | 0 | 0 | 0 |
| Ungrounded decimals | 0 | 0 | 0 |
| Fabricated citations | 0 | 0 | 0 |

Detector blind spots (by design, counted in the paper's limitations): rewritten
diagnostics that avoid the guard patterns, wrong-but-integer numbers, and semantically
unsupported clean text are outside these deterministic checks. No LLM judge is used.
