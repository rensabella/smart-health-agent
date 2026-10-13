# SmartHealth LLM evaluation report

Mode: **dry_run**.

Dry-run records are unexecuted plans. Infrastructure-test records use mocks or scripted decisions and are not live LLM evaluation. A live-mode record is live-exercised only when a dispatched provider request returned a valid decision; attempted connections without a valid response and zero-call router outcomes are separate.

| Setting | Recorded value |
|---|---|
| evaluation_date | 2026-10-13T13:45:32.002152+00:00 |
| provider | openai_compatible |
| model | not configured |
| baseline | full_guard |
| benchmark_hash | a8df7d40850af2abf453465995f6e938516aad44d897cc0b6a936058a4ea7483 |
| config_hash | 71e22952baf115e9f80fd896cb27ba32c972a1869d599671e501a32bf69d1345 |
| prompt_hash | 7c10f6c698d22b6ea4127778156ad419606f5252f164f8e17da797e42ca74fc7 |
| fixture_seed | 10042 |
| temperature | 0.0 |
| seed | 42 |
| max_completion_tokens | 2048 |
| max_tasks | 20 |
| repetitions | 1 |
| concurrency | 1 |

## Execution coverage

| Outcome | Count |
|---|---:|
| selected | 20 |
| executed | 0 |
| attempted | 0 |
| graded | 0 |
| skipped | 0 |
| dry_run | 20 |
| infrastructure_errors | 0 |
| live_exercised | 0 |
| live_attempted | 0 |
| router_only | 0 |
| faults_configured | 0 |
| faults_triggered | 0 |

## Dry run — unexecuted

| Metric | Numerator | Denominator | Value |
|---|---:|---:|---:|
| coverage | 0 | 20 | 0.0000 |
| task_completion_rate | 0 | 0 | unavailable |
| agent_quality_rate | 0 | 0 | unavailable |
| live_exercised_task_completion_rate | 0 | 0 | unavailable |
| workflow_completion_rate | 0 | 0 | unavailable |
| unsupported_request_handling_accuracy | 0 | 0 | unavailable |
| tool_selection_correctness | 0 | 0 | unavailable |
| tool_argument_validity | 0 | 0 | unavailable |
| tool_success_rate | 0 | 0 | unavailable |
| physical_tool_success_rate | 0 | 0 | unavailable |
| citation_identity_precision | 0 | 0 | unavailable |
| citation_complete_report_rate | 0 | 0 | unavailable |
| runtime_constraint_violation_rate | 0 | 0 | unavailable |
| recovery_success_rate | 0 | 0 | unavailable |
| safe_termination_rate | 0 | 0 | unavailable |
| fault_exercise_rate | 0 | 0 | unavailable |
| average_tool_calls | 0 | 0 | unavailable |

Executed latency samples: 0; p50: unavailable ms; p95: unavailable ms.

Known input tokens: unavailable; known output tokens: unavailable; usage complete: False.

## Budget and cost

Admission-control estimates reserve a configurable input bound and maximum output before dispatch. They are not provider billing guarantees. Failed attempts with unknown usage retain reservations; unknown usage is not free.

| Budget measurement | Value |
|---|---|
| admitted_requests | 0 |
| settled_requests | 0 |
| in_flight_requests | 0 |
| known_usage_attempts | 0 |
| unknown_usage_attempts | 0 |
| known_input_tokens | 0 |
| known_output_tokens | 0 |
| known_total_tokens | 0 |
| charged_tokens | 0 |
| in_flight_reserved_tokens | 0 |
| retained_unknown_tokens | 0 |
| estimated_known_cost | unavailable |
| estimated_charged_cost | unavailable |
| retained_unknown_estimated_cost | unavailable |
| usage_complete | False |
| overrun | False |
| max_provider_requests | 32 |
| max_total_tokens | 300000 |
| max_estimated_cost | unavailable |
| reservation_method | serialized_request_bytes_plus_completion_cap |
| billing_guarantee | False |

API cost is unavailable: both operator-supplied token prices were not provided.

## Skips and failure categories

| Category | Count |
|---|---:|
| No recorded failures or skips | 0 |

## Measurement limits

- Ratios show actual denominators; zero-denominator values are unavailable. Skips, setup errors and dry runs do not enter conditional agent-quality denominators. Task completion includes attempted infrastructure failures and partially executed budget-limited tasks. Asset skips remain outside attempts; coverage exposes exclusions.
- Required Skills must succeed. Artifacts, schema, lineage, publication guards, numerical reference tolerance (absolute/relative 1e-9), and citation identities are checked without prescribing one exact tool order.
- Tool selection and arguments count model-chosen actions. Injected extra calls and fault outcomes are separated from model execution metrics; physical counts include duplicates. Rejected policy attempts are distinct from invariant violations.
- Recovery requires a triggered fault and successful completion. Safe termination is reported separately. An untriggered fault is unexercised, never recovered.
- Citation identity and provenance do not establish semantic entailment or medical truth. Semantic grounding and clinical safety remain unverified; no LLM judge is used.
- First-token latency is unavailable for nonstreaming provider decisions. SSE progress events are not model tokens. Throughput requires measured suite wall time and is not inferred by adding concurrent per-task latencies.
- Only the full-guard configuration is available. Guard-off and retry-only variants cannot fairly isolate this runtime's coupled permissions, lineage and publication rules.
- The controlled synthetic corpus does not support authentic conflicting-source resolution or general medical question answering benchmarks.
