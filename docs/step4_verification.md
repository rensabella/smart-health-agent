# Step 4 verification

Verified locally on 2026-09-08 using Python 3.12.14 on macOS arm64. This phase adds Agent execution over the previously trained physiological models. It does not retrain an LLM or claim clinical performance.

| Check | Result |
|---|---|
| Full pytest suite | 87 passed; two pre-existing upstream deprecation warnings |
| Ruff lint/format | Passed |
| pip dependency check | No broken requirements |
| Standalone Planner example | 9 scripted decisions, 6 real Skill calls, 1 delegation, completed |
| Real HTTP/SSE smoke | Subscriber disconnected during running; run continued; contiguous replay through completed |
| HTTP idempotency/auth/images | Same key reused the same run; unauthenticated SSE rejected; protected PNG downloaded |
| Provider adapter integration | Mock HTTP transport exercised actual tool-call parsing and real Skills through Planner and Data Analysis |
| External provider execution | Not run; external LLM calls = 0 |

The standalone example uses the existing default LightGBM bundle. Its highest synthetic anomaly score was `0.9983188764750849`, matching the Tool example. See [run.json](agent_example/run.json), [trace.json](agent_example/trace.json) and [real HTTP smoke record](agent_example/http_smoke.json). The HTTP smoke resumed after event 1 and received events through 39 with no gaps; the run was still running immediately after disconnection. Its loopback server was stopped after verification.

The new tests cover action validation, provider errors/limits/no hidden retries, controlled transient retry accounting, observation-dependent recovery, delegation return to Planner, shared tool budgets, objective mismatch, missing plots, same-session cross-run references, unknown tools, cancellation before execution/during native work/at publication, deadline, final journal failure, bounded result retention, idempotency, session conflict, API body limits, SSE cursor/replay/authorization, parent-run continuation and read-only restart recovery.

Raw queries, delegation tasks, model response text, private exception details and physiological rows are absent from persisted event examples. Public summaries are reconstructed from trusted Tool results or a fixed model card. The publication gate is deliberately limited to data facts; RAG, medical report writing and semantic safety verification remain future work.

The live adapter tests use a mock HTTP transport. Synthetic/scripted and mock-protocol results are not presented as live model planning quality, agent task-completion benchmarks or production throughput. Real provider credentials, model availability, vendor-specific schema compatibility and live decision quality still require a separately configured run. No production deployment, Docker validation or system Evaluation Framework was performed in this phase.

The two warnings remain Starlette TestClient's httpx deprecation and the AnyIO BlockingPortal alias deprecation. The real HTTP smoke independently validated SSE behavior outside TestClient buffering.
