# Live LLM evaluation integration plan

Date: 2026-10-13. Status: architecture inspected before implementation. No paid
provider call is authorized by this implementation request.

Implementation and free verification now follow this plan. See the
[usage guide](LIVE_LLM_EVALUATION.md) and
[verification evidence](LIVE_LLM_EVALUATION_VERIFICATION.md); no paid run was made.

## Existing architecture

| Concern | Existing implementation | Integration decision |
|---|---|---|
| Provider | `agents/backends.py`: `DecisionBackend`, `OpenAICompatibleBackend`; bounded Chat Completions function calls, redacted errors, no adapter retries | Reuse the real adapter; add optional generation settings with unchanged application defaults. Meter each request in the evaluation layer. |
| Planning / execution | `agents/planner.py` builds permission-scoped requests; `AgentRuntime.create`, `wait`, `cancel` in `agents/executor.py` are also used by the API | Invoke these production entry points, with a fresh isolated runtime/session per case/repetition. Never generate an alternative decision policy. |
| Tool calling | Nine typed Skills; `create_registry`, strict input/output validation, Harness allowlists | Use the same implementations, schemas and Registry. Isolated fault wrappers preserve their contracts. |
| State / guards | RunState, atomic tool budgets, role limits, deadlines, cancellation, one transient decision retry, bounded observations | Preserve all protections. Use existing AgentLimits for task ceilings; no whole-task retries. |
| RAG / citations | Pinned E5 ONNX, verified local Qdrant, corpus hashes, evidence IDs; report publication binds current analysis/evidence/draft/review | Require real prepared assets for RAG cases. Score structural citation validity separately from semantic support. No invented or conflicting corpus documents. |
| Artifacts | ArtifactStore checks owner/session/input version, TTL, source lineage; immutable copies; reports bind draft version/hash | Use only generated synthetic uploads. Export safe artifact manifests/hashes and validated final results before cleanup, never raw sensor payloads or model/provider secrets. |
| Evaluation | Existing `evaluation/{cases,runner,metrics,run_eval}.py` supports scripted/live modes and corpus-bound labels | Leave historical reports and evaluator intact. Add `evaluation/live_llm/` for task-oriented runs, shared suite budgets, repetitions, JSONL and reports. |
| Observability | Runtime events, Skill trace sink, bounded execution spans, structured logs and stable error codes | Collect actual events, tool results and provider usage; no hidden reasoning, raw requests, credential values or exception text. |
| Tests | Existing pytest suite covers runtime, provider HTTP behavior, Skills, RAG, reports, API, SSE, frontend and packaging | Preserve suite; new mock transport/scripted tests validate evaluation infrastructure only and are labeled as such. |

## Implementation sequence

1. Define strict task/config contracts and 20 checked-in benchmark tasks, using
   supported objectives (`data_analysis`, `score_definition`, `health_report`).
2. Add opt-in generation controls to the existing provider, request reservations
   and sanitized attempt records. Unknown usage stays unknown; failed attempts
   retain their reservation rather than being counted as free.
3. Add independent dry-run/live CLI, synthetic fixture generation, per-task
   runtimes, bounded concurrency and repeated runs. Defaults cannot issue paid calls.
4. Add isolated fault injection, trace/artifact export and deterministic graders.
   Test guards with explicit injected events, and distinguish those from actions
   chosen by the model. A fault never reached is skipped/unexercised, not recovered.
5. Aggregate denominators, latency percentiles, known token usage, cost estimates,
   unavailable metrics, infrastructure failures and agent outcomes separately.
6. Run free tests and dry-run; document actual results and exact live commands.
   Live quality results remain unverified until separately approved and executed.

## Benchmark and measurement boundaries

- Tasks cover normal analysis, visualization, supported report/RAG topics,
  unsupported/missing input, and bounded failure recovery. Grade observable final
  artifacts/state plus executed tools, not prose quality or one exact sequence.
- A direct single-Skill final result is not a production Agent objective. Tool
  selection can be checked within complete analysis; do not add a fake objective.
- Standalone medical Q&A and authentic conflicting-source resolution are not
  supported benchmarks in this controlled corpus; disclose coverage gaps.
- Router-only outcomes with zero provider calls are valid system control cases,
  but cannot count as evidence of live LLM capability.
- Runtime rejection is not a successful policy bypass. Report attempted violations
  and violated invariants separately; injected violations do not score as model errors.
- A token/spend cap reserves a conservative configurable input bound plus maximum
  output before dispatch. It is an admission control estimate, not a provider
  billing guarantee; missing usage, provider-specific tokenization and pricing
  must stay explicit. Never log keys or assume a price/model.
- The existing runtime already retries transient decisions once. Retrying a whole
  workflow could duplicate tool side effects; the new runner never does so.
- Guard-off and retry-only baselines cannot isolate the same task semantics:
  permissions, artifact lineage and report publication are coupled to guards.
  Ship only the full-guard configuration and document the comparison limitation.
- No optional LLM judge in the initial release: deterministic citation identity
  does not establish medical truth or claim entailment. Retain the existing
  corpus-labeled evaluation for its narrower semantic cases.

## Acceptance evidence

Produce a schema-validated dry-run report explicitly marked `dry_run`, new local
test results, and reproducible commands. Reports must record task/config/corpus/
model/prompt hashes, input generator seed, provider/model identifiers, fault mode,
budget usage and limitations. Mock-based infrastructure checks and historical
scripted results are never relabeled as live LLM results.
