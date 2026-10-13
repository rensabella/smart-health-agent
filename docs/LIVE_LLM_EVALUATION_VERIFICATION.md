# Live evaluation framework verification

Date: 2026-10-13. This records implementation verification, **not live model
performance**. External LLM requests during development: **0**. Paid LLM API cost:
**0**. No provider/model has yet been selected for an approved live benchmark.

## Verified implementation

- The new runner uses production `AgentRuntime.create/wait/cancel`, the existing
  provider adapter, Skills, Harness, artifact store, report service and retrieval.
  Existing HTTP contracts and core business algorithms remain unchanged.
- All 20 packaged task definitions validate. Coverage: 2 basic, 2 workflow,
  3 RAG, 5 failure, 5 safety and 3 unsupported/ambiguous tasks.
- [Recorded dry-run report](../evaluation/live_llm_dry_run/report.md),
  [JSON summary](../evaluation/live_llm_dry_run/summary.json) and
  [JSONL task records](../evaluation/live_llm_dry_run/results.jsonl) contain 20
  unexecuted plans, 0 provider requests and unavailable task-success metrics.
- Final local full regression passed **614 tests, 7 warnings, 16.01 seconds** on
  Python 3.12/macOS. The existing 523 tests are preserved, with 91 added cases
  covering the evaluation framework and additive provider options.
- Ruff lint and format checks passed for 254 Python files; `pip check` reported
  no broken requirements. Existing upstream deprecation warnings were not hidden.
- The isolated wheel smoke passed outside the source checkout: real synthetic
  training, Tool API inference, PNG download, all command-line entry points and
  the new 20-task dry-run. It reused installed third-party dependencies and made
  no provider requests.

Infrastructure tests explicitly use `infrastructure_test` records or MockTransport
fixtures. They establish evaluator behavior, not LLM task completion. They cover
actual Skill/runtime error handling, record preservation, report/citation tampering,
artifact lineage, real plot export, budget races/cancellation/unknown usage,
provider request serialization and error parsing, and metric denominators.

## Cloud verification

The [GitHub Actions run](https://github.com/rensabella/smart-health-agent/actions/runs/38421067133)
completed successfully on 2026-10-13 for source commit
`67a5653959f91c4570af80db2550affe4442dfee`:

| Check | Observed result |
|---|---|
| Full regression, Python 3.12.15 / Ubuntu | **614 passed, 5 warnings, 36.49 seconds** |
| Ruff lint and format | Passed; 254 files already formatted |
| Dependency consistency | No broken requirements |
| Wheel build and independent installation | Passed, including real training/API/PNG behavior, console entries and the 20-task dry-run |
| Docker image build | Passed; image tagged `smart-health-agent:ci` on the CI runner |

Both `python` and `container-build` jobs succeeded. The container check builds an
image; it does not publish that image, launch the deployment or establish runtime
readiness. CI uses a disabled production provider and mocked provider responses
only in infrastructure tests. It makes **0 live LLM evaluation requests**.

This record is a documentation-only follow-up. Markdown-only commits are
excluded by the workflow's path filter; the tested source commit above remains
the verification reference.

## Remaining verification

- Live provider protocol compatibility, planning quality, recovery quality, token
  consumption, latency and prices remain unmeasured.
- The checked-in corpus has no independently labeled conflicting-evidence cases;
  citation identity checks do not establish entailment or clinical correctness.
- A fair Guard-off/retry-only comparison is unavailable without changing coupled
  production authorization and publication semantics.
- This evaluates the in-process Runtime; container deployment, multi-user HTTP
  performance, distributed coordination and model first-token latency are outside
  the verified scope.

To perform the first real run, the owner must select a provider/model, configure
credentials locally, supply current token prices if estimated cost is needed,
and explicitly authorize a bounded `--live` invocation. See
[live evaluation instructions](LIVE_LLM_EVALUATION.md).
