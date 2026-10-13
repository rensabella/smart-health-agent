# Package refactor verification

Date: 2026-10-13, Asia/Shanghai. Scope: the incremental `0.2.0` migration from
source-root `app`/`skills` packages to `src/smarthealth`.

## Change boundary

The repository was inspected and the architecture/migration analysis was provided
before any code changes. Existing Agent, Skill, Runtime, Retrieval, API and
Evaluation implementations were identified; infrastructure already present was
extended rather than presented as new functionality.

The original business implementation was migrated with import updates. A
structural comparison of **22 domain modules** (pipeline, predictors, analysis
service, explanations, visualization, report/schema modules and Skill adapters)
found identical Python ASTs after excluding import statements. The executor's
existing state and decision-request construction were extracted; its policy and
publication logic remain in use. Source-checkout compatibility modules retain
the old entry points and share canonical module identity.

## Executed checks

| Check | Result and scope |
|---|---|
| Existing behavior suite after migration | Original 499 tests passed |
| Local full suite including new regressions | **523 passed, 7 warnings, 13.86 seconds** on Python 3.12/macOS |
| Cloud full suite | **523 passed, 5 warnings, 32.70 seconds** on Python 3.12.15/Ubuntu; source `3c7175d` |
| New regression scope | Settings validation/injection, secret exclusion, packaged resources, legacy module identity, span correlation/failure/cancellation, bounded retention and symlink boundaries |
| Domain preservation | 22 modules structurally unchanged except imports |
| Wheel build | `smart_health_agent-0.2.0-py3-none-any.whl` built successfully |
| Independent wheel installation | Installed in a disposable venv and run from a temporary directory; no checkout/src import path |
| Wheel runtime smoke | Real synthetic dataset → RandomForest training → verified bundle loading → `/health`, `/ready`, `/predict`, `/analyze` → three plots and protected PNG retrieval |
| Console entry points | API, generator, trainer, index builder and evaluation `--help` passed outside the checkout |
| Packaged resources | Both policy profiles and the corpus-bound evaluation cases loaded outside the checkout |
| Ruff / dependencies | Lint passed; 238 Python files formatted; `pip check` reported no broken requirements |
| Documentation / whitespace | 53 local Markdown link targets checked; `git diff --check` passed |

The disposable venv reuses the test environment's third-party packages through a
dependency path. It installs the project's wheel locally and asserts the import
origin. This validates packaging and runtime behavior without claiming fresh
dependency resolution on a second OS. The Streamlit behavior remains covered by
the existing AppTest/HTTP client tests; this smoke does not claim a browser or
container deployment test.

The wheel check found a macOS installation-prefix issue (`/var` resolving to
`/private/var`). Trusted bundled resources now resolve their installed prefix;
operator-supplied index paths retain symlink components so their original strict
checks still reject them. A regression test covers each side of that boundary.

The local seven warnings remain upstream Starlette/AnyIO and Streamlit/NumPy
deprecation notices; the cloud environment reported five warnings. No warning
suppression was added. Historical ML/E5/system evaluation
results and the known semantic-review failure were not relabeled or remeasured.
New training was limited to synthetic test and wheel-smoke fixtures.

## Reproduce

```bash
python -m pip install -e '.[dev,ui]'
make check
make package-check
```

`make package-check` requires build dependencies; no LLM credential or embedding
model download is needed. Mutable outputs and temporary models are outside the
installed package. CI has been updated to run the wheel check after pytest.

## Publication and remaining verification

The refactor is published at source commit
[`3c7175d`](https://github.com/rensabella/smart-health-agent/commit/3c7175d4186eb1bce621b8dcec4e78607bb32ab1).
[GitHub Actions run 38275517363](https://github.com/rensabella/smart-health-agent/actions/runs/38275517363)
completed successfully on 2026-10-13, Asia/Shanghai (2026-10-12 UTC):

- `python`: dependency installation, `pip check`, Ruff lint, formatting of 238
  Python files, 523 tests, wheel build and independent wheel smoke all passed.
- `container-build`: the updated `src` layout successfully built a Docker image.

The wheel smoke passed on both the local macOS environment and the Ubuntu runner.
It includes real synthetic training, API inference, three plots and PNG retrieval;
it does not run a live LLM or the full historical E5 evaluation. These results
belong to the source commit above. Subsequent Markdown-only verification updates
skip CI under the existing workflow rules.

Docker is unavailable on the local host. Container preparation/startup/SSE/restart,
live LLM quality, multi-user isolation and distributed deployment remain separate
work; image-build success does not establish container runtime behavior. A
repository license still needs an owner decision.

See [architecture.md](architecture.md), [deployment.md](deployment.md) and
[evaluation.md](evaluation.md) for the implemented boundaries and benchmark plan.
