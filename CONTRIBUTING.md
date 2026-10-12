# Contributing

Start with the [README](README.md), [architecture](docs/architecture.md), and [evaluation definitions](docs/evaluation.md). This repository currently supports editable and wheel installation with a single API process. Changes should preserve the Tool API and make new behavior observable and reproducible.

## Local setup

Use Python 3.10 or later; the recorded local environment is Python 3.12. Install an OpenMP runtime for LightGBM (`libgomp1` on Debian/Ubuntu or `libomp` on macOS), then run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ui]'
mkdir -p data/generated/matplotlib-cache
export MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache"
python -m pytest
ruff check src app skills scripts evaluation frontend tests examples train.py generate_demo_data.py
ruff format --check src app skills scripts evaluation frontend tests examples train.py generate_demo_data.py
python -m pip check
make package-check
```

The test suite trains small model fixtures and uses explicit test embedding/decision backends. It does not require an API key or download the E5 model. Do not use the test adapters as a silent application fallback.

Canonical code belongs in `src/smarthealth`. Root Python files are compatibility entry points; do not add a second implementation there. When changing default profiles or corpus assets, update their packaged resource copies; tests compare the copies.

## Change boundaries

- Add or change a Skill through its Pydantic input/output contract and explicit registry entry. Permissions belong to the server-side context and Harness, not model-writable arguments.
- Keep preprocessing and feature extraction shared by training and inference. Change model or feature protocol versions when compatibility changes; regenerate model bundles under a new output directory.
- Preserve ownership, session, input-version, TTL and artifact lineage checks. Cover meaningful failure cases as well as the intended path.
- Keep report claims, citations, evidence and review versions bound together. Unreviewed report text must not enter SSE, the UI, long-term Memory or persistent traces.
- Keep original sensor data, credentials, local databases, model binaries and temporary output out of Git. Repository examples must remain synthetic and clearly labeled.
- Preserve failed evaluation cases and their denominators. If a label changes, record its source and rationale; do not change labels to improve a score.

## Evaluation

See [evaluation.md](docs/evaluation.md) for explicit model/index preparation and mode selection. A full local RAG evaluation requires the real pinned E5 assets and exclusive access to its Qdrant directory; stop other processes using that directory first.

```bash
smarthealth-eval
```

The checked-in scripted baseline includes one known semantic-review failure. `--fail-on-regression` intentionally returns a nonzero exit for that baseline. Unit-test success does not certify report semantics. Do not mix scripted and live results in one quality claim; live mode requires operator-supplied credentials and incurs provider usage.

## Pull requests

Describe the problem, resulting behavior, scope and relevant validation. Include updated examples or documentation for API, configuration or score-semantics changes. Do not include provider secrets or private data in issues, pull requests, fixtures or screenshots.

The source is published for review; an open-source license has not yet been selected. Confirm contribution and redistribution terms with the repository owner before contributing or reusing material beyond rights already granted by its original source. Third-party source notes retain their own attribution and terms.
