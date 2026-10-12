# Step 3 verification

Verified locally on 2026-09-08, macOS arm64, Python 3.12.14. Scope: Skill Registry, six physiological analysis Skills and the underlying Tool API. Agent decisions, RAG and system-level evaluations are not part of this run.

| Check | Observed result |
|---|---|
| `python -m pytest` | 38 passed in 14.85s; two upstream deprecation warnings |
| `ruff check app skills tests examples train.py generate_demo_data.py` | All checks passed |
| `ruff format --check app skills tests examples train.py generate_demo_data.py` | 51 files already formatted |
| `python -m pip check` | No broken requirements |
| Default data generation | 57,600 training rows; 300 demo rows |
| Default training | Actual RF and LightGBM bundles; subject-disjoint train/validation/test |
| `python examples/run_skills.py --output-dir docs/example` | Six successful calls, valid JSON/schema exports, three valid PNGs and JSONL trace |
| Uvicorn loopback HTTP smoke | `/health` and `/ready` passed; missing bearer token returned 401; `/analyze` returned 200; three protected PNGs downloaded |
| HTTP log correlation | Request `step3-http-smoke` linked to a Skill run; latency, prediction and model version logged |
| Visual review | Signal, score and importance figures inspected; labels and scope readable |

The default LightGBM demo produced score `0.9983188764750849`, ten windows, maximum-score window index 2, anomaly ratio 0.2. The same score was observed in standalone Skills and real HTTP execution. See `example/api_analysis.json` and `example/training_report.json` for the actual synthetic outputs. Reference IDs in saved examples are no longer live after process shutdown.

The dependency snapshot is `requirements-py312.lock`. This host required an OpenMP runtime in its local virtual environment for LightGBM; its native library setup is not embedded in application code or the Python lock snapshot. See the README for normal platform installation requirements. Other Python and OS combinations have not been verified here.

The two warnings originate from Starlette TestClient's deprecated httpx integration and the deprecated AnyIO BlockingPortal alias. They do not indicate failed application checks. The HTTP smoke run additionally exercised Uvicorn outside TestClient. Its first plot request built a Matplotlib font cache, so that one latency is not a steady-state performance measurement.

The temporary loopback server was stopped after verification. No external LLM calls, medical data upload, public deployment, Docker validation, distributed load test or clinical validation was performed.
