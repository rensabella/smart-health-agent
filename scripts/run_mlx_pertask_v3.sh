#!/bin/zsh
# v3 adapter live rerun: all 20 tasks x 3 reps, per-task isolated budgets.
# Server must be running: mlx_lm.server --model <4bit> --adapter-path <v3 adapter> --port 8800
set -euo pipefail
cd "/Users/rensibo/Smart Health/smart_health_agent"
set -a
source .env.mlx
set +a
ROOT=evaluation/live_runs/20261116-mlx-qwen25-1p5b-sftv3-pertask
TASKS=(ambiguous_task analysis_visualization backend_invalid_json backend_transient duplicate_execution empty_retrieval execution_budget grounded_signal_report invalid_arguments invalid_csv invalid_tool_output missing_upload risk_plot_parameters score_definition stale_artifact tool_timeout unauthorized_tool unsupported_capability unsupported_evidence csv_analysis)
for t in "${TASKS[@]}"; do
  echo "=== [$(date +%H:%M:%S)] task: $t ==="
  .venv/bin/python -m smarthealth.evaluation.live_llm.runner \
    --live --config configs/live_llm.mlx.yaml --task "$t" \
    --output "$ROOT/$t" || echo "task $t exited nonzero (recorded)"
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
