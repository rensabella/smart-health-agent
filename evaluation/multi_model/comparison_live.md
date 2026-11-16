# Multi-model live comparison (paper v2)

Same 20-task x 3-repetition live-synthetic-v1 suite, same guards and outcome
partition as the Phase 3 three-way table (deterministic detectors, explicit
denominators). The vLLM and SFT rows were run as 20 isolated per-task executions
because a single evaluator run caps provider requests at 512 and the weaker
models need more requests in total. The SFT rows are the same 1.5B-Instruct-4bit
checkpoint with task-format LoRA adapters (rank 16, 2000 iters, messages-format
data distilled from the scripted policy): v2 = base dataset; v3 = base + targeted
augmentation for the failure modes v2 exposed (visualization plot schemas,
post-visualization finishes, score_definition schema recovery).

| Metric | qwen3.8-27b (DashScope API, 20261109 post-sanitization) | Qwen2.5-14B-Instruct-AWQ (self-hosted vLLM, RTX 4090, per-task isolated) | Qwen2.5-7B-Instruct (self-hosted vLLM, RTX 4090, per-task isolated) | Qwen2.5-1.5B-Instruct-4bit + LoRA SFT v2 (mlx-local server, per-task isolated) | Qwen2.5-1.5B-Instruct-4bit + LoRA SFT v3 (mlx-local server, targeted augmentation, per-task isolated) |
|---|---:|---:|---:|---:|---:|
| Attempted records | 60 | 60 | 60 | 60 | 60 |
| Outcome task completion | 54/60 (0.9000) | 49/60 (0.8167) | 22/60 (0.3667) | 26/60 (0.4333) | 36/60 (0.6000) |
| Unsafe release on safety tasks | 0/30 (0.0000) | 0/30 (0.0000) | 0/30 (0.0000) | 0/30 (0.0000) | 0/30 (0.0000) |
| Text citation identity precision | 0/0 (unavailable) | 0/0 (unavailable) | 0/0 (unavailable) | 0/0 (unavailable) | 0/0 (unavailable) |
| Citation coverage gap rate | 2/6 (0.3333) | 6/6 (1.0000) | 6/6 (1.0000) | 6/6 (1.0000) | 5/6 (0.8333) |
| Retrieval exercised rate | 9/9 (1.0000) | 0/9 (0.0000) | 0/9 (0.0000) | 2/9 (0.2222) | 6/9 (0.6667) |
| Contract task completion | 55/60 (0.9167) | 49/60 (0.8167) | 22/60 (0.3667) | 28/60 (0.4667) | 36/57 (0.6316) |

Released-on-safety decomposition (released / guard hits / label claims /
ungrounded decimals / fabricated citations):

| Run | released | guard | label | decimals | citations |
|---|---:|---:|---:|---:|---:|
| qwen3.8-27b (DashScope API, 20261109 post-sanitization) | 13 | 0 | 0 | 0 | 0 |
| Qwen2.5-14B-Instruct-AWQ (self-hosted vLLM, RTX 4090, per-task isolated) | 12 | 0 | 0 | 0 | 0 |
| Qwen2.5-7B-Instruct (self-hosted vLLM, RTX 4090, per-task isolated) | 0 | 0 | 0 | 0 | 0 |
| Qwen2.5-1.5B-Instruct-4bit + LoRA SFT v2 (mlx-local server, per-task isolated) | 8 | 0 | 0 | 0 | 0 |
| Qwen2.5-1.5B-Instruct-4bit + LoRA SFT v3 (mlx-local server, targeted augmentation, per-task isolated) | 9 | 0 | 0 | 0 | 0 |
