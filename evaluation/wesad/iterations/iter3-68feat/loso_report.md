# WESAD leave-one-subject-out report

- Data table: `training.csv` (sha256 `4ece28fcb7982bfad92b62bffa4ef2770a5ae4090c82747293a60292841de935`).
- Subjects: 15 (S10, S11, S13, S14, S15, S16, S17, S2, S3, S4, S5, S6, S7, S8, S9).
- Windows: 1015 pure-label (1858 mixed dropped, 1 dirty dropped, 0 segments dropped).
- Protocol: 60-row windows, stride 30, classes baseline/stress/amusement, seed 42.

## random_forest

Pooled out-of-fold macro-F1: **0.6204** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.1995 |
| S11 | 70 | 0.7226 |
| S13 | 69 | 0.8315 |
| S14 | 69 | 0.8040 |
| S15 | 69 | 0.4350 |
| S16 | 67 | 0.6138 |
| S17 | 69 | 0.8478 |
| S2 | 65 | 0.4407 |
| S3 | 67 | 0.6598 |
| S4 | 66 | 0.6145 |
| S5 | 67 | 0.6289 |
| S6 | 67 | 0.7202 |
| S7 | 67 | 0.4780 |
| S8 | 68 | 0.5363 |
| S9 | 66 | 0.6830 |

Per-class F1: baseline 0.8192, stress 0.8341, amusement 0.2080

Stress-vs-rest (pooled OOF): threshold 0.550, F1 0.8723, precision 0.9389, recall 0.8146.

## lightgbm

Pooled out-of-fold macro-F1: **0.6223** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.2129 |
| S11 | 70 | 1.0000 |
| S13 | 69 | 0.7600 |
| S14 | 69 | 0.6552 |
| S15 | 69 | 0.3261 |
| S16 | 67 | 0.6034 |
| S17 | 69 | 0.7290 |
| S2 | 65 | 0.4752 |
| S3 | 67 | 0.4169 |
| S4 | 66 | 0.6016 |
| S5 | 67 | 0.6303 |
| S6 | 67 | 0.8321 |
| S7 | 67 | 0.6693 |
| S8 | 68 | 0.5834 |
| S9 | 66 | 0.6318 |

Per-class F1: baseline 0.8014, stress 0.8305, amusement 0.2348

Stress-vs-rest (pooled OOF): threshold 0.560, F1 0.8365, precision 0.7964, recall 0.8808.

