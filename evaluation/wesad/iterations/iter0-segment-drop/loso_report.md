# WESAD leave-one-subject-out report

- Data table: `training.csv` (sha256 `4ece28fcb7982bfad92b62bffa4ef2770a5ae4090c82747293a60292841de935`).
- Subjects: 15 (S10, S11, S13, S14, S15, S16, S17, S2, S3, S4, S5, S6, S7, S8, S9).
- Windows: 948 pure-label (1711 mixed dropped, 1 segments dropped).
- Protocol: 60-row windows, stride 30, classes baseline/stress/amusement, seed 42.

## random_forest

Pooled out-of-fold macro-F1: **0.6030** over 948 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.1995 |
| S11 | 70 | 0.8742 |
| S13 | 69 | 0.6581 |
| S14 | 69 | 0.6709 |
| S15 | 69 | 0.4085 |
| S16 | 67 | 0.6183 |
| S17 | 69 | 0.6810 |
| S2 | 65 | 0.5111 |
| S3 | 0 | n/a |
| S4 | 66 | 0.6145 |
| S5 | 67 | 0.5686 |
| S6 | 67 | 0.6911 |
| S7 | 67 | 0.6312 |
| S8 | 68 | 0.5458 |
| S9 | 66 | 0.5743 |

Per-class F1: baseline 0.8335, stress 0.8245, amusement 0.1509

Stress-vs-rest (pooled OOF): threshold 0.520, F1 0.8710, precision 0.9209, recall 0.8262.

## lightgbm

Pooled out-of-fold macro-F1: **0.5909** over 948 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.2428 |
| S11 | 70 | 0.9219 |
| S13 | 69 | 0.8562 |
| S14 | 69 | 0.5994 |
| S15 | 69 | 0.3851 |
| S16 | 67 | 0.5877 |
| S17 | 69 | 0.6089 |
| S2 | 65 | 0.4407 |
| S3 | 0 | n/a |
| S4 | 66 | 0.6270 |
| S5 | 67 | 0.3810 |
| S6 | 67 | 0.6144 |
| S7 | 67 | 0.4945 |
| S8 | 68 | 0.5958 |
| S9 | 66 | 0.6684 |

Per-class F1: baseline 0.7759, stress 0.8257, amusement 0.1712

Stress-vs-rest (pooled OOF): threshold 0.760, F1 0.8389, precision 0.8375, recall 0.8404.

