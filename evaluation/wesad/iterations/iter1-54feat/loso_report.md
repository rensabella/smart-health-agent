# WESAD leave-one-subject-out report

- Data table: `training.csv` (sha256 `4ece28fcb7982bfad92b62bffa4ef2770a5ae4090c82747293a60292841de935`).
- Subjects: 15 (S10, S11, S13, S14, S15, S16, S17, S2, S3, S4, S5, S6, S7, S8, S9).
- Windows: 1015 pure-label (1858 mixed dropped, 1 dirty dropped, 0 segments dropped).
- Protocol: 60-row windows, stride 30, classes baseline/stress/amusement, seed 42.

## random_forest

Pooled out-of-fold macro-F1: **0.6230** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.1905 |
| S11 | 70 | 0.7667 |
| S13 | 69 | 0.7989 |
| S14 | 69 | 0.7131 |
| S15 | 69 | 0.4285 |
| S16 | 67 | 0.6224 |
| S17 | 69 | 0.8979 |
| S2 | 65 | 0.4633 |
| S3 | 67 | 0.6354 |
| S4 | 66 | 0.6270 |
| S5 | 67 | 0.5656 |
| S6 | 67 | 0.8324 |
| S7 | 67 | 0.5493 |
| S8 | 68 | 0.5562 |
| S9 | 66 | 0.5611 |

Per-class F1: baseline 0.8237, stress 0.8349, amusement 0.2105

Stress-vs-rest (pooled OOF): threshold 0.510, F1 0.8629, precision 0.8824, recall 0.8444.

## lightgbm

Pooled out-of-fold macro-F1: **0.6327** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.2710 |
| S11 | 70 | 0.9811 |
| S13 | 69 | 0.7623 |
| S14 | 69 | 0.5623 |
| S15 | 69 | 0.3261 |
| S16 | 67 | 0.5747 |
| S17 | 69 | 0.6743 |
| S2 | 65 | 0.4752 |
| S3 | 67 | 0.5066 |
| S4 | 66 | 0.6270 |
| S5 | 67 | 0.7935 |
| S6 | 67 | 0.7917 |
| S7 | 67 | 0.6611 |
| S8 | 68 | 0.6497 |
| S9 | 66 | 0.5863 |

Per-class F1: baseline 0.8103, stress 0.8406, amusement 0.2472

Stress-vs-rest (pooled OOF): threshold 0.800, F1 0.8748, precision 0.9075, recall 0.8444.

