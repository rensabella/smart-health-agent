# WESAD leave-one-subject-out report

- Data table: `training.csv` (sha256 `4ece28fcb7982bfad92b62bffa4ef2770a5ae4090c82747293a60292841de935`).
- Subjects: 15 (S10, S11, S13, S14, S15, S16, S17, S2, S3, S4, S5, S6, S7, S8, S9).
- Windows: 1015 pure-label (1858 mixed dropped, 1 dirty dropped, 0 segments dropped).
- Protocol: 60-row windows, stride 30, classes baseline/stress/amusement, seed 42.

## random_forest

Pooled out-of-fold macro-F1: **0.6320** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.2129 |
| S11 | 70 | 0.6162 |
| S13 | 69 | 0.7966 |
| S14 | 69 | 0.7389 |
| S15 | 69 | 0.4285 |
| S16 | 67 | 0.6183 |
| S17 | 69 | 0.9088 |
| S2 | 65 | 0.4815 |
| S3 | 67 | 0.6478 |
| S4 | 66 | 0.6145 |
| S5 | 67 | 0.6584 |
| S6 | 67 | 0.8149 |
| S7 | 67 | 0.5456 |
| S8 | 68 | 0.6322 |
| S9 | 66 | 0.5565 |

Per-class F1: baseline 0.8254, stress 0.8431, amusement 0.2276

Stress-vs-rest (pooled OOF): threshold 0.510, F1 0.8767, precision 0.9078, recall 0.8477.

## lightgbm

Pooled out-of-fold macro-F1: **0.6311** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.3189 |
| S11 | 70 | 0.9811 |
| S13 | 69 | 0.7520 |
| S14 | 69 | 0.5712 |
| S15 | 69 | 0.4023 |
| S16 | 67 | 0.5931 |
| S17 | 69 | 0.6727 |
| S2 | 65 | 0.4752 |
| S3 | 67 | 0.4769 |
| S4 | 66 | 0.6270 |
| S5 | 67 | 0.7530 |
| S6 | 67 | 0.7917 |
| S7 | 67 | 0.6693 |
| S8 | 68 | 0.5797 |
| S9 | 66 | 0.5996 |

Per-class F1: baseline 0.8072, stress 0.8369, amusement 0.2491

Stress-vs-rest (pooled OOF): threshold 0.740, F1 0.8799, precision 0.8997, recall 0.8609.

