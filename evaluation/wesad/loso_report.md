# WESAD leave-one-subject-out report

- Data table: `training.csv` (sha256 `4ece28fcb7982bfad92b62bffa4ef2770a5ae4090c82747293a60292841de935`).
- Subjects: 15 (S10, S11, S13, S14, S15, S16, S17, S2, S3, S4, S5, S6, S7, S8, S9).
- Windows: 1015 pure-label (1858 mixed dropped, 1 dirty dropped, 0 segments dropped).
- Protocol: 60-row windows, stride 30, classes baseline/stress/amusement, seed 42.

## random_forest

Pooled out-of-fold macro-F1: **0.6061** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.2121 |
| S11 | 70 | 0.7197 |
| S13 | 69 | 0.7093 |
| S14 | 69 | 0.7131 |
| S15 | 69 | 0.4502 |
| S16 | 67 | 0.6183 |
| S17 | 69 | 0.8478 |
| S2 | 65 | 0.4752 |
| S3 | 67 | 0.5931 |
| S4 | 66 | 0.6270 |
| S5 | 67 | 0.6252 |
| S6 | 67 | 0.7453 |
| S7 | 67 | 0.4963 |
| S8 | 68 | 0.5863 |
| S9 | 66 | 0.5611 |

Per-class F1: baseline 0.8275, stress 0.8410, amusement 0.1500

Stress-vs-rest (pooled OOF): threshold 0.490, F1 0.8981, precision 0.9386, recall 0.8609.

## lightgbm

Pooled out-of-fold macro-F1: **0.6428** over 1015 windows.

| subject | windows | macro-F1 |
|---|---|---|
| S10 | 69 | 0.3331 |
| S11 | 70 | 1.0000 |
| S13 | 69 | 0.7393 |
| S14 | 69 | 0.5784 |
| S15 | 69 | 0.3934 |
| S16 | 67 | 0.5877 |
| S17 | 69 | 0.7290 |
| S2 | 65 | 0.4936 |
| S3 | 67 | 0.5076 |
| S4 | 66 | 0.6145 |
| S5 | 67 | 0.7571 |
| S6 | 67 | 0.7917 |
| S7 | 67 | 0.6693 |
| S8 | 68 | 0.6606 |
| S9 | 66 | 0.5863 |

Per-class F1: baseline 0.8139, stress 0.8364, amusement 0.2780

Stress-vs-rest (pooled OOF): threshold 0.740, F1 0.8840, precision 0.9120, recall 0.8576.

