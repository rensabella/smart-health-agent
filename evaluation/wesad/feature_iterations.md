# WESAD LOSO feature iteration log (honest protocol)

Protocol: every iteration is scored only by pooled out-of-fold (LOSO) macro-F1
on the same 15-subject canonical table (`prepared/training.csv`, zip sha256
5e15d2606adf16d819ba535c1786d92e94bae5ff5392f7ae01fb63f9938fd71c). No decision
ever uses a test fold; guards and graders are untouched. The frozen feature set
is the last row before the deploy table is written.

## iter0 — segment-level quality granularity (superseded)

54 features (8 stats x 6 channels + LF/HF + resp freq + eda slope + scr_count),
training-side windowing dropped the WHOLE segment when any window failed the
shared cleaning rule. S3 was excluded entirely (one window at 21.7% HR missing
> 20% per-channel cap; 580 one-second HR gaps elsewhere were all acceptable).

- Windows: 948 pure-label (1711 mixed, 1 segment / 6493 rows dropped)
- random_forest pooled macro-F1: 0.6030 (baseline 0.8335 / stress 0.8245 / amusement 0.1509)
- lightgbm pooled macro-F1: 0.5909
- Stress-vs-rest (RF): threshold 0.520, F1 0.8710

Fix before any feature work: dirty windows are now dropped and counted
individually (same shared acceptance rule; only the failure granularity
changed, mirroring the mixed-label drop-and-count policy). iter1 re-scores the
unchanged 54-feature set on the recovered corpus.

## iter1 — dirty-window drop-and-count (corpus fix, features unchanged)

Same 54 features; training-side windowing now drops and counts individual dirty
windows instead of the whole segment (shared acceptance rule unchanged; S3's
single 21.7%-HR window no longer forfeits its 67 clean windows).

- Windows: 1015 pure-label (1858 mixed, 1 dirty, 0 segments dropped)
- random_forest pooled macro-F1: 0.6230 (baseline 0.8237 / stress 0.8349 / amusement 0.2105)
- lightgbm pooled macro-F1: 0.6327 (baseline 0.8103 / stress 0.8406 / amusement 0.2472)
- S3 fold (RF): 0.6354 over 67 windows; stress-vs-rest (RF) threshold 0.51

Bottleneck unchanged: amusement is confused with baseline (moderate arousal,
positive valence; chest arousal channels carry little valence information).

## iter2 — cross-subject shape helpers (65 features)

Added 11 helpers to the 54: dispersion ratios hr_cv and eda_cv (remove each
subject's own scale), temp_range and eda_range (burst shape), hr_diff_std,
normalized LF/HF power (hr_lf_nu, hr_hf_nu), and first/second-half slopes for
hr and eda.

- Windows: 1015 pure-label (1858 mixed, 1 dirty)
- random_forest pooled macro-F1: 0.6320 (baseline 0.8254 / stress 0.8431 / amusement 0.2276)
- lightgbm pooled macro-F1: 0.6311 (baseline 0.8072 / stress 0.8369 / amusement 0.2491)
- Stress-vs-rest F1: RF 0.8767 (thr 0.51) / LGBM 0.8799 (thr 0.74)

OOF importance diagnostic (RF, pooled out-of-fold): top features
eda_mean_abs_change 0.084, hr_mean 0.075, hr_p75 0.065, eda_range 0.043;
channel totals eda 0.32 / hr 0.27 / resp 0.13 / temp 0.11 / acc 0.09 / emg
0.07 / scr 0.01; spectral HRV (LF/HF) ≈ 0.002 — a 60-sample 1 Hz periodogram
cannot resolve the 0.04–0.15 Hz LF band.

## iter3 — shape expansion, spectral HRV dropped (68 features)

Dropped the near-dead spectral features (hr_lf_power, hr_hf_power,
hr_lf_hf_ratio, eda_spectral_slope; resp_dominant_frequency kept), expanded
ranges to all six channels, added p10/p90 extremes for hr/eda/resp and
successive-diff spread for resp/acc_mag/emg.

- Windows: 1015 pure-label (1858 mixed, 1 dirty)
- random_forest pooled macro-F1: 0.6204 (baseline 0.8192 / stress 0.8341 / amusement 0.2080)
- lightgbm pooled macro-F1: 0.6223 (baseline 0.8014 / stress 0.8305 / amusement 0.2348)
- Stress-vs-rest F1: RF 0.8723 (thr 0.55) / LGBM 0.8365 (thr 0.56)

Both backends worse than iter2: dimension dilution on the fixed shallow
hyperparameters outweighs the removed dead weight. (A naming bug —
`acc_diff_std` produced as `acc_mag_diff_std`, filling a NaN column — was
caught by the finiteness guard and fixed before scoring.)

## iter4 — 54 base + retained helpers (59 features) — FROZEN

The original 54-feature literature-aligned set plus only the five helpers
informed by the iter2 diagnostic: hr_cv, eda_cv, temp_range, eda_range
(strongest addition, OOF importance 0.043), hr_diff_std. Spectral HRV is kept
for literature comparability despite its near-zero OOF importance (the
resolution limit is documented above).

- Windows: 1015 pure-label (1858 mixed, 1 dirty)
- random_forest pooled macro-F1: 0.6061 (baseline 0.8275 / stress 0.8410 / amusement 0.1500)
- lightgbm pooled macro-F1: 0.6428 (baseline 0.8139 / stress 0.8364 / amusement 0.2780)
- Stress-vs-rest F1: RF 0.8981 (thr 0.49) / LGBM 0.8840 (thr 0.74)

## Freeze decision

iter4 (59 features) is frozen for the deploy bundle and the paper table.
Rationale: best pooled OOF on the deploy backend (lightgbm 0.6428 vs 0.6327 /
0.6311 / 0.6223 for iter1/2/3) and the simplest defensible set beyond the
literature-aligned 54. Backend disagreement (RF prefers iter2 at 0.6320) and
the ±0.02 spread across iterations sit within fold-level noise — fold
macro-F1s range 0.21–1.00 — so the choice is documented, not claimed as
significant. The remaining gap to the 0.85–0.93 literature range is structural:
strict leave-one-subject-out generalization (vs within-subject random splits),
chest-only signals, 60 s windows, and amusement/baseline confusability
(moderate arousal, positive valence; chest arousal channels carry little
valence information; amusement F1 ≤ 0.28 in every iteration). Stress-vs-rest —
the deployed task scope — reaches 0.88–0.90 F1.
