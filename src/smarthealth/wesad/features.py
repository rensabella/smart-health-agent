"""Window features for the WESAD chest-signal canonical table (wesad_physiology_v1).

Inputs are cleaned, finite 60-row windows at 1 Hz produced by the shared
preprocessing contract with stride 30. The extractor is a pure function of the
window values; no randomness, no wall-clock dependence, no clinical claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from smarthealth.runtime.errors import SkillError

WESAD_SAMPLE_RATE_HZ = 1
WESAD_WINDOW_ROWS = 60
WESAD_STRIDE_ROWS = 30
WESAD_SIGNAL_COLUMNS = ("hr", "eda", "emg", "temp", "resp", "acc_mag", "scr")
WESAD_STAT_CHANNELS = WESAD_SIGNAL_COLUMNS[:6]
WESAD_FEATURE_SCHEMA_VERSION = "1.0"
WESAD_STATISTICS = ("mean", "std", "min", "max", "p25", "p75", "slope", "mean_abs_change")
WESAD_FEATURE_NAMES = tuple(
    f"{channel}_{stat}" for channel in WESAD_STAT_CHANNELS for stat in WESAD_STATISTICS
) + (
    "hr_lf_power",
    "hr_hf_power",
    "hr_lf_hf_ratio",
    "resp_dominant_frequency",
    "eda_spectral_slope",
    "scr_count",
    "hr_cv",
    "eda_cv",
    "temp_range",
    "eda_range",
    "hr_diff_std",
)
WESAD_FEATURE_CONFIG = {
    "version": WESAD_FEATURE_SCHEMA_VERSION,
    "statistics": list(WESAD_STATISTICS),
    "std_ddof": 0,
    "slope_unit": "signal_unit_per_second",
    "spectral_method": "scipy_periodogram_linear_detrend",
    "hr_lf_band_hz": [0.04, 0.15],
    "hr_hf_band_hz": [0.15, 0.4],
    "resp_band_hz": [0.1, 0.5],
    "zero_power_ratio_fallback": 0.0,
    "cv_channels": ["hr", "eda"],
    "range_channels": ["eda", "temp"],
    "successive_diff_std_channels": ["hr"],
}


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    band = (freqs >= low) & (freqs <= high)
    return float(power[band].sum()) if band.any() else 0.0


def _dominant_frequency(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return 0.0
    return float(freqs[band][int(np.argmax(power[band]))])


def _spectral_slope(freqs: np.ndarray, power: np.ndarray) -> float:
    positive = freqs > 0
    if not positive.any():
        return 0.0
    log_f = np.log(freqs[positive])
    log_p = np.log(power[positive] + 1e-12)
    center_f = log_f.mean()
    center_p = log_p.mean()
    denominator = float(((log_f - center_f) ** 2).sum())
    if denominator <= 0.0:
        return 0.0
    return float(((log_f - center_f) * (log_p - center_p)).sum() / denominator)


def _window_features(window: pd.DataFrame) -> dict:
    record: dict[str, float] = {}
    centered_time = np.arange(len(window), dtype=float) - (len(window) - 1) / 2
    series = {name: window[name].to_numpy(dtype=float) for name in WESAD_STAT_CHANNELS}
    for name, values in series.items():
        stats = (
            values.mean(),
            values.std(ddof=0),
            values.min(),
            values.max(),
            np.quantile(values, 0.25),
            np.quantile(values, 0.75),
            np.dot(centered_time, values) / np.dot(centered_time, centered_time),
            np.abs(np.diff(values)).mean(),
        )
        record.update(
            {f"{name}_{stat}": float(value) for stat, value in zip(WESAD_STATISTICS, stats)}
        )
    # Cross-subject helpers: dispersion ratios remove each person's own scale,
    # and ranges and successive-difference spread expose burst shape that tree
    # splits cannot reconstruct from the raw statistics above.
    for name in ("hr", "eda"):
        mean = float(series[name].mean())
        record[f"{name}_cv"] = float(series[name].std(ddof=0) / mean) if mean > 0.0 else 0.0
    record["temp_range"] = float(series["temp"].max() - series["temp"].min())
    record["eda_range"] = float(series["eda"].max() - series["eda"].min())
    record["hr_diff_std"] = float(np.diff(series["hr"]).std(ddof=0))
    hr_freqs, hr_power = periodogram(series["hr"], fs=WESAD_SAMPLE_RATE_HZ, detrend="linear")
    lf = _band_power(hr_freqs, hr_power, 0.04, 0.15)
    hf = _band_power(hr_freqs, hr_power, 0.15, 0.4)
    record["hr_lf_power"] = lf
    record["hr_hf_power"] = hf
    record["hr_lf_hf_ratio"] = lf / hf if hf > 0.0 else 0.0
    resp_freqs, resp_power = periodogram(series["resp"], fs=WESAD_SAMPLE_RATE_HZ, detrend="linear")
    record["resp_dominant_frequency"] = _dominant_frequency(resp_freqs, resp_power, 0.1, 0.5)
    eda_freqs, eda_power = periodogram(series["eda"], fs=WESAD_SAMPLE_RATE_HZ, detrend="linear")
    record["eda_spectral_slope"] = _spectral_slope(eda_freqs, eda_power)
    record["scr_count"] = float(window["scr"].sum())
    return record


def extract_wesad_features(windows: list[pd.DataFrame]) -> pd.DataFrame:
    if not windows:
        raise SkillError("insufficient_data", "At least one complete window is required.")
    records = []
    for window in windows:
        if (
            len(window) != WESAD_WINDOW_ROWS
            or not set(WESAD_SIGNAL_COLUMNS).issubset(window.columns)
            or not np.isfinite(window[list(WESAD_SIGNAL_COLUMNS)].to_numpy(dtype=float)).all()
        ):
            raise SkillError(
                "invalid_features",
                f"Feature extraction requires complete, finite {WESAD_WINDOW_ROWS}-row windows.",
            )
        records.append(_window_features(window))
    result = pd.DataFrame.from_records(records, columns=WESAD_FEATURE_NAMES)
    if not np.isfinite(result.to_numpy()).all():
        raise SkillError("invalid_features", "Extracted features must all be finite.")
    return result
