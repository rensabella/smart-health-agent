"""Window statistics; no claims to waveform-derived HRV or EEG band power."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smarthealth.pipeline.preprocess import SIGNAL_COLUMNS
from smarthealth.runtime.errors import SkillError

FEATURE_SCHEMA_VERSION = "1.0"
STATISTICS = ("mean", "std", "min", "max", "p25", "p75", "slope", "mean_abs_change")
FEATURE_NAMES = tuple(
    f"{column}_{stat}" for column in SIGNAL_COLUMNS[:-1] for stat in STATISTICS
) + (
    "sleep_quality_mean",
    "heart_rate_activity_correlation",
    "heart_rate_EDA_correlation",
)
FEATURE_CONFIG = {
    "version": FEATURE_SCHEMA_VERSION,
    "statistics": list(STATISTICS),
    "std_ddof": 0,
    "slope_unit": "signal_unit_per_second",
}


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def extract_features(windows: list[pd.DataFrame]) -> pd.DataFrame:
    if not windows:
        raise SkillError("insufficient_data", "At least one complete window is required.")
    records = []
    for window in windows:
        if (
            len(window) != 30
            or not set(SIGNAL_COLUMNS).issubset(window.columns)
            or not np.isfinite(window[list(SIGNAL_COLUMNS)].to_numpy(dtype=float)).all()
        ):
            raise SkillError(
                "invalid_features", "Feature extraction requires complete, finite 30-row windows."
            )
        record = {}
        centered_time = np.arange(len(window), dtype=float) - (len(window) - 1) / 2
        for name in SIGNAL_COLUMNS[:-1]:
            values = window[name].to_numpy(dtype=float)
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
                {f"{name}_{stat}": float(value) for stat, value in zip(STATISTICS, stats)}
            )
        record["sleep_quality_mean"] = float(window["sleep_quality"].mean())
        record["heart_rate_activity_correlation"] = _correlation(
            window["heart_rate"].to_numpy(), window["activity"].to_numpy()
        )
        record["heart_rate_EDA_correlation"] = _correlation(
            window["heart_rate"].to_numpy(), window["EDA"].to_numpy()
        )
        records.append(record)
    result = pd.DataFrame.from_records(records, columns=FEATURE_NAMES)
    if not np.isfinite(result.to_numpy()).all():
        raise SkillError("invalid_features", "Extracted features must all be finite.")
    return result
