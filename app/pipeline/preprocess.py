"""Strict synthetic_physiology_v1 CSV contract and deterministic window cleaning.

Input rows represent seconds, not raw ECG or EEG waveforms. Missing samples are
never synthesized across a timestamp gap or a window boundary.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

from app.core.errors import SkillError

SIGNAL_COLUMNS = ("heart_rate", "EDA", "EEG_feature", "activity", "sleep_quality")
OPTIONAL_COLUMNS = ("timestamp", "subject_id", "session_id")
INPUT_PROFILE = "synthetic_physiology_v1"
PROCESSING_CONFIG = {
    "version": "1.0",
    "sample_rate_hz": 1,
    "window_rows": 30,
    "stride_rows": 30,
    "max_gap_rows": 3,
    "max_window_missing_fraction": 0.15,
    "max_channel_missing_fraction": 0.20,
    "tail_policy": "discard_and_report",
    "imputation": "within_window_linear_interpolation_then_edge_fill",
}
MAX_ROWS = 108_000
MAX_CSV_BYTES = 32 * 1024 * 1024


def _fail(code: str, message: str) -> None:
    raise SkillError(code, message)


def parse_csv(content: bytes) -> pd.DataFrame:
    """Read one session. Targets and arbitrary extra columns are rejected online."""
    if not content:
        _fail("invalid_csv", "CSV must contain a header and data rows.")
    if len(content) > MAX_CSV_BYTES:
        _fail("file_too_large", "CSV exceeds the supported size limit.")
    try:
        decoded = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(decoded), strict=True)
        header = next(reader)
        if len(header) != len(set(header)):
            _fail("invalid_columns", "Duplicate column names are not supported.")
        allowed = set(SIGNAL_COLUMNS + OPTIONAL_COLUMNS)
        if not set(SIGNAL_COLUMNS).issubset(header) or not set(header).issubset(allowed):
            _fail("invalid_columns", "CSV must follow the synthetic_physiology_v1 schema.")
        rows = []
        for row in reader:
            if len(row) != len(header):
                _fail("invalid_csv", "CSV contains an inconsistent row width.")
            rows.append(row)
            if len(rows) > MAX_ROWS:
                _fail("file_too_large", "CSV exceeds the supported row limit.")
        if not rows:
            _fail("invalid_csv", "CSV must contain data rows.")
        frame = pd.DataFrame(rows, columns=header)
        for name in SIGNAL_COLUMNS:
            # Only empty fields mean missing; misspellings and strings are errors.
            frame[name] = pd.to_numeric(frame[name].replace(r"^\s*$", np.nan, regex=True))
    except (UnicodeDecodeError, csv.Error, StopIteration, ValueError, TypeError):
        _fail("invalid_csv", "CSV must be UTF-8 with valid numeric signal values.")
    return _validate_frame(frame)


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        _fail("invalid_data", "A nonempty signal table is required.")
    if not frame.columns.is_unique or not set(SIGNAL_COLUMNS).issubset(frame.columns):
        _fail("invalid_columns", "Required signal columns must be unique and present.")
    if not set(frame.columns).issubset(SIGNAL_COLUMNS + OPTIONAL_COLUMNS):
        _fail("invalid_columns", "Unsupported columns cannot enter signal processing.")
    if len(frame) > MAX_ROWS:
        _fail("file_too_large", "Signal table exceeds the supported row limit.")
    frame = frame.copy(deep=True).reset_index(drop=True)
    for name in SIGNAL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(
            frame[name]
        ):
            _fail("invalid_data", "Signal columns must contain numeric values.")
        frame[name] = frame[name].astype(float)
        values = frame[name].to_numpy()
        if np.isinf(values).any():
            _fail("invalid_data", "Infinite signal values are not supported.")
        valid = frame[name].dropna()
        if name == "heart_rate" and (valid <= 0).any():
            _fail("invalid_data", "Heart rate must be positive when present.")
        if name in ("EDA", "EEG_feature") and (valid < 0).any():
            _fail("invalid_data", "EDA and the synthetic EEG proxy must be nonnegative.")
        if name in ("activity", "sleep_quality") and ((valid < 0) | (valid > 1)).any():
            _fail("invalid_data", "Normalized context values must be within zero and one.")
    for name in ("subject_id", "session_id"):
        if name in frame and (
            frame[name].isna().any()
            or frame[name].astype(str).str.strip().eq("").any()
            or frame[name].nunique() != 1
        ):
            _fail("multiple_sessions", "Each upload must contain exactly one subject and session.")
    sleep = frame["sleep_quality"].dropna()
    if len(sleep) and float(sleep.max() - sleep.min()) > 1e-8:
        _fail("invalid_data", "Sleep quality must be constant within a session.")
    if "timestamp" in frame:
        if pd.api.types.is_numeric_dtype(frame["timestamp"]):
            _fail("invalid_timestamp", "Timestamps must be ISO-8601 datetime values.")
        try:
            stamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise", format="mixed")
        except (ValueError, TypeError, OverflowError):
            _fail("invalid_timestamp", "Timestamps must be valid datetime values.")
        if stamp.isna().any() or (
            len(stamp) > 1
            and not (
                np.diff(stamp.to_numpy(dtype="datetime64[ns]")) == np.timedelta64(1, "s")
            ).all()
        ):
            _fail(
                "invalid_timestamp", "Timestamps must be strictly consecutive one-second samples."
            )
        frame["timestamp"] = stamp
    return frame


def inspect_data(frame: pd.DataFrame) -> dict:
    frame = _validate_frame(frame)
    missing = frame[list(SIGNAL_COLUMNS)].isna()
    return {
        "input_profile": INPUT_PROFILE,
        "row_count": len(frame),
        "columns": list(frame.columns),
        "sample_rate_hz": 1,
        "timestamp_present": "timestamp" in frame,
        "missing_values": {column: int(missing[column].sum()) for column in SIGNAL_COLUMNS},
        "missing_fraction": float(missing.to_numpy().mean()),
        "warnings": (
            [] if "timestamp" in frame else ["Timestamps absent; row order is interpreted as 1 Hz."]
        ),
    }


def _longest_gap(series: pd.Series) -> int:
    missing = series.isna().to_numpy()
    best = current = 0
    for value in missing:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def preprocess(frame: pd.DataFrame) -> tuple[list[pd.DataFrame], dict]:
    frame = _validate_frame(frame)
    quality = inspect_data(frame)
    window_rows = PROCESSING_CONFIG["window_rows"]
    if len(frame) < window_rows:
        _fail("insufficient_data", "At least 30 consecutive one-second rows are required.")
    if any(
        _longest_gap(frame[name]) > PROCESSING_CONFIG["max_gap_rows"] for name in SIGNAL_COLUMNS
    ):
        _fail("poor_data_quality", "A signal contains a missing span longer than three seconds.")
    windows: list[pd.DataFrame] = []
    imputed = 0
    tail = len(frame) % window_rows
    for start in range(0, len(frame) - tail, window_rows):
        window = frame.iloc[start : start + window_rows].copy().reset_index(drop=True)
        signals = window[list(SIGNAL_COLUMNS)]
        if (
            float(signals.isna().to_numpy().mean())
            > PROCESSING_CONFIG["max_window_missing_fraction"]
            or (signals.isna().mean() > PROCESSING_CONFIG["max_channel_missing_fraction"]).any()
            or any(
                _longest_gap(signals[name]) > PROCESSING_CONFIG["max_gap_rows"]
                for name in SIGNAL_COLUMNS
            )
        ):
            _fail(
                "poor_data_quality",
                "A window has excessive missing data or an unsupported long gap.",
            )
        imputed += int(signals.isna().to_numpy().sum())
        window[list(SIGNAL_COLUMNS)] = signals.interpolate(method="linear", limit_direction="both")
        if window[list(SIGNAL_COLUMNS)].isna().any().any():
            _fail("poor_data_quality", "A window cannot be cleaned without crossing its boundary.")
        windows.append(window)
    quality.update(
        {
            "window_count": len(windows),
            "window_rows": window_rows,
            "processed_rows": len(windows) * window_rows,
            "dropped_tail_rows": tail,
            "imputed_values": imputed,
            "processing_version": PROCESSING_CONFIG["version"],
        }
    )
    if tail:
        quality["warnings"].append(f"Discarded {tail} trailing rows shorter than one window.")
    if imputed:
        quality["warnings"].append("Short missing spans were filled within their own windows.")
    return windows, quality
