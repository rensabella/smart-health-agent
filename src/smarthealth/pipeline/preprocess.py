"""Strict per-profile CSV contracts and deterministic window cleaning.

Input rows represent seconds, not raw ECG or EEG waveforms. Missing samples are
never synthesized across a timestamp gap or a window boundary. The default
profile keeps the historical synthetic_physiology_v1 contract byte-for-byte;
registered profiles (see ``pipeline/profiles.py``) add other data origins.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

from smarthealth.runtime.errors import SkillError

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


def _default_profile():
    from smarthealth.pipeline.profiles import SYNTHETIC_V1

    return SYNTHETIC_V1


def parse_csv(content: bytes, *, profile=None) -> pd.DataFrame:
    """Read one session. Targets and arbitrary extra columns are rejected online."""
    if profile is None:
        profile = _default_profile()
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
        allowed = set(profile.signal_columns + profile.optional_columns)
        if not set(profile.signal_columns).issubset(header) or not set(header).issubset(allowed):
            from smarthealth.pipeline.profiles import suggest_profile

            message = f"CSV must follow the {profile.name} schema."
            suggestion = suggest_profile(header, exclude=profile.name)
            if suggestion:
                message += (
                    f" Column layout matches registered profile '{suggestion}';"
                    " retry with that value."
                )
            _fail("invalid_columns", message)
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
        for name in profile.signal_columns:
            # Only empty fields mean missing; misspellings and strings are errors.
            frame[name] = pd.to_numeric(frame[name].replace(r"^\s*$", np.nan, regex=True))
    except (UnicodeDecodeError, csv.Error, StopIteration, ValueError, TypeError):
        _fail("invalid_csv", "CSV must be UTF-8 with valid numeric signal values.")
    return _validate_frame(frame, profile=profile)


def _validate_frame(frame: pd.DataFrame, *, profile=None) -> pd.DataFrame:
    if profile is None:
        profile = _default_profile()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        _fail("invalid_data", "A nonempty signal table is required.")
    if not frame.columns.is_unique or not set(profile.signal_columns).issubset(frame.columns):
        _fail("invalid_columns", "Required signal columns must be unique and present.")
    if not set(frame.columns).issubset(profile.signal_columns + profile.optional_columns):
        _fail("invalid_columns", "Unsupported columns cannot enter signal processing.")
    if len(frame) > MAX_ROWS:
        _fail("file_too_large", "Signal table exceeds the supported row limit.")
    frame = frame.copy(deep=True).reset_index(drop=True)
    rules = {column: rule for rule in profile.value_rules for column in rule.columns}
    for name in profile.signal_columns:
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(
            frame[name]
        ):
            _fail("invalid_data", "Signal columns must contain numeric values.")
        frame[name] = frame[name].astype(float)
        values = frame[name].to_numpy()
        if np.isinf(values).any():
            _fail("invalid_data", "Infinite signal values are not supported.")
        valid = frame[name].dropna()
        rule = rules.get(name)
        if rule is not None and rule.violated(valid.to_numpy(dtype=float)):
            _fail("invalid_data", rule.message)
    for name in ("subject_id", "session_id"):
        if name in frame and (
            frame[name].isna().any()
            or frame[name].astype(str).str.strip().eq("").any()
            or frame[name].nunique() != 1
        ):
            _fail("multiple_sessions", "Each upload must contain exactly one subject and session.")
    for rule in profile.constant_columns:
        column = frame[rule.column].dropna()
        if len(column) and float(column.max() - column.min()) > 1e-8:
            _fail("invalid_data", rule.message)
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


def inspect_data(frame: pd.DataFrame, *, profile=None) -> dict:
    if profile is None:
        profile = _default_profile()
    frame = _validate_frame(frame, profile=profile)
    missing = frame[list(profile.signal_columns)].isna()
    return {
        "input_profile": profile.name,
        "row_count": len(frame),
        "columns": list(frame.columns),
        "sample_rate_hz": profile.processing_config["sample_rate_hz"],
        "timestamp_present": "timestamp" in frame,
        "missing_values": {column: int(missing[column].sum()) for column in profile.signal_columns},
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


def _collect_windows(
    frame: pd.DataFrame, *, profile, fail_closed: bool
) -> tuple[list[pd.DataFrame], int, int, int]:
    """Slice and clean windows by the shared rule; rejects raise or are counted.

    The acceptance rule, interpolation and window arithmetic exist exactly once
    here. ``preprocess`` fails closed on the first rejected window (the runtime
    upload contract); the training-side ``clean_windows`` collector instead drops
    rejects and counts them so quality filtering keeps an honest denominator.
    """
    config = profile.processing_config
    window_rows = config["window_rows"]
    stride_rows = config["stride_rows"]
    windows: list[pd.DataFrame] = []
    imputed = 0
    rejected = 0
    last_start = ((len(frame) - window_rows) // stride_rows) * stride_rows
    tail = len(frame) - (last_start + window_rows)
    for start in range(0, last_start + 1, stride_rows):
        window = frame.iloc[start : start + window_rows].copy().reset_index(drop=True)
        signals = window[list(profile.signal_columns)]
        if (
            float(signals.isna().to_numpy().mean()) > config["max_window_missing_fraction"]
            or (signals.isna().mean() > config["max_channel_missing_fraction"]).any()
            or any(
                _longest_gap(signals[name]) > config["max_gap_rows"]
                for name in profile.signal_columns
            )
        ):
            if fail_closed:
                _fail(
                    "poor_data_quality",
                    "A window has excessive missing data or an unsupported long gap.",
                )
            rejected += 1
            continue
        imputed += int(signals.isna().to_numpy().sum())
        cleaned = signals.interpolate(method="linear", limit_direction="both")
        if cleaned.isna().any().any():
            if fail_closed:
                _fail(
                    "poor_data_quality",
                    "A window cannot be cleaned without crossing its boundary.",
                )
            rejected += 1
            continue
        window[list(profile.signal_columns)] = cleaned
        windows.append(window)
    return windows, imputed, rejected, tail


def preprocess(frame: pd.DataFrame, *, profile=None) -> tuple[list[pd.DataFrame], dict]:
    if profile is None:
        profile = _default_profile()
    frame = _validate_frame(frame, profile=profile)
    quality = inspect_data(frame, profile=profile)
    config = profile.processing_config
    window_rows = config["window_rows"]
    if len(frame) < window_rows:
        _fail(
            "insufficient_data",
            f"At least {window_rows} consecutive one-second rows are required.",
        )
    if any(_longest_gap(frame[name]) > config["max_gap_rows"] for name in profile.signal_columns):
        _fail("poor_data_quality", "A signal contains a missing span longer than three seconds.")
    windows, imputed, _rejected, tail = _collect_windows(frame, profile=profile, fail_closed=True)
    quality.update(
        {
            "window_count": len(windows),
            "window_rows": window_rows,
            "processed_rows": len(windows) * window_rows,
            "dropped_tail_rows": tail,
            "imputed_values": imputed,
            "processing_version": config["version"],
        }
    )
    if tail:
        quality["warnings"].append(f"Discarded {tail} trailing rows shorter than one window.")
    if imputed:
        quality["warnings"].append("Short missing spans were filled within their own windows.")
    return windows, quality


def clean_windows(frame: pd.DataFrame, *, profile=None) -> tuple[list[pd.DataFrame], dict]:
    """Every window that passes the shared cleaning rule, with rejects counted.

    Training-side quality filtering over one clean segment: this never raises on
    a rejected window, it drops and counts it. The rule itself is the shared
    production contract above, so training and inference accept exactly the same
    windows; only the failure policy differs (see ``_collect_windows``).
    """
    if profile is None:
        profile = _default_profile()
    frame = _validate_frame(frame, profile=profile)
    config = profile.processing_config
    window_rows = config["window_rows"]
    if len(frame) < window_rows:
        _fail(
            "insufficient_data",
            f"At least {window_rows} consecutive one-second rows are required.",
        )
    windows, imputed, rejected, tail = _collect_windows(frame, profile=profile, fail_closed=False)
    return windows, {
        "window_rows": window_rows,
        "accepted_windows": len(windows),
        "rejected_windows": rejected,
        "dropped_tail_rows": tail,
        "imputed_values": imputed,
        "processing_version": config["version"],
    }
