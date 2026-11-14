"""Data-profile registry contract: synthetic aliases stay byte-equal to module constants."""

import numpy as np
import pandas as pd
import pytest

from smarthealth.pipeline.feature_engineering import (
    FEATURE_CONFIG,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_features,
    extract_synthetic_features,
)
from smarthealth.pipeline.preprocess import (
    INPUT_PROFILE,
    OPTIONAL_COLUMNS,
    PROCESSING_CONFIG,
    SIGNAL_COLUMNS,
    parse_csv,
    preprocess,
)
from smarthealth.pipeline.profiles import (
    SYNTHETIC_V1,
    WESAD_V1,
    get_profile,
    suggest_profile,
)
from smarthealth.runtime.errors import SkillError
from smarthealth.wesad.features import (
    WESAD_FEATURE_NAMES,
    extract_wesad_features,
)

SYNTHETIC_HEADER = list(SIGNAL_COLUMNS) + ["timestamp", "subject_id", "session_id"]
WESAD_HEADER = list(WESAD_V1.signal_columns)


def synthetic_frame(length: int) -> pd.DataFrame:
    rows = np.arange(length, dtype=float)
    return pd.DataFrame(
        {
            "heart_rate": 70.0 + 5.0 * np.sin(rows / 7.0),
            "EDA": 0.8 + 0.1 * np.cos(rows / 11.0),
            "EEG_feature": 10.0 + np.sin(rows / 5.0),
            "activity": np.clip(0.4 + 0.1 * np.sin(rows / 9.0), 0.0, 1.0),
            "sleep_quality": np.full(length, 0.6),
        }
    )


def wesad_frame(length: int) -> pd.DataFrame:
    rows = np.arange(length, dtype=float)
    return pd.DataFrame(
        {
            "hr": 70.0 + 6.0 * np.sin(rows / 9.0),
            "eda": 4.0 + 0.5 * np.sin(rows / 13.0),
            "emg": 0.02 + 0.005 * np.sin(rows / 3.0),
            "temp": 36.5 + 0.1 * np.sin(rows / 60.0),
            "resp": 0.5 + 0.2 * np.sin(2 * np.pi * rows / 20.0),
            "acc_mag": 1.0 + 0.05 * np.sin(rows / 4.0),
            "scr": (rows % 17 == 0).astype(float),
        }
    )


def test_synthetic_profile_aliases_module_constants():
    assert SYNTHETIC_V1.name == INPUT_PROFILE
    assert SYNTHETIC_V1.signal_columns == SIGNAL_COLUMNS
    assert SYNTHETIC_V1.optional_columns == OPTIONAL_COLUMNS
    assert SYNTHETIC_V1.processing_config == PROCESSING_CONFIG
    assert SYNTHETIC_V1.feature_names == FEATURE_NAMES
    assert SYNTHETIC_V1.feature_config == FEATURE_CONFIG
    assert SYNTHETIC_V1.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert SYNTHETIC_V1.extractor is extract_synthetic_features
    assert SYNTHETIC_V1.label_mapping == {"0": "normal", "1": "anomaly"}
    assert SYNTHETIC_V1.expected_classes == (0, 1)
    assert SYNTHETIC_V1.risk_class == "anomaly"
    assert SYNTHETIC_V1.risk_class_index() == 1
    assert SYNTHETIC_V1.score_semantics == "synthetic_anomaly_model_score"
    assert SYNTHETIC_V1.task_scope == "synthetic_anomaly_only"
    assert SYNTHETIC_V1.window_rows == 30
    assert SYNTHETIC_V1.stride_rows == 30
    assert SYNTHETIC_V1.window_seconds == 30
    assert SYNTHETIC_V1.allowed_columns == SIGNAL_COLUMNS + OPTIONAL_COLUMNS


def test_synthetic_guard_text_matches_harness_constants():
    from smarthealth.runtime.harness.harness_validator import (
        HIGH_SCORE_RECOMMENDATION,
        UNCERTAINTY,
    )

    assert SYNTHETIC_V1.guard_text.uncertainty == UNCERTAINTY
    assert SYNTHETIC_V1.guard_text.high_score_recommendation == HIGH_SCORE_RECOMMENDATION


def test_wesad_profile_contract():
    assert WESAD_V1.name == "wesad_physiology_v1"
    assert WESAD_V1.signal_columns == ("hr", "eda", "emg", "temp", "resp", "acc_mag", "scr")
    assert WESAD_V1.window_rows == 60 and WESAD_V1.stride_rows == 30
    assert WESAD_V1.window_seconds == 60
    assert WESAD_V1.label_mapping == {"0": "baseline", "1": "stress", "2": "amusement"}
    assert WESAD_V1.expected_classes == (0, 1, 2)
    assert WESAD_V1.risk_class == "stress" and WESAD_V1.risk_class_index() == 1
    assert WESAD_V1.score_semantics == "wesad_stress_model_score"
    assert WESAD_V1.task_scope == "wesad_stress_only"
    assert WESAD_V1.feature_names == WESAD_FEATURE_NAMES
    assert len(WESAD_V1.feature_names) == len(set(WESAD_V1.feature_names)) == 59
    assert WESAD_V1.extractor is extract_wesad_features
    assert "scipy" in WESAD_V1.dependency_packages
    assert "scipy" not in SYNTHETIC_V1.dependency_packages


@pytest.mark.parametrize("name", ["synthetic_physiology_v1", "wesad_physiology_v1"])
def test_get_profile_resolves_registered_names(name):
    assert get_profile(name).name == name


@pytest.mark.parametrize("name", ["", "unknown_profile", "synthetic", None])
def test_get_profile_rejects_unknown_names(name):
    with pytest.raises(ValueError):
        get_profile(name)


def test_suggest_profile_recognises_registered_layouts():
    assert suggest_profile(WESAD_HEADER, exclude=SYNTHETIC_V1.name) == WESAD_V1.name
    assert suggest_profile(WESAD_HEADER + ["timestamp"], exclude=SYNTHETIC_V1.name) == WESAD_V1.name
    assert suggest_profile(SYNTHETIC_HEADER, exclude=WESAD_V1.name) == SYNTHETIC_V1.name
    assert suggest_profile(SYNTHETIC_HEADER, exclude=SYNTHETIC_V1.name) is None
    assert suggest_profile(["a", "b"], exclude=SYNTHETIC_V1.name) is None
    assert suggest_profile(WESAD_HEADER + ["label"], exclude=SYNTHETIC_V1.name) is None


def test_parse_csv_suggests_other_registered_profile():
    frame = wesad_frame(5)
    content = frame.to_csv(index=False).encode()
    with pytest.raises(SkillError) as exc:
        parse_csv(content)
    assert exc.value.code == "invalid_columns"
    assert "CSV must follow the synthetic_physiology_v1 schema." in exc.value.message
    assert "Column layout matches registered profile 'wesad_physiology_v1'" in exc.value.message


def test_parse_csv_suggests_synthetic_profile_from_wesad_side():
    frame = synthetic_frame(5)
    content = frame.to_csv(index=False).encode()
    with pytest.raises(SkillError) as exc:
        parse_csv(content, profile=WESAD_V1)
    assert exc.value.code == "invalid_columns"
    assert "Column layout matches registered profile 'synthetic_physiology_v1'" in exc.value.message


@pytest.mark.parametrize("length", [30, 31, 59, 60, 61, 90, 95, 119, 120, 121])
def test_window_partition_matches_legacy_nonoverlapping_loop(length):
    frame = synthetic_frame(length)
    windows, report = preprocess(frame)
    tail = length % 30
    starts = list(range(0, length - tail, 30))
    assert report["dropped_tail_rows"] == tail
    assert report["window_count"] == len(windows) == len(starts)
    assert report["processed_rows"] == len(windows) * 30
    for window, start in zip(windows, starts):
        assert len(window) == 30
        np.testing.assert_array_equal(
            window["heart_rate"].to_numpy(),
            frame["heart_rate"].iloc[start : start + 30].to_numpy(),
        )


@pytest.mark.parametrize("length", [60, 61, 89, 90, 91, 150])
def test_wesad_window_partition_uses_stride(length):
    frame = wesad_frame(length)
    windows, report = preprocess(frame, profile=WESAD_V1)
    starts = list(range(0, length - 59, 30))
    assert report["window_count"] == len(windows) == len(starts)
    assert report["dropped_tail_rows"] == length - (starts[-1] + 60)
    for window, start in zip(windows, starts):
        assert len(window) == 60
        np.testing.assert_array_equal(
            window["hr"].to_numpy(), frame["hr"].iloc[start : start + 60].to_numpy()
        )


def test_wesad_inspect_and_features_end_to_end():
    frame = wesad_frame(130)
    quality = preprocess(frame, profile=WESAD_V1)[1]
    assert quality["input_profile"] == "wesad_physiology_v1"
    assert quality["sample_rate_hz"] == 1
    windows = preprocess(frame, profile=WESAD_V1)[0]
    features = extract_features(windows, profile=WESAD_V1)
    assert tuple(features.columns) == WESAD_FEATURE_NAMES
    assert len(features) == len(windows)
    assert np.isfinite(features.to_numpy()).all()
    assert features["scr_count"].iloc[0] == float(frame["scr"].iloc[:60].sum())
    assert 0.0 < features["resp_dominant_frequency"].iloc[0] <= 0.5


def test_wesad_value_rules_reject_bad_rows():
    frame = wesad_frame(70)
    frame.loc[3, "hr"] = -2.0
    with pytest.raises(SkillError) as exc:
        preprocess(frame, profile=WESAD_V1)
    assert exc.value.code == "invalid_data"
    assert "Heart rate must be positive when present." in exc.value.message
    frame.loc[3, "hr"] = 70.0
    frame.loc[5, "eda"] = -0.1
    with pytest.raises(SkillError):
        preprocess(frame, profile=WESAD_V1)


def test_wesad_insufficient_rows_message_uses_profile_window():
    with pytest.raises(SkillError) as exc:
        preprocess(wesad_frame(59), profile=WESAD_V1)
    assert exc.value.code == "insufficient_data"
    assert "At least 60 consecutive one-second rows are required." in exc.value.message
