import numpy as np
import pytest

from app.core.errors import SkillError
from app.pipeline.feature_engineering import FEATURE_NAMES, extract_features
from app.pipeline.preprocess import parse_csv, preprocess


def test_cleaning_preserves_anomalies_and_reports_tail(csv_bytes):
    frame = parse_csv(csv_bytes).iloc[:95].copy()
    frame.loc[:1, "heart_rate"] = np.nan
    frame.loc[10, "heart_rate"] = 250.0
    windows, report = preprocess(frame)
    assert len(windows) == 3 and report["dropped_tail_rows"] == 5
    assert report["imputed_values"] >= 2
    assert windows[0].loc[10, "heart_rate"] == 250
    features = extract_features(windows)
    assert tuple(features.columns) == FEATURE_NAMES
    assert np.isfinite(features.to_numpy()).all()
    assert "sleep_quality_std" not in features


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("heart_rate", "wrong_column", 1),
        lambda s: s.replace("heart_rate", "EDA", 1),
        lambda s: s + "malformed,row\n",
    ],
)
def test_bad_csv_rejected(csv_bytes, mutate):
    with pytest.raises(SkillError):
        parse_csv(mutate(csv_bytes.decode()).encode())


def test_long_gaps_timestamps_and_sessions(csv_bytes):
    frame = parse_csv(csv_bytes)
    bad = frame.copy()
    bad.loc[10:14, "EDA"] = np.nan
    with pytest.raises(SkillError):
        preprocess(bad)
    bad = frame.copy()
    bad.loc[28:32, "EDA"] = np.nan
    with pytest.raises(SkillError):
        preprocess(bad)
    bad = frame.copy()
    bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
    with pytest.raises(SkillError):
        preprocess(bad)
    bad = frame.copy()
    bad.loc[1, "subject_id"] = "other"
    with pytest.raises(SkillError):
        preprocess(bad)
