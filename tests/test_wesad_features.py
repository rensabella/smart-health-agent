"""Known-value feature checks and training-side window construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from wesad_helpers import canonical_frame

from smarthealth.pipeline.profiles import WESAD_V1
from smarthealth.runtime.errors import SkillError
from smarthealth.wesad.build import clean_segments, corpus_dataset, subject_dataset
from smarthealth.wesad.features import (
    WESAD_FEATURE_NAMES,
    WESAD_SIGNAL_COLUMNS,
    extract_wesad_features,
)


def signal_window(**overrides) -> pd.DataFrame:
    base = {name: np.full(60, 5.0) for name in WESAD_SIGNAL_COLUMNS}
    base.update(overrides)
    return pd.DataFrame(base)


def test_feature_contract_covers_all_channels():
    assert len(WESAD_FEATURE_NAMES) == 59
    assert list(WESAD_FEATURE_NAMES) == list(WESAD_V1.feature_names)
    result = extract_wesad_features([signal_window()])
    assert list(result.columns) == list(WESAD_FEATURE_NAMES)


def test_constant_window_statistics():
    result = extract_wesad_features([signal_window()])
    row = result.iloc[0]
    for channel in ("hr", "eda", "emg", "temp", "resp", "acc_mag"):
        assert row[f"{channel}_mean"] == pytest.approx(5.0)
        assert row[f"{channel}_std"] == pytest.approx(0.0)
        assert row[f"{channel}_min"] == pytest.approx(5.0)
        assert row[f"{channel}_max"] == pytest.approx(5.0)
        assert row[f"{channel}_p25"] == pytest.approx(5.0)
        assert row[f"{channel}_p75"] == pytest.approx(5.0)
        assert row[f"{channel}_slope"] == pytest.approx(0.0)
        assert row[f"{channel}_mean_abs_change"] == pytest.approx(0.0)
    # scr_count is the plain sum of the indicator column.
    assert row["scr_count"] == pytest.approx(300.0)
    # Constant windows make every shape helper degenerate but finite.
    assert row["hr_cv"] == pytest.approx(0.0)
    assert row["eda_cv"] == pytest.approx(0.0)
    assert row["temp_range"] == pytest.approx(0.0)
    assert row["eda_range"] == pytest.approx(0.0)
    assert row["hr_diff_std"] == pytest.approx(0.0)


def test_ramp_slope_and_mean_abs_change():
    ramp = np.arange(60, dtype=float)
    result = extract_wesad_features([signal_window(hr=ramp)])
    assert result["hr_slope"].iloc[0] == pytest.approx(1.0)
    assert result["hr_mean_abs_change"].iloc[0] == pytest.approx(1.0)


def test_breathing_rate_on_exact_bin():
    t = np.arange(60, dtype=float)
    window = signal_window(resp=0.5 + 0.2 * np.sin(2 * np.pi * 0.2 * t))
    result = extract_wesad_features([window])
    assert result["resp_dominant_frequency"].iloc[0] == pytest.approx(0.2, abs=1e-9)


def test_range_and_cv_helpers_expose_burst_shape():
    t = np.arange(60, dtype=float)
    bursts = np.full(60, 4.0)
    bursts[20:23] = 12.0
    window = signal_window(hr=70.0 + 2.0 * np.sin(2 * np.pi * 0.1 * t), eda=bursts)
    result = extract_wesad_features([window])
    row = result.iloc[0]
    assert row["eda_range"] == pytest.approx(8.0)
    assert row["eda_cv"] > 0.0
    assert row["hr_cv"] > 0.0


def test_scr_count_sums_the_indicator_column():
    scr = np.zeros(60)
    scr[[3, 9, 20, 21, 55]] = 1.0
    result = extract_wesad_features([signal_window(scr=scr)])
    assert result["scr_count"].iloc[0] == pytest.approx(5.0)


def test_extract_rejects_incomplete_windows():
    with pytest.raises(SkillError):
        extract_wesad_features([])
    with pytest.raises(SkillError):
        extract_wesad_features([signal_window().iloc[:59]])
    with pytest.raises(SkillError):
        extract_wesad_features([signal_window(hr=np.full(60, np.nan))])
    missing = signal_window().drop(columns="temp")
    with pytest.raises(SkillError):
        extract_wesad_features([missing])


def test_clean_segments_splits_only_on_long_gaps():
    frame = canonical_frame("S2", [(200, 0)])
    frame.loc[90:92, "hr"] = np.nan
    assert clean_segments(frame) == [(0, 200)]
    frame.loc[90:93, "hr"] = np.nan
    assert clean_segments(frame) == [(0, 90), (94, 200)]
    short_tail = canonical_frame("S2", [(70, 0)])
    short_tail.loc[65:70, "hr"] = np.nan
    assert clean_segments(short_tail) == [(0, 65)]


def test_subject_dataset_keeps_pure_label_windows_only():
    frame = canonical_frame("S2", [(130, 0), (130, 1), (130, 2)])
    dataset = subject_dataset(frame)
    assert dataset["window_count"] == 8
    assert dataset["mixed_label_windows"] == 4
    assert dataset["dirty_windows"] == 0
    assert dataset["dropped_segments"] == 0
    assert dataset["labels"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert dataset["features"].shape == (8, 59)
    assert list(dataset["features"].columns) == list(WESAD_FEATURE_NAMES)


def test_subject_dataset_drops_and_counts_one_dirty_window():
    """One window over the per-channel missing cap forfeits only itself."""
    frame = canonical_frame("S2", [(390, 0)])
    # 13 one-second gaps inside the window starting at row 60 (13/60 > 20%),
    # each gap short enough that clean_segments keeps the frame whole.
    frame.loc[60:96:3, "hr"] = np.nan
    dataset = subject_dataset(frame)
    assert dataset["dropped_segments"] == 0
    assert dataset["dirty_windows"] == 1
    assert dataset["window_count"] == 11
    assert dataset["labels"].tolist() == [0] * 11


def test_subject_dataset_drops_unlabeled_windows():
    frame = canonical_frame("S2", [(70, 0), (70, None), (70, 1)])
    dataset = subject_dataset(frame)
    assert dataset["window_count"] == 2
    assert dataset["mixed_label_windows"] == 4
    assert dataset["labels"].tolist() == [0, 1]


def test_corpus_dataset_groups_by_subject():
    frames = [
        canonical_frame("S3", [(130, 0), (130, 1), (130, 2)], seed=1),
        canonical_frame("S2", [(130, 0), (130, 1), (130, 2)], seed=2),
    ]
    corpus = corpus_dataset(pd.concat(frames, ignore_index=True))
    assert sorted(corpus["subjects"]) == ["S2", "S3"]
    assert len(corpus["features"]) == 16
    assert len(corpus["labels"]) == 16
    assert corpus["mixed_label_windows"] == 8
    assert corpus["dirty_windows"] == 0
    assert corpus["subjects"]["S2"]["window_count"] == 8
    assert corpus["subjects"]["S3"]["window_count"] == 8
