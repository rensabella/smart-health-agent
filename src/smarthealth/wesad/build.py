"""Training-side window construction for the WESAD canonical table.

Sessions are split into clean segments where no signal channel has a missing
span longer than the profile gap limit, each segment is windowed by the shared
production rule (identical contract to inference), and only pure-label windows
enter training. Windows that fail the shared cleaning rule, and mixed or
unlabeled windows, are dropped and counted so denominators stay honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smarthealth.pipeline.preprocess import clean_windows
from smarthealth.pipeline.profiles import WESAD_V1
from smarthealth.runtime.errors import SkillError
from smarthealth.wesad.features import extract_wesad_features


def _bad_rows(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Rows inside missing spans longer than the profile gap limit."""
    isnan = np.isnan(values)
    if not isnan.any():
        return np.zeros(len(values), dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([False], isnan, [False])).astype(int)))
    starts, ends = edges[0::2], edges[1::2]
    bad = np.zeros(len(values), dtype=bool)
    for start, end in zip(starts, ends):
        if end - start > max_gap:
            bad[start:end] = True
    return bad


def clean_segments(frame: pd.DataFrame, *, profile=None) -> list[tuple[int, int]]:
    """Maximal row spans that satisfy the profile gap rule in every signal column."""
    if profile is None:
        profile = WESAD_V1
    max_gap = profile.processing_config["max_gap_rows"]
    bad = np.zeros(len(frame), dtype=bool)
    for name in profile.signal_columns:
        bad |= _bad_rows(frame[name].to_numpy(dtype=float), max_gap)
    segments = []
    start = None
    for index, is_bad in enumerate(bad):
        if not is_bad and start is None:
            start = index
        elif is_bad and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(bad)))
    window_rows = profile.window_rows
    return [span for span in segments if span[1] - span[0] >= window_rows]


def subject_dataset(frame: pd.DataFrame, *, profile=None) -> dict:
    """Features and pure-label targets for one subject, with honest denominators."""
    if profile is None:
        profile = WESAD_V1
    frame = frame.reset_index(drop=True)
    labels = frame["label"].to_numpy(dtype=float)
    stride, window_rows = profile.stride_rows, profile.window_rows
    windows: list[pd.DataFrame] = []
    window_labels: list[int] = []
    mixed_label_windows = 0
    dirty_windows = 0
    dropped_segments = 0
    dropped_segment_rows = 0
    for start, end in clean_segments(frame, profile=profile):
        segment = frame.iloc[start:end]
        try:
            segment_windows, cleaning = clean_windows(
                segment.drop(columns="label"), profile=profile
            )
        except SkillError:
            dropped_segments += 1
            dropped_segment_rows += end - start
            continue
        # One dirty window no longer forfeits the whole segment: the shared rule
        # is identical, the training side just drops and counts the reject.
        dirty_windows += int(cleaning["rejected_windows"])
        segment_labels = labels[start:end]
        for index, window in enumerate(segment_windows):
            span = segment_labels[index * stride : index * stride + window_rows]
            unique = np.unique(span)
            if len(unique) == 1 and not np.isnan(unique[0]):
                windows.append(window)
                window_labels.append(int(unique[0]))
            else:
                mixed_label_windows += 1
    if not windows:
        return {
            "features": pd.DataFrame(columns=list(profile.feature_names)),
            "labels": np.empty(0, dtype=int),
            "window_count": 0,
            "mixed_label_windows": mixed_label_windows,
            "dirty_windows": dirty_windows,
            "dropped_segments": dropped_segments,
            "dropped_segment_rows": dropped_segment_rows,
        }
    features = extract_wesad_features(windows)
    return {
        "features": features,
        "labels": np.asarray(window_labels, dtype=int),
        "window_count": len(windows),
        "mixed_label_windows": mixed_label_windows,
        "dirty_windows": dirty_windows,
        "dropped_segments": dropped_segments,
        "dropped_segment_rows": dropped_segment_rows,
    }


def corpus_dataset(frame: pd.DataFrame, *, profile=None) -> dict:
    """Concatenate per-subject datasets; subjects stay the unit of independence."""
    datasets = [
        subject_dataset(session, profile=profile)
        for _, session in frame.groupby("subject_id", sort=True)
    ]
    features = (
        pd.concat([item["features"] for item in datasets], ignore_index=True)
        if datasets
        else pd.DataFrame(columns=list((profile or WESAD_V1).feature_names))
    )
    return {
        "features": features,
        "labels": np.concatenate([item["labels"] for item in datasets])
        if datasets
        else np.empty(0, dtype=int),
        "subjects": {
            subject: item for subject, item in zip(sorted(frame["subject_id"].unique()), datasets)
        },
        "mixed_label_windows": sum(item["mixed_label_windows"] for item in datasets),
        "dirty_windows": sum(item["dirty_windows"] for item in datasets),
        "dropped_segments": sum(item["dropped_segments"] for item in datasets),
        "dropped_segment_rows": sum(item["dropped_segment_rows"] for item in datasets),
    }
