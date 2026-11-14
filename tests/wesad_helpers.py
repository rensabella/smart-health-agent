"""Deterministic fake WESAD fixtures shared by the Phase 4 test modules.

Everything here is synthetic mini-data used only by tests: it is never
packaged into the wheel and never shipped as an evaluation fixture.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CHEST_HZ = 700
CANONICAL_COLUMNS = (
    "timestamp",
    "subject_id",
    "session_id",
    "label",
    "hr",
    "eda",
    "emg",
    "temp",
    "resp",
    "acc_mag",
    "scr",
)


def _chest_signals(
    total_samples: int, *, hr_bpm: float, scr_second: int, seed: int, column_vectors: bool = True
) -> dict:
    """Chest channels; the released pickles carry scalar channels as (n, 1) columns."""
    n = total_samples
    ecg = np.zeros(n)
    period = max(1, int(round(CHEST_HZ * 60.0 / hr_bpm)))
    for peak in range(0, n, period):
        lo, hi = max(0, peak - 20), min(n, peak + 20)
        ecg[lo:hi] += 1.2 * (1.0 - np.abs(np.linspace(-1.0, 1.0, hi - lo)))
    eda = 5.0 + 0.01 * (np.arange(n) / CHEST_HZ)
    decay = np.arange(3 * CHEST_HZ) / CHEST_HZ
    scr_shape = 0.8 * (1.0 - np.exp(-decay / 0.3)) * np.exp(-decay / 2.0)
    start = scr_second * CHEST_HZ
    room = max(0, min(len(scr_shape), n - start))
    eda[start : start + room] += scr_shape[:room]
    phase = np.arange(n) / CHEST_HZ
    emg = 0.05 + 0.01 * np.sin(2 * np.pi * phase / 50.0)
    resp = 0.5 + 0.2 * np.sin(2 * np.pi * 0.25 * phase + seed)
    temp = 36.5 + 0.05 * np.sin(2 * np.pi * phase / 60.0)
    acc = np.tile([0.1, 0.2, 0.3], (n, 1)) + 0.01 * np.sin(2 * np.pi * phase / 30.0)[:, None]
    if column_vectors:
        ecg, eda, emg, resp, temp = (
            values[:, None] for values in (ecg, eda, emg, resp, temp)
        )
    return {"ACC": acc, "ECG": ecg, "EDA": eda, "EMG": emg, "Resp": resp, "Temp": temp}


def protocol_label(seconds: int, blocks, *, transient_samples: int = 0) -> np.ndarray:
    """Protocol labels at 700 Hz; transient spans use 7 and never map to a class."""
    label = np.zeros(seconds * CHEST_HZ, dtype=int)
    cursor = 0
    for rows, protocol in blocks:
        label[cursor * CHEST_HZ : (cursor + rows) * CHEST_HZ] = protocol
        cursor += rows
    if transient_samples:
        half = transient_samples // 2
        boundaries = np.cumsum([rows for rows, _ in blocks])[:-1]
        for boundary in boundaries:
            label[boundary * CHEST_HZ - half : boundary * CHEST_HZ + half] = 7
    return label


def write_fake_subject(
    source: Path,
    subject: str,
    *,
    seconds: int = 90,
    blocks=None,
    hr_bpm: float = 72.0,
    scr_second: int = 45,
    seed: int = 0,
    wrist: bool = True,
    timestamp: datetime | None = None,
    nested: bool = False,
    top_level_label: bool = True,
    column_vectors: bool = True,
    tail_samples: int = 0,
    transient_samples: int = 0,
    extra_top: dict | None = None,
) -> Path:
    """Write one deterministic fake WESAD pickle; defaults mirror the released layout."""
    if blocks is None:
        first = seconds // 3
        blocks = [(first, 1), (first, 2), (seconds - 2 * first, 3)]
    total = seconds * CHEST_HZ + tail_samples
    chest = _chest_signals(
        total, hr_bpm=hr_bpm, scr_second=scr_second, seed=seed, column_vectors=column_vectors
    )
    label = protocol_label(seconds, blocks, transient_samples=transient_samples)
    if tail_samples:
        label = np.concatenate([label, np.full(tail_samples, 7, dtype=int)])
    data = {"subject": subject, "signal": {"chest": chest}}
    if top_level_label:
        data["label"] = label
    else:
        chest["label"] = label
    if timestamp is not None:
        data["timestamp"] = timestamp
    if extra_top:
        data.update(extra_top)
    if wrist:
        data["signal"]["wrist"] = {
            "BVP": np.zeros(64),
            "EDA": np.zeros(64),
            "TEMP": np.zeros(64),
        }
    directory = Path(source) / subject if nested else Path(source)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{subject}.pkl"
    with path.open("wb") as stream:
        pickle.dump(data, stream)
    return path


def canonical_frame(
    subject: str,
    blocks,
    *,
    session: str = "protocol",
    seed: int = 0,
    gap: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Deterministic 1 Hz canonical rows; block labels are canonical 0/1/2 or None."""
    rows = []
    base = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    index = 0
    for block_rows, label in blocks:
        for _ in range(block_rows):
            wiggle = 2.0 * np.sin((index + seed * 13) / 5.0)
            rows.append(
                {
                    "timestamp": (base + timedelta(seconds=index)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "subject_id": subject,
                    "session_id": session,
                    "label": np.nan if label is None else float(label),
                    "hr": 62.0 + 16.0 * (label or 0) + wiggle,
                    "eda": 3.5 + 1.1 * (label or 0) + 0.2 * np.sin(index / 7.0 + seed),
                    "emg": 0.05 + 0.02 * (label or 0) + 0.005 * np.sin(index / 3.0),
                    "temp": 36.4 + 0.15 * (label or 0) + 0.02 * np.sin(index / 40.0),
                    "resp": 0.5 + 0.1 * (label or 0) + 0.15 * np.sin(2 * np.pi * index / 20.0 + seed),
                    "acc_mag": 1.0 + 0.05 * (label or 0) + 0.02 * np.sin(index / 4.0),
                    "scr": 1.0 if (label == 1 and index % 6 == 0) else 0.0,
                }
            )
            index += 1
    frame = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if gap is not None:
        start, length = gap
        frame.loc[start : start + length - 1, "hr"] = np.nan
    return frame
