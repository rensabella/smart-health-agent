"""Deterministic WESAD subject loading and 1 Hz canonical-table derivation.

Consumes the public UCI WESAD pickle files (chest RespiBAN at 700 Hz, wrist
Empatica E4). Only chest-derived signals enter the canonical table in v1; the
wrist signals are recorded for availability only. Every derivation parameter
is fixed here so identical inputs always produce identical tables.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from smarthealth.runtime.errors import SkillError

EXPECTED_SUBJECTS = (
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
)
CHEST_SAMPLE_RATE_HZ = 700
# Key spellings exactly as released in the WESAD pickles (note "Temp").
REQUIRED_CHEST_KEYS = ("ACC", "ECG", "EDA", "EMG", "Resp", "Temp")
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
# WESAD protocol labels: 1=baseline, 2=stress, 3=amusement; others are
# meditation/transient and never map into the three-class vocabulary.
WESAD_LABEL_MAP = {1: 0, 2: 1, 3: 2}
ECG_BANDPASS_HZ = (0.5, 40.0)
ECG_FILTER_ORDER = 3
ECG_MIN_RR_SECONDS = 0.3
ECG_PROMINENCE_MV = 0.25
HR_PLAUSIBLE_BPM = (30.0, 220.0)
EDA_TONIC_CUTOFF_HZ = 0.05
EDA_FILTER_ORDER = 4
SCR_PROMINENCE_U_S = 0.05
SCR_MIN_INTERVAL_S = 0.5
FALLBACK_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DERIVATION_VERSION = "wesad_canonical_v1"


def _fail(message: str) -> None:
    raise SkillError("invalid_wesad_input", message)


def _bandpass(values: np.ndarray, low: float, high: float, order: int) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=CHEST_SAMPLE_RATE_HZ, output="sos")
    return sosfiltfilt(sos, values)


def _lowpass(values: np.ndarray, cutoff: float, order: int) -> np.ndarray:
    sos = butter(order, cutoff, btype="lowpass", fs=CHEST_SAMPLE_RATE_HZ, output="sos")
    return sosfiltfilt(sos, values)


def ecg_to_hr(ecg: np.ndarray) -> np.ndarray:
    """Median instantaneous heart rate per full second; implausible beats are dropped."""
    seconds = len(ecg) // CHEST_SAMPLE_RATE_HZ
    if seconds == 0:
        return np.empty(0)
    filtered = _bandpass(np.asarray(ecg, dtype=float), *ECG_BANDPASS_HZ, order=ECG_FILTER_ORDER)
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(ECG_MIN_RR_SECONDS * CHEST_SAMPLE_RATE_HZ)),
        prominence=ECG_PROMINENCE_MV,
    )
    if len(peaks) < 2:
        return np.full(seconds, np.nan)
    intervals = np.diff(peaks) / CHEST_SAMPLE_RATE_HZ
    rates = 60.0 / intervals
    plausible = (rates >= HR_PLAUSIBLE_BPM[0]) & (rates <= HR_PLAUSIBLE_BPM[1])
    per_second = np.full(seconds, np.nan)
    # Each interval is attributed to the second of its later R peak.
    buckets: list[list[float]] = [[] for _ in range(seconds)]
    for index, (rate, ok) in enumerate(zip(rates, plausible)):
        if ok:
            buckets[peaks[index + 1] // CHEST_SAMPLE_RATE_HZ].append(float(rate))
    for second, values in enumerate(buckets):
        if values:
            per_second[second] = float(np.median(values))
    return per_second


def scr_indicator(eda: np.ndarray) -> np.ndarray:
    """Count of skin-conductance responses per full second (tonic component removed)."""
    seconds = len(eda) // CHEST_SAMPLE_RATE_HZ
    if seconds == 0:
        return np.empty(0)
    values = np.asarray(eda, dtype=float)[: seconds * CHEST_SAMPLE_RATE_HZ]
    tonic = _lowpass(values, EDA_TONIC_CUTOFF_HZ, order=EDA_FILTER_ORDER)
    phasic = values - tonic
    peaks, _ = find_peaks(
        phasic,
        height=0.0,
        prominence=SCR_PROMINENCE_U_S,
        distance=max(1, int(SCR_MIN_INTERVAL_S * CHEST_SAMPLE_RATE_HZ)),
    )
    counts = np.zeros(seconds)
    for peak in peaks:
        counts[peak // CHEST_SAMPLE_RATE_HZ] += 1.0
    return counts


def _as_vector(values) -> np.ndarray:
    """Released WESAD pickles store scalar chest channels as (n, 1) column vectors."""
    array = np.asarray(values)
    return array[:, 0] if array.ndim == 2 and array.shape[1] == 1 else array


def _per_second_mean(values: np.ndarray) -> np.ndarray:
    seconds = len(values) // CHEST_SAMPLE_RATE_HZ
    trimmed = np.asarray(values, dtype=float)[: seconds * CHEST_SAMPLE_RATE_HZ]
    return trimmed.reshape(seconds, CHEST_SAMPLE_RATE_HZ).mean(axis=1)


def _per_second_label(values: np.ndarray) -> np.ndarray:
    """Map a second only when all 700 protocol samples agree on a three-class label."""
    seconds = len(values) // CHEST_SAMPLE_RATE_HZ
    trimmed = np.asarray(values)[: seconds * CHEST_SAMPLE_RATE_HZ].reshape(
        seconds, CHEST_SAMPLE_RATE_HZ
    )
    labels = np.full(seconds, np.nan)
    first = trimmed[:, 0]
    uniform = (trimmed == first[:, None]).all(axis=1)
    for protocol, canonical in WESAD_LABEL_MAP.items():
        labels[uniform & (first == protocol)] = canonical
    return labels


def _recording_epoch(data: dict) -> datetime:
    stamp = data.get("timestamp")
    if isinstance(stamp, datetime):
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return FALLBACK_EPOCH


@dataclass(frozen=True)
class SubjectRecord:
    subject: str
    frame: pd.DataFrame
    wrist_available: bool = False
    wrist_signals: tuple[str, ...] = ()
    extra_top_level_keys: tuple[str, ...] = field(default=())
    extra_chest_keys: tuple[str, ...] = field(default=())
    dropped_partial_samples: int = 0


def load_subject(path: Path, *, expected_subject: str | None = None) -> SubjectRecord:
    """Load one WESAD pickle into the deterministic canonical table."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            data = pickle.load(stream, encoding="latin1")
    except Exception as error:
        raise SkillError("invalid_wesad_input", "WESAD subject file is not loadable.") from error
    if not isinstance(data, dict) or "signal" not in data or "subject" not in data:
        _fail("WESAD subject file misses the expected top-level structure.")
    subject = data["subject"]
    if not isinstance(subject, str) or not subject.strip():
        _fail("WESAD subject identifier is invalid.")
    if expected_subject is not None and subject != expected_subject:
        _fail("WESAD subject identifier does not match its file name.")
    signal = data["signal"]
    if not isinstance(signal, dict) or not isinstance(signal.get("chest"), dict):
        _fail("WESAD subject file misses chest signals.")
    chest = signal["chest"]
    missing = [key for key in REQUIRED_CHEST_KEYS if key not in chest]
    if missing:
        _fail(f"WESAD chest signals are missing required keys: {', '.join(missing)}.")
    label = chest.get("label", data.get("label"))
    if label is None:
        _fail("WESAD subject file has no protocol label array.")
    arrays = {key: np.asarray(chest[key]) for key in REQUIRED_CHEST_KEYS}
    # Both the released column-vector layout and plain one-dimensional arrays
    # are accepted and normalized; any wider layout stays a contract failure.
    label = _as_vector(label)
    for name in ("ECG", "EDA", "EMG", "Resp", "Temp"):
        arrays[name] = _as_vector(arrays[name])
    if label.ndim != 1 or any(value.ndim not in (1, 2) for value in arrays.values()):
        _fail("WESAD signal arrays have unexpected shapes.")
    if arrays["ACC"].ndim != 2 or arrays["ACC"].shape[1] != 3:
        _fail("WESAD chest ACC must be a three-axis array.")
    for name in ("ECG", "EDA", "EMG", "Resp", "Temp"):
        if arrays[name].ndim != 1:
            _fail(f"WESAD chest {name} must be a one-dimensional array.")
    length = len(arrays["ECG"])
    if any(len(value) != length for value in list(arrays.values()) + [label]):
        _fail("WESAD chest signals are not sample-aligned.")
    try:
        seconds = length // CHEST_SAMPLE_RATE_HZ
        acc_mag = np.sqrt((arrays["ACC"].astype(float) ** 2).sum(axis=1))
        frame = pd.DataFrame(
            {
                "timestamp": [
                    (_recording_epoch(data) + timedelta(seconds=second)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    for second in range(seconds)
                ],
                "subject_id": subject,
                "session_id": "protocol",
                "label": _per_second_label(label),
                "hr": ecg_to_hr(arrays["ECG"]),
                "eda": _per_second_mean(arrays["EDA"]),
                "emg": _per_second_mean(arrays["EMG"]),
                "temp": _per_second_mean(arrays["Temp"]),
                "resp": _per_second_mean(arrays["Resp"]),
                "acc_mag": _per_second_mean(acc_mag),
                "scr": scr_indicator(arrays["EDA"]),
            }
        )
    except (ValueError, TypeError) as error:
        raise SkillError("invalid_wesad_input", "WESAD signals cannot be normalized.") from error
    wrist = signal.get("wrist")
    wrist_signals = tuple(sorted(wrist)) if isinstance(wrist, dict) else ()
    extra_top = tuple(
        sorted(key for key in data if key not in {"subject", "signal", "label", "timestamp"})
    )
    extra_chest = tuple(sorted(key for key in chest if key not in (*REQUIRED_CHEST_KEYS, "label")))
    return SubjectRecord(
        subject=subject,
        frame=frame,
        wrist_available=bool(wrist_signals),
        wrist_signals=wrist_signals,
        extra_top_level_keys=extra_top,
        extra_chest_keys=extra_chest,
        dropped_partial_samples=length - seconds * CHEST_SAMPLE_RATE_HZ,
    )


def write_canonical_csv(frame: pd.DataFrame, path: Path) -> None:
    """Deterministic CSV bytes: fixed column order, float format and line endings."""
    Path(path).write_bytes(
        frame.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode("utf-8")
    )


def parse_sha256_file(path: Path) -> str:
    """Read a shasum-style manifest line and return the hex digest."""
    text = Path(path).read_text(encoding="utf-8").strip()
    digest = text.split()[0] if text else ""
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise SkillError("invalid_wesad_input", "WESAD archive checksum file is malformed.")
    return digest.lower()
