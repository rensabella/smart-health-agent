"""Prepare deterministic WESAD artifacts: canonical tables, corpus and manifest.

Usage:
    python -m smarthealth.wesad.prepare --source <extracted WESAD root> \
        --manifest <WESAD.zip.sha256> --output <directory> --holdout-subject S17

The archive checksum is a gate: it is recorded with every derived artifact so a
prepared corpus can always be traced to one reviewed WESAD download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smarthealth.runtime.errors import SkillError
from smarthealth.wesad.loader import (
    CANONICAL_COLUMNS,
    CHEST_SAMPLE_RATE_HZ,
    DERIVATION_VERSION,
    ECG_BANDPASS_HZ,
    ECG_FILTER_ORDER,
    ECG_MIN_RR_SECONDS,
    ECG_PROMINENCE_MV,
    EDA_FILTER_ORDER,
    EDA_TONIC_CUTOFF_HZ,
    EXPECTED_SUBJECTS,
    HR_PLAUSIBLE_BPM,
    SCR_MIN_INTERVAL_S,
    SCR_PROMINENCE_U_S,
    WESAD_LABEL_MAP,
    load_subject,
    parse_sha256_file,
    write_canonical_csv,
)


def _subject_pickle(source: Path, subject: str) -> Path:
    nested = source / subject / f"{subject}.pkl"
    flat = source / f"{subject}.pkl"
    if nested.is_file():
        return nested
    if flat.is_file():
        return flat
    raise SkillError(
        "invalid_wesad_input", f"WESAD subject {subject} is missing from the source directory."
    )


def _subject_summary(record, frame: pd.DataFrame, pkl_sha256: str) -> dict:
    labels = frame["label"].to_numpy(dtype=float)
    hr = frame["hr"].to_numpy(dtype=float)
    valid_hr = hr[~np.isnan(hr)]
    return {
        "pkl_sha256": pkl_sha256,
        "rows": int(len(frame)),
        "dropped_partial_samples": int(record.dropped_partial_samples),
        "label_seconds": {
            str(canonical): int(np.count_nonzero(labels == canonical)) for canonical in (0, 1, 2)
        },
        "unlabeled_seconds": int(np.count_nonzero(np.isnan(labels))),
        "hr": {
            "median_bpm": float(np.median(valid_hr)) if len(valid_hr) else None,
            "p05_bpm": float(np.quantile(valid_hr, 0.05)) if len(valid_hr) else None,
            "p95_bpm": float(np.quantile(valid_hr, 0.95)) if len(valid_hr) else None,
            "nan_seconds": int(np.isnan(hr).sum()),
        },
        "wrist_available": bool(record.wrist_available),
        "wrist_signals": list(record.wrist_signals),
        "extra_top_level_keys": list(record.extra_top_level_keys),
        "extra_chest_keys": list(record.extra_chest_keys),
    }


def prepare(source: Path, manifest: Path, output: Path, *, holdout_subject: str) -> dict:
    source, manifest, output = Path(source), Path(manifest), Path(output)
    if holdout_subject not in EXPECTED_SUBJECTS:
        raise SkillError(
            "invalid_wesad_input",
            "Holdout subject must be one of the fifteen released WESAD subjects.",
        )
    zip_digest = parse_sha256_file(manifest)
    output.mkdir(parents=True, exist_ok=False)
    canonical_dir = output / "canonical"
    canonical_dir.mkdir()
    frames: dict[str, pd.DataFrame] = {}
    subjects_manifest = {}
    for subject in EXPECTED_SUBJECTS:
        path = _subject_pickle(source, subject)
        record = load_subject(path, expected_subject=subject)
        if list(record.frame.columns) != list(CANONICAL_COLUMNS):
            raise SkillError(
                "invalid_wesad_input", f"WESAD subject {subject} produced an unexpected table."
            )
        write_canonical_csv(record.frame, canonical_dir / f"{subject}.csv")
        frames[subject] = record.frame
        subjects_manifest[subject] = _subject_summary(
            record, record.frame, hashlib.sha256(path.read_bytes()).hexdigest()
        )
    corpus = pd.concat([frames[name] for name in EXPECTED_SUBJECTS], ignore_index=True)
    write_canonical_csv(corpus, output / "training.csv")
    write_canonical_csv(frames[holdout_subject], output / "fixture_subject.csv")
    document = {
        "schema_version": "1.0",
        "derivation_version": DERIVATION_VERSION,
        "dataset": {
            "name": "WESAD",
            "license": "CC-BY-4.0",
            "citation": (
                "Schmidt, Philip, et al. 'Introducing WESAD, a multimodal dataset for "
                "wearable stress and affect detection.' ICMI 2018."
            ),
            "wesad_zip_sha256": zip_digest,
            "subjects": list(EXPECTED_SUBJECTS),
            "holdout_subject": holdout_subject,
            "chest_sample_rate_hz": CHEST_SAMPLE_RATE_HZ,
        },
        "derivation": {
            "chest_only": True,
            "ecg_bandpass_hz": list(ECG_BANDPASS_HZ),
            "ecg_filter_order": ECG_FILTER_ORDER,
            "ecg_min_rr_seconds": ECG_MIN_RR_SECONDS,
            "ecg_prominence_mv": ECG_PROMINENCE_MV,
            "hr_plausible_bpm": list(HR_PLAUSIBLE_BPM),
            "eda_tonic_cutoff_hz": EDA_TONIC_CUTOFF_HZ,
            "eda_filter_order": EDA_FILTER_ORDER,
            "scr_prominence_u_s": SCR_PROMINENCE_U_S,
            "scr_min_interval_s": SCR_MIN_INTERVAL_S,
            "label_map_protocol_to_canonical": {
                str(key): value for key, value in WESAD_LABEL_MAP.items()
            },
        },
        "subjects": subjects_manifest,
        "totals": {
            "rows": int(sum(item["rows"] for item in subjects_manifest.values())),
            "label_seconds": {
                str(canonical): sum(
                    item["label_seconds"][str(canonical)] for item in subjects_manifest.values()
                )
                for canonical in (0, 1, 2)
            },
            "unlabeled_seconds": sum(
                item["unlabeled_seconds"] for item in subjects_manifest.values()
            ),
            "wrist_available_subjects": sum(
                1 for item in subjects_manifest.values() if item["wrist_available"]
            ),
        },
        "outputs": {
            "canonical": "canonical/SXX.csv",
            "training": "training.csv",
            "fixture_subject": "fixture_subject.csv",
        },
    }
    (output / "prepare_manifest.json").write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output),
        "holdout_subject": holdout_subject,
        "subjects": len(EXPECTED_SUBJECTS),
        "rows": document["totals"]["rows"],
        "label_seconds": document["totals"]["label_seconds"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-subject", default="S17")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(**vars(args)), indent=2))
    except SkillError as error:
        print(
            json.dumps({"error_code": error.code, "message": error.message}),
            file=__import__("sys").stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
