"""Reproducible, explicitly synthetic physiological events and held-out demo data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from smarthealth.config.settings import Settings

GENERATOR_VERSION = "1.0"


def generate_dataset(*, subjects=60, sessions=4, windows=8, seed=42) -> pd.DataFrame:
    if min(subjects, sessions, windows) < 1:
        raise ValueError("Generator dimensions must be positive")
    rng = np.random.default_rng(seed)
    records = []
    for subject in range(subjects):
        hr_offset = rng.normal(0, 7)
        eda_offset = rng.uniform(0.4, 1.8)
        for session in range(sessions):
            length = windows * 30
            sleep = float(rng.uniform(0.25, 0.95))
            # Targets describe simulated latent events, not thresholds applied to features.
            events = rng.binomial(1, 0.28, windows)
            event_strength = np.repeat(events * rng.uniform(0.4, 1.4, windows), 30)
            activity = np.clip(
                np.repeat(rng.beta(2, 3, windows), 30) + rng.normal(0, 0.035, length), 0, 1
            )
            drift = np.cumsum(rng.normal(0, 0.12, length))
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=session),
                        periods=length,
                        freq="s",
                    ),
                    "subject_id": f"s{subject:04d}",
                    "session_id": f"s{subject:04d}_r{session:03d}",
                    "heart_rate": np.maximum(
                        25,
                        65
                        + hr_offset
                        + 30 * activity
                        + 15 * event_strength
                        + drift
                        + rng.normal(0, 4, length),
                    ),
                    "EDA": np.maximum(
                        0,
                        eda_offset
                        + 0.8 * activity
                        + 1.2 * event_strength
                        + rng.normal(0, 0.35, length),
                    ),
                    "EEG_feature": np.maximum(
                        0,
                        0.4
                        + 0.15 * (1 - sleep)
                        + 0.18 * event_strength
                        + rng.normal(0, 0.15, length),
                    ),
                    "activity": activity,
                    "sleep_quality": sleep,
                    "label": np.repeat(events, 30).astype(int),
                }
            )
            # Missing measurements occur in both classes and are not a label shortcut.
            for column in ("heart_rate", "EDA", "EEG_feature", "activity"):
                mask = rng.random(length) < 0.002
                frame.loc[mask, column] = np.nan
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def generate_files(output_dir: Path, *, subjects=60, seed=42) -> dict:
    sample = output_dir / "sample"
    generated = output_dir / "generated"
    sample.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    training = generate_dataset(subjects=subjects, seed=seed)
    training.to_csv(generated / "training.csv", index=False, float_format="%.8f")
    demo = generate_dataset(subjects=1, sessions=1, windows=10, seed=seed + 10000)
    demo = demo.drop(columns="label")
    demo["subject_id"] = "demo_subject"
    demo["session_id"] = "demo_session"
    demo.to_csv(sample / "demo.csv", index=False, float_format="%.8f")
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "subjects": subjects,
        "training_rows": len(training),
        "demo_rows": len(demo),
        "task": "synthetic_anomaly",
        "label_definition": "simulated latent event; not stress, seizure, or disease diagnosis",
    }
    (generated / "generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    config = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.home / "data")
    parser.add_argument("--subjects", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_files(args.output_dir, subjects=args.subjects, seed=args.seed), indent=2
        )
    )


if __name__ == "__main__":
    main()
