"""Train and publish baseline bundles for a registered input profile.

The default synthetic path keeps its historical subject split, threshold rule
and metadata byte-for-byte. The WESAD path consumes the deterministic canonical
table, selects the deploy threshold on internal leave-one-subject-out
out-of-fold scores, and always holds one subject out as the fixture subject.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from smarthealth.config.settings import Settings
from smarthealth.pipeline.feature_engineering import (
    FEATURE_CONFIG,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_features,
)
from smarthealth.pipeline.preprocess import INPUT_PROFILE, PROCESSING_CONFIG, preprocess
from smarthealth.pipeline.profiles import WESAD_V1
from smarthealth.wesad.build import corpus_dataset
from smarthealth.wesad.loso import LABEL_NAMES, make_estimator, select_stress_threshold


def subject_split(frame: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    subjects = np.array(sorted(frame["subject_id"].unique()))
    if len(subjects) < 10:
        raise ValueError("At least ten independent subjects are required for three-way evaluation")
    np.random.default_rng(seed).shuffle(subjects)
    train_end = int(len(subjects) * 0.6)
    val_end = int(len(subjects) * 0.8)
    return {
        "train": subjects[:train_end].tolist(),
        "validation": subjects[train_end:val_end].tolist(),
        "test": subjects[val_end:].tolist(),
    }


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    all_features, labels = [], []
    for _, session in frame.groupby(["subject_id", "session_id"], sort=True):
        target = session["label"].to_numpy()
        if not np.isin(target, [0, 1]).all():
            raise ValueError("Training labels must be binary synthetic event labels")
        windows, _ = preprocess(session.drop(columns="label"))
        all_features.append(extract_features(windows))
        for index in range(len(windows)):
            window_labels = target[index * 30 : (index + 1) * 30]
            if len(np.unique(window_labels)) != 1:
                raise ValueError("This training profile requires one label per complete window")
            labels.append(int(window_labels[0]))
    x = pd.concat(all_features, ignore_index=True)
    y = np.asarray(labels)
    if len(np.unique(y)) != 2:
        raise ValueError("Every split must contain both classes; adjust the data generation setup")
    return x, y


def metrics(y, scores, threshold: float) -> dict:
    pred = scores >= threshold
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, scores)),
        "window_count": len(y),
        "positive_windows": int(y.sum()),
    }


def _write_bundle(output_dir: Path, backend: str, estimator, metadata: dict) -> Path:
    """Atomically publish one fitted bundle; existing bundles are never overwritten."""
    destination = output_dir / f"{backend}_v1.0"
    if destination.exists():
        raise FileExistsError(
            f"Bundle already exists: {destination}; choose a new output directory"
        )
    temporary = Path(tempfile.mkdtemp(prefix=".training-", dir=output_dir))
    try:
        joblib.dump(estimator, temporary / "estimator.joblib")
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        manifest = {
            "format_version": "1.0",
            "sha256": {
                name: hashlib.sha256((temporary / name).read_bytes()).hexdigest()
                for name in ("metadata.json", "estimator.joblib")
            },
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def train_models(
    data_path: Path, output_dir: Path, *, seed=42, backends=("random_forest", "lightgbm")
) -> dict:
    data_path, output_dir = Path(data_path), Path(output_dir)
    data_bytes = data_path.read_bytes()
    frame = pd.read_csv(data_path, dtype={"subject_id": str, "session_id": str})
    required = {"subject_id", "session_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Training requires subject, session and synthetic label metadata")
    splits = subject_split(frame, seed)
    datasets = {
        name: build_features(frame[frame.subject_id.isin(ids)]) for name, ids in splits.items()
    }
    train_x, train_y = datasets["train"]
    normal = train_x.loc[train_y == 0]
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("scikit-learn", "lightgbm", "numpy", "pandas", "joblib")
    }
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        if backend not in ("random_forest", "lightgbm"):
            raise ValueError("Unknown baseline")
        destination = output_dir / f"{backend}_v1.0"
        if destination.exists():
            raise FileExistsError(
                f"Bundle already exists: {destination}; choose a new output directory"
            )
        estimator = make_estimator(backend, seed=seed)
        started = time.perf_counter()
        estimator.fit(train_x, train_y)
        val_x, val_y = datasets["validation"]
        val_scores = estimator.predict_proba(val_x)[:, 1]
        thresholds = np.linspace(0.2, 0.8, 61)
        # Stable tie-break prefers the threshold closest to 0.5. Test labels never select it.
        threshold = max(thresholds, key=lambda t: (f1_score(val_y, val_scores >= t), -abs(t - 0.5)))
        threshold = float(threshold)
        evaluation = {
            name: metrics(y, estimator.predict_proba(x)[:, 1], threshold)
            for name, (x, y) in datasets.items()
            if name != "train"
        }
        metadata = {
            "name": "RandomForest" if backend == "random_forest" else "LightGBM",
            "backend": backend,
            "version": "1.0",
            "task_scope": "synthetic_anomaly_only",
            "input_schema_version": "1.0",
            "input_profile": INPUT_PROFILE,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "feature_config": FEATURE_CONFIG,
            "processing_config": PROCESSING_CONFIG,
            "label_mapping": {"0": "normal", "1": "anomaly"},
            "score_semantics": "synthetic_anomaly_model_score",
            "aggregation": "max",
            "threshold": threshold,
            "threshold_source": "maximum_validation_f1",
            "risk_thresholds": {"medium": threshold, "high": max(0.8, threshold)},
            "normal_reference": {
                n: {"mean": float(normal[n].mean()), "std": float(normal[n].std(ddof=0))}
                for n in FEATURE_NAMES
            },
            "seed": seed,
            "subject_splits": splits,
            "metrics": evaluation,
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "dependencies": dependencies,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "training_seconds": time.perf_counter() - started,
            "estimator_parameters": estimator.get_params(),
            "limitations": [
                "Synthetic data only; no clinical validation.",
                "Maximum window score is not a session probability.",
                "No calibration on real physiology; metrics reflect the synthetic generator.",
            ],
        }
        _write_bundle(output_dir, backend, estimator, metadata)
        results[backend] = {"bundle": str(destination), "threshold": threshold, **evaluation}
    (output_dir / "training_report.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def train_wesad(
    data_path: Path,
    output_dir: Path,
    *,
    holdout_subject: str,
    seed=42,
    backends=("random_forest", "lightgbm"),
) -> dict:
    """Train WESAD deploy bundles; the threshold comes from internal LOSO out-of-fold scores."""
    data_path, output_dir = Path(data_path), Path(output_dir)
    data_bytes = data_path.read_bytes()
    frame = pd.read_csv(data_path, dtype={"subject_id": str, "session_id": str})
    required = {"subject_id", "session_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Training requires subject, session and canonical label metadata")
    corpus = corpus_dataset(frame)
    subjects = sorted(corpus["subjects"])
    if holdout_subject not in subjects:
        raise ValueError(f"Holdout subject {holdout_subject!r} is not present in the table.")
    deploy_subjects = [name for name in subjects if name != holdout_subject]
    if len(deploy_subjects) < 2:
        raise ValueError("The internal LOSO threshold requires at least two deploy subjects.")
    total_windows = sum(corpus["subjects"][name]["window_count"] for name in subjects)
    if total_windows == 0:
        raise ValueError("No pure-label windows were produced from this table.")
    labels = tuple(WESAD_V1.expected_classes)
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("scikit-learn", "lightgbm", "numpy", "pandas", "joblib", "scipy")
    }
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        if backend not in ("random_forest", "lightgbm"):
            raise ValueError("Unknown baseline")
        destination = output_dir / f"{backend}_v1.0"
        if destination.exists():
            raise FileExistsError(
                f"Bundle already exists: {destination}; choose a new output directory"
            )
        # Internal LOSO over the deploy subjects only: the fixture subject never
        # contributes to threshold selection.
        oof_true, oof_stress = [], []
        for subject in deploy_subjects:
            test = corpus["subjects"][subject]
            if test["window_count"] == 0:
                continue
            others = [name for name in deploy_subjects if name != subject]
            fold_x = pd.concat(
                [corpus["subjects"][name]["features"] for name in others], ignore_index=True
            )
            fold_y = np.concatenate([corpus["subjects"][name]["labels"] for name in others])
            if len(fold_y) == 0 or len(np.unique(fold_y)) < 2:
                raise ValueError(f"Internal LOSO fold {subject} has a degenerate training pool.")
            fold_estimator = make_estimator(backend, seed=seed)
            fold_estimator.fit(fold_x, fold_y)
            fold_classes = [int(value) for value in fold_estimator.classes_]
            if 1 not in fold_classes:
                raise ValueError(f"Internal LOSO fold {subject} trained without stress windows.")
            oof_true.append(test["labels"])
            oof_stress.append(
                np.asarray(
                    fold_estimator.predict_proba(test["features"])[:, fold_classes.index(1)],
                    dtype=float,
                )
            )
        if not oof_true:
            raise ValueError("No pure-label windows among the deploy subjects.")
        threshold_report = select_stress_threshold(
            np.concatenate(oof_true), np.concatenate(oof_stress)
        )
        threshold = threshold_report["threshold"]
        train_x = pd.concat(
            [corpus["subjects"][name]["features"] for name in deploy_subjects], ignore_index=True
        )
        train_y = np.concatenate([corpus["subjects"][name]["labels"] for name in deploy_subjects])
        if sorted(np.unique(train_y)) != list(labels):
            raise ValueError("The deploy pool must contain all three WESAD classes.")
        started = time.perf_counter()
        estimator = make_estimator(backend, seed=seed)
        estimator.fit(train_x, train_y)
        holdout = corpus["subjects"][holdout_subject]
        evaluation = {}
        if holdout["window_count"]:
            prediction = estimator.predict(holdout["features"])
            evaluation["held_out_fixture"] = {
                "window_count": int(holdout["window_count"]),
                "macro_f1": float(
                    f1_score(
                        holdout["labels"],
                        prediction,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "class_windows": {
                    LABEL_NAMES[value]: int((holdout["labels"] == value).sum()) for value in labels
                },
            }
        normal = train_x.loc[train_y == 0]
        metadata = {
            "name": "RandomForest" if backend == "random_forest" else "LightGBM",
            "backend": backend,
            "version": "1.0",
            "task_scope": WESAD_V1.task_scope,
            "input_schema_version": "1.0",
            "input_profile": WESAD_V1.name,
            "feature_schema_version": WESAD_V1.feature_schema_version,
            "feature_names": list(WESAD_V1.feature_names),
            "feature_config": WESAD_V1.feature_config,
            "processing_config": WESAD_V1.processing_config,
            "label_mapping": WESAD_V1.label_mapping,
            "score_semantics": WESAD_V1.score_semantics,
            "aggregation": "max",
            "threshold": threshold,
            "threshold_source": "maximum_loso_out_of_fold_stress_f1",
            "risk_thresholds": {"medium": threshold, "high": max(0.8, threshold)},
            "normal_reference": {
                name: {"mean": float(normal[name].mean()), "std": float(normal[name].std(ddof=0))}
                for name in WESAD_V1.feature_names
            },
            "seed": seed,
            "subject_splits": {"train": deploy_subjects, "held_out_fixture": [holdout_subject]},
            "metrics": evaluation,
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "dependencies": dependencies,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "training_seconds": time.perf_counter() - started,
            "estimator_parameters": estimator.get_params(),
            "limitations": [
                "Trained on laboratory stress recordings (WESAD); no clinical validation.",
                "Maximum window score is not a session probability.",
                "Chest-derived signals only; wrist signals are not modeled in this version.",
            ],
        }
        _write_bundle(output_dir, backend, estimator, metadata)
        results[backend] = {
            "bundle": str(destination),
            "threshold": threshold,
            "threshold_selection": threshold_report,
            **evaluation,
        }
    report = {
        "profile": WESAD_V1.name,
        "holdout_subject": holdout_subject,
        "windowing": {
            "subjects": len(subjects),
            "deploy_subjects": len(deploy_subjects),
            "pure_label_windows": total_windows,
            "mixed_label_windows": int(corpus["mixed_label_windows"]),
            "dropped_segments": int(corpus["dropped_segments"]),
            "dropped_segment_rows": int(corpus["dropped_segment_rows"]),
        },
        "backends": results,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    config = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.home / "data/generated/training.csv")
    parser.add_argument("--output-dir", type=Path, default=config.home / "models_saved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["both", "random_forest", "lightgbm"], default="both")
    parser.add_argument("--profile", choices=["synthetic", "wesad"], default="synthetic")
    parser.add_argument("--cv", choices=["none", "loso"], default="none")
    parser.add_argument("--holdout-subject", default="S17")
    args = parser.parse_args()
    if (args.profile == "wesad") != (args.cv == "loso"):
        parser.error("--profile wesad must be combined with --cv loso; synthetic uses --cv none")
    backends = ("random_forest", "lightgbm") if args.model == "both" else (args.model,)
    if args.profile == "wesad":
        results = train_wesad(
            args.data,
            args.output_dir,
            holdout_subject=args.holdout_subject,
            seed=args.seed,
            backends=backends,
        )
    else:
        results = train_models(args.data, args.output_dir, seed=args.seed, backends=backends)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
