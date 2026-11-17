"""Leave-one-subject-out evaluation over the WESAD canonical table.

Every fold trains on the other subjects and scores the held-out one, so the
pooled numbers measure generalization across people. Window construction is
the shared production preprocess with pure-label windows only; mixed and
unlabeled windows are dropped and counted so every metric keeps an honest
denominator. The stress-vs-rest threshold is selected on pooled out-of-fold
scores with the same rule the deploy trainer applies to its internal LOSO.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from smarthealth.pipeline.profiles import WESAD_V1
from smarthealth.wesad.build import corpus_dataset
from smarthealth.wesad.loader import DERIVATION_VERSION

CANONICAL_LABELS = (0, 1, 2)
LABEL_NAMES = {0: "baseline", 1: "stress", 2: "amusement"}
BACKENDS = ("random_forest", "lightgbm")
THRESHOLD_GRID = tuple(float(value) for value in np.linspace(0.2, 0.8, 61))


def make_estimator(backend: str, *, seed: int):
    """Deterministic baseline estimators shared by LOSO and the deploy trainer."""
    if backend == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=seed, n_jobs=1
        )
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            num_leaves=20,
            min_child_samples=15,
            learning_rate=0.06,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"Unknown baseline backend: {backend!r}")


def select_stress_threshold(y_true: np.ndarray, stress_scores: np.ndarray) -> dict:
    """Maximum stress-vs-rest F1 over the fixed grid; ties prefer the value closest to 0.5."""
    binary = (np.asarray(y_true) == 1).astype(int)
    scores = np.asarray(stress_scores, dtype=float)
    if len(np.unique(binary)) != 2:
        raise ValueError("Threshold selection requires both stress and nonstress windows.")
    best = max(
        THRESHOLD_GRID,
        key=lambda t: (
            f1_score(binary, (scores >= t).astype(int), zero_division=0),
            -abs(t - 0.5),
        ),
    )
    prediction = (scores >= best).astype(int)
    return {
        "threshold": float(best),
        "f1": float(f1_score(binary, prediction, zero_division=0)),
        "precision": float(precision_score(binary, prediction, zero_division=0)),
        "recall": float(recall_score(binary, prediction, zero_division=0)),
        "positive_windows": int(binary.sum()),
        "selection_rule": "maximum_stress_f1_tie_break_closest_to_0.5",
    }


def _stress_probability(estimator, features: pd.DataFrame) -> np.ndarray:
    classes = [int(value) for value in estimator.classes_]
    if 1 not in classes:
        return np.zeros(len(features))
    return np.asarray(estimator.predict_proba(features)[:, classes.index(1)], dtype=float)


def _class_counts(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    return {LABEL_NAMES[value]: int((labels == value).sum()) for value in CANONICAL_LABELS}


def _classification_metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=CANONICAL_LABELS, average=None, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CANONICAL_LABELS, average="macro", zero_division=0)
        ),
        "per_class_f1": {
            LABEL_NAMES[value]: float(score) for value, score in zip(CANONICAL_LABELS, per_class)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=CANONICAL_LABELS).tolist(),
        "window_count": int(len(y_true)),
        "class_windows": _class_counts(y_true),
    }


def loso_evaluate(data_path: Path, *, seed: int = 42, backends: tuple = BACKENDS) -> dict:
    """Compute the leave-one-subject-out report for a canonical training table."""
    data_path = Path(data_path)
    data_bytes = data_path.read_bytes()
    frame = pd.read_csv(data_path, dtype={"subject_id": str, "session_id": str})
    corpus = corpus_dataset(frame)
    subjects = sorted(corpus["subjects"])
    if len(subjects) < 2:
        raise ValueError("Leave-one-subject-out requires at least two subjects.")
    total_windows = sum(item["window_count"] for item in corpus["subjects"].values())
    if total_windows == 0:
        raise ValueError("No pure-label windows were produced from this table.")
    backend_reports = {}
    for backend in backends:
        folds = {}
        oof_true, oof_pred, oof_stress = [], [], []
        for subject in subjects:
            test = corpus["subjects"][subject]
            others = [name for name in subjects if name != subject]
            train_x = pd.concat(
                [corpus["subjects"][name]["features"] for name in others], ignore_index=True
            )
            train_y = np.concatenate([corpus["subjects"][name]["labels"] for name in others])
            fold = {"window_count": int(test["window_count"])}
            if test["window_count"] == 0:
                folds[subject] = {
                    **fold,
                    "macro_f1": None,
                    "confusion": None,
                    "note": "No pure-label windows for this subject.",
                }
                continue
            if len(train_y) == 0 or len(np.unique(train_y)) < 2:
                raise ValueError(f"Fold {subject} has a degenerate training pool.")
            estimator = make_estimator(backend, seed=seed)
            estimator.fit(train_x, train_y)
            prediction = estimator.predict(test["features"])
            folds[subject] = {
                **fold,
                **_classification_metrics(test["labels"], prediction),
                "train_classes": [
                    LABEL_NAMES[value] for value in sorted({int(cls) for cls in estimator.classes_})
                ],
            }
            oof_true.append(test["labels"])
            oof_pred.append(prediction)
            oof_stress.append(_stress_probability(estimator, test["features"]))
        y_true = np.concatenate(oof_true) if oof_true else np.empty(0, dtype=int)
        y_pred = np.concatenate(oof_pred) if oof_pred else np.empty(0, dtype=int)
        stress = np.concatenate(oof_stress) if oof_stress else np.empty(0)
        backend_reports[backend] = {
            "folds": folds,
            "pooled": _classification_metrics(y_true, y_pred),
            "stress_vs_rest": select_stress_threshold(y_true, stress),
        }
    return {
        "schema_version": "1.0",
        "dataset": {
            "table": data_path.name,
            "sha256": hashlib.sha256(data_bytes).hexdigest(),
            "derivation_version": DERIVATION_VERSION,
            "profile": WESAD_V1.name,
            "subjects": subjects,
        },
        "protocol": {
            "name": "leave_one_subject_out",
            "folds": len(subjects),
            "window_rows": WESAD_V1.window_rows,
            "stride_rows": WESAD_V1.stride_rows,
            "labels": [LABEL_NAMES[value] for value in CANONICAL_LABELS],
            "threshold_grid": {"low": 0.2, "high": 0.8, "steps": len(THRESHOLD_GRID)},
        },
        "windowing": {
            "pure_label_windows": total_windows,
            "mixed_label_windows": int(corpus["mixed_label_windows"]),
            "dirty_windows": int(corpus["dirty_windows"]),
            "dropped_segments": int(corpus["dropped_segments"]),
            "dropped_segment_rows": int(corpus["dropped_segment_rows"]),
            "windows_per_subject": {
                subject: int(corpus["subjects"][subject]["window_count"]) for subject in subjects
            },
        },
        "seed": seed,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("scikit-learn", "lightgbm", "numpy", "pandas", "scipy")
        },
        "backends": backend_reports,
    }


def _markdown(document: dict) -> str:
    windowing = document["windowing"]
    protocol = document["protocol"]
    lines = [
        "# WESAD leave-one-subject-out report",
        "",
        f"- Data table: `{document['dataset']['table']}` "
        f"(sha256 `{document['dataset']['sha256']}`).",
        f"- Subjects: {len(document['dataset']['subjects'])} "
        f"({', '.join(document['dataset']['subjects'])}).",
        f"- Windows: {windowing['pure_label_windows']} pure-label "
        f"({windowing['mixed_label_windows']} mixed dropped, "
        f"{windowing['dirty_windows']} dirty dropped, "
        f"{windowing['dropped_segments']} segments dropped).",
        f"- Protocol: {protocol['window_rows']}-row windows, stride "
        f"{protocol['stride_rows']}, classes baseline/stress/amusement, "
        f"seed {document['seed']}.",
        "",
    ]
    for backend, report in document["backends"].items():
        pooled = report["pooled"]
        threshold = report["stress_vs_rest"]
        lines += [
            f"## {backend}",
            "",
            f"Pooled out-of-fold macro-F1: **{pooled['macro_f1']:.4f}** "
            f"over {pooled['window_count']} windows.",
            "",
            "| subject | windows | macro-F1 |",
            "|---|---|---|",
        ]
        for subject, fold in report["folds"].items():
            value = "n/a" if fold["macro_f1"] is None else f"{fold['macro_f1']:.4f}"
            lines.append(f"| {subject} | {fold['window_count']} | {value} |")
        lines += [
            "",
            "Per-class F1: "
            + ", ".join(f"{name} {score:.4f}" for name, score in pooled["per_class_f1"].items()),
            "",
            f"Stress-vs-rest (pooled OOF): threshold {threshold['threshold']:.3f}, "
            f"F1 {threshold['f1']:.4f}, precision {threshold['precision']:.4f}, "
            f"recall {threshold['recall']:.4f}.",
            "",
        ]
    return "\n".join(lines) + "\n"


def write_report(document: dict, output: Path) -> None:
    """Write loso_report.json/.md; existing reports are never overwritten."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("loso_report.json", "loso_report.md"):
        if (output / name).exists():
            raise FileExistsError(f"{output / name} already exists; choose a new output directory")
    (output / "loso_report.json").write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "loso_report.md").write_text(_markdown(document), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["both", *BACKENDS], default="both")
    args = parser.parse_args(argv)
    backends = BACKENDS if args.model == "both" else (args.model,)
    document = loso_evaluate(args.data, seed=args.seed, backends=backends)
    write_report(document, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "folds": document["protocol"]["folds"],
                "windows": document["windowing"]["pure_label_windows"],
                "pooled_macro_f1": {
                    backend: report["pooled"]["macro_f1"]
                    for backend, report in document["backends"].items()
                },
                "stress_threshold": {
                    backend: report["stress_vs_rest"]["threshold"]
                    for backend, report in document["backends"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
