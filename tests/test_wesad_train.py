"""Deploy-bundle training, three-class prediction semantics and LOSO reports."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from wesad_helpers import canonical_frame

from smarthealth.cli.train import train_wesad
from smarthealth.models.model_loader import load_model
from smarthealth.wesad import loso as loso_module
from smarthealth.wesad.build import corpus_dataset
from smarthealth.wesad.loso import loso_evaluate, select_stress_threshold, write_report

SUBJECTS = ["S2", "S3", "S4", "S5"]


@pytest.fixture(scope="module")
def wesad_training_csv(tmp_path_factory):
    root = tmp_path_factory.mktemp("wesad_train")
    frames = [
        canonical_frame(subject, [(130, 0), (130, 1), (130, 2)], seed=index)
        for index, subject in enumerate(SUBJECTS)
    ]
    path = root / "training.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def wesad_bundles(wesad_training_csv, tmp_path_factory):
    output = tmp_path_factory.mktemp("bundles")
    report = train_wesad(
        wesad_training_csv, output, holdout_subject="S5", backends=("random_forest", "lightgbm")
    )
    return output, report


def test_bundle_metadata_pins_the_wesad_contract(wesad_bundles):
    output, report = wesad_bundles
    assert report["profile"] == "wesad_physiology_v1"
    assert report["holdout_subject"] == "S5"
    assert report["windowing"]["pure_label_windows"] == 32
    assert report["windowing"]["mixed_label_windows"] == 16
    assert report["windowing"]["deploy_subjects"] == 3
    for backend in ("random_forest", "lightgbm"):
        assert (output / f"{backend}_v1.0" / "metadata.json").is_file()
    metadata = json.loads(
        (output / "random_forest_v1.0" / "metadata.json").read_text()
    )
    assert metadata["input_profile"] == "wesad_physiology_v1"
    assert metadata["task_scope"] == "wesad_stress_only"
    assert metadata["score_semantics"] == "wesad_stress_model_score"
    assert metadata["label_mapping"] == {"0": "baseline", "1": "stress", "2": "amusement"}
    assert metadata["threshold_source"] == "maximum_loso_out_of_fold_stress_f1"
    assert 0.2 <= metadata["threshold"] <= 0.8
    assert metadata["subject_splits"] == {
        "train": ["S2", "S3", "S4"],
        "held_out_fixture": ["S5"],
    }
    assert metadata["risk_thresholds"]["medium"] == metadata["threshold"]
    assert "scipy" in metadata["dependencies"]
    assert any("no clinical validation" in item for item in metadata["limitations"])
    assert metadata["metrics"]["held_out_fixture"]["window_count"] == 8


def test_bundles_load_and_predict_three_classes(wesad_bundles, wesad_training_csv):
    output, _ = wesad_bundles
    frame = pd.read_csv(wesad_training_csv, dtype={"subject_id": str})
    holdout = corpus_dataset(frame)["subjects"]["S5"]
    features = holdout["features"]
    for backend in ("random_forest", "lightgbm"):
        predictor = load_model(output / f"{backend}_v1.0")
        assert predictor.profile.name == "wesad_physiology_v1"
        result = predictor.predict(features)
        vocabulary = {"baseline", "stress", "amusement"}
        windows = result["window_predictions"]
        assert all(window["prediction"] in vocabulary for window in windows)
        assert result["prediction"] in vocabulary
        threshold = predictor.metadata["threshold"]
        probabilities = predictor.estimator.predict_proba(features)
        stress_column = list(predictor.estimator.classes_).index(1)
        for index, window in enumerate(windows):
            assert window["risk_score"] == pytest.approx(
                float(probabilities[index, stress_column])
            )
            assert (window["risk_score"] >= threshold) == (window["prediction"] == "stress")
        highest = windows[int(np.argmax([w["risk_score"] for w in windows]))]
        assert result["prediction"] == highest["prediction"]
        assert result["risk_score"] == pytest.approx(highest["risk_score"])
        assert result["anomalous_ratio"] == pytest.approx(
            float(np.mean(np.asarray([w["risk_score"] for w in windows]) >= threshold))
        )
        assert result["model"]["task_scope"] == "wesad_stress_only"
        explained = predictor.explain(features, top_k=5)
        names = {item["name"] for item in explained["local_evidence"]}
        assert names <= set(predictor.metadata["feature_names"])
        assert any("WESAD" in item for item in explained["limitations"])


def test_train_wesad_refuses_overwrite_and_unknown_holdout(wesad_training_csv, tmp_path):
    train_wesad(
        wesad_training_csv, tmp_path, holdout_subject="S5", backends=("random_forest",)
    )
    with pytest.raises(FileExistsError):
        train_wesad(
            wesad_training_csv, tmp_path, holdout_subject="S5", backends=("random_forest",)
        )
    with pytest.raises(ValueError):
        train_wesad(
            wesad_training_csv, tmp_path / "other", holdout_subject="S99", backends=()
        )


def test_internal_loso_never_trains_on_the_holdout(wesad_training_csv, tmp_path, monkeypatch):
    from smarthealth.cli import train as train_module

    fits = []
    real = train_module.make_estimator

    def recording(backend, *, seed):
        estimator = real(backend, seed=seed)
        original_fit = estimator.fit

        def fit(x, y, **kwargs):
            # Restore the bound method so the dumped bundle stays picklable.
            estimator.fit = original_fit
            fits.append((len(x), sorted({int(value) for value in np.unique(y)})))
            return original_fit(x, y, **kwargs)

        estimator.fit = fit
        return estimator

    monkeypatch.setattr(train_module, "make_estimator", recording)
    train_wesad(
        wesad_training_csv, tmp_path, holdout_subject="S5", backends=("random_forest",)
    )
    # Three internal folds (16 windows each) plus the final deploy fit (24 windows);
    # the eight holdout windows never enter any training pool.
    assert [count for count, _ in fits] == [16, 16, 16, 24]
    assert all(classes == [0, 1, 2] for _, classes in fits)


def test_loso_evaluate_reports_all_folds(wesad_training_csv, tmp_path):
    document = loso_evaluate(wesad_training_csv)
    assert document["dataset"]["subjects"] == SUBJECTS
    assert document["protocol"]["folds"] == 4
    assert document["windowing"]["pure_label_windows"] == 32
    assert document["windowing"]["windows_per_subject"] == {name: 8 for name in SUBJECTS}
    for backend in ("random_forest", "lightgbm"):
        report = document["backends"][backend]
        assert sorted(report["folds"]) == SUBJECTS
        assert sum(fold["window_count"] for fold in report["folds"].values()) == 32
        assert report["pooled"]["window_count"] == 32
        assert 0.0 <= report["pooled"]["macro_f1"] <= 1.0
        assert 0.2 <= report["stress_vs_rest"]["threshold"] <= 0.8
    write_report(document, tmp_path / "loso")
    assert (tmp_path / "loso" / "loso_report.json").is_file()
    markdown = (tmp_path / "loso" / "loso_report.md").read_text()
    assert "leave-one-subject-out" in markdown
    with pytest.raises(FileExistsError):
        write_report(document, tmp_path / "loso")


def test_loso_folds_train_on_disjoint_subjects(wesad_training_csv, monkeypatch):
    fits = []
    real = loso_module.make_estimator

    def recording(backend, *, seed):
        estimator = real(backend, seed=seed)
        original_fit = estimator.fit

        def fit(x, y, **kwargs):
            estimator.fit = original_fit
            fits.append(len(x))
            return original_fit(x, y, **kwargs)

        estimator.fit = fit
        return estimator

    monkeypatch.setattr(loso_module, "make_estimator", recording)
    document = loso_evaluate(wesad_training_csv, backends=("random_forest",))
    # Every fold trains on exactly the other three subjects' windows.
    assert fits == [24, 24, 24, 24]
    assert document["backends"]["random_forest"]["pooled"]["window_count"] == 32


def test_select_stress_threshold_rule_and_rejection():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.55, 0.45, 0.1])
    report = select_stress_threshold(y, scores)
    assert report["threshold"] == pytest.approx(0.5, abs=1e-9)
    assert report["f1"] == pytest.approx(1.0)
    assert report["positive_windows"] == 2
    assert report["selection_rule"] == "maximum_stress_f1_tie_break_closest_to_0.5"
    with pytest.raises(ValueError):
        select_stress_threshold(np.ones(4), np.full(4, 0.9))
