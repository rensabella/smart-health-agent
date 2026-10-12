import json

import numpy as np
import pytest

from app.core.errors import SkillError
from app.models.model_loader import load_model
from app.pipeline.feature_engineering import extract_features
from app.pipeline.preprocess import parse_csv, preprocess
from generate_demo_data import generate_dataset


@pytest.mark.parametrize("backend", ["random_forest", "lightgbm"])
def test_roundtrip_real_inference(bundles, csv_bytes, backend):
    predictor = load_model(bundles[backend])
    features = extract_features(preprocess(parse_csv(csv_bytes))[0])
    result = predictor.predict(features)
    scores = predictor.estimator.predict_proba(features)[:, 1]
    assert result["risk_score"] == float(scores.max())
    assert result["highest_risk_window"] == int(np.argmax(scores))
    assert result == load_model(bundles[backend]).predict(features)
    explanation = predictor.explain(features, top_k=len(features.columns))
    if backend == "lightgbm":
        reconstructed = explanation["base_value"] + sum(
            e["contribution"] for e in explanation["local_evidence"]
        )
        assert reconstructed == pytest.approx(explanation["raw_margin"], abs=1e-8)
        assert 1 / (1 + np.exp(-reconstructed)) == pytest.approx(result["risk_score"])
    else:
        assert all(e["contribution"] is None for e in explanation["local_evidence"])
    with pytest.raises(SkillError):
        predictor.predict(features[features.columns[::-1]])


def test_training_subjects_disjoint_and_metadata_real(bundles):
    for path in bundles.values():
        metadata = json.loads((path / "metadata.json").read_text())
        groups = [set(s) for s in metadata["subject_splits"].values()]
        assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
        assert metadata["metrics"]["test"]["window_count"] > 0
        assert metadata["threshold_source"] == "maximum_validation_f1"
        assert "label" not in metadata["feature_names"]


def test_synthetic_generation_reproducible():
    a = generate_dataset(subjects=2, sessions=1, windows=2, seed=83)
    b = generate_dataset(subjects=2, sessions=1, windows=2, seed=83)
    assert a.equals(b)
    assert a.groupby("session_id").sleep_quality.nunique().eq(1).all()


def test_corrupted_model_refused(bundles, tmp_path):
    import shutil

    root = tmp_path / "bundle"
    shutil.copytree(bundles["random_forest"], root)
    with (root / "estimator.joblib").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(SkillError) as error:
        load_model(root)
    assert error.value.code == "model_unavailable"
