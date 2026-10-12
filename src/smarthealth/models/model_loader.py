"""Load only administrator-provisioned local bundles, never uploaded pickle files.

The SHA-256 manifest detects corruption; it does not authenticate pickle content.
Only trusted, locally trained model directories may be configured by the service.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np

from smarthealth.models.base import BasePredictor
from smarthealth.models.lightgbm_model import LightGBMPredictor
from smarthealth.models.random_forest_model import RandomForestPredictor
from smarthealth.pipeline.feature_engineering import (
    FEATURE_CONFIG,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from smarthealth.pipeline.preprocess import INPUT_PROFILE, PROCESSING_CONFIG
from smarthealth.runtime.errors import SkillError


def load_model(bundle_dir: str | Path) -> BasePredictor:
    """Load a bundle once during application startup; callers retain the instance."""
    root = Path(bundle_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        meta_bytes = (root / "metadata.json").read_bytes()
        model_bytes = (root / "estimator.joblib").read_bytes()
        if manifest["format_version"] != "1.0" or any(
            hashlib.sha256(value).hexdigest() != manifest["sha256"][name]
            for name, value in (("metadata.json", meta_bytes), ("estimator.joblib", model_bytes))
        ):
            raise ValueError("integrity")
        metadata = json.loads(meta_bytes)
        if (
            metadata["feature_names"] != list(FEATURE_NAMES)
            or metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION
            or metadata["input_profile"] != INPUT_PROFILE
            or metadata["processing_config"] != PROCESSING_CONFIG
            or metadata["feature_config"] != FEATURE_CONFIG
            or metadata["label_mapping"] != {"0": "normal", "1": "anomaly"}
            or metadata["score_semantics"] != "synthetic_anomaly_model_score"
            or not 0 < metadata["threshold"] < 1
        ):
            raise ValueError("contract")
        if metadata["backend"] not in ("lightgbm", "random_forest"):
            raise ValueError("backend")
        for package, version in metadata["dependencies"].items():
            if (
                package in ("scikit-learn", "lightgbm", "numpy", "joblib")
                and importlib.metadata.version(package) != version
            ):
                raise ValueError("dependency")
        if any(name not in metadata["normal_reference"] for name in FEATURE_NAMES):
            raise ValueError("reference")
        # Deserialize exactly the bytes whose checksum was verified (no reopen race).
        import io

        estimator = joblib.load(io.BytesIO(model_bytes))
        if list(estimator.feature_names_in_) != list(FEATURE_NAMES) or not np.array_equal(
            estimator.classes_, np.array([0, 1])
        ):
            raise ValueError("estimator contract")
        cls = LightGBMPredictor if metadata["backend"] == "lightgbm" else RandomForestPredictor
        return cls(estimator, metadata)
    except SkillError:
        raise
    except Exception as error:
        raise SkillError(
            "model_unavailable", "Model bundle is missing, corrupt, or incompatible."
        ) from error
