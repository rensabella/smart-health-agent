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
from smarthealth.pipeline.profiles import get_profile
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
        # The bundle must declare a registered input profile; unknown origins are
        # a contract failure even when every other field happens to match.
        profile = get_profile(metadata["input_profile"])
        if (
            metadata["feature_names"] != list(profile.feature_names)
            or metadata["feature_schema_version"] != profile.feature_schema_version
            or metadata["input_profile"] != profile.name
            or metadata["processing_config"] != profile.processing_config
            or metadata["feature_config"] != profile.feature_config
            or metadata["label_mapping"] != profile.label_mapping
            or metadata["score_semantics"] != profile.score_semantics
            or not 0 < metadata["threshold"] < 1
        ):
            raise ValueError("contract")
        if metadata["backend"] not in ("lightgbm", "random_forest"):
            raise ValueError("backend")
        for package, version in metadata["dependencies"].items():
            if (
                package in profile.dependency_packages
                and importlib.metadata.version(package) != version
            ):
                raise ValueError("dependency")
        if any(name not in metadata["normal_reference"] for name in profile.feature_names):
            raise ValueError("reference")
        # Deserialize exactly the bytes whose checksum was verified (no reopen race).
        import io

        estimator = joblib.load(io.BytesIO(model_bytes))
        if list(estimator.feature_names_in_) != list(profile.feature_names) or not np.array_equal(
            estimator.classes_, np.array(profile.expected_classes)
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
