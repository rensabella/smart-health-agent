"""One scoring contract for independently trained prediction backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd

from smarthealth.pipeline.feature_engineering import FEATURE_NAMES
from smarthealth.runtime.errors import SkillError


class BasePredictor(ABC):
    def __init__(self, estimator, metadata: dict):
        self.estimator = estimator
        self._metadata = deepcopy(metadata)

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    def _validate_features(self, features: pd.DataFrame) -> None:
        if (
            not isinstance(features, pd.DataFrame)
            or features.empty
            or tuple(features.columns) != FEATURE_NAMES
        ):
            raise SkillError(
                "invalid_features", "Feature names and order must match the model schema."
            )
        try:
            valid = np.isfinite(features.to_numpy(dtype=float)).all()
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise SkillError("invalid_features", "Model features must be finite numeric values.")

    @abstractmethod
    def global_importance(self) -> np.ndarray:
        """Return nonnegative importance in feature order."""

    def contributions(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray | None, float | None, float | None]:
        return None, None, None

    def _scores(self, features: pd.DataFrame) -> np.ndarray:
        self._validate_features(features)
        try:
            scores = np.asarray(self.estimator.predict_proba(features)[:, 1], dtype=float)
        except Exception as error:
            raise SkillError(
                "prediction_failed", "The model could not score this feature set."
            ) from error
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise SkillError("prediction_failed", "The model produced an invalid anomaly score.")
        return scores

    def _importance(self, top_k: int = 5) -> list[dict]:
        importance = np.asarray(self.global_importance(), dtype=float)
        total = importance.sum()
        importance = importance / total if total > 0 else np.zeros(len(FEATURE_NAMES))
        order = np.argsort(-importance, kind="stable")[:top_k]
        return [
            {"name": FEATURE_NAMES[index], "importance": float(importance[index])}
            for index in order
        ]

    def predict(self, features: pd.DataFrame) -> dict:
        scores = self._scores(features)
        threshold = self._metadata["threshold"]
        highest = int(np.argmax(scores))
        score = float(scores[highest])
        levels = self._metadata["risk_thresholds"]
        model_fields = (
            "name",
            "version",
            "task_scope",
            "input_schema_version",
            "feature_schema_version",
            "input_profile",
        )
        return {
            "prediction": "anomaly" if score >= threshold else "normal",
            "risk_score": score,
            "risk_level": "high"
            if score >= levels["high"]
            else "medium"
            if score >= levels["medium"]
            else "low",
            "important_features": self._importance(),
            "model": {key: self._metadata[key] for key in model_fields},
            "aggregation": "max",
            "window_count": len(scores),
            "anomalous_ratio": float(np.mean(scores >= threshold)),
            "highest_risk_window": highest,
            "window_predictions": [
                {
                    "window_index": index,
                    "risk_score": float(value),
                    "prediction": "anomaly" if value >= threshold else "normal",
                }
                for index, value in enumerate(scores)
            ],
        }

    def explain(self, features: pd.DataFrame, *, top_k: int = 5) -> dict:
        from smarthealth.explain.feature_importance import explain_prediction

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= len(FEATURE_NAMES)
        ):
            raise SkillError(
                "invalid_input", "Requested feature count is outside the supported range."
            )
        scores = self._scores(features)
        return explain_prediction(self, features, int(np.argmax(scores)), top_k=top_k)
