"""One scoring contract for independently trained prediction backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd

from smarthealth.pipeline.profiles import DataProfile, get_profile
from smarthealth.runtime.errors import SkillError


class BasePredictor(ABC):
    def __init__(self, estimator, metadata: dict):
        self.estimator = estimator
        self._metadata = deepcopy(metadata)
        self._profile = get_profile(self._metadata["input_profile"])

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    @property
    def profile(self) -> DataProfile:
        return self._profile

    def _validate_features(self, features: pd.DataFrame) -> None:
        if (
            not isinstance(features, pd.DataFrame)
            or features.empty
            or tuple(features.columns) != tuple(self._metadata["feature_names"])
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

    def _risk_column(self) -> int:
        """Position of the risk class inside the estimator's class ordering."""
        return list(self.estimator.classes_).index(self._profile.risk_class_index())

    def _label_for(self, probabilities: np.ndarray, risk_score: float) -> str:
        """Threshold-first rule: at or above the threshold the window is the risk class.

        Below the threshold a binary model stays "normal"; a multiclass model
        falls back to the strongest non-risk class. Binary behavior is exactly
        the historical rule.
        """
        if risk_score >= self._metadata["threshold"]:
            return self._profile.risk_class
        risk_column = self._risk_column()
        others = [index for index in range(len(probabilities)) if index != risk_column]
        if not others:
            return self._profile.risk_class
        label_column = others[int(np.argmax(probabilities[others]))]
        label = int(self.estimator.classes_[label_column])
        return self._profile.label_mapping[str(label)]

    def _probabilities(self, features: pd.DataFrame) -> np.ndarray:
        self._validate_features(features)
        try:
            probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)
        except Exception as error:
            raise SkillError(
                "prediction_failed", "The model could not score this feature set."
            ) from error
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or probabilities.shape != (len(features), len(self.estimator.classes_))
        ):
            raise SkillError("prediction_failed", "The model produced an invalid anomaly score.")
        return probabilities

    def _scores(self, features: pd.DataFrame) -> np.ndarray:
        return self._probabilities(features)[:, self._risk_column()]

    def _importance(self, top_k: int = 5) -> list[dict]:
        names = self._metadata["feature_names"]
        importance = np.asarray(self.global_importance(), dtype=float)
        total = importance.sum()
        importance = importance / total if total > 0 else np.zeros(len(names))
        order = np.argsort(-importance, kind="stable")[:top_k]
        return [{"name": names[index], "importance": float(importance[index])} for index in order]

    def predict(self, features: pd.DataFrame) -> dict:
        probabilities = self._probabilities(features)
        scores = probabilities[:, self._risk_column()]
        threshold = self._metadata["threshold"]
        highest = int(np.argmax(scores))
        score = float(scores[highest])
        levels = self._metadata["risk_thresholds"]
        window_predictions = [
            {
                "window_index": index,
                "risk_score": float(value),
                "prediction": self._label_for(probabilities[index], float(value)),
            }
            for index, value in enumerate(scores)
        ]
        model_fields = (
            "name",
            "version",
            "task_scope",
            "input_schema_version",
            "feature_schema_version",
            "input_profile",
        )
        return {
            "prediction": window_predictions[highest]["prediction"],
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
            "window_predictions": window_predictions,
        }

    def explain(self, features: pd.DataFrame, *, top_k: int = 5) -> dict:
        from smarthealth.explain.feature_importance import explain_prediction

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= len(self._metadata["feature_names"])
        ):
            raise SkillError(
                "invalid_input", "Requested feature count is outside the supported range."
            )
        scores = self._scores(features)
        return explain_prediction(self, features, int(np.argmax(scores)), top_k=top_k)
