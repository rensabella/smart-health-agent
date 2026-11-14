"""LightGBM gain importance and native signed raw-margin contributions."""

import numpy as np

from smarthealth.models.base import BasePredictor


class LightGBMPredictor(BasePredictor):
    def global_importance(self) -> np.ndarray:
        return self.estimator.booster_.feature_importance(importance_type="gain")

    def contributions(self, features):
        """Signed raw-margin contributions for the risk class of one window.

        Binary models return one block of (features..., bias); multiclass
        models return one block per class — stacked as ``(width, classes)`` or
        flattened class-major depending on the LightGBM version — so the
        risk-class block is selected explicitly.
        """
        values = np.asarray(self.estimator.booster_.predict(features, pred_contrib=True))[0]
        raw = np.asarray(self.estimator.booster_.predict(features, raw_score=True))[0]
        column = self._risk_column()
        width = len(self._metadata["feature_names"]) + 1
        if values.ndim == 2:
            values, raw = values[:, column], raw[column]
        elif len(values) != width:
            values, raw = values[column * width : (column + 1) * width], raw[column]
        return values[:-1], float(values[-1]), float(raw)
