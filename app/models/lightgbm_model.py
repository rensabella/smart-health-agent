"""LightGBM gain importance and native signed raw-margin contributions."""

import numpy as np

from app.models.base import BasePredictor


class LightGBMPredictor(BasePredictor):
    def global_importance(self) -> np.ndarray:
        return self.estimator.booster_.feature_importance(importance_type="gain")

    def contributions(self, features):
        contributions = np.asarray(self.estimator.booster_.predict(features, pred_contrib=True))[0]
        raw_margin = float(np.asarray(self.estimator.booster_.predict(features, raw_score=True))[0])
        return contributions[:-1], float(contributions[-1]), raw_margin
