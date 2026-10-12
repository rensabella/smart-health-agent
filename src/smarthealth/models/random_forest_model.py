"""RandomForest baseline; reference deviations are not local attribution."""

import numpy as np

from smarthealth.models.base import BasePredictor


class RandomForestPredictor(BasePredictor):
    def global_importance(self) -> np.ndarray:
        return np.asarray(self.estimator.feature_importances_)
