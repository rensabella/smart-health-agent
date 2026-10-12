"""Inference accepts an already loaded predictor; no loading on the request path."""

from smarthealth.models.base import BasePredictor
from smarthealth.pipeline.feature_engineering import extract_features
from smarthealth.pipeline.preprocess import parse_csv, preprocess


def analyze_csv(content: bytes, predictor: BasePredictor) -> dict:
    windows, quality = preprocess(parse_csv(content))
    features = extract_features(windows)
    return {
        "prediction": predictor.predict(features),
        "explanation": predictor.explain(features),
        "data_quality": quality,
    }
