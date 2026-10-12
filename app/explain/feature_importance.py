"""Evidence for the window responsible for the session's maximum score."""

from __future__ import annotations

import numpy as np

from app.pipeline.feature_engineering import FEATURE_NAMES


def explain_prediction(predictor, features, highest: int, *, top_k: int) -> dict:
    current = features.iloc[[highest]]
    contribution, base_value, raw_margin = predictor.contributions(current)
    metadata = predictor.metadata
    reference = metadata["normal_reference"]
    importance = predictor.global_importance()
    order = np.argsort(
        -np.abs(contribution) if contribution is not None else -importance, kind="stable"
    )[:top_k]
    evidence = []
    for index in order:
        name = FEATURE_NAMES[index]
        observed = float(current.iloc[0, index])
        mean, std = reference[name]["mean"], reference[name]["std"]
        value = float(contribution[index]) if contribution is not None else None
        direction = (
            "reference_deviation_only"
            if value is None
            else "raises_score"
            if value > 0
            else "lowers_score"
            if value < 0
            else "no_change"
        )
        evidence.append(
            {
                "name": name,
                "observed_value": observed,
                "reference_mean": mean,
                "reference_std": std,
                "deviation_std": float((observed - mean) / max(std, 1e-6)),
                "contribution": value,
                "direction": direction,
            }
        )
    limitations = [
        "Trained on synthetic anomaly events; outputs are not clinical diagnoses or disease probabilities.",
        "Session score is the maximum window score and increases exposure to isolated extreme windows.",
        "Feature associations do not establish causal explanations of an individual's physiology.",
        "Reference deviations use normal training windows with a standard-deviation floor of 1e-6.",
    ]
    if contribution is None:
        limitations.append(
            "RandomForest global impurity importance and reference deviations are not local model attribution."
        )
    else:
        limitations.append(
            "Local contributions are signed raw-margin units, not probability changes."
        )
    return {
        "global_importance": predictor._importance(top_k),
        "global_importance_method": "gain"
        if contribution is not None
        else "mean_decrease_impurity",
        "local_evidence": evidence,
        "method": "lightgbm_native_contribution"
        if contribution is not None
        else "global_importance_with_reference_deviation",
        "scope": "highest_risk_window",
        "highest_risk_window": highest,
        "contribution_units": "raw_margin" if contribution is not None else "not_available",
        "base_value": base_value,
        "raw_margin": raw_margin,
        "limitations": limitations,
    }
