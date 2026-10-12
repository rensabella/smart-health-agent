"""Versioned public contracts shared by HTTP responses and individual Skills."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from skills.base_skill import SkillModel

Ref = Annotated[str, Field(min_length=1, max_length=128)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class ImportantFeature(SkillModel):
    name: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]


class ModelInfo(SkillModel):
    name: str
    version: str
    task_scope: str = "synthetic_physiology_v1"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class WindowPrediction(SkillModel):
    window_index: Annotated[int, Field(ge=0)]
    risk_score: Score
    prediction: Literal["anomaly", "normal"]


class LocalEvidence(SkillModel):
    name: str
    observed_value: float
    reference_mean: float
    reference_std: float
    deviation_std: float
    contribution: float | None
    direction: Literal["raises_score", "lowers_score", "no_change", "reference_deviation_only"]


class Prediction(SkillModel):
    prediction: Literal["anomaly", "normal"]
    risk_score: Score
    important_features: list[ImportantFeature]
    model: ModelInfo
    aggregation: Literal["max"] = "max"
    window_count: Annotated[int, Field(ge=1)]
    anomalous_ratio: Score
    highest_risk_window: Annotated[int, Field(ge=0)]
    window_predictions: list[WindowPrediction]
    risk_level: Literal["low", "medium", "high"]


class Explanation(SkillModel):
    global_importance: list[ImportantFeature]
    global_importance_method: Literal["gain", "mean_decrease_impurity"]
    local_evidence: list[LocalEvidence]
    method: str
    limitations: list[str]
    highest_risk_window: Annotated[int, Field(ge=0)]
    scope: Literal["highest_risk_window"] = "highest_risk_window"
    contribution_units: Literal["raw_margin", "not_available"] = "not_available"
    base_value: float | None = None
    raw_margin: float | None = None


class PlotArtifact(SkillModel):
    artifact_ref: Ref
    kind: Literal["signals", "risk", "importance"]
    mime_type: Literal["image/png"]
    url: str


class PredictResponse(Prediction):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    session_id: str
    input_version: str
    prediction_ref: Ref
    quality_report: dict[str, JsonValue]
    warnings: list[str]


class AnalyzeResponse(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    session_id: str
    input_version: str
    analysis_ref: Ref
    summary: str
    risk_level: Literal["low", "medium", "high"]
    prediction: Prediction
    explanation: Explanation
    visualization: list[PlotArtifact]
    quality_report: dict[str, JsonValue]
    warnings: list[str]


class UploadResponse(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    session_id: str
    input_version: str
    upload_ref: Ref
    size_bytes: int


class ErrorDetail(SkillModel):
    code: str
    message: str


class ErrorResponse(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    error: ErrorDetail
