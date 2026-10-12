"""Invoke the already-loaded baseline through a stable prediction contract."""

from typing import Literal

from app.core.context import SkillContext
from app.schemas.response import Prediction, Ref
from app.services.analysis_service import AnalysisService
from skills.base_skill import BaseSkill, SkillModel


class AnomalyPredictionInput(SkillModel):
    feature_set_ref: Ref


class AnomalyPredictionOutput(Prediction):
    schema_version: Literal["1.0"] = "1.0"
    prediction_ref: Ref


class AnomalyPredictionSkill(BaseSkill[AnomalyPredictionInput, AnomalyPredictionOutput]):
    name = "anomaly_prediction"
    description = "Score synthetic physiological anomaly windows using a cached baseline model; scores are not medical probabilities."
    input_schema = AnomalyPredictionInput
    output_schema = AnomalyPredictionOutput

    def __init__(self, service: AnalysisService):
        self.service = service

    def execute(
        self, input: AnomalyPredictionInput, context: SkillContext
    ) -> AnomalyPredictionOutput:
        return AnomalyPredictionOutput.model_validate(
            self.service.predict(input.feature_set_ref, context)
        )
