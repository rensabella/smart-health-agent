"""Produce the model's fixed, versioned physiological feature matrix."""

from typing import Literal

from smarthealth.runtime.context import SkillContext
from smarthealth.schemas.response import Ref
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.base_skill import BaseSkill, SkillModel


class FeatureExtractionInput(SkillModel):
    window_set_ref: Ref
    feature_profile: Literal["physiology_statistics_v1"] = "physiology_statistics_v1"


class FeatureExtractionOutput(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    feature_set_ref: Ref
    feature_schema_version: str
    feature_names: list[str]
    window_count: int


class FeatureExtractionSkill(BaseSkill[FeatureExtractionInput, FeatureExtractionOutput]):
    name = "feature_extraction"
    description = "Extract versioned window statistics from validated physiological signals."
    input_schema = FeatureExtractionInput
    output_schema = FeatureExtractionOutput

    def __init__(self, service: AnalysisService):
        self.service = service

    def execute(
        self, input: FeatureExtractionInput, context: SkillContext
    ) -> FeatureExtractionOutput:
        return FeatureExtractionOutput.model_validate(
            self.service.extract(input.window_set_ref, context)
        )
