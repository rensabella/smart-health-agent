"""Explain server-produced predictions without accepting untrusted model outputs."""

from typing import Annotated, Literal

from pydantic import Field

from smarthealth.runtime.context import SkillContext
from smarthealth.schemas.response import Explanation, Ref
from smarthealth.services.analysis_service import AnalysisService
from smarthealth.skills.base_skill import BaseSkill, SkillModel


class FeatureExplanationInput(SkillModel):
    prediction_ref: Ref
    top_k: Annotated[int, Field(ge=1, le=20)] = 5


class FeatureExplanationOutput(Explanation):
    schema_version: Literal["1.0"] = "1.0"
    analysis_ref: Ref


class FeatureExplanationSkill(BaseSkill[FeatureExplanationInput, FeatureExplanationOutput]):
    name = "feature_explanation"
    description = "Return global importance and highest-risk-window evidence with explicit attribution limits."
    input_schema = FeatureExplanationInput
    output_schema = FeatureExplanationOutput

    def __init__(self, service: AnalysisService):
        self.service = service

    def execute(
        self, input: FeatureExplanationInput, context: SkillContext
    ) -> FeatureExplanationOutput:
        return FeatureExplanationOutput.model_validate(
            self.service.explain(input.prediction_ref, context, input.top_k)
        )
