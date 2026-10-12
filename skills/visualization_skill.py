"""Render bounded PNG artifacts from authenticated analysis lineage."""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from app.core.context import SkillContext
from app.schemas.response import PlotArtifact, Ref
from app.services.analysis_service import AnalysisService
from skills.base_skill import BaseSkill, SkillModel


class VisualizationInput(SkillModel):
    analysis_ref: Ref
    plot_types: Annotated[
        list[Literal["signals", "risk", "importance"]], Field(min_length=1, max_length=3)
    ] = ["signals", "risk", "importance"]

    @field_validator("plot_types")
    @classmethod
    def unique_types(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("Plot types must be unique.")
        return values


class VisualizationOutput(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    artifacts: list[PlotArtifact]


class VisualizationSkill(BaseSkill[VisualizationInput, VisualizationOutput]):
    name = "visualization"
    description = "Render signal trends, window risk scores, and feature importance as protected PNG artifacts."
    input_schema = VisualizationInput
    output_schema = VisualizationOutput
    timeout_seconds = 45.0

    def __init__(self, service: AnalysisService):
        self.service = service

    def execute(self, input: VisualizationInput, context: SkillContext) -> VisualizationOutput:
        return VisualizationOutput.model_validate(
            self.service.visualize(input.analysis_ref, input.plot_types, context)
        )
