"""Validate a server-owned CSV upload and create a typed dataset artifact."""

from typing import Literal

from pydantic import JsonValue

from app.core.context import SkillContext
from app.schemas.response import Ref
from app.services.analysis_service import AnalysisService
from skills.base_skill import BaseSkill, SkillModel


class CSVAnalysisInput(SkillModel):
    upload_ref: Ref
    data_profile: Literal["synthetic_physiology_v1"] = "synthetic_physiology_v1"


class CSVAnalysisOutput(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_ref: Ref
    columns: list[str]
    row_count: int
    quality_report: dict[str, JsonValue]


class CSVAnalysisSkill(BaseSkill[CSVAnalysisInput, CSVAnalysisOutput]):
    name = "csv_analysis"
    description = "Validate the five-signal synthetic physiology CSV and report data quality."
    input_schema = CSVAnalysisInput
    output_schema = CSVAnalysisOutput

    def __init__(self, service: AnalysisService):
        self.service = service

    def execute(self, input: CSVAnalysisInput, context: SkillContext) -> CSVAnalysisOutput:
        return CSVAnalysisOutput.model_validate(self.service.inspect_csv(input.upload_ref, context))
