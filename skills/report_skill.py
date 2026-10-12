"""Writer action stores a versioned draft; it cannot publish or approve it."""

from app.core.errors import SkillError
from app.reports.schemas import GenerationInput, GenerationOutput
from skills.base_skill import BaseSkill


class ReportSkill(BaseSkill[GenerationInput, GenerationOutput]):
    name = "report_generation"
    description = "Create a qualitative draft with real evidence IDs; revise only a previously rejected draft with its review. Publication requires an independent Safety Agent."
    input_schema = GenerationInput
    output_schema = GenerationOutput

    def __init__(self, service):
        self.service = service

    def execute(self, input, context):
        if self.service.reports is None:
            raise SkillError("safety_unavailable", "Report review capabilities are unavailable.")
        return GenerationOutput.model_validate(self.service.reports.generate(input, context))
