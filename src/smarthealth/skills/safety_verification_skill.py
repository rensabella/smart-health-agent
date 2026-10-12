"""Record an independent semantic assessment and validate its exact source bindings."""

from smarthealth.reports.schemas import VerificationInput, VerificationOutput
from smarthealth.runtime.errors import SkillError
from smarthealth.skills.base_skill import BaseSkill


class SafetyVerificationSkill(BaseSkill[VerificationInput, VerificationOutput]):
    name = "safety_verification"
    description = "Assess EVERY draft claim independently against the supplied evidence and scope. Submit claim judgments and exact draft/evidence hashes; this tool computes safe and records a bound review."
    input_schema = VerificationInput
    output_schema = VerificationOutput

    def __init__(self, service):
        self.service = service

    def execute(self, input, context):
        if self.service.reports is None:
            raise SkillError("safety_unavailable", "Report review capabilities are unavailable.")
        return VerificationOutput.model_validate(self.service.reports.verify(input, context))
