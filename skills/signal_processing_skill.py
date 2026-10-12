"""Clean validated data and split it into fixed physiological windows."""

from typing import Literal

from pydantic import JsonValue

from app.core.context import SkillContext
from app.schemas.response import Ref
from app.services.analysis_service import AnalysisService
from skills.base_skill import BaseSkill, SkillModel


class SignalProcessingInput(SkillModel):
    dataset_ref: Ref
    processing_profile: Literal["1hz_30s_v1"] = "1hz_30s_v1"


class SignalProcessingOutput(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    window_set_ref: Ref
    window_count: int
    cleaning_report: dict[str, JsonValue]
    warnings: list[str]


class SignalProcessingSkill(BaseSkill[SignalProcessingInput, SignalProcessingOutput]):
    name = "signal_processing"
    description = "Clean short signal gaps and create non-overlapping 30-second windows at 1 Hz."
    input_schema = SignalProcessingInput
    output_schema = SignalProcessingOutput

    def __init__(self, service: AnalysisService):
        self.service = service

    def execute(
        self, input: SignalProcessingInput, context: SkillContext
    ) -> SignalProcessingOutput:
        return SignalProcessingOutput.model_validate(
            self.service.process_signal(input.dataset_ref, context)
        )
