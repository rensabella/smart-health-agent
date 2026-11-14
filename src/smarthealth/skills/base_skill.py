"""Explicit typed contracts, independent of any LLM provider or Agent framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from smarthealth.runtime.context import SkillContext


class SkillModel(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class ValidationIssue(SkillModel):
    category: Annotated[str, Field(pattern=r"^[a-z_]{1,64}$")]
    field_path: Annotated[str, Field(max_length=256)]
    expected: Annotated[str, Field(max_length=400)]
    actual_type: Literal[
        "missing", "null", "boolean", "integer", "number", "string", "array", "object", "other"
    ]


class InputValidationError(SkillModel):
    issues: Annotated[list[ValidationIssue], Field(min_length=1, max_length=4)]
    truncated: bool = False
    retryable: bool = True
    remaining_corrections: Annotated[int, Field(ge=0, le=1)] | None = None


class SkillFailure(SkillModel):
    code: str
    message: str
    validation_error: InputValidationError | None = None
    # Bounded, sanitized context: stable reason/rule codes and reference field names only.
    details: dict[str, Any] | None = None


class SkillResult(SkillModel):
    status: Literal["success", "error"]
    data: dict[str, Any] | None = None
    error: SkillFailure | None = None
    warnings: list[str] = Field(default_factory=list)
    skill: str
    version: str | None = None
    run_id: str
    call_id: str
    latency_ms: float = Field(ge=0)


InputT = TypeVar("InputT", bound=SkillModel)
OutputT = TypeVar("OutputT", bound=SkillModel)


class BaseSkill(ABC, Generic[InputT, OutputT]):
    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str] = "1.0"
    input_schema: type[InputT]
    output_schema: type[OutputT]
    timeout_seconds: ClassVar[float] = 30.0

    @abstractmethod
    def execute(self, input: InputT, context: SkillContext) -> OutputT:
        """Execute one bounded action. Async overrides are also supported by Registry."""
        raise NotImplementedError
