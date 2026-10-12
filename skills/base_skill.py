"""Explicit typed contracts, independent of any LLM provider or Agent framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.core.context import SkillContext


class SkillModel(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class SkillFailure(SkillModel):
    code: str
    message: str


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
