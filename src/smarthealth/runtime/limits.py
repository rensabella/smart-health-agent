"""Validated execution ceilings shared by configuration and the runtime."""

from typing import Annotated

from pydantic import ConfigDict, Field

from smarthealth.skills.base_skill import SkillModel


class AgentLimits(SkillModel):
    model_config = ConfigDict(frozen=True)

    max_decisions: Annotated[int, Field(ge=1, le=64)] = 32
    max_llm_requests: Annotated[int, Field(ge=1, le=64)] = 32
    max_tool_calls: Annotated[int, Field(ge=1, le=32)] = 18
    max_data_tool_calls: Annotated[int, Field(ge=1, le=32)] = 8
    max_rag_tool_calls: Annotated[int, Field(ge=1, le=8)] = 4
    max_writer_tool_calls: Annotated[int, Field(ge=1, le=3)] = 3
    max_safety_tool_calls: Annotated[int, Field(ge=1, le=3)] = 3
    max_delegations: Annotated[int, Field(ge=0, le=8)] = 8
    max_total_tokens: Annotated[int, Field(ge=1, le=1000000)] = 65536
    run_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 180.0
    decision_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 30.0
    max_active_runs: Annotated[int, Field(ge=1, le=64)] = 8
    max_retained_runs: Annotated[int, Field(ge=1, le=1024)] = 128
    retention_seconds: Annotated[float, Field(gt=0)] = 604800.0
    result_ttl_seconds: Annotated[float, Field(gt=0)] = 3600.0
    max_observations: Annotated[int, Field(ge=1, le=64)] = 32
    simple_max_decisions: Annotated[int, Field(ge=1, le=64)] = 4
    simple_max_llm_requests: Annotated[int, Field(ge=1, le=64)] = 4
    simple_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 30.0
