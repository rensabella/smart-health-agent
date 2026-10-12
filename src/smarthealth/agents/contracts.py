"""Provider-independent decisions; no free-form model text can become a published report."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from smarthealth.agents.router import RouteDecision
from smarthealth.runtime.memory.schemas import MemoryContext
from smarthealth.skills.base_skill import SkillModel

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
ArtifactRef = Annotated[str, Field(pattern=r"^art_[0-9a-f]{32}$")]
Role = Literal["planner", "data_analysis", "rag", "report_writer", "safety_verification"]
Objective = Literal["data_analysis", "score_definition", "health_report"]
ResultKind = Literal[
    "data_analysis", "score_definition", "health_report", "evidence", "draft", "review"
]
RunStatus = Literal[
    "queued",
    "running",
    "awaiting_input",
    "verifying",
    "completed",
    "blocked",
    "failed",
    "cancelled",
]
TERMINAL_STATUSES = frozenset({"awaiting_input", "completed", "blocked", "failed", "cancelled"})


class CallSkillAction(SkillModel):
    action: Literal["call_skill"] = "call_skill"
    skill: Annotated[str, Field(min_length=1, max_length=64)]
    arguments: dict[str, JsonValue]


class DelegateAction(SkillModel):
    action: Literal["delegate"] = "delegate"
    agent: Annotated[str, Field(min_length=1, max_length=64)]
    task: Annotated[str, Field(min_length=1, max_length=2000)]


class FinishAction(SkillModel):
    action: Literal["finish"] = "finish"
    result_kind: ResultKind
    result_ref: ArtifactRef | None = None


class RequestInputAction(SkillModel):
    action: Literal["request_input"] = "request_input"
    reason: Literal["missing_upload", "invalid_data", "clarify_task"]


class UpgradeWorkflowAction(SkillModel):
    action: Literal["upgrade_workflow"] = "upgrade_workflow"
    objective: Literal["data_analysis", "health_report"]


AgentAction = Annotated[
    CallSkillAction | DelegateAction | FinishAction | RequestInputAction | UpgradeWorkflowAction,
    Field(discriminator="action"),
]
ACTION_ADAPTER = TypeAdapter(AgentAction)


class Observation(SkillModel):
    step_id: str
    agent: Role
    action: str
    status: Literal["success", "error"]
    skill: str | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)
    error_code: str | None = None


class DecisionRequest(SkillModel):
    run_id: Identifier
    agent: Role
    task: Annotated[str, Field(max_length=4000)]
    objective: Objective
    include_visualizations: bool
    available_refs: dict[str, list[str]]
    observations: list[Observation]
    tools: list[dict[str, JsonValue]]
    allowed_delegates: list[
        Literal["data_analysis", "rag", "report_writer", "safety_verification"]
    ] = Field(default_factory=list, max_length=4)
    remaining_tool_calls: int
    remaining_decisions: int
    workflow_type: Literal["simple", "complex"] | None = None
    can_upgrade_workflow: bool = False
    memory: MemoryContext = Field(default_factory=MemoryContext)
    report_context: dict[str, JsonValue] = Field(default_factory=dict)


class DecisionReply(SkillModel):
    decision: AgentAction
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None


class CreateRunRequest(SkillModel):
    query: Annotated[str, Field(min_length=1, max_length=4000)]
    session_id: Identifier
    input_version: Identifier
    upload_refs: Annotated[list[ArtifactRef], Field(max_length=1)] = Field(default_factory=list)
    objective: Objective = "data_analysis"
    entry_agent: Literal["planner", "data_analysis"] = "data_analysis"
    include_visualizations: bool = False
    parent_run_id: Identifier | None = None
    source_run_id: Identifier | None = None

    @model_validator(mode="after")
    def nonblank_query(self):
        if not self.query.strip():
            raise ValueError("Query must be nonblank")
        return self


class RunSnapshot(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: Identifier
    request_id: str
    session_id: Identifier
    input_version: Identifier
    parent_run_id: Identifier | None
    status: RunStatus
    backend_mode: str
    decision_model: str | None = None
    objective: Objective
    entry_agent: Role
    created_at: str
    updated_at: str
    decision_count: int
    llm_requests: int
    tool_calls: int
    delegation_count: int
    input_tokens: int
    output_tokens: int
    token_usage_available: bool
    last_event_id: int
    final_result: dict[str, JsonValue] | None = None
    result_available: bool = False
    error_code: str | None = None
    message: str | None = None
    events_url: str
    route: RouteDecision | None = None
    initial_workflow_type: Literal["simple", "complex"] | None = None
    workflow_upgrades: Annotated[int, Field(ge=0, le=1)] = 0
    source_run_id: Identifier | None = None
    memory_status: Literal[
        "disabled", "active", "pending", "stored", "skipped", "unavailable", "unknown"
    ] = "disabled"
    harness_version: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")] | None = None
    harness_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


class RunEvent(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: int
    run_id: Identifier
    timestamp: str
    event: str
    agent: str | None = None
    step_id: str | None = None
    parent_step_id: str | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)
