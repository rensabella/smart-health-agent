"""Authoritative in-process execution state; raw inputs remain ephemeral."""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from smarthealth.agents.contracts import CreateRunRequest
from smarthealth.agents.router import RouteDecision
from smarthealth.runtime.context import SkillContext
from smarthealth.runtime.execution_trace import ExecutionTrace
from smarthealth.runtime.memory.schemas import MemoryContext


def utcnow():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunState:
    request: CreateRunRequest
    context: SkillContext
    execution_trace: ExecutionTrace = field(init=False, repr=False)
    request_id: str
    backend_mode: str
    decision_model: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    status: str = "queued"
    decision_count: int = 0
    llm_requests: int = 0
    delegation_count: int = 0
    data_tool_calls: int = 0
    role_tool_calls: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    token_usage_available: bool = False
    final_result: dict | None = None
    error_code: str | None = None
    message: str | None = None
    finished_at: float | None = None
    refs: dict[str, list[str]] = field(default_factory=dict)
    observations: deque = field(default_factory=deque)
    plots: dict[str, list[dict]] = field(default_factory=dict)
    quality_reports: dict[str, dict] = field(default_factory=dict)
    historical_tool_calls: int | None = None
    usage_missing: bool = False
    restored: bool = False
    route: RouteDecision | None = None
    initial_workflow_type: str | None = None
    workflow_upgrades: int = 0
    started_at: float = field(default_factory=time.monotonic)
    result_expires_at: float | None = None
    memory_lease: str | None = None
    memory_context: MemoryContext = field(default_factory=MemoryContext)
    memory_status: str = "disabled"
    harness_version: str | None = None
    harness_hash: str | None = None

    def __post_init__(self):
        self.execution_trace = ExecutionTrace(self.context.run_id)
