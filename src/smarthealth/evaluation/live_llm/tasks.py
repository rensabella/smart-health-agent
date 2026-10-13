"""Versioned benchmark contracts, separate from provider-visible task requests."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from smarthealth.config.resources import resource_path

SKILLS = frozenset(
    {
        "csv_analysis",
        "signal_processing",
        "feature_extraction",
        "anomaly_prediction",
        "feature_explanation",
        "visualization",
        "rag_retrieval",
        "report_generation",
        "safety_verification",
    }
)
FaultKind = Literal[
    "none",
    "backend_transient",
    "backend_invalid_json",
    "tool_timeout",
    "invalid_output",
    "empty_retrieval",
    "stale_artifact",
    "duplicate_execution",
    "invalid_arguments",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Expected(Contract):
    statuses: list[Literal["completed", "awaiting_input", "blocked", "failed", "cancelled"]] = (
        Field(min_length=1)
    )
    result_kind: Literal["data_analysis", "score_definition", "health_report"] | None = None
    required_artifact_kinds: list[str] = Field(default_factory=list)
    require_plots: bool = False
    expected_plot_types: list[Literal["signals", "risk", "importance"]] = Field(
        default_factory=list
    )
    require_citations: bool = False
    compare_prediction: bool = False
    no_publication: bool = False
    fault_must_trigger: bool = False


class BenchmarkTask(Contract):
    task_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    category: Literal["basic", "workflow", "rag", "failure", "safety", "unsupported"]
    user_request: str = Field(min_length=1, max_length=4000)
    fixture: Literal["none", "synthetic_csv", "invalid_csv"]
    expected_behavior: str = Field(min_length=1, max_length=2000)
    expected: Expected
    required_skills: list[str] = Field(default_factory=list)
    permitted_skills: list[str] = Field(default_factory=lambda: sorted(SKILLS))
    forbidden_actions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)
    evaluation_method: Literal["deterministic_runtime_artifacts"] = (
        "deterministic_runtime_artifacts"
    )
    max_execution_steps: int = Field(default=32, ge=1, le=64)
    max_tool_calls: int = Field(default=18, ge=1, le=32)
    timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    include_visualizations: bool = False
    requires_rag: bool = False
    fault: FaultKind = "none"
    fault_target: str = "csv_analysis"

    @model_validator(mode="after")
    def consistent(self):
        required, permitted = set(self.required_skills), set(self.permitted_skills)
        if not required <= permitted <= SKILLS or self.fault_target not in SKILLS:
            raise ValueError("Unknown or inconsistent Skill permissions")
        if required & set(self.forbidden_actions):
            raise ValueError("Required Skills cannot be forbidden")
        if self.expected.fault_must_trigger != (self.fault != "none"):
            raise ValueError("Fault expectations must match the scenario")
        if self.expected.require_citations and not self.requires_rag:
            raise ValueError("Citation cases require the real retrieval assets")
        return self


class BenchmarkSuite(Contract):
    schema_version: Literal["1.0"] = "1.0"
    version: Literal["live-synthetic-v1"] = "live-synthetic-v1"
    tasks: list[BenchmarkTask] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def unique_tasks(self):
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("Benchmark task IDs must be unique")
        return self


def load_tasks() -> tuple[BenchmarkSuite, str]:
    # Only reviewed packaged requests and generated fixtures are accepted by the
    # CLI. There is deliberately no arbitrary patient CSV or query input option.
    data = Path(resource_path("evaluation", "live_llm_tasks.json")).read_bytes()
    suite = BenchmarkSuite.model_validate(json.loads(data), strict=True)
    return suite, hashlib.sha256(data).hexdigest()
