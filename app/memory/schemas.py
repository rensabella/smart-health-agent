"""Memory payload allowlists. Raw queries, measurements and reports have no fields here."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from skills.base_skill import SkillModel

MemoryIdentifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
MemoryRef = Annotated[str, Field(pattern=r"^art_[0-9a-f]{32}$")]
Version = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")]
MemoryObjective = Literal["data_analysis", "score_definition", "health_report"]
MemoryStatus = Literal[
    "queued",
    "running",
    "awaiting_input",
    "verifying",
    "completed",
    "blocked",
    "failed",
    "cancelled",
]


class Preferences(SkillModel):
    language: Literal["zh-CN", "en"] = "zh-CN"
    detail: Literal["brief", "standard"] = "standard"


class ScoreProtocol(SkillModel):
    model_name: Version
    model_version: Version
    model_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    input_profile: Version
    feature_schema_version: Version
    task_scope: Literal["synthetic_anomaly_only"] = "synthetic_anomaly_only"
    window_seconds: Literal[30] = 30
    aggregation: Literal["max"] = "max"
    score_semantics: Literal["synthetic_anomaly_model_score"] = "synthetic_anomaly_model_score"


class ScoreSummary(ScoreProtocol):
    prediction: Literal["normal", "anomaly"]
    risk_score: Annotated[float, Field(ge=0, le=1)]
    risk_level: Literal["low", "medium", "high"]
    window_count: Annotated[int, Field(ge=1, le=100000)]


class TrendPoint(SkillModel):
    source_run_id: MemoryIdentifier
    analyzed_at: Annotated[float, Field(ge=0)]
    risk_score: Annotated[float, Field(ge=0, le=1)]
    risk_level: Literal["low", "medium", "high"]
    window_count: Annotated[int, Field(ge=1, le=100000)]


class TrendGroup(SkillModel):
    group_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    protocol: ScoreProtocol
    points: Annotated[list[TrendPoint], Field(max_length=512)]


class TrendResponse(SkillModel):
    historical_only: Literal[True] = True
    groups: Annotated[list[TrendGroup], Field(max_length=512)]


class LongTermEntry(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["analysis", "preferences"]
    source_run_id: MemoryIdentifier | None = None
    created_at: Annotated[float, Field(ge=0)]
    expires_at: Annotated[float, Field(gt=0)]
    analysis: ScoreSummary | None = None
    preferences: Preferences | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.expires_at <= self.created_at:
            raise ValueError("Memory retention must be positive")
        if self.kind == "analysis":
            if self.source_run_id is None or self.analysis is None or self.preferences is not None:
                raise ValueError("An analysis entry requires approved score facts and provenance")
        elif (
            self.preferences is None or self.analysis is not None or self.source_run_id is not None
        ):
            raise ValueError("A preference entry contains explicit preferences only")
        return self


class MemoryHit(SkillModel):
    entry: LongTermEntry
    score: Annotated[float, Field(ge=-1, le=1)]
    score_type: Literal["cosine_lexical_hash"] = "cosine_lexical_hash"


class TurnSummary(SkillModel):
    run_id: MemoryIdentifier
    input_version: MemoryIdentifier
    user_intent: MemoryObjective
    assistant_status: MemoryStatus
    completed_at: Annotated[float, Field(ge=0)]
    analysis: ScoreSummary | None = None


class ExecutionSummary(SkillModel):
    run_id: MemoryIdentifier
    status: MemoryStatus
    last_event_id: Annotated[int, Field(ge=0)] = 0
    decision_count: Annotated[int, Field(ge=0)] = 0
    tool_calls: Annotated[int, Field(ge=0)] = 0


class SessionMemory(SkillModel):
    session_id: MemoryIdentifier
    input_version: MemoryIdentifier
    upload_refs: Annotated[list[MemoryRef], Field(max_length=1)] = Field(default_factory=list)
    current_result_run_id: MemoryIdentifier | None = None
    prediction: ScoreSummary | None = None
    execution: ExecutionSummary | None = None
    turns: Annotated[list[TurnSummary], Field(max_length=5)] = Field(default_factory=list)
    expires_at: Annotated[float, Field(gt=0)]


class MemoryContext(SkillModel):
    """Background context only: no usable artifact references or arbitrary instructions."""

    recent_turns: Annotated[list[TurnSummary], Field(max_length=5)] = Field(default_factory=list)
    preferences: Preferences | None = None
    historical_summaries: Annotated[list[MemoryHit], Field(max_length=3)] = Field(
        default_factory=list
    )
    historical_only: Literal[True] = True


class MemorySearchRequest(SkillModel):
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    limit: Annotated[int, Field(ge=1, le=10)] = 3

    @model_validator(mode="after")
    def nonblank(self):
        if not self.query.strip():
            raise ValueError("A nonblank search query is required")
        return self


def render_entry(entry: LongTermEntry) -> str:
    """Render from typed facts; never use a model/user supplied narrative for embedding."""
    entry = LongTermEntry.model_validate(entry.model_dump())
    if entry.kind == "preferences":
        return f"Explicit report preferences 语言 {entry.preferences.language} detail {entry.preferences.detail}"
    a = entry.analysis
    level = {"low": "低", "medium": "中", "high": "高"}[a.risk_level]
    return (
        f"Synthetic physiological analysis 合成生理异常分析 {a.prediction}; "
        f"risk 风险 {a.risk_level} {level}; score 分数 {a.risk_score:.3f}; "
        f"windows {a.window_count}; model {a.model_name} {a.model_version}. "
        "Model score only; not a clinical disease probability. 不是临床患病概率。"
    )
