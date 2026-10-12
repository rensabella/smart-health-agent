"""Typed reference labels loaded separately from model-visible decision inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from smarthealth.retrieval.files import read_json
from smarthealth.retrieval.schemas import Digest
from smarthealth.skills.base_skill import SkillModel


class ExpectedRun(SkillModel):
    workflow_type: Literal["simple", "complex"]
    objective: Literal["score_definition", "data_analysis", "health_report"]
    status: Literal["completed", "awaiting_input", "blocked", "failed"]
    result_kind: Literal["score_definition", "data_analysis", "health_report"] | None
    tool_calls_min: int = Field(ge=0, le=18)
    tool_calls_max: int = Field(ge=0, le=18)
    required_scope: str | None
    forbidden_claim_categories: list[str]

    @model_validator(mode="after")
    def bounded(self):
        if self.tool_calls_min > self.tool_calls_max:
            raise ValueError("Invalid tool call interval")
        return self


class SystemCase(SkillModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    query: str = Field(min_length=1, max_length=4000)
    input: Literal["none", "demo_csv", "invalid_csv"]
    include_visualizations: bool
    harness_profile: Literal["report", "tool"]
    retrieval_enabled: bool
    expected: ExpectedRun
    tags: list[str]


class RetrievalCase(SkillModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(ge=1, le=6)
    expected_source_ids: list[str] = Field(min_length=1, max_length=8)


class ExpectedClaim(SkillModel):
    supported: bool
    within_scope: bool
    no_diagnosis: bool
    reason: Literal[
        "supported",
        "insufficient_support",
        "medical_diagnosis",
        "causal_overstatement",
        "unsafe_instruction",
        "out_of_scope",
    ]
    should_publish: bool


class SafetyCase(SkillModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    claim: str = Field(min_length=1, max_length=800)
    cited_source_ids: list[str] = Field(max_length=4)
    evidence_source_ids: list[str] = Field(min_length=1, max_length=6)
    expected: ExpectedClaim
    rationale: str = Field(max_length=2000)
    fabricated_evidence_id: str | None = Field(default=None, pattern=r"^ev_[0-9a-f]{64}$")

    @model_validator(mode="after")
    def citation_labels_are_unambiguous(self):
        # This first suite labels one claim/source pair per case. Multi-citation
        # cases need per-source support labels rather than broadcasting one flag.
        if len(self.cited_source_ids) + int(self.fabricated_evidence_id is not None) != 1:
            raise ValueError("Safety cases require exactly one labeled citation")
        if not set(self.cited_source_ids) <= set(self.evidence_source_ids):
            raise ValueError("A labeled citation must be in provided evidence")
        return self


class CorpusIdentity(SkillModel):
    path: Literal["knowledge/corpus.json"]
    knowledge_version: str
    sha256: Digest


class EvaluationCases(SkillModel):
    schema_version: Literal["1.0"]
    suite_version: str
    corpus: CorpusIdentity
    system_cases: list[SystemCase] = Field(min_length=1, max_length=32)
    retrieval_cases: list[RetrievalCase] = Field(min_length=1, max_length=32)
    safety_cases: list[SafetyCase] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [
            c.id
            for group in (self.system_cases, self.retrieval_cases, self.safety_cases)
            for c in group
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation IDs must be unique")
        return self


def load_cases(path: Path, *, root=Path(".")) -> EvaluationCases:
    suite = EvaluationCases.model_validate(read_json(path, max_bytes=256 * 1024), strict=True)
    corpus_path = root / suite.corpus.path
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != suite.corpus.sha256:
        raise ValueError("Corpus changed: review evaluation labels before running")
    corpus = read_json(corpus_path, max_bytes=512 * 1024)
    if corpus["knowledge_version"] != suite.corpus.knowledge_version:
        raise ValueError("Knowledge version mismatch")
    known = {doc["source"]["id"] for doc in corpus["documents"]}
    if any(not set(c.expected_source_ids) <= known for c in suite.retrieval_cases) or any(
        not (set(c.cited_source_ids) | set(c.evidence_source_ids)) <= known
        for c in suite.safety_cases
    ):
        raise ValueError("Evaluation cites an unknown corpus source")
    return suite
