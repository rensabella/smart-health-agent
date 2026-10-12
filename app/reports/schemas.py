"""Bounded report contracts. Only server-produced facts can contain numerical results."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from app.rag.schemas import EvidenceSource
from skills.base_skill import SkillModel

ArtifactRef = Annotated[str, Field(pattern=r"^art_[0-9a-f]{32}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ClaimId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,47}$")]
EvidenceId = Annotated[str, Field(pattern=r"^ev_[0-9a-f]{64}$")]
ReviewReason = Literal[
    "supported",
    "insufficient_support",
    "medical_diagnosis",
    "causal_overstatement",
    "unsafe_instruction",
    "out_of_scope",
]


class ClaimDraft(SkillModel):
    claim_id: ClaimId
    text: Annotated[str, Field(min_length=1, max_length=800)]
    evidence_ids: Annotated[list[EvidenceId], Field(min_length=1, max_length=4)]

    @field_validator("text")
    @classmethod
    def qualitative_plain_text(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        if not value.strip() or value != value.strip():
            raise ValueError("Claims must be nonblank plain text")
        if any(unicodedata.category(char).startswith("C") for char in normalized):
            raise ValueError("Claims cannot contain control or formatting characters")
        if re.search(r"[\d<>\[\]`]|https?://|www\.", normalized, re.IGNORECASE):
            raise ValueError("Claims are qualitative text without numerical facts or markup")
        if re.search(
            r"[零〇一二两三四五六七八九十百千万亿]+(?:点|成|倍|秒|分钟|小时|毫克|克|岁|次|%)"
            r"|百分之|千分之|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"hundred|thousand)\s+(?:percent|seconds?|minutes?|hours?|times?|milligrams?)\b",
            normalized,
            re.IGNORECASE,
        ):
            raise ValueError("Model-supplied numerical facts cannot enter claim text")
        return value

    @model_validator(mode="after")
    def unique_evidence(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Claim evidence references must be unique")
        return self


class GenerationInput(SkillModel):
    analysis_ref: ArtifactRef
    evidence_ref: ArtifactRef
    claims: Annotated[list[ClaimDraft], Field(min_length=1, max_length=6)]
    previous_draft_ref: ArtifactRef | None = None
    previous_review_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def valid_revision(self):
        if bool(self.previous_draft_ref) != bool(self.previous_review_ref):
            raise ValueError("Revision requires its exact failed draft and review")
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("Claim identifiers must be unique")
        return self


class GenerationOutput(SkillModel):
    draft_ref: ArtifactRef
    draft_version: Annotated[int, Field(ge=1, le=3)]
    draft_hash: Digest
    evidence_hash: Digest
    claim_count: Annotated[int, Field(ge=1, le=6)]


class ClaimReview(SkillModel):
    claim_id: ClaimId
    supported: bool
    within_scope: bool
    no_diagnosis: bool
    reason: ReviewReason


class VerificationInput(SkillModel):
    draft_ref: ArtifactRef
    draft_hash: Digest
    evidence_hash: Digest
    reviews: Annotated[list[ClaimReview], Field(min_length=1, max_length=6)]


class VerificationOutput(SkillModel):
    review_ref: ArtifactRef
    draft_ref: ArtifactRef
    draft_hash: Digest
    safe: bool
    warnings: Annotated[list[str], Field(max_length=48)]
    revision_needed: bool


class DraftArtifact(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    draft_version: Annotated[int, Field(ge=1, le=3)]
    draft_hash: Digest
    evidence_hash: Digest
    analysis_fingerprint: Digest
    policy_hash: Digest
    analysis_ref: ArtifactRef
    evidence_ref: ArtifactRef
    claims: Annotated[list[ClaimDraft], Field(min_length=1, max_length=6)]
    previous_draft_ref: ArtifactRef | None = None
    previous_review_ref: ArtifactRef | None = None


class ReviewArtifact(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    reviewer_role: Literal["safety_verification"] = "safety_verification"
    draft_ref: ArtifactRef
    draft_version: Annotated[int, Field(ge=1, le=3)]
    draft_hash: Digest
    evidence_hash: Digest
    analysis_fingerprint: Digest
    policy_hash: Digest
    review_hash: Digest
    reviews: Annotated[list[ClaimReview], Field(min_length=1, max_length=6)]
    safe: bool
    warnings: Annotated[list[str], Field(max_length=48)]
    revision_needed: bool


class PublishedSource(SkillModel):
    evidence_id: EvidenceId
    source: EvidenceSource
    locator: Annotated[str, Field(min_length=1, max_length=240)]
    content_hash: Digest
    knowledge_version: Annotated[str, Field(min_length=1, max_length=64)]


class PublishedClaim(ClaimDraft):
    sources: Annotated[list[PublishedSource], Field(min_length=1, max_length=4)]


class ReportFacts(SkillModel):
    prediction: Literal["normal", "anomaly"]
    risk_score: Annotated[float, Field(ge=0, le=1)]
    risk_level: Literal["low", "medium", "high"]
    window_count: Annotated[int, Field(ge=1)]
    highest_risk_window: Annotated[int, Field(ge=0)]
    model_name: str
    model_version: str


class PublicationSafety(SkillModel):
    safe: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)
    revision_needed: Literal[False] = False
    draft_version: Annotated[int, Field(ge=1, le=3)]
    draft_hash: Digest
    evidence_hash: Digest
    analysis_fingerprint: Digest
    policy_hash: Digest
    review_hash: Digest
    reviewer_role: Literal["safety_verification"] = "safety_verification"
    verification_mode: Literal["role_claim_review_with_deterministic_guards"] = (
        "role_claim_review_with_deterministic_guards"
    )
    clinical_validation: Literal["not_performed"] = "not_performed"


class ApprovedReport(SkillModel):
    schema_version: Literal["1.0"] = "1.0"
    analysis_ref: ArtifactRef
    evidence_ref: ArtifactRef
    draft_ref: ArtifactRef
    review_ref: ArtifactRef
    title: str
    summary: str
    analysis: ReportFacts
    claims: Annotated[list[PublishedClaim], Field(min_length=1, max_length=6)]
    uncertainty: str
    recommendations: list[str]
    scope: Literal["synthetic_analysis_with_educational_context"] = (
        "synthetic_analysis_with_educational_context"
    )
    safety: PublicationSafety
    body: str
