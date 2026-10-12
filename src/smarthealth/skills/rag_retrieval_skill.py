"""Query a versioned local index; retain only server-owned, scoped evidence references."""

from typing import Annotated, Literal

from pydantic import Field

from smarthealth.runtime.errors import SkillError
from smarthealth.skills.base_skill import BaseSkill, SkillModel


class RetrievalInput(SkillModel):
    analysis_ref: Annotated[str, Field(pattern=r"^art_[0-9a-f]{32}$")]
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    top_k: Annotated[int, Field(ge=1, le=6)] = 4


class RetrievalOutput(SkillModel):
    status: Literal["success", "insufficient_evidence"]
    evidence_ref: Annotated[str, Field(pattern=r"^art_[0-9a-f]{32}$")]
    hit_count: Annotated[int, Field(ge=0, le=6)]
    knowledge_version: str
    embedding_id: str


class RAGRetrievalSkill(BaseSkill[RetrievalInput, RetrievalOutput]):
    name = "rag_retrieval"
    description = "Retrieve sourced educational signal context from the configured local knowledge index; similarity is not diagnostic confidence."
    input_schema = RetrievalInput
    output_schema = RetrievalOutput

    def __init__(self, service):
        self.service = service

    def execute(self, input, context):
        self.service.store.get(input.analysis_ref, kind="analysis", context=context)
        if self.service.retrieval is None:
            raise SkillError("retrieval_unavailable", "Configure a verified local knowledge index.")
        result = self.service.retrieval.retrieve(input.query, top_k=input.top_k)
        context.check_active()
        evidence_ref = self.service.store.put(
            {"analysis_ref": input.analysis_ref, "result": result.model_dump(mode="json")},
            kind="evidence",
            context=context,
            source_ref=input.analysis_ref,
        )
        return RetrievalOutput(
            status=result.status,
            evidence_ref=evidence_ref,
            hit_count=len(result.hits),
            knowledge_version=result.knowledge_version,
            embedding_id=result.embedding_id,
        )
