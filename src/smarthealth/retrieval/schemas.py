"""Frozen source, chunk and retrieval contracts; evidence is data, never authority."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from datetime import date
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator, model_validator

from smarthealth.skills.base_skill import SkillModel

Version = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$", max_length=64)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
EmbeddingId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,199}$")]


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def canonical_hash(value) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    )


def validate_source_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.isascii()
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or len(value) > 1024
        or any(ord(char) <= 32 for char in value)
        or "." not in parsed.hostname
    ):
        raise ValueError("A fixed public HTTPS source URL is required")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return value
    raise ValueError("Source URLs cannot use IP addresses")


class FrozenRagModel(SkillModel):
    model_config = ConfigDict(frozen=True)


class EvidenceSource(FrozenRagModel):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,79}$")]
    title: Annotated[str, Field(min_length=1, max_length=200)]
    url: Annotated[str, Field(min_length=1, max_length=1024)]
    source_type: Literal["signal_reference", "sensor_documentation", "guideline", "research_paper"]

    _url = field_validator("url")(validate_source_url)


class SourceDocument(FrozenRagModel):
    source: EvidenceSource
    locator: Annotated[str, Field(min_length=1, max_length=200)]
    content: Annotated[str, Field(min_length=1, max_length=4000)]
    content_hash: Digest
    accessed_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    rights: Annotated[str, Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def valid_document(self):
        date.fromisoformat(self.accessed_at)
        if not self.content.strip() or sha256(self.content) != self.content_hash:
            raise ValueError("Document content hash does not match")
        if any(ord(char) < 32 and char not in "\n\t" for char in self.content):
            raise ValueError("Document contains unsupported control characters")
        return self


class KnowledgeCorpus(FrozenRagModel):
    schema_version: Literal["1.0"]
    knowledge_version: Version
    allowed_source_urls: Annotated[tuple[str, ...], Field(max_length=64)]
    documents: Annotated[tuple[SourceDocument, ...], Field(max_length=64)]

    @field_validator("allowed_source_urls", "documents", mode="before")
    @classmethod
    def immutable_sequence(cls, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Expected explicit sequence")
        return tuple(value)

    @model_validator(mode="after")
    def source_allowlist(self):
        if len(self.allowed_source_urls) != len(set(self.allowed_source_urls)):
            raise ValueError("Duplicate allowlisted source")
        for url in self.allowed_source_urls:
            validate_source_url(url)
        if set(self.allowed_source_urls) != {doc.source.url for doc in self.documents}:
            raise ValueError("Corpus sources must match the explicit URL allowlist")
        ids = [doc.source.id for doc in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate document source identity")
        return self


class KnowledgeChunk(FrozenRagModel):
    evidence_id: Annotated[str, Field(pattern=r"^ev_[0-9a-f]{64}$")]
    source: EvidenceSource
    content: Annotated[str, Field(min_length=1, max_length=768)]
    locator: Annotated[str, Field(min_length=1, max_length=240)]
    content_hash: Digest
    knowledge_version: Version

    @model_validator(mode="after")
    def content_matches(self):
        if sha256(self.content) != self.content_hash:
            raise ValueError("Evidence content hash does not match")
        expected = evidence_id(
            self.source.id, self.locator, self.content_hash, self.knowledge_version
        )
        if self.evidence_id != expected:
            raise ValueError("Evidence identity does not match provenance")
        return self


def evidence_id(source_id: str, locator: str, content_hash: str, knowledge_version: str) -> str:
    return "ev_" + canonical_hash([source_id, locator, content_hash, knowledge_version])


class Evidence(KnowledgeChunk):
    score: Annotated[float, Field(ge=-1, le=1)]
    score_type: Literal["cosine"] = "cosine"


class RetrievalResult(FrozenRagModel):
    status: Literal["success", "insufficient_evidence"]
    hits: Annotated[tuple[Evidence, ...], Field(max_length=8)]
    knowledge_version: Version
    embedding_id: EmbeddingId

    @field_validator("hits", mode="before")
    @classmethod
    def immutable_hits(cls, value):
        if not isinstance(value, (tuple, list)):
            raise ValueError("Hits must be an explicit sequence")
        return tuple(value)

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "success") != bool(self.hits):
            raise ValueError("Retrieval status disagrees with evidence")
        if any(hit.knowledge_version != self.knowledge_version for hit in self.hits):
            raise ValueError("Mixed knowledge versions")
        if len({hit.evidence_id for hit in self.hits}) != len(self.hits):
            raise ValueError("Duplicate evidence")
        return self
