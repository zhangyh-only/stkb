from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
        extra="forbid",
    )


class SourceAnchor(ApiModel):
    anchor_id: str
    kind: Literal["page", "section", "table", "paragraph", "time_range"]
    page: int | None = None


class DocumentPackage(ApiModel):
    document_package_id: str
    workspace_id: str
    source_file_name: str
    source_file_path: str = ""
    source_sha256: str
    full_markdown_path: str
    full_markdown_sha256: str
    full_markdown: str
    processing_method: Literal["agent_assisted", "capability"]
    status: Literal["available", "unavailable"]
    anchors: list[SourceAnchor]
    quality_issues: list[str]


class SourceMaterial(ApiModel):
    document_package_id: str
    source_file_name: str
    source_file_path: str
    source_sha256: str
    processing_method: Literal["agent_assisted", "capability"]
    status: Literal["available", "unavailable"]


class ModelRequest(ApiModel):
    document_package_id: str
    system_prompt: str
    user_prompt: str


class ModelCompletion(ApiModel):
    provider: str
    model: str
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class EntityMention(ApiModel):
    mention_id: str
    text: str
    proposed_type: str
    reference_role: str
    source_ref: str


class ProposedRelation(ApiModel):
    relation_kind: Literal["entity", "object"]
    relation_type: str
    source_ref: str
    target_ref: str
    evidence: list[str] = Field(min_length=1)


class CandidateKnowledge(ApiModel):
    candidate_id: str
    domain: str
    module: str
    object_type: str
    content: dict[str, Any]
    entity_mentions: list[EntityMention] = Field(default_factory=list)
    evidence: list[str] = Field(min_length=1)
    relations: list[ProposedRelation] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("content must not be empty")
        return value


class RejectedCandidate(ApiModel):
    candidate_id: str
    reasons: list[str]
    raw_candidate: dict[str, Any]


class RejectedAuxiliaryItem(ApiModel):
    kind: Literal["weak_signal", "unresolved_item"]
    reasons: list[str]
    raw_item: dict[str, Any]


class CandidateNormalization(ApiModel):
    candidate_id: str
    field: Literal["domain"]
    original_value: str
    normalized_value: str
    reason: str


CoverageStatus = Literal["hit", "weak_signal", "not_found", "unresolved"]


class WeakSignal(ApiModel):
    module: str
    reason: str
    evidence: list[str] = Field(min_length=1)


class UnresolvedItem(ApiModel):
    description: str
    reason: str
    evidence: list[str] = Field(default_factory=list)
    module: str | None = None


class ModelCallTrace(ApiModel):
    attempt: int
    purpose: Literal["identification", "output_limit_retry", "repair"]
    status: Literal["completed", "failed"]
    duration_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    raw_output: str | None = None
    finish_reason: str | None = None
    segment: str | None = None


class ProcessingStage(ApiModel):
    key: str
    name: str
    status: Literal["completed", "failed"]
    duration_ms: int
    detail: str


class StorageImpact(ApiModel):
    postgres_run_records: int = 1
    formal_knowledge_files: int = 0
    pgvector_records: int = 0
    neo4j_nodes: int = 0
    neo4j_relationships: int = 0


class ModelConfigurationSnapshot(ApiModel):
    temperature: float
    max_output_tokens: int
    timeout_seconds: int
    max_retries: int
    max_candidates: int
    enable_thinking: bool
    document_max_chars: int
    max_concurrency: int
    fingerprint: str


class IdentificationResult(ApiModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    document_package_id: str
    status: Literal["completed", "failed"] = "completed"
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int = 0
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    catalog_version: str
    catalog_fingerprint: str
    raw_model_output: str
    model_calls: list[ModelCallTrace]
    processing_stages: list[ProcessingStage]
    candidates: list[CandidateKnowledge]
    rejected_candidates: list[RejectedCandidate]
    rejected_auxiliary_items: list[RejectedAuxiliaryItem] = Field(default_factory=list)
    normalizations: list[CandidateNormalization] = Field(default_factory=list)
    weak_signals: list[WeakSignal]
    unresolved_items: list[UnresolvedItem]
    coverage_by_module: dict[str, CoverageStatus]
    call_count: int
    prompt_tokens: int
    completion_tokens: int
    model_configuration: ModelConfigurationSnapshot | None = None
    storage_impact: StorageImpact = Field(default_factory=StorageImpact)
