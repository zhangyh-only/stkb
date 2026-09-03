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


ClaimKind = Literal[
    "fact",
    "list",
    "process",
    "rule",
    "comparison",
    "customer_signal",
    "method",
    "strategy",
    "script",
    "objection",
    "qa",
    "term",
    "case",
    "asset",
    "value_proposition",
    "evaluation",
    "benchmark",
]


class ClaimEvidence(ApiModel):
    anchor_id: str
    exact_quote: str = Field(min_length=2)
    selector: str | None = None
    source_text: str = ""

    @field_validator("selector", mode="before")
    @classmethod
    def blank_selector_is_absent(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value


class AtomicClaim(ApiModel):
    claim_id: str
    claim_kind: ClaimKind
    statement: str = Field(min_length=2)
    subject: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    module_hints: list[str] = Field(default_factory=list)
    evidence: list[ClaimEvidence] = Field(min_length=1)


class RejectedAtomicClaim(ApiModel):
    claim_id: str
    reasons: list[str]
    raw_claim: dict[str, Any]


class CandidateObjectPlan(ApiModel):
    plan_id: str
    title: str = ""
    domain: str
    module: str
    object_type: str
    object_boundary: str = ""
    classification_basis: str = ""
    identity_hints: dict[str, Any] = Field(default_factory=dict)
    source_claim_ids: list[str] = Field(min_length=1)


class ContentClaimUsage(ApiModel):
    claim_id: str
    role: Literal["primary", "supporting"]
    content_paths: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=2)


class RejectedObjectPlan(ApiModel):
    plan_id: str
    reasons: list[str]
    raw_plan: dict[str, Any]


class CandidateKnowledgeObject(ApiModel):
    candidate_id: str
    title: str = ""
    domain: str
    module: str
    object_type: str
    object_boundary: str = ""
    classification_basis: str = ""
    identity_hints: dict[str, Any] = Field(default_factory=dict)
    planned_source_claim_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    claim_usage: list[ContentClaimUsage] = Field(default_factory=list)
    content_leaf_count: int = 0
    attributed_content_leaf_count: int = 0
    unattributed_content_paths: list[str] = Field(default_factory=list)
    quality_issues: list[str] = Field(default_factory=list)
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
    field: Literal[
        "domain",
        "entity_mentions",
        "relations",
        "content.expressions",
        "content.items",
        "content.resolutionElements",
        "content.attributionPruning",
        "claimUsage",
    ]
    original_value: Any
    normalized_value: Any
    reason: str


CoverageStatus = Literal["hit", "weak_signal", "not_found", "unresolved"]


class WeakSignal(ApiModel):
    claim_id: str | None = None
    module: str
    reason: str
    evidence: list[str] = Field(min_length=1)


class UnresolvedItem(ApiModel):
    claim_id: str | None = None
    description: str
    reason: str
    evidence: list[str] = Field(default_factory=list)
    module: str | None = None


class ModelCallTrace(ApiModel):
    call_id: str = ""
    stage_id: str = ""
    retry_of: str | None = None
    attempt: int
    purpose: Literal[
        "identification",
        "claim_discovery",
        "object_planning",
        "content_realization",
        "object_formation",
        "output_limit_retry",
        "repair",
    ]
    status: Literal["completed", "failed"]
    duration_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    raw_output: str | None = None
    finish_reason: str | None = None
    segment: str | None = None


class ProcessingStage(ApiModel):
    key: str
    name: str
    status: Literal["completed", "failed"]
    duration_ms: int
    detail: str
    actor: Literal["model", "code"] = "code"
    model_call_ids: list[str] = Field(default_factory=list)


class ObjectGranularityMetrics(ApiModel):
    object_count: int = 0
    single_claim_object_count: int = 0
    single_claim_object_rate: float = 0.0
    average_claims_per_object: float = 0.0
    source_anchors_split_across_objects: int = 0


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


class GoldGroupEvaluation(ApiModel):
    key: str
    expected_count: int
    minimum_expected_count: int = 0
    maximum_expected_count: int | None = None
    predicted_count: int
    matched_count: int
    status: Literal[
        "met", "missed", "under_split_or_recall", "over_split", "contract_failed"
    ]
    predicted_candidate_ids: list[str]
    required_item_count: int | None = None
    predicted_item_count: int | None = None
    required_content_fields: list[str] = Field(default_factory=list)
    missing_content_fields: list[str] = Field(default_factory=list)
    required_item_fields: list[str] = Field(default_factory=list)
    missing_item_fields: list[str] = Field(default_factory=list)
    required_unresolved_evidence: list[str] = Field(default_factory=list)
    missing_unresolved_evidence: list[str] = Field(default_factory=list)
    require_all_evidence: bool = False
    missing_expected_evidence: list[str] = Field(default_factory=list)


class IdentificationQualityReport(ApiModel):
    gold_version: str
    gold_status: str
    overall_status: Literal["pass", "fail", "review"]
    expected_object_count: int
    matched_expected_count: int
    object_recall_proxy: float
    groups_met: int
    group_count: int
    summary_only_count: int
    evidence_backed_rate: float
    claim_consumption_rate: float
    claim_accounting_rate: float = 0.0
    content_attribution_rate: float = 0.0
    median_content_chars: int
    groups: list[GoldGroupEvaluation]
    findings: list[str]


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
    granularity_metrics: ObjectGranularityMetrics = Field(
        default_factory=ObjectGranularityMetrics
    )
    atomic_claims: list[AtomicClaim] = Field(default_factory=list)
    rejected_atomic_claims: list[RejectedAtomicClaim] = Field(default_factory=list)
    object_plans: list[CandidateObjectPlan] = Field(default_factory=list)
    rejected_object_plans: list[RejectedObjectPlan] = Field(default_factory=list)
    candidates: list[CandidateKnowledgeObject]
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
    quality_report: IdentificationQualityReport | None = None
    storage_impact: StorageImpact = Field(default_factory=StorageImpact)


class ResolvedBusinessEntity(ApiModel):
    entity_id: str
    entity_type: str
    canonical_name: str
    source_mentions: list[str]
    action: Literal["created", "reused"]


class KnowledgeObjectEntityReference(ApiModel):
    entity_id: str
    reference_role: str
    evidence: list[str]


class KnowledgeObjectSourceTrace(ApiModel):
    candidate_id: str
    source_claim_ids: list[str]
    claim_usage: list[ContentClaimUsage]
    content_leaf_count: int
    attributed_content_leaf_count: int
    unattributed_content_paths: list[str]


class KnowledgeObjectRevisionProposal(ApiModel):
    title: str
    identity_key: str
    content_fingerprint: str
    content: dict[str, Any]
    entity_references: list[KnowledgeObjectEntityReference]
    evidence: list[str]
    source_traces: list[KnowledgeObjectSourceTrace]
    changed_paths: list[str]


class FormalKnowledgeObject(ApiModel):
    knowledge_object_id: str
    revision: int
    action: Literal["created", "updated", "reused", "review_required"]
    title: str
    domain: str
    module: str
    object_type: str
    identity_key: str
    source_lineage_keys: list[str]
    content_fingerprint: str
    content: dict[str, Any]
    entity_references: list[KnowledgeObjectEntityReference]
    evidence: list[str]
    source_candidate_ids: list[str]
    source_traces: list[KnowledgeObjectSourceTrace]
    revision_proposal: KnowledgeObjectRevisionProposal | None = None
    equivalence_reason: str | None = None
    file_path: str
    file_sha256: str


class FormalKnowledgeRelationship(ApiModel):
    relationship_id: str
    relation_type: str
    source_ref: str
    source_kind: Literal["knowledge_object", "business_entity"]
    source_revision: int | None = None
    target_ref: str
    target_kind: Literal["knowledge_object", "business_entity"]
    target_revision: int | None = None
    direction: Literal["forward"] = "forward"
    inverse_label: str
    scope: dict[str, Any] = Field(default_factory=dict)
    effective_period: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(min_length=1)
    status: Literal["active"] = "active"
    provenance: dict[str, str]


class KnowledgeFormationStage(ApiModel):
    key: Literal[
        "entity_resolution",
        "knowledge_merge",
        "formal_write",
        "pgvector_projection",
        "neo4j_projection",
    ]
    name: str
    status: Literal["completed", "pending", "failed"]
    detail: str
    duration_ms: int = 0


class KnowledgeStorageEvidence(ApiModel):
    postgres_objects: int = 0
    formal_files: int = 0
    pgvector_records: int = 0
    neo4j_knowledge_objects: int = 0
    neo4j_entities: int = 0
    neo4j_relationships: int = 0
    neo4j_document_links: int = 0
    neo4j_entity_references: int = 0
    neo4j_knowledge_relationships: int = 0
    embedding_model: str = ""
    embedding_tokens: int = 0
    vector_duration_ms: int = 0
    graph_duration_ms: int = 0
    errors: list[str] = Field(default_factory=list)


class KnowledgeFormationResult(ApiModel):
    build_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    document_package_id: str
    status: Literal["completed", "review_required", "failed"] = "completed"
    entities: list[ResolvedBusinessEntity]
    knowledge_objects: list[FormalKnowledgeObject]
    relationships: list[FormalKnowledgeRelationship] = Field(default_factory=list)
    stages: list[KnowledgeFormationStage]
    created_count: int
    updated_count: int
    reused_count: int
    review_required_count: int = 0
    superseded_count: int = 0
    quality_blocked_candidate_ids: list[str] = Field(default_factory=list)
    quality_blocked_count: int = 0
    formal_knowledge_files: int
    storage_evidence: KnowledgeStorageEvidence = Field(
        default_factory=KnowledgeStorageEvidence
    )
