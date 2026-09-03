from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity_contracts import canonical_identity, validate_identity_hints
from .models import (
    CandidateKnowledgeObject,
    DocumentPackage,
    FormalKnowledgeObject,
    FormalKnowledgeRelationship,
    IdentificationResult,
    KnowledgeFormationResult,
    KnowledgeFormationStage,
    KnowledgeObjectEntityReference,
    KnowledgeObjectRevisionProposal,
    KnowledgeObjectSourceTrace,
    ResolvedBusinessEntity,
)

CONTENT_FIELD_LABELS = {
    "subject": "知识主体",
    "facts": "事实内容",
    "applicability": "适用范围",
    "limitations": "限制与边界",
    "purpose": "业务目的",
    "preconditions": "前置条件",
    "rulesOrSteps": "规则或步骤",
    "exceptions": "例外情况",
    "strategyName": "策略名称",
    "triggerConditions": "触发条件",
    "decisionLogic": "判断逻辑",
    "actions": "建议动作",
    "communicationGoal": "沟通目标",
    "script": "建议表达",
    "factReferences": "事实依据",
    "complianceConstraints": "合规边界",
    "objectionTheme": "异议主题",
    "expressions": "客户表达",
    "context": "适用情境",
    "rootConcernHypotheses": "有依据的顾虑假设",
    "resolutionElements": "回应要点",
    "items": "知识条目",
    "terms": "术语条目",
}

FORMAL_RELATION_TYPES = {
    "ABOUT",
    "APPLIES_TO",
    "SUPPORTS",
    "ADDRESSES",
    "GUIDES",
    "CONSTRAINED_BY",
    "EVALUATED_BY",
    "EXEMPLIFIED_BY",
}
INVERSE_RELATION_LABELS = {
    "ABOUT": "HAS_KNOWLEDGE",
    "APPLIES_TO": "HAS_APPLICABLE_KNOWLEDGE",
    "SUPPORTS": "SUPPORTED_BY",
    "ADDRESSES": "ADDRESSED_BY",
    "GUIDES": "GUIDED_BY",
    "CONSTRAINED_BY": "CONSTRAINS",
    "EVALUATED_BY": "EVALUATES",
    "EXEMPLIFIED_BY": "EXEMPLIFIES",
}


@dataclass(frozen=True)
class ExistingKnowledgeObjectState:
    revision: int
    content_fingerprint: str
    title: str = ""
    domain: str = ""
    module: str = ""
    object_type: str = ""
    identity_key: str = ""
    content: dict[str, Any] | None = None
    entity_references: tuple[KnowledgeObjectEntityReference, ...] = ()
    evidence: tuple[str, ...] = ()
    file_path: str = ""
    file_sha256: str = ""


class KnowledgeObjectFormationService:
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.knowledge_root = (self.project_root / "workspace/knowledge").resolve()

    def candidate_entity_ids(
        self, workspace_id: str, candidates: list[CandidateKnowledgeObject]
    ) -> set[str]:
        return {
            self._entity_id(workspace_id, mention.proposed_type, mention.text)
            for candidate in candidates
            for mention in candidate.entity_mentions
        }

    def candidate_object_ids(
        self,
        workspace_id: str,
        candidates: list[CandidateKnowledgeObject],
        existing_lineages: dict[str, str] | None = None,
    ) -> set[str]:
        existing_lineages = existing_lineages or {}
        return {
            existing_lineages.get(
                self._source_lineage_key(candidate),
                self._knowledge_object_id(
                    workspace_id,
                    candidate.object_type,
                    self._identity_key(candidate),
                ),
            )
            for candidate in candidates
        }

    def candidate_lineage_keys(
        self, candidates: list[CandidateKnowledgeObject]
    ) -> set[str]:
        return {self._source_lineage_key(candidate) for candidate in candidates}

    def form(
        self,
        *,
        document_package: DocumentPackage,
        identification: IdentificationResult,
        existing_entities: set[str],
        existing_objects: dict[str, ExistingKnowledgeObjectState],
        existing_lineages: dict[str, str] | None = None,
        existing_document_object_ids: set[str] | None = None,
    ) -> KnowledgeFormationResult:
        existing_lineages = existing_lineages or {}
        existing_document_object_ids = existing_document_object_ids or set()
        quality_blocked_candidate_ids = [
            candidate.candidate_id
            for candidate in identification.candidates
            if candidate.quality_issues
        ]
        eligible_candidates = [
            candidate
            for candidate in identification.candidates
            if not candidate.quality_issues
        ]
        entities = self._resolve_entities(
            document_package.workspace_id,
            eligible_candidates,
            existing_entities,
        )
        entity_by_mention = {
            (candidate.candidate_id, mention.mention_id): self._entity_id(
                document_package.workspace_id,
                mention.proposed_type,
                mention.text,
            )
            for candidate in eligible_candidates
            for mention in candidate.entity_mentions
        }
        groups: dict[str, list[CandidateKnowledgeObject]] = defaultdict(list)
        identity_keys: dict[str, str] = {}
        lineage_keys: dict[str, set[str]] = defaultdict(set)
        resolved_object_ids = self._unique_source_slot_matches(
            eligible_candidates,
            existing_objects,
            existing_document_object_ids,
        )
        for candidate in eligible_candidates:
            identity_key = self._identity_key(candidate)
            lineage_key = self._source_lineage_key(candidate)
            object_id = resolved_object_ids.get(
                candidate.candidate_id,
                existing_lineages.get(
                    lineage_key,
                    self._knowledge_object_id(
                        document_package.workspace_id,
                        candidate.object_type,
                        identity_key,
                    ),
                ),
            )
            groups[object_id].append(candidate)
            identity_keys[object_id] = identity_key
            lineage_keys[object_id].add(lineage_key)

        knowledge_objects = [
            self._form_object(
                object_id=object_id,
                candidates=candidates,
                identity_key=identity_keys[object_id],
                document_package=document_package,
                entity_by_mention=entity_by_mention,
                existing=existing_objects.get(object_id),
                source_lineage_keys=sorted(lineage_keys[object_id]),
            )
            for object_id, candidates in groups.items()
        ]
        object_by_candidate = {
            candidate_id: item.knowledge_object_id
            for item in knowledge_objects
            for candidate_id in item.source_candidate_ids
        }
        relationships = self._resolve_relationships(
            eligible_candidates,
            object_by_candidate,
            entity_by_mention,
            {item.knowledge_object_id: item.revision for item in knowledge_objects},
            identification.run_id,
            document_package.document_package_id,
        )
        created_count = sum(item.action == "created" for item in knowledge_objects)
        updated_count = sum(item.action == "updated" for item in knowledge_objects)
        reused_count = sum(item.action == "reused" for item in knowledge_objects)
        review_required_count = sum(
            item.action == "review_required" for item in knowledge_objects
        )
        current_object_ids = set(groups)
        identity_set_changed = bool(
            existing_document_object_ids
            and current_object_ids != existing_document_object_ids
        )
        superseded_count = len(existing_document_object_ids - current_object_ids)
        if identity_set_changed:
            knowledge_objects = [
                item.model_copy(update={"action": "review_required"})
                for item in knowledge_objects
            ]
            created_count = 0
            updated_count = 0
            reused_count = 0
            review_required_count = len(current_object_ids)
        requires_review = bool(review_required_count or quality_blocked_candidate_ids)
        if not requires_review:
            for item in knowledge_objects:
                if item.action == "created" or not item.file_sha256:
                    self._write_accepted_object(
                        item, document_package.document_package_id, relationships
                    )
        return KnowledgeFormationResult(
            run_id=identification.run_id,
            document_package_id=document_package.document_package_id,
            entities=entities,
            knowledge_objects=knowledge_objects,
            relationships=relationships,
            stages=[
                KnowledgeFormationStage(
                    key="entity_resolution",
                    name="业务实体归一",
                    status="completed",
                    detail=f"确认 {len(entities)} 个可稳定引用的业务实体",
                ),
                KnowledgeFormationStage(
                    key="knowledge_merge",
                    name="知识身份归并",
                    status="completed",
                    detail=(
                        f"匹配 {len(knowledge_objects)} 个知识身份："
                        f"待新增 {created_count}、待评审 {review_required_count}、"
                        f"复用 {reused_count}、质量阻断 "
                        f"{len(quality_blocked_candidate_ids)}"
                        if requires_review
                        else f"形成 {len(knowledge_objects)} 个正式知识身份："
                        f"新增 {created_count}、更新 {updated_count}、复用 {reused_count}"
                    ),
                ),
                KnowledgeFormationStage(
                    key="formal_write",
                    name="正式知识写入",
                    status="pending" if requires_review else "completed",
                    detail=(
                        f"有 {review_required_count} 项正文或身份集合变化，"
                        f"{len(quality_blocked_candidate_ids)} 个候选未通过追溯门槛；"
                        "评审前不改写正式知识"
                        if requires_review
                        else f"写入或校验 {len(knowledge_objects)} 份正式 Markdown"
                    ),
                ),
            ],
            status="review_required" if requires_review else "completed",
            created_count=created_count,
            updated_count=updated_count,
            reused_count=reused_count,
            review_required_count=review_required_count,
            superseded_count=superseded_count,
            quality_blocked_candidate_ids=quality_blocked_candidate_ids,
            quality_blocked_count=len(quality_blocked_candidate_ids),
            formal_knowledge_files=sum(bool(item.file_sha256) for item in knowledge_objects),
        )

    @staticmethod
    def _resolve_relationships(
        candidates: list[CandidateKnowledgeObject],
        object_by_candidate: dict[str, str],
        entity_by_mention: dict[tuple[str, str], str],
        revision_by_object: dict[str, int],
        run_id: str,
        document_package_id: str,
    ) -> list[FormalKnowledgeRelationship]:
        refs: dict[str, tuple[str, str]] = {
            candidate_id: (object_id, "knowledge_object")
            for candidate_id, object_id in object_by_candidate.items()
        }
        for candidate in candidates:
            for mention in candidate.entity_mentions:
                refs[mention.mention_id] = (
                    entity_by_mention[(candidate.candidate_id, mention.mention_id)],
                    "business_entity",
                )
        relationships: dict[str, FormalKnowledgeRelationship] = {}
        for candidate in candidates:
            for relation in candidate.relations:
                source = refs.get(relation.source_ref)
                target = refs.get(relation.target_ref)
                if (
                    source is None
                    or target is None
                    or relation.relation_type not in FORMAL_RELATION_TYPES
                ):
                    continue
                identity = "|".join(
                    (relation.relation_type, source[0], target[0])
                )
                relationship_id = (
                    "KR-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20].upper()
                )
                relationships[relationship_id] = FormalKnowledgeRelationship(
                    relationship_id=relationship_id,
                    relation_type=relation.relation_type,
                    source_ref=source[0],
                    source_kind=source[1],
                    source_revision=revision_by_object.get(source[0]),
                    target_ref=target[0],
                    target_kind=target[1],
                    target_revision=revision_by_object.get(target[0]),
                    inverse_label=INVERSE_RELATION_LABELS[relation.relation_type],
                    evidence=relation.evidence,
                    provenance={
                        "runId": run_id,
                        "documentPackageId": document_package_id,
                        "candidateId": candidate.candidate_id,
                    },
                )
        return list(relationships.values())

    def _resolve_entities(
        self,
        workspace_id: str,
        candidates: list[CandidateKnowledgeObject],
        existing_entities: set[str],
    ) -> list[ResolvedBusinessEntity]:
        grouped: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            for mention in candidate.entity_mentions:
                entity_id = self._entity_id(
                    workspace_id,
                    mention.proposed_type,
                    mention.text,
                )
                item = grouped.setdefault(
                    entity_id,
                    {
                        "entity_type": mention.proposed_type,
                        "canonical_name": mention.text.strip(),
                        "source_mentions": set(),
                    },
                )
                item["source_mentions"].add(mention.text.strip())
        return [
            ResolvedBusinessEntity(
                entity_id=entity_id,
                entity_type=item["entity_type"],
                canonical_name=item["canonical_name"],
                source_mentions=sorted(item["source_mentions"]),
                action="reused" if entity_id in existing_entities else "created",
            )
            for entity_id, item in grouped.items()
        ]

    def _form_object(
        self,
        *,
        object_id: str,
        candidates: list[CandidateKnowledgeObject],
        identity_key: str,
        document_package: DocumentPackage,
        entity_by_mention: dict[tuple[str, str], str],
        existing: ExistingKnowledgeObjectState | None,
        source_lineage_keys: list[str],
    ) -> FormalKnowledgeObject:
        primary = candidates[0]
        evidence = sorted({item for candidate in candidates for item in candidate.evidence})
        entity_references = self._entity_references(candidates, entity_by_mention)
        content = self._merge_content(candidates)
        source_candidate_ids = [candidate.candidate_id for candidate in candidates]
        source_traces = [
            KnowledgeObjectSourceTrace(
                candidate_id=candidate.candidate_id,
                source_claim_ids=candidate.source_claim_ids,
                claim_usage=candidate.claim_usage,
                content_leaf_count=candidate.content_leaf_count,
                attributed_content_leaf_count=candidate.attributed_content_leaf_count,
                unattributed_content_paths=candidate.unattributed_content_paths,
            )
            for candidate in candidates
        ]
        payload_fingerprint = self._content_fingerprint(
            title=primary.title,
            domain=primary.domain,
            module=primary.module,
            object_type=primary.object_type,
            content=content,
            entity_references=entity_references,
            evidence=evidence,
            source_traces=source_traces,
        )
        revision_proposal = None
        equivalence_reason = None
        if existing is None:
            action = "created"
            revision = 1
        elif existing.content_fingerprint == payload_fingerprint:
            action = "reused"
            revision = existing.revision
        elif self._equivalent_core_content(
            primary.object_type, existing.content or {}, content
        ):
            action = "reused"
            revision = existing.revision
            equivalence_reason = (
                "来源承载的核心内容未变化；仅顺序、主张编号或辅助描述发生漂移"
            )
            primary = primary.model_copy(
                update={
                    "title": existing.title or primary.title,
                    "domain": existing.domain or primary.domain,
                    "module": existing.module or primary.module,
                    "object_type": existing.object_type or primary.object_type,
                    "content": existing.content or content,
                }
            )
            identity_key = existing.identity_key or identity_key
            content = existing.content or content
            entity_references = list(existing.entity_references) or entity_references
            evidence = list(existing.evidence) or evidence
        else:
            action = "review_required"
            revision = existing.revision
            revision_proposal = KnowledgeObjectRevisionProposal(
                title=primary.title,
                identity_key=identity_key,
                content_fingerprint=payload_fingerprint,
                content=content,
                entity_references=entity_references,
                evidence=evidence,
                source_traces=source_traces,
                changed_paths=self._changed_paths(existing.content or {}, content),
            )
            primary = primary.model_copy(
                update={
                    "title": existing.title or primary.title,
                    "domain": existing.domain or primary.domain,
                    "module": existing.module or primary.module,
                    "object_type": existing.object_type or primary.object_type,
                    "content": existing.content or content,
                }
            )
            identity_key = existing.identity_key or identity_key
            content = existing.content or content
            entity_references = list(existing.entity_references) or entity_references
            evidence = list(existing.evidence) or evidence
        formal_fingerprint = (
            existing.content_fingerprint
            if action in {"review_required", "reused"} and existing is not None
            else payload_fingerprint
        )
        relative_path = Path(
            f"workspace/knowledge/{primary.domain}/{primary.module}/{object_id}.md"
        )
        if existing and existing.file_path:
            relative_path = Path(existing.file_path)
        file_sha256 = existing.file_sha256 if existing else ""
        return FormalKnowledgeObject(
            knowledge_object_id=object_id,
            revision=revision,
            action=action,
            title=primary.title,
            domain=primary.domain,
            module=primary.module,
            object_type=primary.object_type,
            identity_key=identity_key,
            source_lineage_keys=source_lineage_keys,
            content_fingerprint=formal_fingerprint,
            content=content,
            entity_references=entity_references,
            evidence=evidence,
            source_candidate_ids=source_candidate_ids,
            source_traces=source_traces,
            revision_proposal=revision_proposal,
            equivalence_reason=equivalence_reason,
            file_path=relative_path.as_posix(),
            file_sha256=file_sha256,
        )

    def _write_accepted_object(
        self,
        item: FormalKnowledgeObject,
        document_package_id: str,
        relationships: list[FormalKnowledgeRelationship],
    ) -> None:
        markdown = self._render_markdown(
            object_id=item.knowledge_object_id,
            revision=item.revision,
            title=item.title,
            domain=item.domain,
            module=item.module,
            object_type=item.object_type,
            identity_key=item.identity_key,
            source_lineage_keys=item.source_lineage_keys,
            content=item.content,
            entity_references=item.entity_references,
            evidence=item.evidence,
            document_package_id=document_package_id,
            source_traces=item.source_traces,
            relationships=[
                relationship
                for relationship in relationships
                if item.knowledge_object_id
                in {relationship.source_ref, relationship.target_ref}
            ],
        )
        item.file_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        self._write_atomically(Path(item.file_path), markdown)

    def _entity_references(
        self,
        candidates: list[CandidateKnowledgeObject],
        entity_by_mention: dict[tuple[str, str], str],
    ) -> list[KnowledgeObjectEntityReference]:
        references: dict[tuple[str, str], set[str]] = defaultdict(set)
        for candidate in candidates:
            for mention in candidate.entity_mentions:
                key = (
                    entity_by_mention[(candidate.candidate_id, mention.mention_id)],
                    mention.reference_role,
                )
                references[key].add(mention.source_ref)
        return [
            KnowledgeObjectEntityReference(
                entity_id=entity_id,
                reference_role=reference_role,
                evidence=sorted(evidence),
            )
            for (entity_id, reference_role), evidence in references.items()
        ]

    @staticmethod
    def _merge_content(candidates: list[CandidateKnowledgeObject]) -> dict[str, Any]:
        if len(candidates) == 1:
            return candidates[0].content
        return {"mergedItems": [candidate.content for candidate in candidates]}

    @classmethod
    def _unique_source_slot_matches(
        cls,
        candidates: list[CandidateKnowledgeObject],
        existing_objects: dict[str, ExistingKnowledgeObjectState],
        active_object_ids: set[str],
    ) -> dict[str, str]:
        candidate_groups: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
        existing_groups: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
        for candidate in candidates:
            candidate_groups[
                (candidate.module, candidate.object_type, tuple(sorted(candidate.evidence)))
            ].append(candidate.candidate_id)
        for object_id in active_object_ids:
            existing = existing_objects.get(object_id)
            if existing is None:
                continue
            existing_groups[
                (existing.module, existing.object_type, tuple(sorted(existing.evidence)))
            ].append(object_id)
        return {
            candidate_ids[0]: existing_groups[key][0]
            for key, candidate_ids in candidate_groups.items()
            if len(candidate_ids) == 1 and len(existing_groups.get(key, [])) == 1
        }

    @staticmethod
    def _identity_key(candidate: CandidateKnowledgeObject) -> str:
        errors = validate_identity_hints(
            candidate.module, candidate.identity_hints, candidate.object_type
        )
        if errors:
            raise ValueError("; ".join(errors))
        canonical = json.dumps(
            canonical_identity(
                candidate.module, candidate.identity_hints, candidate.object_type
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _source_lineage_key(cls, candidate: CandidateKnowledgeObject) -> str:
        if not candidate.evidence:
            raise ValueError("candidate source lineage requires evidence")
        primary_anchor = sorted(candidate.evidence, key=cls._natural_sort_key)[0]
        value = (
            f"{candidate.module}|{candidate.object_type}|{primary_anchor}|"
            f"{cls._identity_key(candidate)}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _natural_sort_key(value: str) -> list[tuple[int, int | str]]:
        return [
            (0, int(part)) if part.isdigit() else (1, part)
            for part in re.split(r"(\d+)", value.casefold())
        ]

    @classmethod
    def _changed_paths(cls, current: Any, proposed: Any, path: str = "$") -> list[str]:
        if isinstance(current, dict) and isinstance(proposed, dict):
            paths: list[str] = []
            for key in sorted(set(current) | set(proposed)):
                child = f"{path}.{key}"
                if key not in current or key not in proposed:
                    paths.append(child)
                else:
                    paths.extend(cls._changed_paths(current[key], proposed[key], child))
            return paths
        if isinstance(current, list) and isinstance(proposed, list):
            paths = []
            for index in range(max(len(current), len(proposed))):
                child = f"{path}[{index}]"
                if index >= len(current) or index >= len(proposed):
                    paths.append(child)
                else:
                    paths.extend(cls._changed_paths(current[index], proposed[index], child))
            return paths
        return [] if current == proposed else [path]

    @classmethod
    def _equivalent_core_content(
        cls, object_type: str, current: dict[str, Any], proposed: dict[str, Any]
    ) -> bool:
        if object_type == "STANDARD_SCRIPT":
            current_script = cls._normalize_text(current.get("script"))
            proposed_script = cls._normalize_text(proposed.get("script"))
            return bool(current_script) and current_script == proposed_script
        if object_type == "QA_PAIR":
            return cls._qa_pairs(current) == cls._qa_pairs(proposed) and bool(
                cls._qa_pairs(current)
            )
        return False

    @classmethod
    def _qa_pairs(cls, content: dict[str, Any]) -> dict[str, str]:
        pairs: dict[str, str] = {}
        for item in content.get("items", []):
            if not isinstance(item, dict):
                continue
            question = cls._normalize_text(item.get("question"))
            answer = cls._normalize_text(item.get("answer"))
            if question and answer:
                pairs[question] = answer
        return pairs

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", "", value).casefold()

    @staticmethod
    def _entity_id(workspace_id: str, entity_type: str, text: str) -> str:
        canonical_name = re.sub(r"\s+", " ", text.strip()).casefold()
        value = f"{workspace_id}|{entity_type}|{canonical_name}"
        return f"BE-{hashlib.sha256(value.encode()).hexdigest()[:16].upper()}"

    @staticmethod
    def _knowledge_object_id(workspace_id: str, object_type: str, identity_key: str) -> str:
        value = f"{workspace_id}|{object_type}|{identity_key}"
        return f"KO-{hashlib.sha256(value.encode()).hexdigest()[:16].upper()}"

    @staticmethod
    def _content_fingerprint(
        *,
        title: str,
        domain: str,
        module: str,
        object_type: str,
        content: dict[str, Any],
        entity_references: list[KnowledgeObjectEntityReference],
        evidence: list[str],
        source_traces: list[KnowledgeObjectSourceTrace],
    ) -> str:
        value = json.dumps(
            {
                "title": title,
                "domain": domain,
                "module": module,
                "objectType": object_type,
                "content": content,
                "entityReferences": [item.model_dump(by_alias=True) for item in entity_references],
                "evidence": evidence,
                "sourceTrace": [
                    {
                        "sourceClaimIds": item.source_claim_ids,
                        "claimUsage": [
                            usage.model_dump(by_alias=True) for usage in item.claim_usage
                        ],
                    }
                    for item in source_traces
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _render_markdown(
        *,
        object_id: str,
        revision: int,
        title: str,
        domain: str,
        module: str,
        object_type: str,
        identity_key: str,
        source_lineage_keys: list[str],
        content: dict[str, Any],
        entity_references: list[KnowledgeObjectEntityReference],
        evidence: list[str],
        document_package_id: str,
        source_traces: list[KnowledgeObjectSourceTrace],
        relationships: list[FormalKnowledgeRelationship],
    ) -> str:
        metadata: dict[str, str | int | list[str]] = {
            "knowledgeObjectId": object_id,
            "revision": revision,
            "domain": domain,
            "module": module,
            "objectType": object_type,
            "identityKey": identity_key,
            "sourceLineageKeys": source_lineage_keys,
            "sourceDocumentPackage": document_package_id,
        }
        lines = ["---"]
        for key, value in metadata.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                lines.extend(
                    f"  - {json.dumps(item, ensure_ascii=False)}" for item in value
                )
            else:
                lines.append(
                    f"{key}: {json.dumps(value, ensure_ascii=False)}"
                    if isinstance(value, str)
                    else f"{key}: {value}"
                )
        lines.extend(
            [
                "---",
                "",
                f"# {title}",
                "",
                "## 规范内容",
                "",
            ]
        )
        lines.extend(_render_content_markdown(content))
        lines.extend(["", "## 关联知识", ""])
        if relationships:
            lines.extend(
                f"- `{item.source_ref}` — **{item.relation_type}** → "
                f"`{item.target_ref}`（证据：{', '.join(item.evidence)}）"
                for item in relationships
            )
        else:
            lines.append("- 无")
        lines.extend(["", "## 业务实体引用", ""])
        if entity_references:
            lines.extend(
                f"- `{item.entity_id}` / `{item.reference_role}` / "
                f"证据：{', '.join(item.evidence)}"
                for item in entity_references
            )
        else:
            lines.append("- 无")
        lines.extend(["", "## 来源证据", ""])
        lines.extend(f"- `{item}`" for item in evidence)
        lines.extend(["", "## 正文主张追溯", ""])
        for trace in source_traces:
            lines.append(f"### 候选 `{trace.candidate_id}`")
            lines.append("")
            lines.append(
                f"- 正文归因：{trace.attributed_content_leaf_count}/"
                f"{trace.content_leaf_count} 个叶子字段"
            )
            lines.append(
                "- 实际使用主张："
                + (", ".join(f"`{item}`" for item in trace.source_claim_ids) or "无")
            )
            for usage in trace.claim_usage:
                paths = ", ".join(f"`{path}`" for path in usage.content_paths)
                lines.append(
                    f"- `{usage.claim_id}`（{usage.role}）→ {paths}：{usage.explanation}"
                )
            if trace.unattributed_content_paths:
                lines.append(
                    "- 未归因正文路径："
                    + ", ".join(
                        f"`{path}`" for path in trace.unattributed_content_paths
                    )
                )
        return "\n".join(lines) + "\n"

    def _write_atomically(self, relative_path: Path, content: str) -> None:
        target = (self.project_root / relative_path).resolve()
        if not target.is_relative_to(self.knowledge_root):
            raise ValueError("formal knowledge file path is outside knowledge root")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)


def _render_content_markdown(content: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for field, value in content.items():
        label = CONTENT_FIELD_LABELS.get(field, field)
        lines.extend([f"### {label}", ""])
        lines.extend(_render_markdown_value(value))
        lines.append("")
    if not lines:
        return ["- 未提供规范内容"]
    return lines[:-1]


def _render_markdown_value(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return ["- 未提供"]
    if isinstance(value, bool):
        return ["是" if value else "否"]
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, list):
        lines: list[str] = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, dict):
                lines.append(f"#### 条目 {index}")
                lines.append("")
                lines.extend(_render_mapping_items(item))
            else:
                lines.append(f"- {_scalar_markdown(item)}")
        return lines
    if isinstance(value, dict):
        return _render_mapping_items(value)
    return [str(value)]


def _render_mapping_items(value: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for field, item in value.items():
        label = CONTENT_FIELD_LABELS.get(field, field)
        if isinstance(item, (dict, list)) and item not in ({}, []):
            lines.extend([f"- **{label}**", ""])
            lines.extend(f"  {line}" if line else "" for line in _render_markdown_value(item))
        else:
            lines.append(f"- **{label}**：{_scalar_markdown(item)}")
    return lines


def _scalar_markdown(value: Any) -> str:
    if value in (None, "", [], {}):
        return "未提供"
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)
