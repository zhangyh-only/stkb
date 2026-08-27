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
    IdentificationResult,
    KnowledgeFormationResult,
    KnowledgeFormationStage,
    KnowledgeObjectEntityReference,
    KnowledgeObjectSourceTrace,
    ResolvedBusinessEntity,
)


@dataclass(frozen=True)
class ExistingKnowledgeObjectState:
    revision: int
    content_fingerprint: str


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
    ) -> KnowledgeFormationResult:
        existing_lineages = existing_lineages or {}
        entities = self._resolve_entities(
            document_package.workspace_id,
            identification.candidates,
            existing_entities,
        )
        entity_by_mention = {
            (candidate.candidate_id, mention.mention_id): self._entity_id(
                document_package.workspace_id,
                mention.proposed_type,
                mention.text,
            )
            for candidate in identification.candidates
            for mention in candidate.entity_mentions
        }
        groups: dict[str, list[CandidateKnowledgeObject]] = defaultdict(list)
        identity_keys: dict[str, str] = {}
        lineage_keys: dict[str, set[str]] = defaultdict(set)
        for candidate in identification.candidates:
            identity_key = self._identity_key(candidate)
            lineage_key = self._source_lineage_key(candidate)
            object_id = existing_lineages.get(
                lineage_key,
                self._knowledge_object_id(
                    document_package.workspace_id,
                    candidate.object_type,
                    identity_key,
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
        created_count = sum(item.action == "created" for item in knowledge_objects)
        updated_count = sum(item.action == "updated" for item in knowledge_objects)
        reused_count = sum(item.action == "reused" for item in knowledge_objects)
        return KnowledgeFormationResult(
            run_id=identification.run_id,
            document_package_id=document_package.document_package_id,
            entities=entities,
            knowledge_objects=knowledge_objects,
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
                        f"形成 {len(knowledge_objects)} 个正式知识身份："
                        f"新增 {created_count}、更新 {updated_count}、复用 {reused_count}"
                    ),
                ),
                KnowledgeFormationStage(
                    key="formal_write",
                    name="正式知识写入",
                    status="completed",
                    detail=f"写入或校验 {len(knowledge_objects)} 份正式 Markdown",
                ),
            ],
            created_count=created_count,
            updated_count=updated_count,
            reused_count=reused_count,
            formal_knowledge_files=len(knowledge_objects),
        )

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
        if existing is None:
            action = "created"
            revision = 1
        elif existing.content_fingerprint == payload_fingerprint:
            action = "reused"
            revision = existing.revision
        else:
            action = "updated"
            revision = existing.revision + 1
        relative_path = Path(
            f"workspace/knowledge/{primary.domain}/{primary.module}/{object_id}.md"
        )
        markdown = self._render_markdown(
            object_id=object_id,
            revision=revision,
            title=primary.title,
            domain=primary.domain,
            module=primary.module,
            object_type=primary.object_type,
            identity_key=identity_key,
            source_lineage_keys=source_lineage_keys,
            content=content,
            entity_references=entity_references,
            evidence=evidence,
            document_package_id=document_package.document_package_id,
            source_traces=source_traces,
        )
        file_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        self._write_atomically(relative_path, markdown)
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
            content_fingerprint=payload_fingerprint,
            content=content,
            entity_references=entity_references,
            evidence=evidence,
            source_candidate_ids=source_candidate_ids,
            source_traces=source_traces,
            file_path=relative_path.as_posix(),
            file_sha256=file_sha256,
        )

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

    @staticmethod
    def _identity_key(candidate: CandidateKnowledgeObject) -> str:
        errors = validate_identity_hints(candidate.module, candidate.identity_hints)
        if errors:
            raise ValueError("; ".join(errors))
        canonical = json.dumps(
            canonical_identity(candidate.module, candidate.identity_hints),
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
        value = f"{candidate.module}|{candidate.object_type}|{primary_anchor}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _natural_sort_key(value: str) -> list[tuple[int, int | str]]:
        return [
            (0, int(part)) if part.isdigit() else (1, part)
            for part in re.split(r"(\d+)", value.casefold())
        ]

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
    ) -> str:
        metadata = {
            "knowledgeObjectId": object_id,
            "revision": revision,
            "domain": domain,
            "module": module,
            "objectType": object_type,
            "identityKey": identity_key,
            "sourceLineageKeys": ",".join(source_lineage_keys),
            "sourceDocumentPackage": document_package_id,
        }
        lines = ["---"]
        lines.extend(f"{key}: {value}" for key, value in metadata.items())
        lines.extend(
            [
                "---",
                "",
                f"# {title}",
                "",
                "## 规范内容",
                "",
                "```json",
                json.dumps(content, ensure_ascii=False, indent=2),
                "```",
                "",
                "## 业务实体引用",
                "",
            ]
        )
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
