from pathlib import Path

from app.features.sales_knowledge_identification.formalizer import (
    ExistingKnowledgeObjectState,
    KnowledgeObjectFormationService,
)
from app.features.sales_knowledge_identification.models import (
    CandidateKnowledgeObject,
    DocumentPackage,
    EntityMention,
    IdentificationResult,
    ProposedRelation,
    SourceAnchor,
)


def _package() -> DocumentPackage:
    return DocumentPackage(
        document_package_id="DP-FORMAL",
        workspace_id="WS-FORMAL",
        source_file_name="source.md",
        source_sha256="source-sha",
        full_markdown_path="workspace/documents/DP-FORMAL/full.md",
        full_markdown_sha256="markdown-sha",
        full_markdown="# 示例",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-FORMAL#section-1", kind="section")],
        quality_issues=[],
    )


def _identification(summary: str = "药享保提供在线问诊") -> IdentificationResult:
    candidate = CandidateKnowledgeObject(
        candidate_id="C1",
        title="药享保在线问诊服务事实",
        domain="D1",
        module="D1.1",
        object_type="PRODUCT_FACT",
        object_boundary="围绕同一产品服务共同更新",
        classification_basis="属于可核验产品事实",
        identity_hints={
            "subject": "药享保",
            "versionScope": "当前产品版本",
            "factTheme": "在线问诊服务",
        },
        content={"summary": summary},
        entity_mentions=[
            EntityMention(
                mention_id="M1",
                text="药享保",
                proposed_type="PRODUCT",
                reference_role="ABOUT_PRODUCT",
                source_ref="DP-FORMAL#section-1",
            )
        ],
        evidence=["DP-FORMAL#section-1"],
        relations=[],
    )
    return IdentificationResult(
        run_id="11111111-1111-1111-1111-111111111111",
        document_package_id="DP-FORMAL",
        provider="test",
        model="test",
        prompt_version="test",
        schema_version="test",
        catalog_version="test",
        catalog_fingerprint="fingerprint",
        raw_model_output="{}",
        model_calls=[],
        processing_stages=[],
        candidates=[candidate],
        rejected_candidates=[],
        weak_signals=[],
        unresolved_items=[],
        coverage_by_module={"D1.1": "hit"},
        call_count=1,
        prompt_tokens=1,
        completion_tokens=1,
    )


def test_formalizer_creates_stable_entity_object_and_markdown(tmp_path: Path) -> None:
    service = KnowledgeObjectFormationService(project_root=tmp_path)
    result = service.form(
        document_package=_package(),
        identification=_identification(),
        existing_entities=set(),
        existing_objects={},
    )

    assert result.status == "completed"
    assert result.created_count == 1
    assert result.updated_count == 0
    assert len(result.entities) == 1
    assert result.entities[0].entity_id.startswith("BE-")
    knowledge_object = result.knowledge_objects[0]
    assert knowledge_object.knowledge_object_id.startswith("KO-")
    assert knowledge_object.revision == 1
    assert knowledge_object.action == "created"
    assert knowledge_object.entity_references[0].entity_id == result.entities[0].entity_id
    file_path = tmp_path / knowledge_object.file_path
    assert file_path.is_file()
    markdown = file_path.read_text(encoding="utf-8")
    assert "# 药享保在线问诊服务事实" in markdown
    assert "## 规范内容" in markdown
    assert "### summary" in markdown
    assert "```json" not in markdown


def test_formalizer_promotes_supported_candidate_relation_to_stable_refs(
    tmp_path: Path,
) -> None:
    identification = _identification()
    first = identification.candidates[0]
    second = first.model_copy(
        deep=True,
        update={
            "candidate_id": "C2",
            "title": "药享保服务限制",
            "identity_hints": {
                **first.identity_hints,
                "factTheme": "服务限制",
            },
        },
    )
    first.relations = [
        ProposedRelation(
            relation_kind="object",
            relation_type="SUPPORTS",
            source_ref="C1",
            target_ref="C2",
            evidence=["DP-FORMAL#section-1"],
        )
    ]
    identification.candidates.append(second)

    result = KnowledgeObjectFormationService(project_root=tmp_path).form(
        document_package=_package(),
        identification=identification,
        existing_entities=set(),
        existing_objects={},
    )

    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.relationship_id.startswith("KR-")
    assert relationship.relation_type == "SUPPORTS"
    assert relationship.inverse_label == "SUPPORTED_BY"
    assert relationship.source_revision == 1
    assert relationship.target_revision == 1
    assert relationship.provenance["runId"] == identification.run_id
    assert {relationship.source_ref, relationship.target_ref} == {
        item.knowledge_object_id for item in result.knowledge_objects
    }
    source_object = next(
        item
        for item in result.knowledge_objects
        if item.knowledge_object_id == relationship.source_ref
    )
    markdown = (tmp_path / source_object.file_path).read_text(encoding="utf-8")
    assert "## 关联知识" in markdown
    assert "**SUPPORTS**" in markdown


def test_source_lineage_does_not_merge_distinct_objects_from_one_anchor() -> None:
    service = KnowledgeObjectFormationService(project_root=Path("."))
    first = _identification().candidates[0]
    second = first.model_copy(
        deep=True,
        update={
            "candidate_id": "C2",
            "identity_hints": {
                **first.identity_hints,
                "versionScope": "另一产品版本",
            },
        },
    )

    assert service._source_lineage_key(first) != service._source_lineage_key(second)


def test_multi_object_source_slots_use_product_version_business_key() -> None:
    service = KnowledgeObjectFormationService(project_root=Path("."))
    first = _identification().candidates[0]
    candidates = [
        first.model_copy(
            deep=True,
            update={
                "candidate_id": f"C-{version}",
                "object_type": "PRODUCT_VERSION_FACT",
                "content": {
                    "subject": "示例产品",
                    "applicability": {"product": "示例产品", "version": version},
                },
            },
        )
        for version in ("基础版", "专业版")
    ]
    existing = {
        f"KO-{version}": ExistingKnowledgeObjectState(
            revision=1,
            content_fingerprint="old",
            module="D1.1",
            object_type="PRODUCT_VERSION_FACT",
            content={
                "subject": "示例产品",
                "applicability": {"product": "示例产品", "version": version},
            },
            evidence=("DP-FORMAL#section-1",),
        )
        for version in ("基础版", "专业版")
    }

    matches, tentative = service._source_slot_matches(
        candidates, existing, set(existing)
    )

    assert matches == {"C-基础版": "KO-基础版", "C-专业版": "KO-专业版"}
    assert tentative == set()


def test_multi_object_source_slots_use_strategy_product_set() -> None:
    service = KnowledgeObjectFormationService(project_root=Path("."))
    first = _identification().candidates[0]
    candidate = first.model_copy(
        deep=True,
        update={
            "candidate_id": "C-STRATEGY",
            "module": "D3.2",
            "object_type": "SALES_STRATEGY",
            "content": {
                "applicability": {"products": ["产品B", "产品A"]},
            },
        },
    )
    existing = {
        "KO-STRATEGY": ExistingKnowledgeObjectState(
            revision=1,
            content_fingerprint="old",
            module="D3.2",
            object_type="SALES_STRATEGY",
            content={"applicability": {"products": ["产品A", "产品B"]}},
            evidence=("DP-FORMAL#section-1",),
        ),
        "KO-OTHER": ExistingKnowledgeObjectState(
            revision=1,
            content_fingerprint="old",
            module="D3.2",
            object_type="SALES_STRATEGY",
            content={"applicability": {"products": ["产品A", "产品C"]}},
            evidence=("DP-FORMAL#section-1",),
        ),
    }

    matches, tentative = service._source_slot_matches(
        [candidate], existing, set(existing)
    )

    assert matches == {"C-STRATEGY": "KO-STRATEGY"}
    assert tentative == set()


def test_product_version_near_subject_match_is_tentative_and_unique() -> None:
    service = KnowledgeObjectFormationService(project_root=Path("."))
    first = _identification().candidates[0]
    candidates = [
        first.model_copy(
            deep=True,
            update={
                "candidate_id": candidate_id,
                "object_type": "PRODUCT_VERSION_FACT",
                "content": {
                    "subject": subject,
                    "applicability": {"product": subject, "version": "专业版"},
                },
            },
        )
        for candidate_id, subject in (
            ("C-PRODUCT", "示例互联网问诊产品"),
            ("C-BENEFIT", "示例权益商城"),
        )
    ]
    existing = {
        "KO-PRODUCT": ExistingKnowledgeObjectState(
            revision=1,
            content_fingerprint="old",
            module="D1.1",
            object_type="PRODUCT_VERSION_FACT",
            content={
                "subject": "示例互联网门诊产品",
                "applicability": {
                    "product": "示例互联网门诊产品",
                    "version": "专业版",
                },
            },
            evidence=("DP-FORMAL#section-1",),
        )
    }

    matches, tentative = service._source_slot_matches(
        candidates, existing, set(existing)
    )

    assert matches == {"C-PRODUCT": "KO-PRODUCT"}
    assert tentative == {"C-PRODUCT"}


def test_formalizer_reuses_unchanged_object_and_requires_review_for_changed_content(
    tmp_path: Path,
) -> None:
    service = KnowledgeObjectFormationService(project_root=tmp_path)
    first = service.form(
        document_package=_package(),
        identification=_identification(),
        existing_entities=set(),
        existing_objects={},
    )
    first_object = first.knowledge_objects[0]
    existing = {
        first_object.knowledge_object_id: ExistingKnowledgeObjectState(
            revision=first_object.revision,
            content_fingerprint=first_object.content_fingerprint,
        )
    }

    reused = service.form(
        document_package=_package(),
        identification=_identification(),
        existing_entities={first.entities[0].entity_id},
        existing_objects=existing,
    )
    updated = service.form(
        document_package=_package(),
        identification=_identification("药享保提供7×24小时在线问诊"),
        existing_entities={first.entities[0].entity_id},
        existing_objects=existing,
    )

    assert reused.knowledge_objects[0].action == "reused"
    assert reused.knowledge_objects[0].revision == 1
    assert reused.entities[0].action == "reused"
    assert updated.status == "review_required"
    assert updated.review_required_count == 1
    assert updated.knowledge_objects[0].action == "review_required"
    assert updated.knowledge_objects[0].revision == 1
    assert updated.knowledge_objects[0].revision_proposal is not None
    assert updated.knowledge_objects[0].revision_proposal.content == {
        "summary": "药享保提供7×24小时在线问诊"
    }


def test_formalizer_requires_review_when_document_identity_set_changes(
    tmp_path: Path,
) -> None:
    service = KnowledgeObjectFormationService(project_root=tmp_path)
    first = service.form(
        document_package=_package(),
        identification=_identification(),
        existing_entities=set(),
        existing_objects={},
    )
    first_id = first.knowledge_objects[0].knowledge_object_id
    changed = _identification().model_copy(deep=True)
    changed.candidates[0].identity_hints["factTheme"] = "线上医疗服务"

    result = service.form(
        document_package=_package(),
        identification=changed,
        existing_entities=set(),
        existing_objects={},
        existing_document_object_ids={first_id},
    )

    assert result.status == "review_required"
    assert result.review_required_count == 1
    assert result.superseded_count == 1
    assert result.formal_knowledge_files == 0
    assert result.knowledge_objects[0].action == "review_required"


def test_formalizer_reuses_unique_source_slot_when_identity_wording_drifts(
    tmp_path: Path,
) -> None:
    service = KnowledgeObjectFormationService(project_root=tmp_path)
    first = service.form(
        document_package=_package(),
        identification=_identification(),
        existing_entities=set(),
        existing_objects={},
    )
    existing = first.knowledge_objects[0]
    changed = _identification().model_copy(deep=True)
    changed.candidates[0].identity_hints["factTheme"] = "线上医疗服务"

    result = service.form(
        document_package=_package(),
        identification=changed,
        existing_entities=set(),
        existing_objects={
            existing.knowledge_object_id: ExistingKnowledgeObjectState(
                revision=existing.revision,
                content_fingerprint=existing.content_fingerprint,
                title=existing.title,
                domain=existing.domain,
                module=existing.module,
                object_type=existing.object_type,
                identity_key=existing.identity_key,
                content=existing.content,
                entity_references=tuple(existing.entity_references),
                evidence=tuple(existing.evidence),
                file_path=existing.file_path,
                file_sha256=existing.file_sha256,
            )
        },
        existing_document_object_ids={existing.knowledge_object_id},
    )

    assert result.status == "completed"
    assert result.knowledge_objects[0].knowledge_object_id == existing.knowledge_object_id
    assert result.knowledge_objects[0].action == "reused"


def test_formalizer_honors_an_explicit_existing_lineage_mapping(
    tmp_path: Path,
) -> None:
    service = KnowledgeObjectFormationService(project_root=tmp_path)
    first_identification = _identification()
    first = service.form(
        document_package=_package(),
        identification=first_identification,
        existing_entities=set(),
        existing_objects={},
    )
    first_object = first.knowledge_objects[0]
    changed_identity = _identification().model_copy(deep=True)
    changed_identity.candidates[0].identity_hints["factTheme"] = "线上医疗服务"
    lineage_key = next(
        iter(service.candidate_lineage_keys(changed_identity.candidates))
    )

    resolved_ids = service.candidate_object_ids(
        _package().workspace_id,
        changed_identity.candidates,
        {lineage_key: first_object.knowledge_object_id},
    )
    second = service.form(
        document_package=_package(),
        identification=changed_identity,
        existing_entities=set(),
        existing_objects={
            first_object.knowledge_object_id: ExistingKnowledgeObjectState(
                revision=first_object.revision,
                content_fingerprint=first_object.content_fingerprint,
            )
        },
        existing_lineages={lineage_key: first_object.knowledge_object_id},
    )

    assert resolved_ids == {first_object.knowledge_object_id}
    assert second.knowledge_objects[0].knowledge_object_id == first_object.knowledge_object_id
    assert second.knowledge_objects[0].action == "reused"


def test_formalizer_keeps_quality_failed_candidate_out_of_formal_knowledge(
    tmp_path: Path,
) -> None:
    service = KnowledgeObjectFormationService(project_root=tmp_path)
    identification = _identification().model_copy(deep=True)
    identification.candidates[0].quality_issues = [
        "关键正文缺少主张归因：$.facts[0].description"
    ]

    result = service.form(
        document_package=_package(),
        identification=identification,
        existing_entities=set(),
        existing_objects={},
    )

    assert result.status == "review_required"
    assert result.knowledge_objects == []
    assert result.entities == []
    assert result.quality_blocked_candidate_ids == ["C1"]
    assert result.quality_blocked_count == 1
    assert result.formal_knowledge_files == 0
    assert not list(tmp_path.rglob("*.md"))


def test_core_equivalence_ignores_script_metadata_and_qa_order_drift() -> None:
    service = KnowledgeObjectFormationService(project_root=Path("."))

    assert service._equivalent_core_content(
        "STANDARD_SCRIPT",
        {"script": "您好，先确认需求。", "applicability": "客户A"},
        {"script": "您好，先确认需求。", "applicability": "通用客户"},
    )
    assert service._equivalent_core_content(
        "QA_PAIR",
        {
            "items": [
                {"question": "问题一", "answer": "答案一", "claimRef": "CL1"},
                {"question": "问题二", "answer": "答案二", "claimRef": "CL2"},
            ]
        },
        {
            "items": [
                {"question": "问题二", "answer": "答案二", "claimRef": "CL9"},
                {"question": "问题一", "answer": "答案一", "claimRef": "CL8"},
            ]
        },
    )
