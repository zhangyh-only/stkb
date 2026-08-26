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
        identity_hints={"product": "药享保", "factType": "在线问诊服务"},
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
    assert "# 药享保在线问诊服务事实" in file_path.read_text(encoding="utf-8")


def test_formalizer_reuses_unchanged_object_and_revises_changed_content(
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
    assert updated.knowledge_objects[0].action == "updated"
    assert updated.knowledge_objects[0].revision == 2
