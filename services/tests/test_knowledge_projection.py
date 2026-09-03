from app.features.sales_knowledge_identification.models import (
    FormalKnowledgeObject,
    KnowledgeFormationResult,
)
from app.features.sales_knowledge_identification.projection import (
    KnowledgeProjectionService,
)


def _formation() -> KnowledgeFormationResult:
    return KnowledgeFormationResult(
        run_id="11111111-1111-1111-1111-111111111111",
        document_package_id="DP-PROJECTION",
        entities=[],
        knowledge_objects=[
            FormalKnowledgeObject(
                knowledge_object_id="KO-1",
                revision=1,
                action="created",
                title="测试知识",
                domain="D1",
                module="D1.1",
                object_type="PRODUCT_FACT",
                identity_key="identity",
                source_lineage_keys=["lineage"],
                content_fingerprint="content",
                content={"summary": "测试知识"},
                entity_references=[],
                evidence=["DP-PROJECTION#page-1"],
                source_candidate_ids=["C1"],
                source_traces=[],
                file_path="workspace/knowledge/D1/D1.1/KO-1.md",
                file_sha256="sha256",
            )
        ],
        relationships=[],
        stages=[],
        created_count=1,
        updated_count=0,
        reused_count=0,
        formal_knowledge_files=1,
    )


def test_projection_reports_each_store_independently() -> None:
    class PartiallyFailingProjection(KnowledgeProjectionService):
        def _embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
            return [[0.0] * 1024], 3

        def _write_vectors(self, *args, **kwargs) -> int:  # type: ignore[no-untyped-def]
            raise RuntimeError("vector unavailable")

        def _write_graph(self, *args, **kwargs) -> tuple[int, int, int, int, int]:  # type: ignore[no-untyped-def]
            return (1, 2, 1, 2, 3)

        def _read_vector_records(self, document_package_id: str) -> list[dict]:
            return []

        def _read_graph_records(
            self, document_package_id: str
        ) -> tuple[list[dict], list[dict]]:
            return [], []

    service = PartiallyFailingProjection(
        postgres_dsn="postgresql://unused",
        neo4j_uri="bolt://unused",
        neo4j_user="unused",
        neo4j_password="unused",
        embedding_base_url="https://example.invalid/v1",
        embedding_api_key="unused",
        embedding_model="test-embedding",
        embedding_dimension=1024,
        embedding_batch_size=10,
        embedding_timeout_seconds=5,
    )

    result = service.project(workspace_id="WS-TEST", formation=_formation())

    assert [stage.status for stage in result.stages] == ["failed", "completed"]
    assert result.evidence.pgvector_records == 0
    assert result.evidence.neo4j_relationships == 6
    assert result.evidence.neo4j_document_links == 1
    assert result.evidence.neo4j_entity_references == 2
    assert result.evidence.neo4j_knowledge_relationships == 3
    assert result.evidence.errors == ["pgvector: vector unavailable"]


def test_retrieval_units_are_derived_from_internal_qa_items() -> None:
    item = _formation().knowledge_objects[0].model_copy(
        update={
            "object_type": "QA_PAIR",
            "content": {
                "items": [
                    {"question": "问题一", "answer": "答案一", "claimRef": "CL1"},
                    {"question": "问题二", "answer": "答案二", "claimRef": "CL2"},
                ]
            },
        }
    )

    units = KnowledgeProjectionService._retrieval_units(item)

    assert [unit["itemId"] for unit in units] == ["CL1", "CL2"]
    assert [unit["contentPath"] for unit in units] == [
        "$.items[0]",
        "$.items[1]",
    ]
    assert all("identityKey" not in unit["retrievalText"] for unit in units)
    assert units[0]["embeddingText"] == "问题：问题一"
    assert "答案：答案一" in units[0]["retrievalText"]
    assert units[0]["retrievalText"].index("问题一") < units[0]["retrievalText"].index(
        "答案一"
    )


def test_process_is_one_retrieval_unit_instead_of_isolated_steps() -> None:
    item = _formation().knowledge_objects[0].model_copy(
        update={
            "object_type": "BUSINESS_PROCESS",
            "content": {"rulesOrSteps": ["第一步", "第二步", "第三步"]},
        }
    )

    units = KnowledgeProjectionService._retrieval_units(item)

    assert len(units) == 1
    assert units[0]["contentPath"] == "$"
    assert "第一步" in units[0]["retrievalText"]
    assert "第三步" in units[0]["retrievalText"]


def test_policy_rule_set_is_one_retrieval_unit_with_related_rules() -> None:
    item = _formation().knowledge_objects[0].model_copy(
        update={
            "object_type": "POLICY_RULE_SET",
            "content": {"rulesOrSteps": ["处方保留72小时", "待付款订单可以继续支付"]},
        }
    )

    units = KnowledgeProjectionService._retrieval_units(item)

    assert len(units) == 1
    assert "处方保留72小时" in units[0]["retrievalText"]
    assert "待付款订单可以继续支付" in units[0]["retrievalText"]
