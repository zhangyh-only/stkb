from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx
import psycopg
from neo4j import GraphDatabase
from psycopg.rows import dict_row

from .models import (
    FormalKnowledgeObject,
    KnowledgeFormationResult,
    KnowledgeFormationStage,
    KnowledgeStorageEvidence,
)


@dataclass(frozen=True)
class ProjectionOutcome:
    evidence: KnowledgeStorageEvidence
    stages: list[KnowledgeFormationStage]


class KnowledgeProjectionService:
    def __init__(
        self,
        *,
        postgres_dsn: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        embedding_base_url: str,
        embedding_api_key: str,
        embedding_model: str,
        embedding_dimension: int,
        embedding_batch_size: int,
        embedding_timeout_seconds: int,
    ) -> None:
        if embedding_dimension != 1024:
            raise ValueError("the current pgvector schema requires 1024 dimensions")
        self.postgres_dsn = postgres_dsn
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = (neo4j_user, neo4j_password)
        self.embedding_url = embedding_base_url.rstrip("/") + "/embeddings"
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.embedding_batch_size = embedding_batch_size
        self.embedding_timeout_seconds = embedding_timeout_seconds

    def project(
        self, *, workspace_id: str, formation: KnowledgeFormationResult
    ) -> ProjectionOutcome:
        if formation.status != "completed":
            raise ValueError("only completed formal knowledge can be projected")
        errors: list[str] = []
        stages: list[KnowledgeFormationStage] = []
        vector_records = 0
        vector_record_data: list[dict[str, Any]] = []
        units: list[dict[str, Any]] = []
        embedding_tokens = 0
        vector_started = perf_counter()
        try:
            units = [
                unit
                for item in formation.knowledge_objects
                for unit in self._retrieval_units(item)
            ]
            texts = [unit["retrievalText"] for unit in units]
            embeddings, embedding_tokens = self._embed(texts)
            vector_records = self._write_vectors(
                workspace_id, formation, units, embeddings
            )
            vector_record_data = self._read_vector_records(
                formation.document_package_id
            )
            vector_duration_ms = round((perf_counter() - vector_started) * 1000)
            stages.append(
                KnowledgeFormationStage(
                    key="pgvector_projection",
                    name="向量检索投影",
                    status="completed",
                    duration_ms=vector_duration_ms,
                    detail=(
                        f"使用 {self.embedding_model} 写入 {vector_records} 条"
                        f" {self.embedding_dimension} 维检索向量，消耗 {embedding_tokens} tokens"
                    ),
                )
            )
        except Exception as error:  # noqa: BLE001 - projection evidence must retain failures
            vector_duration_ms = round((perf_counter() - vector_started) * 1000)
            errors.append(f"pgvector: {error}")
            stages.append(
                KnowledgeFormationStage(
                    key="pgvector_projection",
                    name="向量检索投影",
                    status="failed",
                    duration_ms=vector_duration_ms,
                    detail=str(error),
                )
            )

        graph_started = perf_counter()
        graph_counts = (0, 0, 0, 0, 0)
        graph_nodes: list[dict[str, Any]] = []
        graph_relationships: list[dict[str, Any]] = []
        try:
            graph_counts = self._write_graph(workspace_id, formation)
            graph_nodes, graph_relationships = self._read_graph_records(
                formation.document_package_id
            )
            graph_duration_ms = round((perf_counter() - graph_started) * 1000)
            stages.append(
                KnowledgeFormationStage(
                    key="neo4j_projection",
                    name="知识图谱投影",
                    status="completed",
                    duration_ms=graph_duration_ms,
                    detail=(
                        f"写入 {graph_counts[0]} 个知识对象节点、{graph_counts[1]} 个实体节点；"
                        f"关系包含承载 {graph_counts[2]}、实体引用 {graph_counts[3]}、"
                        f"知识关系 {graph_counts[4]}"
                    ),
                )
            )
        except Exception as error:  # noqa: BLE001 - projection evidence must retain failures
            graph_duration_ms = round((perf_counter() - graph_started) * 1000)
            errors.append(f"neo4j: {error}")
            stages.append(
                KnowledgeFormationStage(
                    key="neo4j_projection",
                    name="知识图谱投影",
                    status="failed",
                    duration_ms=graph_duration_ms,
                    detail=str(error),
                )
            )

        return ProjectionOutcome(
            evidence=KnowledgeStorageEvidence(
                postgres_objects=len(formation.knowledge_objects),
                formal_files=formation.formal_knowledge_files,
                internal_items=sum(unit["contentPath"] != "$" for unit in units),
                pgvector_records=vector_records,
                neo4j_knowledge_objects=graph_counts[0],
                neo4j_entities=graph_counts[1],
                neo4j_relationships=sum(graph_counts[2:]),
                neo4j_document_links=graph_counts[2],
                neo4j_entity_references=graph_counts[3],
                neo4j_knowledge_relationships=graph_counts[4],
                embedding_model=self.embedding_model,
                embedding_tokens=embedding_tokens,
                vector_duration_ms=vector_duration_ms,
                graph_duration_ms=graph_duration_ms,
                errors=errors,
                formal_records=[
                    {
                        "knowledgeObjectId": item.knowledge_object_id,
                        "revision": item.revision,
                        "filePath": item.file_path,
                        "fileSha256": item.file_sha256,
                    }
                    for item in formation.knowledge_objects
                ],
                vector_records=vector_record_data,
                graph_nodes=graph_nodes,
                graph_relationships=graph_relationships,
            ),
            stages=stages,
        )

    @staticmethod
    def _retrieval_units(item: FormalKnowledgeObject) -> list[dict[str, Any]]:
        item_fields = {
            "PRODUCT_VERSION_FACT": "facts",
            "LIST_FACT": "entryStructure",
            "SELLING_POINT": "observableChecks",
            "SALES_STRATEGY": "actions",
            "QA_PAIR": "items",
            "TERM": "terms",
        }
        field = item_fields.get(item.object_type)
        values = item.content.get(field) if field else None
        if not isinstance(values, list) or not values:
            text = f"{item.title}\n{json.dumps(item.content, ensure_ascii=False)}"
            return [
                {
                    "retrievalUnitId": f"RU-{item.knowledge_object_id}-R{item.revision}-ROOT",
                    "itemId": None,
                    "contentPath": "$",
                    "retrievalText": text,
                    "object": item,
                }
            ]
        units = []
        for index, value in enumerate(values):
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
            explicit_id = (
                next(
                    (
                        value.get(key)
                        for key in ("itemId", "claimRef", "stepId", "ruleId")
                        if isinstance(value, dict) and value.get(key)
                    ),
                    None,
                )
                if isinstance(value, dict)
                else None
            )
            item_id = str(explicit_id or hashlib.sha256(serialized.encode()).hexdigest()[:12])
            units.append(
                {
                    "retrievalUnitId": (
                        f"RU-{item.knowledge_object_id}-R{item.revision}-{item_id}"
                    ),
                    "itemId": item_id,
                    "contentPath": f"$.{field}[{index}]",
                    "retrievalText": (
                        f"{item.title}\n问题：{value.get('question')}\n"
                        f"答案：{value.get('answer')}"
                        if item.object_type == "QA_PAIR" and isinstance(value, dict)
                        else f"{item.title}\n{field}: {serialized}"
                    ),
                    "object": item,
                }
            )
        return units

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        embeddings: list[list[float]] = []
        total_tokens = 0
        with httpx.Client(timeout=self.embedding_timeout_seconds) as client:
            for offset in range(0, len(texts), self.embedding_batch_size):
                response = client.post(
                    self.embedding_url,
                    headers={"Authorization": f"Bearer {self.embedding_api_key}"},
                    json={
                        "model": self.embedding_model,
                        "input": texts[offset : offset + self.embedding_batch_size],
                        "dimensions": self.embedding_dimension,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
                payload = response.json()
                embeddings.extend(
                    item["embedding"]
                    for item in sorted(payload["data"], key=lambda item: item["index"])
                )
                total_tokens += int(payload.get("usage", {}).get("total_tokens", 0))
        if len(embeddings) != len(texts) or any(
            len(embedding) != self.embedding_dimension for embedding in embeddings
        ):
            raise ValueError("embedding response does not match the requested inputs")
        return embeddings, total_tokens

    def _write_vectors(
        self,
        workspace_id: str,
        formation: KnowledgeFormationResult,
        units: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        with psycopg.connect(self.postgres_dsn) as connection:
            connection.execute(
                """
                UPDATE knowledge_retrieval_units SET active = FALSE, updated_at = NOW()
                WHERE document_package_id = %s AND active = TRUE
                """,
                (formation.document_package_id,),
            )
            for unit, embedding in zip(units, embeddings, strict=True):
                item = unit["object"]
                connection.execute(
                    """
                    INSERT INTO knowledge_retrieval_units (
                        retrieval_unit_id, workspace_id, document_package_id,
                        knowledge_object_id, revision, domain, module, object_type,
                        title, item_id, content_path, retrieval_text,
                        source_file_sha256, embedding_model,
                        embedding_dimension, embedding, active
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s::vector, TRUE
                    )
                    ON CONFLICT (retrieval_unit_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        item_id = EXCLUDED.item_id,
                        content_path = EXCLUDED.content_path,
                        retrieval_text = EXCLUDED.retrieval_text,
                        source_file_sha256 = EXCLUDED.source_file_sha256,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dimension = EXCLUDED.embedding_dimension,
                        embedding = EXCLUDED.embedding,
                        active = TRUE,
                        updated_at = NOW()
                    """,
                    (
                        unit["retrievalUnitId"],
                        workspace_id,
                        formation.document_package_id,
                        item.knowledge_object_id,
                        item.revision,
                        item.domain,
                        item.module,
                        item.object_type,
                        item.title,
                        unit["itemId"],
                        unit["contentPath"],
                        unit["retrievalText"],
                        item.file_sha256,
                        self.embedding_model,
                        self.embedding_dimension,
                        "[" + ",".join(str(value) for value in embedding) + "]",
                    ),
                )
        return len(units)

    def _write_graph(
        self, workspace_id: str, formation: KnowledgeFormationResult
    ) -> tuple[int, int, int, int, int]:
        objects = [
            {
                "id": item.knowledge_object_id,
                "revision": item.revision,
                "domain": item.domain,
                "module": item.module,
                "objectType": item.object_type,
                "title": item.title,
                "fileSha256": item.file_sha256,
            }
            for item in formation.knowledge_objects
        ]
        entities = [
            {"id": item.entity_id, "type": item.entity_type, "name": item.canonical_name}
            for item in formation.entities
        ]
        references = [
            {
                "objectId": item.knowledge_object_id,
                "entityId": reference.entity_id,
                "role": reference.reference_role,
                "evidence": reference.evidence,
            }
            for item in formation.knowledge_objects
            for reference in item.entity_references
        ]
        relationships = [
            {
                "id": item.relationship_id,
                "source": item.source_ref,
                "target": item.target_ref,
                "type": item.relation_type,
                "evidence": item.evidence,
            }
            for item in formation.relationships
        ]
        with GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth) as driver:
            with driver.session() as session:
                for query in (
                    "CREATE CONSTRAINT knowledge_object_id IF NOT EXISTS "
                    "FOR (node:KnowledgeObject) REQUIRE node.id IS UNIQUE",
                    "CREATE CONSTRAINT business_entity_id IF NOT EXISTS "
                    "FOR (node:BusinessEntity) REQUIRE node.id IS UNIQUE",
                    "CREATE CONSTRAINT document_package_id IF NOT EXISTS "
                    "FOR (node:DocumentPackage) REQUIRE node.id IS UNIQUE",
                ):
                    session.run(query).consume()
                session.run(
                    """
                    MERGE (document:DocumentPackage {id: $documentId})
                    SET document.workspaceId = $workspaceId
                    WITH document
                    OPTIONAL MATCH (document)-[old:CONTAINS]->(object:KnowledgeObject)
                    SET object.active = false
                    DELETE old
                    """,
                    documentId=formation.document_package_id,
                    workspaceId=workspace_id,
                ).consume()
                session.run(
                    """
                    UNWIND $objects AS item
                    MERGE (object:KnowledgeObject {id: item.id})
                    SET object += item, object.workspaceId = $workspaceId, object.active = true
                    WITH object
                    MATCH (document:DocumentPackage {id: $documentId})
                    MERGE (document)-[:CONTAINS]->(object)
                    """,
                    objects=objects,
                    workspaceId=workspace_id,
                    documentId=formation.document_package_id,
                ).consume()
                session.run(
                    """
                    UNWIND $entities AS item
                    MERGE (entity:BusinessEntity {id: item.id})
                    SET entity += item, entity.workspaceId = $workspaceId
                    """,
                    entities=entities,
                    workspaceId=workspace_id,
                ).consume()
                object_ids = [item["id"] for item in objects]
                session.run(
                    """
                    MATCH (object:KnowledgeObject)-[old:REFERS_TO]->()
                    WHERE object.id IN $objectIds
                    DELETE old
                    """,
                    objectIds=object_ids,
                ).consume()
                session.run(
                    """
                    UNWIND $references AS item
                    MATCH (object:KnowledgeObject {id: item.objectId})
                    MATCH (entity:BusinessEntity {id: item.entityId})
                    MERGE (object)-[reference:REFERS_TO {role: item.role}]->(entity)
                    SET reference.evidence = item.evidence
                    """,
                    references=references,
                ).consume()
                session.run(
                    """
                    MATCH (source:KnowledgeObject)-[old:RELATED]->()
                    WHERE source.id IN $objectIds
                    DELETE old
                    """,
                    objectIds=object_ids,
                ).consume()
                session.run(
                    """
                    UNWIND $relationships AS item
                    MATCH (source) WHERE source.id = item.source
                    MATCH (target) WHERE target.id = item.target
                    MERGE (source)-[relation:RELATED {id: item.id}]->(target)
                    SET relation.type = item.type, relation.evidence = item.evidence
                    """,
                    relationships=relationships,
                ).consume()
        return (
            len(objects),
            len(entities),
            len(objects),
            len(references),
            len(relationships),
        )

    def _read_vector_records(self, document_package_id: str) -> list[dict[str, Any]]:
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT retrieval_unit_id, knowledge_object_id, revision,
                       item_id, content_path, embedding_model,
                       embedding_dimension, active,
                       LEFT(retrieval_text, 160) AS text_preview
                FROM knowledge_retrieval_units
                WHERE document_package_id = %s AND active = TRUE
                ORDER BY retrieval_unit_id
                """,
                (document_package_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _read_graph_records(
        self, document_package_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth) as driver:
            nodes = driver.execute_query(
                """
                MATCH (document:DocumentPackage {id: $documentId})-[:CONTAINS]->
                      (object:KnowledgeObject)
                OPTIONAL MATCH (object)-[:REFERS_TO]->(entity:BusinessEntity)
                WITH document, collect(DISTINCT object) AS objects,
                     collect(DISTINCT entity) AS entities
                UNWIND [document] + objects + entities AS node
                RETURN labels(node) AS labels, node.id AS id,
                       node.revision AS revision, node.title AS title
                ORDER BY labels, id
                """,
                documentId=document_package_id,
                result_transformer_=lambda result: [record.data() for record in result],
            )
            relationships = driver.execute_query(
                """
                MATCH (document:DocumentPackage {id: $documentId})-[:CONTAINS]->
                      (object:KnowledgeObject)
                WITH document, collect(object.id) AS objectIds
                MATCH (source)-[relation]->(target)
                WHERE (source.id = $documentId AND type(relation) = 'CONTAINS')
                   OR source.id IN objectIds
                RETURN source.id AS source, type(relation) AS type,
                       target.id AS target, relation.role AS role,
                       relation.id AS relationshipId
                ORDER BY type, source, target
                """,
                documentId=document_package_id,
                result_transformer_=lambda result: [record.data() for record in result],
            )
        return nodes, relationships
