from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .formalizer import ExistingKnowledgeObjectState
from .models import (
    DocumentPackage,
    KnowledgeFormationResult,
    KnowledgeObjectEntityReference,
    SourceMaterial,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class IdentificationRecordNotFound(LookupError):
    pass


class DocumentPackageIntegrityError(RuntimeError):
    pass


class PsycopgIdentificationRepository:
    def __init__(
        self,
        *,
        postgres_dsn: str,
        project_root: Path,
        retention_hours: int,
    ) -> None:
        self.postgres_dsn = postgres_dsn
        self.project_root = project_root.resolve()
        self.retention_hours = retention_hours

    def ensure_schema(self) -> None:
        with psycopg.connect(self.postgres_dsn) as connection:
            connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    def register_manifest(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with psycopg.connect(self.postgres_dsn) as connection:
            connection.execute(
                """
                INSERT INTO document_packages (
                    document_package_id, workspace_id, source_file_name, source_file_path,
                    source_sha256,
                    full_markdown_path, full_markdown_sha256, processing_method, status,
                    anchors, quality_issues
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_package_id) DO UPDATE SET
                    workspace_id = EXCLUDED.workspace_id,
                    source_file_name = EXCLUDED.source_file_name,
                    source_file_path = EXCLUDED.source_file_path,
                    source_sha256 = EXCLUDED.source_sha256,
                    full_markdown_path = EXCLUDED.full_markdown_path,
                    full_markdown_sha256 = EXCLUDED.full_markdown_sha256,
                    processing_method = EXCLUDED.processing_method,
                    status = EXCLUDED.status,
                    anchors = EXCLUDED.anchors,
                    quality_issues = EXCLUDED.quality_issues
                """,
                (
                    manifest["documentPackageId"],
                    manifest["workspaceId"],
                    manifest["sourceFileName"],
                    manifest["sourceFilePath"],
                    manifest["sourceSha256"],
                    manifest["fullMarkdownPath"],
                    manifest["fullMarkdownSha256"],
                    manifest["processingMethod"],
                    manifest["status"],
                    Jsonb(manifest["anchors"]),
                    Jsonb(manifest["qualityIssues"]),
                ),
            )

    def list_source_materials(self) -> list[SourceMaterial]:
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT document_package_id, source_file_name, source_file_path, source_sha256,
                       processing_method, status
                FROM document_packages
                ORDER BY created_at DESC, source_file_name
                """
            ).fetchall()
        return [
            SourceMaterial(
                document_package_id=row["document_package_id"],
                source_file_name=row["source_file_name"],
                source_file_path=row["source_file_path"],
                source_sha256=row["source_sha256"],
                processing_method=row["processing_method"],
                status=row["status"],
            )
            for row in rows
        ]

    def get_document_package(self, document_package_id: str) -> DocumentPackage:
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT * FROM document_packages WHERE document_package_id = %s",
                (document_package_id,),
            ).fetchone()
        if row is None:
            raise IdentificationRecordNotFound(document_package_id)
        source_file_path = self._resolve_workspace_path(row["source_file_path"])
        actual_source_checksum = hashlib.sha256(source_file_path.read_bytes()).hexdigest()
        if actual_source_checksum != row["source_sha256"]:
            raise DocumentPackageIntegrityError(
                f"source file checksum mismatch for {document_package_id}"
            )
        full_markdown_path = self._resolve_workspace_path(row["full_markdown_path"])
        full_markdown = full_markdown_path.read_text(encoding="utf-8")
        actual_checksum = hashlib.sha256(full_markdown.encode("utf-8")).hexdigest()
        if actual_checksum != row["full_markdown_sha256"]:
            raise DocumentPackageIntegrityError(
                f"full Markdown checksum mismatch for {document_package_id}"
            )
        return DocumentPackage(
            document_package_id=row["document_package_id"],
            workspace_id=row["workspace_id"],
            source_file_name=row["source_file_name"],
            source_file_path=row["source_file_path"],
            source_sha256=row["source_sha256"],
            full_markdown_path=row["full_markdown_path"],
            full_markdown_sha256=row["full_markdown_sha256"],
            full_markdown=full_markdown,
            processing_method=row["processing_method"],
            status=row["status"],
            anchors=row["anchors"],
            quality_issues=row["quality_issues"],
        )

    def save_run(self, result: dict[str, Any]) -> None:
        expires_at = datetime.now(UTC) + timedelta(hours=self.retention_hours)
        with psycopg.connect(self.postgres_dsn) as connection:
            connection.execute(
                "DELETE FROM sales_knowledge_identification_runs WHERE expires_at <= NOW()"
            )
            connection.execute(
                """
                INSERT INTO sales_knowledge_identification_runs (
                    run_id, document_package_id, status, provider, model, result_json, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    result["runId"],
                    result["documentPackageId"],
                    result["status"],
                    result["provider"],
                    result["model"],
                    Jsonb(result),
                    expires_at,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            row = connection.execute(
                """
                SELECT result_json
                FROM sales_knowledge_identification_runs
                WHERE run_id = %s AND expires_at > NOW()
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise IdentificationRecordNotFound(run_id)
        return row["result_json"]

    def list_runs(self, document_package_id: str, limit: int) -> list[dict[str, Any]]:
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT result_json
                FROM sales_knowledge_identification_runs
                WHERE document_package_id = %s AND expires_at > NOW()
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (document_package_id, limit),
            ).fetchall()
        return [row["result_json"] for row in reversed(rows)]

    def get_evaluation_report(self, document_package_id: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+", document_package_id) is None:
            raise IdentificationRecordNotFound(document_package_id)
        report_path = self._resolve_workspace_path(
            f"workspace/evaluations/{document_package_id}/validation-report.md"
        )
        if not report_path.is_file():
            raise IdentificationRecordNotFound(document_package_id)
        return report_path.read_text(encoding="utf-8")

    def get_existing_entity_ids(self, entity_ids: set[str]) -> set[str]:
        if not entity_ids:
            return set()
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            rows = connection.execute(
                "SELECT entity_id FROM business_entities WHERE entity_id = ANY(%s)",
                (list(entity_ids),),
            ).fetchall()
        return {row["entity_id"] for row in rows}

    def get_existing_object_states(
        self, object_ids: set[str]
    ) -> dict[str, ExistingKnowledgeObjectState]:
        if not object_ids:
            return {}
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT knowledge_object_id, revision, content_fingerprint, title,
                       domain, module, object_type, identity_key, content,
                       entity_references, evidence, file_path, file_sha256
                FROM knowledge_objects
                WHERE knowledge_object_id = ANY(%s)
                """,
                (list(object_ids),),
            ).fetchall()
        return {
            row["knowledge_object_id"]: ExistingKnowledgeObjectState(
                revision=row["revision"],
                content_fingerprint=row["content_fingerprint"],
                title=row["title"],
                domain=row["domain"],
                module=row["module"],
                object_type=row["object_type"],
                identity_key=row["identity_key"],
                content=row["content"],
                entity_references=tuple(
                    KnowledgeObjectEntityReference.model_validate(item)
                    for item in row["entity_references"]
                ),
                evidence=tuple(row["evidence"]),
                file_path=row["file_path"],
                file_sha256=row["file_sha256"],
            )
            for row in rows
        }

    def get_existing_lineage_object_ids(
        self, workspace_id: str, lineage_keys: set[str]
    ) -> dict[str, str]:
        if not lineage_keys:
            return {}
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT source_lineage_key, knowledge_object_id
                FROM knowledge_object_lineages
                WHERE workspace_id = %s AND source_lineage_key = ANY(%s)
                """,
                (workspace_id, list(lineage_keys)),
            ).fetchall()
        return {
            row["source_lineage_key"]: row["knowledge_object_id"] for row in rows
        }

    def save_knowledge_formation(
        self,
        *,
        workspace_id: str,
        formation: KnowledgeFormationResult,
    ) -> None:
        with psycopg.connect(self.postgres_dsn) as connection:
            if formation.status == "review_required":
                self._save_knowledge_build_result(connection, formation)
                return
            current_object_ids = {
                item.knowledge_object_id for item in formation.knowledge_objects
            }
            previous_rows = connection.execute(
                """
                SELECT DISTINCT knowledge_object_id
                FROM knowledge_object_sources
                WHERE document_package_id = %s AND active = TRUE
                """,
                (formation.document_package_id,),
            ).fetchall()
            previous_object_ids = {row[0] for row in previous_rows}
            superseded_object_ids = previous_object_ids - current_object_ids
            formation.superseded_count = len(superseded_object_ids)
            connection.execute(
                """
                UPDATE knowledge_object_sources
                SET active = FALSE
                WHERE document_package_id = %s AND active = TRUE
                """,
                (formation.document_package_id,),
            )
            for entity in formation.entities:
                connection.execute(
                    """
                    INSERT INTO business_entities (
                        entity_id, workspace_id, entity_type, canonical_name, source_mentions
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO UPDATE SET
                        canonical_name = EXCLUDED.canonical_name,
                        source_mentions = EXCLUDED.source_mentions,
                        updated_at = NOW()
                    """,
                    (
                        entity.entity_id,
                        workspace_id,
                        entity.entity_type,
                        entity.canonical_name,
                        Jsonb(entity.source_mentions),
                    ),
                )
            for item in formation.knowledge_objects:
                connection.execute(
                    """
                    INSERT INTO knowledge_objects (
                        knowledge_object_id, workspace_id, revision, domain, module,
                        object_type, title, identity_key, content_fingerprint, content,
                        entity_references, evidence, file_path, file_sha256
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (knowledge_object_id) DO UPDATE SET
                        revision = EXCLUDED.revision,
                        domain = EXCLUDED.domain,
                        module = EXCLUDED.module,
                        object_type = EXCLUDED.object_type,
                        title = EXCLUDED.title,
                        identity_key = EXCLUDED.identity_key,
                        content_fingerprint = EXCLUDED.content_fingerprint,
                        content = EXCLUDED.content,
                        entity_references = EXCLUDED.entity_references,
                        evidence = EXCLUDED.evidence,
                        file_path = EXCLUDED.file_path,
                        file_sha256 = EXCLUDED.file_sha256,
                        status = 'active',
                        updated_at = NOW()
                    """,
                    (
                        item.knowledge_object_id,
                        workspace_id,
                        item.revision,
                        item.domain,
                        item.module,
                        item.object_type,
                        item.title,
                        item.identity_key,
                        item.content_fingerprint,
                        Jsonb(item.content),
                        Jsonb(
                            [
                                reference.model_dump(mode="json", by_alias=True)
                                for reference in item.entity_references
                            ]
                        ),
                        Jsonb(item.evidence),
                        item.file_path,
                        item.file_sha256,
                    ),
                )
                for trace in item.source_traces:
                    connection.execute(
                        """
                        INSERT INTO knowledge_object_sources (
                            knowledge_object_id, document_package_id, run_id, candidate_id,
                            evidence, source_claim_ids, claim_usage, active
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
                        ON CONFLICT (
                            knowledge_object_id, document_package_id, run_id, candidate_id
                        ) WHERE run_id IS NOT NULL
                        DO UPDATE SET
                            evidence = EXCLUDED.evidence,
                            source_claim_ids = EXCLUDED.source_claim_ids,
                            claim_usage = EXCLUDED.claim_usage,
                            active = TRUE
                        """,
                        (
                            item.knowledge_object_id,
                            formation.document_package_id,
                            formation.run_id,
                            trace.candidate_id,
                            Jsonb(item.evidence),
                            Jsonb(trace.source_claim_ids),
                            Jsonb(
                                [
                                    usage.model_dump(mode="json", by_alias=True)
                                    for usage in trace.claim_usage
                                ]
                            ),
                        ),
                    )
                for lineage_key in item.source_lineage_keys:
                    connection.execute(
                        """
                        INSERT INTO knowledge_object_lineages (
                            workspace_id, source_lineage_key, knowledge_object_id,
                            document_package_id, module, object_type
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (workspace_id, source_lineage_key) DO UPDATE SET
                            knowledge_object_id = EXCLUDED.knowledge_object_id,
                            document_package_id = EXCLUDED.document_package_id,
                            module = EXCLUDED.module,
                            object_type = EXCLUDED.object_type,
                            updated_at = NOW()
                        """,
                        (
                            workspace_id,
                            lineage_key,
                            item.knowledge_object_id,
                            formation.document_package_id,
                            item.module,
                            item.object_type,
                        ),
                    )
            if superseded_object_ids:
                connection.execute(
                    """
                    UPDATE knowledge_objects object_record
                    SET status = 'superseded', updated_at = NOW()
                    WHERE object_record.knowledge_object_id = ANY(%s)
                      AND NOT EXISTS (
                          SELECT 1
                          FROM knowledge_object_sources source_record
                          WHERE source_record.knowledge_object_id =
                                object_record.knowledge_object_id
                            AND source_record.active = TRUE
                      )
                    """,
                    (list(superseded_object_ids),),
                )
            self._save_knowledge_build_result(connection, formation)

    @staticmethod
    def _save_knowledge_build_result(
        connection: psycopg.Connection[Any], formation: KnowledgeFormationResult
    ) -> None:
        serialized = formation.model_dump(mode="json", by_alias=True)
        connection.execute(
                """
                INSERT INTO knowledge_build_results (
                    build_id, run_id, document_package_id, status, result_json
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    build_id = EXCLUDED.build_id,
                    status = EXCLUDED.status,
                    result_json = EXCLUDED.result_json
                """,
            (
                formation.build_id,
                formation.run_id,
                formation.document_package_id,
                formation.status,
                Jsonb(serialized),
            ),
        )

    def get_knowledge_formation(self, run_id: str) -> dict[str, Any]:
        with psycopg.connect(self.postgres_dsn, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT result_json FROM knowledge_build_results WHERE run_id = %s",
                (run_id,),
            ).fetchone()
        if row is None:
            raise IdentificationRecordNotFound(run_id)
        return row["result_json"]

    def _resolve_workspace_path(self, relative_path: str) -> Path:
        path = (self.project_root / relative_path).resolve()
        workspace_root = (self.project_root / "workspace").resolve()
        if not path.is_relative_to(workspace_root):
            raise DocumentPackageIntegrityError("full Markdown path is outside workspace")
        return path
