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

from .models import DocumentPackage, SourceMaterial

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

    def _resolve_workspace_path(self, relative_path: str) -> Path:
        path = (self.project_root / relative_path).resolve()
        workspace_root = (self.project_root / "workspace").resolve()
        if not path.is_relative_to(workspace_root):
            raise DocumentPackageIntegrityError("full Markdown path is outside workspace")
        return path
