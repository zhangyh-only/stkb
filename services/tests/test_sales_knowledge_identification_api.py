import json

import pytest
from fastapi.testclient import TestClient

from app.api import sales_knowledge_identification as identification_api
from app.api.sales_knowledge_identification import (
    get_identification_repository,
)
from app.features.sales_knowledge_identification.models import (
    DocumentPackage,
    ModelCompletion,
    SourceAnchor,
    SourceMaterial,
)
from app.main import app


class InMemoryRepository:
    def __init__(self, document_package: DocumentPackage) -> None:
        self.document_package = document_package
        self.runs: dict[str, dict[str, object]] = {}

    def get_document_package(self, document_package_id: str) -> DocumentPackage:
        assert document_package_id == self.document_package.document_package_id
        return self.document_package

    def list_source_materials(self) -> list[SourceMaterial]:
        return [
            SourceMaterial(
                document_package_id=self.document_package.document_package_id,
                source_file_name=self.document_package.source_file_name,
                source_file_path=self.document_package.source_file_path,
                source_sha256=self.document_package.source_sha256,
                processing_method=self.document_package.processing_method,
                status=self.document_package.status,
            )
        ]

    def save_run(self, result: dict[str, object]) -> None:
        self.runs[str(result["runId"])] = result

    def get_run(self, run_id: str) -> dict[str, object]:
        return self.runs[run_id]

    def list_runs(self, document_package_id: str, limit: int) -> list[dict[str, object]]:
        assert document_package_id == self.document_package.document_package_id
        return list(self.runs.values())[-limit:]

    def get_evaluation_report(self, document_package_id: str) -> str:
        assert document_package_id == self.document_package.document_package_id
        return "# 代理评估\n\n节点机制阶段通过。"


class ApiStubGateway:
    def complete(self, request):  # type: ignore[no-untyped-def]
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(
                {
                    "candidates": [
                        {
                            "candidateId": "C1",
                            "domain": "D1",
                            "module": "D1.1",
                            "objectType": "PRODUCT_FACT",
                            "content": {"summary": "药享保提供在线问诊服务"},
                            "entityMentions": [],
                            "evidence": ["DP-API#page-1"],
                            "relations": [],
                        }
                    ],
                    "weakSignals": [],
                    "unresolvedItems": [],
                },
                ensure_ascii=False,
            ),
        )


def test_api_runs_identification_and_reads_the_saved_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_package = DocumentPackage(
        document_package_id="DP-API",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-API/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n药享保提供在线问诊服务。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-API#page-1", kind="page", page=1)],
        quality_issues=[],
    )
    repository = InMemoryRepository(document_package)
    app.dependency_overrides[get_identification_repository] = lambda: repository
    monkeypatch.setattr(identification_api, "get_model_gateway", lambda: ApiStubGateway())
    client = TestClient(app)

    try:
        catalog_response = client.get("/api/sales-knowledge-identification/catalog")
        materials_response = client.get(
            "/api/sales-knowledge-identification/source-materials"
        )
        document_response = client.get(
            "/api/sales-knowledge-identification/document-packages/DP-API"
        )
        run_response = client.post(
            "/api/sales-knowledge-identification/runs",
            json={"documentPackageId": "DP-API"},
        )
        run_id = run_response.json()["runId"]
        saved_response = client.get(f"/api/sales-knowledge-identification/runs/{run_id}")
        history_response = client.get(
            "/api/sales-knowledge-identification/runs",
            params={"documentPackageId": "DP-API", "limit": 3},
        )
        evaluation_response = client.get(
            "/api/sales-knowledge-identification/evaluations/DP-API"
        )
    finally:
        app.dependency_overrides.clear()

    assert document_response.status_code == 200
    assert materials_response.status_code == 200
    assert materials_response.json()[0]["sourceFileName"] == "sample.pdf"
    assert materials_response.json()[0]["documentPackageId"] == "DP-API"
    assert catalog_response.status_code == 200
    assert len(catalog_response.json()["modules"]) == 22
    assert catalog_response.json()["version"] == "d1-d5-v0.2"
    assert catalog_response.json()["status"] == "sample_validation"
    assert len(catalog_response.json()["fingerprint"]) == 64
    assert catalog_response.json()["source"].endswith(
        "STKB-D1-D5知识对象与业务图模型映射矩阵.md"
    )
    assert catalog_response.json()["modules"][0]["meaning"]
    assert catalog_response.json()["modules"][0]["boundary"]
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "completed"
    assert run_response.json()["modelConfiguration"]["documentMaxChars"] == 3500
    assert run_response.json()["candidates"][0]["candidateId"] == "C1"
    assert saved_response.status_code == 200
    assert saved_response.json() == run_response.json()
    assert history_response.status_code == 200
    assert history_response.json() == [run_response.json()]
    assert evaluation_response.status_code == 200
    assert evaluation_response.json()["markdown"].startswith("# 代理评估")


def test_api_rejects_an_unavailable_document_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = DocumentPackage(
        document_package_id="DP-UNAVAILABLE",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-UNAVAILABLE/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="",
        processing_method="agent_assisted",
        status="unavailable",
        anchors=[],
        quality_issues=["全文不可用"],
    )
    repository = InMemoryRepository(package)
    app.dependency_overrides[get_identification_repository] = lambda: repository
    monkeypatch.setattr(
        identification_api,
        "get_model_gateway",
        lambda: (_ for _ in ()).throw(AssertionError("gateway must not be initialized")),
    )
    client = TestClient(app)

    try:
        response = client.post(
            "/api/sales-knowledge-identification/runs",
            json={"documentPackageId": "DP-UNAVAILABLE"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json()["detail"] == "DocumentPackage is unavailable"
    assert repository.runs == {}
