import json

import pytest
from fastapi.testclient import TestClient

from app.api import sales_knowledge_identification as identification_api
from app.api.sales_knowledge_identification import (
    get_identification_repository,
    get_knowledge_formation_service,
)
from app.features.sales_knowledge_identification.content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
)
from app.features.sales_knowledge_identification.formalizer import (
    KnowledgeObjectFormationService,
)
from app.features.sales_knowledge_identification.models import (
    DocumentPackage,
    ModelCompletion,
    SourceAnchor,
    SourceMaterial,
)
from app.features.sales_knowledge_identification.repository import (
    IdentificationRecordNotFound,
)
from app.main import app


class InMemoryRepository:
    def __init__(self, document_package: DocumentPackage) -> None:
        self.document_package = document_package
        self.runs: dict[str, dict[str, object]] = {}
        self.formations: dict[str, dict[str, object]] = {}

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

    def get_existing_entity_ids(self, entity_ids: set[str]) -> set[str]:
        return set()

    def get_existing_object_states(self, object_ids: set[str]) -> dict[str, object]:
        return {}

    def save_knowledge_formation(self, *, workspace_id: str, formation) -> None:  # type: ignore[no-untyped-def]
        self.formations[formation.run_id] = formation.model_dump(mode="json", by_alias=True)

    def get_knowledge_formation(self, run_id: str) -> dict[str, object]:
        return self.formations[run_id]


class ApiStubGateway:
    def complete(self, request):  # type: ignore[no-untyped-def]
        if "原子主张发现器" in request.system_prompt:
            payload = {
                "claims": [
                    {
                        "claimId": "CL1",
                        "claimKind": "fact",
                        "statement": "药享保提供在线问诊服务",
                        "subject": "药享保",
                        "attributes": {},
                        "moduleHints": ["D1.1"],
                        "evidence": [
                            {
                                "anchorId": "DP-API#page-1",
                                "exactQuote": "药享保提供在线问诊服务",
                            }
                        ],
                    }
                ]
            }
        elif "对象边界规划器" in request.system_prompt:
            payload = {
                "objectPlans": [
                    {
                        "planId": "P1",
                        "title": "测试对象 C1",
                        "objectBoundary": "共享测试业务身份与更新边界",
                        "classificationBasis": "依据测试模块规则分类",
                        "identityHints": {
                            "subject": "药享保",
                            "versionScope": "当前版本",
                            "factTheme": "在线问诊服务",
                        },
                        "sourceClaimIds": ["CL1"],
                        "domain": "D1",
                        "module": "D1.1",
                        "objectType": "PRODUCT_FACT",
                    }
                ],
                "weakSignals": [],
                "unresolvedItems": [],
            }
        else:
            payload = {
                "realizations": [
                    {
                        "planId": "P1",
                        "content": _api_contract_content(
                            "D1.1", "药享保提供在线问诊服务"
                        ),
                        "entityMentions": [],
                        "relations": [],
                    }
                ]
            }
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(payload, ensure_ascii=False),
        )


class NoEvaluationRepository(InMemoryRepository):
    def get_evaluation_report(self, document_package_id: str) -> str:
        raise IdentificationRecordNotFound(document_package_id)


def _api_contract_content(module: str, summary: str) -> dict[str, object]:
    contract = CONTENT_CONTRACT_BY_MODULE[module]
    content: dict[str, object] = {
        field: f"测试字段 {field}"
        for field in contract.required_fields
    }
    content["summary"] = summary
    content["contractDetail"] = "用于验证内容合同的结构化测试详情。" * 20
    return content


def test_api_runs_identification_and_reads_the_saved_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
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
    app.dependency_overrides[get_knowledge_formation_service] = lambda: (
        KnowledgeObjectFormationService(project_root=tmp_path)
    )
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
        formation_response = client.post(
            f"/api/sales-knowledge-identification/runs/{run_id}/knowledge-objects"
        )
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
    assert catalog_response.json()["version"] == "d1-d5-v0.4"
    assert catalog_response.json()["status"] == "sample_validation"
    assert len(catalog_response.json()["fingerprint"]) == 64
    assert catalog_response.json()["source"].endswith(
        "STKB-D1-D5知识对象与业务图模型映射矩阵.md"
    )
    assert catalog_response.json()["modules"][0]["meaning"]
    assert catalog_response.json()["modules"][0]["boundary"]
    assert len(catalog_response.json()["domains"]) == 5
    assert catalog_response.json()["domains"][0]["question"] == "卖什么"
    assert set(catalog_response.json()["scopeDefinitions"]) == {"core", "optional"}
    assert catalog_response.json()["contentContractVersion"] == (
        "object-content-contracts-v0.8"
    )
    assert catalog_response.json()["identityContractVersion"] == (
        "object-identity-contracts-v0.2"
    )
    assert catalog_response.json()["modules"][0]["contentContract"][
        "requiredFields"
    ]
    d43 = next(
        module
        for module in catalog_response.json()["modules"]
        if module["code"] == "D4.3"
    )
    assert d43["contentContract"]["requiredFieldsByType"]["TERM"] == [
        "terms",
        "applicability",
    ]
    assert d43["contentContract"]["itemFieldsByType"]["TERM"] == [
        "termText",
        "standardExplanation",
        "sourceStance",
        "usageBoundary",
    ]
    assert catalog_response.json()["modules"][0]["identityContract"][
        "identityFields"
    ]
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "completed"
    assert run_response.json()["modelConfiguration"]["documentMaxChars"] == 3500
    assert run_response.json()["candidates"][0]["candidateId"] == "P1"
    assert formation_response.status_code == 200
    assert formation_response.json()["status"] == "completed"
    assert formation_response.json()["createdCount"] == 1
    assert formation_response.json()["knowledgeObjects"][0]["knowledgeObjectId"].startswith(
        "KO-"
    )
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


def test_missing_optional_evaluation_returns_an_empty_report() -> None:
    package = DocumentPackage(
        document_package_id="DP-NO-EVALUATION",
        workspace_id="WS-TEST",
        source_file_name="sample.md",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-NO-EVALUATION/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例",
        processing_method="agent_assisted",
        status="available",
        anchors=[],
        quality_issues=[],
    )
    app.dependency_overrides[get_identification_repository] = lambda: NoEvaluationRepository(
        package
    )
    client = TestClient(app)

    try:
        response = client.get(
            "/api/sales-knowledge-identification/evaluations/DP-NO-EVALUATION"
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "documentPackageId": "DP-NO-EVALUATION",
        "markdown": "",
    }
