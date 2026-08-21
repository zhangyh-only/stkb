import json

import pytest

from app.features.sales_knowledge_identification.models import (
    DocumentPackage,
    ModelCompletion,
    ModelRequest,
    SourceAnchor,
)
from app.features.sales_knowledge_identification.segmenter import segment_document
from app.features.sales_knowledge_identification.service import (
    DocumentPackageUnavailable,
    SalesKnowledgeIdentificationService,
)


class StubModelGateway:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(self.payload, ensure_ascii=False),
            prompt_tokens=120,
            completion_tokens=80,
        )


class SequencedModelGateway:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=next(self.contents),
        )


class FlakyModelGateway:
    def __init__(self, successful_payload: dict[str, object]) -> None:
        self.successful_payload = successful_payload
        self.call_count = 0

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("temporary model service failure")
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(self.successful_payload, ensure_ascii=False),
        )


class FailingModelGateway:
    def complete(self, request: ModelRequest) -> ModelCompletion:
        raise RuntimeError("model service unavailable")


class LengthLimitedModelGateway:
    def __init__(self, repaired_payload: dict[str, object]) -> None:
        self.repaired_payload = repaired_payload
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelCompletion(
                provider="test-provider",
                model="test-model",
                content='{"candidates":[{"candidateId":"C1"',
                finish_reason="length",
            )
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(self.repaired_payload, ensure_ascii=False),
            finish_reason="stop",
        )


class SegmentAwareGateway:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        if "DP-SEGMENT#page-1" in request.user_prompt:
            module = "D1.1"
            object_type = "PRODUCT_FACT"
            anchor = "DP-SEGMENT#page-1"
        else:
            module = "D4.3"
            object_type = "QA_PAIR"
            anchor = "DP-SEGMENT#page-2"
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(
                {
                    "candidates": [
                        {
                            "candidateId": "C1",
                            "domain": module.split(".")[0],
                            "module": module,
                            "objectType": object_type,
                            "content": {"summary": anchor},
                            "entityMentions": [],
                            "evidence": [anchor],
                            "relations": [],
                        }
                    ],
                    "weakSignals": [],
                    "unresolvedItems": [],
                }
            ),
            finish_reason="stop",
        )


class CrossSegmentEvidenceGateway:
    def complete(self, request: ModelRequest) -> ModelCompletion:
        candidates: list[dict[str, object]] = []
        if "DP-CROSS#page-1 -->" in request.user_prompt:
            candidates.append(
                {
                    "candidateId": "C1",
                    "domain": "D1",
                    "module": "D1.1",
                    "objectType": "PRODUCT_FACT",
                    "content": {"summary": "错误引用另一分段"},
                    "entityMentions": [],
                    "evidence": ["DP-CROSS#page-2"],
                    "relations": [],
                }
            )
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(
                {"candidates": candidates, "weakSignals": [], "unresolvedItems": []}
            ),
        )


def test_identification_accepts_only_candidates_with_valid_catalog_and_evidence() -> None:
    gateway = StubModelGateway(
        {
            "candidates": [
                {
                    "candidateId": "C1",
                    "domain": "D4",
                    "module": "D4.2",
                    "objectType": "CUSTOMER_OBJECTION",
                    "content": {"rootConcern": "客户担心药品不在保障目录内"},
                    "entityMentions": [
                        {
                            "mentionId": "M1",
                            "text": "药享保",
                            "proposedType": "PRODUCT",
                            "referenceRole": "ABOUT_PRODUCT",
                            "sourceRef": "DP-TEST#page-1",
                        }
                    ],
                    "evidence": ["DP-TEST#page-1"],
                    "relations": [],
                },
                {
                    "candidateId": "C2",
                    "domain": "D9",
                    "module": "D9.1",
                    "objectType": "UNKNOWN",
                    "content": {"summary": "非法分类"},
                    "entityMentions": [],
                    "evidence": ["DP-TEST#page-1"],
                    "relations": [],
                },
                {
                    "candidateId": "C3",
                    "domain": "D1",
                    "module": "D1.1",
                    "objectType": "PRODUCT_FACT",
                    "content": {"summary": "缺少有效证据"},
                    "entityMentions": [],
                    "evidence": ["DP-TEST#page-99"],
                    "relations": [],
                },
            ],
            "weakSignals": [],
            "unresolvedItems": [],
        }
    )
    service = SalesKnowledgeIdentificationService(gateway=gateway)
    document_package = DocumentPackage(
        document_package_id="DP-TEST",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-TEST/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n药品需在保障目录内。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-TEST#page-1", kind="page", page=1)],
        quality_issues=[],
    )

    result = service.identify(document_package)

    assert [candidate.candidate_id for candidate in result.candidates] == ["C1"]
    assert {item.candidate_id for item in result.rejected_candidates} == {"C2", "C3"}
    assert result.coverage_by_module["D4.2"] == "hit"
    assert result.coverage_by_module["D1.1"] == "not_found"
    assert result.call_count == 1
    assert gateway.requests[0].document_package_id == "DP-TEST"
    assert "22个知识内容模块" in gateway.requests[0].system_prompt


def test_identification_rejects_relations_to_a_rejected_candidate() -> None:
    gateway = StubModelGateway(
        {
            "candidates": [
                {
                    "candidateId": "C1",
                    "domain": "D1",
                    "module": "D1.1",
                    "objectType": "PRODUCT_FACT",
                    "content": {"summary": "有效产品"},
                    "entityMentions": [],
                    "evidence": ["DP-REL#page-1"],
                    "relations": [
                        {
                            "relationKind": "object",
                            "relationType": "DEPENDS_ON",
                            "sourceRef": "C1",
                            "targetRef": "C2",
                            "evidence": ["DP-REL#page-1"],
                        }
                    ],
                },
                {
                    "candidateId": "C2",
                    "domain": "D9",
                    "module": "D9.1",
                    "objectType": "UNKNOWN",
                    "content": {"summary": "非法候选"},
                    "entityMentions": [],
                    "evidence": ["DP-REL#page-1"],
                    "relations": [],
                },
            ],
            "weakSignals": [],
            "unresolvedItems": [],
        }
    )
    service = SalesKnowledgeIdentificationService(gateway=gateway)
    package = DocumentPackage(
        document_package_id="DP-REL",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-REL/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n关系验证。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-REL#page-1", kind="page", page=1)],
        quality_issues=[],
    )

    result = service.identify(package)

    assert result.candidates == []
    rejected_by_id = {item.candidate_id: item for item in result.rejected_candidates}
    assert "relation references rejected or missing objects" in rejected_by_id[
        "C1"
    ].reasons[0]


def test_identification_rejects_unavailable_packages_at_capability_boundary() -> None:
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
        quality_issues=["不可用"],
    )

    with pytest.raises(DocumentPackageUnavailable):
        SalesKnowledgeIdentificationService(
            gateway=StubModelGateway({})
        ).identify(package)


def test_identification_canonicalizes_domain_when_model_repeats_module_code() -> None:
    gateway = StubModelGateway(
        {
            "candidates": [
                {
                    "candidateId": "C1",
                    "domain": "D1.3",
                    "module": "D1.3",
                    "objectType": "PROCESS_STEP",
                    "content": {"summary": "提交问诊"},
                    "entityMentions": [],
                    "evidence": ["DP-DOMAIN#page-1"],
                    "relations": [],
                }
            ],
            "weakSignals": [],
            "unresolvedItems": [],
        }
    )
    service = SalesKnowledgeIdentificationService(gateway=gateway)
    package = DocumentPackage(
        document_package_id="DP-DOMAIN",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-DOMAIN/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n提交问诊。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-DOMAIN#page-1", kind="page", page=1)],
        quality_issues=[],
    )

    result = service.identify(package)

    assert len(result.candidates) == 1
    assert result.candidates[0].domain == "D1"
    assert result.coverage_by_module["D1.3"] == "hit"


def test_identification_uses_an_explicit_repair_call_for_invalid_json() -> None:
    repaired_payload = {
        "candidates": [
            {
                "candidateId": "C1",
                "domain": "D1",
                "module": "D1.1",
                "objectType": "PRODUCT_FACT",
                "content": {"summary": "药享保提供在线问诊"},
                "entityMentions": [],
                "evidence": ["DP-REPAIR#page-1"],
                "relations": [],
            }
        ],
        "weakSignals": [],
        "unresolvedItems": [],
    }
    gateway = SequencedModelGateway(
        ["```json\nnot valid json\n```", json.dumps(repaired_payload, ensure_ascii=False)]
    )
    service = SalesKnowledgeIdentificationService(gateway=gateway)
    document_package = DocumentPackage(
        document_package_id="DP-REPAIR",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-REPAIR/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n药享保提供在线问诊。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-REPAIR#page-1", kind="page", page=1)],
        quality_issues=[],
    )

    result = service.identify(document_package)

    assert result.call_count == 2
    assert [trace.purpose for trace in result.model_calls] == [
        "identification",
        "repair",
    ]
    assert result.model_calls[0].raw_output.startswith("```json")
    assert result.model_calls[1].raw_output.startswith('{"candidates"')
    assert "修复" in gateway.requests[1].system_prompt
    assert [candidate.candidate_id for candidate in result.candidates] == ["C1"]


def test_identification_records_failed_model_attempt_before_retrying() -> None:
    gateway = FlakyModelGateway({"candidates": [], "weakSignals": [], "unresolvedItems": []})
    service = SalesKnowledgeIdentificationService(gateway=gateway, max_retries=1)
    document_package = DocumentPackage(
        document_package_id="DP-RETRY",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-RETRY/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n没有销售知识。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-RETRY#page-1", kind="page", page=1)],
        quality_issues=[],
    )

    result = service.identify(document_package)

    assert result.call_count == 2
    assert [trace.status for trace in result.model_calls] == ["failed", "completed"]
    assert result.model_calls[0].error == "temporary model service failure"
    assert result.model_calls[1].attempt == 2


def test_identification_returns_auditable_failed_result_after_final_retry() -> None:
    service = SalesKnowledgeIdentificationService(
        gateway=FailingModelGateway(),
        max_retries=1,
        provider="test-provider",
        model="test-model",
    )
    document_package = DocumentPackage(
        document_package_id="DP-FAILED",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-FAILED/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n模型调用失败。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-FAILED#page-1", kind="page", page=1)],
        quality_issues=[],
    )

    result = service.identify(document_package)

    assert result.status == "failed"
    assert result.provider == "test-provider"
    assert result.model == "test-model"
    assert result.call_count == 2
    assert [call.status for call in result.model_calls] == ["failed", "failed"]
    assert result.processing_stages[0].status == "failed"


def test_identification_reduces_candidate_limit_after_output_truncation() -> None:
    gateway = LengthLimitedModelGateway(
        {"candidates": [], "weakSignals": [], "unresolvedItems": []}
    )
    service = SalesKnowledgeIdentificationService(gateway=gateway, max_candidates=20)
    document_package = DocumentPackage(
        document_package_id="DP-LENGTH",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-LENGTH/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown="# 示例\n\n需要综合识别。",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="DP-LENGTH#page-1", kind="page", page=1)],
        quality_issues=[],
    )

    result = service.identify(document_package)

    assert result.call_count == 2
    assert [trace.purpose for trace in result.model_calls] == [
        "identification",
        "output_limit_retry",
    ]
    assert "最多输出 10 个候选" in gateway.requests[1].system_prompt


def test_identification_aggregates_structural_segments_without_id_collisions() -> None:
    gateway = SegmentAwareGateway()
    service = SalesKnowledgeIdentificationService(
        gateway=gateway,
        document_max_chars=90,
    )
    document_package = DocumentPackage(
        document_package_id="DP-SEGMENT",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-SEGMENT/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown=(
            "# 示例\n\n"
            "## 第 1 页\n\n<!-- source-anchor: DP-SEGMENT#page-1 -->\n\n"
            "产品事实内容。产品事实内容。产品事实内容。\n\n"
            "## 第 2 页\n\n<!-- source-anchor: DP-SEGMENT#page-2 -->\n\n"
            "问答内容。问答内容。问答内容。"
        ),
        processing_method="agent_assisted",
        status="available",
        anchors=[
            SourceAnchor(anchor_id="DP-SEGMENT#page-1", kind="page", page=1),
            SourceAnchor(anchor_id="DP-SEGMENT#page-2", kind="page", page=2),
        ],
        quality_issues=[],
    )

    result = service.identify(document_package)

    assert result.call_count == 2
    assert [candidate.candidate_id for candidate in result.candidates] == [
        "S1-C1",
        "S2-C1",
    ]
    assert result.coverage_by_module["D1.1"] == "hit"
    assert result.coverage_by_module["D4.3"] == "hit"
    assert {trace.segment for trace in result.model_calls} == {"S1/2", "S2/2"}
    assert all("22个知识内容模块" in request.system_prompt for request in gateway.requests)


def test_segmenter_matches_source_anchors_exactly() -> None:
    package = DocumentPackage(
        document_package_id="DP-ANCHOR",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-ANCHOR/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown=(
            "## 第 1 页\n\n<!-- source-anchor: DP-ANCHOR#page-1 -->\n\n一。\n\n"
            "## 第 10 页\n\n<!-- source-anchor: DP-ANCHOR#page-10 -->\n\n十。"
        ),
        processing_method="agent_assisted",
        status="available",
        anchors=[
            SourceAnchor(anchor_id="DP-ANCHOR#page-1", kind="page", page=1),
            SourceAnchor(anchor_id="DP-ANCHOR#page-10", kind="page", page=10),
        ],
        quality_issues=[],
    )

    segments = segment_document(package, max_chars=70)

    assert [[anchor.anchor_id for anchor in segment.anchors] for segment in segments] == [
        ["DP-ANCHOR#page-1"],
        ["DP-ANCHOR#page-10"],
    ]


def test_identification_rejects_evidence_from_another_structural_segment() -> None:
    service = SalesKnowledgeIdentificationService(
        gateway=CrossSegmentEvidenceGateway(), document_max_chars=80
    )
    package = DocumentPackage(
        document_package_id="DP-CROSS",
        workspace_id="WS-TEST",
        source_file_name="sample.pdf",
        source_sha256="source-checksum",
        full_markdown_path="workspace/documents/DP-CROSS/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown=(
            "## 第 1 页\n\n<!-- source-anchor: DP-CROSS#page-1 -->\n\n一段内容。\n\n"
            "## 第 2 页\n\n<!-- source-anchor: DP-CROSS#page-2 -->\n\n二段内容。"
        ),
        processing_method="agent_assisted",
        status="available",
        anchors=[
            SourceAnchor(anchor_id="DP-CROSS#page-1", kind="page", page=1),
            SourceAnchor(anchor_id="DP-CROSS#page-2", kind="page", page=2),
        ],
        quality_issues=[],
    )

    result = service.identify(package)

    assert result.candidates == []
    assert result.rejected_candidates[0].reasons == [
        "unknown evidence anchors: DP-CROSS#page-2"
    ]
