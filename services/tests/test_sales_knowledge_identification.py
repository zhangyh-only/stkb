import json
from copy import deepcopy

import pytest

from app.features.sales_knowledge_identification.claims import validate_atomic_claims
from app.features.sales_knowledge_identification.content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
)
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


class TwoStageGateway:
    def __init__(
        self,
        claims: list[dict[str, object]],
        object_payload: dict[str, object] | None = None,
    ) -> None:
        self.claims = deepcopy(claims)
        self.object_payload = deepcopy(
            object_payload
            or {"candidates": [], "weakSignals": [], "unresolvedItems": []}
        )
        for candidate in self.object_payload.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            module = candidate.get("module")
            if (
                isinstance(module, str)
                and module in CONTENT_CONTRACT_BY_MODULE
                and candidate.get("candidateId") != "C-INCOMPLETE"
            ):
                candidate["content"] = _contract_content(
                    module, candidate.get("content", {})
                )
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        payload: dict[str, object]
        if "原子主张发现器" in request.system_prompt:
            payload = {"claims": self.claims}
        else:
            payload = self.object_payload
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(payload, ensure_ascii=False),
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
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content='{"claims":[{"claimId":"CL1"',
            finish_reason="length",
        )


class SegmentAwareGateway:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        if "原子主张发现器" in request.system_prompt:
            if "DP-SEGMENT#page-1" in request.user_prompt:
                claim = _claim(
                    "DP-SEGMENT#page-1", "产品事实内容", kind="fact"
                )
            else:
                claim = _claim("DP-SEGMENT#page-2", "问答内容", kind="qa")
            payload: dict[str, object] = {"claims": [claim]}
        else:
            is_page_one = "S1-CL1" in request.user_prompt
            module = "D1.1" if is_page_one else "D4.3"
            object_type = "PRODUCT_FACT" if is_page_one else "QA_PAIR"
            payload = {
                "candidates": [
                    _candidate(
                        module,
                        object_type,
                        ["S1-CL1" if is_page_one else "S2-CL1"],
                    )
                ],
                "weakSignals": [],
                "unresolvedItems": [],
            }
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(payload, ensure_ascii=False),
        )


class CrossSegmentClaimGateway:
    def complete(self, request: ModelRequest) -> ModelCompletion:
        if "DP-CROSS#page-1" in request.user_prompt:
            payload = {
                "claims": [
                    _claim("DP-CROSS#page-2", "二段内容", kind="fact")
                ]
            }
        else:
            payload = {"claims": []}
        return ModelCompletion(
            provider="test-provider",
            model="test-model",
            content=json.dumps(payload, ensure_ascii=False),
        )


def _contract_content(module: str, original: object) -> dict[str, object]:
    contract = CONTENT_CONTRACT_BY_MODULE[module]
    content = {field: f"测试字段 {field}" for field in contract.required_fields}
    if isinstance(original, dict):
        content.update(original)
    content["contractDetail"] = "用于验证内容合同的结构化测试详情。" * 20
    return content


def _claim(
    anchor: str,
    quote: str,
    *,
    kind: str = "fact",
    claim_id: str = "CL1",
    selector: str | None = None,
) -> dict[str, object]:
    return {
        "claimId": claim_id,
        "claimKind": kind,
        "statement": f"关于{quote}的可核验主张",
        "subject": quote,
        "attributes": {},
        "moduleHints": [],
        "evidence": [
            {
                "anchorId": anchor,
                "exactQuote": quote,
                "selector": selector,
            }
        ],
    }


def _candidate(
    module: str,
    object_type: str,
    source_claim_ids: list[str],
    *,
    candidate_id: str = "C1",
) -> dict[str, object]:
    return {
        "candidateId": candidate_id,
        "title": f"测试对象 {candidate_id}",
        "objectBoundary": "共享测试业务身份与更新边界",
        "classificationBasis": "依据测试模块规则分类",
        "identityHints": {"testKey": candidate_id},
        "domain": module.split(".")[0],
        "module": module,
        "objectType": object_type,
        "sourceClaimIds": source_claim_ids,
        "content": _contract_content(module, {}),
        "entityMentions": [],
        "relations": [],
    }


def _package(
    package_id: str,
    markdown: str,
    anchors: list[SourceAnchor] | None = None,
) -> DocumentPackage:
    return DocumentPackage(
        document_package_id=package_id,
        workspace_id="WS-TEST",
        source_file_name="sample.md",
        source_sha256="source-checksum",
        full_markdown_path=f"workspace/documents/{package_id}/full.md",
        full_markdown_sha256="markdown-checksum",
        full_markdown=markdown,
        processing_method="agent_assisted",
        status="available",
        anchors=anchors
        or [SourceAnchor(anchor_id=f"{package_id}#page-1", kind="page", page=1)],
        quality_issues=[],
    )


def test_two_stage_identification_validates_catalog_and_source_claims() -> None:
    object_payload = {
        "candidates": [
            _candidate("D4.2", "CUSTOMER_OBJECTION", ["CL1"]),
            {
                **_candidate("D1.1", "PRODUCT_FACT", ["UNKNOWN"], candidate_id="C2"),
            },
            {
                **_candidate("D1.1", "PRODUCT_FACT", ["CL1"], candidate_id="C3"),
                "domain": "D9",
                "module": "D9.1",
                "objectType": "UNKNOWN",
            },
        ],
        "weakSignals": [],
        "unresolvedItems": [],
    }
    gateway = TwoStageGateway(
        [_claim("DP-TEST#page-1", "药品需在保障目录内")], object_payload
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-TEST", "# 示例\n\n药品需在保障目录内。")
    )

    assert [candidate.candidate_id for candidate in result.candidates] == ["G1-C1"]
    assert {item.candidate_id for item in result.rejected_candidates} == {
        "G1-C2",
        "G1-C3",
    }
    assert result.atomic_claims[0].evidence[0].source_text.endswith(
        "药品需在保障目录内。"
    )
    assert result.coverage_by_module["D4.2"] == "hit"
    assert result.call_count == 2
    assert [call.purpose for call in result.model_calls] == [
        "claim_discovery",
        "object_formation",
    ]
    assert "原子主张发现器" in gateway.requests[0].system_prompt
    assert "对象形成器" in gateway.requests[1].system_prompt


def test_identification_rejects_relations_to_a_rejected_candidate() -> None:
    first = _candidate("D1.1", "PRODUCT_FACT", ["CL1"])
    first["relations"] = [
        {
            "relationKind": "object",
            "relationType": "DEPENDS_ON",
            "sourceRef": "C1",
            "targetRef": "C2",
            "evidence": ["DP-REL#page-1"],
        }
    ]
    second = _candidate("D1.1", "PRODUCT_FACT", ["UNKNOWN"], candidate_id="C2")
    gateway = TwoStageGateway(
        [_claim("DP-REL#page-1", "关系验证")],
        {
            "candidates": [first, second],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-REL", "# 示例\n\n关系验证。")
    )

    assert result.candidates == []
    rejected = {item.candidate_id: item for item in result.rejected_candidates}
    assert (
        "relation references rejected or missing objects"
        in rejected["G1-C1"].reasons[0]
    )


def test_identification_rejects_incomplete_candidate_object_contract() -> None:
    incomplete = {
        "candidateId": "C-INCOMPLETE",
        "domain": "D1",
        "module": "D1.1",
        "objectType": "PRODUCT_FACT",
        "sourceClaimIds": ["CL1"],
        "content": {"summary": "只有摘要"},
        "entityMentions": [],
        "relations": [],
    }
    gateway = TwoStageGateway(
        [_claim("DP-INCOMPLETE#page-1", "只有摘要")],
        {
            "candidates": [incomplete],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-INCOMPLETE", "# 示例\n\n只有摘要。")
    )

    assert result.candidates == []
    assert set(result.rejected_candidates[0].reasons) >= {
        "candidate title is required",
        "candidate object boundary is required",
        "candidate classification basis is required",
        "candidate identity hints are required",
    }
    assert any(
        "missing required content fields" in reason
        for reason in result.rejected_candidates[0].reasons
    )


def test_identification_rejects_unavailable_packages_at_capability_boundary() -> None:
    package = _package("DP-OFFLINE", "# 示例")
    package = package.model_copy(update={"status": "unavailable"})
    with pytest.raises(DocumentPackageUnavailable):
        SalesKnowledgeIdentificationService(
            gateway=TwoStageGateway([])
        ).identify(package)


def test_identification_canonicalizes_domain_when_model_repeats_module_code() -> None:
    candidate = _candidate("D1.3", "PROCESS_STEP", ["CL1"])
    candidate["domain"] = "D1.3"
    gateway = TwoStageGateway(
        [_claim("DP-DOMAIN#page-1", "提交问诊", kind="process")],
        {
            "candidates": [candidate],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-DOMAIN", "# 示例\n\n提交问诊。")
    )

    assert result.candidates[0].domain == "D1"
    assert result.normalizations[0].original_value == "D1.3"


def test_identification_uses_an_explicit_repair_call_for_invalid_json() -> None:
    claim_payload = {"claims": [_claim("DP-REPAIR#page-1", "药享保提供在线问诊")]}
    object_payload = {
        "candidates": [_candidate("D1.1", "PRODUCT_FACT", ["CL1"])],
        "weakSignals": [],
        "unresolvedItems": [],
    }
    gateway = SequencedModelGateway(
        [
            "```json\nnot valid json\n```",
            json.dumps(claim_payload, ensure_ascii=False),
            json.dumps(object_payload, ensure_ascii=False),
        ]
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-REPAIR", "# 示例\n\n药享保提供在线问诊。")
    )

    assert [trace.purpose for trace in result.model_calls] == [
        "claim_discovery",
        "repair",
        "object_formation",
    ]
    assert result.candidates[0].candidate_id == "G1-C1"


def test_identification_records_failed_model_attempt_before_retrying() -> None:
    gateway = FlakyModelGateway({"claims": []})
    result = SalesKnowledgeIdentificationService(
        gateway=gateway, max_retries=1
    ).identify(_package("DP-RETRY", "# 示例\n\n重试。"))

    assert result.status == "completed"
    assert result.call_count == 2
    assert [trace.status for trace in result.model_calls] == ["failed", "completed"]


def test_identification_returns_auditable_failed_result_after_final_retry() -> None:
    result = SalesKnowledgeIdentificationService(
        gateway=FailingModelGateway(), max_retries=1
    ).identify(_package("DP-FAIL", "# 示例\n\n失败。"))

    assert result.status == "failed"
    assert result.call_count == 2
    assert all(trace.purpose == "claim_discovery" for trace in result.model_calls)
    assert "model service unavailable" in result.processing_stages[0].detail


def test_identification_fails_closed_when_model_output_is_truncated() -> None:
    gateway = LengthLimitedModelGateway()
    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-LENGTH", "# 示例\n\n需要综合识别。")
    )

    assert result.status == "failed"
    assert result.call_count == 1
    assert "truncated" in result.processing_stages[0].detail


def test_identification_runs_discovery_and_object_formation_per_structural_group() -> None:
    gateway = SegmentAwareGateway()
    package = _package(
        "DP-SEGMENT",
        (
            "# 示例\n\n"
            "## 第 1 页\n\n<!-- source-anchor: DP-SEGMENT#page-1 -->\n\n"
            "产品事实内容。产品事实内容。产品事实内容。\n\n"
            "## 第 2 页\n\n<!-- source-anchor: DP-SEGMENT#page-2 -->\n\n"
            "问答内容。问答内容。问答内容。"
        ),
        [
            SourceAnchor(anchor_id="DP-SEGMENT#page-1", kind="page", page=1),
            SourceAnchor(anchor_id="DP-SEGMENT#page-2", kind="page", page=2),
        ],
    )

    result = SalesKnowledgeIdentificationService(
        gateway=gateway, document_max_chars=90
    ).identify(package)

    assert result.call_count == 4
    assert [candidate.candidate_id for candidate in result.candidates] == [
        "G1-C1",
        "G2-C1",
    ]
    assert {claim.claim_id for claim in result.atomic_claims} == {"S1-CL1", "S2-CL1"}


def test_segmenter_matches_source_anchors_exactly_and_supports_spreadsheet_rows() -> None:
    package = _package(
        "DP-ROWS",
        (
            "# 示例\n\n## 工作表：话术\n\n"
            "### 第 1 行\n\n<!-- source-anchor: DP-ROWS#row-1 -->\n\n- **D列**：第一行内容。\n\n"
            "### 第 2 行\n\n<!-- source-anchor: DP-ROWS#row-2 -->\n\n- **D列**：第二行内容。"
        ),
        [
            SourceAnchor(anchor_id="DP-ROWS#row-1", kind="table"),
            SourceAnchor(anchor_id="DP-ROWS#row-2", kind="table"),
        ],
    )

    segments = segment_document(package, max_chars=70)

    assert len(segments) == 2
    assert [anchor.anchor_id for anchor in segments[0].anchors] == ["DP-ROWS#row-1"]
    assert [anchor.anchor_id for anchor in segments[1].anchors] == ["DP-ROWS#row-2"]


def test_claim_validation_rejects_cross_segment_evidence() -> None:
    package = _package(
        "DP-CROSS",
        (
            "## 第 1 页\n\n<!-- source-anchor: DP-CROSS#page-1 -->\n\n一段内容。\n\n"
            "## 第 2 页\n\n<!-- source-anchor: DP-CROSS#page-2 -->\n\n二段内容。"
        ),
        [
            SourceAnchor(anchor_id="DP-CROSS#page-1", kind="page", page=1),
            SourceAnchor(anchor_id="DP-CROSS#page-2", kind="page", page=2),
        ],
    )

    result = SalesKnowledgeIdentificationService(
        gateway=CrossSegmentClaimGateway(), document_max_chars=55
    ).identify(package)

    assert result.candidates == []
    assert result.atomic_claims == []
    assert result.rejected_atomic_claims[0].reasons == [
        "unknown evidence anchor: DP-CROSS#page-2"
    ]


def test_claim_selector_and_verbatim_macro_preserve_complete_source_field() -> None:
    package = _package(
        "DP-SCRIPT",
        (
            "### 第 3 行\n\n<!-- source-anchor: DP-SCRIPT#row-3 -->\n\n"
            "- **C列**：忙碌上班族推荐\n"
            "- **D列**：先生您好，这是一段必须被完整保留的标准销售话术。"
        ),
        [SourceAnchor(anchor_id="DP-SCRIPT#row-3", kind="table")],
    )
    claim = _claim(
        "DP-SCRIPT#row-3",
        "必须被完整保留",
        kind="script",
        selector="D列",
    )
    candidate = _candidate("D4.1", "STANDARD_SCRIPT", ["CL1"])
    candidate["content"] = _contract_content(
        "D4.1", {"script": {"$verbatimFromClaim": "CL1"}}
    )

    result = SalesKnowledgeIdentificationService(
        gateway=TwoStageGateway(
            [claim],
            {
                "candidates": [candidate],
                "weakSignals": [],
                "unresolvedItems": [],
            },
        )
    ).identify(package)

    assert result.atomic_claims[0].evidence[0].source_text == (
        "先生您好，这是一段必须被完整保留的标准销售话术。"
    )
    assert result.candidates[0].content["script"] == (
        "先生您好，这是一段必须被完整保留的标准销售话术。"
    )


def test_validate_atomic_claims_rejects_non_verbatim_quote() -> None:
    package = _package("DP-QUOTE", "# 示例\n\n药享保提供在线问诊。")
    accepted, rejected = validate_atomic_claims(
        [_claim("DP-QUOTE#page-1", "药享保提供线下问诊")], package
    )

    assert accepted == []
    assert rejected[0].reasons == ["exact quote not found in DP-QUOTE#page-1"]
