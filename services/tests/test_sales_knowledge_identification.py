import json
from copy import deepcopy

import pytest

from app.features.sales_knowledge_identification.claims import (
    supplement_structured_table_claims,
    validate_atomic_claims,
)
from app.features.sales_knowledge_identification.content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
)
from app.features.sales_knowledge_identification.identity_contracts import (
    IDENTITY_CONTRACT_BY_MODULE,
)
from app.features.sales_knowledge_identification.models import (
    AtomicClaim,
    CandidateObjectPlan,
    ClaimEvidence,
    DocumentPackage,
    ModelCompletion,
    ModelRequest,
    SourceAnchor,
)
from app.features.sales_knowledge_identification.segmenter import segment_document
from app.features.sales_knowledge_identification.service import (
    DocumentPackageUnavailable,
    SalesKnowledgeIdentificationService,
    _enforce_plan_granularity,
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
        elif "对象边界规划器" in request.system_prompt:
            plans = []
            for candidate in self.object_payload.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                plan = deepcopy(candidate)
                plan["planId"] = plan.pop("candidateId")
                for field in ("content", "entityMentions", "relations"):
                    plan.pop(field, None)
                plans.append(plan)
            payload = {
                "objectPlans": plans,
                "weakSignals": self.object_payload.get("weakSignals", []),
                "unresolvedItems": self.object_payload.get("unresolvedItems", []),
            }
        else:
            requested = {
                candidate.get("candidateId")
                for candidate in self.object_payload.get("candidates", [])
                if isinstance(candidate, dict)
                and candidate.get("candidateId") in request.user_prompt
            }
            payload = {
                "realizations": [
                    {
                        "planId": candidate["candidateId"],
                        "content": candidate.get("content", {}),
                        "entityMentions": candidate.get("entityMentions", []),
                        "relations": candidate.get("relations", []),
                    }
                    for candidate in self.object_payload.get("candidates", [])
                    if isinstance(candidate, dict)
                    and candidate.get("candidateId") in requested
                ]
            }
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
        elif "对象边界规划器" in request.system_prompt:
            payload = {
                "objectPlans": [
                    {
                        **_candidate("D1.1", "PRODUCT_FACT", ["S1-CL1"], candidate_id="P1"),
                        "planId": "P1",
                    },
                    {
                        **_candidate("D4.3", "QA_PAIR", ["S2-CL1"], candidate_id="P2"),
                        "planId": "P2",
                    },
                ],
                "weakSignals": [],
                "unresolvedItems": [],
            }
            for plan in payload["objectPlans"]:
                plan.pop("candidateId", None)
                plan.pop("content", None)
                plan.pop("entityMentions", None)
                plan.pop("relations", None)
        else:
            payload = {
                "realizations": [
                    {
                        "planId": plan_id,
                        "content": _contract_content(module, {}),
                        "entityMentions": [],
                        "relations": [],
                    }
                    for plan_id, module in (("P1", "D1.1"), ("P2", "D4.3"))
                    if plan_id in request.user_prompt
                ]
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
        "identityHints": {
            field: f"{candidate_id}-{field}"
            for field in IDENTITY_CONTRACT_BY_MODULE[module].identity_fields
        },
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

    assert [candidate.candidate_id for candidate in result.candidates] == ["C1"]
    assert {item.plan_id for item in result.rejected_object_plans} == {
        "C2",
        "C3",
    }
    assert result.atomic_claims[0].evidence[0].source_text.endswith(
        "药品需在保障目录内。"
    )
    assert result.coverage_by_module["D4.2"] == "hit"
    assert result.call_count == 3
    assert [call.purpose for call in result.model_calls] == [
        "claim_discovery",
        "object_planning",
        "content_realization",
    ]
    assert "原子主张发现器" in gateway.requests[0].system_prompt
    assert "对象边界规划器" in gateway.requests[1].system_prompt
    assert "内容编制器" in gateway.requests[2].system_prompt


def test_uncovered_claim_does_not_promote_module_hint_to_classification() -> None:
    claim = _claim("DP-HINT#page-1", "经验性客户判断", kind="customer_signal")
    claim["moduleHints"] = ["D3.1"]
    gateway = TwoStageGateway(
        [claim],
        {"candidates": [], "weakSignals": [], "unresolvedItems": []},
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-HINT", "# 示例\n\n经验性客户判断。")
    )

    assert len(result.unresolved_items) == 1
    assert result.unresolved_items[0].module is None
    assert "禁止静默丢失" in result.unresolved_items[0].reason


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
    assert "unknown relation reference: C2" in rejected["C1"].reasons


def test_identification_rejects_incomplete_candidate_object_contract() -> None:
    incomplete = {
        "candidateId": "C-INCOMPLETE",
        "domain": "D1",
        "module": "D1.1",
        "objectType": "PRODUCT_FACT",
        "title": "不完整内容对象",
        "objectBoundary": "同一产品与更新周期",
        "classificationBasis": "符合产品事实边界",
        "identityHints": {
            "subject": "药享保",
            "versionScope": "当前版本",
            "factTheme": "产品责任",
        },
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
    assert result.object_plans[0].domain == "D1"


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
            json.dumps(
                {
                    "realizations": [
                        {
                            "planId": "C1",
                            "content": object_payload["candidates"][0]["content"],
                            "entityMentions": [],
                            "relations": [],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-REPAIR", "# 示例\n\n药享保提供在线问诊。")
    )

    assert [trace.purpose for trace in result.model_calls] == [
        "claim_discovery",
        "repair",
        "object_planning",
        "content_realization",
    ]
    assert result.candidates[0].candidate_id == "C1"


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

    assert result.call_count == 5
    assert [candidate.candidate_id for candidate in result.candidates] == ["P1", "P2"]
    assert {claim.claim_id for claim in result.atomic_claims} == {"S1-CL1", "S2-CL1"}
    realization_prompts = [
        request.system_prompt
        for request in gateway.requests
        if "内容编制器" in request.system_prompt
    ]
    assert len(realization_prompts) == 2
    assert any(
        "### D1.1 内容合同" in prompt and "### D4.3 内容合同" not in prompt
        for prompt in realization_prompts
    )
    assert any(
        "### D4.3 内容合同" in prompt and "### D1.1 内容合同" not in prompt
        for prompt in realization_prompts
    )


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


def test_claim_validation_recovers_literal_quote_with_markdown_emphasis() -> None:
    package = _package(
        "DP-MARKDOWN",
        "# 示例\n\n- **A 类（已报价）**：接通电话并完成报价。",
    )
    accepted, rejected = validate_atomic_claims(
        [_claim("DP-MARKDOWN#page-1", "A 类（已报价）：接通电话并完成报价")],
        package,
    )

    assert rejected == []
    assert accepted[0].evidence[0].exact_quote == (
        "A 类（已报价）**：接通电话并完成报价"
    )


def test_claim_validation_supports_anchor_before_its_markdown_heading() -> None:
    package = _package(
        "DP-BEFORE",
        (
            "<!-- source-anchor: DP-BEFORE#section-1 -->\n\n"
            "### 知识点 1\n\n嫌货才是买货人\n\n"
            "<!-- source-anchor: DP-BEFORE#section-2 -->\n\n"
            "### 知识点 2\n\n第二条知识"
        ),
        [
            SourceAnchor(anchor_id="DP-BEFORE#section-1", kind="section"),
            SourceAnchor(anchor_id="DP-BEFORE#section-2", kind="section"),
        ],
    )

    accepted, rejected = validate_atomic_claims(
        [_claim("DP-BEFORE#section-1", "嫌货才是买货人")], package
    )

    assert rejected == []
    assert "第二条知识" not in accepted[0].evidence[0].source_text


def test_blank_selector_from_model_is_treated_as_plain_markdown_evidence() -> None:
    package = _package("DP-BLANK", "# 示例\n\n嫌货才是买货人。")
    raw = _claim("DP-BLANK#page-1", "嫌货才是买货人")
    raw["evidence"][0]["selector"] = ""

    accepted, rejected = validate_atomic_claims([raw], package)

    assert rejected == []
    assert accepted[0].evidence[0].selector is None


def test_claim_validation_uses_complete_selected_cell_when_model_paraphrases_quote() -> None:
    package = _package(
        "DP-CELL",
        (
            "### 第 3 行\n\n<!-- source-anchor: DP-CELL#row-3 -->\n\n"
            "- **D列**：先认同客户顾虑，再说明线上问诊和药品直赔的价值。"
        ),
        [SourceAnchor(anchor_id="DP-CELL#row-3", kind="table")],
    )
    raw = _claim(
        "DP-CELL#row-3",
        "认同顾虑并强调产品价值",
        kind="strategy",
        selector="D列",
    )

    accepted, rejected = validate_atomic_claims([raw], package)

    assert rejected == []
    assert accepted[0].evidence[0].exact_quote == (
        "先认同客户顾虑，再说明线上问诊和药品直赔的价值。"
    )


def test_structured_table_completeness_adds_missing_qa_and_inherited_objection() -> None:
    package = _package(
        "DP-STRUCTURED",
        (
            "### 第 1 行\n\n<!-- source-anchor: DP-STRUCTURED#row-1 -->\n\n"
            "- **A列**：异议处理\n- **B列**：价格贵\n- **D列**：先确认顾虑再解释价值。\n\n"
            "### 第 2 行\n\n<!-- source-anchor: DP-STRUCTURED#row-2 -->\n\n"
            "- **B列**：一次能买几年\n- **D列**：产品按年续交。\n\n"
            "### 第 3 行\n\n<!-- source-anchor: DP-STRUCTURED#row-3 -->\n\n"
            "- **A列**：常见FAQ\n\n"
            "### 第 4 行\n\n<!-- source-anchor: DP-STRUCTURED#row-4 -->\n\n"
            "- **A列**：序号\n- **B列**：问题\n- **C列**：解答\n\n"
            "### 第 5 行\n\n<!-- source-anchor: DP-STRUCTURED#row-5 -->\n\n"
            "- **B列**：如何问诊\n- **C列**：进入服务页发起问诊。"
        ),
        [
            SourceAnchor(anchor_id="DP-STRUCTURED#row-1", kind="table"),
            SourceAnchor(anchor_id="DP-STRUCTURED#row-2", kind="table"),
            SourceAnchor(anchor_id="DP-STRUCTURED#row-3", kind="table"),
            SourceAnchor(anchor_id="DP-STRUCTURED#row-4", kind="table"),
            SourceAnchor(anchor_id="DP-STRUCTURED#row-5", kind="table"),
        ],
    )

    supplemented = supplement_structured_table_claims(package, [])

    assert [(claim.claim_kind, claim.subject) for claim in supplemented] == [
        ("objection", "价格贵"),
        ("objection", "一次能买几年"),
        ("qa", "如何问诊"),
    ]
    assert supplemented[-1].evidence[1].source_text == "进入服务页发起问诊。"


def test_global_planning_can_merge_cross_kind_claims_into_one_object() -> None:
    claims = [
        _claim("DP-GLOBAL#page-1", "尊享版保障责任", kind="fact", claim_id="CL1"),
        _claim("DP-GLOBAL#page-1", "尊享版赔付限制", kind="rule", claim_id="CL2"),
    ]
    candidate = _candidate(
        "D1.1", "PRODUCT_VERSION_FACT", ["CL1", "CL2"], candidate_id="P1"
    )
    gateway = TwoStageGateway(
        claims,
        {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-GLOBAL", "# 示例\n\n尊享版保障责任，尊享版赔付限制。")
    )

    assert len(result.object_plans) == 1
    assert result.object_plans[0].source_claim_ids == ["CL1", "CL2"]
    planning_request = gateway.requests[1]
    assert '"claimKind": "fact"' in planning_request.user_prompt
    assert '"claimKind": "rule"' in planning_request.user_prompt


def test_unassigned_claim_is_exposed_as_unresolved_instead_of_disappearing() -> None:
    gateway = TwoStageGateway(
        [
            _claim("DP-MISSING#page-1", "已覆盖主张", claim_id="CL1"),
            _claim("DP-MISSING#page-1", "遗漏主张", claim_id="CL2"),
        ],
        {
            "candidates": [_candidate("D1.1", "PRODUCT_FACT", ["CL1"])],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-MISSING", "# 示例\n\n已覆盖主张，遗漏主张。")
    )

    assert any("CL2" in item.description for item in result.unresolved_items)
    assert result.coverage_by_module["D1.1"] == "hit"


def test_content_realization_cannot_change_planned_identity_or_classification() -> None:
    candidate = _candidate("D1.1", "PRODUCT_FACT", ["CL1"], candidate_id="P1")

    class MutatingRealizationGateway(TwoStageGateway):
        def complete(self, request: ModelRequest) -> ModelCompletion:
            completion = super().complete(request)
            if "内容编制器" not in request.system_prompt:
                return completion
            payload = json.loads(completion.content)
            payload["realizations"][0].update(
                {
                    "module": "D9.9",
                    "objectType": "MUTATED",
                    "identityHints": {"tampered": True},
                    "sourceClaimIds": ["UNKNOWN"],
                    "relations": [
                        {"source": "药享保", "target": "车险", "type": "ANALOGY"}
                    ],
                }
            )
            return completion.model_copy(
                update={"content": json.dumps(payload, ensure_ascii=False)}
            )

    result = SalesKnowledgeIdentificationService(
        gateway=MutatingRealizationGateway(
            [_claim("DP-LOCKED#page-1", "可信事实")],
            {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
        )
    ).identify(_package("DP-LOCKED", "# 示例\n\n可信事实。"))

    assert result.candidates[0].module == "D1.1"
    assert result.candidates[0].object_type == "PRODUCT_FACT"
    assert result.candidates[0].identity_hints == {
        "subject": "P1-subject",
        "versionScope": "P1-versionScope",
        "factTheme": "P1-factTheme",
    }
    assert result.candidates[0].source_claim_ids == ["CL1"]
    assert result.candidates[0].relations == []
    assert result.normalizations[-1].field == "relations"


def test_granularity_gate_splits_objections_sharing_one_source_anchor() -> None:
    claims = [
        AtomicClaim(
            claim_id=f"CL{index}",
            claim_kind="objection",
            statement=f"客户异议：{subject}",
            subject=subject,
            evidence=[
                ClaimEvidence(
                    anchor_id="DP-SPLIT#section-1",
                    exact_quote=subject,
                    source_text=subject,
                )
            ],
        )
        for index, subject in enumerate(("价格贵", "已经有医保"), start=1)
    ]
    plan = CandidateObjectPlan(
        plan_id="P1",
        title="常见异议",
        domain="D4",
        module="D4.2",
        object_type="CUSTOMER_OBJECTION",
        object_boundary="不同根本顾虑必须拆分",
        classification_basis="属于客户异议",
        identity_hints={"rootConcern": "常见异议", "context": "通用"},
        source_claim_ids=["CL1", "CL2"],
    )

    split = _enforce_plan_granularity([plan], claims)

    assert [item.plan_id for item in split] == ["P1-1", "P1-2"]
    assert [item.identity_hints["rootConcern"] for item in split] == [
        "价格贵",
        "已经有医保",
    ]
    assert [item.source_claim_ids for item in split] == [["CL1"], ["CL2"]]
