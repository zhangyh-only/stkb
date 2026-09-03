import json
from copy import deepcopy

import pytest

from app.features.sales_knowledge_identification.catalog import MODULE_BY_OBJECT_TYPE
from app.features.sales_knowledge_identification.claims import (
    resolve_verbatim_claim_references,
    supplement_explicit_internal_term_claims,
    supplement_numbered_qa_claims,
    supplement_structured_table_claims,
    validate_atomic_claims,
)
from app.features.sales_knowledge_identification.content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
    FIELD_TYPES_BY_OBJECT_TYPE,
)
from app.features.sales_knowledge_identification.identity_contracts import (
    IDENTITY_CONTRACT_BY_MODULE,
)
from app.features.sales_knowledge_identification.models import (
    AtomicClaim,
    CandidateObjectPlan,
    ClaimEvidence,
    ContentClaimUsage,
    DocumentPackage,
    ModelCompletion,
    ModelRequest,
    SourceAnchor,
)
from app.features.sales_knowledge_identification.prompt_builder import (
    build_document_object_planning_request,
)
from app.features.sales_knowledge_identification.segmenter import segment_document
from app.features.sales_knowledge_identification.service import (
    DocumentPackageUnavailable,
    SalesKnowledgeIdentificationService,
    _apply_plan_augmentations,
    _automatic_uncovered_claim_ids,
    _auxiliary_claim_id,
    _claim_explicitly_all_versions,
    _constrain_plan_source_claim_scope,
    _enforce_plan_granularity,
    _ensure_enumeration_dimension_plans,
    _ensure_explicit_script_plans,
    _ensure_explicit_term_plans,
    _ensure_rule_policy_plans,
    _expand_compact_claim_payload,
    _expand_compact_object_plan,
    _expand_content_path_to_leaf_paths,
    _group_claims_for_planning,
    _merge_repair_plans,
    _normalize_content_shape,
    _object_granularity_metrics,
    _plan_satisfies_primary_claim_role,
    _prune_unattributed_d33_inferences,
    _split_composite_product_version_plan,
    _split_explicit_strategy_combinations,
    _supplement_exact_match_claim_usage,
    _unclaimed_source_anchor_inputs,
    _validate_content_claim_usage,
    _validate_object_plans,
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
                    module,
                    candidate.get("content", {}),
                    str(candidate.get("objectType", "")) or None,
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
                for field in (
                    "content",
                    "claimUsage",
                    "omittedClaims",
                    "entityMentions",
                    "relations",
                ):
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
                        **(
                            {"claimUsage": candidate["claimUsage"]}
                            if "claimUsage" in candidate
                            else {}
                        ),
                        **(
                            {"omittedClaims": candidate["omittedClaims"]}
                            if "omittedClaims" in candidate
                            else {}
                        ),
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
                        **_candidate("D4.2", "QA_PAIR", ["S2-CL1"], candidate_id="P2"),
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
                        "content": _contract_content(module, {}, object_type),
                        "entityMentions": [],
                        "relations": [],
                    }
                    for plan_id, module, object_type in (
                        ("P1", "D1.1", "PRODUCT_FACT"),
                        ("P2", "D4.2", "QA_PAIR"),
                    )
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


def _contract_content(
    module: str, original: object, object_type: str | None = None
) -> dict[str, object]:
    contract = CONTENT_CONTRACT_BY_MODULE[module]
    selected_type = object_type or contract.object_types[0]
    field_types = FIELD_TYPES_BY_OBJECT_TYPE.get(selected_type, {})
    content: dict[str, object] = {}
    for field in contract.required_fields_by_type.get(
        selected_type, contract.required_fields
    ):
        expected_type = field_types.get(field, str)
        if field == "items":
            content[field] = [
                {
                    "question": "测试问题",
                    "answer": "测试答案",
                    "claimRef": "CL1",
                }
            ]
        elif field == "facts":
            content[field] = [{"description": "测试事实"}]
        elif field == "rootConcernHypotheses":
            content[field] = []
        elif field == "resolutionElements":
            content[field] = [
                {
                    "element": "先确认客户表达",
                    "detail": "依据资料提供的回复说明处理方向和事实边界。",
                }
            ]
        elif expected_type is list:
            content[field] = [f"测试字段 {field}"]
        elif expected_type is dict:
            content[field] = {"scope": f"测试字段 {field}"}
        else:
            content[field] = f"测试字段 {field}"
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


def test_planning_batches_keep_source_location_together() -> None:
    claims = [
        AtomicClaim.model_validate(_claim("DP-BATCH#row-1", "事实一", claim_id="CL1")),
        AtomicClaim.model_validate(_claim("DP-BATCH#row-1", "事实二", claim_id="CL2")),
        AtomicClaim.model_validate(_claim("DP-BATCH#row-2", "事实三", claim_id="CL3")),
    ]

    batches = _group_claims_for_planning(claims, max_claims=2)

    assert [[claim.claim_id for claim in batch] for _label, batch in batches] == [
        ["CL1", "CL2"],
        ["CL3"],
    ]


def test_compact_claim_supports_multiple_verbatim_evidence_spans() -> None:
    payload = _expand_compact_claim_payload(
        {
            "claims": [
                [
                    "CL1",
                    "fact",
                    "产品事实",
                    {},
                    [
                        ["DP#page-1", "第一段原文...附加原文", None],
                        ["DP#page-2", "第二段原文", None],
                    ],
                ]
            ]
        }
    )

    assert payload["claims"][0]["statement"] == "第一段原文；附加原文；第二段原文"
    assert len(payload["claims"][0]["evidence"]) == 3


def test_integrated_planning_keeps_versions_strategies_and_qa_boundaries() -> None:
    request = build_document_object_planning_request(
        _package("DP-PLAN", "# 资料\n\n产品与问答"), max_candidates=20
    )

    assert "不同版本存在不同价格" in request.system_prompt
    assert "每个组合形成独立 SALES_STRATEGY" in request.system_prompt
    assert "每个问题及答案分别形成 qa claim" in request.system_prompt
    assert "最多\n20 个" in request.system_prompt


def test_claim_validation_recovers_exact_span_across_markdown_table_formatting() -> None:
    package = _package(
        "DP-TABLE-QUOTE",
        "<!-- source-anchor: DP-TABLE-QUOTE#page-1 -->\n| 版本 | 尊享版 | 全能版 |",
        [SourceAnchor(anchor_id="DP-TABLE-QUOTE#page-1", kind="table")],
    )
    claim = {
        **_claim(
            "DP-TABLE-QUOTE#page-1",
            "版本 尊享版 全能版",
            claim_id="CL1",
        ),
        "evidence": [
            {
                "anchorId": "DP-TABLE-QUOTE#page-1",
                "exactQuote": "版本 尊享版 全能版",
            }
        ],
    }

    accepted, rejected = validate_atomic_claims([claim], package)

    assert rejected == []
    assert accepted[0].evidence[0].exact_quote == "版本 | 尊享版 | 全能版"


def test_claim_validation_recovers_ordered_source_lines_with_intervening_labels() -> None:
    package = _package(
        "DP-LINE-QUOTE",
        (
            "<!-- source-anchor: DP-LINE-QUOTE#page-1 -->\n"
            "每笔订单不超过4种\n药房购药\n每种药品不超过1盒"
        ),
        [SourceAnchor(anchor_id="DP-LINE-QUOTE#page-1", kind="page")],
    )
    claim = {
        **_claim(
            "DP-LINE-QUOTE#page-1",
            "每笔订单不超过4种 每种药品不超过1盒",
            claim_id="CL1",
        ),
        "evidence": [
            {
                "anchorId": "DP-LINE-QUOTE#page-1",
                "exactQuote": "每笔订单不超过4种 每种药品不超过1盒",
            }
        ],
    }

    accepted, rejected = validate_atomic_claims([claim], package)

    assert rejected == []
    assert "药房购药" in accepted[0].evidence[0].exact_quote

def test_object_granularity_metrics_expose_single_claim_and_anchor_splits() -> None:
    claims = [
        AtomicClaim.model_validate(
            _claim("DP-METRICS#section-1", text, claim_id=f"CL{index}")
        )
        for index, text in enumerate(("事实一", "事实二"), start=1)
    ]
    plans = [
        CandidateObjectPlan(
            plan_id="P1",
            title="聚合对象",
            domain="D1",
            module="D1.1",
            object_type="PRODUCT_FACT",
            identity_hints={},
            source_claim_ids=["CL1", "CL2"],
        ),
        CandidateObjectPlan(
            plan_id="P2",
            title="单主张对象",
            domain="D1",
            module="D1.1",
            object_type="PRODUCT_FACT",
            identity_hints={},
            source_claim_ids=["CL2"],
        ),
    ]

    metrics = _object_granularity_metrics(plans, claims)

    assert metrics.single_claim_object_rate == 0.5
    assert metrics.average_claims_per_object == 1.5
    assert metrics.source_anchors_split_across_objects == 1


def test_multiple_planning_batches_receive_unique_plan_ids() -> None:
    first_claim = AtomicClaim.model_validate(
        _claim("DP-BATCH#row-1", "产品事实", claim_id="CL1")
    )
    second_claim = AtomicClaim.model_validate(
        _claim("DP-BATCH#row-2", "另一个产品事实", claim_id="CL2")
    )
    results = [
        (
            "planning-1",
            [first_claim],
            {
                "objectPlans": [
                    [
                        "P1",
                        "产品事实一",
                        "D1.1",
                        "PRODUCT_FACT",
                        {
                            "subject": "产品A",
                            "versionScope": "版本A",
                            "factTheme": "价格",
                        },
                        ["CL1"],
                    ]
                ]
            },
            None,
            [],
        ),
        (
            "planning-2",
            [second_claim],
            {
                "objectPlans": [
                    [
                        "P1",
                        "产品事实二",
                        "D1.1",
                        "PRODUCT_FACT",
                        {
                            "subject": "产品B",
                            "versionScope": "版本B",
                            "factTheme": "责任",
                        },
                        ["CL2"],
                    ]
                ]
            },
            None,
            [],
        ),
    ]

    plans, rejected, _weak, _unresolved = _validate_object_plans(
        results, [first_claim, second_claim]
    )

    assert rejected == []
    assert [plan.plan_id for plan in plans] == ["planning-1-P1", "planning-2-P1"]


def test_named_version_product_fact_is_normalized_to_version_fact() -> None:
    claim = AtomicClaim.model_validate(
        _claim("DP-VERSION#row-1", "全能版药品覆盖", claim_id="CL1")
    )
    results = [
        (
            "planning-1",
            [claim],
            {
                "objectPlans": [
                    [
                        "P1",
                        "全能版药品覆盖事实",
                        "D1.1",
                        "PRODUCT_FACT",
                        {
                            "subject": "药享保",
                            "versionScope": "全能版语境",
                            "factTheme": "药品覆盖",
                        },
                        ["CL1"],
                    ]
                ]
            },
            None,
            [],
        )
    ]

    plans, rejected, _weak, _unresolved = _validate_object_plans(results, [claim])

    assert rejected == []
    assert plans[0].object_type == "PRODUCT_VERSION_FACT"


def test_unknown_version_product_fact_stays_unresolved() -> None:
    claim = AtomicClaim.model_validate(
        _claim("DP-VERSION#row-1", "药享保年费180元", claim_id="CL1")
    )
    results = [
        (
            "planning-1",
            [claim],
            {
                "objectPlans": [
                    [
                        "P1",
                        "药享保价格事实",
                        "D1.1",
                        "PRODUCT_FACT",
                        {
                            "subject": "药享保",
                            "versionScope": "未明确版本，需核实",
                            "factTheme": "保费价格",
                        },
                        ["CL1"],
                    ]
                ]
            },
            None,
            [],
        )
    ]

    plans, rejected, _weak, unresolved = _validate_object_plans(results, [claim])

    assert plans == []
    assert any(
        "unknown version must remain unresolved" in reason
        for item in rejected
        for reason in item.reasons
    )
    assert unresolved[0][0]["claimId"] == "CL1"


def test_objection_identity_uses_observable_customer_expression() -> None:
    raw_claim = _claim(
        "DP-OBJECTION#row-1",
        "可以一次性买几年的吗",
        kind="objection",
        claim_id="CL1",
    )
    raw_claim["attributes"] = {
        "expression": "可以一次性买几年的吗",
        "responseContext": "产品按年续交。",
    }
    claim = AtomicClaim.model_validate(raw_claim)
    results = [
        (
            "planning-1",
            [claim],
            {
                "objectPlans": [
                    [
                        "P1",
                        "缴费周期异议",
                            "D4.1",
                        "CUSTOMER_OBJECTION",
                        {
                            "objectionIntent": "担心涨价并希望锁定长期权益",
                            "context": "产品咨询",
                        },
                        ["CL1"],
                    ]
                ]
            },
            None,
            [],
        )
    ]

    plans, rejected, _weak, _unresolved = _validate_object_plans(results, [claim])

    assert rejected == []
    assert plans[0].identity_hints["objectionIntent"] == "可以一次性买几年的吗"


def test_script_plan_drops_supporting_claims_from_unrelated_source_rows() -> None:
    script_claim = AtomicClaim.model_validate(
        _claim("DP-SCOPE#row-1", "完整销售话术", kind="script", claim_id="CL1")
    )
    same_row_fact = AtomicClaim.model_validate(
        _claim("DP-SCOPE#row-1", "话术引用事实", claim_id="CL2")
    )
    unrelated_fact = AtomicClaim.model_validate(
        _claim("DP-SCOPE#row-9", "其他场景事实", claim_id="CL3")
    )
    plan = CandidateObjectPlan(
        plan_id="P1",
        title="销售话术",
        domain="D4",
        module="D4.1",
        object_type="STANDARD_SCRIPT",
        object_boundary="同一沟通目标",
        classification_basis="来源提供完整话术",
        identity_hints={
            "communicationGoal": "产品引入",
            "method": "事实说明",
            "applicability": "通用",
        },
        source_claim_ids=["CL1", "CL2", "CL3"],
    )

    constrained = _constrain_plan_source_claim_scope(
        plan,
        {claim.claim_id: claim for claim in [script_claim, same_row_fact, unrelated_fact]},
    )

    assert constrained.source_claim_ids == ["CL1", "CL2"]


def test_strategy_plan_keeps_same_anchor_script_as_supporting_evidence() -> None:
    strategy_claim = AtomicClaim.model_validate(
        _claim("DP-STRATEGY#row-1", "产品引入策略", kind="strategy", claim_id="CL1")
    )
    script_claim = AtomicClaim.model_validate(
        _claim("DP-STRATEGY#row-1", "面向优质客户", kind="script", claim_id="CL2")
    )
    results = [
        (
            "planning-1",
            [strategy_claim, script_claim],
            {
                "objectPlans": [
                    [
                        "P1",
                        "产品引入策略",
                            "D3.2",
                        "SALES_STRATEGY",
                        {
                            "strategyGoal": "产品引入",
                            "triggerContext": "面向优质客户",
                            "applicability": "测试产品",
                        },
                        ["CL1"],
                    ]
                ]
            },
            None,
            [],
        )
    ]

    plans, rejected, _weak, _unresolved = _validate_object_plans(
        results, [strategy_claim, script_claim]
    )

    assert rejected == []
    assert plans[0].source_claim_ids == ["CL1", "CL2"]


def test_verified_script_claim_gets_independent_d41_plan_when_planner_only_uses_strategy() -> None:
    script_payload = _claim(
        "DP-SCRIPT-GUARD#row-1",
        "完整异议处理话术",
        kind="script",
        claim_id="CL-SCRIPT",
    )
    script_payload["attributes"] = {
        "communicationGoal": "异议化解与促成",
        "audience": "犹豫客户",
    }
    script_claim = AtomicClaim.model_validate(script_payload)
    strategy_claim = AtomicClaim.model_validate(
        _claim(
            "DP-SCRIPT-GUARD#row-1",
            "探询顾虑后强化价值",
            kind="strategy",
            claim_id="CL-STRATEGY",
        )
    )
    strategy_plan = CandidateObjectPlan(
        plan_id="P1",
        title="异议处理策略",
        domain="D3",
        module="D3.2",
        object_type="SALES_STRATEGY",
        object_boundary="同一策略目标与适用范围",
        classification_basis="来源提供策略主张",
        identity_hints={
            "strategyGoal": "异议化解",
            "triggerContext": "客户犹豫",
            "applicability": "药享保",
        },
        source_claim_ids=["CL-SCRIPT", "CL-STRATEGY"],
    )

    plans = _ensure_explicit_script_plans(
        [strategy_plan], [script_claim, strategy_claim]
    )

    assert plans[0].source_claim_ids == ["CL-SCRIPT", "CL-STRATEGY"]
    assert plans[1].module == "D4.2"
    assert plans[1].object_type == "STANDARD_SCRIPT"
    assert plans[1].source_claim_ids == ["CL-SCRIPT"]
    assert plans[1].identity_hints == {
        "communicationGoal": "异议化解与促成",
        "method": "完整原文复用",
        "applicability": "犹豫客户",
    }


def test_script_guard_groups_variants_with_same_goal_and_audience() -> None:
    claims = []
    for index, text in enumerate(("先确认现有流程", "再说明方案价值"), start=1):
        payload = _claim(
            f"DP-SCRIPT-GROUP#row-{index}", text, kind="script", claim_id=f"CL{index}"
        )
        payload["attributes"] = {
            "communicationGoal": "说明方案价值",
            "audience": "财务负责人",
        }
        claims.append(AtomicClaim.model_validate(payload))

    plans = _ensure_explicit_script_plans([], claims)

    assert len(plans) == 1
    assert plans[0].source_claim_ids == ["CL1", "CL2"]


def test_template_reference_does_not_become_standard_script_plan() -> None:
    payload = _claim(
        "DP-TEMPLATE#row-1",
        "客户询问信息来源时，必须按照范本说",
        kind="script",
        claim_id="CL-TEMPLATE",
    )
    claim = AtomicClaim.model_validate(payload)

    assert _ensure_explicit_script_plans([], [claim]) == []


def test_explicit_internal_identifier_definition_gets_term_duty() -> None:
    package = _package(
        "DP-TERM",
        "<!-- source-anchor: DP-TERM#rule-1 -->\n1010的电话是质检的，内部使用。",
        [SourceAnchor(anchor_id="DP-TERM#rule-1", kind="section")],
    )

    claims = supplement_explicit_internal_term_claims(package, [])
    plans = _ensure_explicit_term_plans([], claims)

    assert len(claims) == 1
    assert claims[0].subject == "1010电话"
    assert claims[0].evidence[0].exact_quote == "1010的电话是质检的"
    assert len(plans) == 1
    assert plans[0].module == "D4.1"
    assert plans[0].object_type == "TERM"


def test_term_guard_groups_one_document_glossary_across_anchors() -> None:
    claims = [
        AtomicClaim.model_validate(
            _claim(f"DP-TERM-GROUP#section-{index}", term, kind="term", claim_id=f"CL{index}")
        )
        for index, term in enumerate(("商机", "有效线索"), start=1)
    ]

    plans = _ensure_explicit_term_plans([], claims)

    assert len(plans) == 1
    assert plans[0].source_claim_ids == ["CL1", "CL2"]


def test_rule_and_strategy_section_keeps_independent_policy_duty() -> None:
    anchor = "DP-REGION#rule-1"
    rule_one = AtomicClaim.model_validate(
        {
            **_claim(anchor, "上海身份证可以投保", kind="rule", claim_id="CL1"),
            "attributes": {"condition": "上海身份证", "action": "可以投保"},
        }
    )
    rule_two = AtomicClaim.model_validate(
        {
            **_claim(anchor, "双非客户大概率不通过", kind="rule", claim_id="CL2"),
            "attributes": {"condition": "双非客户", "outcome": "大概率不通过"},
        }
    )
    strategy = AtomicClaim.model_validate(
        _claim(anchor, "不通过时只推荐健康险", kind="strategy", claim_id="CL3")
    )
    plan = CandidateObjectPlan(
        plan_id="P1",
        title="异地客户产品选择策略",
        domain="D3",
        module="D3.2",
        object_type="SALES_STRATEGY",
        identity_hints={
            "strategyGoal": "受限客户产品选择",
            "triggerContext": "异地客户",
            "applicability": "上海分公司",
        },
        source_claim_ids=["CL1", "CL2", "CL3"],
    )

    guarded = _ensure_rule_policy_plans(
        [plan], [rule_one, rule_two, strategy]
    )

    policy = next(item for item in guarded if item.module == "D1.2")
    assert policy.object_type == "POLICY_RULE_SET"
    assert policy.source_claim_ids == ["CL1", "CL2"]


def test_malformed_auxiliary_item_has_no_claim_id_before_validation() -> None:
    assert _auxiliary_claim_id(("模型错误输出", {"DP#row-1"})) is None
    assert _auxiliary_claim_id(({"claimId": "CL1"}, {"DP#row-1"})) == "CL1"


def test_exact_source_text_can_recover_missing_critical_attribution() -> None:
    claim = AtomicClaim.model_validate(
        _claim(
            "DP-ATTR#row-1",
            "根据客户特点定制",
            kind="strategy",
            claim_id="CL1",
        )
    )
    content = {
        "strategyName": "全能版引入策略",
        "triggerConditions": ["根据客户特点定制"],
        "decisionLogic": "强调产品特点",
        "actions": ["说明产品范围"],
        "applicability": {"products": ["全能版"], "scenarios": ["产品引入"]},
    }

    usage = _supplement_exact_match_claim_usage(
        module="D3.2",
        object_type="SALES_STRATEGY",
        content=content,
        source_claims=[claim],
        existing_usage=[],
    )

    assert [(item.claim_id, item.content_paths) for item in usage] == [
        ("CL1", ["$.triggerConditions[0]"])
    ]


def test_d33_prunes_only_unattributed_extra_actions() -> None:
    content = {
        "strategyName": "回访引入策略",
        "triggerConditions": ["续保客户回访"],
        "decisionLogic": "以调研降低防备",
        "actions": ["以回访调研发起对话", "建立信任后自然转向推销"],
        "applicability": {"products": ["药享保"], "scenarios": ["客户回访"]},
    }

    normalized, _usage = _prune_unattributed_d33_inferences(
        content,
        {
            "$.triggerConditions[0]",
            "$.decisionLogic",
            "$.actions[0]",
        },
    )

    assert normalized["actions"] == ["以回访调研发起对话"]


def _candidate(
    module: str,
    object_type: str,
    source_claim_ids: list[str],
    *,
    candidate_id: str = "C1",
) -> dict[str, object]:
    module = MODULE_BY_OBJECT_TYPE[object_type].code
    return {
        "candidateId": candidate_id,
        "title": f"测试对象 {candidate_id}",
        "objectBoundary": "共享测试业务身份与更新边界",
        "classificationBasis": "依据测试模块规则分类",
        "identityHints": {
            field: f"{candidate_id}-{field}"
            for field in IDENTITY_CONTRACT_BY_MODULE[
                module
            ].identity_fields_by_type[object_type]
        },
        "domain": module.split(".")[0],
        "module": module,
        "objectType": object_type,
        "sourceClaimIds": source_claim_ids,
        "content": _contract_content(module, {}, object_type),
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


def test_content_claim_usage_distinguishes_planned_evidence_from_written_content() -> None:
    package_id = "DP-CONTENT-USAGE"
    anchor = f"{package_id}#page-1"
    candidate = _candidate("D1.1", "PRODUCT_FACT", ["CL1", "CL2"])
    content_paths = [
        f"$.{field}"
        for field, value in candidate["content"].items()
        if value not in (None, "", [], {})
    ]
    object_payload = {
        "candidates": [
            {
                **candidate,
                "claimUsage": [
                    {
                        "claimId": "CL1",
                        "role": "primary",
                        "contentPaths": content_paths,
                        "explanation": "正文详情表达了该产品事实",
                    }
                ],
                "omittedClaims": [
                    {"claimId": "CL2", "reason": "该主张未实际写入当前对象正文"}
                ],
            }
        ],
        "weakSignals": [],
        "unresolvedItems": [],
    }
    gateway = TwoStageGateway(
        [
            _claim(anchor, "事实一", claim_id="CL1"),
            _claim(anchor, "事实二", claim_id="CL2"),
        ],
        object_payload,
    )
    service = SalesKnowledgeIdentificationService(gateway)

    result = service.identify(_package(package_id, "事实一；事实二"))

    assert result.status == "completed"
    assert result.candidates[0].planned_source_claim_ids == ["CL1", "CL2"]
    assert result.candidates[0].source_claim_ids == ["CL1"]
    assert [item.claim_id for item in result.candidates[0].claim_usage] == ["CL1"]
    assert any(
        item.claim_id == "CL2" and "未实际写入" in item.reason
        for item in result.unresolved_items
    )
    assert sum(item.claim_id == "CL2" for item in result.unresolved_items) == 1


def test_missing_critical_claim_attribution_blocks_formalization_quality() -> None:
    package_id = "DP-CRITICAL-USAGE"
    anchor = f"{package_id}#page-1"
    candidate = _candidate("D1.1", "PRODUCT_FACT", ["CL1"])
    candidate["claimUsage"] = [
        {
            "claimId": "CL1",
            "role": "primary",
            "contentPaths": ["$.summary"],
            "explanation": "仅把主张写入摘要，没有逐项支撑事实正文",
        }
    ]
    candidate["content"] = {
        **candidate["content"],
        "summary": "产品事实摘要",
    }
    gateway = TwoStageGateway(
        [_claim(anchor, "药享保提供在线问诊", claim_id="CL1")],
        {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
    )

    result = SalesKnowledgeIdentificationService(gateway).identify(
        _package(package_id, "药享保提供在线问诊")
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].quality_issues == [
        "关键正文缺少主张归因：$.facts[0].description"
    ]


def test_d13_prunes_unattributed_inferred_preconditions_and_exception_handling() -> None:
    package_id = "DP-D13-ATTRIBUTION"
    anchor = f"{package_id}#page-1"
    candidate = _candidate("D1.3", "BUSINESS_PROCESS", ["CL1", "CL2"])
    candidate["content"] = {
        "purpose": "完成线上问诊购药",
        "preconditions": ["用户必须已持有有效权益"],
        "rulesOrSteps": [
            {"stepOrder": 1, "description": "进入在线问诊并填写病情。"},
            {"stepOrder": 2, "description": "医生问诊后开具处方。"},
        ],
        "exceptions": [
            {
                "condition": "医生可能基于用药安全调整药品和剂量。",
                "handling": "用户应无条件接受调整。",
            }
        ],
        "contractDetail": "用于验证没有来源归因的推演字段会被清理。" * 20,
    }
    candidate["claimUsage"] = [
        {
            "claimId": "CL1",
            "role": "primary",
            "contentPaths": [
                "$.rulesOrSteps[0].description",
                "$.rulesOrSteps[1].description",
            ],
            "explanation": "来源明确给出两个流程步骤",
        },
        {
            "claimId": "CL2",
            "role": "supporting",
            "contentPaths": ["$.exceptions[0].condition"],
            "explanation": "来源只给出异常条件，没有给出处置动作",
        },
    ]
    gateway = TwoStageGateway(
        [
            _claim(anchor, "进入在线问诊并由医生开具处方", kind="process", claim_id="CL1"),
            _claim(anchor, "医生可能调整药品和剂量", kind="rule", claim_id="CL2"),
        ],
        {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
    )

    result = SalesKnowledgeIdentificationService(gateway).identify(
        _package(package_id, "进入在线问诊并由医生开具处方；医生可能调整药品和剂量。")
    )

    assert result.candidates[0].content["preconditions"] == []
    assert result.candidates[0].content["exceptions"] == [
        {"condition": "医生可能基于用药安全调整药品和剂量。"}
    ]
    assert result.candidates[0].quality_issues == []
    assert any(
        item.field == "content.attributionPruning"
        for item in result.normalizations
    )


def test_objection_expression_is_normalized_to_customer_source_field() -> None:
    package_id = "DP-OBJECTION-EXPRESSION"
    anchor = f"{package_id}#page-1"
    claim = _claim(
        anchor,
        "可以一次性买几年的吗",
        kind="objection",
        claim_id="CL1",
    )
    claim["attributes"] = {
        "expression": "可以一次性买几年的吗",
        "responseContext": "产品按年续交，次年可根据实际情况决定是否续保。",
    }
    candidate = _candidate("D4.2", "CUSTOMER_OBJECTION", ["CL1"])
    candidate["content"] = {
        "objectionTheme": "缴费周期咨询",
        "expressions": [
            "可以一次性买几年的吗\n\n产品按年续交，次年可根据实际情况决定是否续保。"
        ],
        "context": "客户询问缴费周期",
        "rootConcernHypotheses": [],
        "resolutionElements": [
            {
                "element": "说明按年续交",
                "detail": "产品按年续交，次年可根据实际情况决定是否续保。",
            }
        ],
        "contractDetail": "用于验证客户原话与销售回复必须分离。" * 20,
    }
    gateway = TwoStageGateway(
        [claim],
        {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
    )

    result = SalesKnowledgeIdentificationService(gateway).identify(
        _package(
            package_id,
            "可以一次性买几年的吗\n产品按年续交，次年可根据实际情况决定是否续保。",
        )
    )

    assert result.candidates[0].content["expressions"] == [
        "可以一次性买几年的吗"
    ]
    assert any(
        item.field == "content.expressions" for item in result.normalizations
    )


def test_objection_resolution_repeating_expression_uses_verified_response() -> None:
    package_id = "DP-OBJECTION-RESPONSE"
    anchor = f"{package_id}#page-1"
    claim = _claim(
        anchor,
        "可以一次性买几年的吗",
        kind="objection",
        claim_id="CL1",
    )
    claim["attributes"] = {
        "expression": "可以一次性买几年的吗",
        "responseContext": "产品按年续交，次年可根据实际情况决定是否续保。",
    }
    candidate = _candidate("D4.2", "CUSTOMER_OBJECTION", ["CL1"])
    candidate["content"] = {
        "objectionTheme": "缴费周期咨询",
        "expressions": ["可以一次性买几年的吗"],
        "context": "客户询问缴费周期",
        "rootConcernHypotheses": [],
        "resolutionElements": [
            {
                "element": "说明按年续交",
                "detail": "可以一次性买几年的吗",
            }
        ],
        "contractDetail": "用于验证客户原话不能冒充销售回复。" * 20,
    }
    gateway = TwoStageGateway(
        [claim],
        {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
    )

    result = SalesKnowledgeIdentificationService(gateway).identify(
        _package(
            package_id,
            "可以一次性买几年的吗\n产品按年续交，次年可根据实际情况决定是否续保。",
        )
    )

    assert result.candidates[0].content["resolutionElements"][0]["detail"] == (
        "产品按年续交，次年可根据实际情况决定是否续保。"
    )
    assert any(
        item.field == "content.resolutionElements"
        for item in result.normalizations
    )


def test_qa_candidate_recovers_missing_structured_question_answer() -> None:
    package_id = "DP-QA-RECOVERY"
    anchor = f"{package_id}#page-1"
    first = _claim(anchor, "问题一", kind="qa", claim_id="CL1")
    first["attributes"] = {"question": "问题一", "answer": "答案一"}
    second = _claim(anchor, "问题二", kind="qa", claim_id="CL2")
    second["attributes"] = {"question": "问题二", "answer": "答案二"}
    candidate = _candidate("D4.3", "QA_PAIR", ["CL1"])
    candidate["content"] = {
        "items": [{"question": "问题一", "answer": "答案一", "claimRef": "CL1"}],
        "factReferences": ["CL1"],
        "applicability": "测试产品",
        "contractDetail": "用于验证结构化问答遗漏时能够从核验字段补回。" * 20,
    }
    candidate["claimUsage"] = [
        {
            "claimId": "CL1",
            "role": "primary",
            "contentPaths": ["$.items[0].question", "$.items[0].answer"],
            "explanation": "第一组问答已写入正文",
        }
    ]
    gateway = TwoStageGateway(
        [first, second],
        {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
    )

    result = SalesKnowledgeIdentificationService(gateway).identify(
        _package(package_id, "问题一 答案一 问题二 答案二")
    )

    assert [item["claimRef"] for item in result.candidates[0].content["items"]] == [
        "CL1",
        "CL2",
    ]
    assert {usage.claim_id for usage in result.candidates[0].claim_usage} == {
        "CL1",
        "CL2",
    }
    assert any(item.field == "content.items" for item in result.normalizations)


def test_invalid_claim_usage_path_cannot_count_as_written_content() -> None:
    package_id = "DP-INVALID-USAGE"
    anchor = f"{package_id}#page-1"
    candidate = _candidate("D1.1", "PRODUCT_FACT", ["CL1"])
    candidate["claimUsage"] = [
        {
            "claimId": "CL1",
            "role": "primary",
            "contentPaths": ["$.factReferences"],
            "explanation": "只引用来源编号",
        }
    ]
    candidate["content"] = {
        **candidate["content"],
        "factReferences": ["CL1"],
    }
    gateway = TwoStageGateway(
        [_claim(anchor, "事实一", claim_id="CL1")],
        {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
    )
    service = SalesKnowledgeIdentificationService(gateway)

    result = service.identify(_package(package_id, "事实一"))

    assert result.candidates == []
    assert result.rejected_candidates[0].candidate_id == "C1"
    assert any(
        item.claim_id == "CL1" and "claimUsage" in item.reason
        for item in result.unresolved_items
    )


def test_valid_claim_usage_survives_an_extra_metadata_path() -> None:
    package_id = "DP-MIXED-USAGE"
    anchor = f"{package_id}#page-1"
    content = {
        "facts": [{"description": "事实一"}],
        "factReferences": ["CL1"],
    }
    usage, unresolved = _validate_content_claim_usage(
        [
        {
            "claimId": "CL1",
            "role": "primary",
            "contentPaths": [
                "$.facts[0].description",
                "$.factReferences[0]",
            ],
            "explanation": "事实已写入正文，并附带来源编号",
        }
        ],
        content,
        {
            "CL1": AtomicClaim.model_validate(
                _claim(anchor, "事实一", claim_id="CL1")
            )
        },
        "C1",
    )

    assert unresolved == []
    assert usage[0].content_paths == [
        "$.facts[0].description"
    ]


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
    objection_claim = _claim(
        "DP-TEST#page-1", "药品需在保障目录内", kind="objection"
    )
    objection_claim["attributes"] = {"responseContext": "目录异议应对依据"}
    gateway = TwoStageGateway([objection_claim], object_payload)

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
    assert result.coverage_by_module["D4.1"] == "hit"
    assert result.call_count == 3
    assert [call.purpose for call in result.model_calls] == [
        "claim_discovery",
        "object_planning",
        "content_realization",
    ]
    model_stages = [stage for stage in result.processing_stages if stage.actor == "model"]
    assert [stage.model_call_ids for stage in model_stages] == [
        ["call-001"],
        ["call-002"],
        ["call-003"],
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


def test_explicitly_unresolved_claim_is_not_duplicated_by_fallback() -> None:
    gateway = TwoStageGateway(
        [_claim("DP-UNRESOLVED#page-1", "待补充范本", kind="rule")],
        {
            "candidates": [],
            "weakSignals": [],
            "unresolvedItems": [
                {
                    "claimId": "CL1",
                    "description": "资料要求使用范本但没有提供范本文字",
                    "reason": "无法形成标准话术",
                    "evidence": ["DP-UNRESOLVED#page-1"],
                    "module": None,
                }
            ],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-UNRESOLVED", "# 示例\n\n待补充范本。")
    )

    assert len(result.unresolved_items) == 1
    assert result.unresolved_items[0].claim_id == "CL1"


def test_planning_rejects_objection_built_from_expression_only() -> None:
    gateway = TwoStageGateway(
        [_claim("DP-EXPRESSION#page-1", "我没时间", kind="objection")],
        {
            "candidates": [
                _candidate("D4.2", "CUSTOMER_OBJECTION", ["CL1"])
            ],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-EXPRESSION", "# 示例\n\n我没时间。")
    )

    assert result.candidates == []
    assert result.rejected_object_plans[0].reasons == [
        "customer objection lacks source-backed root concern or response"
    ]
    assert result.unresolved_items[0].evidence == ["DP-EXPRESSION#page-1"]


def test_objection_plan_inherits_same_anchor_script_and_strategy_claims() -> None:
    objection = _claim(
        "DP-OBJECTION#row-1",
        "价格太贵",
        kind="objection",
        claim_id="CL1",
    )
    script = _claim(
        "DP-OBJECTION#row-1",
        "先认可价格顾虑再说明价值",
        kind="script",
        claim_id="CL2",
    )
    gateway = TwoStageGateway(
        [objection, script],
        {
            "candidates": [
                _candidate("D4.2", "CUSTOMER_OBJECTION", ["CL1"])
            ],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package(
            "DP-OBJECTION",
            "# 示例\n\n价格太贵。先认可价格顾虑再说明价值。",
            [SourceAnchor(anchor_id="DP-OBJECTION#row-1", kind="table")],
        )
    )

    assert result.rejected_object_plans == []
    assert result.candidates[0].source_claim_ids == ["CL1", "CL2"]


def test_planning_rejects_list_only_action_rule() -> None:
    gateway = TwoStageGateway(
        [_claim("DP-LIST#page-1", "A、B、C三级", kind="list")],
        {
            "candidates": [_candidate("D3.3", "DECISION_RULE", ["CL1"])],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-LIST", "# 示例\n\nA、B、C三级。")
    )

    assert result.candidates == []
    assert result.rejected_object_plans[0].reasons == [
        "category enumeration cannot become an action rule without source actions"
    ]


def test_planning_rejects_service_guidance_as_sales_decision_rule() -> None:
    gateway = TwoStageGateway(
        [_claim("DP-QA#row-1", "库存不足时等待后重试", kind="method")],
        {
            "candidates": [_candidate("D3.3", "DECISION_RULE", ["CL1"])],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package(
            "DP-QA",
            "# 示例\n\n库存不足时等待后重试。",
            [SourceAnchor(anchor_id="DP-QA#row-1", kind="table")],
        )
    )

    assert result.candidates == []
    assert any(
            "not service operation guidance" in reason
        for reason in result.rejected_object_plans[0].reasons
    )


def test_planning_rejects_standard_script_when_source_only_mentions_template() -> None:
    gateway = TwoStageGateway(
        [_claim("DP-TEMPLATE#page-1", "必须按照范本说", kind="rule")],
        {
            "candidates": [_candidate("D4.1", "STANDARD_SCRIPT", ["CL1"])],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-TEMPLATE", "# 示例\n\n必须按照范本说。")
    )

    assert result.candidates == []
    assert result.rejected_object_plans[0].reasons == [
        "standard script source text is not provided"
    ]


def test_planning_rejects_single_compliance_action_as_business_process() -> None:
    gateway = TwoStageGateway(
        [_claim("DP-REPORT#page-1", "及时上报投诉", kind="process")],
        {
            "candidates": [_candidate("D1.3", "BUSINESS_PROCESS", ["CL1"])],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-REPORT", "# 示例\n\n及时上报投诉。")
    )

    assert result.candidates == []
    assert result.rejected_object_plans[0].reasons == [
        "business process requires a source-backed multi-step sequence"
    ]


def test_planning_rejects_sales_conversation_branch_as_business_process() -> None:
    claim = _claim(
        "DP-BRANCH#page-1",
        "客户允许授权则继续销售，不允许则礼貌挂机",
        kind="process",
    )
    claim["attributes"] = {
        "trigger": "客户询问授权",
        "branch_yes": "继续销售",
        "branch_no": "礼貌挂机",
    }
    gateway = TwoStageGateway(
        [claim],
        {
            "candidates": [_candidate("D1.3", "BUSINESS_PROCESS", ["CL1"])],
            "weakSignals": [],
            "unresolvedItems": [],
        },
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package(
            "DP-BRANCH",
            "# 示例\n\n客户允许授权则继续销售，不允许则礼貌挂机。",
        )
    )

    assert result.candidates == []
    assert any(
            "sales-conversation conditional branches belong to D3.2" in reason
        for reason in result.rejected_object_plans[0].reasons
    )


def test_unclaimed_source_anchor_is_exposed_for_review() -> None:
    package = _package(
        "DP-ANCHOR",
        "<!-- source-anchor: DP-ANCHOR#section-1 -->\n有知识。\n"
        "<!-- source-anchor: DP-ANCHOR#section-2 -->\n只有客户原话。",
        [
            SourceAnchor(anchor_id="DP-ANCHOR#section-1", kind="section"),
            SourceAnchor(anchor_id="DP-ANCHOR#section-2", kind="section"),
        ],
    )
    claim = AtomicClaim.model_validate(
        _claim("DP-ANCHOR#section-1", "有知识")
    )

    unresolved = _unclaimed_source_anchor_inputs(package, [claim], [])

    assert len(unresolved) == 1
    assert unresolved[0][0]["evidence"] == ["DP-ANCHOR#section-2"]
    assert "未形成原子主张" in unresolved[0][0]["description"]


def test_enumeration_terms_used_by_rule_also_form_profile_dimension() -> None:
    source_text = "### 知识点 3：系统名单 ABC 分类定义\nA类、B类、C类。"
    claims = [
        AtomicClaim(
            claim_id=f"CL{index}",
            claim_kind="term",
            statement=f"{label}名单定义",
            subject=f"{label}名单定义",
            attributes={"condition": condition},
            module_hints=["D2.1"],
            evidence=[
                ClaimEvidence(
                    anchor_id="DP-LIST#section-3",
                    exact_quote=f"{label}类：{condition}",
                    source_text=source_text,
                )
            ],
        )
        for index, (label, condition) in enumerate(
            (("A", "已报价"), ("B", "未报价"), ("C", "未接触")), start=1
        )
    ]
    plans = [
        CandidateObjectPlan(
            plan_id="P1",
            title="系统名单分类规则",
            domain="D3",
            module="D3.2",
            object_type="DECISION_RULE",
            identity_hints={
                "strategyGoal": "名单分类",
                "triggerContext": "通话结束",
                "applicability": "外呼",
            },
            source_claim_ids=[claim.claim_id for claim in claims],
        )
    ]

    guarded = _ensure_enumeration_dimension_plans(plans, claims)

    dimension = next(plan for plan in guarded if plan.module == "D2.1")
    assert dimension.object_type == "PROFILE_DIMENSION"
    assert dimension.source_claim_ids == ["CL1", "CL2", "CL3"]
    assert dimension.identity_hints["dimensionName"] == "系统名单 ABC 分类"


def test_coded_term_set_forms_dimension_even_when_planner_only_creates_glossary() -> None:
    claims = [
        AtomicClaim(
            claim_id=f"CL{index}",
            claim_kind="term",
            statement=f"{code}类定义为{meaning}",
            subject=f"{code}类名单定义",
            attributes={"code": code, "meaning": meaning},
            evidence=[
                ClaimEvidence(
                    anchor_id="DP-LIST#section-3",
                    exact_quote=f"{code}类：{meaning}",
                    source_text="### 系统名单 ABC 分类定义\nA类、B类、C类。",
                )
            ],
        )
        for index, (code, meaning) in enumerate(
            (("A", "已报价"), ("B", "未报价"), ("C", "未接触")), start=1
        )
    ]
    glossary = CandidateObjectPlan(
        plan_id="P1",
        title="ABC术语",
        domain="D4",
        module="D4.3",
        object_type="TERM",
        identity_hints={"subject": "ABC分类", "applicability": "名单管理"},
        source_claim_ids=[claim.claim_id for claim in claims],
    )

    guarded = _ensure_enumeration_dimension_plans([glossary], claims)

    assert [plan.module for plan in guarded] == ["D4.3", "D2.1"]


def test_identification_drops_relation_to_rejected_candidate_but_keeps_object() -> None:
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

    assert [candidate.candidate_id for candidate in result.candidates] == ["C1"]
    assert result.candidates[0].relations == []
    assert any(
        item.candidate_id == "C1"
        and item.field == "relations"
        and "悬空关系" in item.original_value
        for item in result.normalizations
    )


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


def test_identification_repairs_valid_json_with_wrong_top_level_shape() -> None:
    claim_payload = {"claims": [_claim("DP-SHAPE#page-1", "药享保提供在线问诊")]}
    object_payload = {
        "candidates": [_candidate("D1.1", "PRODUCT_FACT", ["CL1"])],
        "weakSignals": [],
        "unresolvedItems": [],
    }
    realization = {
        "planId": "C1",
        "content": object_payload["candidates"][0]["content"],
        "entityMentions": [],
        "relations": [],
    }
    gateway = SequencedModelGateway(
        [
            json.dumps(claim_payload, ensure_ascii=False),
            json.dumps(object_payload, ensure_ascii=False),
            json.dumps([realization], ensure_ascii=False),
            json.dumps({"realizations": [realization]}, ensure_ascii=False),
        ]
    )

    result = SalesKnowledgeIdentificationService(gateway=gateway).identify(
        _package("DP-SHAPE", "# 示例\n\n药享保提供在线问诊。")
    )

    assert result.status == "completed"
    assert [trace.purpose for trace in result.model_calls] == [
        "claim_discovery",
        "object_planning",
        "content_realization",
        "repair",
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
    assert result.model_calls[0].retry_of is None
    assert result.model_calls[1].retry_of == "call-001"


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


def test_identification_batches_mixed_object_contracts_in_one_realization_call() -> None:
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
    assert [candidate.candidate_id for candidate in result.candidates] == ["P1", "P2"]
    assert {claim.claim_id for claim in result.atomic_claims} == {"S1-CL1", "S2-CL1"}
    realization_prompts = [
        request.system_prompt
        for request in gateway.requests
        if "内容编制器" in request.system_prompt
    ]
    assert len(realization_prompts) == 1
    realization_requests = [
        request
        for request in gateway.requests
        if "内容编制器" in request.system_prompt
    ]
    assert all('"sourceText"' not in request.user_prompt for request in realization_requests)
    assert "### D1.1 内容合同" in realization_prompts[0]
    assert "### D4.2 内容合同" in realization_prompts[0]
    assert [call.call_id for call in result.model_calls] == [
        "call-001",
        "call-002",
        "call-003",
        "call-004",
    ]
    assert all(call.stage_id for call in result.model_calls)


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


def test_segmenter_packs_adjacent_headings_until_capacity() -> None:
    package = _package(
        "DP-PACK",
        "\n\n".join(
            f"## 第 {index} 页\n\n<!-- source-anchor: DP-PACK#page-{index} -->\n\n内容{index}。"
            for index in range(1, 4)
        ),
        [
            SourceAnchor(anchor_id=f"DP-PACK#page-{index}", kind="page", page=index)
            for index in range(1, 4)
        ],
    )

    segments = segment_document(package, max_chars=125)

    assert len(segments) == 2
    assert [
        anchor.anchor_id for segment in segments for anchor in segment.anchors
    ] == ["DP-PACK#page-1", "DP-PACK#page-2", "DP-PACK#page-3"]


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


def test_exact_quote_macro_does_not_expand_to_the_full_source_section() -> None:
    claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="rule",
        statement="禁止表达",
        subject="返现",
        evidence=[
            ClaimEvidence(
                anchor_id="DP-QUOTE#section-1",
                exact_quote="不允许说返现",
                source_text="完整规则段落：不允许说返现，还应说明合规替代动作。",
            )
        ],
    )

    resolved, reasons = resolve_verbatim_claim_references(
        {"expression": {"$exactQuoteFromClaim": "CL1"}}, {"CL1": claim}
    )

    assert reasons == []
    assert resolved == {"expression": "不允许说返现"}


def test_objection_verbatim_macro_keeps_customer_expression_separate() -> None:
    claim = AtomicClaim(
        claim_id="CL-OBJECTION",
        claim_kind="objection",
        statement="客户咨询缴费周期",
        subject="可以一次性买几年的吗",
        attributes={
            "expression": "可以一次性买几年的吗",
            "responseContext": "产品按年续交，客户可以按实际情况决定是否续保。",
        },
        evidence=[
            ClaimEvidence(
                anchor_id="DP-OBJECTION#row-1",
                exact_quote="可以一次性买几年的吗",
                source_text="可以一次性买几年的吗",
            ),
            ClaimEvidence(
                anchor_id="DP-OBJECTION#row-1",
                exact_quote="产品按年续交",
                source_text="产品按年续交，客户可以按实际情况决定是否续保。",
            ),
        ],
    )

    resolved, reasons = resolve_verbatim_claim_references(
        {"expressions": [{"$verbatimFromClaim": "CL-OBJECTION"}]},
        {"CL-OBJECTION": claim},
    )

    assert reasons == []
    assert resolved == {"expressions": ["可以一次性买几年的吗"]}


def test_claim_attribute_macro_selects_objection_response_only() -> None:
    claim = AtomicClaim(
        claim_id="CL-OBJECTION",
        claim_kind="objection",
        statement="客户咨询缴费周期",
        subject="可以一次性买几年的吗",
        attributes={
            "expression": "可以一次性买几年的吗",
            "responseContext": "产品按年续交，客户可以按实际情况决定是否续保。",
        },
        evidence=[
            ClaimEvidence(
                anchor_id="DP-OBJECTION#row-1",
                exact_quote="可以一次性买几年的吗",
                source_text="可以一次性买几年的吗",
            )
        ],
    )

    resolved, reasons = resolve_verbatim_claim_references(
        {
            "resolutionElement": {
                "$attributeFromClaim": {
                    "claimId": "CL-OBJECTION",
                    "attribute": "responseContext",
                }
            }
        },
        {"CL-OBJECTION": claim},
    )

    assert reasons == []
    assert resolved == {
        "resolutionElement": "产品按年续交，客户可以按实际情况决定是否续保。"
    }


def test_alternate_verbatim_reference_is_normalized_to_verified_source() -> None:
    claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="script",
        statement="完整话术",
        subject="价格异议",
        evidence=[
            ClaimEvidence(
                anchor_id="DP-SCRIPT#row-1",
                exact_quote="先认可价格顾虑",
                source_text="先认可价格顾虑，再完整说明产品价值与适用边界。",
            )
        ],
    )

    resolved, reasons = resolve_verbatim_claim_references(
        {"script": {"verbatim": True, "sourceRef": "CL1"}}, {"CL1": claim}
    )

    assert reasons == []
    assert resolved == {"script": "先认可价格顾虑，再完整说明产品价值与适用边界。"}


def test_verbatim_content_wrapper_is_normalized_to_plain_text() -> None:
    resolved, reasons = resolve_verbatim_claim_references(
        {"script": {"verbatimContent": "来自资料的完整标准话术。"}}, {}
    )

    assert reasons == []
    assert resolved == {"script": "来自资料的完整标准话术。"}


def test_nested_verbatim_wrappers_are_normalized_to_plain_text() -> None:
    claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="script",
        statement="完整话术",
        subject="产品引入",
        evidence=[
            ClaimEvidence(
                anchor_id="DP-SCRIPT#row-1",
                exact_quote="完整话术",
                source_text="来自资料的完整标准话术。",
            )
        ],
    )

    first, first_reasons = resolve_verbatim_claim_references(
        {"script": {"verbatim": True, "text": {"$verbatimFromClaim": "CL1"}}},
        {"CL1": claim},
    )
    second, second_reasons = resolve_verbatim_claim_references(
        {"script": {"verbatimContent": {"$verbatimFromClaim": "CL1"}}},
        {"CL1": claim},
    )

    assert first_reasons == second_reasons == []
    assert first == second == {"script": "来自资料的完整标准话术。"}


def test_json_encoded_verbatim_macro_is_normalized_to_verified_source() -> None:
    claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="script",
        statement="完整话术",
        subject="产品引入",
        evidence=[
            ClaimEvidence(
                anchor_id="DP-SCRIPT#row-1",
                exact_quote="完整话术",
                source_text="来自资料的完整标准话术。",
            )
        ],
    )

    resolved, reasons = resolve_verbatim_claim_references(
        {"script": '{"$verbatimFromClaim":"CL1"}'}, {"CL1": claim}
    )

    assert reasons == []
    assert resolved == {"script": "来自资料的完整标准话术。"}


def test_nested_script_object_is_rejected_without_crashing_source_gate() -> None:
    claim = _claim(
        "DP-SCRIPT-GATE#page-1",
        "来自资料的完整标准话术。",
        kind="script",
    )
    candidate = _candidate("D4.1", "STANDARD_SCRIPT", ["CL1"])
    candidate["content"] = {
        "script": {"fullText": "未解析的模型结构"},
    }

    result = SalesKnowledgeIdentificationService(
        gateway=TwoStageGateway(
            [claim],
            {
                "candidates": [candidate],
                "weakSignals": [],
                "unresolvedItems": [],
            },
        )
    ).identify(
        _package("DP-SCRIPT-GATE", "# 示例\n\n来自资料的完整标准话术。")
    )

    assert result.candidates == []
    assert any(
        "standard script must equal verified source text" in reason
        for reason in result.rejected_candidates[0].reasons
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
        ("qa", "一次能买几年"),
        ("qa", "如何问诊"),
    ]
    assert supplemented[-1].evidence[1].source_text == "进入服务页发起问诊。"


def test_numbered_qa_completeness_adds_only_missing_pairs() -> None:
    package = _package(
        "DP-NUMBERED-QA",
        (
            "## 常见问答\n\n<!-- source-anchor: DP-NUMBERED-QA#page-1 -->\n\n"
            "Q1：如何开通服务\nA1：进入服务页提交申请。\n"
            "Q2：多久可以使用\nA2：审核通过后即可使用。"
        ),
        [SourceAnchor(anchor_id="DP-NUMBERED-QA#page-1", kind="page")],
    )
    existing = AtomicClaim.model_validate(
        {
            **_claim("DP-NUMBERED-QA#page-1", "如何开通服务", kind="qa"),
            "attributes": {
                "question": "如何开通服务",
                "answer": "进入服务页提交申请。",
            },
        }
    )

    supplemented = supplement_numbered_qa_claims(package, [existing])

    assert [claim.attributes["question"] for claim in supplemented] == [
        "如何开通服务",
        "多久可以使用",
    ]
    assert supplemented[-1].attributes["answer"] == "审核通过后即可使用。"


def test_numbered_qa_completeness_enriches_matching_model_claim() -> None:
    package = _package(
        "DP-QA-ENRICH",
        (
            "## 常见问答\n\n<!-- source-anchor: DP-QA-ENRICH#page-1 -->\n\n"
            "Q1：如何开通服务\nA1：进入服务页提交申请。"
        ),
        [SourceAnchor(anchor_id="DP-QA-ENRICH#page-1", kind="page")],
    )
    existing = AtomicClaim.model_validate(
        _claim(
            "DP-QA-ENRICH#page-1",
            "Q1：如何开通服务",
            kind="qa",
            claim_id="CL7",
        )
    )

    supplemented = supplement_numbered_qa_claims(package, [existing])

    assert [claim.claim_id for claim in supplemented] == ["CL7"]
    assert supplemented[0].attributes == {
        "question": "如何开通服务",
        "answer": "进入服务页提交申请。",
    }


def test_numbered_qa_completeness_replaces_combined_model_claim() -> None:
    package = _package(
        "DP-QA-SPLIT",
        (
            "## 常见问答\n\n<!-- source-anchor: DP-QA-SPLIT#page-1 -->\n\n"
            "Q1：如何开通服务\nA1：进入服务页提交申请。\n"
            "Q2：多久可以使用\nA2：审核通过后即可使用。"
        ),
        [SourceAnchor(anchor_id="DP-QA-SPLIT#page-1", kind="page")],
    )
    combined = AtomicClaim.model_validate(
        _claim(
            "DP-QA-SPLIT#page-1",
            "Q1：如何开通服务；Q2：多久可以使用",
            kind="qa",
            claim_id="CL9",
        )
    )

    supplemented = supplement_numbered_qa_claims(package, [combined])

    assert len(supplemented) == 2
    assert all(claim.claim_id.startswith("STRUCTURED-QA-") for claim in supplemented)


def test_process_claim_derives_explicit_arrow_steps() -> None:
    package = _package("DP-PROCESS", "登录 -> 选择服务 -> 提交申请")
    raw = _claim(
        "DP-PROCESS#page-1",
        "登录 -> 选择服务 -> 提交申请",
        kind="process",
    )
    raw["statement"] = "登录 -> 选择服务 -> 提交申请"

    accepted, rejected = validate_atomic_claims([raw], package)

    assert rejected == []
    assert accepted[0].attributes["steps"] == ["登录", "选择服务", "提交申请"]


def test_process_claim_derives_steps_from_verified_multiline_evidence() -> None:
    package = _package("DP-PROCESS-LINES", "填写信息\n确认内容\n提交申请")
    raw = _claim(
        "DP-PROCESS-LINES#page-1",
        "填写信息\n确认内容\n提交申请",
        kind="process",
    )
    raw["statement"] = "填写信息 确认内容 提交申请"

    accepted, rejected = validate_atomic_claims([raw], package)

    assert rejected == []
    assert accepted[0].attributes["steps"] == ["填写信息", "确认内容", "提交申请"]


def test_structured_table_completeness_adds_explicit_strategy_column() -> None:
    package = _package(
        "DP-STRATEGY",
        (
            "### 第 1 行\n\n<!-- source-anchor: DP-STRATEGY#row-1 -->\n\n"
            "- **B列**：客群引入\n- **C列**：面向异地医保父母推荐\n"
            "- **D列**：完整销售话术。"
        ),
        [SourceAnchor(anchor_id="DP-STRATEGY#row-1", kind="table")],
    )

    supplemented = supplement_structured_table_claims(package, [])

    assert [(claim.claim_kind, claim.subject) for claim in supplemented] == [
        ("strategy", "客群引入")
    ]
    assert supplemented[0].attributes["strategyDescription"] == (
        "面向异地医保父母推荐"
    )


def test_structured_objection_removes_model_invented_root_cause() -> None:
    package = _package(
        "DP-ROOT-CAUSE",
        (
            "### 第 1 行\n\n<!-- source-anchor: DP-ROOT-CAUSE#row-1 -->\n\n"
            "- **A列**：异议处理\n- **B列**：可以一次性买几年的吗\n"
            "- **D列**：产品按年续交。"
        ),
        [SourceAnchor(anchor_id="DP-ROOT-CAUSE#row-1", kind="table")],
    )
    model_claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="objection",
        statement="客户询问缴费周期",
        subject="客户希望长期锁定权益",
        attributes={"rootCause": "担心涨价或续费麻烦"},
        evidence=[
            ClaimEvidence(
                anchor_id="DP-ROOT-CAUSE#row-1",
                exact_quote="可以一次性买几年的吗",
                source_text="可以一次性买几年的吗",
            )
        ],
    )

    supplemented = supplement_structured_table_claims(package, [model_claim])

    assert supplemented[0].subject == "客户希望长期锁定权益"
    assert supplemented[0].attributes == {
        "expression": "可以一次性买几年的吗",
        "responseContext": "产品按年续交。",
    }


def test_content_path_container_and_content_prefix_expand_to_leaf_paths() -> None:
    content = {"applicability": {"products": ["车险", "药享保"]}}

    assert _expand_content_path_to_leaf_paths(
        content, "$.applicability.products"
    ) == ["$.applicability.products[0]", "$.applicability.products[1]"]


def test_all_versions_scope_requires_explicit_source_scope() -> None:
    unscoped = AtomicClaim.model_validate(
        _claim("DP-SCOPE#row-1", "药享保保费180元", kind="fact")
    )
    scoped = unscoped.model_copy(
        update={
            "statement": "药享保两个版本通用按年续交",
            "attributes": {"applicability": "全版本"},
        }
    )

    assert _claim_explicitly_all_versions(unscoped) is False
    assert _claim_explicitly_all_versions(scoped) is True


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
    assert '"claimKind":"fact"' in planning_request.user_prompt
    assert '"claimKind":"rule"' in planning_request.user_prompt


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


def test_invalid_content_is_recompiled_once_without_replanning() -> None:
    candidate = _candidate("D1.1", "PRODUCT_FACT", ["CL1"], candidate_id="P1")

    class RepairingContentGateway(TwoStageGateway):
        realization_calls = 0

        def complete(self, request: ModelRequest) -> ModelCompletion:
            if "内容编制器" not in request.system_prompt:
                return super().complete(request)
            self.realization_calls += 1
            if self.realization_calls > 1:
                return super().complete(request)
            self.requests.append(request)
            return ModelCompletion(
                provider="test-provider",
                model="test-model",
                content=json.dumps(
                    {
                        "realizations": [
                            {
                                "planId": "P1",
                                "content": {"subject": "测试产品"},
                                "entityMentions": [],
                                "relations": [],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            )

    result = SalesKnowledgeIdentificationService(
        gateway=RepairingContentGateway(
            [_claim("DP-RECOMPILE#page-1", "可信事实")],
            {"candidates": [candidate], "weakSignals": [], "unresolvedItems": []},
        )
    ).identify(_package("DP-RECOMPILE", "# 示例\n\n可信事实。"))

    assert result.call_count == 4
    assert [item.candidate_id for item in result.candidates] == ["P1"]
    assert result.rejected_candidates == []
    assert not any(
        item.reason.startswith("候选被拒绝：") for item in result.unresolved_items
    )


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
        module="D4.1",
        object_type="CUSTOMER_OBJECTION",
        object_boundary="不同根本顾虑必须拆分",
        classification_basis="属于客户异议",
        identity_hints={"objectionIntent": "常见异议", "context": "通用"},
        source_claim_ids=["CL1", "CL2"],
    )

    split = _enforce_plan_granularity([plan], claims)

    assert [item.plan_id for item in split] == ["P1-1", "P1-2"]
    assert [item.identity_hints["objectionIntent"] for item in split] == [
        "价格贵",
        "已经有医保",
    ]
    assert [item.source_claim_ids for item in split] == [["CL1"], ["CL2"]]


def test_granularity_gate_merges_same_product_version_fact_fragments() -> None:
    claims = [
        AtomicClaim.model_validate(
            _claim("DP-VERSION#row-1", "尊享版年保费100元", claim_id="CL1")
        ),
        AtomicClaim.model_validate(
            _claim("DP-VERSION#row-2", "尊享版首次普药赔付100%", claim_id="CL2")
        ),
    ]
    plans = [
        CandidateObjectPlan(
            plan_id="P1",
            title="尊享版价格事实",
            domain="D1",
            module="D1.1",
            object_type="PRODUCT_VERSION_FACT",
            object_boundary="同一产品版本共同更新",
            classification_basis="属于版本事实",
            identity_hints={
                "subject": "药享保-尊享版",
                "versionScope": "尊享版",
                "factTheme": "价格",
            },
            source_claim_ids=["CL1"],
        ),
        CandidateObjectPlan(
            plan_id="P2",
            title="尊享版赔付事实补充",
            domain="D1",
            module="D1.1",
            object_type="PRODUCT_VERSION_FACT",
            object_boundary="同一产品版本共同更新",
            classification_basis="属于版本事实",
            identity_hints={
                "subject": "药享保",
                "versionScope": "尊享版语境",
                "factTheme": "赔付比例",
            },
            source_claim_ids=["CL2"],
        ),
    ]

    merged = _enforce_plan_granularity(plans, claims)

    assert len(merged) == 1
    assert merged[0].title == "药享保尊享版产品版本事实"
    assert merged[0].identity_hints == {
        "subject": "药享保",
        "versionScope": "尊享版",
        "factTheme": "产品版本综合事实",
    }
    assert merged[0].source_claim_ids == ["CL1", "CL2"]


def test_granularity_gate_merges_policy_rules_for_same_business_subject() -> None:
    claims = [
        AtomicClaim.model_validate(
            _claim("DP-POLICY#row-1", "处方不超过5种药品", kind="rule", claim_id="CL1")
        ),
        AtomicClaim.model_validate(
            _claim("DP-POLICY#row-2", "同功效药品不能同时开具", kind="rule", claim_id="CL2")
        ),
    ]
    plans = [
        CandidateObjectPlan(
            plan_id=f"P{index}",
            title=title,
            domain="D1",
                module="D1.2",
            object_type="POLICY_RULE_SET",
            object_boundary="同一处方规则主题共同维护",
            classification_basis="属于处方约束",
            identity_hints={
                "purpose": purpose,
                "subject": "线上问诊处方",
                "scope": scope,
            },
            source_claim_ids=[f"CL{index}"],
        )
        for index, (title, purpose, scope) in enumerate(
            [
                ("处方数量规则", "限制药品数量", "西药与中成药"),
                ("同功效药规则", "保障用药安全", "同功效药品"),
            ],
            start=1,
        )
    ]

    merged = _enforce_plan_granularity(plans, claims)

    assert len(merged) == 1
    assert merged[0].title == "线上问诊处方规则集"
    assert merged[0].identity_hints == {
        "purpose": "规范线上问诊处方",
        "subject": "线上问诊处方",
        "scope": "来源资料明确的规则范围",
    }
    assert merged[0].source_claim_ids == ["CL1", "CL2"]


def test_granularity_gate_merges_qa_plans_from_one_document_unit() -> None:
    claims = [
        AtomicClaim(
            claim_id=f"CL{index}",
            claim_kind="qa",
            statement=question,
            subject=question,
            evidence=[
                ClaimEvidence(
                    anchor_id=f"DP-FAQ#row-{index}",
                    exact_quote=question,
                    source_text=question,
                )
            ],
        )
        for index, question in enumerate(("如何问诊", "如何开发票"), start=1)
    ]
    plans = [
        CandidateObjectPlan(
            plan_id=f"P{index}",
            title=question,
            domain="D4",
                module="D4.2",
            object_type="QA_PAIR",
            object_boundary="共同维护的问答集合",
            classification_basis="标准问答",
            identity_hints={"subject": question, "applicability": "药享保"},
            source_claim_ids=[f"CL{index}"],
        )
        for index, question in enumerate(("如何问诊", "如何开发票"), start=1)
    ]

    merged = _enforce_plan_granularity(plans, claims)

    assert len(merged) == 1
    assert merged[0].source_claim_ids == ["CL1", "CL2"]


def test_granularity_gate_keeps_related_decision_rules_in_one_object() -> None:
    claims = [
        AtomicClaim(
            claim_id="CL1",
            claim_kind="rule",
            statement="首次接触后按报价结果分类",
            subject="首次分类",
            evidence=[
                ClaimEvidence(
                    anchor_id="DP-RULE#section-1",
                    exact_quote="首次分类",
                    source_text="首次分类",
                )
            ],
        ),
        AtomicClaim(
            claim_id="CL2",
            claim_kind="rule",
            statement="复播时只升不降",
            subject="复播迁移",
            evidence=[
                ClaimEvidence(
                    anchor_id="DP-RULE#section-2",
                    exact_quote="复播迁移",
                    source_text="复播迁移",
                )
            ],
        ),
    ]
    plan = CandidateObjectPlan(
        plan_id="P1",
        title="名单状态规则",
        domain="D3",
        module="D3.2",
        object_type="DECISION_RULE",
        object_boundary="触发上下文不同必须拆分",
        classification_basis="属于行动规则",
        identity_hints={
            "strategyGoal": "维护名单状态",
            "triggerContext": "名单更新",
            "applicability": "外呼名单管理",
        },
        source_claim_ids=["CL1", "CL2"],
    )

    split = _enforce_plan_granularity([plan], claims)

    assert len(split) == 1
    assert split[0].source_claim_ids == ["CL1", "CL2"]
    assert split[0].identity_hints["triggerContext"] == "名单更新"


def test_composite_product_versions_split_into_independent_identities() -> None:
    plan = CandidateObjectPlan(
        plan_id="P1",
        title="产品权益",
        domain="D1",
        module="D1.1",
        object_type="PRODUCT_FACT",
        identity_hints={
            "subject": "示例产品",
            "versionScope": "基础版 vs 专业版",
            "factTheme": "权益",
        },
        source_claim_ids=["CL1"],
    )

    split = _split_composite_product_version_plan(plan)

    assert [item.identity_hints["versionScope"] for item in split] == [
        "基础版",
        "专业版",
    ]
    assert [item.plan_id for item in split] == ["P1-V1", "P1-V2"]


def test_explicit_numbered_strategy_combinations_become_independent_plans() -> None:
    source = "方案一\n产品A+产品B\n• 补充能力\n方案二\n产品A+产品C\n• 降低成本"
    claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="strategy",
        statement="方案一 产品A+产品B；方案二 产品A+产品C",
        subject="组合销售",
        attributes={"strategyGoal": "组合销售"},
        evidence=[
            ClaimEvidence(
                anchor_id="DP-STRATEGY#page-1",
                exact_quote=source,
                source_text=source,
            )
        ],
    )
    plan = CandidateObjectPlan(
        plan_id="P1",
        title="组合销售策略",
        domain="D3",
        module="D3.2",
        object_type="SALES_STRATEGY",
        identity_hints={
            "strategyGoal": "组合销售",
            "triggerContext": "销售场景",
            "applicability": "产品组合",
        },
        source_claim_ids=["CL1"],
    )

    claims, plans = _split_explicit_strategy_combinations([claim], [plan])
    split = _enforce_plan_granularity(plans, claims)

    assert [item.attributes["combination"] for item in claims] == [
        "产品A+产品B",
        "产品A+产品C",
    ]
    assert [item.source_claim_ids for item in split] == [["CL1-S1"], ["CL1-S2"]]


def test_coverage_repair_augments_existing_plan_and_merges_new_identity() -> None:
    existing = CandidateObjectPlan(
        plan_id="P1",
        title="药享保问答",
        domain="D4",
        module="D4.3",
        object_type="QA_PAIR",
        object_boundary="同一维护单元",
        classification_basis="标准问答",
        identity_hints={"subject": "药享保FAQ", "applicability": "全版本"},
        source_claim_ids=["CL1"],
    )
    repair_claim = AtomicClaim(
        claim_id="CL2",
        claim_kind="qa",
        statement="是否可以开具发票",
        subject="发票问答",
        evidence=[
            ClaimEvidence(
                anchor_id="DP-FAQ#row-2",
                exact_quote="是否可以开具发票",
                source_text="是否可以开具发票",
            )
        ],
    )
    augmented, augmentation_rejections = _apply_plan_augmentations(
        [existing], [{"planId": "P1", "sourceClaimIds": ["CL2"]}], [repair_claim]
    )
    new_plan = CandidateObjectPlan(
        plan_id="R1",
        title="短时突出版本优势",
        domain="D3",
        module="D3.2",
        object_type="SALES_TECHNIQUE",
        object_boundary="机制不同必须拆分",
        classification_basis="可复用销售方法",
        identity_hints={
            "techniqueName": "短时优势表达",
            "purpose": "快速说明产品价值",
            "mechanism": "聚焦核心特点",
        },
        source_claim_ids=["CL3"],
    )
    merged, duplicate_rejections = _merge_repair_plans(augmented, [new_plan])

    assert augmentation_rejections == []
    assert augmented[0].source_claim_ids == ["CL1", "CL2"]
    assert duplicate_rejections == []
    assert [plan.plan_id for plan in merged] == ["P1", "R1"]


def test_coverage_repair_cannot_attach_conversation_branch_to_business_process() -> None:
    existing = CandidateObjectPlan(
        plan_id="P1",
        title="禁呼屏蔽流程",
        domain="D1",
        module="D1.2",
        object_type="BUSINESS_PROCESS",
        identity_hints={
            "purpose": "执行禁呼屏蔽",
            "subject": "电销坐席",
            "scope": "明确拒绝客户",
        },
        source_claim_ids=["CL1"],
    )
    branch_payload = _claim(
        "DP-BRANCH#rule-1",
        "客户拒绝授权则礼貌挂机",
        kind="process",
        claim_id="CL2",
    )
    branch_payload["attributes"] = {
        "condition": "客户拒绝授权",
        "action": "礼貌挂机",
    }
    branch = AtomicClaim.model_validate(branch_payload)

    augmented, rejected = _apply_plan_augmentations(
        [existing], [{"planId": "P1", "sourceClaimIds": ["CL2"]}], [branch]
    )

    assert augmented[0].source_claim_ids == ["CL1"]
    assert "cannot augment a D1.2" in rejected[0].reasons[0]


def test_coverage_repair_only_retries_automatic_planning_omissions() -> None:
    automatic = {
        "claimId": "CL1",
        "reason": "模型未将该主张分配给任何对象计划，禁止静默丢失",
    }
    explicit = {"claimId": "CL2", "reason": "资料不足，保留为未决项"}

    assert _automatic_uncovered_claim_ids(
        [(automatic, {"A1"}), (explicit, {"A2"})]
    ) == {"CL1"}


def test_script_plan_does_not_consume_independent_strategy_role() -> None:
    strategy_claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="strategy",
        statement="短时间讲清产品特点",
        subject="产品引入方法",
        evidence=[
            ClaimEvidence(
                anchor_id="DP-SCRIPT#row-1",
                exact_quote="短时间讲清产品特点",
                source_text="短时间讲清产品特点",
            )
        ],
    )
    script_plan = CandidateObjectPlan(
        plan_id="P1",
        title="产品引入话术",
        domain="D4",
        module="D4.1",
        object_type="STANDARD_SCRIPT",
        object_boundary="话术边界",
        classification_basis="完整表达",
        identity_hints={
            "communicationGoal": "产品引入",
            "method": "优势说明",
            "applicability": "目标客户",
        },
        source_claim_ids=["CL1"],
    )

    assert not _plan_satisfies_primary_claim_role(script_plan, strategy_claim)


def test_qa_content_normalizes_single_fact_reference_alias() -> None:
    content = _normalize_content_shape(
        "D4.2",
        "QA_PAIR",
        {
            "items": [
                {
                    "question": "是否可以开具发票？",
                    "answer": "全额自费订单可以开具。",
                    "factReferences": ["CL1"],
                }
            ]
        },
    )

    assert content["items"][0]["claimRef"] == "CL1"


def test_content_shape_removes_runtime_refs_and_normalizes_applicability() -> None:
    content = _normalize_content_shape(
        "D1.1",
        "PRODUCT_VERSION_FACT",
        {
            "subject": "药享保尊享版",
            "facts": [{"description": "年保费100元", "sourceRef": "CL1"}],
            "applicability": "尊享版用户",
            "limitations": [],
        },
        {"subject": "药享保", "versionScope": "尊享版"},
    )

    assert content["applicability"] == {"product": "药享保", "version": "尊享版"}
    assert content["facts"] == [{"description": "年保费100元"}]


def test_compact_object_plan_expands_to_validation_shape() -> None:
    expanded = _expand_compact_object_plan(
        [
            "P1",
            "产品引入话术",
            "D4.1",
            "STANDARD_SCRIPT",
            {
                "communicationGoal": "产品引入",
                "method": "优势说明",
                "applicability": "目标客户",
            },
            ["CL1"],
        ]
    )

    assert expanded["planId"] == "P1"
    assert expanded["sourceClaimIds"] == ["CL1"]


def test_d33_pruning_rebuilds_unattributed_summary_from_verified_claims() -> None:
    source_text = "### 复播名单状态迁移规则\nA类不能降级。"
    claim = AtomicClaim(
        claim_id="CL1",
        claim_kind="rule",
        statement="复播时A类名单不得降级",
        subject="A类迁移规则",
        attributes={"constraint": "A类不得降级"},
        module_hints=["D3.3"],
        evidence=[
            ClaimEvidence(
                anchor_id="DP-D33#section-1",
                exact_quote="A类不能降级",
                source_text=source_text,
            )
        ],
    )
    content = {
        "strategyName": "复播规则",
        "triggerConditions": ["客户再次触达"],
        "decisionLogic": "保护高意向客户，避免资源错配",
        "actions": [
            {
                "condition": "A类",
                "actionType": "TRANSITION_RULE",
                "targetValue": "维持A类",
            }
        ],
        "applicability": {"products": [], "scenarios": ["复播"]},
    }
    usage = [
        ContentClaimUsage(
            claim_id="CL1",
            role="primary",
            content_paths=["$.actions[0].condition", "$.actions[0].targetValue"],
            explanation="来源规则",
        )
    ]

    normalized, normalized_usage = _prune_unattributed_d33_inferences(
        content,
        {path for item in usage for path in item.content_paths},
        [claim],
        usage,
    )

    assert normalized["decisionLogic"] == "复播时A类名单不得降级"
    assert normalized["triggerConditions"] == ["复播名单状态迁移规则"]
    assert "actionType" not in normalized["actions"][0]
    assert {path for item in normalized_usage for path in item.content_paths} >= {
        "$.decisionLogic",
        "$.triggerConditions[0]",
    }


def test_d33_pruning_rebuilds_fully_unattributed_actions_from_rule_claims() -> None:
    claim = AtomicClaim.model_validate(
        {
            **_claim(
                "DP-D33#section-1",
                "成功报价后标识为A类",
                kind="rule",
                claim_id="CL1",
            ),
            "statement": "成功报价后应将客户标识为A类",
            "attributes": {"condition": "成功报价", "result": "A类"},
        }
    )
    content = {
        "strategyName": "名单判定",
        "triggerConditions": ["成功报价"],
        "decisionLogic": "成功报价判定A类",
        "actions": ["记录渠道", "发送通知"],
        "applicability": {"products": [], "scenarios": []},
    }

    normalized, usage = _prune_unattributed_d33_inferences(
        content,
        {"$.triggerConditions[0]", "$.decisionLogic"},
        [claim],
        [],
    )

    assert normalized["actions"] == ["成功报价后应将客户标识为A类"]
    assert any(item.content_paths == ["$.actions[0]"] for item in usage)
