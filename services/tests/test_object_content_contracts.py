from app.features.sales_knowledge_identification.catalog import KNOWLEDGE_MODULES
from app.features.sales_knowledge_identification.content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
    CONTENT_CONTRACT_VERSION,
    validate_candidate_content,
)


def test_content_contracts_cover_every_module_and_object_type() -> None:
    assert CONTENT_CONTRACT_VERSION == "object-content-contracts-v1.3"
    assert set(CONTENT_CONTRACT_BY_MODULE) == {
        module.code for module in KNOWLEDGE_MODULES
    }
    for module in KNOWLEDGE_MODULES:
        contract = CONTENT_CONTRACT_BY_MODULE[module.code]
        assert set(contract.object_types) == set(module.object_types)
        assert contract.required_fields
        assert contract.minimum_content_chars >= 120
        assert contract.inclusion and contract.exclusion
        assert contract.positive_example and contract.negative_example


def test_summary_only_content_fails_quality_gate() -> None:
    errors = validate_candidate_content(
        "D4.1",
        "STANDARD_SCRIPT",
        {"summary": "包含多种标准话术。"},
    )

    assert "summary-only content is not a valid knowledge object" in errors
    assert any(error.startswith("missing required content fields") for error in errors)
    assert any(error.startswith("content is too thin") for error in errors)


def test_complete_type_specific_content_passes_quality_gate() -> None:
    content = {
        "communicationGoal": "向忙碌上班族说明在线问诊购药价值并确认购买意愿",
        "applicability": {
            "customer": "工作繁忙、在意线下就医时间成本的客户",
            "product": "药享保尊享版",
            "stage": "产品引入",
        },
        "script": (
            "您平时工作忙，小病小痛去医院挂号排队很花时间。"
            "药享保提供线上问诊和购药直赔，可减少往返医院的时间成本。"
        ),
        "factReferences": [
            "7×24小时线上图文问诊",
            "保险责任范围内线上购药直赔",
        ],
        "complianceConstraints": [
            "不得承诺所有疾病或药品均可赔付",
            "保费、赔付比例和等待期必须引用当前版本事实",
        ],
    }

    assert validate_candidate_content("D4.1", "STANDARD_SCRIPT", content) == []


def test_explicitly_allowed_empty_fields_do_not_force_model_invention() -> None:
    content = {
        "communicationGoal": "完整保留资料中的产品引入表达",
        "applicability": {"customer": "优质车险客户", "stage": "产品引入"},
        "script": "这是一段来自原始资料且已经过逐字证据校验的完整标准话术。" * 10,
        "factReferences": [],
        "complianceConstraints": [],
    }

    assert validate_candidate_content("D4.1", "STANDARD_SCRIPT", content) == []


def test_d13_allows_empty_preconditions_when_source_does_not_define_them() -> None:
    content = {
        "purpose": "规范互联网医院医生的在线处方开具规则",
        "preconditions": [],
        "rulesOrSteps": [
            {"stepId": "R1", "description": "每张处方不超过5种药品且不超过7日用量。"}
        ],
        "exceptions": [],
        "contractDetail": "规则来自原始资料，资料未定义额外进入条件。" * 20,
    }

    assert validate_candidate_content("D1.3", "POLICY_RULE_SET", content) == []


def test_validates_stable_field_shapes_for_formal_content() -> None:
    errors = validate_candidate_content(
        "D1.1",
        "PRODUCT_VERSION_FACT",
        {
            "subject": "药享保尊享版",
            "facts": [{"description": "年保费100元"}],
            "applicability": "尊享版用户",
            "limitations": [],
            "detail": "用于确保测试内容达到最小长度。" * 20,
        },
    )

    assert "invalid content field types: applicability must be dict" in errors


def test_standard_script_rejects_nested_wrapper_after_normalization_stage() -> None:
    content = {
        "communicationGoal": "处理价格异议并准确说明产品价值",
        "applicability": "客户明确提出价格顾虑时使用",
        "script": {"verbatimContent": "不应残留在正式对象中的模型中间结构"},
        "factReferences": ["DP-SCRIPT#row-1"],
        "complianceConstraints": [],
        "detail": "补充文字不能掩盖 script 字段类型不符合正式合同。" * 20,
    }

    errors = validate_candidate_content("D4.1", "STANDARD_SCRIPT", content)

    assert "content fields must be non-empty strings: script" in errors


def test_term_cannot_use_qa_pair_content_shape() -> None:
    errors = validate_candidate_content(
        "D4.3",
        "TERM",
        {
            "items": [{"question": "什么是嫌货", "answer": "提出异议的客户"}],
            "factReferences": ["CL1"],
            "applicability": "销售基础培训",
            "detail": "用于确保内容长度达到基础阈值。" * 20,
        },
    )

    assert any(error.startswith("missing required content fields: terms") for error in errors)


def test_term_requires_explanation_for_each_term_item() -> None:
    errors = validate_candidate_content(
        "D4.3",
        "TERM",
        {
            "terms": [{"termText": "嫌货才是买货人"}],
            "applicability": "用于销售基础培训中的异议识别，不能直接推断购买意愿。",
            "detail": "这个扩展说明只用于满足对象总体内容量，不应替代术语条目中的标准解释。" * 10,
        },
    )

    assert any(
        error.startswith("content item 1 missing fields: standardExplanation")
        for error in errors
    )


def test_term_accepts_complete_nested_items() -> None:
    content = {
        "terms": [
            {
                "termText": "嫌货才是买货人",
                "standardExplanation": (
                    "这是销售培训中的经验表达，指异议可能代表客户正在评估价值，"
                    "但不能据此断定客户一定有购买意愿。"
                ),
                "sourceStance": "销售课程将其作为理解异议的经验性表述。",
                "usageBoundary": (
                    "只作为理解异议的启发式术语，不生成客户心理规则，也不生成自动行动决策。"
                ),
            }
        ],
        "applicability": (
            "用于销售基础培训中的异议识别。实际应用时仍需通过追问确认异议原因，"
            "不得替代客户需求与购买意愿证据。"
        ),
    }

    assert validate_candidate_content("D4.3", "TERM", content) == []


def test_objection_allows_empty_root_hypotheses_without_invention() -> None:
    content = {
        "objectionTheme": "缴费周期咨询",
        "expressions": ["可以一次性买几年的吗"],
        "context": "客户在药享保产品咨询中询问是否支持多年期购买。",
        "rootConcernHypotheses": [],
        "resolutionElements": [
            {
                "element": "说明按年续交规则",
                "detail": (
                    "资料明确说明药享保与车险类似按年续交，客户次年可根据产品实际情况决定是否续保。"
                ),
            }
        ],
        "sourceDetail": (
            "该对象只保留客户原话和资料明确给出的回复依据，"
            "不推演客户担心涨价或操作麻烦。"
        )
        * 3,
    }

    assert validate_candidate_content("D4.2", "CUSTOMER_OBJECTION", content) == []


def test_objection_rejects_unstable_resolution_element_shape() -> None:
    content = {
        "objectionTheme": "缴费周期咨询",
        "expressions": ["可以一次性买几年的吗"],
        "context": "客户在产品咨询中询问缴费周期。",
        "rootConcernHypotheses": [],
        "resolutionElements": [
            {
                "elementText": "产品按年续交",
                "elementRole": "standard_response",
            }
        ],
        "sourceDetail": (
            "用于确认嵌套字段名称漂移时，即使总体文本足够长"
            "也不能通过正式内容合同。"
        )
        * 5,
    }

    errors = validate_candidate_content("D4.2", "CUSTOMER_OBJECTION", content)

    assert (
        "resolutionElements item 1 missing string fields: element, detail"
        in errors
    )


def test_objection_resolution_detail_cannot_repeat_customer_expression() -> None:
    content = {
        "objectionTheme": "缴费周期咨询",
        "expressions": ["可以一次性买几年的吗"],
        "context": "客户在产品咨询中询问缴费周期。",
        "rootConcernHypotheses": [],
        "resolutionElements": [
            {
                "element": "说明按年续交",
                "detail": "可以一次性买几年的吗",
            }
        ],
        "sourceDetail": "用于确认客户原话不能被当作化解要素正文重复写入。" * 8,
    }

    errors = validate_candidate_content("D4.2", "CUSTOMER_OBJECTION", content)

    assert any("repeats the customer expression" in error for error in errors)
