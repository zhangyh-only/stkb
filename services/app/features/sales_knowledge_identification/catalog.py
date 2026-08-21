from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeModule:
    domain: str
    code: str
    name: str
    object_types: tuple[str, ...]


KNOWLEDGE_MODULES = (
    KnowledgeModule(
        "D1",
        "D1.1",
        "产品事实库",
        ("PRODUCT_FACT", "PRODUCT_VERSION_FACT", "PRODUCT_COMPONENT_FACT"),
    ),
    KnowledgeModule(
        "D1", "D1.2", "清单事实库", ("LIST_FACT", "LIST_ITEM_FACT", "ELIGIBILITY_FACT")
    ),
    KnowledgeModule(
        "D1", "D1.3", "政策流程库", ("POLICY_RULE_SET", "BUSINESS_PROCESS", "PROCESS_STEP")
    ),
    KnowledgeModule("D1", "D1.4", "竞品事实库", ("COMPETITOR_FACT", "COMPARISON_SNAPSHOT")),
    KnowledgeModule("D2", "D2.1", "画像坐标轴库", ("PROFILE_DIMENSION", "PROFILE_DIMENSION_VALUE")),
    KnowledgeModule(
        "D2", "D2.2", "画像库", ("CUSTOMER_PROFILE", "PROFILE_BACKGROUND_FACT", "ROLEPLAY_RULE")
    ),
    KnowledgeModule(
        "D2",
        "D2.3",
        "需求与决策动因库",
        ("CUSTOMER_NEED", "PAIN_POINT", "PURCHASE_MOTIVATION", "URGENCY_RULE"),
    ),
    KnowledgeModule(
        "D2",
        "D2.4",
        "打动点与触发机制库",
        ("PERSUASION_TRIGGER", "TRUST_TRIGGER", "REJECTION_TRIGGER", "RESPONSE_PATTERN"),
    ),
    KnowledgeModule(
        "D2", "D2.5", "决策链角色库", ("DECISION_ROLE", "INFLUENCE_RELATION", "ROLE_CONCERN")
    ),
    KnowledgeModule(
        "D3",
        "D3.1",
        "场景与流程阶段库",
        ("SALES_SCENARIO", "SALES_STAGE", "STAGE_TRANSITION_RULE", "STAGE_COMPLETION_MARKER"),
    ),
    KnowledgeModule(
        "D3", "D3.2", "技巧模式库", ("SALES_TECHNIQUE", "METHOD_STEP", "APPLICABILITY_CONDITION")
    ),
    KnowledgeModule(
        "D3",
        "D3.3",
        "策略与判断规则库",
        ("SALES_STRATEGY", "DECISION_RULE", "NEXT_BEST_ACTION", "CLARIFYING_QUESTION"),
    ),
    KnowledgeModule(
        "D3",
        "D3.4",
        "合规红线库",
        ("COMPLIANCE_RULE", "PROHIBITED_EXPRESSION", "RISK_PATTERN", "COMPLIANT_REWRITE_GUIDE"),
    ),
    KnowledgeModule("D4", "D4.1", "标准话术库", ("STANDARD_SCRIPT", "SCRIPT_VARIANT")),
    KnowledgeModule(
        "D4",
        "D4.2",
        "异议库",
        ("CUSTOMER_OBJECTION", "ROOT_CONCERN_HYPOTHESIS", "RESOLUTION_ELEMENT"),
    ),
    KnowledgeModule("D4", "D4.3", "Q&A与术语库", ("QA_PAIR", "TERM", "STANDARD_EXPLANATION")),
    KnowledgeModule(
        "D4",
        "D4.4",
        "案例库",
        ("SALES_CASE", "KEY_EVENT", "CASE_OUTCOME", "SUCCESS_FAILURE_CAUSE", "LESSON_LEARNED"),
    ),
    KnowledgeModule(
        "D4", "D4.5", "物料与微课库", ("SALES_MATERIAL", "SCRIPT_CARD", "MICRO_LESSON_CARD")
    ),
    KnowledgeModule(
        "D5", "D5.1", "卖点与检测单元库", ("SELLING_POINT", "CHECK_UNIT", "TRIGGER_CONDITION")
    ),
    KnowledgeModule(
        "D5", "D5.2", "能力模型库", ("COMPETENCY_DIMENSION", "BEHAVIOR_ANCHOR", "PROFICIENCY_LEVEL")
    ),
    KnowledgeModule(
        "D5",
        "D5.3",
        "评分规则库",
        ("EVALUATION_METRIC", "SCORING_RULE", "EVIDENCE_REQUIREMENT", "TOLERANCE_RULE"),
    ),
    KnowledgeModule(
        "D5",
        "D5.4",
        "基准与黄金测集库",
        ("EVALUATION_BENCHMARK", "GOLDEN_SAMPLE", "REGRESSION_PROBE", "ACCEPTANCE_CHECKLIST"),
    ),
)

MODULE_BY_CODE = {module.code: module for module in KNOWLEDGE_MODULES}
DOMAIN_NAMES = {
    "D1": "业务事实",
    "D2": "客户与动因",
    "D3": "销售策略",
    "D4": "话术与案例",
    "D5": "评估与训练",
}
CATALOG_VERSION = "d1-d5-v0.1"


def validate_candidate_classification(domain: str, module: str, object_type: str) -> list[str]:
    definition = MODULE_BY_CODE.get(module)
    if definition is None:
        return [f"unknown knowledge module: {module}"]
    errors: list[str] = []
    if definition.domain != domain:
        errors.append(f"module {module} belongs to {definition.domain}, not {domain}")
    if object_type not in definition.object_types:
        errors.append(f"object type {object_type} is not allowed for module {module}")
    return errors


def render_catalog_for_prompt() -> str:
    return "\n".join(
        f"- {module.code} / {module.name} / allowed object types: {', '.join(module.object_types)}"
        for module in KNOWLEDGE_MODULES
    )
