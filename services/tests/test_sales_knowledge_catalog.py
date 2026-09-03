from app.features.sales_knowledge_identification.catalog import (
    CANONICAL_OBJECT_TYPES,
    CATALOG_FINGERPRINT,
    CATALOG_SOURCE,
    CATALOG_STATUS,
    CATALOG_VERSION,
    ITEM_OBJECT_TYPES,
    KNOWLEDGE_DOMAINS,
    KNOWLEDGE_MODULES,
    MODULE_BY_CODE,
    MODULE_SCOPE_DEFINITIONS,
    render_catalog_for_prompt,
)


def test_rule_package_defines_complete_versioned_d1_d5_catalog() -> None:
    assert CATALOG_VERSION == "d1-d5-v0.9"
    assert CATALOG_STATUS == "sample_validation"
    assert len(CATALOG_FINGERPRINT) == 64
    assert CATALOG_SOURCE.endswith("STKB-通用销售知识规则体系与识别机制重审建议.md")
    assert len(KNOWLEDGE_MODULES) == 12
    assert [domain.code for domain in KNOWLEDGE_DOMAINS] == [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]
    assert [domain.question for domain in KNOWLEDGE_DOMAINS] == [
        "卖什么，事实和使用规则是什么",
        "卖给谁，客户为什么行动",
        "在什么场景下应该怎么做",
        "客户怎么表达，销售怎么回应",
        "怎样判断做得好不好",
    ]
    assert {module.domain for module in KNOWLEDGE_MODULES} == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    }
    assert all(module.scope == "core" for module in KNOWLEDGE_MODULES)
    assert "12个候选知识内容模块" in MODULE_SCOPE_DEFINITIONS["core"]
    assert "下沉到对象合同" in MODULE_SCOPE_DEFINITIONS["optional"]
    assert all(module.meaning and module.boundary for module in KNOWLEDGE_MODULES)
    assert all(module.object_types and module.sources for module in KNOWLEDGE_MODULES)
    assert len(CANONICAL_OBJECT_TYPES) == 37
    assert len(ITEM_OBJECT_TYPES) == 34
    assert not CANONICAL_OBJECT_TYPES & ITEM_OBJECT_TYPES


def test_internal_item_type_cannot_be_planned_as_knowledge_object() -> None:
    from app.features.sales_knowledge_identification.catalog import (
        validate_candidate_classification,
    )

    assert validate_candidate_classification("D1", "D1.2", "PROCESS_STEP") == [
        "object type PROCESS_STEP is an internal item and cannot be a KnowledgeObject"
    ]


def test_prompt_catalog_contains_rules_that_distinguish_stage_and_strategy() -> None:
    prompt_catalog = render_catalog_for_prompt()

    assert "### D3.1 销售场景与旅程" in prompt_catalog
    assert "某次会话当前阶段属于运行状态" in prompt_catalog
    assert "### D3.2 销售方法与策略" in prompt_catalog
    assert "场景骨架、完整话术和产品事实不进入" in prompt_catalog
    assert MODULE_BY_CODE["D3.2"].object_types[-4:] == (
        "SALES_STRATEGY",
        "DECISION_RULE",
        "NEXT_BEST_ACTION",
        "CLARIFYING_QUESTION",
    )
