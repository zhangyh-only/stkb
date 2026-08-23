from app.features.sales_knowledge_identification.catalog import (
    CATALOG_FINGERPRINT,
    CATALOG_SOURCE,
    CATALOG_STATUS,
    CATALOG_VERSION,
    KNOWLEDGE_MODULES,
    MODULE_BY_CODE,
    render_catalog_for_prompt,
)


def test_rule_package_defines_complete_versioned_d1_d5_catalog() -> None:
    assert CATALOG_VERSION == "d1-d5-v0.2"
    assert CATALOG_STATUS == "sample_validation"
    assert len(CATALOG_FINGERPRINT) == 64
    assert CATALOG_SOURCE.endswith("STKB-D1-D5知识对象与业务图模型映射矩阵.md")
    assert len(KNOWLEDGE_MODULES) == 22
    assert {module.domain for module in KNOWLEDGE_MODULES} == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    }
    assert {
        module.code for module in KNOWLEDGE_MODULES if module.lifecycle == "optional"
    } == {"D1.4", "D2.5"}
    assert all(module.meaning and module.boundary for module in KNOWLEDGE_MODULES)
    assert all(module.object_types and module.sources for module in KNOWLEDGE_MODULES)


def test_prompt_catalog_contains_rules_that_distinguish_stage_and_strategy() -> None:
    prompt_catalog = render_catalog_for_prompt()

    assert "### D3.1 场景与流程阶段库" in prompt_catalog
    assert "具体会话当前处于哪个阶段属于运行时状态" in prompt_catalog
    assert "仅描述特定条件下应采取动作的内容优先归入D3.3" in prompt_catalog
    assert "### D3.3 策略与判断规则库" in prompt_catalog
    assert "场景的稳定阶段骨架归D3.1" in prompt_catalog
    assert "产品事实或卖点本身不归策略" in prompt_catalog
    assert MODULE_BY_CODE["D3.3"].object_types == (
        "SALES_STRATEGY",
        "DECISION_RULE",
        "NEXT_BEST_ACTION",
        "CLARIFYING_QUESTION",
    )
