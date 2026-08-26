from app.features.sales_knowledge_identification.catalog import KNOWLEDGE_MODULES
from app.features.sales_knowledge_identification.content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
    CONTENT_CONTRACT_VERSION,
    validate_candidate_content,
)


def test_content_contracts_cover_every_module_and_object_type() -> None:
    assert CONTENT_CONTRACT_VERSION == "object-content-contracts-v0.2"
    assert set(CONTENT_CONTRACT_BY_MODULE) == {
        module.code for module in KNOWLEDGE_MODULES
    }
    for module in KNOWLEDGE_MODULES:
        contract = CONTENT_CONTRACT_BY_MODULE[module.code]
        assert set(contract.object_types) == set(module.object_types)
        assert contract.required_fields
        assert contract.minimum_content_chars >= 150
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
