from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACTS_PATH = (
    Path(__file__).with_name("rules") / "object-content-contracts-v1.3.toml"
)

STRING_FIELDS_BY_OBJECT_TYPE: dict[str, tuple[str, ...]] = {
    "STANDARD_SCRIPT": ("communicationGoal", "script"),
}

FIELD_TYPES_BY_OBJECT_TYPE: dict[str, dict[str, type]] = {
    "PRODUCT_VERSION_FACT": {
        "subject": str,
        "facts": list,
        "applicability": dict,
        "limitations": list,
    },
    "BUSINESS_PROCESS": {
        "purpose": str,
        "preconditions": list,
        "rulesOrSteps": list,
        "exceptions": list,
    },
    "POLICY_RULE_SET": {
        "purpose": str,
        "preconditions": list,
        "rulesOrSteps": list,
        "exceptions": list,
    },
    "SALES_STRATEGY": {
        "strategyName": str,
        "triggerConditions": list,
        "decisionLogic": str,
        "actions": list,
        "applicability": dict,
    },
    "DECISION_RULE": {
        "strategyName": str,
        "triggerConditions": list,
        "decisionLogic": str,
        "actions": list,
        "applicability": dict,
    },
    "STANDARD_SCRIPT": {
        "communicationGoal": str,
        "applicability": dict,
        "script": str,
        "factReferences": list,
        "complianceConstraints": list,
    },
    "CUSTOMER_OBJECTION": {
        "objectionTheme": str,
        "expressions": list,
        "context": str,
        "rootConcernHypotheses": list,
        "resolutionElements": list,
    },
    "QA_PAIR": {
        "items": list,
        "factReferences": list,
        "applicability": str,
    },
}

ITEM_FIELD_TYPES_BY_OBJECT_TYPE: dict[str, dict[str, type]] = {
    "PRODUCT_VERSION_FACT": {"description": str},
    "QA_PAIR": {"question": str, "answer": str, "claimRef": str},
}

NESTED_ITEM_FIELDS_BY_OBJECT_TYPE: dict[
    str, dict[str, tuple[str, ...]]
] = {
    "CUSTOMER_OBJECTION": {
        "rootConcernHypotheses": (
            "hypothesis",
            "sourceStance",
            "usageBoundary",
        ),
        "resolutionElements": ("element", "detail"),
    }
}

CONTENT_SHAPES_BY_OBJECT_TYPE: dict[str, str] = {
    "PRODUCT_VERSION_FACT": (
        "subject:string; facts:[{description:string}]; "
        "applicability:{product:string,version:string}; limitations:array"
    ),
    "BUSINESS_PROCESS": (
        "purpose:string; preconditions:array; rulesOrSteps:array; exceptions:array"
    ),
    "POLICY_RULE_SET": (
        "purpose:string; preconditions:array; rulesOrSteps:array; exceptions:array"
    ),
    "SALES_STRATEGY": (
        "strategyName:string; triggerConditions:array; decisionLogic:string; "
        "actions:array; applicability:{products:array,scenarios:array}"
    ),
    "DECISION_RULE": (
        "与 SALES_STRATEGY 同形；仅允许销售决策，不收录服务操作答疑"
    ),
    "STANDARD_SCRIPT": (
        "communicationGoal:string; applicability:object; script:string; "
        "factReferences:array; complianceConstraints:array"
    ),
    "CUSTOMER_OBJECTION": (
        "objectionTheme:string; expressions:array; context:string; "
        "rootConcernHypotheses:[{hypothesis:string,sourceStance:string,"
        "usageBoundary:string}]（无来源可为空）；"
        "resolutionElements:[{element:string,detail:string}]"
    ),
    "QA_PAIR": (
        "items:[{question:string,answer:string,claimRef:string}]; "
        "factReferences:array; applicability:string"
    ),
}


@dataclass(frozen=True)
class ObjectContentContract:
    module: str
    object_types: tuple[str, ...]
    required_fields: tuple[str, ...]
    required_fields_by_type: dict[str, tuple[str, ...]]
    item_fields_by_type: dict[str, tuple[str, ...]]
    allow_empty_fields: tuple[str, ...]
    minimum_content_chars: int
    granularity: str
    inclusion: str
    exclusion: str
    positive_example: str
    negative_example: str


def _load_contracts() -> tuple[str, tuple[ObjectContentContract, ...]]:
    with CONTRACTS_PATH.open("rb") as contract_file:
        payload = tomllib.load(contract_file)
    contracts = tuple(
        ObjectContentContract(
            module=item["module"],
            object_types=tuple(item["object_types"]),
            required_fields=tuple(item["required_fields"]),
            required_fields_by_type={
                object_type: tuple(fields)
                for object_type, fields in item.get(
                    "required_fields_by_type", {}
                ).items()
            },
            item_fields_by_type={
                object_type: tuple(fields)
                for object_type, fields in item.get("item_fields_by_type", {}).items()
            },
            allow_empty_fields=tuple(item.get("allow_empty_fields", [])),
            minimum_content_chars=item["minimum_content_chars"],
            granularity=item["granularity"],
            inclusion=item["inclusion"],
            exclusion=item["exclusion"],
            positive_example=item["positive_example"],
            negative_example=item["negative_example"],
        )
        for item in payload["contracts"]
    )
    if len(contracts) != 22:
        raise RuntimeError("object content contracts must cover all 22 modules")
    if len({item.module for item in contracts}) != len(contracts):
        raise RuntimeError("object content contracts contain duplicate modules")
    return payload["version"], contracts


CONTENT_CONTRACT_VERSION, OBJECT_CONTENT_CONTRACTS = _load_contracts()
CONTENT_CONTRACT_BY_MODULE = {item.module: item for item in OBJECT_CONTENT_CONTRACTS}


def validate_candidate_content(module: str, object_type: str, content: dict[str, Any]) -> list[str]:
    contract = CONTENT_CONTRACT_BY_MODULE.get(module)
    if contract is None:
        return [f"content contract is missing for module {module}"]
    errors: list[str] = []
    if object_type not in contract.object_types:
        errors.append(f"object type {object_type} is not covered by content contract {module}")
    required_fields = contract.required_fields_by_type.get(
        object_type, contract.required_fields
    )
    missing_fields = [
        field
        for field in required_fields
        if field not in content
        or (
            field not in contract.allow_empty_fields
            and content[field] in (None, "", [], {})
        )
    ]
    if missing_fields:
        errors.append("missing required content fields: " + ", ".join(missing_fields))
    invalid_string_fields = [
        field
        for field in STRING_FIELDS_BY_OBJECT_TYPE.get(object_type, ())
        if field in content
        and (not isinstance(content[field], str) or not content[field].strip())
    ]
    if invalid_string_fields:
        errors.append(
            "content fields must be non-empty strings: "
            + ", ".join(invalid_string_fields)
        )
    invalid_type_fields = [
        f"{field} must be {expected_type.__name__}"
        for field, expected_type in FIELD_TYPES_BY_OBJECT_TYPE.get(
            object_type, {}
        ).items()
        if field in content and not isinstance(content[field], expected_type)
    ]
    if invalid_type_fields:
        errors.append("invalid content field types: " + ", ".join(invalid_type_fields))
    item_fields = contract.item_fields_by_type.get(object_type)
    if item_fields:
        collection_field = next(
            (field for field in required_fields if isinstance(content.get(field), list)),
            None,
        )
        items = content.get(collection_field, []) if collection_field else []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                errors.append(f"content item {index} must be an object")
                continue
            missing_item_fields = [
                field for field in item_fields if item.get(field) in (None, "", [], {})
            ]
            if missing_item_fields:
                errors.append(
                    f"content item {index} missing fields: "
                    + ", ".join(missing_item_fields)
                )
            invalid_item_types = [
                f"{field} must be {expected_type.__name__}"
                for field, expected_type in ITEM_FIELD_TYPES_BY_OBJECT_TYPE.get(
                    object_type, {}
                ).items()
                if field in item and not isinstance(item[field], expected_type)
            ]
            if invalid_item_types:
                errors.append(
                    f"content item {index} invalid field types: "
                    + ", ".join(invalid_item_types)
                )
    for collection_field, nested_fields in NESTED_ITEM_FIELDS_BY_OBJECT_TYPE.get(
        object_type, {}
    ).items():
        for index, item in enumerate(content.get(collection_field, []), start=1):
            if not isinstance(item, dict):
                errors.append(
                    f"{collection_field} item {index} must be an object"
                )
                continue
            missing_nested_fields = [
                field
                for field in nested_fields
                if not isinstance(item.get(field), str) or not item[field].strip()
            ]
            if missing_nested_fields:
                errors.append(
                    f"{collection_field} item {index} missing string fields: "
                    + ", ".join(missing_nested_fields)
                )
            unexpected_nested_fields = sorted(set(item) - set(nested_fields))
            if unexpected_nested_fields:
                errors.append(
                    f"{collection_field} item {index} has unsupported fields: "
                    + ", ".join(unexpected_nested_fields)
                )
    if object_type == "CUSTOMER_OBJECTION":
        expressions = {
            expression.strip()
            for expression in content.get("expressions", [])
            if isinstance(expression, str) and expression.strip()
        }
        for index, item in enumerate(content.get("resolutionElements", []), start=1):
            if isinstance(item, dict) and item.get("detail") in expressions:
                errors.append(
                    f"resolutionElements item {index} repeats the customer expression "
                    "instead of a source-backed resolution detail"
                )
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True)
    if len(serialized) < contract.minimum_content_chars:
        errors.append(
            f"content is too thin for {module}: {len(serialized)} < "
            f"{contract.minimum_content_chars} chars"
        )
    if set(content) == {"summary"}:
        errors.append("summary-only content is not a valid knowledge object")
    return errors


def render_content_contracts_for_prompt(
    module_codes: set[str] | None = None,
) -> str:
    sections = []
    for item in OBJECT_CONTENT_CONTRACTS:
        if module_codes is not None and item.module not in module_codes:
            continue
        sections.append(
            "\n".join(
                [
                    f"### {item.module} 内容合同",
                    f"- 适用对象类型：{', '.join(item.object_types)}",
                    f"- content 必填字段：{', '.join(item.required_fields)}",
                    *(
                        [
                            "- 分对象类型必填："
                            + "；".join(
                                f"{object_type}=[{', '.join(fields)}]"
                                for object_type, fields in item.required_fields_by_type.items()
                            )
                        ]
                        if item.required_fields_by_type
                        else []
                    ),
                    *(
                        [
                            "- 字段形态："
                            + "；".join(
                                f"{object_type}={CONTENT_SHAPES_BY_OBJECT_TYPE[object_type]}"
                                for object_type in item.object_types
                                if object_type in CONTENT_SHAPES_BY_OBJECT_TYPE
                            )
                        ]
                        if any(
                            object_type in CONTENT_SHAPES_BY_OBJECT_TYPE
                            for object_type in item.object_types
                        )
                        else []
                    ),
                    *(
                        [
                            "- 分对象类型条目必填："
                            + "；".join(
                                f"{object_type}=[{', '.join(fields)}]"
                                for object_type, fields in item.item_fields_by_type.items()
                            )
                        ]
                        if item.item_fields_by_type
                        else []
                    ),
                    (
                        "- 可显式为空的字段："
                        + (", ".join(item.allow_empty_fields) or "无")
                    ),
                    f"- 最小有效内容量：序列化后 {item.minimum_content_chars} 字符",
                    f"- 对象粒度：{item.granularity}",
                    f"- 纳入：{item.inclusion}",
                    f"- 排除：{item.exclusion}",
                    f"- 正例：{item.positive_example}",
                    f"- 反例：{item.negative_example}",
                ]
            )
        )
    return "\n\n".join(sections)
