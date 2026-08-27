from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACTS_PATH = (
    Path(__file__).with_name("rules") / "object-content-contracts-v1.0.toml"
)

STRING_FIELDS_BY_OBJECT_TYPE: dict[str, tuple[str, ...]] = {
    "STANDARD_SCRIPT": ("communicationGoal", "script"),
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
