from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACTS_PATH = (
    Path(__file__).with_name("rules") / "object-content-contracts-v0.3.toml"
)


@dataclass(frozen=True)
class ObjectContentContract:
    module: str
    object_types: tuple[str, ...]
    required_fields: tuple[str, ...]
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
    missing_fields = [
        field
        for field in contract.required_fields
        if field not in content
        or (
            field not in contract.allow_empty_fields
            and content[field] in (None, "", [], {})
        )
    ]
    if missing_fields:
        errors.append("missing required content fields: " + ", ".join(missing_fields))
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True)
    if len(serialized) < contract.minimum_content_chars:
        errors.append(
            f"content is too thin for {module}: {len(serialized)} < "
            f"{contract.minimum_content_chars} chars"
        )
    if set(content) == {"summary"}:
        errors.append("summary-only content is not a valid knowledge object")
    return errors


def render_content_contracts_for_prompt() -> str:
    sections = []
    for item in OBJECT_CONTENT_CONTRACTS:
        sections.append(
            "\n".join(
                [
                    f"### {item.module} 内容合同",
                    f"- 适用对象类型：{', '.join(item.object_types)}",
                    f"- content 必填字段：{', '.join(item.required_fields)}",
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
