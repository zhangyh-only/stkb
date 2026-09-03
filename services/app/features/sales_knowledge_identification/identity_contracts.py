from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import KNOWLEDGE_MODULES, KnowledgeModule
from .content_contracts import SOURCE_CONTENT_CONTRACTS

CONTRACTS_PATH = (
    Path(__file__).with_name("rules") / "object-identity-contracts-v0.4.toml"
)


@dataclass(frozen=True)
class ObjectIdentityContract:
    module: str
    identity_fields: tuple[str, ...]
    identity_fields_by_type: dict[str, tuple[str, ...]]
    same_object_when: str
    different_object_when: str
    merge_strategy: str
    conflict_rule: str


def _load_contracts() -> tuple[str, str, tuple[ObjectIdentityContract, ...]]:
    with CONTRACTS_PATH.open("rb") as contract_file:
        payload = tomllib.load(contract_file)
    contracts = tuple(
        ObjectIdentityContract(
            module=item["module"],
            identity_fields=tuple(item["identity_fields"]),
            identity_fields_by_type={},
            same_object_when=item["same_object_when"],
            different_object_when=item["different_object_when"],
            merge_strategy=item["merge_strategy"],
            conflict_rule=item["conflict_rule"],
        )
        for item in payload["contracts"]
    )
    if len(contracts) != 22:
        raise RuntimeError("source identity contracts must cover 22 legacy modules")
    if any(not item.identity_fields for item in contracts):
        raise RuntimeError("every identity contract must define identity fields")
    return payload["version"], payload["status"], contracts


IDENTITY_CONTRACT_VERSION, IDENTITY_CONTRACT_STATUS, SOURCE_IDENTITY_CONTRACTS = (
    _load_contracts()
)
_SOURCE_IDENTITY_BY_MODULE = {
    item.module: item for item in SOURCE_IDENTITY_CONTRACTS
}
IDENTITY_CONTRACT_BY_OBJECT_TYPE = {
    object_type: _SOURCE_IDENTITY_BY_MODULE[content_contract.module]
    for content_contract in SOURCE_CONTENT_CONTRACTS
    for object_type in content_contract.object_types
}


def _module_contract(module: KnowledgeModule) -> ObjectIdentityContract:
    object_types = module.object_types
    identity_fields_by_type = {
        object_type: IDENTITY_CONTRACT_BY_OBJECT_TYPE[object_type].identity_fields
        for object_type in object_types
    }
    return ObjectIdentityContract(
        module=module.code,
        identity_fields=tuple(
            f"{object_type}: {' + '.join(fields)}"
            for object_type, fields in identity_fields_by_type.items()
        ),
        identity_fields_by_type=identity_fields_by_type,
        same_object_when="由 objectType 对应身份字段共同决定；模块码不参与对象身份。",
        different_object_when="任一对象合同的关键身份、适用范围或生效周期不同。",
        merge_strategy="使用 objectType 合同归并，同名事实冲突时隔离而不覆盖。",
        conflict_rule="优先正式版本和生效依据；证据不足时进入待评审。",
    )


OBJECT_IDENTITY_CONTRACTS = tuple(
    _module_contract(module) for module in KNOWLEDGE_MODULES
)
IDENTITY_CONTRACT_BY_MODULE = {item.module: item for item in OBJECT_IDENTITY_CONTRACTS}


def validate_identity_hints(
    module: str, identity_hints: dict[str, Any], object_type: str | None = None
) -> list[str]:
    contract = (
        IDENTITY_CONTRACT_BY_OBJECT_TYPE.get(object_type)
        if object_type
        else IDENTITY_CONTRACT_BY_MODULE.get(module)
    )
    if contract is None:
        return [f"identity contract is missing for module {module}"]
    identity_fields = contract.identity_fields
    if not identity_fields:
        return [f"identity contract is missing for object type {object_type or 'unknown'}"]
    missing = [
        field
        for field in identity_fields
        if field not in identity_hints or identity_hints[field] in (None, "", [], {})
    ]
    if missing:
        return ["missing required identity fields: " + ", ".join(missing)]
    return []


def canonical_identity(
    module: str, identity_hints: dict[str, Any], object_type: str | None = None
) -> dict[str, Any]:
    contract = (
        IDENTITY_CONTRACT_BY_OBJECT_TYPE[object_type]
        if object_type
        else IDENTITY_CONTRACT_BY_MODULE[module]
    )
    identity_fields = contract.identity_fields
    return {field: identity_hints[field] for field in identity_fields}


def render_identity_contracts_for_prompt() -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"### {item.module} 身份与归并合同",
                "- objectType 对应 identityHints："
                + "；".join(
                    f"{object_type}=[{', '.join(fields)}]"
                    for object_type, fields in item.identity_fields_by_type.items()
                ),
                f"- 同一对象：{item.same_object_when}",
                f"- 必须拆分：{item.different_object_when}",
                f"- 归并方式：{item.merge_strategy}",
                f"- 冲突裁决：{item.conflict_rule}",
            ]
        )
        for item in OBJECT_IDENTITY_CONTRACTS
    )
