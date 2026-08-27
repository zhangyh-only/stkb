from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import MODULE_BY_CODE

CONTRACTS_PATH = (
    Path(__file__).with_name("rules") / "object-identity-contracts-v0.3.toml"
)


@dataclass(frozen=True)
class ObjectIdentityContract:
    module: str
    identity_fields: tuple[str, ...]
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
            same_object_when=item["same_object_when"],
            different_object_when=item["different_object_when"],
            merge_strategy=item["merge_strategy"],
            conflict_rule=item["conflict_rule"],
        )
        for item in payload["contracts"]
    )
    modules = {item.module for item in contracts}
    if modules != set(MODULE_BY_CODE):
        missing = sorted(set(MODULE_BY_CODE) - modules)
        extra = sorted(modules - set(MODULE_BY_CODE))
        raise RuntimeError(
            f"identity contracts coverage mismatch: missing={missing}, extra={extra}"
        )
    if any(not item.identity_fields for item in contracts):
        raise RuntimeError("every identity contract must define identity fields")
    return payload["version"], payload["status"], contracts


IDENTITY_CONTRACT_VERSION, IDENTITY_CONTRACT_STATUS, OBJECT_IDENTITY_CONTRACTS = (
    _load_contracts()
)
IDENTITY_CONTRACT_BY_MODULE = {item.module: item for item in OBJECT_IDENTITY_CONTRACTS}


def validate_identity_hints(module: str, identity_hints: dict[str, Any]) -> list[str]:
    contract = IDENTITY_CONTRACT_BY_MODULE.get(module)
    if contract is None:
        return [f"identity contract is missing for module {module}"]
    missing = [
        field
        for field in contract.identity_fields
        if field not in identity_hints or identity_hints[field] in (None, "", [], {})
    ]
    if missing:
        return ["missing required identity fields: " + ", ".join(missing)]
    return []


def canonical_identity(module: str, identity_hints: dict[str, Any]) -> dict[str, Any]:
    contract = IDENTITY_CONTRACT_BY_MODULE[module]
    return {field: identity_hints[field] for field in contract.identity_fields}


def render_identity_contracts_for_prompt() -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"### {item.module} 身份与归并合同",
                f"- identityHints 必填：{', '.join(item.identity_fields)}",
                f"- 同一对象：{item.same_object_when}",
                f"- 必须拆分：{item.different_object_when}",
                f"- 归并方式：{item.merge_strategy}",
                f"- 冲突裁决：{item.conflict_rule}",
            ]
        )
        for item in OBJECT_IDENTITY_CONTRACTS
    )
