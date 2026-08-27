from __future__ import annotations

import tomllib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

RULE_PACKAGE_PATH = Path(__file__).with_name("rules") / "d1-d5-v0.6.toml"
EXPECTED_MODULE_COUNT = 22
EXPECTED_DOMAINS = {"D1", "D2", "D3", "D4", "D5"}


@dataclass(frozen=True)
class KnowledgeModule:
    domain: str
    code: str
    name: str
    scope: Literal["core", "optional"]
    meaning: str
    object_types: tuple[str, ...]
    core_objects: tuple[str, ...]
    boundary: str
    sources: tuple[str, ...]
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeDomain:
    code: str
    name: str
    question: str
    meaning: str
    boundary: str


def _load_rule_package() -> tuple[
    str,
    str,
    str,
    str,
    dict[str, str],
    tuple[KnowledgeDomain, ...],
    tuple[KnowledgeModule, ...],
]:
    rule_content = RULE_PACKAGE_PATH.read_bytes()
    payload = tomllib.loads(rule_content.decode("utf-8"))
    scope_definitions = payload["scope_definitions"]
    domains = tuple(
        KnowledgeDomain(
            code=item["code"],
            name=item["name"],
            question=item["question"],
            meaning=item["meaning"],
            boundary=item["boundary"],
        )
        for item in payload["domains"]
    )
    modules = tuple(
        KnowledgeModule(
            domain=item["domain"],
            code=item["code"],
            name=item["name"],
            scope=item["scope"],
            meaning=item["meaning"],
            object_types=tuple(item["object_types"]),
            core_objects=tuple(item["core_objects"]),
            boundary=item["boundary"],
            sources=tuple(item["sources"]),
            consumers=tuple(item["consumers"]),
        )
        for item in payload["modules"]
    )
    _validate_rule_package(scope_definitions, domains, modules)
    fingerprint = sha256(rule_content).hexdigest()
    return (
        payload["version"],
        payload["status"],
        payload["source"],
        fingerprint,
        scope_definitions,
        domains,
        modules,
    )


def _validate_rule_package(
    scope_definitions: dict[str, str],
    domains: tuple[KnowledgeDomain, ...],
    modules: tuple[KnowledgeModule, ...],
) -> None:
    if set(scope_definitions) != {"core", "optional"} or not all(scope_definitions.values()):
        raise RuntimeError("knowledge rule package must define core and optional scopes")
    if {domain.code for domain in domains} != EXPECTED_DOMAINS:
        raise RuntimeError("knowledge rule package must define D1-D5 domain rules")
    if any(
        not all((domain.name, domain.question, domain.meaning, domain.boundary))
        for domain in domains
    ):
        raise RuntimeError("knowledge rule package contains incomplete domain rules")
    if len(modules) != EXPECTED_MODULE_COUNT:
        raise RuntimeError(
            f"knowledge rule package must define {EXPECTED_MODULE_COUNT} modules"
        )
    codes = [module.code for module in modules]
    if len(codes) != len(set(codes)):
        raise RuntimeError("knowledge rule package contains duplicate module codes")
    if {module.domain for module in modules} != EXPECTED_DOMAINS:
        raise RuntimeError("knowledge rule package must cover D1-D5")
    for module in modules:
        if not module.code.startswith(f"{module.domain}."):
            raise RuntimeError(f"module {module.code} does not belong to {module.domain}")
        required_values = (
            module.name,
            module.meaning,
            module.boundary,
            module.object_types,
            module.core_objects,
            module.sources,
            module.consumers,
        )
        if not all(required_values):
            raise RuntimeError(f"module {module.code} has incomplete identification rules")


(
    CATALOG_VERSION,
    CATALOG_STATUS,
    CATALOG_SOURCE,
    CATALOG_FINGERPRINT,
    MODULE_SCOPE_DEFINITIONS,
    KNOWLEDGE_DOMAINS,
    KNOWLEDGE_MODULES,
) = _load_rule_package()
MODULE_BY_CODE = {module.code: module for module in KNOWLEDGE_MODULES}
DOMAIN_BY_CODE = {domain.code: domain for domain in KNOWLEDGE_DOMAINS}


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
    sections = []
    for module in KNOWLEDGE_MODULES:
        scope = "可选范围模块" if module.scope == "optional" else "核心范围模块"
        sections.append(
            "\n".join(
                [
                    f"### {module.code} {module.name}（{scope}）",
                    f"- 业务含义：{module.meaning}",
                    f"- 允许对象类型：{', '.join(module.object_types)}",
                    f"- 核心对象：{'、'.join(module.core_objects)}",
                    f"- 对象边界与分类裁决：{module.boundary}",
                    f"- 典型来源：{'、'.join(module.sources)}",
                    f"- 明确消费方：{'、'.join(module.consumers)}",
                ]
            )
        )
    return "\n\n".join(sections)
