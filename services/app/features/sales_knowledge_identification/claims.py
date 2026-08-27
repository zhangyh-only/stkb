from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .catalog import MODULE_BY_CODE
from .models import AtomicClaim, ClaimEvidence, DocumentPackage, RejectedAtomicClaim

SOURCE_ANCHOR = re.compile(r"<!--\s*source-anchor:\s*([^\s>]+)\s*-->")


def extract_anchor_sections(document_package: DocumentPackage) -> dict[str, str]:
    """Return the smallest structural Markdown section containing each anchor."""
    markdown = document_package.full_markdown
    anchor_matches = list(SOURCE_ANCHOR.finditer(markdown))
    if not anchor_matches:
        if len(document_package.anchors) == 1:
            return {document_package.anchors[0].anchor_id: markdown}
        return {}

    result: dict[str, str] = {}
    for index, anchor_match in enumerate(anchor_matches):
        start = anchor_match.start()
        end = (
            anchor_matches[index + 1].start()
            if index + 1 < len(anchor_matches)
            else len(markdown)
        )
        result[anchor_match.group(1)] = markdown[start:end].strip()
    return result


def validate_atomic_claims(
    raw_claims: list[Any],
    document_package: DocumentPackage,
) -> tuple[list[AtomicClaim], list[RejectedAtomicClaim]]:
    anchor_sections = extract_anchor_sections(document_package)
    accepted: list[AtomicClaim] = []
    rejected: list[RejectedAtomicClaim] = []
    seen_ids: set[str] = set()

    for index, raw_claim in enumerate(raw_claims, start=1):
        claim_id = _claim_id(raw_claim, index)
        if not isinstance(raw_claim, dict):
            rejected.append(
                RejectedAtomicClaim(
                    claim_id=claim_id,
                    reasons=["claim must be a JSON object"],
                    raw_claim={"value": raw_claim},
                )
            )
            continue
        try:
            claim = AtomicClaim.model_validate(raw_claim)
        except ValidationError as error:
            rejected.append(
                RejectedAtomicClaim(
                    claim_id=claim_id,
                    reasons=[item["msg"] for item in error.errors()],
                    raw_claim=raw_claim,
                )
            )
            continue

        reasons: list[str] = []
        if claim.claim_id in seen_ids:
            reasons.append(f"duplicate claim id: {claim.claim_id}")
        seen_ids.add(claim.claim_id)
        valid_module_hints = [
            module for module in claim.module_hints if module in MODULE_BY_CODE
        ]

        resolved_evidence: list[ClaimEvidence] = []
        for evidence in claim.evidence:
            section = anchor_sections.get(evidence.anchor_id)
            if section is None:
                reasons.append(f"unknown evidence anchor: {evidence.anchor_id}")
                continue
            source_text = _select_source_text(section, evidence.selector)
            if source_text is None:
                reasons.append(
                    f"unknown evidence selector for {evidence.anchor_id}: {evidence.selector}"
                )
                continue
            resolved_quote = _resolve_markdown_formatted_quote(
                source_text, evidence.exact_quote
            )
            if resolved_quote is None:
                if evidence.selector is not None:
                    resolved_evidence.append(
                        evidence.model_copy(
                            update={
                                "exact_quote": source_text,
                                "source_text": source_text,
                            }
                        )
                    )
                    continue
                reasons.append(
                    f"exact quote not found in {evidence.anchor_id}"
                    + (f" selector {evidence.selector}" if evidence.selector else "")
                )
                continue
            resolved_evidence.append(
                evidence.model_copy(
                    update={
                        "exact_quote": resolved_quote,
                        "source_text": source_text,
                    }
                )
            )

        if reasons:
            rejected.append(
                RejectedAtomicClaim(
                    claim_id=claim.claim_id,
                    reasons=reasons,
                    raw_claim=raw_claim,
                )
            )
            continue
        accepted.append(
            claim.model_copy(
                update={
                    "module_hints": valid_module_hints,
                    "evidence": resolved_evidence,
                }
            )
        )
    return accepted, rejected


def _resolve_markdown_formatted_quote(source_text: str, quote: str) -> str | None:
    """Recover a literal source span when the model omits inline Markdown marks."""
    if quote in source_text:
        return quote
    ignored = {"*", "`"}
    normalized_chars: list[str] = []
    source_indexes: list[int] = []
    for index, char in enumerate(source_text):
        if char in ignored:
            continue
        normalized_chars.append(char)
        source_indexes.append(index)
    normalized_source = "".join(normalized_chars)
    normalized_quote = "".join(char for char in quote if char not in ignored)
    offset = normalized_source.find(normalized_quote)
    if offset < 0 or not normalized_quote:
        return None
    start = source_indexes[offset]
    end = source_indexes[offset + len(normalized_quote) - 1] + 1
    return source_text[start:end]


def supplement_structured_table_claims(
    document_package: DocumentPackage,
    claims: list[AtomicClaim],
) -> list[AtomicClaim]:
    """Enforce source-table row duties that a probabilistic discovery call may miss."""
    sections = extract_anchor_sections(document_package)
    supplemented = list(claims)
    active_group = ""
    sequence = 0
    for anchor in document_package.anchors:
        section = sections.get(anchor.anchor_id)
        if section is None:
            continue
        group = _select_source_text(section, "A列")
        if group and any(label in group for label in ("FAQ", "问答", "异议")):
            active_group = group
        question = _select_source_text(section, "B列")
        answer = _select_source_text(section, "C列")
        script = _select_source_text(section, "D列")
        if question in {"问题", "客户问题"} and answer in {"解答", "答案"}:
            continue
        if answer and script and "异议" not in active_group:
            strategy_indexes = [
                index
                for index, claim in enumerate(supplemented)
                if claim.claim_kind == "strategy"
                and any(
                    evidence.anchor_id == anchor.anchor_id
                    for evidence in claim.evidence
                )
            ]
            strategy_evidence = [
                ClaimEvidence(
                    anchor_id=anchor.anchor_id,
                    exact_quote=answer,
                    selector="C列",
                    source_text=answer,
                )
            ]
            if strategy_indexes:
                strategy_index = strategy_indexes[0]
                current = supplemented[strategy_index]
                supplemented[strategy_index] = current.model_copy(
                    update={
                        "attributes": {
                            **current.attributes,
                            "strategyDescription": answer,
                        },
                        "module_hints": list(
                            dict.fromkeys([*current.module_hints, "D3.3"])
                        ),
                        "evidence": strategy_evidence,
                    }
                )
            else:
                sequence += 1
                supplemented.append(
                    AtomicClaim(
                        claim_id=f"STRUCTURED-STRATEGY-{sequence}",
                        claim_kind="strategy",
                        statement=f"资料明确的销售策略：{answer}",
                        subject=question or answer,
                        attributes={"strategyDescription": answer},
                        module_hints=["D3.3"],
                        evidence=strategy_evidence,
                    )
                )
        duties: list[tuple[str, str, str, str]] = []
        if question and answer and any(label in active_group for label in ("FAQ", "问答")):
            duties.append(("qa", "D4.3", answer, "C列"))
        if question and script and "异议" in active_group:
            # 同一行既是“客户异议”证据，也是可直接消费的标准问答。
            # 两种知识对象职责不同，重叠使用同一来源是明确允许的。
            duties.append(("objection", "D4.2", script, "D列"))
            if _looks_like_information_question(question):
                duties.append(("qa", "D4.3", script, "D列"))
        for claim_kind, module_hint, response, response_selector in duties:
            matching_indexes = [
                index
                for index, claim in enumerate(supplemented)
                if claim.claim_kind == claim_kind
                and any(
                    evidence.anchor_id == anchor.anchor_id
                    for evidence in claim.evidence
                )
            ]
            structured_attributes = {
                "question" if claim_kind == "qa" else "expression": question,
                "answer" if claim_kind == "qa" else "responseContext": response,
            }
            structured_evidence = [
                ClaimEvidence(
                    anchor_id=anchor.anchor_id,
                    exact_quote=value,
                    selector=selector,
                    source_text=value,
                )
                for selector, value in [
                    ("B列", question),
                    (response_selector, response),
                ]
            ]
            if matching_indexes:
                primary_index = matching_indexes[0]
                current = supplemented[primary_index]
                supplemented[primary_index] = current.model_copy(
                    update={
                        "attributes": {
                            **current.attributes,
                            **structured_attributes,
                        },
                        "module_hints": list(
                            dict.fromkeys([*current.module_hints, module_hint])
                        ),
                        "evidence": structured_evidence,
                    }
                )
                continue
            sequence += 1
            supplemented.append(
                AtomicClaim(
                    claim_id=f"STRUCTURED-{claim_kind.upper()}-{sequence}",
                    claim_kind=claim_kind,
                    statement=(
                        f"标准问答：{question}"
                        if claim_kind == "qa"
                        else f"客户异议：{question}"
                    ),
                    subject=question,
                    attributes=structured_attributes,
                    module_hints=[module_hint],
                    evidence=structured_evidence,
                )
            )
    return supplemented


def _looks_like_information_question(value: str) -> bool:
    normalized = value.strip()
    return (
        normalized.endswith(("?", "？", "吗", "呢"))
        or normalized.startswith(
            ("如何", "怎么", "为什么", "是否", "能否", "可以", "可不可以")
        )
        or any(marker in normalized for marker in ("几年", "多少", "多久", "多长"))
    )


def resolve_verbatim_claim_references(
    value: Any,
    claim_by_id: dict[str, AtomicClaim],
) -> tuple[Any, list[str]]:
    """Resolve model macros to verified source text without asking it to copy long text."""
    if isinstance(value, dict):
        if set(value) == {"verbatimContent"}:
            return resolve_verbatim_claim_references(
                value["verbatimContent"], claim_by_id
            )
        if "fullText" in value and set(value) <= {"strategy", "fullText"}:
            return resolve_verbatim_claim_references(value["fullText"], claim_by_id)
        if value.get("verbatim") is True and set(value) == {"verbatim", "text"}:
            return resolve_verbatim_claim_references(value["text"], claim_by_id)
        if (
            value.get("verbatim") is True
            and isinstance(value.get("sourceRef"), str)
        ):
            claim_id = value["sourceRef"]
            if claim_id not in claim_by_id:
                return value, [f"unknown verbatim claim reference: {claim_id}"]
            source_parts = [
                evidence.source_text for evidence in claim_by_id[claim_id].evidence
            ]
            return "\n\n".join(dict.fromkeys(source_parts)), []
        if set(value) == {"$exactQuoteFromClaim"}:
            claim_id = value["$exactQuoteFromClaim"]
            if not isinstance(claim_id, str) or claim_id not in claim_by_id:
                return value, [f"unknown exact quote claim reference: {claim_id}"]
            quote_parts = [
                evidence.exact_quote for evidence in claim_by_id[claim_id].evidence
            ]
            return "\n\n".join(dict.fromkeys(quote_parts)), []
        if set(value) == {"$verbatimFromClaim"}:
            claim_id = value["$verbatimFromClaim"]
            if not isinstance(claim_id, str) or claim_id not in claim_by_id:
                return value, [f"unknown verbatim claim reference: {claim_id}"]
            source_parts = [
                evidence.source_text for evidence in claim_by_id[claim_id].evidence
            ]
            return "\n\n".join(dict.fromkeys(source_parts)), []
        resolved: dict[str, Any] = {}
        reasons: list[str] = []
        for key, item in value.items():
            resolved_item, item_reasons = resolve_verbatim_claim_references(
                item, claim_by_id
            )
            resolved[key] = resolved_item
            reasons.extend(item_reasons)
        return resolved, reasons
    if isinstance(value, list):
        resolved_items: list[Any] = []
        reasons: list[str] = []
        for item in value:
            resolved_item, item_reasons = resolve_verbatim_claim_references(
                item, claim_by_id
            )
            resolved_items.append(resolved_item)
            reasons.extend(item_reasons)
        return resolved_items, reasons
    if isinstance(value, str) and value.startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value, []
        if isinstance(parsed, dict) and (
            set(parsed) == {"$verbatimFromClaim"}
            or set(parsed) == {"$exactQuoteFromClaim"}
        ):
            return resolve_verbatim_claim_references(parsed, claim_by_id)
    return value, []


def _select_source_text(section: str, selector: str | None) -> str | None:
    if selector is None:
        return section
    escaped = re.escape(selector.strip())
    field = re.search(
        rf"(?ms)^-\s+\*\*{escaped}\*\*[:：]\s*(.*?)(?=\n-\s+\*\*|\n#{{2,4}}\s|\Z)",
        section,
    )
    return field.group(1).strip() if field else None


def _claim_id(raw_claim: Any, index: int) -> str:
    if isinstance(raw_claim, dict) and isinstance(raw_claim.get("claimId"), str):
        return raw_claim["claimId"]
    return f"INVALID-CLAIM-{index}"
