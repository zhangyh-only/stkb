from __future__ import annotations

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
    existing = {
        (claim.claim_kind, evidence.anchor_id)
        for claim in claims
        for evidence in claim.evidence
    }
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
        claim_kind: str | None = None
        module_hint = ""
        source_values: list[tuple[str, str]] = []
        if question and answer and any(label in active_group for label in ("FAQ", "问答")):
            claim_kind = "qa"
            module_hint = "D4.3"
            source_values = [("B列", question), ("C列", answer)]
        elif question and script and "异议" in active_group:
            claim_kind = "objection"
            module_hint = "D4.2"
            source_values = [("B列", question), ("D列", script)]
        if claim_kind is None or (claim_kind, anchor.anchor_id) in existing:
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
                attributes={
                    "question" if claim_kind == "qa" else "expression": question,
                    "answer" if claim_kind == "qa" else "responseContext": (
                        answer if claim_kind == "qa" else script
                    ),
                },
                module_hints=[module_hint],
                evidence=[
                    ClaimEvidence(
                        anchor_id=anchor.anchor_id,
                        exact_quote=value,
                        selector=selector,
                        source_text=value,
                    )
                    for selector, value in source_values
                ],
            )
        )
        existing.add((claim_kind, anchor.anchor_id))
    return supplemented


def resolve_verbatim_claim_references(
    value: Any,
    claim_by_id: dict[str, AtomicClaim],
) -> tuple[Any, list[str]]:
    """Resolve model macros to verified source text without asking it to copy long text."""
    if isinstance(value, dict):
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
