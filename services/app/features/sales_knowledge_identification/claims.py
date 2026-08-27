from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .catalog import MODULE_BY_CODE
from .models import AtomicClaim, ClaimEvidence, DocumentPackage, RejectedAtomicClaim

SOURCE_ANCHOR = re.compile(r"<!--\s*source-anchor:\s*([^\s>]+)\s*-->")
MARKDOWN_HEADING = re.compile(r"(?m)^#{2,4}\s+[^\n]+$")


def extract_anchor_sections(document_package: DocumentPackage) -> dict[str, str]:
    """Return the smallest structural Markdown section containing each anchor."""
    markdown = document_package.full_markdown
    anchor_matches = list(SOURCE_ANCHOR.finditer(markdown))
    if not anchor_matches:
        if len(document_package.anchors) == 1:
            return {document_package.anchors[0].anchor_id: markdown}
        return {}

    heading_matches = list(MARKDOWN_HEADING.finditer(markdown))
    result: dict[str, str] = {}
    for index, anchor_match in enumerate(anchor_matches):
        start = _nearest_heading_start(heading_matches, anchor_match.start())
        if index + 1 < len(anchor_matches):
            end = _nearest_heading_start(heading_matches, anchor_matches[index + 1].start())
            if end <= start:
                end = anchor_matches[index + 1].start()
        else:
            end = len(markdown)
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
            if evidence.exact_quote not in source_text:
                reasons.append(
                    f"exact quote not found in {evidence.anchor_id}"
                    + (f" selector {evidence.selector}" if evidence.selector else "")
                )
                continue
            resolved_evidence.append(
                evidence.model_copy(update={"source_text": source_text})
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


def resolve_verbatim_claim_references(
    value: Any,
    claim_by_id: dict[str, AtomicClaim],
) -> tuple[Any, list[str]]:
    """Resolve model macros to verified source text without asking it to copy long text."""
    if isinstance(value, dict):
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


def _nearest_heading_start(matches: list[re.Match[str]], position: int) -> int:
    starts = [match.start() for match in matches if match.start() < position]
    return starts[-1] if starts else 0


def _select_source_text(section: str, selector: str | None) -> str | None:
    if selector is None:
        return section
    escaped = re.escape(selector.strip())
    field = re.search(
        rf"(?ms)^-\s+\*\*{escaped}\*\*[:：]\s*(.*?)(?=\n-\s+\*\*|\Z)",
        section,
    )
    return field.group(1).strip() if field else None


def _claim_id(raw_claim: Any, index: int) -> str:
    if isinstance(raw_claim, dict) and isinstance(raw_claim.get("claimId"), str):
        return raw_claim["claimId"]
    return f"INVALID-CLAIM-{index}"
