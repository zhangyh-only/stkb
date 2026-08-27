from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .models import (
    GoldGroupEvaluation,
    IdentificationQualityReport,
    IdentificationResult,
)


def evaluate_against_gold(
    result: IdentificationResult,
    gold_path: Path,
) -> IdentificationQualityReport:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    group_results = [_evaluate_group(item, result) for item in gold["expectedObjectGroups"]]
    expected_total = sum(item.expected_count for item in group_results)
    matched_total = sum(item.matched_count for item in group_results)
    used_claim_ids = {
        claim_id for candidate in result.candidates for claim_id in candidate.source_claim_ids
    }
    content_lengths = [
        len(json.dumps(candidate.content, ensure_ascii=False, sort_keys=True))
        for candidate in result.candidates
    ]
    evidence_backed = sum(
        bool(candidate.source_claim_ids and candidate.evidence)
        for candidate in result.candidates
    )
    summary_only_count = sum(
        set(candidate.content) == {"summary"} for candidate in result.candidates
    )
    findings = _build_findings(group_results, result)
    groups_met = sum(item.status == "met" for item in group_results)
    overall_status = "pass" if groups_met == len(group_results) else "fail"
    if gold.get("status") != "approved" and overall_status == "pass":
        overall_status = "review"
    return IdentificationQualityReport(
        gold_version=gold_path.name,
        gold_status=gold.get("status", "unknown"),
        overall_status=overall_status,
        expected_object_count=expected_total,
        matched_expected_count=matched_total,
        object_recall_proxy=round(matched_total / expected_total, 4)
        if expected_total
        else 0.0,
        groups_met=groups_met,
        group_count=len(group_results),
        summary_only_count=summary_only_count,
        evidence_backed_rate=round(evidence_backed / len(result.candidates), 4)
        if result.candidates
        else 0.0,
        claim_consumption_rate=round(len(used_claim_ids) / len(result.atomic_claims), 4)
        if result.atomic_claims
        else 0.0,
        median_content_chars=int(statistics.median(content_lengths))
        if content_lengths
        else 0,
        groups=group_results,
        findings=findings,
    )


def find_gold_path(workspace_root: Path, document_package_id: str) -> Path | None:
    evaluation_root = workspace_root / "evaluations" / document_package_id
    candidates = sorted(evaluation_root.glob("gold-v*.json"))
    return candidates[-1] if candidates else None


def _evaluate_group(
    group: dict[str, Any], result: IdentificationResult
) -> GoldGroupEvaluation:
    expected_evidence = set(group.get("evidence", []))
    object_types = set(group.get("objectTypes", []))
    matched_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.module == group["module"]
        and candidate.object_type in object_types
        and (not expected_evidence or expected_evidence.intersection(candidate.evidence))
    ]
    predicted_count = len(matched_candidates)
    expected_count = int(group["expectedCount"])
    required_item_count = group.get("requiredItemCount")
    predicted_item_count = None
    if required_item_count is not None:
        predicted_item_count = sum(
            len(candidate.content.get("items", []))
            for candidate in matched_candidates
            if isinstance(candidate.content.get("items", []), list)
        )
    if predicted_count == expected_count and (
        required_item_count is None or predicted_item_count == required_item_count
    ):
        status = "met"
    elif predicted_count == 0:
        status = "missed"
    elif predicted_count > expected_count:
        status = "over_split"
    else:
        status = "under_split_or_recall"
    return GoldGroupEvaluation(
        key=group["key"],
        expected_count=expected_count,
        predicted_count=predicted_count,
        matched_count=min(expected_count, predicted_count),
        status=status,
        predicted_candidate_ids=[
            candidate.candidate_id for candidate in matched_candidates
        ],
        required_item_count=required_item_count,
        predicted_item_count=predicted_item_count,
    )


def _build_findings(
    groups: list[GoldGroupEvaluation], result: IdentificationResult
) -> list[str]:
    findings = [
        f"{group.key}: 期望 {group.expected_count}，识别 {group.predicted_count}，{group.status}"
        for group in groups
        if group.status != "met"
    ]
    if result.rejected_atomic_claims:
        findings.append(f"有 {len(result.rejected_atomic_claims)} 条主张未通过逐字证据校验")
    if result.rejected_candidates:
        findings.append(f"有 {len(result.rejected_candidates)} 项对象提议未通过合同校验")
    return findings
