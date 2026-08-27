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
        usage.claim_id for candidate in result.candidates for usage in candidate.claim_usage
    }
    unresolved_evidence = {
        evidence
        for item in [*result.unresolved_items, *result.weak_signals]
        for evidence in item.evidence
    }
    accounted_claim_ids = used_claim_ids | {
        claim.claim_id
        for claim in result.atomic_claims
        if any(
            evidence.anchor_id in unresolved_evidence for evidence in claim.evidence
        )
    }
    content_lengths = [
        len(json.dumps(candidate.content, ensure_ascii=False, sort_keys=True))
        for candidate in result.candidates
    ]
    evidence_backed = sum(
        bool(candidate.claim_usage and candidate.evidence)
        for candidate in result.candidates
    )
    summary_only_count = sum(
        set(candidate.content) == {"summary"} for candidate in result.candidates
    )
    findings = _build_findings(group_results, result)
    groups_met = sum(item.status == "met" for item in group_results)
    total_content_leaves = sum(
        candidate.content_leaf_count for candidate in result.candidates
    )
    attributed_content_leaves = sum(
        candidate.attributed_content_leaf_count for candidate in result.candidates
    )
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
        claim_accounting_rate=round(
            len(accounted_claim_ids) / len(result.atomic_claims), 4
        )
        if result.atomic_claims
        else 0.0,
        content_attribution_rate=round(
            attributed_content_leaves / total_content_leaves, 4
        )
        if total_content_leaves
        else 0.0,
        median_content_chars=int(statistics.median(content_lengths))
        if content_lengths
        else 0,
        groups=group_results,
        findings=findings,
    )


def find_gold_path(
    workspace_root: Path,
    document_package_id: str,
    samples_root: Path | None = None,
) -> Path | None:
    workspace_candidates = sorted(
        (workspace_root / "evaluations" / document_package_id).glob("gold-v*.json")
    )
    sample_candidates: list[Path] = []
    if samples_root is not None:
        sample_candidates = sorted(
            (samples_root / document_package_id).glob("gold-v*.json")
        )
    # Committed samples are the reviewable regression truth.  A stale local
    # workspace draft must not silently shadow the project baseline.
    for candidates in (sample_candidates, workspace_candidates):
        for candidate in reversed(candidates):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload.get("expectedObjectGroups"), list):
                return candidate
    return None


def _evaluate_group(
    group: dict[str, Any], result: IdentificationResult
) -> GoldGroupEvaluation:
    expected_evidence = set(group.get("evidence", []))
    require_all_evidence = bool(group.get("requireAllEvidence", False))
    object_types = set(group.get("objectTypes", []))
    matched_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.module == group["module"]
        and candidate.object_type in object_types
        and (not expected_evidence or expected_evidence.intersection(candidate.evidence))
    ]
    predicted_count = len(matched_candidates)
    predicted_evidence = {
        evidence
        for candidate in matched_candidates
        for evidence in candidate.evidence
    }
    missing_expected_evidence = (
        sorted(expected_evidence - predicted_evidence) if require_all_evidence else []
    )
    expected_count = int(group["expectedCount"])
    minimum_expected_count = int(group.get("minimumExpectedCount", expected_count))
    raw_maximum_expected_count = group.get("maximumExpectedCount", expected_count)
    maximum_expected_count = (
        int(raw_maximum_expected_count)
        if raw_maximum_expected_count is not None
        else None
    )
    required_item_count = group.get("requiredItemCount")
    required_item_field = group.get("requiredItemField", "items")
    required_item_fields = group.get("requiredItemFields", [])
    required_unresolved_evidence = group.get("requiredUnresolvedEvidence", [])
    unresolved_evidence = {
        evidence
        for item in [*result.unresolved_items, *result.weak_signals]
        for evidence in item.evidence
    }
    missing_unresolved_evidence = sorted(
        set(required_unresolved_evidence) - unresolved_evidence
    )
    required_content_fields = group.get("requiredContentFields", [])
    allow_empty_content_fields = set(group.get("allowEmptyContentFields", []))
    missing_content_fields = sorted(
        {
            field
            for candidate in matched_candidates
            for field in required_content_fields
            if field not in candidate.content
            or (
                field not in allow_empty_content_fields
                and candidate.content.get(field) in (None, "", [], {})
            )
        }
    )
    missing_item_fields = sorted(
        {
            field
            for candidate in matched_candidates
            for item in candidate.content.get(required_item_field, [])
            if isinstance(item, dict)
            for field in required_item_fields
            if item.get(field) in (None, "", [], {})
        }
    )
    predicted_item_count = None
    if required_item_count is not None:
        predicted_item_count = sum(
            len(candidate.content.get(required_item_field, []))
            for candidate in matched_candidates
            if isinstance(candidate.content.get(required_item_field, []), list)
        )
    if (
        minimum_expected_count <= predicted_count
        and (
            maximum_expected_count is None
            or predicted_count <= maximum_expected_count
        )
        and (required_item_count is None or predicted_item_count == required_item_count)
        and not missing_content_fields
        and not missing_item_fields
        and not missing_unresolved_evidence
        and not missing_expected_evidence
    ):
        status = "met"
    elif predicted_count == 0:
        status = "missed"
    elif (
        maximum_expected_count is not None
        and predicted_count > maximum_expected_count
    ):
        status = "over_split"
    elif (
        missing_content_fields
        or missing_item_fields
        or missing_unresolved_evidence
        or missing_expected_evidence
    ):
        status = "contract_failed"
    elif predicted_count < minimum_expected_count:
        status = "under_split_or_recall"
    else:
        status = "contract_failed"
    return GoldGroupEvaluation(
        key=group["key"],
        expected_count=expected_count,
        minimum_expected_count=minimum_expected_count,
        maximum_expected_count=maximum_expected_count,
        predicted_count=predicted_count,
        matched_count=min(minimum_expected_count, predicted_count),
        status=status,
        predicted_candidate_ids=[
            candidate.candidate_id for candidate in matched_candidates
        ],
        required_item_count=required_item_count,
        predicted_item_count=predicted_item_count,
        required_content_fields=required_content_fields,
        missing_content_fields=missing_content_fields,
        required_item_fields=required_item_fields,
        missing_item_fields=missing_item_fields,
        required_unresolved_evidence=required_unresolved_evidence,
        missing_unresolved_evidence=missing_unresolved_evidence,
        require_all_evidence=require_all_evidence,
        missing_expected_evidence=missing_expected_evidence,
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
    candidates_without_usage = sum(
        not candidate.claim_usage for candidate in result.candidates
    )
    if candidates_without_usage:
        findings.append(
            f"有 {candidates_without_usage} 项对象没有正文级主张消费路径，"
            "不能按计划来源计入消费率"
        )
    unattributed_fields = sum(
        len(candidate.unattributed_content_paths) for candidate in result.candidates
    )
    if unattributed_fields:
        findings.append(
            f"有 {unattributed_fields} 个正文业务字段尚未建立 claimUsage 归因路径"
        )
    return findings
