from __future__ import annotations

import json
import re
import statistics
from hashlib import sha256
from pathlib import Path
from typing import Any

from .content_contracts import CONTENT_CONTRACT_VERSION
from .identity_contracts import IDENTITY_CONTRACT_VERSION
from .models import (
    DocumentPackage,
    GoldGroupEvaluation,
    IdentificationQualityReport,
    IdentificationResult,
)


def evaluate_against_gold(
    result: IdentificationResult,
    gold_path: Path,
    *,
    document_package: DocumentPackage | None = None,
) -> IdentificationQualityReport:
    gold_bytes = gold_path.read_bytes()
    gold = json.loads(gold_bytes)
    compatibility_issues = []
    if gold.get("catalogVersion") not in (None, result.catalog_version):
        compatibility_issues.append(
            f"catalogVersion: {gold.get('catalogVersion')} != {result.catalog_version}"
        )
    if gold.get("contentContractVersion") not in (None, CONTENT_CONTRACT_VERSION):
        compatibility_issues.append(
            "contentContractVersion: "
            f"{gold.get('contentContractVersion')} != {CONTENT_CONTRACT_VERSION}"
        )
    if gold.get("identityContractVersion") not in (None, IDENTITY_CONTRACT_VERSION):
        compatibility_issues.append(
            "identityContractVersion: "
            f"{gold.get('identityContractVersion')} != {IDENTITY_CONTRACT_VERSION}"
        )
    if document_package is not None:
        if gold.get("sourceSha256") not in (None, document_package.source_sha256):
            compatibility_issues.append("Gold绑定的原始文件哈希与DocumentPackage不一致")
        if gold.get("fullMarkdownSha256") not in (
            None,
            document_package.full_markdown_sha256,
        ):
            compatibility_issues.append("Gold绑定的全文Markdown哈希与DocumentPackage不一致")
        if gold.get("anchorCount") not in (None, len(document_package.anchors)):
            compatibility_issues.append("Gold绑定的来源锚点数量与DocumentPackage不一致")
    expected_groups = gold.get("expectedObjectGroups", [])
    if not isinstance(expected_groups, list):
        expected_groups = []
    positive_groups: list[dict[str, Any]] = []
    negative_groups: list[dict[str, Any]] = []
    for item in expected_groups:
        if not isinstance(item, dict):
            continue
        target = negative_groups if int(item.get("expectedCount", 0)) == 0 else positive_groups
        target.append(item)
    for key in ("negativeObjectGroups", "forbiddenObjectGroups", "negativeGroups"):
        raw_groups = gold.get(key, [])
        if not isinstance(raw_groups, list):
            continue
        for item in raw_groups:
            if isinstance(item, dict):
                negative_groups.append(
                    {
                        **item,
                        "expectedCount": 0,
                        "minimumExpectedCount": 0,
                        "maximumExpectedCount": 0,
                    }
                )
    for item in gold.get("deferredEvidenceGroups", []):
        if isinstance(item, dict) and item.get("requiredAuthoritativeSource"):
            negative_groups.append(
                {
                    **item,
                    "expectedCount": 0,
                    "minimumExpectedCount": 0,
                    "maximumExpectedCount": 0,
                    "requiredUnresolvedEvidence": item.get("evidence", []),
                }
            )
    strict_field_evidence = bool(
        gold.get("requireFieldEvidence", gold.get("checkFieldEvidence", False))
    )
    group_results = [
        _evaluate_group(item, result, require_field_evidence=strict_field_evidence)
        for item in [*positive_groups, *negative_groups]
    ]
    positive_result_count = len(positive_groups)
    candidate_groups: dict[str, list[int]] = {}
    for index, group_result in enumerate(group_results[:positive_result_count]):
        for candidate_id in group_result.predicted_candidate_ids:
            candidate_groups.setdefault(candidate_id, []).append(index)
    for candidate_id, indexes in candidate_groups.items():
        if len(indexes) < 2:
            continue
        for index in indexes:
            current = group_results[index]
            group_results[index] = current.model_copy(
                update={
                    "status": "contract_failed",
                    "matched_count": 0,
                    "ambiguous_candidate_ids": sorted(
                        {*current.ambiguous_candidate_ids, candidate_id}
                    ),
                }
            )
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
    groups_met = sum(item.status == "met" for item in group_results)
    total_content_leaves = sum(
        candidate.content_leaf_count for candidate in result.candidates
    )
    attributed_content_leaves = sum(
        candidate.attributed_content_leaf_count for candidate in result.candidates
    )
    derived_paths_by_candidate: dict[str, set[str]] = {}
    for group, group_result in zip(
        positive_groups,
        group_results[:positive_result_count],
        strict=True,
    ):
        for candidate_id in group_result.predicted_candidate_ids:
            derived_paths_by_candidate.setdefault(candidate_id, set()).update(
                group.get("derivedContentPaths", [])
            )
    (
        source_content_leaf_count,
        attributed_source_content_leaf_count,
        system_derived_content_leaf_count,
        unsupported_content_paths,
    ) = _content_attribution_metrics(result, derived_paths_by_candidate)
    source_content_attribution_rate = round(
        attributed_source_content_leaf_count / source_content_leaf_count, 4
    ) if source_content_leaf_count else 0.0
    expected_candidate_ids = {
        candidate.candidate_id
        for group in positive_groups
        for candidate in result.candidates
        if _candidate_matches_group(candidate, group, include_identity=True)
    }
    forbidden_candidate_ids = {
        candidate.candidate_id
        for group in negative_groups
        for candidate in result.candidates
        if _candidate_matches_group(candidate, group, include_identity=False)
    }
    unexpected_candidate_ids = sorted(
        {candidate.candidate_id for candidate in result.candidates}
        - expected_candidate_ids
        - forbidden_candidate_ids
    )
    findings = _build_findings(group_results, result)
    if unexpected_candidate_ids:
        findings.append(
            "识别出 Gold 未声明的多余对象：" + "、".join(unexpected_candidate_ids)
        )
    if forbidden_candidate_ids:
        findings.append(
            "识别出 Gold 明确禁止的对象：" + "、".join(sorted(forbidden_candidate_ids))
        )
    if strict_field_evidence and unsupported_content_paths:
        findings.append(
            "有正文业务字段的 claimUsage 只声明了路径，未被对应来源主张值支持："
            + "、".join(unsupported_content_paths)
        )
    rejected_noise_count, rejected_knowledge_count = _rejected_claim_counts(
        result, gold
    )
    if rejected_noise_count:
        findings.append(
            f"合理拒绝 {rejected_noise_count} 条非知识噪声，不作为发布阻断"
        )
    non_blocking_unresolved_claim_ids = _non_blocking_unresolved_claim_ids(
        result,
        negative_groups,
        group_results[positive_result_count:],
    )
    overall_status = "pass" if groups_met == len(group_results) else "fail"
    if unexpected_candidate_ids or forbidden_candidate_ids:
        overall_status = "fail"
    if strict_field_evidence and unsupported_content_paths:
        overall_status = "fail"
    if compatibility_issues:
        overall_status = "review"
    if gold.get("status") != "approved" and overall_status == "pass":
        overall_status = "review"
    return IdentificationQualityReport(
        gold_version=gold_path.name,
        gold_sha256=sha256(gold_bytes).hexdigest(),
        gold_status=gold.get("status", "unknown"),
        gold_compatible=not compatibility_issues,
        compatibility_issues=compatibility_issues,
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
        source_content_leaf_count=source_content_leaf_count,
        attributed_source_content_leaf_count=attributed_source_content_leaf_count,
        source_content_attribution_rate=source_content_attribution_rate,
        system_derived_content_leaf_count=system_derived_content_leaf_count,
        groups=group_results,
        findings=[*compatibility_issues, *findings],
        unexpected_candidate_ids=unexpected_candidate_ids,
        forbidden_candidate_ids=sorted(forbidden_candidate_ids),
        rejected_noise_count=rejected_noise_count,
        rejected_knowledge_count=rejected_knowledge_count,
        non_blocking_unresolved_claim_ids=sorted(non_blocking_unresolved_claim_ids),
        source_sha256=document_package.source_sha256 if document_package else "",
        full_markdown_sha256=(
            document_package.full_markdown_sha256 if document_package else ""
        ),
    )


def knowledge_release_blockers(result: IdentificationResult) -> list[str]:
    report = result.quality_report
    if report is None:
        return ["缺少与当前合同兼容的质量基准"]
    blockers = []
    if report.gold_status != "approved":
        blockers.append(f"Gold状态为 {report.gold_status}，尚未人工批准")
    if not report.gold_compatible:
        blockers.append("Gold与当前规则或合同版本不兼容")
    if report.overall_status != "pass":
        blockers.append(f"Gold总体结果为 {report.overall_status}")
    if report.rejected_knowledge_count:
        blockers.append(f"有 {len(result.rejected_atomic_claims)} 条来源主张被拒绝")
    elif result.rejected_atomic_claims and not report.rejected_noise_count:
        # Legacy reports predate rejection classification; preserve their gate.
        blockers.append(f"有 {len(result.rejected_atomic_claims)} 条来源主张被拒绝")
    critical_unresolved = [
        item
        for item in result.unresolved_items
        if item.claim_id
        and item.claim_id not in report.non_blocking_unresolved_claim_ids
    ]
    if critical_unresolved:
        blockers.append(f"有 {len(critical_unresolved)} 条知识主张尚未处理")
    if report.source_content_leaf_count:
        attribution_rate = report.source_content_attribution_rate
        attribution_label = "来源事实字段归因率"
    else:
        attribution_rate = report.content_attribution_rate
        attribution_label = "正文归因率"
    if attribution_rate < 1:
        blockers.append(
            f"{attribution_label} {attribution_rate:.2%}，未达到 100%"
        )
    if report.unexpected_candidate_ids:
        blockers.append(
            "Gold 未声明的多余对象：" + "、".join(report.unexpected_candidate_ids)
        )
    if report.forbidden_candidate_ids:
        blockers.append(
            "Gold 明确禁止的对象已被识别："
            + "、".join(report.forbidden_candidate_ids)
        )
    return blockers


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
    group: dict[str, Any],
    result: IdentificationResult,
    *,
    require_field_evidence: bool = False,
) -> GoldGroupEvaluation:
    expected_evidence = set(group.get("evidence", []))
    require_all_evidence = bool(group.get("requireAllEvidence", False))
    object_types = set(group.get("objectTypes", []))
    structural_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.module == group["module"]
        and candidate.object_type in object_types
        and (not expected_evidence or expected_evidence.intersection(candidate.evidence))
    ]
    required_identity_hints = _expected_identity_hints(group)
    required_identity_fields = group.get("requiredIdentityFields", [])
    if isinstance(required_identity_fields, list):
        required_identity_hints = {
            **required_identity_hints,
            **{
                field: "<non-empty>"
                for field in required_identity_fields
                if isinstance(field, str) and field not in required_identity_hints
            },
        }
    matched_candidates = [
        candidate
        for candidate in structural_candidates
        if _matches_identity_hints(candidate.identity_hints, required_identity_hints)
        and _matches_identity_rules(
            candidate.identity_hints, group.get("identityRules", {})
        )
    ]
    identity_mismatches = [
        _identity_mismatch(candidate, required_identity_hints)
        for candidate in structural_candidates
        if required_identity_hints
        and not _matches_identity_hints(candidate.identity_hints, required_identity_hints)
    ]
    identity_mismatches.extend(
        _identity_rule_mismatches(candidate, group.get("identityRules", {}))
        for candidate in structural_candidates
        if not _matches_identity_rules(
            candidate.identity_hints, group.get("identityRules", {})
        )
    )
    missing_identity_hints = sorted(
        {
            field
            for candidate in structural_candidates
            for field, expected in required_identity_hints.items()
            if expected == "<non-empty>"
            and candidate.identity_hints.get(field) in (None, "", [], {})
        }
    )
    predicted_count = len(structural_candidates)
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
    required_content_terms = group.get("requiredContentTerms", [])
    content_text = _normalize_evidence_text(
        [candidate.content for candidate in matched_candidates]
    )
    missing_content_terms = sorted(
        term
        for term in required_content_terms
        if _normalize_evidence_text(term) not in content_text
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
    required_item_keys = group.get("requiredItemKeys", [])
    item_text = _normalize_evidence_text(
        [
            item
            for candidate in matched_candidates
            for item in candidate.content.get(required_item_field, [])
        ]
    )
    missing_item_keys = sorted(
        item
        for item in required_item_keys
        if _normalize_evidence_text(item) not in item_text
    )
    predicted_item_count = None
    if required_item_count is not None:
        predicted_item_count = sum(
            len(candidate.content.get(required_item_field, []))
            for candidate in matched_candidates
            if isinstance(candidate.content.get(required_item_field, []), list)
        )
    unsupported_content_paths = (
        _unsupported_candidate_content_paths(
            matched_candidates,
            result,
            set(group.get("derivedContentPaths", [])),
        )
        if require_field_evidence
        else []
    )
    if (
        (required_identity_hints or group.get("identityRules"))
        and identity_mismatches
        and not matched_candidates
    ):
        status = "identity_failed"
    elif int(group.get("expectedCount", 1)) == 0 and predicted_count > 0:
        status = "negative_hit"
    elif int(group.get("expectedCount", 1)) == 0 and missing_unresolved_evidence:
        status = "contract_failed"
    elif (
        minimum_expected_count <= predicted_count
        and (
            maximum_expected_count is None
            or predicted_count <= maximum_expected_count
        )
        and (required_item_count is None or predicted_item_count == required_item_count)
        and not missing_content_fields
        and not missing_content_terms
        and not missing_item_fields
        and not missing_item_keys
        and not missing_unresolved_evidence
        and not missing_expected_evidence
        and not missing_identity_hints
        and not unsupported_content_paths
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
        or missing_content_terms
        or missing_item_fields
        or missing_item_keys
        or missing_unresolved_evidence
        or missing_expected_evidence
        or missing_identity_hints
        or unsupported_content_paths
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
        matched_count=(
            min(minimum_expected_count, predicted_count) if status == "met" else 0
        ),
        status=status,
        predicted_candidate_ids=[
            candidate.candidate_id for candidate in matched_candidates
        ],
        required_item_count=required_item_count,
        predicted_item_count=predicted_item_count,
        required_content_fields=required_content_fields,
        missing_content_fields=missing_content_fields,
        missing_content_terms=missing_content_terms,
        required_item_fields=required_item_fields,
        missing_item_fields=missing_item_fields,
        missing_item_keys=missing_item_keys,
        required_unresolved_evidence=required_unresolved_evidence,
        missing_unresolved_evidence=missing_unresolved_evidence,
        require_all_evidence=require_all_evidence,
        missing_expected_evidence=missing_expected_evidence,
        required_identity_hints=required_identity_hints,
        missing_identity_hints=missing_identity_hints,
        identity_mismatches=identity_mismatches,
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
    return findings


_REFERENCE_CONTENT_KEYS = {
    "anchorId",
    "claimRef",
    "evidence",
    "evidenceRefs",
    "factReferences",
    "sourceClaimIds",
    "sourceClaimId",
    "sourceRef",
    "evidenceRef",
    "exactQuote",
    "elementId",
    "stepId",
    "stepOrder",
    "category",
    "type",
}


def _expected_identity_hints(group: dict[str, Any]) -> dict[str, Any]:
    for key in ("identityHints", "expectedIdentityHints", "expectedIdentity", "identity"):
        value = group.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _matches_identity_hints(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for field, value in expected.items():
        if value == "<non-empty>":
            if actual.get(field) in (None, "", [], {}):
                return False
        elif field not in actual or not _same_identity_value(actual[field], value):
            return False
    return True


def _matches_identity_rules(actual: dict[str, Any], rules: Any) -> bool:
    if not isinstance(rules, dict):
        return True
    for field, rule in rules.items():
        if not isinstance(rule, dict):
            return False
        actual_text = _normalize_evidence_text(actual.get(field, ""))
        contains_all = rule.get("containsAll", [])
        contains_any = rule.get("containsAny", [])
        if any(_normalize_evidence_text(item) not in actual_text for item in contains_all):
            return False
        if contains_any and not any(
            _normalize_evidence_text(item) in actual_text for item in contains_any
        ):
            return False
    return True


def _identity_rule_mismatches(candidate: Any, rules: Any) -> str:
    failed_fields = [
        field
        for field, rule in rules.items()
        if not _matches_identity_rules(
            candidate.identity_hints,
            {field: rule},
        )
    ]
    return f"{candidate.candidate_id}: 身份语义不匹配 {', '.join(failed_fields)}"


def _same_identity_value(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return re.sub(r"\s+", "", actual).casefold() == re.sub(
            r"\s+", "", expected
        ).casefold()
    if isinstance(actual, list) and isinstance(expected, list):
        return sorted(map(str, actual)) == sorted(map(str, expected))
    return actual == expected


def _identity_mismatch(candidate: Any, expected: dict[str, Any]) -> str:
    differences = []
    for field, value in expected.items():
        actual = candidate.identity_hints.get(field)
        if value == "<non-empty>":
            if actual in (None, "", [], {}):
                differences.append(f"{field}=空")
        elif not _same_identity_value(actual, value):
            differences.append(f"{field}期望{value!r}实际{actual!r}")
    return f"{candidate.candidate_id}: " + "；".join(differences)


def _candidate_matches_group(
    candidate: Any, group: dict[str, Any], *, include_identity: bool
) -> bool:
    object_types = set(group.get("objectTypes", []))
    expected_evidence = set(group.get("evidence", []))
    if candidate.module != group.get("module") or candidate.object_type not in object_types:
        return False
    if expected_evidence and not expected_evidence.intersection(candidate.evidence):
        return False
    return not include_identity or _matches_identity_hints(
        candidate.identity_hints, _expected_identity_hints(group)
    ) and _matches_identity_rules(candidate.identity_hints, group.get("identityRules", {}))


def _business_content_leaf_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            if key in _REFERENCE_CONTENT_KEYS:
                continue
            paths.extend(_business_content_leaf_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        paths: list[str] = []
        for index, item in enumerate(value):
            paths.extend(_business_content_leaf_paths(item, f"{path}[{index}]"))
        return paths
    return [] if value in (None, "") else [path]


def _content_value_at_path(content: dict[str, Any], path: str) -> Any:
    current: Any = content
    for field, index_text in re.findall(r"(?:^|\.)([^.\[]+)|\[([0-9]+)\]", path[1:]):
        if field:
            if not isinstance(current, dict):
                return None
            current = current.get(field)
        else:
            if not isinstance(current, list) or int(index_text) >= len(current):
                return None
            current = current[int(index_text)]
    return current


def _content_attribution_metrics(
    result: IdentificationResult,
    derived_paths_by_candidate: dict[str, set[str]],
) -> tuple[int, int, int, list[str]]:
    source_count = attributed_count = derived_count = 0
    unsupported: list[str] = []
    claims = {claim.claim_id: claim for claim in result.atomic_claims}
    for candidate in result.candidates:
        leaves = _business_content_leaf_paths(candidate.content)
        derived = set(leaves) & derived_paths_by_candidate.get(
            candidate.candidate_id, set()
        )
        source_paths = set(leaves) - derived
        source_count += len(source_paths)
        derived_count += len(derived)
        for path in source_paths:
            if _content_path_supported(candidate, path, claims):
                attributed_count += 1
            else:
                unsupported.append(f"{candidate.candidate_id}:{path}")
    return source_count, attributed_count, derived_count, sorted(unsupported)


def _unsupported_candidate_content_paths(
    candidates: list[Any],
    result: IdentificationResult,
    derived_paths: set[str] | None = None,
) -> list[str]:
    derived_paths = derived_paths or set()
    claims = {claim.claim_id: claim for claim in result.atomic_claims}
    return sorted(
        f"{candidate.candidate_id}:{path}"
        for candidate in candidates
        for path in _business_content_leaf_paths(candidate.content)
        if path not in derived_paths and not _content_path_supported(candidate, path, claims)
    )


def _content_path_supported(
    candidate: Any, path: str, claims: dict[str, Any]
) -> bool:
    value = _content_value_at_path(candidate.content, path)
    if value in (None, "", [], {}):
        return False
    normalized_value = _normalize_evidence_text(value)
    if not normalized_value:
        return False
    for usage in candidate.claim_usage:
        if not any(
            usage_path == path
            or usage_path.startswith(path + ".")
            or path.startswith(usage_path + ".")
            for usage_path in usage.content_paths
        ):
            continue
        claim = claims.get(usage.claim_id)
        if claim is None:
            continue
        claim_text = " ".join(
            text
            for evidence in claim.evidence
            for text in (evidence.exact_quote, evidence.source_text)
        )
        if _value_supported_by_source(value, claim_text):
            return True
    return False


def _normalize_evidence_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"[\s，。；：、,.!?！？'\"“”‘’（）()]+", "", value.casefold())


def _value_supported_by_source(value: Any, source_text: str) -> bool:
    if not isinstance(value, str):
        return _normalize_evidence_text(value) in _normalize_evidence_text(source_text)
    value = value.replace("（两版本相同）", "").replace("(两版本相同)", "")
    fragments = []
    for fragment in re.split(r"[；;。：:\n，,（）()]", value):
        normalized = _normalize_evidence_text(fragment)
        for prefix in (
            "尊享版",
            "全能版",
            "问诊购药",
            "在线药房购药",
            "药房购药",
        ):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].removeprefix("为")
                break
        normalized = normalized.removeprefix("及")
        if normalized:
            fragments.append(normalized)
    normalized_source = _normalize_evidence_text(source_text)
    return bool(fragments) and all(fragment in normalized_source for fragment in fragments)


def _rejected_claim_counts(
    result: IdentificationResult, gold: dict[str, Any]
) -> tuple[int, int]:
    allowed_ids = {
        item
        for key in ("allowedRejectedClaimIds", "nonKnowledgeRejectedClaimIds")
        for item in gold.get(key, [])
        if isinstance(item, str)
    }
    noise = sum(item.claim_id in allowed_ids for item in result.rejected_atomic_claims)
    return noise, len(result.rejected_atomic_claims) - noise


def _non_blocking_unresolved_claim_ids(
    result: IdentificationResult,
    negative_groups: list[dict[str, Any]],
    negative_results: list[GoldGroupEvaluation],
) -> set[str]:
    claims = {claim.claim_id: claim for claim in result.atomic_claims}
    allowed: set[str] = set()
    for group, group_result in zip(negative_groups, negative_results, strict=True):
        if group_result.status != "met":
            continue
        evidence = set(group.get("requiredUnresolvedEvidence", []))
        claim_kinds = set(group.get("claimKinds", []))
        for item in result.unresolved_items:
            claim = claims.get(item.claim_id or "")
            if claim is None or not evidence.intersection(item.evidence):
                continue
            if claim_kinds and claim.claim_kind not in claim_kinds:
                continue
            allowed.add(claim.claim_id)
    return allowed
