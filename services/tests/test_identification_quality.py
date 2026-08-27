import json

from app.features.sales_knowledge_identification.models import IdentificationResult
from app.features.sales_knowledge_identification.quality import (
    evaluate_against_gold,
    find_gold_path,
)


def test_find_gold_path_falls_back_to_reviewable_samples(tmp_path) -> None:
    incompatible_dir = tmp_path / "workspace/evaluations/DP-SAMPLE"
    incompatible_dir.mkdir(parents=True)
    (incompatible_dir / "gold-v9.json").write_text(
        '{"expectedObjects": []}', encoding="utf-8"
    )
    samples_root = tmp_path / "samples"
    sample_dir = samples_root / "DP-SAMPLE"
    sample_dir.mkdir(parents=True)
    gold_path = sample_dir / "gold-v0.2.json"
    gold_path.write_text('{"expectedObjectGroups": []}', encoding="utf-8")

    found = find_gold_path(tmp_path / "workspace", "DP-SAMPLE", samples_root)

    assert found == gold_path


def test_quality_report_detects_group_recall_and_over_split(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.1.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "proxy_draft",
                "expectedObjectGroups": [
                    {
                        "key": "scripts",
                        "expectedCount": 2,
                        "module": "D4.1",
                        "objectTypes": ["STANDARD_SCRIPT"],
                        "evidence": ["A1", "A2"],
                    },
                    {
                        "key": "faq",
                        "expectedCount": 1,
                        "module": "D4.3",
                        "objectTypes": ["QA_PAIR"],
                        "requiredItemCount": 2,
                        "requiredContentFields": ["usageBoundary"],
                        "evidence": ["A3", "A4"],
                    },
                    {
                        "key": "unsupported-rule",
                        "expectedCount": 0,
                        "module": "D3.3",
                        "objectTypes": ["DECISION_RULE"],
                        "evidence": ["A2"],
                        "requiredUnresolvedEvidence": ["A2"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = IdentificationResult.model_validate(
        {
            "documentPackageId": "DP-QUALITY",
            "provider": "test",
            "model": "test",
            "promptVersion": "test",
            "schemaVersion": "test",
            "catalogVersion": "test",
            "catalogFingerprint": "test",
            "rawModelOutput": "{}",
            "modelCalls": [],
            "processingStages": [],
            "atomicClaims": [
                _claim("CL1", "A1"),
                _claim("CL2", "A2"),
                _claim("CL3", "A3"),
                _claim("CL4", "A4"),
            ],
            "candidates": [
                _candidate("C1", "D4.1", "STANDARD_SCRIPT", ["CL1"], ["A1"]),
                _candidate("C2", "D4.3", "QA_PAIR", ["CL3"], ["A3"], 1),
                _candidate("C3", "D4.3", "QA_PAIR", ["CL4"], ["A4"], 1),
            ],
            "rejectedCandidates": [],
            "weakSignals": [],
            "unresolvedItems": [
                {
                    "description": "CL2 不具备稳定规则证据",
                    "reason": "保留为待验证项",
                    "evidence": ["A2"],
                }
            ],
            "coverageByModule": {},
            "callCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
        }
    )

    report = evaluate_against_gold(result, gold_path)

    assert report.overall_status == "fail"
    assert report.object_recall_proxy == 0.6667
    assert report.claim_consumption_rate == 0.75
    assert report.claim_accounting_rate == 1.0
    assert report.content_attribution_rate == 0.0
    assert report.groups[0].status == "under_split_or_recall"
    assert report.groups[1].status == "over_split"
    assert report.groups[1].predicted_item_count == 2
    assert report.groups[2].status == "met"
    assert report.groups[2].missing_unresolved_evidence == []


def test_quality_report_detects_missing_required_nested_item_fields(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.1.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "proxy_draft",
                "expectedObjectGroups": [
                    {
                        "key": "term",
                        "expectedCount": 1,
                        "module": "D4.3",
                        "objectTypes": ["TERM"],
                        "evidence": ["A1", "A2"],
                        "requireAllEvidence": True,
                        "requiredItemCount": 1,
                        "requiredItemField": "items",
                        "requiredItemFields": ["sourceStance", "usageBoundary"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = IdentificationResult.model_validate(
        {
            "documentPackageId": "DP-QUALITY",
            "provider": "test",
            "model": "test",
            "promptVersion": "test",
            "schemaVersion": "test",
            "catalogVersion": "test",
            "catalogFingerprint": "test",
            "rawModelOutput": "{}",
            "modelCalls": [],
            "processingStages": [],
            "atomicClaims": [_claim("CL1", "A1")],
            "candidates": [
                _candidate("C1", "D4.3", "TERM", ["CL1"], ["A1"], 1)
            ],
            "rejectedCandidates": [],
            "weakSignals": [],
            "unresolvedItems": [],
            "coverageByModule": {},
            "callCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
        }
    )

    report = evaluate_against_gold(result, gold_path)

    assert report.overall_status == "fail"
    assert report.groups[0].status == "contract_failed"
    assert report.groups[0].missing_item_fields == [
        "sourceStance",
        "usageBoundary",
    ]
    assert report.groups[0].missing_expected_evidence == ["A2"]


def test_quality_gold_can_require_a_field_while_allowing_explicit_empty_value(
    tmp_path,
) -> None:
    gold_path = tmp_path / "gold-v0.1.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [
                    {
                        "key": "process",
                        "expectedCount": 1,
                        "module": "D1.3",
                        "objectTypes": ["BUSINESS_PROCESS"],
                        "requiredContentFields": ["purpose", "preconditions"],
                        "allowEmptyContentFields": ["preconditions"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate = _candidate("C1", "D1.3", "BUSINESS_PROCESS", ["CL1"], ["A1"])
    candidate["content"] = {
        "purpose": "完成线上问诊",
        "preconditions": [],
        "rulesOrSteps": ["进入问诊", "医生开方"],
        "exceptions": [],
    }
    result = IdentificationResult.model_validate(
        {
            "documentPackageId": "DP-QUALITY",
            "provider": "test",
            "model": "test",
            "promptVersion": "test",
            "schemaVersion": "test",
            "catalogVersion": "test",
            "catalogFingerprint": "test",
            "rawModelOutput": "{}",
            "modelCalls": [],
            "processingStages": [],
            "atomicClaims": [_claim("CL1", "A1")],
            "candidates": [candidate],
            "rejectedCandidates": [],
            "weakSignals": [],
            "unresolvedItems": [],
            "coverageByModule": {},
            "callCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
        }
    )

    report = evaluate_against_gold(result, gold_path)

    assert report.overall_status == "pass"
    assert report.groups[0].status == "met"


def _claim(claim_id: str, anchor: str) -> dict[str, object]:
    return {
        "claimId": claim_id,
        "claimKind": "script",
        "statement": "测试主张",
        "subject": "测试主体",
        "evidence": [
            {
                "anchorId": anchor,
                "exactQuote": "测试原文",
                "sourceText": "测试原文",
            }
        ],
    }


def _candidate(
    candidate_id: str,
    module: str,
    object_type: str,
    claim_ids: list[str],
    evidence: list[str],
    item_count: int = 0,
) -> dict[str, object]:
    content: dict[str, object] = {"body": "结构化知识内容" * 20}
    if item_count:
        content["items"] = [{"question": "问", "answer": "答"}] * item_count
    return {
        "candidateId": candidate_id,
        "title": "测试对象",
        "domain": module.split(".")[0],
        "module": module,
        "objectType": object_type,
        "objectBoundary": "独立消费与更新边界",
        "classificationBasis": "依据模块规则归类",
        "identityHints": {"subject": candidate_id},
        "sourceClaimIds": claim_ids,
        "claimUsage": [
            {
                "claimId": claim_id,
                "role": "primary",
                "contentPaths": ["$.body"],
                "explanation": "测试正文表达该主张",
            }
            for claim_id in claim_ids
        ],
        "content": content,
        "evidence": evidence,
    }
