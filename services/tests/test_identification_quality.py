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
                        "evidence": ["A3", "A4"],
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
            "unresolvedItems": [],
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
    assert report.groups[0].status == "under_split_or_recall"
    assert report.groups[1].status == "over_split"
    assert report.groups[1].predicted_item_count == 2


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
        "content": content,
        "evidence": evidence,
    }
