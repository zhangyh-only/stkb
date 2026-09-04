import json

from app.features.sales_knowledge_identification.models import (
    DocumentPackage,
    IdentificationResult,
    SourceAnchor,
)
from app.features.sales_knowledge_identification.quality import (
    evaluate_against_gold,
    find_gold_path,
    knowledge_release_blockers,
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


def test_find_gold_path_prefers_reviewable_sample_over_stale_workspace_copy(
    tmp_path,
) -> None:
    workspace_dir = tmp_path / "workspace/evaluations/DP-SAMPLE"
    workspace_dir.mkdir(parents=True)
    workspace_gold = workspace_dir / "gold-v9.json"
    workspace_gold.write_text(
        '{"expectedObjectGroups": [{"key": "stale"}]}', encoding="utf-8"
    )
    sample_dir = tmp_path / "samples/DP-SAMPLE"
    sample_dir.mkdir(parents=True)
    sample_gold = sample_dir / "gold-v0.2.json"
    sample_gold.write_text(
        '{"expectedObjectGroups": [{"key": "reviewable"}]}', encoding="utf-8"
    )

    found = find_gold_path(
        tmp_path / "workspace", "DP-SAMPLE", tmp_path / "samples"
    )

    assert found == sample_gold


def test_quality_report_marks_stale_catalog_gold_incompatible(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.1.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "proxy_draft",
                "catalogVersion": "d1-d5-v0.8",
                "expectedObjectGroups": [],
            }
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
            "catalogVersion": "d1-d5-v0.9",
            "catalogFingerprint": "test",
                "rawModelOutput": "{}",
                "modelCalls": [],
                "processingStages": [],
                "candidates": [],
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

    assert report.gold_compatible is False
    assert report.overall_status == "review"
    assert report.compatibility_issues == [
        "catalogVersion: d1-d5-v0.8 != d1-d5-v0.9"
    ]


def test_quality_report_binds_gold_to_source_package_hashes(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "sourceSha256": "wrong",
                "fullMarkdownSha256": "markdown-sha",
                "anchorCount": 1,
                "expectedObjectGroups": [],
            }
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
            "candidates": [],
            "rejectedCandidates": [],
            "weakSignals": [],
            "unresolvedItems": [],
            "coverageByModule": {},
            "callCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
        }
    )
    package = DocumentPackage(
        document_package_id="DP-QUALITY",
        workspace_id="WS-TEST",
        source_file_name="sample.pptx",
        source_sha256="source-sha",
        full_markdown_path="workspace/documents/DP-QUALITY/full.md",
        full_markdown_sha256="markdown-sha",
        full_markdown="正文",
        processing_method="agent_assisted",
        status="available",
        anchors=[SourceAnchor(anchor_id="A1", kind="page", page=1)],
        quality_issues=[],
    )

    report = evaluate_against_gold(result, gold_path, document_package=package)

    assert report.gold_compatible is False
    assert report.source_sha256 == "source-sha"
    assert report.full_markdown_sha256 == "markdown-sha"


def test_release_gate_blocks_unapproved_or_incomplete_quality() -> None:
    result = IdentificationResult.model_validate(
        {
            "documentPackageId": "DP-QUALITY",
            "provider": "test",
            "model": "test",
            "promptVersion": "test",
            "schemaVersion": "test",
            "catalogVersion": "d1-d5-v0.9",
            "catalogFingerprint": "test",
            "rawModelOutput": "{}",
            "modelCalls": [],
            "processingStages": [],
            "candidates": [],
            "rejectedCandidates": [],
            "weakSignals": [],
            "unresolvedItems": [],
            "coverageByModule": {},
            "callCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
            "qualityReport": {
                "goldVersion": "gold-v0.2.json",
                "goldStatus": "proxy_draft",
                "overallStatus": "review",
                "expectedObjectCount": 0,
                "matchedExpectedCount": 0,
                "objectRecallProxy": 0,
                "groupsMet": 0,
                "groupCount": 0,
                "summaryOnlyCount": 0,
                "evidenceBackedRate": 1,
                "claimConsumptionRate": 1,
                "contentAttributionRate": 0.8,
                "medianContentChars": 0,
                "groups": [],
                "findings": [],
            },
        }
    )

    blockers = knowledge_release_blockers(result)

    assert blockers == [
        "Gold状态为 proxy_draft，尚未人工批准",
        "Gold总体结果为 review",
        "正文归因率 80.00%，未达到 100%",
    ]


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
    assert report.object_recall_proxy == 0.0
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


def test_quality_gold_accepts_reviewed_object_count_range(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.1.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [
                    {
                        "key": "compliance",
                        "expectedCount": 1,
                        "minimumExpectedCount": 1,
                        "maximumExpectedCount": 2,
                        "module": "D3.4",
                        "objectTypes": ["COMPLIANCE_RULE"],
                    }
                ],
            }
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
            "atomicClaims": [_claim("CL1", "A1"), _claim("CL2", "A2")],
            "candidates": [
                _candidate("C1", "D3.4", "COMPLIANCE_RULE", ["CL1"], ["A1"]),
                _candidate("C2", "D3.4", "COMPLIANCE_RULE", ["CL2"], ["A2"]),
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

    assert report.groups[0].status == "met"
    assert report.groups[0].minimum_expected_count == 1
    assert report.groups[0].maximum_expected_count == 2


def test_legacy_quality_report_defaults_new_trace_metrics() -> None:
    result = IdentificationResult.model_validate(
        {
            "documentPackageId": "DP-LEGACY",
            "provider": "test",
            "model": "test",
            "promptVersion": "old",
            "schemaVersion": "old",
            "catalogVersion": "old",
            "catalogFingerprint": "old",
            "rawModelOutput": "{}",
            "modelCalls": [],
            "processingStages": [],
            "candidates": [],
            "rejectedCandidates": [],
            "weakSignals": [],
            "unresolvedItems": [],
            "coverageByModule": {},
            "callCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
            "qualityReport": {
                "goldVersion": "gold-v0.json",
                "goldStatus": "proxy_draft",
                "overallStatus": "fail",
                "expectedObjectCount": 0,
                "matchedExpectedCount": 0,
                "objectRecallProxy": 0,
                "groupsMet": 0,
                "groupCount": 0,
                "summaryOnlyCount": 0,
                "evidenceBackedRate": 0,
                "claimConsumptionRate": 0,
                "medianContentChars": 0,
                "groups": [],
                "findings": [],
            },
        }
    )

    assert result.quality_report is not None
    assert result.quality_report.claim_accounting_rate == 0
    assert result.quality_report.content_attribution_rate == 0


def test_quality_report_rejects_undeclared_and_negative_candidates(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [
                    {
                        "key": "facts",
                        "expectedCount": 1,
                        "module": "D1.1",
                        "objectTypes": ["PRODUCT_FACT"],
                        "evidence": ["A1"],
                    }
                ],
                "negativeObjectGroups": [
                    {
                        "key": "bundle-strategy",
                        "module": "D3.2",
                        "objectTypes": ["SALES_STRATEGY"],
                        "evidence": ["A2"],
                    }
                ],
            }
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
            "atomicClaims": [_claim("CL1", "A1"), _claim("CL2", "A2")],
            "candidates": [
                _candidate("C1", "D1.1", "PRODUCT_FACT", ["CL1"], ["A1"]),
                _candidate("C2", "D3.2", "SALES_STRATEGY", ["CL2"], ["A2"]),
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
    assert report.forbidden_candidate_ids == ["C2"]
    assert report.unexpected_candidate_ids == []
    assert report.groups[-1].status == "negative_hit"


def test_quality_report_treats_authority_deferred_groups_as_unresolved_only(
    tmp_path,
) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [],
                "deferredEvidenceGroups": [
                    {
                        "key": "policy-background",
                        "module": "D1.1",
                        "objectTypes": ["PRODUCT_FACT"],
                        "evidence": ["A1"],
                        "requiredAuthoritativeSource": True,
                    }
                ],
            }
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
                _candidate("C1", "D1.1", "PRODUCT_FACT", ["CL1"], ["A1"])
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
    assert report.forbidden_candidate_ids == ["C1"]
    assert report.groups[0].missing_unresolved_evidence == ["A1"]


def test_quality_report_checks_gold_identity_hints(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [
                    {
                        "key": "facts",
                        "expectedCount": 1,
                        "module": "D1.1",
                        "objectTypes": ["PRODUCT_FACT"],
                        "evidence": ["A1"],
                        "identityHints": {"subject": "权威产品"},
                    }
                ],
            }
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
                _candidate("C1", "D1.1", "PRODUCT_FACT", ["CL1"], ["A1"])
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
    assert report.groups[0].status == "identity_failed"
    assert report.groups[0].identity_mismatches == [
        "C1: subject期望'权威产品'实际'C1'"
    ]
    assert report.unexpected_candidate_ids == ["C1"]


def test_quality_report_separates_supported_source_fields_from_derived_fields(
    tmp_path,
) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "requireFieldEvidence": True,
                "expectedObjectGroups": [
                    {
                        "key": "version",
                        "expectedCount": 1,
                        "module": "D1.1",
                        "objectTypes": ["PRODUCT_VERSION_FACT"],
                        "evidence": ["A1"],
                        "derivedContentPaths": [
                            "$.applicability.product",
                            "$.applicability.version",
                        ],
                    }
                ],
            }
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
                    {
                        **_claim("A1", "权威产品"),
                        "claimId": "CL1",
                        "statement": "权威产品事实",
                        "subject": "权威产品",
                        "evidence": [
                            {
                                "anchorId": "A1",
                                "exactQuote": "权威产品事实",
                                "sourceText": "权威产品事实",
                            }
                        ],
                }
            ],
            "candidates": [
                {
                    **_candidate(
                        "C1", "D1.1", "PRODUCT_VERSION_FACT", ["CL1"], ["A1"]
                    ),
                    "identityHints": {
                        "subject": "权威产品",
                        "versionScope": "V1",
                        "factTheme": "事实",
                    },
                    "content": {
                        "subject": "权威产品",
                        "facts": [{"description": "权威产品事实"}],
                        "applicability": {"product": "权威产品", "version": "V1"},
                        "limitations": [],
                    },
                    "claimUsage": [
                        {
                            "claimId": "CL1",
                            "role": "primary",
                            "contentPaths": ["$.subject", "$.facts[0].description"],
                            "explanation": "来源字段",
                        }
                    ],
                }
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

    assert report.overall_status == "pass"
    assert report.source_content_leaf_count == 2
    assert report.attributed_source_content_leaf_count == 2
    assert report.source_content_attribution_rate == 1
    assert report.system_derived_content_leaf_count == 2


def test_quality_report_classifies_allowed_noise_rejection_as_non_blocking(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [],
                "allowedRejectedClaimIds": ["NOISE-1"],
            }
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
            "atomicClaims": [],
            "rejectedAtomicClaims": [
                {
                    "claimId": "NOISE-1",
                    "reasons": ["内容不构成销售知识"],
                    "rawClaim": {"claimId": "NOISE-1", "claimKind": "fact"},
                }
            ],
            "candidates": [],
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
    result = result.model_copy(update={"quality_report": report})

    blockers = knowledge_release_blockers(result)

    assert report.rejected_noise_count == 1
    assert report.rejected_knowledge_count == 0
    assert not any("来源主张被拒绝" in item for item in blockers)


def test_one_candidate_cannot_satisfy_two_gold_objects(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [
                    {
                        "key": "flow-a",
                        "expectedCount": 1,
                        "module": "D1.2",
                        "objectTypes": ["BUSINESS_PROCESS"],
                        "evidence": ["A1"],
                    },
                    {
                        "key": "flow-b",
                        "expectedCount": 1,
                        "module": "D1.2",
                        "objectTypes": ["BUSINESS_PROCESS"],
                        "evidence": ["A2"],
                    },
                ],
            }
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
            "atomicClaims": [_claim("CL1", "A1"), _claim("CL2", "A2")],
            "candidates": [
                _candidate(
                    "C1", "D1.2", "BUSINESS_PROCESS", ["CL1", "CL2"], ["A1", "A2"]
                )
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

    assert report.object_recall_proxy == 0
    assert [group.ambiguous_candidate_ids for group in report.groups] == [
        ["C1"],
        ["C1"],
    ]


def test_identity_rules_reject_nonempty_but_wrong_identity(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [
                    {
                        "key": "faq",
                        "expectedCount": 1,
                        "module": "D4.2",
                        "objectTypes": ["QA_PAIR"],
                        "identityRules": {
                            "subject": {"containsAll": ["药享保"]}
                        },
                    }
                ],
            }
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
            "candidates": [
                _candidate("C1", "D4.2", "QA_PAIR", ["CL1"], ["A1"])
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

    assert report.groups[0].status == "identity_failed"
    assert report.groups[0].matched_count == 0


def test_model_attributes_cannot_self_prove_content_field(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "requireFieldEvidence": True,
                "expectedObjectGroups": [
                    {
                        "key": "fact",
                        "expectedCount": 1,
                        "module": "D1.1",
                        "objectTypes": ["PRODUCT_FACT"],
                        "evidence": ["A1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate = _candidate("C1", "D1.1", "PRODUCT_FACT", ["CL1"], ["A1"])
    candidate["content"] = {"invented": "不存在于原文的值"}
    candidate["claimUsage"] = [
        {
            "claimId": "CL1",
            "role": "primary",
            "contentPaths": ["$.invented"],
            "explanation": "模型声称已支持",
        }
    ]
    claim = _claim("CL1", "A1")
    claim["attributes"] = {"invented": "不存在于原文的值"}
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
            "atomicClaims": [claim],
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

    assert report.overall_status == "fail"
    assert report.source_content_attribution_rate == 0


def test_non_blocking_unresolved_uses_claim_kind_not_whole_anchor(tmp_path) -> None:
    gold_path = tmp_path / "gold-v0.3.json"
    gold_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "expectedObjectGroups": [],
                "negativeObjectGroups": [
                    {
                        "key": "partial-list",
                        "module": "D1.1",
                        "objectTypes": ["LIST_FACT"],
                        "evidence": ["A1"],
                        "requiredUnresolvedEvidence": ["A1"],
                        "claimKinds": ["list"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fact_claim = _claim("CL2", "A1")
    list_claim = {**_claim("CL1", "A1"), "claimKind": "list"}
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
            "atomicClaims": [list_claim, fact_claim],
            "candidates": [],
            "rejectedCandidates": [],
            "weakSignals": [],
            "unresolvedItems": [
                {
                    "claimId": "CL1",
                    "description": "清单不完整",
                    "reason": "等待完整清单",
                    "evidence": ["A1"],
                },
                {
                    "claimId": "CL2",
                    "description": "事实遗漏",
                    "reason": "仍需处理",
                    "evidence": ["A1"],
                },
            ],
            "coverageByModule": {},
            "callCount": 0,
            "promptTokens": 0,
            "completionTokens": 0,
        }
    )
    report = evaluate_against_gold(result, gold_path)
    result = result.model_copy(update={"quality_report": report})

    assert report.non_blocking_unresolved_claim_ids == ["CL1"]
    assert "有 1 条知识主张尚未处理" in knowledge_release_blockers(result)


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
