from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from .catalog import (
    CATALOG_FINGERPRINT,
    CATALOG_VERSION,
    KNOWLEDGE_MODULES,
    MODULE_BY_CODE,
    validate_candidate_classification,
)
from .claims import (
    resolve_verbatim_claim_references,
    supplement_structured_table_claims,
    validate_atomic_claims,
)
from .content_contracts import CONTENT_CONTRACT_BY_MODULE, validate_candidate_content
from .identity_contracts import (
    IDENTITY_CONTRACT_BY_MODULE,
    canonical_identity,
    validate_identity_hints,
)
from .models import (
    AtomicClaim,
    CandidateKnowledgeObject,
    CandidateNormalization,
    CandidateObjectPlan,
    ContentClaimUsage,
    DocumentPackage,
    IdentificationResult,
    ModelCallTrace,
    ModelCompletion,
    ModelConfigurationSnapshot,
    ModelRequest,
    ProcessingStage,
    RejectedAtomicClaim,
    RejectedAuxiliaryItem,
    RejectedCandidate,
    RejectedObjectPlan,
    UnresolvedItem,
    WeakSignal,
)
from .prompt_builder import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_claim_discovery_request,
    build_content_realization_request,
    build_object_planning_request,
    build_plan_coverage_repair_request,
    build_repair_request,
)
from .segmenter import DocumentSegment, segment_document

CallPurpose = Literal["claim_discovery", "object_planning", "content_realization"]
CONTENT_REALIZATION_BATCH_SIZE = 5
PRIMARY_MODULE_PREFIXES_BY_CLAIM_KIND: dict[str, tuple[str, ...]] = {
    "fact": ("D1.",),
    "list": ("D1.2",),
    "process": ("D1.3",),
    "rule": ("D1.3", "D3.3", "D3.4", "D5.3"),
    "comparison": ("D1.4",),
    "customer_signal": ("D2.",),
    "method": ("D3.2",),
    "strategy": ("D3.2", "D3.3"),
    "script": ("D4.1",),
    "objection": ("D4.2",),
    "qa": ("D4.3",),
    "term": ("D4.3",),
    "case": ("D4.4",),
    "asset": ("D4.5",),
    "value_proposition": ("D5.1",),
    "evaluation": ("D5.2", "D5.3"),
    "benchmark": ("D5.4",),
}


class ModelGateway(Protocol):
    def complete(self, request: ModelRequest) -> ModelCompletion: ...


class SegmentIdentificationFailure(RuntimeError):
    def __init__(self, message: str, model_calls: list[ModelCallTrace]) -> None:
        super().__init__(message)
        self.model_calls = model_calls


class DocumentPackageUnavailable(ValueError):
    pass


class SalesKnowledgeIdentificationService:
    def __init__(
        self,
        gateway: ModelGateway,
        max_retries: int = 0,
        max_candidates: int = 10,
        document_max_chars: int = 3500,
        max_concurrency: int = 3,
        provider: str = "unknown",
        model: str = "unknown",
        model_configuration: ModelConfigurationSnapshot | None = None,
    ) -> None:
        self.gateway = gateway
        self.max_retries = max_retries
        self.max_candidates = max_candidates
        self.document_max_chars = document_max_chars
        self.claim_discovery_max_chars = max(1, document_max_chars // 2)
        self.max_concurrency = max_concurrency
        self.provider = provider
        self.model = model
        self.model_configuration = model_configuration

    def identify(self, document_package: DocumentPackage) -> IdentificationResult:
        if document_package.status != "available":
            raise DocumentPackageUnavailable(document_package.document_package_id)
        started_at = datetime.now(UTC)
        run_started = perf_counter()
        segments = segment_document(
            document_package, self.claim_discovery_max_chars
        )
        model_calls: list[ModelCallTrace] = []

        discovery_results, discovery_failures = self._discover_claims(
            document_package, segments
        )
        model_calls.extend(_renumber_calls(discovery_results, model_calls))
        if discovery_failures:
            return self._failed_result(
                document_package=document_package,
                started_at=started_at,
                run_started=run_started,
                model_calls=model_calls,
                failures=discovery_failures,
            )

        atomic_claims: list[AtomicClaim] = []
        rejected_atomic_claims: list[RejectedAtomicClaim] = []
        for segment, payload, _completion, _calls in discovery_results:
            namespaced = _namespace_claim_payload(
                payload, f"S{segment.index}" if len(segments) > 1 else None
            )
            segment_package = document_package.model_copy(
                update={"full_markdown": segment.markdown, "anchors": segment.anchors}
            )
            accepted, rejected = validate_atomic_claims(
                namespaced.get("claims", []), segment_package
            )
            atomic_claims.extend(accepted)
            rejected_atomic_claims.extend(rejected)

        atomic_claims = supplement_structured_table_claims(
            document_package, atomic_claims
        )

        planning_result, planning_failures = self._plan_candidate_objects(
            document_package.document_package_id, atomic_claims
        )
        model_calls.extend(_renumber_calls(planning_result, model_calls))
        if planning_failures:
            return self._failed_result(
                document_package=document_package,
                started_at=started_at,
                run_started=run_started,
                model_calls=model_calls,
                failures=planning_failures,
                atomic_claims=atomic_claims,
                rejected_atomic_claims=rejected_atomic_claims,
            )

        planning_validation_started = perf_counter()
        (
            object_plans,
            rejected_object_plans,
            planning_weak_inputs,
            planning_unresolved_inputs,
        ) = _validate_object_plans(planning_result, atomic_claims)
        for _repair_pass in range(2):
            uncovered_claim_ids = _automatic_uncovered_claim_ids(
                planning_unresolved_inputs
            )
            if not uncovered_claim_ids:
                break
            repair_claims = [
                claim
                for claim in atomic_claims
                if claim.claim_id in uncovered_claim_ids
            ]
            repair_result, repair_failures = self._repair_uncovered_object_plans(
                document_package.document_package_id,
                object_plans,
                repair_claims,
            )
            model_calls.extend(_renumber_calls(repair_result, model_calls))
            if repair_failures:
                break
            repair_payload = repair_result[0][2] if repair_result else {}
            object_plans, augmentation_rejections = _apply_plan_augmentations(
                object_plans,
                repair_payload.get("planAugmentations", []),
                repair_claims,
            )
            (
                repair_plans,
                repair_rejections,
                repair_weak_inputs,
                repair_unresolved_inputs,
            ) = _validate_object_plans(repair_result, repair_claims)
            object_plans, duplicate_rejections = _merge_repair_plans(
                object_plans, repair_plans
            )
            rejected_object_plans.extend(
                [
                    *augmentation_rejections,
                    *repair_rejections,
                    *duplicate_rejections,
                ]
            )
            claim_by_id = {
                claim.claim_id: claim for claim in atomic_claims
            }
            covered_claim_ids = {
                claim_id
                for plan in object_plans
                for claim_id in plan.source_claim_ids
                if claim_id in claim_by_id
                and _plan_satisfies_primary_claim_role(
                    plan, claim_by_id[claim_id]
                )
            }
            planning_unresolved_inputs = [
                item
                for item in planning_unresolved_inputs
                if item[0].get("claimId") not in uncovered_claim_ids
            ]
            planning_unresolved_inputs.extend(
                item
                for item in repair_unresolved_inputs
                if item[0].get("claimId") not in covered_claim_ids
            )
            planning_weak_inputs.extend(repair_weak_inputs)
        remaining_uncovered_ids = _automatic_uncovered_claim_ids(
            planning_unresolved_inputs
        )
        structured_objections = [
            claim
            for claim in atomic_claims
            if claim.claim_id in remaining_uncovered_ids
            and claim.claim_id.startswith("STRUCTURED-OBJECTION-")
        ]
        if structured_objections:
            guard_result = [
                (
                    "structured-duty-guard",
                    structured_objections,
                    {
                        "objectPlans": [
                            [
                                f"G-D42-{index}",
                                f"客户异议：{claim.subject}",
                                "D4.2",
                                "CUSTOMER_OBJECTION",
                                {
                                    "rootConcern": claim.subject,
                                    "context": "来源资料中的销售咨询与异议处理",
                                },
                                [claim.claim_id],
                            ]
                            for index, claim in enumerate(
                                structured_objections, start=1
                            )
                        ],
                        "weakSignals": [],
                        "unresolvedItems": [],
                    },
                    None,
                    [],
                )
            ]
            (
                guard_plans,
                guard_rejections,
                _guard_weak_inputs,
                guard_unresolved_inputs,
            ) = _validate_object_plans(guard_result, structured_objections)
            object_plans, guard_duplicate_rejections = _merge_repair_plans(
                object_plans, guard_plans
            )
            rejected_object_plans.extend(
                [*guard_rejections, *guard_duplicate_rejections]
            )
            guarded_claim_ids = {
                claim_id
                for plan in guard_plans
                for claim_id in plan.source_claim_ids
            }
            planning_unresolved_inputs = [
                item
                for item in planning_unresolved_inputs
                if item[0].get("claimId") not in guarded_claim_ids
            ]
            planning_unresolved_inputs.extend(guard_unresolved_inputs)
        object_plans = _enforce_plan_granularity(object_plans, atomic_claims)
        planning_validation_duration_ms = round(
            (perf_counter() - planning_validation_started) * 1000
        )

        realization_groups = _group_object_plans(
            object_plans, atomic_claims, CONTENT_REALIZATION_BATCH_SIZE
        )
        object_results, object_failures = self._realize_candidate_objects(
            document_package.document_package_id, realization_groups
        )
        model_calls.extend(_renumber_calls(object_results, model_calls))
        if object_failures:
            return self._failed_result(
                document_package=document_package,
                started_at=started_at,
                run_started=run_started,
                model_calls=model_calls,
                failures=object_failures,
                atomic_claims=atomic_claims,
                rejected_atomic_claims=rejected_atomic_claims,
                object_plans=object_plans,
                rejected_object_plans=rejected_object_plans,
            )

        validation_started = perf_counter()
        (
            accepted_candidates,
            rejected_candidates,
            rejected_auxiliary_items,
            normalizations,
            weak_signals,
            unresolved_items,
            coverage,
        ) = self._validate_object_results(
            object_results, planning_weak_inputs, planning_unresolved_inputs
        )

        completion = _last_completion(object_results, planning_result, discovery_results)
        validation_duration_ms = round((perf_counter() - validation_started) * 1000)
        finished_at = datetime.now(UTC)
        return IdentificationResult(
            document_package_id=document_package.document_package_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=round((perf_counter() - run_started) * 1000),
            provider=completion.provider if completion else self.provider,
            model=completion.model if completion else self.model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            catalog_version=CATALOG_VERSION,
            catalog_fingerprint=CATALOG_FINGERPRINT,
            raw_model_output=json.dumps(
                [
                    {
                        "purpose": call.purpose,
                        "segment": call.segment,
                        "rawOutput": call.raw_output,
                    }
                    for call in model_calls
                    if call.status == "completed"
                ],
                ensure_ascii=False,
            ),
            model_calls=model_calls,
            processing_stages=[
                ProcessingStage(
                    key="claim_discovery",
                    name="原子主张发现",
                    status="completed",
                    duration_ms=sum(
                        call.duration_ms
                        for call in model_calls
                        if call.purpose == "claim_discovery"
                    ),
                    detail=(
                        f"{len(segments)} 个结构分段发现 {len(atomic_claims)} 条可核验主张，"
                        f"拒绝 {len(rejected_atomic_claims)} 条证据不成立主张；"
                        f"发现分段上限 {self.claim_discovery_max_chars} 字符"
                    ),
                ),
                ProcessingStage(
                    key="claim_evidence_validation",
                    name="原文引句校验",
                    status="completed",
                    duration_ms=0,
                    detail="逐条校验来源锚点、列选择器与逐字引句，并回填完整来源字段",
                ),
                ProcessingStage(
                    key="object_planning",
                    name="全局对象边界规划",
                    status="completed",
                    duration_ms=sum(
                        call.duration_ms
                        for call in model_calls
                        if call.purpose == "object_planning"
                    ),
                    detail=(
                        f"一次比较全部 {len(atomic_claims)} 条主张，形成 "
                        f"{len(object_plans)} 个对象计划，拒绝 {len(rejected_object_plans)} 个；"
                        "未按主张类型或22个模块割裂对象边界"
                    ),
                ),
                ProcessingStage(
                    key="object_plan_validation",
                    name="对象身份与主张覆盖校验",
                    status="completed",
                    duration_ms=planning_validation_duration_ms,
                    detail="校验分类、身份要素、重复对象和主张引用；未覆盖主张显式进入未决项",
                ),
                ProcessingStage(
                    key="content_realization",
                    name="完整知识内容编制",
                    status="completed",
                    duration_ms=sum(
                        call.duration_ms
                        for call in model_calls
                        if call.purpose == "content_realization"
                    ),
                    detail=(
                        f"按对象计划分成 {len(realization_groups)} 个内容批次；"
                        "边界与分类由全局计划锁定，内容批次不可改写"
                    ),
                ),
                ProcessingStage(
                    key="contract_validation",
                    name="对象合同校验",
                    status="completed",
                    duration_ms=max(validation_duration_ms, 0),
                    detail=(
                        f"接受 {len(accepted_candidates)} 项，"
                        f"拒绝 {len(rejected_candidates)} 项"
                    ),
                ),
                ProcessingStage(
                    key="evidence_validation",
                    name="主张到对象追溯校验",
                    status="completed",
                    duration_ms=0,
                    detail="候选来源锚点由 sourceClaimIds 程序推导，长原文由已核验主张回填",
                ),
            ],
            atomic_claims=atomic_claims,
            rejected_atomic_claims=rejected_atomic_claims,
            object_plans=object_plans,
            rejected_object_plans=rejected_object_plans,
            candidates=accepted_candidates,
            rejected_candidates=rejected_candidates,
            rejected_auxiliary_items=rejected_auxiliary_items,
            normalizations=normalizations,
            weak_signals=weak_signals,
            unresolved_items=unresolved_items,
            coverage_by_module=coverage,
            call_count=len(model_calls),
            prompt_tokens=sum(call.prompt_tokens for call in model_calls),
            completion_tokens=sum(call.completion_tokens for call in model_calls),
            model_configuration=self.model_configuration,
        )

    def _discover_claims(
        self,
        document_package: DocumentPackage,
        segments: list[DocumentSegment],
    ) -> tuple[
        list[tuple[DocumentSegment, dict[str, Any], ModelCompletion, list[ModelCallTrace]]],
        list[str],
    ]:
        def run(
            segment: DocumentSegment,
        ) -> tuple[DocumentSegment, dict[str, Any], ModelCompletion, list[ModelCallTrace]]:
            segment_package = document_package.model_copy(
                update={"full_markdown": segment.markdown, "anchors": segment.anchors}
            )
            request = build_claim_discovery_request(
                segment_package, segment.label if len(segments) > 1 else None
            )
            payload, completion, calls = self._complete_json_request(
                request, "claim_discovery"
            )
            return segment, payload, completion, calls

        return _run_parallel(
            items=segments,
            worker=run,
            label=lambda segment: segment.label,
            max_concurrency=self.max_concurrency,
        )

    def _plan_candidate_objects(
        self,
        document_package_id: str,
        claims: list[AtomicClaim],
    ) -> tuple[
        list[tuple[str, list[AtomicClaim], dict[str, Any], ModelCompletion, list[ModelCallTrace]]],
        list[str],
    ]:
        if not claims:
            return [], []
        request = build_object_planning_request(
            document_package_id, claims, len(claims)
        )
        try:
            payload, completion, calls = self._complete_json_request(
                request, "object_planning"
            )
        except SegmentIdentificationFailure as error:
            return [("global", claims, {}, None, error.model_calls)], [
                f"global: {error}"
            ]
        return [("global", claims, payload, completion, calls)], []

    def _realize_candidate_objects(
        self,
        document_package_id: str,
        groups: list[tuple[str, list[CandidateObjectPlan], list[AtomicClaim]]],
    ) -> tuple[list[Any], list[str]]:
        if not groups:
            return [], []

        def run(item: tuple[str, list[CandidateObjectPlan], list[AtomicClaim]]) -> tuple[Any, ...]:
            label, plans, claims = item
            request = build_content_realization_request(
                document_package_id, plans, claims, label
            )
            payload, completion, calls = self._complete_json_request(
                request, "content_realization"
            )
            return label, plans, claims, payload, completion, calls

        return _run_parallel(
            items=groups,
            worker=run,
            label=lambda item: item[0],
            max_concurrency=self.max_concurrency,
        )

    def _repair_uncovered_object_plans(
        self,
        document_package_id: str,
        existing_plans: list[CandidateObjectPlan],
        uncovered_claims: list[AtomicClaim],
    ) -> tuple[list[Any], list[str]]:
        request = build_plan_coverage_repair_request(
            document_package_id, existing_plans, uncovered_claims
        )
        try:
            payload, completion, calls = self._complete_json_request(
                request, "object_planning"
            )
        except SegmentIdentificationFailure as error:
            return [("coverage-repair", uncovered_claims, {}, None, error.model_calls)], [
                f"coverage-repair: {error}"
            ]
        return [
            ("coverage-repair", uncovered_claims, payload, completion, calls)
        ], []

    def _complete_json_request(
        self,
        request: ModelRequest,
        purpose: CallPurpose,
    ) -> tuple[dict[str, Any], ModelCompletion, list[ModelCallTrace]]:
        completion, model_calls = self._complete_with_retries(
            request, purpose=purpose, first_attempt=1
        )
        try:
            payload = json.loads(completion.content)
        except json.JSONDecodeError as parse_error:
            if completion.finish_reason == "length":
                raise SegmentIdentificationFailure(
                    "model output was truncated; reduce the structural batch size",
                    model_calls,
                ) from parse_error
            repair_request = build_repair_request(
                request.document_package_id, completion.content, str(parse_error)
            )
            repaired, repair_calls = self._complete_with_retries(
                repair_request,
                purpose="repair",
                first_attempt=len(model_calls) + 1,
            )
            model_calls.extend(repair_calls)
            completion = repaired
            try:
                payload = json.loads(completion.content)
            except json.JSONDecodeError as final_error:
                raise SegmentIdentificationFailure(
                    "model output is not valid JSON after repair", model_calls
                ) from final_error
        if not isinstance(payload, dict):
            raise SegmentIdentificationFailure(
                "model output top level must be a JSON object", model_calls
            )
        return payload, completion, model_calls

    def _complete_with_retries(
        self,
        request: ModelRequest,
        *,
        purpose: Literal[
            "claim_discovery", "object_planning", "content_realization", "repair"
        ],
        first_attempt: int,
    ) -> tuple[ModelCompletion, list[ModelCallTrace]]:
        traces: list[ModelCallTrace] = []
        for retry_index in range(self.max_retries + 1):
            attempt = first_attempt + retry_index
            started = perf_counter()
            try:
                completion = self.gateway.complete(request)
            except RuntimeError as error:
                traces.append(
                    ModelCallTrace(
                        attempt=attempt,
                        purpose=purpose,
                        status="failed",
                        duration_ms=round((perf_counter() - started) * 1000),
                        error=str(error),
                        system_prompt=request.system_prompt,
                        user_prompt=request.user_prompt,
                    )
                )
                if retry_index == self.max_retries:
                    raise SegmentIdentificationFailure(str(error), traces) from error
                continue
            traces.append(
                ModelCallTrace(
                    attempt=attempt,
                    purpose=purpose,
                    status="completed",
                    duration_ms=round((perf_counter() - started) * 1000),
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    raw_output=completion.content,
                    finish_reason=completion.finish_reason,
                )
            )
            return completion, traces
        raise RuntimeError("model call exhausted without a result")

    def _validate_object_results(
        self,
        object_results: list[Any],
        weak_signal_inputs: list[tuple[Any, set[str]]],
        unresolved_inputs: list[tuple[Any, set[str]]],
    ) -> tuple[
        list[CandidateKnowledgeObject],
        list[RejectedCandidate],
        list[RejectedAuxiliaryItem],
        list[CandidateNormalization],
        list[WeakSignal],
        list[UnresolvedItem],
        dict[str, Literal["hit", "weak_signal", "not_found", "unresolved"]],
    ]:
        raw_candidates: list[dict[str, Any]] = []
        candidate_claim_scopes: dict[str, dict[str, AtomicClaim]] = {}
        for _group_index, (_label, plans, claims, payload, _completion, _calls) in enumerate(
            object_results, start=1
        ):
            claim_by_id = {claim.claim_id: claim for claim in claims}
            plan_by_id = {plan.plan_id: plan for plan in plans}
            raw_realizations = payload.get("realizations")
            if raw_realizations is None:
                raw_realizations = payload.get("candidates", [])
            realized_plan_ids: set[str] = set()
            for index, realization in enumerate(raw_realizations, start=1):
                if not isinstance(realization, dict):
                    continue
                plan_id = realization.get("planId", realization.get("candidateId"))
                plan = plan_by_id.get(plan_id)
                if plan is None:
                    raw_candidates.append(
                        {"candidateId": f"INVALID-REALIZATION-{index}", "content": {}}
                    )
                    continue
                realized_plan_ids.add(plan.plan_id)
                candidate_payload = plan.model_dump(by_alias=True)
                candidate_payload["candidateId"] = plan.plan_id
                candidate_payload.pop("planId", None)
                candidate_payload["content"] = realization.get("content", {})
                if "claimUsage" in realization:
                    candidate_payload["claimUsage"] = realization.get("claimUsage")
                if "omittedClaims" in realization:
                    candidate_payload["omittedClaims"] = realization.get(
                        "omittedClaims"
                    )
                candidate_payload["entityMentions"] = realization.get("entityMentions", [])
                candidate_payload["relations"] = realization.get("relations", [])
                raw_candidates.append(candidate_payload)
                candidate_claim_scopes[plan.plan_id] = {
                    claim_id: claim_by_id[claim_id]
                    for claim_id in plan.source_claim_ids
                    if claim_id in claim_by_id
                }
            for plan in plans:
                if plan.plan_id in realized_plan_ids:
                    continue
                missing_payload = plan.model_dump(by_alias=True)
                missing_payload["candidateId"] = plan.plan_id
                missing_payload.pop("planId", None)
                missing_payload.update(
                    {"content": {}, "entityMentions": [], "relations": []}
                )
                raw_candidates.append(missing_payload)
                candidate_claim_scopes[plan.plan_id] = {
                    claim_id: claim_by_id[claim_id]
                    for claim_id in plan.source_claim_ids
                    if claim_id in claim_by_id
                }

        accepted: list[CandidateKnowledgeObject] = []
        rejected: list[RejectedCandidate] = []
        normalizations: list[CandidateNormalization] = []
        seen_candidate_ids: set[str] = set()
        seen_fingerprints: set[str] = set()
        content_unresolved_inputs: list[tuple[Any, set[str]]] = []
        usage_contract_candidate_ids: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates, start=1):
            candidate_id = _candidate_id(raw_candidate, index)
            candidate_payload = raw_candidate.copy()
            reasons: list[str] = []
            claim_by_id = candidate_claim_scopes.get(candidate_id, {})
            source_claim_ids = candidate_payload.get("sourceClaimIds")
            if not isinstance(source_claim_ids, list) or not source_claim_ids:
                reasons.append("source claim ids are required")
                source_claim_ids = []
            unknown_claim_ids = sorted(
                claim_id
                for claim_id in source_claim_ids
                if not isinstance(claim_id, str) or claim_id not in claim_by_id
            )
            if unknown_claim_ids:
                reasons.append("unknown source claim ids: " + ", ".join(unknown_claim_ids))
            source_claims = [
                claim_by_id[claim_id]
                for claim_id in source_claim_ids
                if isinstance(claim_id, str) and claim_id in claim_by_id
            ]
            resolved_content, macro_reasons = resolve_verbatim_claim_references(
                candidate_payload.get("content", {}), claim_by_id
            )
            candidate_payload["content"] = _normalize_content_shape(
                candidate_payload.get("module"),
                candidate_payload.get("objectType"),
                resolved_content,
            )
            reasons.extend(macro_reasons)
            raw_claim_usage = candidate_payload.get("claimUsage", [])
            if "claimUsage" in raw_candidate:
                usage_contract_candidate_ids.add(candidate_id)
            valid_claim_usage, usage_unresolved = _validate_content_claim_usage(
                raw_claim_usage,
                candidate_payload["content"],
                claim_by_id,
                candidate_id,
            )
            candidate_payload["claimUsage"] = [
                item.model_dump(by_alias=True) for item in valid_claim_usage
            ]
            used_claim_ids = list(
                dict.fromkeys(item.claim_id for item in valid_claim_usage)
            )
            content = candidate_payload["content"]
            if isinstance(content, dict) and "factReferences" in content and used_claim_ids:
                candidate_payload["content"] = {
                    **content,
                    "factReferences": used_claim_ids,
                }
            if "claimUsage" in raw_candidate:
                covered_content_paths = {
                    path for item in valid_claim_usage for path in item.content_paths
                }
                content_leaf_paths = _business_content_leaf_paths(
                    candidate_payload["content"]
                )
                unsupported_content_paths = sorted(
                    set(content_leaf_paths) - covered_content_paths
                )
                candidate_payload["contentLeafCount"] = len(content_leaf_paths)
                candidate_payload["attributedContentLeafCount"] = (
                    len(content_leaf_paths) - len(unsupported_content_paths)
                )
                candidate_payload["unattributedContentPaths"] = (
                    unsupported_content_paths
                )
            candidate_payload["plannedSourceClaimIds"] = source_claim_ids
            # 新合同明确输出 claimUsage 时，sourceClaimIds 收紧为真正进入正文的主张；
            # 旧运行与测试载荷没有该字段时保留兼容读取，但质量指标不会再按它计分。
            if "claimUsage" in raw_candidate:
                candidate_payload["sourceClaimIds"] = used_claim_ids
                source_claims = [claim_by_id[claim_id] for claim_id in used_claim_ids]
            candidate_payload["evidence"] = sorted(
                {
                    evidence.anchor_id
                    for claim in source_claims
                    for evidence in claim.evidence
                }
            )
            valid_anchors = set(candidate_payload["evidence"])
            content_unresolved_inputs.extend(usage_unresolved)
            content_unresolved_inputs.extend(
                _validate_omitted_claims(
                    candidate_payload.pop("omittedClaims", []),
                    claim_by_id,
                    used_claim_ids,
                    candidate_id,
                )
            )

            raw_mentions = candidate_payload.get("entityMentions", [])
            if isinstance(raw_mentions, list):
                structured_mentions = [
                    mention
                    for mention in raw_mentions
                    if isinstance(mention, dict)
                    and mention.get("sourceRef") in valid_anchors
                ]
                discarded_count = len(raw_mentions) - len(structured_mentions)
                if discarded_count:
                    candidate_payload["entityMentions"] = structured_mentions
                    normalizations.append(
                        CandidateNormalization(
                            candidate_id=candidate_id,
                            field="entity_mentions",
                            original_value=f"{discarded_count}个非结构化实体提及",
                            normalized_value="已移除，保留对象内容与证据",
                            reason=(
                                "实体提及缺少类型、引用角色或使用了非来源锚点；"
                                "不据此创建低质量正式实体，但保留已核验对象内容"
                            ),
                        )
                    )

            raw_relations = candidate_payload.get("relations", [])
            if isinstance(raw_relations, list):
                structured_relations = [
                    relation
                    for relation in raw_relations
                    if isinstance(relation, dict)
                    and set(relation) >= {
                        "relationKind",
                        "relationType",
                        "sourceRef",
                        "targetRef",
                        "evidence",
                    }
                ]
                discarded_count = len(raw_relations) - len(structured_relations)
                if discarded_count:
                    candidate_payload["relations"] = structured_relations
                    normalizations.append(
                        CandidateNormalization(
                            candidate_id=candidate_id,
                            field="relations",
                            original_value=f"{discarded_count}个非合同关系建议",
                            normalized_value="已隔离，保留对象内容与证据",
                            reason=(
                                "关系建议缺少关系种类、对象引用或证据；"
                                "辅助关系失败不应删除已通过内容合同的知识对象"
                            ),
                        )
                    )

            module_code = candidate_payload.get("module")
            if (
                candidate_payload.get("domain") == module_code
                and isinstance(module_code, str)
                and module_code in MODULE_BY_CODE
            ):
                normalized_domain = MODULE_BY_CODE[module_code].domain
                candidate_payload["domain"] = normalized_domain
                normalizations.append(
                    CandidateNormalization(
                        candidate_id=candidate_id,
                        field="domain",
                        original_value=module_code,
                        normalized_value=normalized_domain,
                        reason="模型将主域重复写为模块码，按已发布目录映射规范化",
                    )
                )
            try:
                candidate = CandidateKnowledgeObject.model_validate(candidate_payload)
            except ValidationError as error:
                reasons.extend(item["msg"] for item in error.errors())
                rejected.append(
                    RejectedCandidate(
                        candidate_id=candidate_id,
                        reasons=reasons,
                        raw_candidate=raw_candidate,
                    )
                )
                continue

            reasons.extend(
                validate_candidate_classification(
                    candidate.domain, candidate.module, candidate.object_type
                )
            )
            reasons.extend(
                validate_candidate_content(
                    candidate.module, candidate.object_type, candidate.content
                )
            )
            if candidate.module == "D4.1":
                verified_script_texts = {
                    evidence.source_text
                    for claim in source_claims
                    if claim.claim_kind == "script"
                    for evidence in claim.evidence
                }
                script = candidate.content.get("script")
                if (
                    not isinstance(script, str)
                    or script not in verified_script_texts
                ):
                    reasons.append(
                        "standard script must equal verified source text from a script claim"
                    )
            if not candidate.title.strip():
                reasons.append("candidate title is required")
            if not candidate.object_boundary.strip():
                reasons.append("candidate object boundary is required")
            if not candidate.classification_basis.strip():
                reasons.append("candidate classification basis is required")
            if not candidate.identity_hints:
                reasons.append("candidate identity hints are required")
            if candidate.candidate_id in seen_candidate_ids:
                reasons.append(f"duplicate candidate id: {candidate.candidate_id}")
            seen_candidate_ids.add(candidate.candidate_id)

            valid_anchors = set(candidate.evidence)
            referenced_anchors = {
                mention.source_ref for mention in candidate.entity_mentions
            }
            for relation in candidate.relations:
                referenced_anchors.update(relation.evidence)
            invalid_anchors = referenced_anchors - valid_anchors
            if invalid_anchors:
                reasons.append(
                    "evidence not backed by source claims: "
                    + ", ".join(sorted(invalid_anchors))
                )
            fingerprint = json.dumps(
                [candidate.module, candidate.object_type, candidate.content],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if fingerprint in seen_fingerprints:
                reasons.append("duplicate candidate content in the same document")
            seen_fingerprints.add(fingerprint)
            if reasons:
                rejected.append(
                    RejectedCandidate(
                        candidate_id=candidate.candidate_id,
                        reasons=reasons,
                        raw_candidate=raw_candidate,
                    )
                )
                continue
            accepted.append(candidate)

        accepted = _drop_dangling_relations(accepted, normalizations)
        used_content_claim_ids = {
            usage.claim_id for candidate in accepted for usage in candidate.claim_usage
        }
        accepted_by_id = {candidate.candidate_id: candidate for candidate in accepted}
        for candidate_id in usage_contract_candidate_ids:
            candidate = accepted_by_id.get(candidate_id)
            if candidate is None:
                continue
            for claim_id in candidate.planned_source_claim_ids:
                if claim_id in used_content_claim_ids:
                    continue
                claim = candidate_claim_scopes.get(candidate_id, {}).get(claim_id)
                if claim is None:
                    continue
                content_unresolved_inputs.append(
                    (
                        {
                            "claimId": claim_id,
                            "description": (
                                f"{claim_id}：已进入对象计划 {candidate_id}，"
                                "但未证明写入对象正文"
                            ),
                            "reason": (
                                "内容编制结果未提供指向真实正文值的 claimUsage；"
                                "不能仅凭计划 sourceClaimIds 计为已消费"
                            ),
                            "evidence": sorted(
                                {item.anchor_id for item in claim.evidence}
                            ),
                            "module": candidate.module,
                        },
                        {item.anchor_id for item in claim.evidence},
                    )
                )
        unresolved_inputs = [
            item
            for item in unresolved_inputs
            if not (
                isinstance(item[0], dict)
                and item[0].get("claimId") in used_content_claim_ids
            )
        ]
        deduplicated_content_unresolved: list[tuple[Any, set[str]]] = []
        seen_content_unresolved_claim_ids: set[str] = set()
        for item in content_unresolved_inputs:
            claim_id = item[0].get("claimId") if isinstance(item[0], dict) else None
            if isinstance(claim_id, str):
                if claim_id in seen_content_unresolved_claim_ids:
                    continue
                seen_content_unresolved_claim_ids.add(claim_id)
            deduplicated_content_unresolved.append(item)
        unresolved_inputs = [
            item
            for item in unresolved_inputs
            if not (
                isinstance(item[0], dict)
                and item[0].get("claimId") in seen_content_unresolved_claim_ids
            )
        ]
        unresolved_inputs.extend(
            item
            for item in deduplicated_content_unresolved
            if not (
                isinstance(item[0], dict)
                and item[0].get("claimId") in used_content_claim_ids
            )
        )
        coverage: dict[
            str, Literal["hit", "weak_signal", "not_found", "unresolved"]
        ] = {module.code: "not_found" for module in KNOWLEDGE_MODULES}
        for candidate in accepted:
            coverage[candidate.module] = "hit"

        weak_signals, unresolved_items, rejected_auxiliary_items = (
            _validate_auxiliary_items(weak_signal_inputs, unresolved_inputs, coverage)
        )
        return (
            accepted,
            rejected,
            rejected_auxiliary_items,
            normalizations,
            weak_signals,
            unresolved_items,
            coverage,
        )

    def _failed_result(
        self,
        *,
        document_package: DocumentPackage,
        started_at: datetime,
        run_started: float,
        model_calls: list[ModelCallTrace],
        failures: list[str],
        atomic_claims: list[AtomicClaim] | None = None,
        rejected_atomic_claims: list[RejectedAtomicClaim] | None = None,
        object_plans: list[CandidateObjectPlan] | None = None,
        rejected_object_plans: list[RejectedObjectPlan] | None = None,
    ) -> IdentificationResult:
        return IdentificationResult(
            document_package_id=document_package.document_package_id,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_ms=round((perf_counter() - run_started) * 1000),
            provider=self.provider,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            catalog_version=CATALOG_VERSION,
            catalog_fingerprint=CATALOG_FINGERPRINT,
            raw_model_output="",
            model_calls=model_calls,
            processing_stages=[
                ProcessingStage(
                    key="model_call",
                    name="两阶段模型调用",
                    status="failed",
                    duration_ms=sum(call.duration_ms for call in model_calls),
                    detail="；".join(failures),
                )
            ],
            atomic_claims=atomic_claims or [],
            rejected_atomic_claims=rejected_atomic_claims or [],
            object_plans=object_plans or [],
            rejected_object_plans=rejected_object_plans or [],
            candidates=[],
            rejected_candidates=[],
            weak_signals=[],
            unresolved_items=[],
            coverage_by_module={
                module.code: "not_found" for module in KNOWLEDGE_MODULES
            },
            call_count=len(model_calls),
            prompt_tokens=sum(call.prompt_tokens for call in model_calls),
            completion_tokens=sum(call.completion_tokens for call in model_calls),
            model_configuration=self.model_configuration,
        )


def _run_parallel(
    *,
    items: list[Any],
    worker: Any,
    label: Any,
    max_concurrency: int,
) -> tuple[list[Any], list[str]]:
    results: list[Any] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(len(items), max_concurrency)) as executor:
        futures = [(item, executor.submit(worker, item)) for item in items]
        for item, future in futures:
            try:
                results.append(future.result())
            except SegmentIdentificationFailure as error:
                results.append((label(item), {}, None, error.model_calls))
                failures.append(f"{label(item)}: {error}")
    return results, failures


def _renumber_calls(results: list[Any], existing: list[ModelCallTrace]) -> list[ModelCallTrace]:
    calls: list[ModelCallTrace] = []
    for result in results:
        label = result[0].label if isinstance(result[0], DocumentSegment) else result[0]
        result_calls = result[-1]
        for call in result_calls:
            calls.append(
                call.model_copy(
                    update={
                        "attempt": len(existing) + len(calls) + 1,
                        "segment": label,
                    }
                )
            )
    return calls


def _last_completion(*result_sets: list[Any]) -> ModelCompletion | None:
    for results in result_sets:
        for result in reversed(results):
            completion = result[-2]
            if isinstance(completion, ModelCompletion):
                return completion
    return None


def _validate_object_plans(
    planning_results: list[Any],
    claims: list[AtomicClaim],
) -> tuple[
    list[CandidateObjectPlan],
    list[RejectedObjectPlan],
    list[tuple[Any, set[str]]],
    list[tuple[Any, set[str]]],
]:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    valid_anchors = {
        evidence.anchor_id for claim in claims for evidence in claim.evidence
    }
    payload = planning_results[0][2] if planning_results else {}
    accepted: list[CandidateObjectPlan] = []
    rejected: list[RejectedObjectPlan] = []
    seen_plan_ids: set[str] = set()
    seen_identities: set[str] = set()
    covered_claim_ids: set[str] = set()

    raw_plans = payload.get("objectPlans")
    if raw_plans is None:
        raw_plans = payload.get("candidates", [])
    for index, raw_plan in enumerate(raw_plans, start=1):
        raw_plan = _expand_compact_object_plan(raw_plan)
        plan_id = (
            raw_plan.get("planId", f"INVALID-{index}")
            if isinstance(raw_plan, dict)
            else f"INVALID-{index}"
        )
        if not isinstance(raw_plan, dict):
            rejected.append(
                RejectedObjectPlan(
                    plan_id=plan_id,
                    reasons=["object plan must be a JSON object"],
                    raw_plan={"value": raw_plan},
                )
            )
            continue
        candidate_payload = raw_plan.copy()
        if "planId" not in candidate_payload and "candidateId" in candidate_payload:
            candidate_payload["planId"] = candidate_payload.pop("candidateId")
        for field in ("content", "entityMentions", "relations", "evidence"):
            candidate_payload.pop(field, None)
        module_code = candidate_payload.get("module")
        if isinstance(module_code, str) and module_code in MODULE_BY_CODE:
            candidate_payload["domain"] = MODULE_BY_CODE[module_code].domain
            identity_contract = IDENTITY_CONTRACT_BY_MODULE[module_code]
            content_contract = CONTENT_CONTRACT_BY_MODULE[module_code]
            candidate_payload["objectBoundary"] = (
                f"同一对象：{identity_contract.same_object_when} "
                f"必须拆分：{identity_contract.different_object_when}"
            )
            candidate_payload["classificationBasis"] = (
                f"纳入：{content_contract.inclusion} 排除：{content_contract.exclusion}"
            )
        reasons: list[str] = []
        try:
            plan = CandidateObjectPlan.model_validate(candidate_payload)
        except ValidationError as error:
            rejected.append(
                RejectedObjectPlan(
                    plan_id=str(plan_id),
                    reasons=[item["msg"] for item in error.errors()],
                    raw_plan=raw_plan,
                )
            )
            continue
        if plan.module == "D4.2" and plan.object_type == "CUSTOMER_OBJECTION":
            objection_anchors = {
                evidence.anchor_id
                for claim_id in plan.source_claim_ids
                if claim_id in claim_by_id
                for evidence in claim_by_id[claim_id].evidence
            }
            supporting_claim_ids = [
                claim.claim_id
                for claim in claims
                if claim.claim_kind in {"strategy", "script"}
                and objection_anchors
                & {evidence.anchor_id for evidence in claim.evidence}
            ]
            plan = plan.model_copy(
                update={
                    "source_claim_ids": list(
                        dict.fromkeys(
                            [*plan.source_claim_ids, *supporting_claim_ids]
                        )
                    )
                }
            )
        if plan.module == "D4.3" and plan.object_type == "QA_PAIR":
            qa_claim_ids = [
                claim.claim_id for claim in claims if claim.claim_kind == "qa"
            ]
            plan = plan.model_copy(
                update={
                    "source_claim_ids": list(
                        dict.fromkeys([*plan.source_claim_ids, *qa_claim_ids])
                    )
                }
            )
        if plan.module in {"D4.1", "D4.2", "D4.3"}:
            plan_anchors = {
                evidence.anchor_id
                for claim_id in plan.source_claim_ids
                if claim_id in claim_by_id
                for evidence in claim_by_id[claim_id].evidence
            }
            scope_claim_ids = [
                claim.claim_id
                for claim in claims
                if claim.claim_kind == "fact"
                and {"scope", "applicability", "applicableVersions"}
                & set(claim.attributes)
                and plan_anchors
                & {evidence.anchor_id for evidence in claim.evidence}
            ]
            plan = plan.model_copy(
                update={
                    "source_claim_ids": list(
                        dict.fromkeys(
                            [*plan.source_claim_ids, *scope_claim_ids]
                        )
                    )
                }
            )
        reasons.extend(
            validate_candidate_classification(plan.domain, plan.module, plan.object_type)
        )
        unknown_claim_ids = sorted(set(plan.source_claim_ids) - set(claim_by_id))
        if unknown_claim_ids:
            reasons.append("unknown source claim ids: " + ", ".join(unknown_claim_ids))
        plan_claims = [
            claim_by_id[claim_id]
            for claim_id in plan.source_claim_ids
            if claim_id in claim_by_id
        ]
        if plan.module == "D4.2" and plan.object_type == "CUSTOMER_OBJECTION":
            support_keys = {
                "rootConcern",
                "rootCause",
                "response",
                "responseContext",
                "resolution",
                "handling",
            }
            has_complete_objection_support = any(
                claim.claim_kind in {"strategy", "script"}
                or support_keys.intersection(claim.attributes)
                for claim in plan_claims
            )
            if not has_complete_objection_support:
                reasons.append(
                    "customer objection lacks source-backed root concern or response"
                )
        if plan.module == "D4.1" and plan.object_type == "STANDARD_SCRIPT":
            script_keys = {"script", "wording", "response", "verbatim"}
            has_source_script = any(
                claim.claim_kind == "script"
                or script_keys.intersection(claim.attributes)
                for claim in plan_claims
            )
            if not has_source_script:
                reasons.append("standard script source text is not provided")
        if plan.module == "D3.2":
            if not any(claim.claim_kind == "method" for claim in plan_claims):
                reasons.append(
                    "D3.2 requires an explicit source method with steps; "
                    "a product or audience strategy belongs to D3.3"
                )
        if plan.module == "D1.3" and plan.object_type == "BUSINESS_PROCESS":
            has_embedded_sequence = any(
                isinstance(claim.attributes.get(field), list)
                and len(claim.attributes[field]) >= 2
                for claim in plan_claims
                for field in ("steps", "actions", "rulesOrSteps")
            )
            has_sourced_sequence = (
                len(plan_claims) >= 2
                and any(claim.claim_kind == "process" for claim in plan_claims)
            ) or has_embedded_sequence or any(
                claim.attributes.get("answerType") == "process"
                or any(
                    evidence.exact_quote.count("→") >= 2
                    for evidence in claim.evidence
                )
                for claim in plan_claims
            )
            if not has_sourced_sequence:
                reasons.append(
                    "business process requires a source-backed multi-step sequence"
                )
        if plan.module == "D1.3" and plan.object_type == "POLICY_RULE_SET":
            statements = "\n".join(claim.statement for claim in plan_claims)
            has_advisory_language = any(
                marker in statements for marker in ("可能", "建议", "等待", "重试")
            )
            has_formal_constraint = any(
                marker in statements
                for marker in ("规定", "必须", "不得", "不能", "无法", "允许", "采用")
            )
            if has_advisory_language and not has_formal_constraint:
                reasons.append(
                    "advisory handling guidance is not a formal D1.3 policy rule"
                )
        if (
            plan.module == "D3.3"
            and plan_claims
            and all(claim.claim_kind == "list" for claim in plan_claims)
        ):
            reasons.append(
                "category enumeration cannot become an action rule without source actions"
            )
        if not plan.title.strip():
            reasons.append("object plan title is required")
        if not plan.object_boundary.strip():
            reasons.append("object plan boundary is required")
        if not plan.classification_basis.strip():
            reasons.append("object plan classification basis is required")
        if not plan.identity_hints:
            reasons.append("object plan identity hints are required")
        else:
            reasons.extend(validate_identity_hints(plan.module, plan.identity_hints))
        if plan.plan_id in seen_plan_ids:
            reasons.append(f"duplicate object plan id: {plan.plan_id}")
        seen_plan_ids.add(plan.plan_id)
        if plan.module in MODULE_BY_CODE and not validate_identity_hints(
            plan.module, plan.identity_hints
        ):
            identity_fingerprint = json.dumps(
                [
                    plan.module,
                    plan.object_type,
                    canonical_identity(plan.module, plan.identity_hints),
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).casefold()
            if identity_fingerprint in seen_identities:
                reasons.append("duplicate object identity in the same document")
            seen_identities.add(identity_fingerprint)
        if reasons:
            rejected.append(
                RejectedObjectPlan(
                    plan_id=plan.plan_id,
                    reasons=reasons,
                    raw_plan=raw_plan,
                )
            )
            continue
        accepted.append(plan)
        covered_claim_ids.update(
            claim_id
            for claim_id in plan.source_claim_ids
            if claim_id in claim_by_id
            and _plan_satisfies_primary_claim_role(plan, claim_by_id[claim_id])
        )

    weak_inputs = [(item, valid_anchors) for item in payload.get("weakSignals", [])]
    unresolved_inputs = [
        (item, valid_anchors) for item in payload.get("unresolvedItems", [])
    ]
    explicitly_accounted_claim_ids = {
        item.get("claimId")
        for item in [
            *payload.get("weakSignals", []),
            *payload.get("unresolvedItems", []),
        ]
        if isinstance(item, dict) and item.get("claimId") in claim_by_id
    }
    for claim_id in sorted(
        set(claim_by_id) - covered_claim_ids - explicitly_accounted_claim_ids
    ):
        claim = claim_by_id[claim_id]
        unresolved_inputs.append(
            (
                {
                    "claimId": claim_id,
                    "description": f"{claim_id}：对象规划未覆盖的原子主张",
                    "reason": "模型未将该主张分配给任何对象计划，禁止静默丢失",
                    "evidence": sorted(
                        {evidence.anchor_id for evidence in claim.evidence}
                    ),
                    # moduleHints 是发现阶段的模型提示，不是已完成的分类裁决。
                    # 未形成对象时不能把它升级为未决项的正式模块归属。
                    "module": None,
                },
                valid_anchors,
            )
        )
    return accepted, rejected, weak_inputs, unresolved_inputs


def _expand_compact_object_plan(raw_plan: Any) -> Any:
    if not isinstance(raw_plan, list) or len(raw_plan) != 6:
        return raw_plan
    return dict(
        zip(
            (
                "planId",
                "title",
                "module",
                "objectType",
                "identityHints",
                "sourceClaimIds",
            ),
            raw_plan,
            strict=True,
        )
    )


def _plan_satisfies_primary_claim_role(
    plan: CandidateObjectPlan, claim: AtomicClaim
) -> bool:
    if (
        claim.claim_kind == "fact"
        and {"scope", "applicability", "applicableVersions"}
        & set(claim.attributes)
        and plan.module in {"D4.1", "D4.2", "D4.3"}
    ):
        return True
    prefixes = PRIMARY_MODULE_PREFIXES_BY_CLAIM_KIND.get(claim.claim_kind, ())
    return any(plan.module == prefix or plan.module.startswith(prefix) for prefix in prefixes)


def _normalize_content_shape(
    module: Any, object_type: Any, content: Any
) -> Any:
    if module != "D4.3" or object_type != "QA_PAIR" or not isinstance(content, dict):
        return content
    items = content.get("items")
    if not isinstance(items, list):
        return content
    normalized_items: list[Any] = []
    for item in items:
        if not isinstance(item, dict) or item.get("claimRef") not in (None, ""):
            normalized_items.append(item)
            continue
        fact_references = item.get("factReferences")
        if isinstance(fact_references, list) and len(fact_references) == 1:
            normalized_items.append(
                {**item, "claimRef": fact_references[0]}
            )
        else:
            normalized_items.append(item)
    return {**content, "items": normalized_items}


_CONTENT_PATH_PATTERN = re.compile(
    r"^\$(?:(?:\.[A-Za-z_][A-Za-z0-9_-]*)|(?:\[[0-9]+\]))+$"
)
_NON_CONTENT_REFERENCE_KEYS = {
    "claimRef",
    "evidenceRefs",
    "factReferences",
    "sourceClaimIds",
    "sourceClaimId",
    "evidenceRef",
    "elementId",
    "stepId",
    "stepOrder",
    "category",
    "type",
}


def _validate_content_claim_usage(
    raw_usage: Any,
    content: dict[str, Any],
    claim_by_id: dict[str, AtomicClaim],
    candidate_id: str,
) -> tuple[list[ContentClaimUsage], list[tuple[Any, set[str]]]]:
    if not isinstance(raw_usage, list):
        return [], []
    accepted: list[ContentClaimUsage] = []
    unresolved: list[tuple[Any, set[str]]] = []
    seen_claim_paths: set[tuple[str, tuple[str, ...]]] = set()
    for raw_item in raw_usage:
        try:
            usage = ContentClaimUsage.model_validate(raw_item)
        except ValidationError:
            continue
        claim = claim_by_id.get(usage.claim_id)
        expanded_paths: list[str] = []
        invalid_paths: list[str] = []
        for path in usage.content_paths:
            if path.startswith("$.content."):
                path = "$." + path.removeprefix("$.content.")
            expanded = _expand_content_path_to_leaf_paths(content, path)
            if not expanded:
                invalid_paths.append(path)
            else:
                expanded_paths.extend(expanded)
        if not invalid_paths:
            usage = usage.model_copy(
                update={"content_paths": list(dict.fromkeys(expanded_paths))}
            )
        key = (usage.claim_id, tuple(usage.content_paths))
        if claim is not None and not invalid_paths and key not in seen_claim_paths:
            accepted.append(usage)
            seen_claim_paths.add(key)
            continue
        if claim is None:
            continue
        anchors = {item.anchor_id for item in claim.evidence}
        unresolved.append(
            (
                {
                    "claimId": usage.claim_id,
                    "description": (
                        f"{usage.claim_id}：对象 {candidate_id} 的正文消费声明无效"
                    ),
                    "reason": (
                        "claimUsage 未指向真实、非空且承载业务知识的 content 路径："
                        + "、".join(invalid_paths or usage.content_paths)
                    ),
                    "evidence": sorted(anchors),
                    "module": None,
                },
                anchors,
            )
        )
    return accepted, unresolved


def _expand_content_path_to_leaf_paths(
    content: dict[str, Any], path: str
) -> list[str]:
    if not isinstance(path, str) or not _CONTENT_PATH_PATTERN.fullmatch(path):
        return []
    current: Any = content
    tokens = re.findall(r"\.([A-Za-z_][A-Za-z0-9_-]*)|\[([0-9]+)\]", path)
    for field, index_text in tokens:
        if field:
            if field in _NON_CONTENT_REFERENCE_KEYS:
                return []
            if not isinstance(current, dict) or field not in current:
                return []
            current = current[field]
        else:
            index = int(index_text)
            if not isinstance(current, list) or index >= len(current):
                return []
            current = current[index]
    paths: list[str] = []
    _collect_business_leaf_paths(current, path, paths)
    return paths


def _business_content_leaf_paths(content: dict[str, Any]) -> list[str]:
    leaf_paths: list[str] = []
    _collect_business_leaf_paths(content, "$", leaf_paths)
    return sorted(leaf_paths)


def _collect_business_leaf_paths(value: Any, path: str, output: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _NON_CONTENT_REFERENCE_KEYS:
                continue
            _collect_business_leaf_paths(item, f"{path}.{key}", output)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_business_leaf_paths(item, f"{path}[{index}]", output)
        return
    if value not in (None, ""):
        output.append(path)


def _validate_omitted_claims(
    raw_omissions: Any,
    claim_by_id: dict[str, AtomicClaim],
    used_claim_ids: list[str],
    candidate_id: str,
) -> list[tuple[Any, set[str]]]:
    if not isinstance(raw_omissions, list):
        return []
    unresolved: list[tuple[Any, set[str]]] = []
    for raw_item in raw_omissions:
        if not isinstance(raw_item, dict):
            continue
        claim_id = raw_item.get("claimId")
        reason = raw_item.get("reason")
        claim = claim_by_id.get(claim_id) if isinstance(claim_id, str) else None
        if (
            claim is None
            or claim_id in used_claim_ids
            or not isinstance(reason, str)
            or len(reason.strip()) < 2
        ):
            continue
        anchors = {item.anchor_id for item in claim.evidence}
        unresolved.append(
            (
                {
                    "claimId": claim_id,
                    "description": f"{claim_id}：对象 {candidate_id} 明确未写入正文",
                    "reason": reason.strip(),
                    "evidence": sorted(anchors),
                    "module": None,
                },
                anchors,
            )
        )
    return unresolved


def _automatic_uncovered_claim_ids(
    unresolved_inputs: list[tuple[Any, set[str]]],
) -> set[str]:
    return {
        item.get("claimId")
        for item, _valid_anchors in unresolved_inputs
        if isinstance(item, dict)
        and item.get("reason")
        == "模型未将该主张分配给任何对象计划，禁止静默丢失"
        and isinstance(item.get("claimId"), str)
    }


def _apply_plan_augmentations(
    plans: list[CandidateObjectPlan],
    raw_augmentations: Any,
    repair_claims: list[AtomicClaim],
) -> tuple[list[CandidateObjectPlan], list[RejectedObjectPlan]]:
    if not isinstance(raw_augmentations, list):
        return plans, [
            RejectedObjectPlan(
                plan_id="AUGMENTATIONS",
                reasons=["plan augmentations must be a list"],
                raw_plan={"value": raw_augmentations},
            )
        ]
    plan_by_id = {plan.plan_id: plan for plan in plans}
    valid_claim_ids = {claim.claim_id for claim in repair_claims}
    rejected: list[RejectedObjectPlan] = []
    for index, augmentation in enumerate(raw_augmentations, start=1):
        if not isinstance(augmentation, dict):
            rejected.append(
                RejectedObjectPlan(
                    plan_id=f"AUGMENT-{index}",
                    reasons=["plan augmentation must be a JSON object"],
                    raw_plan={"value": augmentation},
                )
            )
            continue
        plan_id = augmentation.get("planId")
        claim_ids = augmentation.get("sourceClaimIds")
        reasons: list[str] = []
        if plan_id not in plan_by_id:
            reasons.append(f"unknown augmentation plan id: {plan_id}")
        if not isinstance(claim_ids, list) or not claim_ids:
            reasons.append("augmentation sourceClaimIds must be a non-empty list")
            claim_ids = []
        unknown_claim_ids = sorted(
            claim_id for claim_id in claim_ids if claim_id not in valid_claim_ids
        )
        if unknown_claim_ids:
            reasons.append(
                "unknown augmentation claim ids: " + ", ".join(unknown_claim_ids)
            )
        if reasons:
            rejected.append(
                RejectedObjectPlan(
                    plan_id=str(plan_id or f"AUGMENT-{index}"),
                    reasons=reasons,
                    raw_plan=augmentation,
                )
            )
            continue
        plan = plan_by_id[plan_id]
        plan_by_id[plan_id] = plan.model_copy(
            update={
                "source_claim_ids": list(
                    dict.fromkeys([*plan.source_claim_ids, *claim_ids])
                )
            }
        )
    return [plan_by_id[plan.plan_id] for plan in plans], rejected


def _merge_repair_plans(
    existing: list[CandidateObjectPlan],
    repairs: list[CandidateObjectPlan],
) -> tuple[list[CandidateObjectPlan], list[RejectedObjectPlan]]:
    merged = list(existing)
    seen_ids = {plan.plan_id for plan in existing}
    seen_identities = {
        (
            plan.module,
            plan.object_type,
            json.dumps(
                canonical_identity(plan.module, plan.identity_hints),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).casefold(),
        )
        for plan in existing
    }
    rejected: list[RejectedObjectPlan] = []
    for plan in repairs:
        reasons: list[str] = []
        identity = (
            plan.module,
            plan.object_type,
            json.dumps(
                canonical_identity(plan.module, plan.identity_hints),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).casefold(),
        )
        if plan.plan_id in seen_ids:
            reasons.append(f"duplicate repair plan id: {plan.plan_id}")
        if identity in seen_identities:
            reasons.append("repair plan duplicates an existing object identity")
        if reasons:
            rejected.append(
                RejectedObjectPlan(
                    plan_id=plan.plan_id,
                    reasons=reasons,
                    raw_plan=plan.model_dump(by_alias=True),
                )
            )
            continue
        merged.append(plan)
        seen_ids.add(plan.plan_id)
        seen_identities.add(identity)
    return merged, rejected


def _enforce_plan_granularity(
    plans: list[CandidateObjectPlan], claims: list[AtomicClaim]
) -> list[CandidateObjectPlan]:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    qa_plans = [
        plan
        for plan in plans
        if plan.module == "D4.3" and plan.object_type == "QA_PAIR"
    ]
    if len(qa_plans) > 1:
        primary_qa_plan = qa_plans[0].model_copy(
            update={
                "source_claim_ids": list(
                    dict.fromkeys(
                        claim_id
                        for plan in qa_plans
                        for claim_id in plan.source_claim_ids
                    )
                )
            }
        )
        plans = [
            primary_qa_plan
            if plan.plan_id == qa_plans[0].plan_id
            else plan
            for plan in plans
            if plan not in qa_plans[1:]
        ]
    enforced: list[CandidateObjectPlan] = []
    for plan in plans:
        plan_claims = [
            claim_by_id[claim_id]
            for claim_id in plan.source_claim_ids
            if claim_id in claim_by_id
        ]
        if plan.module == "D4.2" and plan.object_type == "CUSTOMER_OBJECTION":
            objections = [claim for claim in plan_claims if claim.claim_kind == "objection"]
            if len(objections) > 1:
                objection_context = plan.identity_hints.get(
                    "context", "来源资料适用场景"
                )
                enforced.extend(
                    _split_plan_by_anchor(
                        plan,
                        plan_claims,
                        objections,
                        title=lambda claim: f"客户异议：{claim.subject}",
                        identity=lambda claim, context=objection_context: {
                            "rootConcern": claim.subject,
                            "context": context,
                        },
                    )
                )
                continue
        if plan.module == "D3.3" and plan.object_type == "SALES_STRATEGY":
            combination_claims = [
                claim
                for claim in plan_claims
                if claim.claim_kind == "strategy"
                and claim.attributes.get("combination")
            ]
            combinations = {
                str(claim.attributes["combination"]) for claim in combination_claims
            }
            if len(combinations) > 1:
                trigger_context = plan.identity_hints.get(
                    "triggerContext", "组合营销场景"
                )
                enforced.extend(
                    _split_plan_by_anchor(
                        plan,
                        plan_claims,
                        combination_claims,
                        title=lambda claim: (
                            f"组合营销策略：{claim.attributes['combination']}"
                        ),
                        identity=lambda claim, context=trigger_context: {
                            "strategyGoal": (
                                f"组合营销-{claim.attributes['combination']}"
                            ),
                            "triggerContext": context,
                            "applicability": claim.attributes["combination"],
                        },
                    )
                )
                continue
        if plan.module == "D3.3" and plan.object_type == "DECISION_RULE":
            claims_by_anchor: dict[str, list[AtomicClaim]] = {}
            for claim in plan_claims:
                if not claim.evidence:
                    continue
                claims_by_anchor.setdefault(claim.evidence[0].anchor_id, []).append(
                    claim
                )
            if len(claims_by_anchor) > 1:
                for index, anchor_claims in enumerate(
                    claims_by_anchor.values(), start=1
                ):
                    trigger = anchor_claims[0].subject
                    identity_hints = dict(plan.identity_hints)
                    identity_hints["triggerContext"] = trigger
                    enforced.append(
                        plan.model_copy(
                            update={
                                "plan_id": f"{plan.plan_id}-{index}",
                                "title": f"{plan.title}：{trigger}",
                                "identity_hints": identity_hints,
                                "source_claim_ids": [
                                    claim.claim_id for claim in anchor_claims
                                ],
                            }
                        )
                    )
                continue
        enforced.append(plan)
    return enforced


def _split_plan_by_anchor(
    plan: CandidateObjectPlan,
    plan_claims: list[AtomicClaim],
    pivots: list[AtomicClaim],
    *,
    title: Any,
    identity: Any,
) -> list[CandidateObjectPlan]:
    results = []
    pivot_ids = {pivot.claim_id for pivot in pivots}
    for index, pivot in enumerate(pivots, start=1):
        anchors = {evidence.anchor_id for evidence in pivot.evidence}
        related_ids = [
            claim.claim_id
            for claim in plan_claims
            if claim.claim_id == pivot.claim_id
            or (
                claim.claim_id not in pivot_ids
                and anchors & {evidence.anchor_id for evidence in claim.evidence}
            )
        ]
        results.append(
            plan.model_copy(
                update={
                    "plan_id": f"{plan.plan_id}-{index}",
                    "title": title(pivot),
                    "identity_hints": identity(pivot),
                    "source_claim_ids": related_ids,
                }
            )
        )
    return results


def _group_object_plans(
    plans: list[CandidateObjectPlan],
    claims: list[AtomicClaim],
    batch_size: int,
) -> list[tuple[str, list[CandidateObjectPlan], list[AtomicClaim]]]:
    groups = []
    compatible_plans: dict[tuple[str, str], list[CandidateObjectPlan]] = {}
    for plan in plans:
        compatible_plans.setdefault((plan.module, plan.object_type), []).append(plan)
    group_index = 0
    for (module, object_type), same_contract_plans in compatible_plans.items():
        for offset in range(0, len(same_contract_plans), batch_size):
            group_index += 1
            batch_plans = same_contract_plans[offset : offset + batch_size]
            claim_ids = {
                claim_id
                for plan in batch_plans
                for claim_id in plan.source_claim_ids
            }
            groups.append(
                (
                    f"{module}-{object_type}-{group_index}",
                    batch_plans,
                    [claim for claim in claims if claim.claim_id in claim_ids],
                )
            )
    return groups


def _namespace_claim_payload(payload: dict[str, Any], namespace: str | None) -> dict[str, Any]:
    namespaced = json.loads(json.dumps(payload, ensure_ascii=False))
    if namespace is None:
        return namespaced
    for raw_claim in namespaced.get("claims", []):
        if isinstance(raw_claim, dict) and isinstance(raw_claim.get("claimId"), str):
            raw_claim["claimId"] = f"{namespace}-{raw_claim['claimId']}"
    return namespaced


def _namespace_candidate_payload(payload: dict[str, Any], namespace: str) -> dict[str, Any]:
    namespaced = json.loads(json.dumps(payload, ensure_ascii=False))
    id_mapping: dict[str, str] = {}
    for raw_candidate in namespaced.get("candidates", []):
        if not isinstance(raw_candidate, dict):
            continue
        candidate_id = raw_candidate.get("candidateId")
        if isinstance(candidate_id, str):
            id_mapping[candidate_id] = f"{namespace}-{candidate_id}"
        for mention in raw_candidate.get("entityMentions", []):
            if isinstance(mention, dict) and isinstance(mention.get("mentionId"), str):
                id_mapping[mention["mentionId"]] = f"{namespace}-{mention['mentionId']}"
    for raw_candidate in namespaced.get("candidates", []):
        if not isinstance(raw_candidate, dict):
            continue
        for field in ("candidateId",):
            if raw_candidate.get(field) in id_mapping:
                raw_candidate[field] = id_mapping[raw_candidate[field]]
        for mention in raw_candidate.get("entityMentions", []):
            if isinstance(mention, dict) and mention.get("mentionId") in id_mapping:
                mention["mentionId"] = id_mapping[mention["mentionId"]]
        for relation in raw_candidate.get("relations", []):
            if not isinstance(relation, dict):
                continue
            for field in ("sourceRef", "targetRef"):
                if relation.get(field) in id_mapping:
                    relation[field] = id_mapping[relation[field]]
    return namespaced


def _candidate_id(raw_candidate: Any, index: int) -> str:
    if isinstance(raw_candidate, dict) and isinstance(raw_candidate.get("candidateId"), str):
        return raw_candidate["candidateId"]
    return f"INVALID-{index}"


def _drop_dangling_relations(
    candidates: list[CandidateKnowledgeObject],
    normalizations: list[CandidateNormalization],
) -> list[CandidateKnowledgeObject]:
    valid_refs = {
        reference
        for candidate in candidates
        for reference in (
            candidate.candidate_id,
            *(mention.mention_id for mention in candidate.entity_mentions),
        )
    }
    normalized: list[CandidateKnowledgeObject] = []
    for candidate in candidates:
        valid_relations = [
            relation
            for relation in candidate.relations
            if relation.source_ref in valid_refs and relation.target_ref in valid_refs
        ]
        discarded_count = len(candidate.relations) - len(valid_relations)
        if discarded_count:
            normalizations.append(
                CandidateNormalization(
                    candidate_id=candidate.candidate_id,
                    field="relations",
                    original_value=f"{discarded_count}个悬空关系建议",
                    normalized_value="已隔离，保留对象内容与证据",
                    reason=(
                        "关系引用了未形成的对象或实体；辅助关系失败不应删除"
                        "已通过内容合同的知识对象"
                    ),
                )
            )
            candidate = candidate.model_copy(update={"relations": valid_relations})
        normalized.append(candidate)
    return normalized


def _validate_auxiliary_items(
    weak_signal_inputs: list[tuple[Any, set[str]]],
    unresolved_inputs: list[tuple[Any, set[str]]],
    coverage: dict[str, Literal["hit", "weak_signal", "not_found", "unresolved"]],
) -> tuple[list[WeakSignal], list[UnresolvedItem], list[RejectedAuxiliaryItem]]:
    weak_signals: list[WeakSignal] = []
    unresolved_items: list[UnresolvedItem] = []
    rejected: list[RejectedAuxiliaryItem] = []
    for raw_signal, valid_anchors in weak_signal_inputs:
        try:
            signal = WeakSignal.model_validate(raw_signal)
            reasons = []
            if signal.module not in MODULE_BY_CODE:
                reasons.append(f"unknown knowledge module: {signal.module}")
            invalid = set(signal.evidence) - valid_anchors
            if invalid:
                reasons.append("unknown evidence anchors: " + ", ".join(sorted(invalid)))
            if reasons:
                raise ValueError("; ".join(reasons))
        except (ValidationError, ValueError) as error:
            rejected.append(
                RejectedAuxiliaryItem(
                    kind="weak_signal",
                    reasons=_validation_reasons(error),
                    raw_item=_raw_item(raw_signal),
                )
            )
            continue
        weak_signals.append(signal)
        if coverage[signal.module] == "not_found":
            coverage[signal.module] = "weak_signal"

    for raw_item, valid_anchors in unresolved_inputs:
        try:
            item = UnresolvedItem.model_validate(raw_item)
            reasons = []
            if item.module is not None and item.module not in MODULE_BY_CODE:
                reasons.append(f"unknown knowledge module: {item.module}")
            invalid = set(item.evidence) - valid_anchors
            if invalid:
                reasons.append("unknown evidence anchors: " + ", ".join(sorted(invalid)))
            if reasons:
                raise ValueError("; ".join(reasons))
        except (ValidationError, ValueError) as error:
            rejected.append(
                RejectedAuxiliaryItem(
                    kind="unresolved_item",
                    reasons=_validation_reasons(error),
                    raw_item=_raw_item(raw_item),
                )
            )
            continue
        unresolved_items.append(item)
        if item.module in MODULE_BY_CODE and coverage[item.module] == "not_found":
            coverage[item.module] = "unresolved"
    return weak_signals, unresolved_items, rejected


def _validation_reasons(error: ValidationError | ValueError) -> list[str]:
    if isinstance(error, ValidationError):
        return [item["msg"] for item in error.errors()]
    return [str(error)]


def _raw_item(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}
