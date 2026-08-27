from __future__ import annotations

import json
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
from .claims import resolve_verbatim_claim_references, validate_atomic_claims
from .content_contracts import validate_candidate_content
from .identity_contracts import canonical_identity, validate_identity_hints
from .models import (
    AtomicClaim,
    CandidateKnowledgeObject,
    CandidateNormalization,
    CandidateObjectPlan,
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
    build_repair_request,
)
from .segmenter import DocumentSegment, segment_document

CallPurpose = Literal["claim_discovery", "object_planning", "content_realization"]
CONTENT_REALIZATION_BATCH_SIZE = 5


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
                candidate_payload["entityMentions"] = realization.get("entityMentions", [])
                candidate_payload["relations"] = realization.get("relations", [])
                raw_candidates.append(candidate_payload)
                candidate_claim_scopes[plan.plan_id] = claim_by_id
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
                candidate_claim_scopes[plan.plan_id] = claim_by_id

        accepted: list[CandidateKnowledgeObject] = []
        rejected: list[RejectedCandidate] = []
        normalizations: list[CandidateNormalization] = []
        seen_candidate_ids: set[str] = set()
        seen_fingerprints: set[str] = set()
        known_relation_refs = {
            reference
            for raw_candidate in raw_candidates
            for reference in _candidate_relation_refs(raw_candidate)
        }

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
            candidate_payload["evidence"] = sorted(
                {
                    evidence.anchor_id
                    for claim in source_claims
                    for evidence in claim.evidence
                }
            )
            resolved_content, macro_reasons = resolve_verbatim_claim_references(
                candidate_payload.get("content", {}), claim_by_id
            )
            candidate_payload["content"] = resolved_content
            reasons.extend(macro_reasons)

            raw_mentions = candidate_payload.get("entityMentions", [])
            if isinstance(raw_mentions, list):
                structured_mentions = [
                    mention for mention in raw_mentions if isinstance(mention, dict)
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
                                "模型只返回实体名称，缺少类型、引用角色和来源；"
                                "不据此创建低质量正式实体"
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
                for relation_ref in (relation.source_ref, relation.target_ref):
                    if relation_ref not in known_relation_refs:
                        reasons.append(f"unknown relation reference: {relation_ref}")
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

        accepted = _reject_dangling_relations(accepted, rejected)
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
        if (
            candidate_payload.get("domain") == module_code
            and isinstance(module_code, str)
            and module_code in MODULE_BY_CODE
        ):
            candidate_payload["domain"] = MODULE_BY_CODE[module_code].domain
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
        reasons.extend(
            validate_candidate_classification(plan.domain, plan.module, plan.object_type)
        )
        unknown_claim_ids = sorted(set(plan.source_claim_ids) - set(claim_by_id))
        if unknown_claim_ids:
            reasons.append("unknown source claim ids: " + ", ".join(unknown_claim_ids))
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
        covered_claim_ids.update(plan.source_claim_ids)

    weak_inputs = [(item, valid_anchors) for item in payload.get("weakSignals", [])]
    unresolved_inputs = [
        (item, valid_anchors) for item in payload.get("unresolvedItems", [])
    ]
    for claim_id in sorted(set(claim_by_id) - covered_claim_ids):
        claim = claim_by_id[claim_id]
        unresolved_inputs.append(
            (
                {
                    "description": f"{claim_id}：对象规划未覆盖的原子主张",
                    "reason": "模型未将该主张分配给任何对象计划，禁止静默丢失",
                    "evidence": sorted(
                        {evidence.anchor_id for evidence in claim.evidence}
                    ),
                    "module": claim.module_hints[0] if claim.module_hints else None,
                },
                valid_anchors,
            )
        )
    return accepted, rejected, weak_inputs, unresolved_inputs


def _group_object_plans(
    plans: list[CandidateObjectPlan],
    claims: list[AtomicClaim],
    batch_size: int,
) -> list[tuple[str, list[CandidateObjectPlan], list[AtomicClaim]]]:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    groups = []
    for offset in range(0, len(plans), batch_size):
        batch_plans = plans[offset : offset + batch_size]
        claim_ids = {
            claim_id for plan in batch_plans for claim_id in plan.source_claim_ids
        }
        batch_claims = [
            claim for claim_id, claim in claim_by_id.items() if claim_id in claim_ids
        ]
        groups.append(
            (
                f"objects-{offset // batch_size + 1}",
                batch_plans,
                batch_claims,
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


def _candidate_relation_refs(raw_candidate: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    candidate_id = raw_candidate.get("candidateId")
    if isinstance(candidate_id, str):
        refs.add(candidate_id)
    for mention in raw_candidate.get("entityMentions", []):
        if isinstance(mention, dict) and isinstance(mention.get("mentionId"), str):
            refs.add(mention["mentionId"])
    return refs


def _reject_dangling_relations(
    candidates: list[CandidateKnowledgeObject],
    rejected: list[RejectedCandidate],
) -> list[CandidateKnowledgeObject]:
    remaining = candidates
    while True:
        valid_refs = {
            reference
            for candidate in remaining
            for reference in (
                candidate.candidate_id,
                *(mention.mention_id for mention in candidate.entity_mentions),
            )
        }
        invalid_candidates: list[tuple[CandidateKnowledgeObject, list[str]]] = []
        for candidate in remaining:
            invalid_refs = {
                reference
                for relation in candidate.relations
                for reference in (relation.source_ref, relation.target_ref)
                if reference not in valid_refs
            }
            if invalid_refs:
                invalid_candidates.append(
                    (
                        candidate,
                        [
                            "relation references rejected or missing objects: "
                            + ", ".join(sorted(invalid_refs))
                        ],
                    )
                )
        if not invalid_candidates:
            return remaining
        invalid_ids = {candidate.candidate_id for candidate, _ in invalid_candidates}
        for candidate, reasons in invalid_candidates:
            rejected.append(
                RejectedCandidate(
                    candidate_id=candidate.candidate_id,
                    reasons=reasons,
                    raw_candidate=candidate.model_dump(by_alias=True),
                )
            )
        remaining = [
            candidate for candidate in remaining if candidate.candidate_id not in invalid_ids
        ]


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
