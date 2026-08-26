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
from .models import (
    CandidateKnowledgeObject,
    CandidateNormalization,
    DocumentPackage,
    IdentificationResult,
    ModelCallTrace,
    ModelCompletion,
    ModelConfigurationSnapshot,
    ModelRequest,
    ProcessingStage,
    RejectedAuxiliaryItem,
    RejectedCandidate,
    UnresolvedItem,
    WeakSignal,
)
from .prompt_builder import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_model_request,
    build_repair_request,
)
from .segmenter import DocumentSegment, segment_document


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
        self.max_concurrency = max_concurrency
        self.provider = provider
        self.model = model
        self.model_configuration = model_configuration

    def identify(self, document_package: DocumentPackage) -> IdentificationResult:
        if document_package.status != "available":
            raise DocumentPackageUnavailable(document_package.document_package_id)
        started_at = datetime.now(UTC)
        run_started = perf_counter()
        segments = segment_document(document_package, self.document_max_chars)
        segment_payloads: list[tuple[DocumentSegment, dict[str, Any]]] = []
        model_calls: list[ModelCallTrace] = []

        def identify_segment(
            segment: DocumentSegment,
        ) -> tuple[DocumentSegment, dict[str, Any], ModelCompletion, list[ModelCallTrace]]:
            segment_package = document_package.model_copy(
                update={"full_markdown": segment.markdown, "anchors": segment.anchors}
            )
            payload, segment_completion, segment_calls = self._identify_segment(
                segment_package,
                segment.label if len(segments) > 1 else None,
            )
            return segment, payload, segment_completion, segment_calls

        segment_results: list[
            tuple[DocumentSegment, dict[str, Any], ModelCompletion, list[ModelCallTrace]]
        ] = []
        failures: list[str] = []
        with ThreadPoolExecutor(
            max_workers=min(len(segments), self.max_concurrency)
        ) as executor:
            futures = [
                (segment, executor.submit(identify_segment, segment))
                for segment in segments
            ]
            for segment, future in futures:
                try:
                    segment_results.append(future.result())
                except SegmentIdentificationFailure as error:
                    failures.append(f"{segment.label}: {error}")
                    for call in error.model_calls:
                        model_calls.append(
                            call.model_copy(
                                update={
                                    "attempt": len(model_calls) + 1,
                                    "segment": segment.label,
                                }
                            )
                        )

        if failures:
            for segment, _payload, _completion, segment_calls in segment_results:
                for call in segment_calls:
                    model_calls.append(
                        call.model_copy(
                            update={
                                "attempt": len(model_calls) + 1,
                                "segment": segment.label,
                            }
                        )
                    )
            finished_at = datetime.now(UTC)
            return IdentificationResult(
                document_package_id=document_package.document_package_id,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
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
                        name="真实模型调用",
                        status="failed",
                        duration_ms=sum(call.duration_ms for call in model_calls),
                        detail="；".join(failures),
                    )
                ],
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

        for segment, payload, _completion, segment_calls in segment_results:
            if len(segments) > 1:
                payload = _namespace_payload(payload, f"S{segment.index}")
            for call in segment_calls:
                model_calls.append(
                    call.model_copy(
                        update={
                            "attempt": len(model_calls) + 1,
                            "segment": segment.label if len(segments) > 1 else None,
                        }
                    )
                )
            segment_payloads.append((segment, payload))

        if not segment_results:
            raise RuntimeError("document segmentation produced no model result")
        completion = segment_results[-1][2]
        payload = {
            "candidates": [
                candidate
                for _, segment_payload in segment_payloads
                for candidate in segment_payload.get("candidates", [])
            ],
            "weakSignals": [
                signal
                for _, segment_payload in segment_payloads
                for signal in segment_payload.get("weakSignals", [])
            ],
            "unresolvedItems": [
                item
                for _, segment_payload in segment_payloads
                for item in segment_payload.get("unresolvedItems", [])
            ],
        }
        candidate_anchor_scopes: dict[str, set[str]] = {}
        weak_signal_inputs: list[tuple[Any, set[str]]] = []
        unresolved_inputs: list[tuple[Any, set[str]]] = []
        for segment, segment_payload in segment_payloads:
            segment_anchor_ids = {anchor.anchor_id for anchor in segment.anchors}
            for index, raw_candidate in enumerate(
                segment_payload.get("candidates", []), start=1
            ):
                candidate_anchor_scopes[_candidate_id(raw_candidate, index)] = (
                    segment_anchor_ids
                )
            weak_signal_inputs.extend(
                (raw_signal, segment_anchor_ids)
                for raw_signal in segment_payload.get("weakSignals", [])
            )
            unresolved_inputs.extend(
                (raw_item, segment_anchor_ids)
                for raw_item in segment_payload.get("unresolvedItems", [])
            )
        validation_started = perf_counter()
        raw_candidates = payload.get("candidates", [])
        accepted: list[CandidateKnowledgeObject] = []
        rejected: list[RejectedCandidate] = []
        seen_candidate_ids: set[str] = set()
        seen_fingerprints: set[str] = set()
        normalizations: list[CandidateNormalization] = []
        known_relation_refs = {
            reference
            for raw_candidate in raw_candidates
            if isinstance(raw_candidate, dict)
            for reference in _candidate_relation_refs(raw_candidate)
        }

        for index, raw_candidate in enumerate(raw_candidates, start=1):
            candidate_id = _candidate_id(raw_candidate, index)
            if not isinstance(raw_candidate, dict):
                rejected.append(
                    RejectedCandidate(
                        candidate_id=candidate_id,
                        reasons=["candidate must be a JSON object"],
                        raw_candidate={"value": raw_candidate},
                    )
                )
                continue
            candidate_payload = raw_candidate.copy()
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
                rejected.append(
                    RejectedCandidate(
                        candidate_id=candidate_id,
                        reasons=[item["msg"] for item in error.errors()],
                        raw_candidate=raw_candidate,
                    )
                )
                continue

            reasons = validate_candidate_classification(
                candidate.domain, candidate.module, candidate.object_type
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
            referenced_anchors = set(candidate.evidence)
            referenced_anchors.update(mention.source_ref for mention in candidate.entity_mentions)
            for relation in candidate.relations:
                referenced_anchors.update(relation.evidence)
                for relation_ref in (relation.source_ref, relation.target_ref):
                    if relation_ref not in known_relation_refs:
                        reasons.append(f"unknown relation reference: {relation_ref}")
            invalid_anchors = referenced_anchors - candidate_anchor_scopes.get(
                candidate.candidate_id, set()
            )
            if invalid_anchors:
                reasons.append("unknown evidence anchors: " + ", ".join(sorted(invalid_anchors)))
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

        coverage = {module.code: "not_found" for module in KNOWLEDGE_MODULES}
        for candidate in accepted:
            coverage[candidate.module] = "hit"

        weak_signals: list[WeakSignal] = []
        rejected_auxiliary_items: list[RejectedAuxiliaryItem] = []
        for raw_signal, valid_anchors in weak_signal_inputs:
            try:
                signal = WeakSignal.model_validate(raw_signal)
            except ValidationError as error:
                rejected_auxiliary_items.append(
                    RejectedAuxiliaryItem(
                        kind="weak_signal",
                        reasons=[item["msg"] for item in error.errors()],
                        raw_item=_raw_item(raw_signal),
                    )
                )
                continue
            signal_reasons: list[str] = []
            if signal.module not in MODULE_BY_CODE:
                signal_reasons.append(f"unknown knowledge module: {signal.module}")
            invalid_anchors = set(signal.evidence) - valid_anchors
            if invalid_anchors:
                signal_reasons.append(
                    "unknown evidence anchors: " + ", ".join(sorted(invalid_anchors))
                )
            if signal_reasons:
                rejected_auxiliary_items.append(
                    RejectedAuxiliaryItem(
                        kind="weak_signal",
                        reasons=signal_reasons,
                        raw_item=_raw_item(raw_signal),
                    )
                )
                continue
            weak_signals.append(signal)
            if coverage[signal.module] == "not_found":
                coverage[signal.module] = "weak_signal"

        unresolved_items: list[UnresolvedItem] = []
        for raw_item, valid_anchors in unresolved_inputs:
            try:
                item = UnresolvedItem.model_validate(raw_item)
            except ValidationError as error:
                rejected_auxiliary_items.append(
                    RejectedAuxiliaryItem(
                        kind="unresolved_item",
                        reasons=[item["msg"] for item in error.errors()],
                        raw_item=_raw_item(raw_item),
                    )
                )
                continue
            invalid_anchors = set(item.evidence) - valid_anchors
            item_reasons: list[str] = []
            if item.module is not None and item.module not in MODULE_BY_CODE:
                item_reasons.append(f"unknown knowledge module: {item.module}")
            if invalid_anchors:
                item_reasons.append(
                    "unknown evidence anchors: " + ", ".join(sorted(invalid_anchors))
                )
            if item_reasons:
                rejected_auxiliary_items.append(
                    RejectedAuxiliaryItem(
                        kind="unresolved_item",
                        reasons=item_reasons,
                        raw_item=_raw_item(raw_item),
                    )
                )
                continue
            unresolved_items.append(item)
            if item.module in MODULE_BY_CODE and coverage[item.module] == "not_found":
                coverage[item.module] = "unresolved"

        total_model_duration_ms = sum(call.duration_ms for call in model_calls)
        validation_duration_ms = round((perf_counter() - validation_started) * 1000)
        finished_at = datetime.now(UTC)
        duration_ms = round((perf_counter() - run_started) * 1000)

        return IdentificationResult(
            document_package_id=document_package.document_package_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            provider=completion.provider,
            model=completion.model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            catalog_version=CATALOG_VERSION,
            catalog_fingerprint=CATALOG_FINGERPRINT,
            raw_model_output=json.dumps(
                {
                    "segments": [
                        {"segment": segment.label, "rawOutput": segment_completion.content}
                        for segment, _payload, segment_completion, _calls in segment_results
                    ]
                },
                ensure_ascii=False,
            ),
            model_calls=model_calls,
            processing_stages=[
                ProcessingStage(
                    key="model_call",
                    name="真实模型调用",
                    status="completed",
                    duration_ms=total_model_duration_ms,
                    detail=(
                        f"按 {len(segments)} 个文档结构分段综合识别，"
                        f"共 {len(model_calls)} 次模型调用，"
                        "未按22个模块循环调用"
                    ),
                ),
                ProcessingStage(
                    key="json_parse",
                    name="JSON 顶层解析",
                    status="completed",
                    duration_ms=0,
                    detail=f"{len(segments)} 个结构分段均解析为 JSON 对象",
                ),
                ProcessingStage(
                    key="contract_validation",
                    name="候选合同校验",
                    status="completed",
                    duration_ms=max(validation_duration_ms, 0),
                    detail=f"接受 {len(accepted)} 项，拒绝 {len(rejected)} 项",
                ),
                ProcessingStage(
                    key="evidence_validation",
                    name="分段证据校验",
                    status="completed",
                    duration_ms=0,
                    detail="有效候选、关系、弱线索和未决项仅允许引用所属结构段锚点",
                ),
                ProcessingStage(
                    key="document_aggregation",
                    name="文档内聚合检查",
                    status="completed",
                    duration_ms=0,
                    detail="本轮仅拒绝完全重复候选；跨分段同义聚合仍待下一轮验证",
                ),
            ],
            candidates=accepted,
            rejected_candidates=rejected,
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

    def _identify_segment(
        self,
        document_package: DocumentPackage,
        segment_label: str | None,
    ) -> tuple[dict[str, Any], ModelCompletion, list[ModelCallTrace]]:
        request = build_model_request(
            document_package,
            self.max_candidates,
            segment_label,
        )
        completion, model_calls = self._complete_with_retries(
            request,
            purpose="identification",
            first_attempt=1,
        )
        try:
            payload = json.loads(completion.content)
        except json.JSONDecodeError as parse_error:
            if completion.finish_reason == "length":
                reduced_candidate_limit = max(1, self.max_candidates // 2)
                limited_request = build_model_request(
                    document_package,
                    reduced_candidate_limit,
                    segment_label,
                )
                completion, retry_calls = self._complete_with_retries(
                    limited_request,
                    purpose="output_limit_retry",
                    first_attempt=len(model_calls) + 1,
                )
            else:
                repair_request = build_repair_request(
                    document_package.document_package_id,
                    completion.content,
                    str(parse_error),
                )
                completion, retry_calls = self._complete_with_retries(
                    repair_request,
                    purpose="repair",
                    first_attempt=len(model_calls) + 1,
                )
            model_calls.extend(retry_calls)
            try:
                payload = json.loads(completion.content)
            except json.JSONDecodeError as final_error:
                raise SegmentIdentificationFailure(
                    f"model output is not valid JSON after {model_calls[-1].purpose}",
                    model_calls,
                ) from final_error
        if not isinstance(payload, dict):
            raise SegmentIdentificationFailure(
                "model output top level must be a JSON object",
                model_calls,
            )
        return payload, completion, model_calls

    def _complete_with_retries(
        self,
        request: ModelRequest,
        *,
        purpose: Literal["identification", "output_limit_retry", "repair"],
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


def _candidate_id(raw_candidate: Any, index: int) -> str:
    if isinstance(raw_candidate, dict) and isinstance(raw_candidate.get("candidateId"), str):
        return raw_candidate["candidateId"]
    return f"INVALID-{index}"


def _raw_item(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}


def _namespace_payload(payload: dict[str, Any], namespace: str) -> dict[str, Any]:
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
                mention_id = mention["mentionId"]
                id_mapping[mention_id] = f"{namespace}-{mention_id}"

    for raw_candidate in namespaced.get("candidates", []):
        if not isinstance(raw_candidate, dict):
            continue
        candidate_id = raw_candidate.get("candidateId")
        if isinstance(candidate_id, str) and candidate_id in id_mapping:
            raw_candidate["candidateId"] = id_mapping[candidate_id]
        for mention in raw_candidate.get("entityMentions", []):
            if (
                isinstance(mention, dict)
                and isinstance(mention.get("mentionId"), str)
                and mention["mentionId"] in id_mapping
            ):
                mention_id = mention["mentionId"]
                mention["mentionId"] = id_mapping[mention_id]
        for relation in raw_candidate.get("relations", []):
            if not isinstance(relation, dict):
                continue
            for field in ("sourceRef", "targetRef"):
                reference = relation.get(field)
                if reference in id_mapping:
                    relation[field] = id_mapping[reference]
    return namespaced


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
