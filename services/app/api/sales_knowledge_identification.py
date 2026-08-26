import hashlib
import json
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from app.core.config import PROJECT_ROOT, get_settings
from app.features.sales_knowledge_identification.adapters.openai_compatible import (
    OpenAICompatibleGateway,
)
from app.features.sales_knowledge_identification.catalog import (
    CATALOG_FINGERPRINT,
    CATALOG_SOURCE,
    CATALOG_STATUS,
    CATALOG_VERSION,
    DOMAIN_BY_CODE,
    KNOWLEDGE_DOMAINS,
    KNOWLEDGE_MODULES,
    MODULE_SCOPE_DEFINITIONS,
)
from app.features.sales_knowledge_identification.content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
    CONTENT_CONTRACT_VERSION,
)
from app.features.sales_knowledge_identification.formalizer import (
    KnowledgeObjectFormationService,
)
from app.features.sales_knowledge_identification.models import (
    DocumentPackage,
    IdentificationResult,
    KnowledgeFormationResult,
    ModelConfigurationSnapshot,
    SourceMaterial,
    to_camel,
)
from app.features.sales_knowledge_identification.repository import (
    IdentificationRecordNotFound,
    PsycopgIdentificationRepository,
)
from app.features.sales_knowledge_identification.service import (
    DocumentPackageUnavailable,
    ModelGateway,
    SalesKnowledgeIdentificationService,
)

router = APIRouter(prefix="/sales-knowledge-identification")


class RunIdentificationRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    document_package_id: str


@router.get("/catalog")
def identification_catalog() -> dict[str, object]:
    return {
        "version": CATALOG_VERSION,
        "fingerprint": CATALOG_FINGERPRINT,
        "status": CATALOG_STATUS,
        "source": CATALOG_SOURCE,
        "contentContractVersion": CONTENT_CONTRACT_VERSION,
        "scopeDefinitions": MODULE_SCOPE_DEFINITIONS,
        "domains": [
            {
                "code": domain.code,
                "name": domain.name,
                "question": domain.question,
                "meaning": domain.meaning,
                "boundary": domain.boundary,
            }
            for domain in KNOWLEDGE_DOMAINS
        ],
        "modules": [
            {
                "domain": module.domain,
                "domainName": DOMAIN_BY_CODE[module.domain].name,
                "code": module.code,
                "name": module.name,
                "scope": module.scope,
                "meaning": module.meaning,
                "objectTypes": module.object_types,
                "coreObjects": module.core_objects,
                "boundary": module.boundary,
                "sources": module.sources,
                "consumers": module.consumers,
                "contentContract": {
                    "requiredFields": CONTENT_CONTRACT_BY_MODULE[module.code].required_fields,
                    "minimumContentChars": CONTENT_CONTRACT_BY_MODULE[
                        module.code
                    ].minimum_content_chars,
                    "granularity": CONTENT_CONTRACT_BY_MODULE[module.code].granularity,
                    "inclusion": CONTENT_CONTRACT_BY_MODULE[module.code].inclusion,
                    "exclusion": CONTENT_CONTRACT_BY_MODULE[module.code].exclusion,
                    "positiveExample": CONTENT_CONTRACT_BY_MODULE[
                        module.code
                    ].positive_example,
                    "negativeExample": CONTENT_CONTRACT_BY_MODULE[
                        module.code
                    ].negative_example,
                },
            }
            for module in KNOWLEDGE_MODULES
        ],
    }


@lru_cache
def get_identification_repository() -> PsycopgIdentificationRepository:
    settings = get_settings()
    repository = PsycopgIdentificationRepository(
        postgres_dsn=settings.postgres_dsn,
        project_root=PROJECT_ROOT,
        retention_hours=settings.identification_debug_retention_hours,
    )
    repository.ensure_schema()
    for manifest_path in sorted((settings.workspace_root / "documents").glob("*/manifest.json")):
        repository.register_manifest(manifest_path)
    return repository


@lru_cache
def get_model_gateway() -> ModelGateway:
    settings = get_settings()
    api_key = settings.llm_api_key
    if not api_key:
        raise RuntimeError(
            f"model API key environment variable is missing: {settings.llm_api_key_env}"
        )
    return OpenAICompatibleGateway(
        provider=settings.llm_provider,
        base_url=settings.llm_base_url,
        api_key=api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        enable_thinking=settings.llm_enable_thinking,
    )


@lru_cache
def get_knowledge_formation_service() -> KnowledgeObjectFormationService:
    return KnowledgeObjectFormationService(project_root=PROJECT_ROOT)


@router.get(
    "/source-materials",
    response_model=list[SourceMaterial],
)
def source_materials(
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
) -> list[SourceMaterial]:
    return repository.list_source_materials()


@router.get(
    "/document-packages/{document_package_id}",
    response_model=DocumentPackage,
)
def document_package(
    document_package_id: str,
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
) -> DocumentPackage:
    try:
        return repository.get_document_package(document_package_id)
    except IdentificationRecordNotFound as error:
        raise HTTPException(status_code=404, detail="DocumentPackage not found") from error


@router.post("/runs", response_model=IdentificationResult)
def run_identification(
    request: RunIdentificationRequest,
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
) -> IdentificationResult:
    try:
        package = repository.get_document_package(request.document_package_id)
    except IdentificationRecordNotFound as error:
        raise HTTPException(status_code=404, detail="DocumentPackage not found") from error
    if package.status != "available":
        raise HTTPException(status_code=409, detail="DocumentPackage is unavailable")
    settings = get_settings()
    configuration_values = {
        "temperature": settings.llm_temperature,
        "maxOutputTokens": settings.llm_max_output_tokens,
        "timeoutSeconds": settings.llm_timeout_seconds,
        "maxRetries": settings.llm_max_retries,
        "maxCandidates": settings.llm_max_candidates,
        "enableThinking": settings.llm_enable_thinking,
        "documentMaxChars": settings.llm_document_max_chars,
        "maxConcurrency": settings.llm_max_concurrency,
    }
    configuration_fingerprint = hashlib.sha256(
        json.dumps(configuration_values, sort_keys=True).encode()
    ).hexdigest()
    configuration = ModelConfigurationSnapshot.model_validate(
        {**configuration_values, "fingerprint": configuration_fingerprint}
    )
    service = SalesKnowledgeIdentificationService(
        gateway=get_model_gateway(),
        max_retries=settings.llm_max_retries,
        max_candidates=settings.llm_max_candidates,
        document_max_chars=settings.llm_document_max_chars,
        max_concurrency=settings.llm_max_concurrency,
        provider=settings.llm_provider,
        model=settings.llm_model,
        model_configuration=configuration,
    )
    try:
        result = service.identify(package)
    except DocumentPackageUnavailable as error:
        raise HTTPException(
            status_code=409, detail="DocumentPackage is unavailable"
        ) from error
    serialized = result.model_dump(mode="json", by_alias=True)
    repository.save_run(serialized)
    return result


@router.post(
    "/runs/{run_id}/knowledge-objects",
    response_model=KnowledgeFormationResult,
)
def form_knowledge_objects(
    run_id: str,
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
    service: Annotated[
        KnowledgeObjectFormationService, Depends(get_knowledge_formation_service)
    ],
) -> KnowledgeFormationResult:
    try:
        identification = IdentificationResult.model_validate(repository.get_run(run_id))
        package = repository.get_document_package(identification.document_package_id)
    except IdentificationRecordNotFound as error:
        raise HTTPException(status_code=404, detail="identification run not found") from error
    if identification.status != "completed":
        raise HTTPException(status_code=409, detail="identification run is not completed")
    entity_ids = service.candidate_entity_ids(package.workspace_id, identification.candidates)
    object_ids = service.candidate_object_ids(package.workspace_id, identification.candidates)
    formation = service.form(
        document_package=package,
        identification=identification,
        existing_entities=repository.get_existing_entity_ids(entity_ids),
        existing_objects=repository.get_existing_object_states(object_ids),
    )
    repository.save_knowledge_formation(
        workspace_id=package.workspace_id,
        formation=formation,
    )
    return formation


@router.get(
    "/runs/{run_id}/knowledge-objects",
    response_model=KnowledgeFormationResult,
)
def knowledge_formation(
    run_id: str,
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
) -> dict[str, object]:
    try:
        return repository.get_knowledge_formation(run_id)
    except IdentificationRecordNotFound as error:
        raise HTTPException(status_code=404, detail="knowledge formation not found") from error


@router.get("/runs", response_model=list[IdentificationResult])
def identification_runs(
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
    document_package_id: Annotated[str, Query(alias="documentPackageId")],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[dict[str, object]]:
    return repository.list_runs(document_package_id, limit)


@router.get("/runs/{run_id}")
def identification_run(
    run_id: str,
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
) -> dict[str, object]:
    try:
        return repository.get_run(run_id)
    except IdentificationRecordNotFound as error:
        raise HTTPException(status_code=404, detail="identification run not found") from error


@router.get("/evaluations/{document_package_id}")
def identification_evaluation(
    document_package_id: str,
    repository: Annotated[PsycopgIdentificationRepository, Depends(get_identification_repository)],
) -> dict[str, str]:
    try:
        markdown = repository.get_evaluation_report(document_package_id)
    except IdentificationRecordNotFound:
        markdown = ""
    return {"documentPackageId": document_package_id, "markdown": markdown}
