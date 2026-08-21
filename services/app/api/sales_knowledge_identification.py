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
    CATALOG_VERSION,
    DOMAIN_NAMES,
    KNOWLEDGE_MODULES,
)
from app.features.sales_knowledge_identification.models import (
    DocumentPackage,
    IdentificationResult,
    ModelConfigurationSnapshot,
    to_camel,
)
from app.features.sales_knowledge_identification.repository import (
    IdentificationRecordNotFound,
    PsycopgIdentificationRepository,
)
from app.features.sales_knowledge_identification.service import (
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
        "modules": [
            {
                "domain": module.domain,
                "domainName": DOMAIN_NAMES[module.domain],
                "code": module.code,
                "name": module.name,
                "objectTypes": module.object_types,
            }
            for module in KNOWLEDGE_MODULES
        ],
    }


@lru_cache
def get_identification_repository() -> PsycopgIdentificationRepository:
    settings = get_settings()
    return PsycopgIdentificationRepository(
        postgres_dsn=settings.postgres_dsn,
        project_root=PROJECT_ROOT,
        retention_hours=settings.identification_debug_retention_hours,
    )


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
    gateway: Annotated[ModelGateway, Depends(get_model_gateway)],
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
    result = SalesKnowledgeIdentificationService(
        gateway=gateway,
        max_retries=settings.llm_max_retries,
        max_candidates=settings.llm_max_candidates,
        document_max_chars=settings.llm_document_max_chars,
        max_concurrency=settings.llm_max_concurrency,
        provider=settings.llm_provider,
        model=settings.llm_model,
        model_configuration=configuration,
    ).identify(package)
    serialized = result.model_dump(mode="json", by_alias=True)
    repository.save_run(serialized)
    return result


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
    except IdentificationRecordNotFound as error:
        raise HTTPException(status_code=404, detail="evaluation report not found") from error
    return {"documentPackageId": document_package_id, "markdown": markdown}
