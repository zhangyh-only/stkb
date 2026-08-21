from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.foundation import router as foundation_router
from app.api.sales_knowledge_identification import router as identification_router
from app.core.config import get_settings
from app.core.constants import API_PREFIX, SERVICE_NAME, SERVICE_VERSION

settings = get_settings()
app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(foundation_router, prefix=API_PREFIX)
app.include_router(identification_router, prefix=API_PREFIX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}
