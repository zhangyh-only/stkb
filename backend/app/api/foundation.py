from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.constants import (
    KNOWLEDGE_FORM_FILE,
    KNOWLEDGE_FORM_GRAPH,
    KNOWLEDGE_FORM_VECTOR,
)

router = APIRouter()


class KnowledgeForm(BaseModel):
    key: str
    name: str
    technology: str
    role: str
    status: Literal["foundation_ready", "capability_pending"]


class FoundationResponse(BaseModel):
    stage: str
    positioning: str
    primary_store: str
    knowledge_forms: list[KnowledgeForm]


@router.get("/foundation", response_model=FoundationResponse)
def foundation() -> FoundationResponse:
    settings = get_settings()
    return FoundationResponse(
        stage="engineering_baseline",
        positioning="方案探索为主，代码与页面用于运行和观察验证结果",
        primary_store=f"PostgreSQL/{settings.postgres_db}",
        knowledge_forms=[
            KnowledgeForm(
                key=KNOWLEDGE_FORM_FILE,
                name="正式知识文件",
                technology="Markdown / local filesystem",
                role="Agent 可直接阅读的规范知识内容",
                status="capability_pending",
            ),
            KnowledgeForm(
                key=KNOWLEDGE_FORM_VECTOR,
                name="向量检索投影",
                technology="PostgreSQL / pgvector",
                role="从正式知识派生检索单元和 embedding",
                status="foundation_ready",
            ),
            KnowledgeForm(
                key=KNOWLEDGE_FORM_GRAPH,
                name="知识图谱投影",
                technology="Neo4j",
                role="从正式对象和关系派生图查询结构",
                status="foundation_ready",
            ),
        ],
    )
