import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="STKB_",
        extra="ignore",
    )

    env: str = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 54329
    postgres_db: str = "stkb_lab"
    postgres_user: str = "stkb"
    postgres_password: str = "stkb_local"
    neo4j_http_port: int = 7474
    neo4j_bolt_port: int = 7687
    neo4j_host: str = "127.0.0.1"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "stkb_local_password"
    knowledge_file_root: Path = Path("workspace/knowledge")
    workspace_root: Path = Path("workspace")
    llm_provider: str = "dashscope"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key_env: str = "DASHSCOPE_API_KEY"
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = Field(default=5000, ge=256, le=32000)
    llm_timeout_seconds: int = Field(default=90, ge=5, le=600)
    llm_max_retries: int = Field(default=1, ge=0, le=5)
    llm_max_candidates: int = Field(default=10, ge=1, le=100)
    llm_enable_thinking: bool = False
    llm_document_max_chars: int = Field(default=8000, ge=500, le=200000)
    llm_max_concurrency: int = Field(default=3, ge=1, le=8)
    identification_debug_retention_hours: int = Field(default=168, ge=1, le=2160)

    @field_validator("knowledge_file_root")
    @classmethod
    def resolve_knowledge_file_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("workspace_root")
    @classmethod
    def resolve_workspace_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def neo4j_uri(self) -> str:
        return f"bolt://{self.neo4j_host}:{self.neo4j_bolt_port}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def llm_api_key(self) -> str | None:
        return os.getenv(self.llm_api_key_env)


@lru_cache
def get_settings() -> Settings:
    return Settings()
