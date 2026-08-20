from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="STKB_",
        extra="ignore",
    )

    env: str = "local"
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

    @field_validator("knowledge_file_root")
    @classmethod
    def resolve_knowledge_file_root(cls, value: Path) -> Path:
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
