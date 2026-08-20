from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_foundation_exposes_three_knowledge_forms() -> None:
    response = client.get("/api/foundation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stage"] == "engineering_baseline"
    assert [item["key"] for item in payload["knowledge_forms"]] == [
        "knowledge_file",
        "pgvector_projection",
        "neo4j_projection",
    ]


def test_knowledge_file_root_is_inside_project_workspace() -> None:
    settings = get_settings()

    assert settings.knowledge_file_root.parts[-2:] == ("workspace", "knowledge")
    assert settings.knowledge_file_root.is_absolute()
