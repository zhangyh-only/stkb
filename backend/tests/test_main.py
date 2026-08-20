from fastapi.testclient import TestClient

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
