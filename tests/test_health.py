"""Phase 0 validation tests: health check and API documentation endpoints."""

from fastapi.testclient import TestClient
from app.main import app

# Create a test client for FastAPI
client = TestClient(app)


def test_health_check_returns_ok():
    """Verify that GET /health returns HTTP 200 and status 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_page_accessible():
    """Verify that Swagger UI documentation at /docs is served successfully."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


def test_openapi_schema_accessible():
    """Verify that the OpenAPI JSON specification is generated correctly."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "WhatsApp Land Records RAG Chatbot API"
    assert "/health" in schema["paths"]
