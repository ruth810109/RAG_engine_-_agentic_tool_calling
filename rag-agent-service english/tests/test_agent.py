"""
Basic tests: confirm the FastAPI service starts up correctly and the health check
responds. (The full Agent / RAG flow needs a real GOOGLE_API_KEY to test, so for
now this only covers the parts that don't require a key.)
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
