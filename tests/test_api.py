#FastAPI's testing utilities
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

# Test the root endpont
def test_root():

    response = client.get("/")

    assert response.status_code == 200
    
# Test the health Endpoint
def test_health():

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"

# Test the prediction Endpoint
def test_predict():

    response = client.post(
        "/api/v1/predict",
        json={
            "text": "NASA discovers water beneath Mars."
        }
    )

    assert response.status_code == 200

    assert "prediction" in response.json()
    
# Test API version info
def test_info():

    response = client.get("/api/v1/info")

    assert response.status_code == 200

    assert response.json()["project"] == "Multilingual Fake News Detection"

    assert response.json()["status"] == "online"