from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Test that the health endpoint returns 200 OK"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive", "model_loaded": True}

def test_prediction_flow():
    """Test that the prediction endpoint accepts data and returns a score"""
    payload = {"message_text": "I have experience in Python and Data Science.", "source": "linkedin"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert data["confidence"] > 0.0