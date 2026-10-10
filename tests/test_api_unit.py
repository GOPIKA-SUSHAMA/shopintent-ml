
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import app, get_model



class FakeModel:
    """Mock ML model that matches Scikit-learn's predict_proba output."""

    def predict_proba(self, data):
        return np.array([[0.8, 0.2]])



@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: FakeModel()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_model, None)


sample_data = {
    "Administrative": 1,
    "Administrative_Duration": 20,
    "Informational": 0,
    "Informational_Duration": 0,
    "ProductRelated": 12,
    "ProductRelated_Duration": 300,
    "BounceRates": 0.02,
    "ExitRates": 0.04,
    "SpecialDay": 0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction(client):
    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert response.json()["purchase_probability"] == 0.2
    assert response.json()["purchase_prediction"] == 0


def test_invalid_input(client):
    invalid = sample_data.copy()
    invalid["BounceRates"] = 1.5

    response = client.post("/predict", json=invalid)

    assert response.status_code == 422
