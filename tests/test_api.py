
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Valid shopping-session input
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


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json=sample_data
    )

    assert response.status_code == 200

    result = response.json()

    assert result["purchase_prediction"] in [0, 1]
    assert 0 <= result["purchase_probability"] <= 1
    assert result["threshold"] == 0.5


def test_invalid_bounce_rate():
    invalid_data = sample_data.copy()
    invalid_data["BounceRates"] = 1.5

    response = client.post(
        "/predict",
        json=invalid_data
    )

    assert response.status_code == 422


def test_missing_required_feature():
    invalid_data = sample_data.copy()
    del invalid_data["Month"]

    response = client.post(
        "/predict",
        json=invalid_data
    )

    assert response.status_code == 422


def test_repeatable_prediction():
    first = client.post("/predict", json=sample_data)
    second = client.post("/predict", json=sample_data)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
