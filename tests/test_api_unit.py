import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import CATEGORY_COLUMNS, app, get_model


class FakeModel:
    def predict_proba(self, data):
        assert len(data) == 1
        assert all(isinstance(data.at[0, c], str) for c in CATEGORY_COLUMNS)
        return np.array([[0.8, 0.2]], dtype=float)


@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: FakeModel()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_model, None)


SAMPLE = {
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
    "Weekend": False,
}


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "1.1.0"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_prediction_and_timings(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    data = response.json()
    assert data["purchase_prediction"] == 0
    assert data["purchase_probability"] == 0.2
    assert data["threshold"] == 0.5
    assert data["feature_preparation_ms"] >= 0
    assert data["model_pipeline_ms"] >= 0


def test_invalid_bounce_rate(client):
    bad = {**SAMPLE, "BounceRates": 1.5}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_feature(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "Month"}
    assert client.post("/predict", json=bad).status_code == 422


def test_repeatable_prediction(client):
    a = client.post("/predict", json=SAMPLE).json()
    b = client.post("/predict", json=SAMPLE).json()
    assert a["purchase_probability"] == b["purchase_probability"]
    assert a["purchase_prediction"] == b["purchase_prediction"]
