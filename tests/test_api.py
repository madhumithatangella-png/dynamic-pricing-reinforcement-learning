"""Unit tests for the FastAPI backend endpoints."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import status
from api.main import app
from api.dependencies import get_model_manager


@pytest.fixture
def mock_service() -> MagicMock:
    """Fixture providing a mocked ModelManagerService."""
    service = MagicMock()
    service.active_model_name = "dqn"
    service.price_actions = [3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0]
    service.max_inventory = 50
    service.max_days = 100
    service.dqn_agent = MagicMock()
    service.q_learning_agent = MagicMock()
    service.prediction_count = 5
    service.total_latency_ms = 100.0
    service.total_inference_time_ms = 10.0
    service.prediction_history = []
    
    # Mock predict_price response
    service.predict_price.return_value = {
        "recommended_price": 5500.0,
        "expected_revenue": 3960.0,
        "booking_probability": 0.72,
        "inventory_status": "Moderate",
        "occupancy_estimate": 0.20,
        "reason": "Seasonal demand test.",
        "model_used": "dqn"
    }
    
    # Mock customer features retrieval
    service.get_customer_features.return_value = {
        "hotel": "Resort Hotel",
        "arrival_season": "Summer",
        "market_segment": "Online TA",
        "customer_type": "Transient",
        "lead_time": 30,
        "booking_month": 7,
        "total_nights": 2,
        "total_guests": 2,
        "booking_window": 30
    }
    
    # Mock demand simulator capabilities
    service.demand_simulator = MagicMock()
    service.demand_simulator.get_booking_probability.return_value = 0.72
    service.demand_simulator.simulate_booking.return_value = True
    service.data_path = "mock_data.csv"
    
    return service


@pytest.fixture
def client(mock_service) -> TestClient:
    """Fixture providing a TestClient with dependency override."""
    app.dependency_overrides[get_model_manager] = lambda: mock_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_endpoint(client) -> None:
    """Verifies GET /health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["status"] == "healthy"


def test_model_info_endpoint(client) -> None:
    """Verifies GET /model-info endpoint."""
    resp = client.get("/model-info")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["active_model"] == "dqn"
    assert data["dqn_model_loaded"] is True


def test_metrics_endpoint(client) -> None:
    """Verifies GET /metrics endpoint."""
    resp = client.get("/metrics")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["prediction_count"] == 5
    assert data["average_latency_ms"] == 20.0  # 100 / 5
    assert data["average_inference_time_ms"] == 2.0  # 10 / 5


def test_predict_price_endpoint(client) -> None:
    """Verifies POST /predict-price endpoint with Pydantic serialization check."""
    payload = {
        "remaining_inventory": 40,
        "days_until_departure": 30,
        "arrival_season": "Summer",
        "booking_month": 7,
        "hotel_type": "Resort Hotel",
        "model_type": "dqn"
    }
    resp = client.post("/predict-price", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["recommended_price"] == 5500.0
    assert data["booking_probability"] == 0.72
    assert data["model_used"] == "dqn"


def test_simulate_booking_endpoint(client) -> None:
    """Verifies POST /simulate-booking endpoint."""
    payload = {
        "price": 5500.0,
        "hotel_type": "City Hotel",
        "arrival_season": "Spring",
        "booking_month": 4,
        "days_until_departure": 12
    }
    resp = client.post("/simulate-booking", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["booking_successful"] is True
    assert data["booking_probability"] == 0.72
