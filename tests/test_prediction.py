"""Unit tests for XAI reasoning and helper functions in ModelManagerService."""

import pytest
from unittest.mock import MagicMock
from api.services import ModelManagerService


def test_inventory_status_categorization() -> None:
    """Verifies inventory status mapping ratios."""
    # We can instantiate a mock service with only max_inventory defined to avoid loading simulator
    service = MagicMock(spec=ModelManagerService)
    service.max_inventory = 50
    service.get_inventory_status = ModelManagerService.get_inventory_status.__get__(service, ModelManagerService)

    # 4 rooms left: 4 / 50 = 8.0% <= 15% -> Critical
    assert service.get_inventory_status(4) == "Critical"
    
    # 20 rooms left: 20 / 50 = 40.0% <= 50% -> Moderate
    assert service.get_inventory_status(20) == "Moderate"
    
    # 45 rooms left: 45 / 50 = 90.0% > 50% -> High
    assert service.get_inventory_status(45) == "High"


def test_recommendation_reason_generation() -> None:
    """Verifies natural language XAI reasoning statement outputs for different contexts."""
    service = MagicMock(spec=ModelManagerService)
    service.max_inventory = 50
    service.max_days = 100
    service.get_inventory_status = ModelManagerService.get_inventory_status.__get__(service, ModelManagerService)
    service.generate_recommendation_reason = ModelManagerService.generate_recommendation_reason.__get__(service, ModelManagerService)

    # Context 1: Critical inventory
    reason_critical = service.generate_recommendation_reason(
        price=7000.0,
        expected_price=5000.0,
        remaining_inventory=5,
        days_until_departure=50,
        arrival_season="Summer"
    )
    assert "Critical inventory level" in reason_critical

    # Context 2: Close to departure, discounted
    reason_discount = service.generate_recommendation_reason(
        price=3000.0,
        expected_price=5000.0,
        remaining_inventory=25,
        days_until_departure=5,
        arrival_season="Summer"
    )
    assert "Close to booking horizon limit" in reason_discount
    assert "discounted" in reason_discount

    # Context 3: Premium pricing during high demand
    reason_premium = service.generate_recommendation_reason(
        price=6500.0,
        expected_price=5000.0,
        remaining_inventory=25,
        days_until_departure=50,
        arrival_season="Summer"
    )
    assert "Strong seasonal booking demand" in reason_premium
    assert "premium" in reason_premium
