"""Schemas module defining Pydantic models for request/response serialization."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PredictionRequest(BaseModel):
    """Pydantic schema representing input parameters for room pricing prediction."""
    remaining_inventory: int = Field(..., ge=0, le=1000, description="Count of remaining rooms.")
    days_until_departure: int = Field(..., ge=0, le=365, description="Days remaining in booking window.")
    arrival_season: str = Field(..., description="Season name (Spring, Summer, Autumn, Winter).")
    booking_month: int = Field(..., ge=1, le=12, description="Calendar month (1 to 12).")
    hotel_type: str = Field(..., description="Hotel category (Resort Hotel or City Hotel).")
    model_type: Optional[str] = Field(None, description="Select agent model ('dqn' or 'q_learning').")

    class Config:
        json_schema_extra = {
            "example": {
                "remaining_inventory": 40,
                "days_until_departure": 30,
                "arrival_season": "Summer",
                "booking_month": 7,
                "hotel_type": "Resort Hotel",
                "model_type": "dqn"
            }
        }


class PredictionResponse(BaseModel):
    """Pydantic schema representing the recommendation response."""
    recommended_price: float = Field(..., description="Optimized recommended price (INR).")
    expected_revenue: float = Field(..., description="Expected revenue from potential booking (INR).")
    booking_probability: float = Field(..., description="Probability of booking acceptance (0.0 to 1.0).")
    inventory_status: str = Field(..., description="Categorical inventory status (e.g. Critical, Moderate, High).")
    occupancy_estimate: float = Field(..., description="Current occupancy rate estimate (0.0 to 1.0).")
    reason: str = Field(..., description="Explainable AI rationale statement.")
    model_used: str = Field(..., description="Active model type used for evaluation.")


class BookingSimulationRequest(BaseModel):
    """Pydantic schema representing parameters to simulate a booking transition."""
    price: float = Field(..., ge=0.0, description="Offered price value.")
    hotel_type: str = Field(..., description="Hotel type category.")
    arrival_season: str = Field(..., description="Season value.")
    booking_month: int = Field(..., ge=1, le=12, description="Month value.")
    days_until_departure: int = Field(..., ge=0, description="Days left until departure.")

    class Config:
        json_schema_extra = {
            "example": {
                "price": 5500.0,
                "hotel_type": "City Hotel",
                "arrival_season": "Spring",
                "booking_month": 4,
                "days_until_departure": 12
            }
        }


class BookingSimulationResponse(BaseModel):
    """Pydantic schema representing booking simulation results."""
    booking_successful: bool = Field(..., description="True if booking succeeds, False otherwise.")
    booking_probability: float = Field(..., description="Computed booking confirmation probability.")


class EvaluationRequest(BaseModel):
    """Pydantic schema representing model policy evaluation options."""
    model_type: str = Field(..., description="Model selection ('dqn' or 'q_learning').")
    episodes: int = Field(10, ge=1, le=100, description="Number of episodes for evaluation.")


class EvaluationResponse(BaseModel):
    """Pydantic schema representing metrics aggregated over evaluation episodes."""
    model_type: str = Field(..., description="Evaluated model category.")
    episodes_evaluated: int = Field(..., description="Number of episodes evaluated.")
    average_revenue: float = Field(..., description="Average revenue per episode (INR).")
    average_occupancy: float = Field(..., description="Average occupancy rate achieved.")
    average_reward: float = Field(..., description="Average episode total reward.")
    rooms_sold: int = Field(..., description="Total rooms sold during evaluation.")


class ComparisonRequest(BaseModel):
    """Pydantic schema representing benchmark model comparison options."""
    episodes: int = Field(10, ge=1, le=100, description="Number of evaluation episodes per agent.")


class ComparisonResponse(BaseModel):
    """Pydantic schema representing benchmark summary across all agents."""
    results: List[Dict[str, Any]] = Field(..., description="List of metric dictionaries for each agent.")


class ModelInfoResponse(BaseModel):
    """Pydantic schema representing model info and configurations."""
    active_model: str = Field(..., description="Active configured model.")
    price_actions: List[float] = Field(..., description="Price action space.")
    max_inventory: int = Field(..., description="Maximum room inventory.")
    max_days: int = Field(..., description="Maximum booking days horizon.")
    dqn_model_loaded: bool = Field(..., description="True if DQN model weights are active.")
    q_learning_loaded: bool = Field(..., description="True if Q-Learning table is active.")


class MetricsResponse(BaseModel):
    """Pydantic schema representing logging and monitoring metrics."""
    prediction_count: int = Field(..., description="Total price predictions served.")
    average_latency_ms: float = Field(..., description="Average prediction latency in milliseconds.")
    average_inference_time_ms: float = Field(..., description="Average ML inference time in milliseconds.")
    device: str = Field(..., description="Computational device used (cpu or cuda).")
    prediction_history: List[Dict[str, Any]] = Field(..., description="Recent prediction request history logs.")
