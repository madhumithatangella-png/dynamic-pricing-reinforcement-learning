"""Routes module mapping HTTP endpoints to business service invocations."""

from fastapi import APIRouter, Depends, HTTPException, status
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.rl.environment import HotelPricingEnvironment
from src.rl.baseline import FixedPricingAgent, DiscountPricingAgent, RandomPricingAgent
from src.dqn.evaluator import evaluate_agent
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BookingSimulationRequest,
    BookingSimulationResponse,
    EvaluationRequest,
    EvaluationResponse,
    ComparisonRequest,
    ComparisonResponse,
    ModelInfoResponse,
    MetricsResponse,
)
from api.dependencies import get_model_manager
from api.services import ModelManagerService

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Health check endpoint to verify API server liveness."""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "service": "dynamic-pricing-api"
    }


@router.get("/model-info", response_model=ModelInfoResponse, status_code=status.HTTP_200_OK)
def get_model_info(model_manager: ModelManagerService = Depends(get_model_manager)) -> ModelInfoResponse:
    """Returns active model details and capabilities."""
    return ModelInfoResponse(
        active_model=model_manager.active_model_name,
        price_actions=model_manager.price_actions,
        max_inventory=model_manager.max_inventory,
        max_days=model_manager.max_days,
        dqn_model_loaded=model_manager.dqn_agent is not None,
        q_learning_loaded=model_manager.q_learning_agent is not None,
    )


@router.get("/metrics", response_model=MetricsResponse, status_code=status.HTTP_200_OK)
def get_metrics(model_manager: ModelManagerService = Depends(get_model_manager)) -> MetricsResponse:
    """Retrieves API diagnostic counters and latency metrics."""
    avg_latency = (
        model_manager.total_latency_ms / model_manager.prediction_count
        if model_manager.prediction_count > 0
        else 0.0
    )
    avg_inference = (
        model_manager.total_inference_time_ms / model_manager.prediction_count
        if model_manager.prediction_count > 0
        else 0.0
    )
    device_name = "cpu"
    if model_manager.dqn_agent is not None:
        device_name = str(model_manager.dqn_agent.device)

    return MetricsResponse(
        prediction_count=model_manager.prediction_count,
        average_latency_ms=avg_latency,
        average_inference_time_ms=avg_inference,
        device=device_name,
        prediction_history=model_manager.prediction_history,
    )


@router.post("/predict-price", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_price(
    request: PredictionRequest,
    model_manager: ModelManagerService = Depends(get_model_manager)
) -> PredictionResponse:
    """Optimizes and recommends room pricing with explainable reasoning."""
    try:
        req_dict = request.model_dump()
        res_dict = model_manager.predict_price(req_dict)
        return PredictionResponse(**res_dict)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/simulate-booking", response_model=BookingSimulationResponse, status_code=status.HTTP_200_OK)
def simulate_booking(
    request: BookingSimulationRequest,
    model_manager: ModelManagerService = Depends(get_model_manager)
) -> BookingSimulationResponse:
    """Simulates booking transaction outcome based on offered price and demand curves."""
    req_dict = request.model_dump()
    cust_features = model_manager.get_customer_features(
        hotel_type=req_dict["hotel_type"],
        arrival_season=req_dict["arrival_season"],
        booking_month=req_dict["booking_month"],
        days_until_departure=req_dict["days_until_departure"],
    )
    prob = model_manager.demand_simulator.get_booking_probability(
        price=req_dict["price"],
        customer_features=cust_features
    )
    outcome = model_manager.demand_simulator.simulate_booking(
        price=req_dict["price"],
        customer_features=cust_features
    )
    return BookingSimulationResponse(
        booking_successful=outcome,
        booking_probability=prob
    )


@router.post("/evaluate-policy", response_model=EvaluationResponse, status_code=status.HTTP_200_OK)
def evaluate_policy(
    request: EvaluationRequest,
    model_manager: ModelManagerService = Depends(get_model_manager)
) -> EvaluationResponse:
    """Evaluates agent policy performance over simulated episodes."""
    model_type = request.model_type
    episodes = request.episodes

    if model_type == "dqn":
        agent = model_manager.dqn_agent
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trained DQN model state is not loaded."
            )
    elif model_type == "q_learning":
        agent = model_manager.q_learning_agent
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trained Q-Learning agent is not loaded."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model selection: {model_type}"
        )

    # Initialize environment
    env = HotelPricingEnvironment(
        data_path=str(model_manager.data_path),
        demand_simulator=model_manager.demand_simulator,
        max_inventory=model_manager.max_inventory,
        max_days=model_manager.max_days,
        price_actions=model_manager.price_actions,
        random_seed=42
    )

    metrics_list, _, _, _ = evaluate_agent(agent, env, episodes)
    df = pd.DataFrame(metrics_list)

    return EvaluationResponse(
        model_type=model_type,
        episodes_evaluated=episodes,
        average_revenue=float(df["revenue"].mean()),
        average_occupancy=float(df["occupancy_rate"].mean()),
        average_reward=float(df["reward"].mean()),
        rooms_sold=int(df["rooms_sold"].sum()),
    )


@router.post("/compare-models", response_model=ComparisonResponse, status_code=status.HTTP_200_OK)
def compare_models(
    request: ComparisonRequest,
    model_manager: ModelManagerService = Depends(get_model_manager)
) -> ComparisonResponse:
    """Evaluates and benchmarks all agents (DQN, Q-learning, baselines) concurrently."""
    episodes = request.episodes

    env = HotelPricingEnvironment(
        data_path=str(model_manager.data_path),
        demand_simulator=model_manager.demand_simulator,
        max_inventory=model_manager.max_inventory,
        max_days=model_manager.max_days,
        price_actions=model_manager.price_actions,
        random_seed=42
    )

    # Define test agents map
    test_agents = {}
    if model_manager.dqn_agent is not None:
        test_agents["DQN"] = model_manager.dqn_agent
    if model_manager.q_learning_agent is not None:
        test_agents["Q-Learning"] = model_manager.q_learning_agent

    test_agents["Fixed Pricing"] = FixedPricingAgent(price_actions=model_manager.price_actions)
    test_agents["Discount Pricing"] = DiscountPricingAgent(
        price_actions=model_manager.price_actions, max_days=model_manager.max_days
    )
    test_agents["Random Pricing"] = RandomPricingAgent(price_actions=model_manager.price_actions)

    comparison_results = []
    for name, agent in test_agents.items():
        metrics_list, _, prices, avg_inf_ms = evaluate_agent(agent, env, episodes)
        df = pd.DataFrame(metrics_list)

        comparison_results.append({
            "Agent": name,
            "Total Revenue": float(df["revenue"].sum()),
            "Revenue per Episode": float(df["revenue"].mean()),
            "Average Reward": float(df["reward"].mean()),
            "Booking Acceptance Rate": float(df["acceptance_rate"].mean()),
            "Occupancy Rate": float(df["occupancy_rate"].mean()),
            "Average Room Price": float(np.mean(prices)),
            "Rooms Sold": int(df["rooms_sold"].sum()),
            "Inference Time": avg_inf_ms
        })

    return ComparisonResponse(results=comparison_results)
