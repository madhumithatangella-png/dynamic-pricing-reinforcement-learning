# Hotel Pricing RL API Documentation

This API hosts the pre-trained Reinforcement Learning models (DQN & Q-learning) to serve room pricing recommendations, transaction simulations, and benchmarking metrics.

## Base URL
When running locally: `http://localhost:8000`

---

## Endpoint References

### 1. Health Status
Verify API server liveness.

*   **URL**: `/health`
*   **Method**: `GET`
*   **Response Headers**: `Content-Type: application/json`
*   **Response Body**:
    ```json
    {
      "status": "healthy",
      "timestamp": "2026-08-06 19:15:30",
      "service": "dynamic-pricing-api"
    }
    ```
*   **cURL Example**:
    ```bash
    curl -X GET http://localhost:8000/health
    ```

---

### 2. Model Information
Inspect active model metadata and loaded weights.

*   **URL**: `/model-info`
*   **Method**: `GET`
*   **Response Body**:
    ```json
    {
      "active_model": "dqn",
      "price_actions": [3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0],
      "max_inventory": 50,
      "max_days": 100,
      "dqn_model_loaded": true,
      "q_learning_loaded": true
    }
    ```
*   **cURL Example**:
    ```bash
    curl -X GET http://localhost:8000/model-info
    ```

---

### 3. Monitoring Metrics
Retrieve transaction counters, compute device, and average execution durations.

*   **URL**: `/metrics`
*   **Method**: `GET`
*   **Response Body**:
    ```json
    {
      "prediction_count": 5,
      "average_latency_ms": 32.45,
      "average_inference_time_ms": 1.25,
      "device": "cpu",
      "prediction_history": [
        {
          "timestamp": "2026-08-06 19:16:02",
          "request": {
            "remaining_inventory": 40,
            "days_until_departure": 30,
            "arrival_season": "Summer",
            "booking_month": 7,
            "hotel_type": "Resort Hotel"
          },
          "response": {
            "recommended_price": 5500.0,
            "expected_revenue": 3960.0,
            "booking_probability": 0.72,
            "inventory_status": "Moderate",
            "occupancy_estimate": 0.20,
            "reason": "Strong seasonal booking demand (Summer) matching high segment ADR expectation...",
            "model_used": "dqn"
          },
          "latency_ms": 34.2
        }
      ]
    }
    ```
*   **cURL Example**:
    ```bash
    curl -X GET http://localhost:8000/metrics
    ```

---

### 4. Optimize Room Price
Calculates optimal room rate with explainable reasoning.

*   **URL**: `/predict-price`
*   **Method**: `POST`
*   **Request Headers**: `Content-Type: application/json`
*   **Request Body**:
    ```json
    {
      "remaining_inventory": 40,
      "days_until_departure": 30,
      "arrival_season": "Summer",
      "booking_month": 7,
      "hotel_type": "Resort Hotel",
      "model_type": "dqn"
    }
    ```
*   **Response Body**:
    ```json
    {
      "recommended_price": 5500.0,
      "expected_revenue": 3960.0,
      "booking_probability": 0.72,
      "inventory_status": "Moderate",
      "occupancy_estimate": 0.20,
      "reason": "Strong seasonal booking demand (Summer) matching high segment ADR expectation. Pricing set to premium ₹5500.",
      "model_used": "dqn"
    }
    ```
*   **cURL Example**:
    ```bash
    curl -X POST http://localhost:8000/predict-price \
      -H "Content-Type: application/json" \
      -d '{"remaining_inventory":40,"days_until_departure":30,"arrival_season":"Summer","booking_month":7,"hotel_type":"Resort Hotel","model_type":"dqn"}'
    ```

---

### 5. Simulate Booking
Directly query simulator likelihood of room conversion for custom rates.

*   **URL**: `/simulate-booking`
*   **Method**: `POST`
*   **Request Body**:
    ```json
    {
      "price": 5500.0,
      "hotel_type": "City Hotel",
      "arrival_season": "Spring",
      "booking_month": 4,
      "days_until_departure": 12
    }
    ```
*   **Response Body**:
    ```json
    {
      "booking_successful": true,
      "booking_probability": 0.684
    }
    ```
*   **cURL Example**:
    ```bash
    curl -X POST http://localhost:8000/simulate-booking \
      -H "Content-Type: application/json" \
      -d '{"price":5500.0,"hotel_type":"City Hotel","arrival_season":"Spring","booking_month":4,"days_until_departure":12}'
    ```

---

### 6. Evaluate Policy
Benchmark an agent's policy stability over specified episode iterations.

*   **URL**: `/evaluate-policy`
*   **Method**: `POST`
*   **Request Body**:
    ```json
    {
      "model_type": "dqn",
      "episodes": 10
    }
    ```
*   **Response Body**:
    ```json
    {
      "model_type": "dqn",
      "episodes_evaluated": 10,
      "average_revenue": 251250.0,
      "average_occupancy": 0.153,
      "average_reward": -185.59,
      "rooms_sold": 153
    }
    ```
*   **cURL Example**:
    ```bash
    curl -X POST http://localhost:8000/evaluate-policy \
      -H "Content-Type: application/json" \
      -d '{"model_type":"dqn","episodes":10}'
    ```

---

### 7. Compare Models
Evaluates and benchmarks all policy agents side-by-side.

*   **URL**: `/compare-models`
*   **Method**: `POST`
*   **Request Body**:
    ```json
    {
      "episodes": 10
    }
    ```
*   **Response Body**:
    ```json
    {
      "results": [
        {
          "Agent": "DQN",
          "Total Revenue": 2512500.0,
          "Revenue per Episode": 251250.0,
          "Average Reward": -185.59,
          "Booking Acceptance Rate": 0.515,
          "Occupancy Rate": 0.153,
          "Average Room Price": 5500.0,
          "Rooms Sold": 153,
          "Inference Time": 1.25
        },
        {
          "Agent": "Q-Learning",
          "Total Revenue": 2742000.0,
          "Revenue per Episode": 274200.0,
          "Average Reward": -144.65,
          "Booking Acceptance Rate": 0.544,
          "Occupancy Rate": 0.174,
          "Average Room Price": 5500.0,
          "Rooms Sold": 174,
          "Inference Time": 0.05
        }
      ]
    }
    ```
*   **cURL Example**:
    ```bash
    curl -X POST http://localhost:8000/compare-models \
      -H "Content-Type: application/json" \
      -d '{"episodes":10}'
    ```
