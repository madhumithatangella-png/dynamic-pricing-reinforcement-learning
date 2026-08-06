# Dynamic Pricing using Reinforcement Learning

An industry-level Reinforcement Learning project designed to optimize hotel room pricing dynamically based on demand, lead time, and guest demographics.

---

## Project Structure

```text
Dynamic-Pricing-RL/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   ├── services.py
│   └── dependencies.py
├── dashboard/
│   ├── app.py
│   ├── charts.py
│   └── pages/
│       ├── __init__.py
│       ├── dashboard.py
│       ├── prediction.py
│       ├── training.py
│       ├── comparison.py
│       └── evaluation.py
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv
│   ├── processed/
│   └── external/
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── docs/
│   ├── API.md
│   └── DEPLOYMENT.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 03_DQN.ipynb
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── logger.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   ├── data_preprocessing.py
│   │   └── feature_engineering.py
│   ├── dqn/
│   │   ├── __init__.py
│   │   ├── replay_buffer.py
│   │   ├── network.py
│   │   ├── dqn_agent.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── utils.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── environment.py
│   │   ├── demand_simulator.py
│   │   ├── q_learning.py
│   │   ├── baseline.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── utils.py
│   └── visualization/
│       ├── __init__.py
│       └── eda.py
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── models/
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── requirements-prod.txt
├── .dockerignore
└── README.md
```

---

## Phases Overview

### Phase 1: Data Pipeline & EDA
Establishes the foundational data loading, validation, pre-processing, feature engineering, and exploratory data analysis.
*   **Orchestration Command**: `python src/main.py --phase 1`

### Phase 2: Tabular Q-Learning RL Framework
Implements the Gym-like hotel pricing MDP environment, fits demand simulator curves, and trains tabular Q-learning agents against Fixed, Discount, and Random pricing baselines.
*   **Orchestration Command**: `python src/main.py --phase 2`

### Phase 3: PyTorch Deep Q-Network (DQN)
Implements continuous neural networks in PyTorch, epsilon-greedy exploration buffers, policy-target synchronizations, and benchmarking evaluation reports.
*   **Orchestration Command**: `python src/main.py --phase 3`

### Phase 4: Production Deployment & Dashboard
Exposes the active models via a FastAPI backend service and maps dynamic rate optimization fields to an interactive Streamlit dashboard.
*   **Orchestration Command**: `python src/main.py --phase 4`

---

## Installation & Setup

### Local Installation
To set up all runtime libraries, execute:
```bash
pip install -r requirements-prod.txt
```

### Running Backend REST API
Launch the FastAPI uvicorn server locally:
```bash
uvicorn api.main:app --port 8000 --reload
```
Interactive Swagger docs will be hosted at: `http://localhost:8000/docs`

### Running Streamlit Dashboard
Launch the dashboard locally:
```bash
streamlit run dashboard/app.py --server.port 8501
```
The dashboard interface will be hosted at: `http://localhost:8501`

---

## Docker Containerization

To build and spin up the complete multi-container system (Nginx + FastAPI + Streamlit), execute Docker Compose:
```bash
docker compose -f deployment/docker-compose.yml up -d --build
```

Verify service liveness:
*   **API Health**: `curl http://localhost:8000/health`
*   **Dashboard UI**: Open `http://localhost:8501` in your browser.

---

## Unit Testing
Execute the complete test suite verifying all 4 phases:
```bash
pytest
```
