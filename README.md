# Dynamic Pricing using Reinforcement Learning

An industry-level Reinforcement Learning project designed to optimize hotel room pricing dynamically based on demand, lead time, and guest demographics.

## Phase 1: Data Pipeline & Exploratory Data Analysis (EDA)

This phase establishes the foundational data loading, validation, pre-processing, feature engineering, and exploratory data analysis.

### Project Structure

```text
Dynamic-Pricing-RL/
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv
│   ├── processed/
│   └── external/
├── notebooks/
│   └── 01_EDA.ipynb
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
│   └── visualization/
│       ├── __init__.py
│       └── eda.py
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── pipeline.log
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

### Installation

To set up the project runtime dependencies, run:
```bash
pip install -r requirements.txt
```

To install dev dependencies:
```bash
pip install -r requirements-dev.txt
```

### Running the Pipeline

Run the end-to-end data pipeline (load, validate, clean, engineer features, generate plots):
```bash
python src/main.py
```

### Testing

Run the test suite:
```bash
pytest
```
