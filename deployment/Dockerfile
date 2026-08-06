# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from project root (since context is root)
COPY requirements-prod.txt .

# Install production python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy directories to container
COPY src/ /app/src/
COPY api/ /app/api/
COPY dashboard/ /app/dashboard/
COPY outputs/ /app/outputs/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default runtime command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
