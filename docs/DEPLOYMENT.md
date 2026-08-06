# Deployment Guide: Hotel Pricing RL Platform

This guide outlines deployment steps to host the FastAPI backend and Streamlit dashboard containers on cloud platforms.

## Environment Variables Configuration
Configure the following parameters in your cloud console:
*   `API_HOST`: Address of the FastAPI backend container (default: `localhost` or `api` in Docker Compose).
*   `API_PORT`: Port of the FastAPI backend container (default: `8000`).
*   `DEVICE`: PyTorch computation hardware (`cpu` or `cuda`). Defaults to `cpu` for server hosting.
*   `MODEL_PATH`: Directory location containing `.pth` and `.npy` state files (default: `/app/outputs/models`).

---

## 1. Render Deployment
Render supports native multi-service builds via Blueprints or single Docker Web Services.

### Option A: Render Blueprint (`render.yaml`)
Create a `render.yaml` file at the root to deploy both containers:
```yaml
services:
  - type: web
    name: dynamic-pricing-api
    env: docker
    dockerfilePath: deployment/Dockerfile
    dockerCommand: uvicorn api.main:app --host 0.0.0.0 --port 8000
    envVars:
      - key: PORT
        value: 8000
      - key: DEVICE
        value: cpu

  - type: web
    name: dynamic-pricing-dashboard
    env: docker
    dockerfilePath: deployment/Dockerfile
    dockerCommand: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    envVars:
      - key: API_HOST
        value: dynamic-pricing-api
      - key: API_PORT
        value: 8000
```

---

## 2. Railway Deployment
Railway builds and containerizes projects automatically using root-level Dockerfiles.

1.  Connect your GitHub repository to Railway.
2.  Deploy the **FastAPI Service**:
    *   Set build path to `deployment/Dockerfile` or override Command: `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
    *   Expose port `8000`.
3.  Deploy the **Streamlit Dashboard**:
    *   Set command override: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`.
    *   Add Environment Variable: `API_HOST = <your-deployed-railway-api-url>`.
    *   Expose port `8501`.

---

## 3. AWS EC2 (Elastic Compute Cloud)
For EC2, run Docker Compose directly.

### Step 1: Install Docker & Docker Compose
Connect to your EC2 instance via SSH and run:
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 2: Clone & Deploy
Clone the repository and spin up Nginx, the API, and the Dashboard:
```bash
git clone https://github.com/your-username/Dynamic-Pricing-RL.git
cd Dynamic-Pricing-RL

# Build and run containers in detached mode
docker-compose -f deployment/docker-compose.yml up -d --build
```
Ensure AWS Security Group ingress rules permit TCP traffic on ports:
*   `8000` (FastAPI Swagger)
*   `8501` (Streamlit Dashboard)

---

## 4. Microsoft Azure (App Service / Container Instances)
Deploy using Azure Container Instances (ACI) or Web App for Containers.

### Option A: Azure Container Registry (ACR) & ACI
```bash
# Log in to Azure CLI
az login

# Create resource group
az group create --name DynamicPricingRG --location eastus

# Create container registry
az acr create --resource-group DynamicPricingRG --name pricingregistry --sku Basic

# Log in to registry
az acr login --name pricingregistry

# Build and push to registry
docker build -t pricingregistry.azurecr.io/pricing-platform:latest -f deployment/Dockerfile .
docker push pricingregistry.azurecr.io/pricing-platform:latest

# Deploy API container
az container create --resource-group DynamicPricingRG --name pricing-api \
    --image pricingregistry.azurecr.io/pricing-platform:latest \
    --cpu 1 --memory 1.5 \
    --registry-login-server pricingregistry.azurecr.io \
    --ports 8000 --command-line "uvicorn api.main:app --host 0.0.0.0 --port 8000" \
    --ip-address Public
```
Retrieving the public IP of `pricing-api` and setting it as `API_HOST` environment variable to deploy the corresponding dashboard container in Azure Container Instances.
