# 🚀 Production MLOps Microservice Lab (Lab_04)
## Dual-Model Machine Learning API & Docker Containerization

![MLOps Pipeline Architecture](https://img.shields.io/badge/MLOps-Production--Ready-blue?style=for-the-badge&logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-24.0+-2496ED?style=for-the-badge&logo=docker)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn)

A complete, production-grade end-to-end MLOps pipeline and microservice architecture. This lab assignment demonstrates the complete workflow from dual-model training (Regression + Classification) and serialization to RESTful API serving using **FastAPI**, container orchestration via **Docker Compose**, and deployment strategies to cloud platforms.

---

## 📑 Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Folder Structure](#2-folder-structure)
3. [Model Training](#3-model-training)
4. [Saving Models](#4-saving-models)
5. [API Development](#5-api-development)
6. [Testing Locally](#6-testing-locally)
7. [Dockerfile Explanation](#7-dockerfile)
8. [Docker Compose Explanation](#8-docker-compose)
9. [Build Commands](#9-build-commands)
10. [Run Commands](#10-run-commands)
11. [Testing Commands](#11-testing-commands)
12. [Deployment Guide](#12-deployment)
13. [Public API Usage](#13-public-api-usage)
14. [Common Errors & Troubleshooting](#14-common-errors-and-fixes)
15. [Best Practices & Features](#15-best-practices)

---

## 1. System Architecture

The microservice follows a modular MLOps architecture separating model training pipelines from API inference orchestration and containerized execution environments.

```mermaid
flowchart TD
    subgraph Data & Training
        A1[California Housing Dataset] --> B1[Regression Pipeline: StandardScaler + RandomForestRegressor]
        A2[Breast Cancer Dataset] --> B2[Classification Pipeline: StandardScaler + RandomForestClassifier]
        B1 --> C1[Export models/regression.pkl]
        B2 --> C2[Export models/classification.pkl]
    end

    subgraph API Microservice Layer (FastAPI)
        C1 & C2 --> D[Model Predictor Engine]
        E[HTTP Client / Postman / Browser] -->|POST /predict/regression| F1[Regression Router]
        E -->|POST /predict/classification| F2[Classification Router]
        E -->|GET /health| F3[Health Router]
        
        F1 -->|Validate Pydantic Schema| D
        F2 -->|Validate Pydantic Schema| D
        D -->|Return Predictions & Confidence| E
    end

    subgraph Container Environment (Docker)
        G[Dockerfile - python:3.11-slim] --> H[Docker Image: lab04-mlops-api]
        H --> I[Docker Container: lab04_mlops_container]
    end
```

---

## 2. Folder Structure

```text
Lab_04/
├── datasets/
│   ├── housing_sample.csv        # Reference California housing sample dataset (100 rows)
│   └── cancer_sample.csv         # Reference Breast Cancer sample dataset (100 rows)
├── notebooks/
│   ├── regression.ipynb          # California Housing Regression model training notebook
│   └── classification.ipynb      # Breast Cancer Classification model training notebook
├── models/
│   ├── regression.pkl            # Trained Scikit-Learn Regression Pipeline (Model + Scaler)
│   └── classification.pkl        # Trained Scikit-Learn Classification Pipeline (Model + Scaler)
├── app/
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Application configurations & environment variables
│   ├── predictor.py              # Model loading logic & prediction engine
│   ├── schemas.py                # Pydantic V2 input validation & response schemas
│   ├── routes.py                 # FastAPI endpoints (/health, /predict/*)
│   └── main.py                   # FastAPI application initialization & middleware
├── scripts/
│   ├── train_models.py           # CLI script to train and serialize both models
│   └── test_api.py               # Automated HTTP client verification test script
├── Dockerfile                    # Production multi-stage slim Dockerfile
├── docker-compose.yml            # Local container orchestration file
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git tracking rules
└── README.md                     # Comprehensive project documentation
```

---

## 3. Model Training

The project builds **TWO completely independent machine learning pipelines**:

### Model 1: Regression Model (California Housing)
* **Dataset**: Scikit-Learn California Housing Dataset (8 numeric features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`).
* **Algorithm**: `RandomForestRegressor(n_estimators=50, random_state=42)` bundled inside a `StandardScaler` pipeline.
* **Evaluation Metrics**:
  * **MAE (Mean Absolute Error)**: `~0.32`
  * **MSE (Mean Squared Error)**: `~0.25`
  * **RMSE (Root Mean Squared Error)**: `~0.50`
  * **$R^2$ Score**: `~0.81`

### Model 2: Classification Model (Breast Cancer)
* **Dataset**: Scikit-Learn Breast Cancer Dataset (5 clinical attributes: `mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_smoothness`).
* **Algorithm**: `RandomForestClassifier(n_estimators=50, random_state=42)` bundled inside a `StandardScaler` pipeline.
* **Evaluation Metrics**:
  * **Accuracy**: `~94.7%`
  * **Precision**: `~95.8%`
  * **Recall**: `~95.8%`
  * **F1-Score**: `~95.8%`
  * **ROC-AUC**: `~98.9%`

To train both models from the command line:
```bash
python scripts/train_models.py
```

---

## 4. Saving Models

Models are exported using `joblib` as full Scikit-Learn `Pipeline` objects bundling pre-processing (`StandardScaler`) with model estimators:
* **Regression Binary**: `models/regression.pkl`
* **Classification Binary**: `models/classification.pkl`

Bundling standard scalers directly inside serialized pipelines ensures that input JSON payloads arriving at the API endpoints are automatically scaled without manual transformation code inside the route handlers.

---

## 5. API Development

Built using **FastAPI** with structured modular packaging (`app/` package):
* **Input Validation**: Strict type enforcement and bounds checking via **Pydantic V2**.
* **Startup Loading**: Models are loaded into memory once during application startup via FastAPI's `lifespan` context manager (`app/main.py`).
* **Exception Handling**: Global exception handler catches errors and returns structured JSON responses with appropriate HTTP status codes (`400`, `422`, `500`, `53`).
* **CORS Support**: Cross-Origin Resource Sharing enabled for web frontend integrations.

---

## 6. Testing Locally

### Option A: Running with Virtual Environment
```bash
# 1. Create & activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models
python scripts/train_models.py

# 4. Start local server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Accessing Interactive UI (Swagger)
Open your browser and navigate to:
```text
http://localhost:8000/docs
```
You can execute test requests directly through the interactive Swagger UI.

---

## 7. Dockerfile

Production-optimized multi-stage dockerfile featuring security and performance best practices:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY datasets/ ./datasets/
COPY models/ ./models/
COPY app/ ./app/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key Features**:
* **Minimal Base Image**: Uses `python:3.11-slim` to reduce container vulnerability footprint and image size.
* **Non-Root Execution**: Runs under a dedicated `appuser` system account to prevent root privilege escalation inside containers.
* **Layer Caching**: Copies `requirements.txt` before code assets to speed up incremental docker builds.

---

## 8. Docker Compose

Orchestrates container lifecycle with resource management and health checks (`docker-compose.yml`):

```yaml
version: '3.8'

services:
  mlops_api:
    build:
      context: .
      dockerfile: Dockerfile
    image: lab04-mlops-api:latest
    container_name: lab04_mlops_container
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 5s
```

---

## 9. Build Commands

To build the container image:

### Using Docker CLI:
```bash
docker build -t lab04-mlops-api:latest .
```

### Using Docker Compose:
```bash
docker-compose build
```

---

## 10. Run Commands

### Using Docker CLI:
```bash
docker run -d -p 8000:8000 --name lab04_mlops_container lab04-mlops-api:latest
```

### Using Docker Compose:
```bash
docker-compose up -d
```

### Container Management Commands:
```bash
# Check container status
docker ps

# Inspect logs
docker logs -f lab04_mlops_container

# Stop container
docker-compose down
```

---

## 11. Testing Commands

### Automated Test Script:
```bash
python scripts/test_api.py
```

### Manual Testing with `curl`:

#### 1. Health Check (`GET /health`)
```bash
curl -X GET "http://localhost:8000/health"
```
**Expected Response**:
```json
{
  "status": "ok",
  "regression_model_loaded": true,
  "classification_model_loaded": true,
  "version": "1.0.0"
}
```

#### 2. Regression Prediction (`POST /predict/regression`)
```bash
curl -X POST "http://localhost:8000/predict/regression" \
     -H "Content-Type: application/json" \
     -d '{
           "MedInc": 8.3252,
           "HouseAge": 41.0,
           "AveRooms": 6.9841,
           "AveBedrms": 1.0238,
           "Population": 322.0,
           "AveOccup": 2.5555,
           "Latitude": 37.88,
           "Longitude": -122.23
         }'
```
**Expected Response**:
```json
{
  "status": "success",
  "model_type": "California Housing RandomForestRegressor Pipeline",
  "predicted_value": 4.526,
  "unit": "$100,000"
}
```

#### 3. Classification Prediction (`POST /predict/classification`)
```bash
curl -X POST "http://localhost:8000/predict/classification" \
     -H "Content-Type: application/json" \
     -d '{
           "mean_radius": 17.99,
           "mean_texture": 10.38,
           "mean_perimeter": 122.8,
           "mean_area": 1001.0,
           "mean_smoothness": 0.1184
         }'
```
**Expected Response**:
```json
{
  "status": "success",
  "model_type": "Breast Cancer RandomForestClassifier Pipeline",
  "predicted_class_id": 0,
  "predicted_class_name": "Malignant",
  "confidence": 0.98,
  "probabilities": {
    "Malignant": 0.98,
    "Benign": 0.02
  }
}
```

---

## 12. Deployment

### Deploying to Render / Railway / Cloud Run

#### Option 1: Render (Web Service)
1. Push your repository to GitHub.
2. Log into [Render Dashboard](https://dashboard.render.com/) and click **New > Web Service**.
3. Connect your GitHub repository.
4. Select environment: **Docker**.
5. Render will automatically detect the `Dockerfile` in `Lab_04/`.
6. Click **Create Web Service**. Render will build and deploy the container giving you a public URL (e.g., `https://lab04-mlops-api.onrender.com`).

#### Option 2: Google Cloud Run
```bash
# 1. Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push image to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/lab04-mlops-api

# 3. Deploy to Cloud Run
gcloud run deploy lab04-mlops-service \
  --image gcr.io/YOUR_PROJECT_ID/lab04-mlops-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 13. Public API Usage

Once deployed to a cloud platform (e.g., Render), external users can invoke predictions without running code locally:

### Sending Requests via Postman:
1. Open **Postman**.
2. Create a `POST` request to `https://<YOUR-PUBLIC-URL>/predict/regression` or `/predict/classification`.
3. Set Header: `Content-Type: application/json`.
4. Add JSON body payloads (as shown in [Section 11](#11-testing-commands)).
5. Click **Send** to receive instant real-time predictions.

---

## 14. Common Errors and Fixes

| Problem / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `422 Unprocessable Entity` | JSON payload fields missing or wrong data types | Verify input JSON keys against `schemas.py` definitions. |
| `53 Service Unavailable` | Model file `.pkl` not found on disk | Ensure model training script (`scripts/train_models.py`) ran before API startup. |
| `Docker build fails: joblib missing` | Layer ordering issue in Dockerfile | Ensure `COPY requirements.txt .` precedes `RUN pip install`. |
| `Port 8000 already in use` | Another process is occupying port 8000 | Kill occupying process (`lsof -i :8000 | xargs kill -9`) or change `PORT=8001` in docker-compose. |

---

## 15. Best Practices & Bonus Features

* ✅ **Automatic OpenAPI / Swagger Documentation**: Self-documenting API endpoints live at `/docs`.
* ✅ **Comprehensive Input Schema Validation**: Prevents invalid values and schema mismatches via Pydantic V2.
* ✅ **Container Security**: Non-root system account execution inside Docker containers.
* ✅ **Health Monitoring**: Native `/health` status route integrated with Docker HEALTHCHECK instructions.
* ✅ **Structured Logging**: Timed application logs formatted for centralized log aggregators (Elasticsearch / CloudWatch).
