# 🧪 MLflow 101: Experiment Tracking & DagsHub Integration

Hands-on guide and lab repository for **MLflow** experiment tracking, model logging, and cloud integration with **DagsHub**.

---

## 📌 Project Overview

This repository demonstrates how to track, log, and manage machine learning experiments using **MLflow** both locally (with SQLite and local storage) and remotely via **DagsHub**.

🔗 **Live DagsHub MLflow Dashboard**: [Kishor-9361/DevOps MLflow Experiments](https://dagshub.com/Kishor-9361/DevOps.mlflow/#/experiments/0)

Key learning goals include:
- Tracking hyperparameters, performance metrics, and model artifacts across different experiment runs.
- Comparing multiple algorithms (Logistic Regression, Random Forest, XGBoost, Ridge, AdaBoost, etc.).
- Logging custom metrics (Accuracy, F1-Score, RMSE, MAE, $R^2$).
- Storing tracking data in a local SQLite database (`mlflow.db`) vs remote MLflow tracking servers hosted on DagsHub.

---

## 📁 Repository Structure

```text
Lab_03/
├── MLFlow_dagshub.ipynb    # Boston Housing Regression experiment tracked on DagsHub
├── ml_flow_1.ipynb         # Breast Cancer Classification experiment with local SQLite MLflow backend
├── README.md               # Documentation & setup instructions
└── .gitignore              # Ignores virtual environments, MLflow artifacts, and checkpoints
```

---

## 📓 Notebook Summaries

### 1. `ml_flow_1.ipynb` — Local MLflow Tracking (Classification)
- **Dataset**: Scikit-Learn Breast Cancer Dataset (569 samples, 30 features).
- **Task**: Binary Classification (Malignant vs. Benign).
- **Models Evaluated**: Logistic Regression, Random Forest, XGBoost (with SMOTETomek resampling).
- **Backend Store**: SQLite (`mlflow.db`) + local artifact store (`mlruns/`).
- **Logged Metrics**: Precision, Recall, F1-Score, Accuracy.

### 2. `MLFlow_dagshub.ipynb` — Remote DagsHub Integration (Regression)
- **Dataset**: Boston Housing Price Dataset.
- **Task**: Regression (Predicting median housing values).
- **Models Evaluated**: Linear Regression, Ridge, Random Forest, Gradient Boosting, AdaBoost, XGBoost.
- **Backend Store**: DagsHub Remote MLflow Tracking Server.
- **Logged Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), $R^2$ Score.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone git@github.com:Kishor-9361/DevOps.git
cd DevOps/Lab_03

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn mlflow dagshub jupyter matplotlib seaborn
```

---

## 📊 Viewing Local Experiments with MLflow UI

If you ran experiments locally with `ml_flow_1.ipynb`:

```bash
# If using SQLite backend store:
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Or standard local directory:
mlflow ui
```

Open your browser and navigate to `http://127.0.0.1:5000` to inspect runs, parameters, metrics, and artifact artifacts.

---

## 🌐 Remote Tracking with DagsHub

You can view the live logged experiments directly on the **[DagsHub MLflow Server](https://dagshub.com/Kishor-9361/DevOps.mlflow/#/experiments/0)**.

To log runs directly to your DagsHub repository:

```python
import dagshub
import mlflow

# Initialize DagsHub tracking integration
dagshub.init(repo_owner='Kishor-9361', repo_name='DevOps', mlflow=True)

# Start MLflow logging run
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("rmse", rmse_val)
    mlflow.sklearn.log_model(model, "model")
```

---

## 🛡️ Ignored Files (`.gitignore`)

The following auto-generated or heavy artifacts are excluded from Git tracking:
- Local Python virtual environments (`.venv/`)
- Jupyter checkpoints (`.ipynb_checkpoints/`)
- MLflow local databases (`mlflow.db`, `*.sqlite`)
- MLflow run logs and artifacts (`mlruns/`)
