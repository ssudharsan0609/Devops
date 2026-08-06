"""
Stage 5: Model Evaluation
---------------------------
Loads the trained model and test set, computes regression evaluation metrics,
and writes them to metrics.json.

Input:
    model.pkl
    data/features/test.csv
Output:
    metrics.json
"""

import json
import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def load_model(path: str = "model.pkl"):
    model = joblib.load(path)
    print(f"[model_evaluation] Loaded model <- {path}")
    return model


def load_test_data(path: str = "data/features/test.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[model_evaluation] Loaded test data (shape={df.shape})")
    return df


def evaluate(model, df: pd.DataFrame) -> dict:
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": mse**0.5,
        "r2_score": r2_score(y_test, y_pred),
        "mape": (abs(y_test - y_pred) / y_test.abs()).mean() * 100,
    }
    return metrics


def save_metrics(metrics: dict, path: str = "metrics.json") -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[model_evaluation] Saved metrics -> {path}")
    print(json.dumps(metrics, indent=4))


def main():
    model = load_model()
    df = load_test_data()
    metrics = evaluate(model, df)
    save_metrics(metrics)


if __name__ == "__main__":
    main()
