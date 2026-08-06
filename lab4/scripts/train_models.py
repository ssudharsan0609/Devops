import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Paths setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def train_regression_model():
    print("=" * 60)
    print("1. Training Regression Model (California Housing)...")
    print("=" * 60)
    
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Feature columns and Target
    feature_names = housing.feature_names
    X = df[feature_names]
    y = df['MedHouseVal']
    
    # Save a small dataset sample for local reference
    sample_df = df.head(100)
    sample_path = os.path.join(DATASETS_DIR, "housing_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"[*] Saved sample dataset to: {sample_path}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scikit-Learn Pipeline: Scaler + Estimator
    reg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    
    # Train pipeline
    reg_pipeline.fit(X_train, y_train)
    
    # Predictions & Metrics
    y_pred = reg_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Regression Model Metrics:")
    print(f"   • MAE:      {mae:.4f}")
    print(f"   • MSE:      {mse:.4f}")
    print(f"   • RMSE:     {rmse:.4f}")
    print(f"   • R² Score: {r2:.4f}")
    
    # Export trained pipeline model
    model_path = os.path.join(MODELS_DIR, "regression.pkl")
    joblib.dump(reg_pipeline, model_path)
    print(f"\n[✓] Saved trained regression model to: {model_path}\n")


def train_classification_model():
    print("=" * 60)
    print("2. Training Classification Model (Breast Cancer)...")
    print("=" * 60)
    
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    
    # Use top 5 clinical features for clean API payloads
    selected_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness"
    ]
    
    X = df[selected_features]
    y = df['target']
    
    # Save a small dataset sample for local reference
    sample_df = df[selected_features + ['target']].head(100)
    sample_path = os.path.join(DATASETS_DIR, "cancer_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"[*] Saved sample dataset to: {sample_path}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scikit-Learn Pipeline: Scaler + Estimator
    clf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    # Train pipeline
    clf_pipeline.fit(X_train, y_train)
    
    # Predictions & Metrics
    y_pred = clf_pipeline.predict(X_test)
    y_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n📊 Classification Model Metrics:")
    print(f"   • Accuracy:  {acc:.4f}")
    print(f"   • Precision: {prec:.4f}")
    print(f"   • Recall:    {rec:.4f}")
    print(f"   • F1 Score:  {f1:.4f}")
    print(f"   • ROC-AUC:   {roc_auc:.4f}")
    
    # Export trained pipeline model
    model_path = os.path.join(MODELS_DIR, "classification.pkl")
    joblib.dump(clf_pipeline, model_path)
    print(f"\n[✓] Saved trained classification model to: {model_path}\n")


if __name__ == "__main__":
    train_regression_model()
    train_classification_model()
    print("✨ Model training pipeline completed successfully!")
