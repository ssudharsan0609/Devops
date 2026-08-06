import os

class Settings:
    PROJECT_NAME: str = "Dual-Model MLOps Production API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REGRESSION_MODEL_PATH: str = os.getenv(
        "REGRESSION_MODEL_PATH",
        os.path.join(BASE_DIR, "models", "regression.pkl")
    )
    CLASSIFICATION_MODEL_PATH: str = os.getenv(
        "CLASSIFICATION_MODEL_PATH",
        os.path.join(BASE_DIR, "models", "classification.pkl")
    )
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()

