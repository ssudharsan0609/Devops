import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas import (
    RegressionInput,
    RegressionResponse,
    ClassificationInput,
    ClassificationResponse,
    HealthResponse
)
from app.predictor import predictor
from app.config import settings

logger = logging.getLogger("mlops_app")

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    tags=["Health"]
)
def health_check():
    """Returns the operational status of the service and loaded model statuses."""
    reg_loaded = predictor.regression_model is not None
    clf_loaded = predictor.classification_model is not None
    
    return HealthResponse(
        status="ok",
        regression_model_loaded=reg_loaded,
        classification_model_loaded=clf_loaded,
        version=settings.VERSION
    )


@router.post(
    "/predict/regression",
    response_model=RegressionResponse,
    status_code=status.HTTP_200_OK,
    summary="California Housing Regression Endpoint",
    tags=["Inference"]
)
def predict_regression(payload: RegressionInput):
    """Predicts median house value ($100,000s) given housing block group features."""
    try:
        input_dict = payload.model_dump()
        prediction = predictor.predict_regression(input_dict)
        
        return RegressionResponse(
            status="success",
            model_type="California Housing RandomForestRegressor Pipeline",
            predicted_value=prediction,
            unit="$100,000"
        )
    except RuntimeError as re:
        logger.error(f"Regression prediction error: {re}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re)
        )
    except Exception as e:
        logger.error(f"Unexpected error in /predict/regression: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(e)}"
        )


@router.post(
    "/predict/classification",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Breast Cancer Tumor Classification Endpoint",
    tags=["Inference"]
)
def predict_classification(payload: ClassificationInput):
    """Classifies tumor sample as Malignant (0) or Benign (1) based on cell attributes."""
    try:
        input_dict = payload.model_dump()
        pred_id, pred_name, confidence, prob_dict = predictor.predict_classification(input_dict)
        
        return ClassificationResponse(
            status="success",
            model_type="Breast Cancer RandomForestClassifier Pipeline",
            predicted_class_id=pred_id,
            predicted_class_name=pred_name,
            confidence=confidence,
            probabilities=prob_dict
        )
    except RuntimeError as re:
        logger.error(f"Classification prediction error: {re}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re)
        )
    except Exception as e:
        logger.error(f"Unexpected error in /predict/classification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(e)}"
        )
