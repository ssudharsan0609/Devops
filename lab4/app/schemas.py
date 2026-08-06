from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class RegressionInput(BaseModel):
    MedInc: float = Field(..., description="Median Income in block group ($10,000s)", example=8.3252)
    HouseAge: float = Field(..., description="Median House Age in block group", example=41.0)
    AveRooms: float = Field(..., description="Average number of rooms per household", example=6.9841)
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", example=1.0238)
    Population: float = Field(..., description="Block group population", example=322.0)
    AveOccup: float = Field(..., description="Average number of household members", example=2.5555)
    Latitude: float = Field(..., description="Block group latitude", example=37.88)
    Longitude: float = Field(..., description="Block group longitude", example=-122.23)

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0,
                "AveOccup": 2.5555,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }


class RegressionResponse(BaseModel):
    status: str = Field(..., example="success")
    model_type: str = Field(..., example="California Housing RandomForestRegressor")
    predicted_value: float = Field(..., description="Predicted median house value in $100,000s", example=4.526)
    unit: str = Field(..., example="$100,000")


class ClassificationInput(BaseModel):
    mean_radius: float = Field(..., description="Mean radius of tumor cells", example=17.99)
    mean_texture: float = Field(..., description="Mean texture (standard deviation of gray-scale values)", example=10.38)
    mean_perimeter: float = Field(..., description="Mean perimeter of tumor", example=122.8)
    mean_area: float = Field(..., description="Mean area of tumor", example=1001.0)
    mean_smoothness: float = Field(..., description="Mean smoothness (local variation in radius lengths)", example=0.1184)

    class Config:
        json_schema_extra = {
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184
            }
        }


class ClassificationResponse(BaseModel):
    status: str = Field(..., example="success")
    model_type: str = Field(..., example="Breast Cancer RandomForestClassifier")
    predicted_class_id: int = Field(..., description="0 = Malignant, 1 = Benign", example=0)
    predicted_class_name: str = Field(..., example="Malignant")
    confidence: float = Field(..., description="Probability score of predicted class", example=0.98)
    probabilities: Dict[str, float] = Field(
        ..., 
        example={"Malignant": 0.98, "Benign": 0.02}
    )


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    regression_model_loaded: bool = Field(..., example=True)
    classification_model_loaded: bool = Field(..., example=True)
    version: str = Field(..., example="1.0.0")
