from pydantic import BaseModel, Field
from typing import Literal


class HeartDiseaseInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="Sex (0=Female, 1=Male)")
    cp: Literal[0, 1, 2, 3] = Field(..., description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl (1=True, 0=False)")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina (1=Yes, 0=No)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: Literal[0, 1, 2] = Field(..., description="Slope of peak exercise ST segment (0-2)")
    ca: Literal[0, 1, 2, 3] = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: Literal[0, 1, 2, 3] = Field(..., description="Thal (0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Other)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 2,
                "thal": 3
            }
        }
    }


class HeartDiseasePrediction(BaseModel):
    prediction: int = Field(..., description="0 = No Disease, 1 = Disease")
    label: str = Field(..., description="Human-readable prediction label")
    probability_no_disease: float = Field(..., description="Probability of no heart disease")
    probability_disease: float = Field(..., description="Probability of heart disease")
    f1_score_on_test: float = Field(..., description="Model F1 score on test set")


class ModelInfo(BaseModel):
    model_name: str
    features: list[str]
    f1_score: float
    accuracy: float
    description: str
