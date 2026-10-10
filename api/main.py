
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field



app = FastAPI(
    title="ShopIntent ML API",
    description="Customer purchase prediction using Machine Learning",
    version="1.0.0"
)



PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "shopintent_model.joblib"
)


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the trained ML model."""
    return joblib.load(MODEL_PATH)



class ShoppingSession(BaseModel):
    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0)

    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)

    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)

    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)

    SpecialDay: float = Field(ge=0, le=1)

    Month: str

    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int

    VisitorType: str
    Weekend: bool



@app.get("/")
def home():
    return {
        "message": "ShopIntent ML API is running"
    }



@app.get("/health")
def health(model=Depends(get_model)):
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict_purchase(
    session: ShoppingSession,
    model=Depends(get_model)
):

    # Convert input data to DataFrame
    data = pd.DataFrame([session.model_dump()])

    # Match categorical types used during training
    category_codes = [
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType"
    ]

    for column in category_codes:
        data[column] = data[column].astype(str)

    # Generate purchase probability
    probability = float(
        model.predict_proba(data)[0, 1]
    )

    # Apply classification threshold
    threshold = 0.5
    prediction = int(probability >= threshold)

    return {
        "purchase_prediction": prediction,
        "purchase_probability": round(probability, 4),
        "threshold": threshold
    }
