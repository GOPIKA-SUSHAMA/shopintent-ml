from functools import lru_cache
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="ShopIntent ML API",
    description="Purchase-probability prediction from shopping-session features",
    version="1.1.0",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "shopintent_model.joblib"
CATEGORY_COLUMNS = ("OperatingSystems", "Browser", "Region", "TrafficType")


@lru_cache(maxsize=1)
def get_model():
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
    Month: str = Field(min_length=1)
    OperatingSystems: int = Field(ge=0)
    Browser: int = Field(ge=0)
    Region: int = Field(ge=0)
    TrafficType: int = Field(ge=0)
    VisitorType: str = Field(min_length=1)
    Weekend: bool


@app.get("/")
def home():
    return {"message": "ShopIntent ML API is running", "version": "1.1.0"}


@app.get("/health")
def health(model=Depends(get_model)):
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_purchase(session: ShoppingSession, model=Depends(get_model)):
    preparation_start = perf_counter()
    data = pd.DataFrame([session.model_dump()])
    for column in CATEGORY_COLUMNS:
        data[column] = data[column].astype(str)
    preparation_ms = (perf_counter() - preparation_start) * 1000

    inference_start = perf_counter()
    purchase_probability = float(model.predict_proba(data)[0, 1])
    model_pipeline_ms = (perf_counter() - inference_start) * 1000

    threshold = 0.5
    return {
        "purchase_prediction": int(purchase_probability >= threshold),
        "purchase_probability": round(purchase_probability, 4),
        "threshold": threshold,
        "feature_preparation_ms": round(preparation_ms, 3),
        "model_pipeline_ms": round(model_pipeline_ms, 3),
    }
