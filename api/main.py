
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="ShopIntent ML API",
    description="Customer purchase prediction using Machine Learning",
    version="1.0.0"
)

# Load the trained model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "shopintent_model.joblib"

model = joblib.load(MODEL_PATH)


# Define input features
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
    return {"message": "ShopIntent ML API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_purchase(session: ShoppingSession):

    # Convert request to DataFrame
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

    # Predict probability
    probability = float(
        model.predict_proba(data)[0, 1]
    )

    # Classification threshold
    prediction = int(probability >= 0.5)

    return {
        "purchase_prediction": prediction,
        "purchase_probability": round(probability, 4),
        "threshold": 0.5
    }
