
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Locate our saved model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "shopintent_model.joblib"

model = joblib.load(MODEL_PATH)

# Same features used by our FastAPI endpoint
sample = pd.DataFrame([{
    "Administrative": 1,
    "Administrative_Duration": 20,
    "Informational": 0,
    "Informational_Duration": 0,
    "ProductRelated": 12,
    "ProductRelated_Duration": 300,
    "BounceRates": 0.02,
    "ExitRates": 0.04,
    "SpecialDay": 0,
    "Month": "Nov",
    "OperatingSystems": "2",
    "Browser": "2",
    "Region": "1",
    "TrafficType": "2",
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}])

# Warm up the model
for _ in range(20):
    model.predict_proba(sample)

# Measure prediction execution time
latencies = []

for _ in range(300):
    start = time.perf_counter()

    model.predict_proba(sample)

    elapsed = (time.perf_counter() - start) * 1000
    latencies.append(elapsed)

print("\nMODEL INFERENCE BENCHMARK")
print("--------------------------")
print(f"Average: {np.mean(latencies):.2f} ms")
print(f"p50: {np.percentile(latencies, 50):.2f} ms")
print(f"p95: {np.percentile(latencies, 95):.2f} ms")
print(f"p99: {np.percentile(latencies, 99):.2f} ms")
