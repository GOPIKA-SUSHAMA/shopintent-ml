import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SAMPLE = {
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
    "Weekend": False,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained ML pipeline only")
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    root = Path(__file__).resolve().parent.parent
    model = joblib.load(root / "models" / "shopintent_model.joblib")
    frame = pd.DataFrame([SAMPLE])
    for _ in range(20):
        model.predict_proba(frame)
    elapsed = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        model.predict_proba(frame)
        elapsed.append((time.perf_counter() - start) * 1000)
    results = {
        "iterations": args.iterations,
        "mean_ms": round(float(np.mean(elapsed)), 3),
        "p50_ms": round(float(np.percentile(elapsed, 50)), 3),
        "p95_ms": round(float(np.percentile(elapsed, 95)), 3),
        "p99_ms": round(float(np.percentile(elapsed, 99)), 3),
    }
    print(json.dumps(results, indent=2))
    output = root / "reports" / "performance" / "latest_model.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
