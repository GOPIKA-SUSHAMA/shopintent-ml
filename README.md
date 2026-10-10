# ShopIntent ML — Purchase Prediction and Low-Latency Model Serving

[![CI](https://github.com/GOPIKA-SUSHAMA/shopintent-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/GOPIKA-SUSHAMA/shopintent-ml/actions/workflows/ci.yml)

An **entry-level Machine Learning Engineering portfolio project** using shopping-session data to compare classifiers, estimate purchase probability, serve inference through a FastAPI endpoint, and benchmark API responsiveness under concurrency. Built with Python, Pandas, NumPy, Scikit-learn, XGBoost, Pytest and GitHub Actions. **No Docker required.**

## Problem and dataset

Given numerical and categorical observations from an online shopping session, estimate whether it ended in a purchase (`Revenue`). The project uses the [Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (also [on Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)), with 12,330 sessions and 17 explanatory attributes. Purchases form roughly 15.5% of observations, so Accuracy alone is insufficient.

**Data is not included in this repository.** Download the CSV from the dataset provider into `data/raw/online_shoppers_intention.csv` and follow the notebook's loading instructions. Local datasets and model artifacts are ignored by Git.

## Implemented ML experiments

The Jupyter notebook (`notebooks/01_data_exploration.ipynb`) explores missing values, feature types, duplicates, target imbalance and feature distributions, then trains and compares:

- Logistic Regression baseline, Random Forest and XGBoost classifiers
- Scikit-learn `Pipeline` / `ColumnTransformer` for numeric scaling and categorical encoding
- Stratified training/validation splitting, model comparison, XGBoost hyperparameter tuning
- Model probability evaluation and optional sigmoid calibration

The model-selection experiment prioritises **Average Precision**, complemented by F1, Precision, Recall, ROC-AUC, Brier Score and Log Loss. Actual notebook metrics should be read from its saved outputs; this README does not invent results.

**Important methodology:** The intended final 20% test split should remain untouched until model selection is complete. For a stricter deployment-focused experiment, use a chronological holdout and audit all features for time availability. Session-summary features may only exist after the visit, so the current demonstration is not proof of accurate early-session or real-time bidding predictions.

## Architecture

```text
Kaggle CSV -> Jupyter EDA -> Scikit-learn preprocessing -> ML experiments
                                                        |
                                                  fitted pipeline
                                                        |
                                                local Joblib artifact
                                                        |
PowerShell/HTTP client -> Pydantic input validation -> FastAPI -> model.predict_proba()
                                                        |
                                            probability + decision + timing

GitHub push -> GitHub Actions -> mock-model API unit tests
```

## Project structure

```text
api/main.py                         FastAPI prediction server
notebooks/01_data_exploration.ipynb  EDA, classifiers, tuning and calibration
tests/test_api.py                  Local integration tests with saved model
tests/test_api_unit.py             CI tests using a mock model
tests/benchmark_api.py             HTTP benchmark: p50/p95/p99, RPS, concurrency
tests/benchmark_model.py           ML pipeline-only inference benchmark
reports/performance/initial_baseline.md  Recorded initial benchmark outcomes
.github/workflows/ci.yml           GitHub Actions (Docker-free)
requirements-ci.txt               Lightweight CI dependencies
```

## Quick start (Windows PowerShell)

The existing trained pipeline is expected at `models/shopintent_model.joblib`. It was produced in Jupyter and is deliberately not committed. Recreate the environment with compatible Scikit-learn and XGBoost versions; cross-version loading of Joblib models is not guaranteed.

```powershell
cd "$HOME\Documents\shopintent-ml"
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_api_unit.py -q
python -m pytest tests/test_api.py -q
python -m uvicorn api.main:app --port 8001 --no-access-log
```

Open <http://127.0.0.1:8001/docs> for the interactive API. A `/predict` request must contain the 16 session features used during training (the API excludes `PageValues`). The response includes:

```json
{
  "purchase_prediction": 0,
  "purchase_probability": 0.1277,
  "threshold": 0.5,
  "feature_preparation_ms": 0.45,
  "model_pipeline_ms": 28.5
}
```

This JSON shows the observed example probability reported during development, with **illustrative timing values**. `model_pipeline_ms` includes the saved preprocessing/calibration model path; end-to-end HTTP latency is greater than this measurement.

## Benchmark inference

Run in a **second** PowerShell window while the API is serving port 8001:

```powershell
cd "$HOME\Documents\shopintent-ml"
.\.venv\Scripts\Activate.ps1
python tests/benchmark_api.py --url http://127.0.0.1:8001/predict --requests 100 --workers 1 5 10
python tests/benchmark_model.py --iterations 300
```

Benchmarks save machine-readable JSON files under `reports/performance/` (ignored by Git). For the first verified numbers, see [initial baseline](reports/performance/initial_baseline.md).

| Concurrent clients | HTTP mean | p95 | Throughput | Errors |
|---:|---:|---:|---:|---:|
| 1 | 52.49 ms | 68.20 ms | 19.00 req/s | 0/50 |
| 5 | 195.42 ms | 300.07 ms | 24.80 req/s | 0/50 |

These are **local laptop measurements** based on 50 requests each, not production-scale claims. Five clients achieved more throughput but increased per-request latency. Repeat with multiple runs, consistent model and server configuration, and larger request samples before making scalability claims. The synthetic single request exercises one path, not realistic session traffic.

## Tests and CI

`tests/test_api_unit.py` uses FastAPI dependency injection to replace the trained model with a NumPy-returning fake. CI can validate API responses, input rejection, timing-field presence and deterministic outputs **without downloading a Kaggle CSV or committing a binary model**.

The local integration suite `tests/test_api.py` needs the saved trained model and should be run on a machine with the correct ML dependencies. CI deliberately runs the mock suite only.

## Key engineering decisions and limits

- **Reusable model loading:** Joblib pipeline loaded once using `lru_cache` and injected through FastAPI dependencies.
- **Validation:** Pydantic enforces required fields and basic ranges; categorical integer IDs are converted to string before the saved pipeline.
- **Observability:** API reports feature preparation and model pipeline timing separately.
- **Reproducibility:** CI uses an isolated Python 3.12 Linux runner with minimal dependencies. The real-model runtime must match the local training environment.
- **Scope:** This is a single-machine educational system, not a distributed ad-auction service; no production traffic, A/B experiment or latency SLA is claimed.
- **Data leakage:** `PageValues` was intentionally excluded. Further work would require an explicit prediction-time feature contract and ideally chronological validation.

## Future experiments

Profile preprocessing versus model inference more deeply, test batch prediction and calibrate thresholds on held-out validation data. Optional extensions include clustering shopper sessions, regression on a suitable continuous target, and more rigorous statistical comparisons. These are future directions, not delivered capabilities.
