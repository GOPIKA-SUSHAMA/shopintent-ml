# ShopIntent ML  Customer Purchase Prediction API

A reproducible machine-learning project that predicts the probability of a purchase from online shopping session features. It combines exploratory data analysis, supervised classification, model comparison, probability calibration, a FastAPI inference service, and automated API testing.

**Status:** Working local prediction API; GitHub Actions CI passing as of 10 October 2026. Model-quality and latency benchmark results should be added after they are reviewed.

## Why this project?

Purchase outcomes are imbalanced: most browsing sessions do not end in a purchase. This project examines how classification models handle that imbalance and how a trained model can be integrated into a testable Python application.

The project demonstrates concepts relevant to junior ML engineering roles: **Python, Pandas, NumPy, Scikit-learn, XGBoost, feature preprocessing, model evaluation, probability estimates, REST APIs, Pytest, and CI**.

## Dataset

**Online Shoppers Purchasing Intention Dataset** — approximately 12,330 sessions, with a binary `Revenue` target indicating whether a session resulted in a purchase.

- [Kaggle dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
- [Original UCI dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

Download `online_shoppers_intention.csv` yourself and place it in `data/raw/`. Dataset files are deliberately excluded from Git; please follow the dataset's source terms.

## ML workflow

1. **Explore the data:** inspect types, missing values, duplicates, numerical distributions, and the `Revenue` class balance.
2. **Prepare features:** split training/validation/test data and preprocess numerical and categorical features with a Scikit-learn `ColumnTransformer` and `Pipeline`.
3. **Train and compare:** experiment with Logistic Regression, Random Forest, and XGBoost.
4. **Evaluate:** use precision, recall, F1, ROC-AUC, and average precision; experiment with cross-validation, tuning, and Brier score/probability calibration.
5. **Serve predictions:** load the selected, locally saved pipeline via Joblib and expose a typed FastAPI `/predict` endpoint.
6. **Test the API:** run integration tests against the real local model and unit tests using a fake model that requires no model artifact.

**Prediction-time caveat:** The source data contains session-derived features. This is a prototype using recorded session information; it is not evidence of predictions being available at the start of a live browsing session. `PageValues` was excluded from the initial API feature set to reduce one potential leakage risk.

## Repository structure

```text
shopintent-ml/
├── .github/workflows/ci.yml    # Python GitHub Actions workflow
├── api/main.py                 # FastAPI app and validated request schema
├── notebooks/01_data_exploration.ipynb
├── tests/test_api.py           # Local tests using saved model
├── tests/test_api_unit.py      # Mock-model API tests for CI
├── requirements.txt             # Full local environment
├── requirements-ci.txt          # Lightweight CI environment
├── .gitignore
├── data/raw/                   # Local CSV, not committed
└── models/                     # Saved model, not committed
```

## Local setup (Windows PowerShell)

Use Python 3.12 and compatible dependency versions for training and inference.

```powershell
cd "$HOME\Documents\shopintent-ml"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the CSV to `data/raw/`, open `notebooks/01_data_exploration.ipynb` in Jupyter, and run the model development cells. Save the selected fitted pipeline as `models/shopintent_model.joblib`. **The API requires this file to serve real predictions.** Model artifacts are not included in the repository.

### Start the API

```powershell
python -m uvicorn api.main:app --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Basic API response |
| `GET /health` | Confirm model can be loaded |
| `POST /predict` | Return predicted purchase class and probability |

### Example prediction request

Send the following JSON to `POST /predict` in Swagger UI:

```json
{
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
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

An example **observed local** response from this project was:

```json
{
  "purchase_prediction": 0,
  "purchase_probability": 0.1277,
  "threshold": 0.5
}
```

This is a single example prediction, **not** a model-accuracy or calibration result.

## Testing and continuous integration

```powershell
# No saved model needed: tests use a fake model
python -m pytest tests/test_api_unit.py -v

# Requires models/shopintent_model.joblib
python -m pytest tests/test_api.py -v
```

The `.github/workflows/ci.yml` workflow installs the Linux-compatible dependencies in `requirements-ci.txt` and runs model-independent API tests on pushes and pull requests to `main`. A successful run was confirmed on 10 October 2026; check the [Actions tab](https://github.com/GOPIKA-SUSHAMA/shopintent-ml/actions) for the latest status.

## Model evaluation results

Add actual validation/test measurements from the notebook once reviewed. Avoid selecting or tuning models on the held-out test set.

| Model | F1 | ROC-AUC | Average Precision | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| Tuned/calibrated XGBoost (if selected) | TBD | TBD | TBD | TBD |

## Next improvements

- Benchmark local prediction throughput and p50/p95/p99 request latency without Uvicorn development reload.
- Add model confidence intervals, feature-impact analysis, and calibration comparison.
- Improve input feature availability guarantees for a truly live session-prediction scenario.
- Document reproducible model training and dependency versions in a standalone script.

## Limitations

This is a **local ML engineering prototype**, not a deployed advertising auction system. It has not been validated for millions of requests per second, and no production latency or business-lift claim is made. Model training and deployment should use compatible Python and library versions; only load Joblib artifacts from trusted sources.
