# Loan Default Prediction ML Pipeline

## Overview
End-to-end ML pipeline for predicting loan default risk, demonstrating production-style data engineering, feature pipelines, and model deployment practices.

Key formula: **EL = PD × LGD × EAD**

- **PD** = Probability of Default — probability the borrower will default within a time horizon (usually 12 months)
- **LGD** = Loss Given Default — percentage of loan lost if default occurs
- **EAD** = Exposure at Default — how much is owed when default occurs

This pipeline predicts PD.

---

## Stack
Python · Pandas · XGBoost · scikit-learn · MLflow · FastAPI · Docker

---

## Data Setup

**Data files are not committed to this repo** (raw files exceed GitHub's size limits). You must download and build locally.

### Small dataset (OpenIntro — 10k rows, for development)
Download from [OpenIntro](https://www.openintro.org/data/index.php?data=loans_full_schema) and place at:
```
data/raw/loans_full_schema.csv
```

Then rebuild processed data:
```bash
source venv/bin/activate
python -c "
from loan_default_ml_pipeline.ingestion.load_data import load_raw_data
from loan_default_ml_pipeline.features.build_features import build_features
df = build_features(load_raw_data())
df.to_parquet('data/processed/loans_featured.parquet', index=False, engine='pyarrow')
"
```

### Large dataset (LendingClub full — 2.2M rows, for training)
Requires [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and a Kaggle account:
```bash
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)
kaggle datasets download wordsforthewise/lending-club \
  -f accepted_2007_to_2018Q4.csv.gz -p data/raw/ --force
cd data/raw && gunzip -k accepted_2007_to_2018Q4.csv.gz
```

Then rebuild processed data:
```bash
source venv/bin/activate
python -c "
from loan_default_ml_pipeline.ingestion.load_data import load_large_data
from loan_default_ml_pipeline.features.build_features import build_features
df = build_features(load_large_data())
df.to_parquet('data/processed/loans_featured.parquet', index=False, engine='pyarrow')
"
```

---

## Quickstart

```bash
make setup          # create venv and install dependencies
pip install -e .    # editable install for package-style imports

make data           # build processed dataset (small) — or make data-large
make train          # train Random Forest model
make run-api        # start FastAPI inference server on :8000
make mlflow-ui      # browse MLflow experiment runs

make docker-build   # build inference API image
make docker-run     # serve the API in a container on :8000

make lint           # flake8 src/
make test           # pytest tests/
```

XGBoost (the best-performing model) is trained via `train_xgboost_model()` and tracked
with MLflow — see `src/loan_default_ml_pipeline/training/train_model.py`.

---

## Architecture
```
data/raw/
  → ingestion       (load_raw_data / load_large_data)
  → features        (build_features — 8-step pipeline)
  → validation      (run_validation — GO/NO-GO before saving)
  → data/processed/loans_featured.parquet
  → training        (RF or XGBoost → models/*.pkl, MLflow tracking)
  → evaluation      (AUC-ROC, classification report, confusion matrix)
  → inference/API   (GET /health, POST /predict — FastAPI, dockerized)
```

CI (`.github/workflows/ci.yml`) runs lint and the full test suite on every push/PR to `main`.

---

## Experiment Results

| Exp | Dataset | Model | AUC-ROC | Recall (Default) |
|---|---|---|---|---|
| 1–3 | 625 rows | Random Forest variants | 0.61–0.63 | 0.05–0.14 |
| 4–6 | 625 rows | XGBoost + tuning | 0.61 | 0.33 |
| 7 | 1.38M rows | XGBoost | **0.73** | **0.68** |

Data volume was the primary constraint. Moving to the full LendingClub dataset broke the modeling ceiling.

---

## Docs
- `docs/feature_log.md` — feature decisions and rationale
- `docs/experiment_log.md` — full experiment history with metrics
- `docs/data_notes.md` — dataset observations and format decisions
- `docs/handoff/` — session handoff notes
