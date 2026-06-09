# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A self-directed ML engineering learning project predicting loan default (PD) using the LendingClub dataset. The domain is real enough to require genuine ML thinking, but it is not the point.

**The point:** develop the engineering discipline of an ML engineer — someone who wraps a structured, reproducible, maintainable *system* around a data science problem. A mediocre model in a clean, testable, deployable pipeline is a better outcome than an accurate model in a notebook.

Primary skill: **feature engineering and pipeline architecture**. Secondary: data validation, experiment tracking, API deployment, MLOps.

Business formula: `EL = PD × LGD × EAD` — this pipeline predicts PD.

---

## Data Setup

**Data files are not in the repo** — excluded via `.gitignore` (raw files exceed GitHub's 100MB limit). Must be downloaded and built locally before running anything.

**Small dataset** (OpenIntro, 10k rows — for development/testing):
```bash
# Download loans_full_schema.csv from openintro.org and place at data/raw/
venv/bin/python -c "
from loan_default_ml_pipeline.ingestion.load_data import load_raw_data
from loan_default_ml_pipeline.features.build_features import build_features
df = build_features(load_raw_data())
df.to_parquet('data/processed/loans_featured.parquet', index=False, engine='pyarrow')
"
```

**Large dataset** (LendingClub full, 2.2M rows — for real training):
```bash
# Requires ~/.kaggle/kaggle.json with valid credentials
kaggle datasets download wordsforthewise/lending-club \
  -f accepted_2007_to_2018Q4.csv.gz -p data/raw/ --force
cd data/raw && gunzip -k accepted_2007_to_2018Q4.csv.gz
venv/bin/python -c "
from loan_default_ml_pipeline.ingestion.load_data import load_large_data
from loan_default_ml_pipeline.features.build_features import build_features
df = build_features(load_large_data())
df.to_parquet('data/processed/loans_featured.parquet', index=False, engine='pyarrow')
"
```
Processing takes ~55s. Output: `data/processed/loans_featured.parquet` (75MB, 1.38M rows).

---

## Commands

```bash
make setup                 # Create venv and install dependencies
pip install -e .           # Editable install (enables package-style imports)

make ingestion_demo        # Run data ingestion
make train                 # Train RF model
make run-api               # Start FastAPI inference server
make lint                  # flake8 src/
make test                  # pytest tests/
```

Activate venv: `source venv/bin/activate`

**Import convention:** package-style only:
```python
from loan_default_ml_pipeline.features.build_features import build_features
```

---

## Architecture

**Data flow:**
```
data/raw/loans_full_schema.csv
  → ingestion       (load_raw_data / load_large_data)
  → features        (build_features — 8-step pipeline)
  → validation      (run_validation — GO/NO-GO before saving)
  → data/processed/loans_featured.parquet
  → training        (RF or XGBoost → models/*.pkl)
  → evaluation      (classification report, AUC-ROC, confusion matrix)
  → inference/API   (GET /health, POST /predict — FastAPI)
```

Run the full data pipeline: `make data` (small) or `make data-large` (1.38M rows).
Start the API: `make run-api` → `http://localhost:8000`

**Processed format is parquet** (pyarrow engine) — not CSV. Rebuild with:
```python
df = build_features(load_raw_data())
df.to_parquet('data/processed/loans_featured.parquet', index=False, engine='pyarrow')
```

### Source modules (`src/loan_default_ml_pipeline/`)

| Module | Status | Purpose |
|---|---|---|
| `ingestion/load_data.py` | Done | `load_raw_data()` |
| `features/build_features.py` | Done | `build_features(df)` — full 7-step pipeline |
| `training/train_model.py` | Done | RF + XGBoost training + hyperparameter tuning |
| `validation/validate_model.py` | Done | Post-training eval: classification report, AUC-ROC, confusion matrix |
| `validation/data_quality.py` | Done | 6 checks + `run_validation()` with GO/NO-GO |
| `inference/predict.py` | Done | Feature transformation + XGBoost prediction |
| `inference/api.py` | Done | FastAPI — `GET /health`, `POST /predict` |
| `pipelines/data_pipeline.py` | Done | `run_data_pipeline(mode, validate)` — end-to-end |

### Feature engineering pipeline (`build_features.py`)

Steps run in order:
1. Drop leakage cols (`paid_total`, `paid_principal`, `paid_interest`, `paid_late_fees`, `balance`)
2. Encode target: `default=1` for `{Charged Off, Late (31-120 days), Late (16-30 days), In Grace Period}`
3. Filter ambiguous statuses — keep defaulted + Fully Paid only, drop `loan_status`
4. Handle joint nulls — adds `is_joint_app` flag, fills joint cols with 0
5. Handle delinquency nulls — adds `ever_delinquent`/`ever_90d_late` flags; sentinel-fills `months_since_*` with 999
6. Engineer ratio features — `loan_income_ratio`, `installment_to_income`, `credit_utilization_rate`, `credit_age`
7. Encode categoricals — ordinal grade (A=1…G=7), one-hot nominal cols, drop high-cardinality cols

Output (small dataset): 625 rows × 66 cols. Output (large dataset): 1,382,351 rows × 73 cols.
Step 8 (`impute_remaining_nulls`) handles sparse bureau fields in pre-2012 LendingClub records.

### Models

Both artifacts in `models/`:
- `rf_loan_default.pkl` — RandomForest, `class_weight="balanced"`, 62 features
- `xgb_loan_default.pkl` — XGBoost, `scale_pos_weight=neg/pos`, tuned via RandomizedSearchCV

XGBoost is the current best model. Training functions:
- `train_model()` — trains RF, saves to `rf_loan_default.pkl`
- `train_xgboost_model()` — trains XGBoost with default params
- `tune_xgboost_model()` — 50-iter RandomizedSearchCV, 5-fold CV, scores on AUC-ROC

---

## Experiment Summary

| Exp | Dataset | Change | AUC-ROC | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 558 rows | RF baseline | 0.6298 | 0.05 | 0.08 |
| 2 | 625 rows | +In Grace Period | 0.6309 | 0.14 | 0.22 |
| 3 | 625 rows | +loan_income_ratio | 0.6072 | 0.08 | 0.14 |
| 4 | 625 rows | XGBoost | 0.6217 | 0.33 | 0.37 |
| 5 | 625 rows | +3 ratio features | 0.6092 | 0.33 | 0.34 |
| 6 | 625 rows | Hyperparameter tuning | 0.6092 | 0.33 | 0.36 |
| **7** | **1.38M rows** | **XGBoost, large dataset** | **0.7305** | **0.68** | **0.47** |

Data volume was the hard constraint. Moving to the full LendingClub dataset broke the ceiling — AUC-ROC crossed 0.73, recall more than doubled. Full results in `docs/experiment_log.md`.

---

## Known Gaps (priority order)

1. **MLflow tracking** — not yet integrated into training
2. **Pinned dependencies** — `requirements.txt` has no versions
3. **Docker, CI/CD, monitoring** — not yet started

---

## Agents

10 agents available in `.claude/agents/`. Full index, trigger conditions, and recommended sequences: `.claude/agents/CLAUDE.md`.

Quick reference — most common:
- `ml-tester` — lint + tests + smoke test (run before every commit)
- `pr-ready` — full pre-PR checklist
- `data-validator` — validate parquet before training
- `pipeline-rerun` — full experiment orchestration
- `session-handoff` — end-of-session handoff doc

---

## Development Workflow

**Notebook → Discover → Document → Implement → Commit**

- `docs/feature_log.md` — log every feature decision with rationale
- `docs/experiment_log.md` — log every model run with metrics and conclusion
- `docs/data_notes.md` — log dataset observations and format decisions
- `tests/test_features.py` — 18 tests; update fixture (`make_minimal_df`) when adding source columns to the pipeline
- `tests/test_validation.py` — 33 tests covering all validation checks
- Total: 51 tests, all passing
