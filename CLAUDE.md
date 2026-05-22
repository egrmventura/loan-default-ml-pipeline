# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

This is a self-directed ML engineering learning project. The domain is loan default prediction (LendingClub dataset). The domain is real enough to require genuine ML thinking, but it is not the point.

**The point:** develop the engineering discipline of an ML engineer — someone who can wrap a structured, reproducible, maintainable *system* around a data science problem. A mediocre model in a clean, testable, deployable pipeline is a better outcome here than an accurate model in a notebook.

Primary skill being developed: **feature engineering and pipeline architecture**, not model tuning.
Secondary skills: data validation, experiment tracking, API deployment, MLOps.

The project is explicitly not a Kaggle submission or a production service. It is a portfolio-grade engineering exercise.

---

## Commands

```bash
# Setup
make setup                 # Create venv and install dependencies
pip install -e .           # Editable install (enables package-style imports)

# Run pipeline stages
make ingestion_demo        # Run data ingestion
make train                 # Train model (reads data/processed/loans_featured.csv)
make run-api               # Start FastAPI inference server

# Quality
make lint                  # flake8 src/
make test                  # pytest tests/
```

Activate venv manually: `source venv/bin/activate`

**Import convention** — use package-style imports throughout:
```python
from loan_default_ml_pipeline.features.build_features import build_features
```
Not `from src.features...`. The Makefile still uses `src/` path style in some targets — prefer the package style in new code.

---

## Architecture

**Business formula:** `EL = PD × LGD × EAD` — this pipeline predicts PD (Probability of Default).

**Data flow:**
```
data/raw/loans_full_schema.csv
  → ingestion       (load only — no profiling logic here)
  → validation      (schema, nulls, quality checks)
  → features        (build_features pipeline)
  → data/processed/loans_featured.csv
  → training        (RandomForest → models/rf_loan_default.pkl)
  → evaluation      (classification report, AUC-ROC, confusion matrix)
  → inference/API   (stub — not yet implemented)
```

### Source modules (`src/loan_default_ml_pipeline/`)

| Module | Status | Purpose |
|---|---|---|
| `ingestion/load_data.py` | Done | `load_raw_data()` — loads raw CSV only |
| `features/build_features.py` | Done | `build_features(df)` — full feature pipeline |
| `training/train_model.py` | Done | Trains RandomForest, saves `.pkl` artifact |
| `validation/validate_model.py` | Done | Post-training eval: classification report, AUC-ROC, confusion matrix |
| `validation/data_quality.py` | Partial | `null_summary(df)` only — full validation layer not yet built |
| `inference/predict.py` | Stub | Empty — FastAPI endpoint not yet implemented |
| `pipelines/data_pipeline.py` | Missing | End-to-end orchestrator not yet built |

### Feature engineering pipeline (`build_features.py`)

Steps run in order inside `build_features(df)`:
1. Drop leakage columns (`paid_total`, `paid_principal`, `paid_interest`, `paid_late_fees`, `balance`)
2. Encode target: `default = 1` if `loan_status` in `{Charged Off, Late (31-120 days), Late (16-30 days)}`
3. Filter ambiguous statuses — keep only defaulted + Fully Paid, drop `loan_status`
4. Handle joint application nulls — adds `is_joint_app` flag, fills joint cols with 0
5. Handle delinquency nulls — adds `ever_delinquent` / `ever_90d_late` flags; sentinel-fills `months_since_*` with 999
6. Encode categoricals — ordinal grade map (A=1…G=7), one-hot nominal cols, drop high-cardinality cols

### Model

`RandomForestClassifier(n_estimators=200, class_weight="balanced")` — `balanced` handles the ~4:1 non-default:default imbalance. Target AUC-ROC > 0.75. Artifact at `models/rf_loan_default.pkl`.

### Notebooks

`notebooks/` is for exploration only. Naming: `01_data_profiling`, `02_feature_builds`, `03_model_training`. No pipeline logic lives here.

---

## Current State (as of 2026-05-22)

**Working:**
- Full feature pipeline (`build_features`) with 18 passing unit tests
- Model trained and saved (`rf_loan_default.pkl`)
- `make lint` clean, `make test` passing

**Known gaps (in priority order):**
1. `validation/data_quality.py` — only `null_summary` exists; schema enforcement, dtype checks, range checks, duplicate detection, and drift awareness are all missing
2. `pipelines/data_pipeline.py` — no end-to-end orchestrator; stages must be run manually
3. `make data` target — not yet in Makefile; needed for reproducible pipeline execution
4. `inference/predict.py` — stub only; FastAPI endpoint not built
5. MLflow experiment tracking — not yet integrated into training
6. Dependencies unpinned — `requirements.txt` has package names only, no versions
7. `docs/feature_log.md` and `docs/experiment_log.md` — partially seeded but need active maintenance
8. Parquet outputs — processed data is CSV; goal calls for parquet
9. Docker, CI/CD, monitoring — not yet started

**Pandas FutureWarning** in `handle_joint_nulls` and `handle_delinq_nulls` — `.fillna()` on object-dtype columns will change behavior in a future pandas version. Fix is to cast to float before filling.

---

## Development Workflow

**Notebook → Discover → Document → Implement → Commit**

1. Discover signals or features in notebooks
2. Document rationale in `docs/feature_log.md` and findings in `docs/experiment_log.md`
3. Move production logic into `src/`
4. Write tests
5. Commit clean implementation

The docs are not optional overhead — tracking *why* decisions were made is a first-class goal of this project. See `docs/feature_log.md` for the log format.
