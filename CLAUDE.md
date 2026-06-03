# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A self-directed ML engineering learning project predicting loan default (PD) using the LendingClub dataset. The domain is real enough to require genuine ML thinking, but it is not the point.

**The point:** develop the engineering discipline of an ML engineer — someone who wraps a structured, reproducible, maintainable *system* around a data science problem. A mediocre model in a clean, testable, deployable pipeline is a better outcome than an accurate model in a notebook.

Primary skill: **feature engineering and pipeline architecture**. Secondary: data validation, experiment tracking, API deployment, MLOps.

Business formula: `EL = PD × LGD × EAD` — this pipeline predicts PD.

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
  → ingestion       (load_raw_data — CSV only, no profiling)
  → features        (build_features — 7-step pipeline)
  → data/processed/loans_featured.parquet
  → training        (RF or XGBoost → models/*.pkl)
  → evaluation      (classification report, AUC-ROC, confusion matrix)
  → inference/API   (stub — not yet implemented)
```

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
| `validation/data_quality.py` | Partial | `null_summary(df)` only |
| `inference/predict.py` | Stub | Empty |
| `pipelines/data_pipeline.py` | Missing | End-to-end orchestrator not yet built |

### Feature engineering pipeline (`build_features.py`)

Steps run in order:
1. Drop leakage cols (`paid_total`, `paid_principal`, `paid_interest`, `paid_late_fees`, `balance`)
2. Encode target: `default=1` for `{Charged Off, Late (31-120 days), Late (16-30 days), In Grace Period}`
3. Filter ambiguous statuses — keep defaulted + Fully Paid only, drop `loan_status`
4. Handle joint nulls — adds `is_joint_app` flag, fills joint cols with 0
5. Handle delinquency nulls — adds `ever_delinquent`/`ever_90d_late` flags; sentinel-fills `months_since_*` with 999
6. Engineer ratio features — `loan_income_ratio`, `installment_to_income`, `credit_utilization_rate`, `credit_age`
7. Encode categoricals — ordinal grade (A=1…G=7), one-hot nominal cols, drop high-cardinality cols

Output: 625 rows × 66 columns (after filtering ambiguous statuses from 10k raw rows).

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

| Exp | Change | Test AUC-ROC | Recall (Default) | F1 (Default) |
|---|---|---|---|---|
| 1 | RF baseline (558 rows) | 0.6298 | 0.05 | 0.08 |
| 2 | +In Grace Period (625 rows) | 0.6309 | 0.14 | 0.22 |
| 3 | +loan_income_ratio | 0.6072 | 0.08 | 0.14 |
| 4 | XGBoost | 0.6217 | 0.33 | 0.37 |
| 5 | +3 ratio features | 0.6092 | 0.33 | 0.34 |
| 6 | Hyperparameter tuning | 0.6092 | 0.33 | 0.36 |

**Current ceiling:** Test AUC-ROC ~0.61, recall ~0.33. Dataset size (625 rows) is the hard constraint — every modeling technique has been applied with marginal returns. Next meaningful step is the LendingClub full dataset (~2.2M rows).

---

## Known Gaps (priority order)

1. **Larger dataset** — 625 usable rows is the primary modeling bottleneck
2. **Validation layer** — `data_quality.py` has only `null_summary`; schema enforcement, range checks, duplicate detection missing
3. **Pipeline orchestrator** — `pipelines/data_pipeline.py` + `make data` target not yet built
4. **Inference endpoint** — `inference/predict.py` is a stub
5. **MLflow tracking** — not yet integrated
6. **FutureWarnings** — `.fillna()` on object-dtype cols in `handle_joint_nulls` / `handle_delinq_nulls`; fix with explicit `.astype(float)` cast before fill
7. **Pinned dependencies** — `requirements.txt` has no versions
8. **Docker, CI/CD, monitoring** — not yet started

---

## Development Workflow

**Notebook → Discover → Document → Implement → Commit**

- `docs/feature_log.md` — log every feature decision with rationale
- `docs/experiment_log.md` — log every model run with metrics and conclusion
- `docs/data_notes.md` — log dataset observations and format decisions
- `tests/test_features.py` — 18 tests, all passing; update fixture (`make_minimal_df`) when adding source columns to the pipeline
