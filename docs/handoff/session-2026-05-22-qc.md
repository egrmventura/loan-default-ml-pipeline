# QC Report & Session Log — 2026-05-22

## QC Results

| Check | Result |
|---|---|
| `make lint` (flake8 src/) | PASS — 0 violations |
| `make test` (pytest tests/) | PASS — 18/18 |
| Package import | PASS |
| Raw data present | PASS — `data/raw/loans_full_schema.csv` |
| Processed data present | PASS — `data/processed/loans_featured.csv` (625 rows, 63 cols) |
| Model artifact present | PASS — `models/rf_loan_default.pkl` |

**Known warnings (non-blocking):**
- 13 `FutureWarning` from pandas on `.fillna()` calls in `handle_joint_nulls` and `handle_delinq_nulls` — columns need explicit `.astype(float)` cast before fill. Does not affect correctness today; will break on a future pandas version.

---

## What Was Built This Session

### Infrastructure (was broken, now fixed)
- `flake8` and `pytest` added to `requirements.txt` and installed
- 35 lint violations resolved across all 5 source files
- 18 unit tests written from scratch (`tests/test_features.py`) — covers every feature pipeline step and `data_quality`

### Feature pipeline changes (`src/loan_default_ml_pipeline/features/build_features.py`)
- Added `In Grace Period` to `DEFAULT_STATUSES` and `KEEP_STATUSES` — increases working dataset from 558 → 625 rows, test defaults from 22 → 36
- Added `engineer_ratio_features()` as Step 6 — computes `loan_income_ratio = loan_amount / annual_income` with zero-income guard

### Documents created
- `CLAUDE.md` — full project guidance for future Claude instances
- `docs/feature_log.md` — seeded with all feature decisions and rationale
- `docs/experiment_log.md` — seeded with all 3 experiments
- `docs/handoff/session-2026-05-22.md` — session context and next steps
- `docs/handoff/session-2026-05-22-qc.md` — this file

---

## Experiment Summary

| Exp | Change | AUC-ROC | Recall (Default) | F1 (Default) |
|---|---|---|---|---|
| 1 | RF baseline (558 rows) | 0.6298 | 0.05 | 0.08 |
| 2 | +In Grace Period (625 rows) | 0.6309 | 0.14 | 0.22 |
| 3 | +loan_income_ratio | 0.6072 | 0.08 | 0.14 |

**Current best model:** Experiment 2 artifact (`models/rf_loan_default.pkl` — trained on 625 rows, 62 features, without `loan_income_ratio` — note: artifact on disk reflects Exp 3 since it was overwritten; retrain from Exp 2 state if needed by reverting `engineer_ratio_features` step).

**Key finding:** Dataset size is the primary bottleneck. The model consistently fails to recall defaults (catching 3–5 of 36 in test). Feature engineering and model changes have marginal effect until data volume increases or a different model class is tried.

---

## Immediate Next Steps

1. **Experiment 4 — XGBoost** with current feature set (62 features including `loan_income_ratio`)
   - XGBoost handles small tabular datasets better than Random Forest
   - Add `xgboost` training path to `training/train_model.py` or create `train_xgboost.py`

2. **Fix FutureWarnings** — cast joint and delinquency columns to float before `.fillna()`

3. **Pipeline orchestrator** — `src/loan_default_ml_pipeline/pipelines/data_pipeline.py` + `make data` target

4. **Validation layer** — `data_quality.py` has only `null_summary`; needs schema enforcement, range checks, duplicate detection

---

## Repo State at Commit

```
src/
  loan_default_ml_pipeline/
    ingestion/load_data.py        — loads raw CSV only
    features/build_features.py   — 7-step pipeline, 62 features out
    training/train_model.py       — RF training, joblib artifact
    validation/validate_model.py  — classification report, AUC-ROC, confusion matrix
    validation/data_quality.py    — null_summary only

tests/
  test_features.py                — 18 tests, all passing

docs/
  feature_log.md                  — all feature decisions logged
  experiment_log.md               — 3 experiments logged
  data_notes.md                   — loan_status mapping, missing value notes
  handoff/
    chatgpt-session-summary.md    — original ChatGPT session context
    session-2026-05-22.md         — this session narrative
    session-2026-05-22-qc.md      — this file

CLAUDE.md                         — project guidance for Claude Code
```
