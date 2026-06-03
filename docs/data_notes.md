# Data Notes

## loan_status
- Values: `Fully Paid`, `Charged Off`, `Late (31-120 days)`, `Late (16-30 days)`, `Current`, `In Grace Period`
- Mappings:
    - Default = 1: `Charged Off`, `Late (31-120 days)`, `Late (16-30 days)`, `In Grace Period`
    - Default = 0: `Fully Paid`
    - Drop entirely: `Current`
- Logic: `Current` (93.8% of raw data) has unknown outcome and cannot be labeled. `In Grace Period` added to default=1 in session 2026-05-22 — loans 1–15 days past due, consistent with treating Late (16-30 days) as default. Increased working dataset from 558 → 625 rows.

## missing_values -> post_default_filter
- months_since_* -> 9%-78%
    - Logic: Reliant on missed payments which have small probability in `Fully Paid` loan_status, making ~80% of list.
- emp_length, emp_title -> 6%

---

## Large Dataset Test — Branch: `feature/large-dataset-test`

**Decision (2026-05-25):** Pulling the LendingClub full historical dataset (~2.2M rows) to test whether increased data volume breaks the modeling ceiling reached at 625 rows (best test AUC-ROC: 0.61, recall: 0.33).

**Branch:** `feature/large-dataset-test` — work done here is isolated from `main` until validated.

**Hardware assessment (pre-pull):**
- Disk: ~1.8 GB total needed — safe (628 GB free)
- RAM: ~0.8 GB in-memory — safe (32 GB available)
- Training: single XGBoost fit ~10 min — acceptable
- Tuning (50-iter CV): ~7 hours — not practical; use known best params instead: `n_estimators=100, max_depth=4, learning_rate=0.2, subsample=0.8, min_child_weight=3, colsample_bytree=1.0`

**Merge criteria:** Branch merges to `main` only if the larger dataset produces a meaningful improvement in test AUC-ROC (target: >0.70) and recall on the default class (target: >0.40).

---

## Processed Data Format

**Format:** Parquet (was CSV as of 2026-05-25)

**Path:** `data/processed/loans_featured.parquet`

**Engine:** `pyarrow` (added to `requirements.txt`)

**Why parquet over CSV:**
- 2.2x smaller at current scale (92 KB vs 198 KB for 625 rows)
- Compression advantage grows significantly at scale — expected 5–10x at 2.2M rows
- Preserves dtypes natively — no silent type casting on read (CSV reads all bool columns as int, one-hot cols as object, etc.)
- Columnar format allows selective column reads — important when feature count grows
- `pd.read_parquet()` is faster than `pd.read_csv()` at any meaningful row count

**Scale context:** Moving to the LendingClub full dataset (~2.2M rows, ~1.5GB CSV) makes parquet format essential. Estimated processed parquet size: ~200–400MB vs ~1GB+ CSV.

**Migration:** `loans_featured.csv` removed. Pipeline now writes and reads parquet exclusively. Rebuild processed data with:
```python
from loan_default_ml_pipeline.ingestion.load_data import load_raw_data
from loan_default_ml_pipeline.features.build_features import build_features

df = build_features(load_raw_data())
df.to_parquet('data/processed/loans_featured.parquet', index=False, engine='pyarrow')
```