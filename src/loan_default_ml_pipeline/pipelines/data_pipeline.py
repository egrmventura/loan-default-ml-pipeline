import sys
import time
import os
from pathlib import Path

from loan_default_ml_pipeline.ingestion.load_data import (
    load_raw_data,
    load_large_data,
)
from loan_default_ml_pipeline.features.build_features import build_features
from loan_default_ml_pipeline.validation.data_quality import run_validation

PARQUET_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/processed/loans_featured.parquet"
)


def run_data_pipeline(
    mode: str = "small",
    validate: bool = True,
) -> None:
    if mode not in ("small", "large"):
        raise ValueError(f"mode must be 'small' or 'large', got '{mode}'")

    t0 = time.time()
    print(f"[data_pipeline] mode={mode}")

    print("[1/3] Loading raw data...")
    t1 = time.time()
    df = load_raw_data() if mode == "small" else load_large_data()
    print(f"      {df.shape[0]:,} rows loaded in {time.time() - t1:.1f}s")

    print("[2/3] Building features...")
    t2 = time.time()
    featured = build_features(df)
    print(f"      {featured.shape[0]:,} rows × {featured.shape[1]} cols "
          f"in {time.time() - t2:.1f}s")
    print(f"      Default rate : {featured['default'].mean():.3f}")
    print(f"      Nulls        : {featured.isnull().sum().sum()}")

    if validate:
        print("[3/4] Validating...")
        run_validation(featured, raise_on_fail=True)

    print("[4/4] Saving parquet..." if validate else "[3/3] Saving parquet...")
    t3 = time.time()
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")
    size_mb = os.path.getsize(PARQUET_PATH) / 1024 / 1024
    print(f"      Saved to {PARQUET_PATH}")
    print(f"      Size: {size_mb:.1f} MB  ({time.time() - t3:.1f}s)")

    print(f"[data_pipeline] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    mode = "large" if "--large" in sys.argv else "small"
    run_data_pipeline(mode=mode)
