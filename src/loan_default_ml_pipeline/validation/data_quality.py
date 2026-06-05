import pandas as pd

# -- Schema contract ----------------------------------
# Columns the pipeline guarantees to produce.
# dtype class: 'numeric' = any of float/int/uint, 'bool' = boolean
REQUIRED_COLUMNS = {
    "default": "numeric",
    "loan_amount": "numeric",
    "interest_rate": "numeric",
    "annual_income": "numeric",
    "debt_to_income": "numeric",
    "grade": "numeric",
    "term": "numeric",
    "installment": "numeric",
    "emp_length": "numeric",
    "loan_income_ratio": "numeric",
    "installment_to_income": "numeric",
    "credit_utilization_rate": "numeric",
    "credit_age": "numeric",
    "is_joint_app": "numeric",
    "ever_delinquent": "numeric",
    "ever_90d_late": "numeric",
}

# -- Range contracts ----------------------------------
RANGE_CHECKS = {
    "interest_rate": (0, 100),
    "loan_amount": (0, 1_500_000),
    "annual_income": (0, 10_000_000),
    "debt_to_income": (0, 500),
    "grade": (1, 7),
    "term": (12, 84),
    "credit_utilization_rate": (0, 10),
    "default": (0, 1),
    "is_joint_app": (0, 1),
    "ever_delinquent": (0, 1),
    "ever_90d_late": (0, 1),
}

# -- Categorical contracts ----------------------------
CATEGORICAL_VALUES = {
    "default": {0, 1},
    "is_joint_app": {0, 1},
    "ever_delinquent": {0, 1},
    "ever_90d_late": {0, 1},
}


# -- Check functions ----------------------------------

def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.isnull()
        .sum()
        .to_frame("missing_count")
        .assign(
            percent_missing=lambda x: (x["missing_count"] / len(df)) * 100,
            dtype=df.dtypes
        )
        .sort_values("missing_count", ascending=False)
    )
    return summary


def check_schema(df: pd.DataFrame) -> dict:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    wrong_type = {
        col: str(df[col].dtype)
        for col, expected in REQUIRED_COLUMNS.items()
        if col in df.columns
        and expected == "numeric"
        and df[col].dtype.kind not in ("f", "i", "u")
    }
    passed = len(missing) == 0 and len(wrong_type) == 0
    return {
        "passed": passed,
        "missing_columns": missing,
        "wrong_dtype": wrong_type,
    }


def check_nulls(df: pd.DataFrame, threshold: float = 0.0) -> dict:
    null_counts = df.isnull().sum()
    violations = null_counts[null_counts / len(df) > threshold]
    passed = len(violations) == 0
    return {
        "passed": passed,
        "total_nulls": int(null_counts.sum()),
        "columns_above_threshold": violations.to_dict(),
    }


def check_ranges(df: pd.DataFrame) -> dict:
    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if out > 0:
            violations[col] = {
                "count": out,
                "expected": f"[{lo}, {hi}]",
                "actual_min": float(df[col].min()),
                "actual_max": float(df[col].max()),
            }
    return {"passed": len(violations) == 0, "violations": violations}


def check_duplicates(df: pd.DataFrame) -> dict:
    count = int(df.duplicated().sum())
    return {
        "passed": count == 0,
        "duplicate_rows": count,
        "duplicate_pct": round(count / len(df) * 100, 3),
    }


def check_class_balance(
    df: pd.DataFrame,
    target: str = "default",
    min_rate: float = 0.05,
    max_rate: float = 0.50,
) -> dict:
    if target not in df.columns:
        return {"passed": False, "error": f"Column '{target}' not found"}
    rate = float(df[target].mean())
    passed = min_rate <= rate <= max_rate
    return {
        "passed": passed,
        "default_rate": round(rate, 4),
        "expected_range": f"[{min_rate}, {max_rate}]",
    }


def check_categoricals(df: pd.DataFrame) -> dict:
    violations = {}
    for col, valid in CATEGORICAL_VALUES.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        unexpected = actual - valid
        if unexpected:
            violations[col] = {
                "unexpected_values": list(unexpected),
                "expected": list(valid),
            }
    return {"passed": len(violations) == 0, "violations": violations}


# -- Master orchestrator ------------------------------

def run_validation(
    df: pd.DataFrame,
    raise_on_fail: bool = False
) -> dict:
    results = {
        "schema": check_schema(df),
        "nulls": check_nulls(df),
        "ranges": check_ranges(df),
        "duplicates": check_duplicates(df),
        "class_balance": check_class_balance(df),
        "categoricals": check_categoricals(df),
    }

    overall = all(r["passed"] for r in results.values())
    results["overall_passed"] = overall

    _print_report(results)

    if raise_on_fail and not overall:
        failed = [k for k, v in results.items() if k != "overall_passed"
                  and not v["passed"]]
        raise ValueError(f"Validation failed: {failed}")

    return results


def _print_report(results: dict) -> None:
    labels = {
        "schema": "Schema",
        "nulls": "Zero nulls",
        "ranges": "Feature ranges",
        "duplicates": "No duplicates",
        "class_balance": "Class balance",
        "categoricals": "Categoricals",
    }
    print("=== Validation Report ===")
    for key, label in labels.items():
        r = results[key]
        status = "PASS" if r["passed"] else "FAIL"
        detail = ""
        if key == "schema" and not r["passed"]:
            detail = (f"missing={r['missing_columns']}, "
                      f"wrong_dtype={r['wrong_dtype']}")
        elif key == "nulls":
            detail = f"total={r['total_nulls']}"
        elif key == "ranges" and not r["passed"]:
            detail = f"violations={list(r['violations'].keys())}"
        elif key == "duplicates":
            detail = f"count={r['duplicate_rows']}"
        elif key == "class_balance":
            detail = f"rate={r['default_rate']}"
        elif key == "categoricals" and not r["passed"]:
            detail = f"violations={list(r['violations'].keys())}"
        print(f"  {label:<20} {status}  {detail}")
    verdict = "GO" if results["overall_passed"] else "NO-GO"
    print(f"  {'':20} ----")
    print(f"  {'Verdict':<20} {verdict}")
