import pytest
import pandas as pd
from loan_default_ml_pipeline.validation.data_quality import (
    check_schema,
    check_nulls,
    check_ranges,
    check_duplicates,
    check_class_balance,
    check_categoricals,
    run_validation,
    null_summary,
    REQUIRED_COLUMNS,
    RANGE_CHECKS,
)


def make_valid_df(n=10):
    """Minimal valid featured DataFrame satisfying all validation contracts.
    Rows are unique — loan_amount varies by index."""
    return pd.DataFrame({
        "default": [0, 1] * (n // 2),
        "loan_amount": [10000.0 + i for i in range(n)],  # unique per row
        "interest_rate": [12.5] * n,
        "annual_income": [60000.0] * n,
        "debt_to_income": [20.0] * n,
        "grade": [2] * n,
        "term": [36] * n,
        "installment": [300.0] * n,
        "emp_length": [5.0] * n,
        "loan_income_ratio": [0.17] * n,
        "installment_to_income": [0.06] * n,
        "credit_utilization_rate": [0.3] * n,
        "credit_age": [15] * n,
        "is_joint_app": [0] * n,
        "ever_delinquent": [0] * n,
        "ever_90d_late": [0] * n,
    })


# -- check_schema -------------------------------------

def test_schema_passes_on_valid_df():
    result = check_schema(make_valid_df())
    assert result["passed"] is True
    assert result["missing_columns"] == []
    assert result["wrong_dtype"] == {}


def test_schema_fails_on_missing_column():
    df = make_valid_df().drop(columns=["interest_rate"])
    result = check_schema(df)
    assert result["passed"] is False
    assert "interest_rate" in result["missing_columns"]


def test_schema_fails_on_string_dtype():
    df = make_valid_df()
    df["grade"] = df["grade"].astype(str)
    result = check_schema(df)
    assert result["passed"] is False
    assert "grade" in result["wrong_dtype"]


def test_schema_accepts_int_for_numeric_cols():
    df = make_valid_df()
    df["loan_amount"] = df["loan_amount"].astype(int)
    result = check_schema(df)
    assert result["passed"] is True


def test_schema_reports_all_missing_columns():
    df = make_valid_df().drop(columns=["grade", "term", "installment"])
    result = check_schema(df)
    assert set(result["missing_columns"]) == {"grade", "term", "installment"}


# -- check_nulls --------------------------------------

def test_nulls_passes_on_clean_df():
    result = check_nulls(make_valid_df())
    assert result["passed"] is True
    assert result["total_nulls"] == 0


def test_nulls_fails_when_null_present():
    df = make_valid_df()
    df.loc[0, "interest_rate"] = None
    result = check_nulls(df)
    assert result["passed"] is False
    assert result["total_nulls"] == 1


def test_nulls_threshold_allows_small_null_rate():
    df = make_valid_df(n=100)
    df.loc[:2, "interest_rate"] = None  # 3% null rate
    result = check_nulls(df, threshold=0.05)  # allow up to 5%
    assert result["passed"] is True


def test_nulls_threshold_fails_above_limit():
    df = make_valid_df(n=10)
    df.loc[:6, "interest_rate"] = None  # 70% null rate
    result = check_nulls(df, threshold=0.05)
    assert result["passed"] is False
    assert "interest_rate" in result["columns_above_threshold"]


# -- check_ranges -------------------------------------

def test_ranges_passes_on_valid_df():
    result = check_ranges(make_valid_df())
    assert result["passed"] is True
    assert result["violations"] == {}


def test_ranges_fails_on_negative_interest_rate():
    df = make_valid_df()
    df.loc[0, "interest_rate"] = -1.0
    result = check_ranges(df)
    assert result["passed"] is False
    assert "interest_rate" in result["violations"]
    assert result["violations"]["interest_rate"]["count"] == 1


def test_ranges_fails_on_grade_out_of_bounds():
    df = make_valid_df()
    df.loc[0, "grade"] = 9  # valid range is 1-7
    result = check_ranges(df)
    assert result["passed"] is False
    assert "grade" in result["violations"]


def test_ranges_skips_missing_columns():
    df = make_valid_df().drop(columns=["interest_rate"])
    result = check_ranges(df)
    assert "interest_rate" not in result["violations"]


def test_ranges_reports_violation_details():
    df = make_valid_df()
    df.loc[0, "interest_rate"] = 150.0
    result = check_ranges(df)
    v = result["violations"]["interest_rate"]
    assert v["count"] == 1
    assert v["actual_max"] == 150.0


# -- check_duplicates ---------------------------------

def test_duplicates_passes_on_unique_df():
    result = check_duplicates(make_valid_df())
    assert result["passed"] is True
    assert result["duplicate_rows"] == 0


def test_duplicates_fails_on_repeated_rows():
    df = make_valid_df(n=4)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = check_duplicates(df)
    assert result["passed"] is False
    assert result["duplicate_rows"] == 1


def test_duplicates_reports_percentage():
    df = make_valid_df(n=10)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = check_duplicates(df)
    assert result["duplicate_pct"] > 0


# -- check_class_balance ------------------------------

def test_class_balance_passes_in_range():
    result = check_class_balance(make_valid_df())  # 50% default rate
    assert result["passed"] is True


def test_class_balance_fails_all_zeros():
    df = make_valid_df()
    df["default"] = 0
    result = check_class_balance(df)
    assert result["passed"] is False


def test_class_balance_fails_all_ones():
    df = make_valid_df()
    df["default"] = 1
    result = check_class_balance(df)
    assert result["passed"] is False


def test_class_balance_fails_missing_target():
    df = make_valid_df().drop(columns=["default"])
    result = check_class_balance(df)
    assert result["passed"] is False
    assert "error" in result


def test_class_balance_reports_rate():
    df = make_valid_df(n=10)
    df["default"] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10% rate
    result = check_class_balance(df)
    assert result["default_rate"] == 0.1


# -- check_categoricals -------------------------------

def test_categoricals_passes_on_valid_df():
    result = check_categoricals(make_valid_df())
    assert result["passed"] is True


def test_categoricals_fails_on_unexpected_value():
    df = make_valid_df()
    df.loc[0, "default"] = 2  # only 0 and 1 are valid
    result = check_categoricals(df)
    assert result["passed"] is False
    assert "default" in result["violations"]
    assert 2 in result["violations"]["default"]["unexpected_values"]


def test_categoricals_skips_missing_columns():
    df = make_valid_df().drop(columns=["is_joint_app"])
    result = check_categoricals(df)
    assert "is_joint_app" not in result["violations"]


# -- run_validation -----------------------------------

def test_run_validation_passes_on_valid_df(capsys):
    result = run_validation(make_valid_df())
    assert result["overall_passed"] is True


def test_run_validation_returns_all_check_keys(capsys):
    result = run_validation(make_valid_df())
    for key in ["schema", "nulls", "ranges", "duplicates",
                "class_balance", "categoricals", "overall_passed"]:
        assert key in result


def test_run_validation_raises_on_fail():
    df = make_valid_df().drop(columns=["interest_rate"])
    with pytest.raises(ValueError, match="Validation failed"):
        run_validation(df, raise_on_fail=True)


def test_run_validation_does_not_raise_when_flag_false(capsys):
    df = make_valid_df().drop(columns=["interest_rate"])
    result = run_validation(df, raise_on_fail=False)
    assert result["overall_passed"] is False


def test_run_validation_overall_false_if_any_check_fails(capsys):
    df = make_valid_df()
    df.loc[0, "grade"] = 99  # range violation
    result = run_validation(df, raise_on_fail=False)
    assert result["overall_passed"] is False


# -- null_summary -------------------------------------

def test_null_summary_returns_dataframe():
    df = make_valid_df()
    df.loc[0, "interest_rate"] = None
    summary = null_summary(df)
    assert isinstance(summary, pd.DataFrame)
    assert "missing_count" in summary.columns
    assert "percent_missing" in summary.columns


def test_null_summary_correct_count():
    df = make_valid_df(n=10)
    df.loc[0:2, "grade"] = None
    summary = null_summary(df)
    assert summary.loc["grade", "missing_count"] == 3


def test_null_summary_sorted_descending():
    df = make_valid_df(n=10)
    df.loc[0:4, "grade"] = None       # 5 nulls
    df.loc[0:1, "interest_rate"] = None  # 2 nulls
    summary = null_summary(df)
    counts = summary["missing_count"].tolist()
    assert counts == sorted(counts, reverse=True)
