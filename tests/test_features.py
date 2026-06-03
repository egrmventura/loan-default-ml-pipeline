import pandas as pd
import pytest
from loan_default_ml_pipeline.features.build_features import (
    build_features,
    drop_leakage_cols,
    encode_target,
    filter_ambiguous,
    handle_joint_nulls,
    handle_delinq_nulls,
    LEAKAGE_COLS,
)


def make_minimal_df(**overrides):
    """Minimal valid row that survives the full feature pipeline."""
    base = {
        "loan_status": "Fully Paid",
        "paid_total": 1000,
        "paid_principal": 900,
        "paid_interest": 100,
        "paid_late_fees": 0,
        "balance": 0,
        "application_type": "individual",
        "annual_income_joint": None,
        "debt_to_income_joint": None,
        "verification_income_joint": None,
        "months_since_last_delinq": None,
        "months_since_90d_late": None,
        "months_since_last_credit_inquiry": 6.0,
        "grade": "B",
        "emp_length": 5.0,
        "debt_to_income": 15.0,
        "num_accounts_120d_past_due": 0.0,
        "loan_amount": 10000,
        "annual_income": 60000,
        "installment": 300.0,
        "total_credit_utilized": 15000,
        "total_credit_limit": 50000,
        "earliest_credit_line": 2005,
        "homeownership": "RENT",
        "verified_income": "Verified",
        "loan_purpose": "debt_consolidation",
        "initial_listing_status": "w",
        "disbursement_method": "Cash",
        "emp_title": "Engineer",
        "sub_grade": "B2",
        "issue_month": "Jan-2019",
        "state": "CA",
    }
    base.update(overrides)
    return pd.DataFrame([base])


# -- drop_leakage_cols ----------------------------

def test_drop_leakage_cols_removes_all():
    df = make_minimal_df()
    result = drop_leakage_cols(df)
    for col in LEAKAGE_COLS:
        assert col not in result.columns


def test_drop_leakage_cols_ignores_missing():
    df = make_minimal_df().drop(columns=["balance"])
    result = drop_leakage_cols(df)
    assert "paid_total" not in result.columns


# -- encode_target --------------------------------

def test_encode_target_default_statuses():
    for status in ["Charged Off", "Late (31-120 days)", "Late (16-30 days)"]:
        df = make_minimal_df(loan_status=status)
        result = encode_target(df)
        assert result["default"].iloc[0] == 1


def test_encode_target_non_default():
    df = make_minimal_df(loan_status="Fully Paid")
    result = encode_target(df)
    assert result["default"].iloc[0] == 0


# -- filter_ambiguous -----------------------------

def test_filter_ambiguous_drops_loan_status_col():
    df = make_minimal_df()
    df = encode_target(df)
    result = filter_ambiguous(df)
    assert "loan_status" not in result.columns


def test_filter_ambiguous_removes_unknown_status():
    df = make_minimal_df(loan_status="Current")
    df = encode_target(df)
    result = filter_ambiguous(df)
    assert len(result) == 0


# -- handle_joint_nulls ---------------------------

def test_handle_joint_nulls_adds_flag():
    df = make_minimal_df(application_type="individual")
    result = handle_joint_nulls(df)
    assert "is_joint_app" in result.columns
    assert result["is_joint_app"].iloc[0] == 0


def test_handle_joint_nulls_fills_nulls():
    df = make_minimal_df()
    result = handle_joint_nulls(df)
    assert result["annual_income_joint"].iloc[0] == 0


# -- handle_delinq_nulls --------------------------

def test_handle_delinq_nulls_sentinel_fill():
    df = make_minimal_df(months_since_last_delinq=None)
    result = handle_delinq_nulls(df)
    assert result["months_since_last_delinq"].iloc[0] == 999


def test_handle_delinq_nulls_ever_delinquent_flag():
    df_no = make_minimal_df(months_since_last_delinq=None)
    df_yes = make_minimal_df(months_since_last_delinq=12.0)
    assert handle_delinq_nulls(df_no)["ever_delinquent"].iloc[0] == 0
    assert handle_delinq_nulls(df_yes)["ever_delinquent"].iloc[0] == 1


# -- build_features (integration) -----------------

def test_build_features_no_nulls():
    df = make_minimal_df()
    result = build_features(df)
    assert result.isnull().sum().sum() == 0


def test_build_features_has_target_col():
    df = make_minimal_df()
    result = build_features(df)
    assert "default" in result.columns


def test_build_features_does_not_mutate_input():
    df = make_minimal_df()
    original_cols = set(df.columns)
    build_features(df)
    assert set(df.columns) == original_cols


def test_build_features_filters_ambiguous_status():
    df = pd.concat([
        make_minimal_df(loan_status="Fully Paid"),
        make_minimal_df(loan_status="Current"),  # should be dropped
    ], ignore_index=True)
    result = build_features(df)
    assert len(result) == 1


# -- data_quality ---------------------------------

def test_null_summary_counts_correctly():
    from loan_default_ml_pipeline.validation.data_quality import null_summary
    df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
    summary = null_summary(df)
    assert summary.loc["a", "missing_count"] == 2
    assert summary.loc["b", "missing_count"] == 0


@pytest.mark.parametrize("col", ["missing_count", "percent_missing", "dtype"])
def test_null_summary_has_expected_columns(col):
    from loan_default_ml_pipeline.validation.data_quality import null_summary
    df = pd.DataFrame({"x": [1, 2, None]})
    assert col in null_summary(df).columns
