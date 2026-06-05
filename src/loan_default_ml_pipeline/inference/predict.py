import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models/xgb_loan_default.pkl"
)

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

# One-hot columns the model was trained on — order must match training output
OHE_COLUMNS = [
    "homeownership_OWN",
    "homeownership_RENT",
    "verified_income_Source Verified",
    "verified_income_Verified",
    "loan_purpose_credit_card",
    "loan_purpose_debt_consolidation",
    "loan_purpose_home_improvement",
    "loan_purpose_house",
    "loan_purpose_major_purchase",
    "loan_purpose_medical",
    "loan_purpose_moving",
    "loan_purpose_other",
    "loan_purpose_small_business",
    "loan_purpose_vacation",
    "application_type_joint",
    "verification_income_joint_Not Verified",
    "verification_income_joint_Source Verified",
    "verification_income_joint_Verified",
    "initial_listing_status_whole",
    "disbursement_method_DirectPay",
]

NUMERIC_COLUMNS = [
    "emp_length", "annual_income", "debt_to_income",
    "annual_income_joint", "debt_to_income_joint", "delinq_2y",
    "months_since_last_delinq", "earliest_credit_line",
    "inquiries_last_12m", "total_credit_lines", "open_credit_lines",
    "total_credit_limit", "total_credit_utilized",
    "num_collections_last_12m", "num_historical_failed_to_pay",
    "months_since_90d_late", "current_accounts_delinq",
    "total_collection_amount_ever", "current_installment_accounts",
    "accounts_opened_24m", "months_since_last_credit_inquiry",
    "num_satisfactory_accounts", "num_accounts_120d_past_due",
    "num_accounts_30d_past_due", "num_active_debit_accounts",
    "total_debit_limit", "num_total_cc_accounts", "num_open_cc_accounts",
    "num_cc_carrying_balance", "num_mort_accounts",
    "account_never_delinq_percent", "tax_liens", "public_record_bankrupt",
    "loan_amount", "term", "interest_rate", "installment", "grade",
    "is_joint_app", "ever_delinquent", "ever_90d_late",
    "loan_income_ratio", "installment_to_income",
    "credit_utilization_rate", "credit_age",
]


def _get(inputs: dict, key: str, default):
    """Return inputs[key] if present and not None, else default."""
    v = inputs.get(key)
    return v if v is not None else default


def _load_model():
    return joblib.load(MODEL_PATH)


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Transform raw loan inputs into the model's feature vector."""
    nan = float("nan")
    income = _get(inputs, "annual_income", nan)
    safe_income = income if income and income > 0 else nan

    loan_income_ratio = (
        _get(inputs, "loan_amount", 0) / safe_income
        if safe_income else 0.17
    )
    installment_to_income = (
        _get(inputs, "installment", 0) / safe_income
        if safe_income else 0.06
    )
    credit_limit = _get(inputs, "total_credit_limit", nan)
    credit_utilization_rate = (
        _get(inputs, "total_credit_utilized", 0) / credit_limit
        if credit_limit and credit_limit > 0 else 0.3
    )
    credit_age = 2024 - _get(inputs, "earliest_credit_line", 2009)

    months_since_delinq = inputs.get("months_since_last_delinq")
    months_since_90d = inputs.get("months_since_90d_late")

    row = {
        "emp_length": _get(inputs, "emp_length", 5.0),
        "annual_income": income,
        "debt_to_income": _get(inputs, "debt_to_income", 20.0),
        "annual_income_joint": _get(inputs, "annual_income_joint", 0.0),
        "debt_to_income_joint": _get(inputs, "debt_to_income_joint", 0.0),
        "delinq_2y": _get(inputs, "delinq_2y", 0),
        "months_since_last_delinq": (
            months_since_delinq if months_since_delinq is not None else 999
        ),
        "earliest_credit_line": _get(inputs, "earliest_credit_line", 2009),
        "inquiries_last_12m": _get(inputs, "inquiries_last_12m", 0),
        "total_credit_lines": _get(inputs, "total_credit_lines", 10),
        "open_credit_lines": _get(inputs, "open_credit_lines", 5),
        "total_credit_limit": _get(inputs, "total_credit_limit", 50000),
        "total_credit_utilized": _get(inputs, "total_credit_utilized", 15000),
        "num_collections_last_12m": _get(
            inputs, "num_collections_last_12m", 0),
        "num_historical_failed_to_pay": _get(
            inputs, "num_historical_failed_to_pay", 0),
        "months_since_90d_late": (
            months_since_90d if months_since_90d is not None else 999
        ),
        "current_accounts_delinq": _get(
            inputs, "current_accounts_delinq", 0),
        "total_collection_amount_ever": _get(
            inputs, "total_collection_amount_ever", 0),
        "current_installment_accounts": _get(
            inputs, "current_installment_accounts", 0),
        "accounts_opened_24m": _get(inputs, "accounts_opened_24m", 0),
        "months_since_last_credit_inquiry": _get(
            inputs, "months_since_last_credit_inquiry", 6),
        "num_satisfactory_accounts": _get(
            inputs, "num_satisfactory_accounts", 5),
        "num_accounts_120d_past_due": _get(
            inputs, "num_accounts_120d_past_due", 0),
        "num_accounts_30d_past_due": _get(
            inputs, "num_accounts_30d_past_due", 0),
        "num_active_debit_accounts": _get(
            inputs, "num_active_debit_accounts", 0),
        "total_debit_limit": _get(inputs, "total_debit_limit", 10000),
        "num_total_cc_accounts": _get(inputs, "num_total_cc_accounts", 3),
        "num_open_cc_accounts": _get(inputs, "num_open_cc_accounts", 2),
        "num_cc_carrying_balance": _get(
            inputs, "num_cc_carrying_balance", 1),
        "num_mort_accounts": _get(inputs, "num_mort_accounts", 0),
        "account_never_delinq_percent": _get(
            inputs, "account_never_delinq_percent", 95.0),
        "tax_liens": _get(inputs, "tax_liens", 0),
        "public_record_bankrupt": _get(inputs, "public_record_bankrupt", 0),
        "loan_amount": _get(inputs, "loan_amount", 10000),
        "term": _get(inputs, "term", 36),
        "interest_rate": _get(inputs, "interest_rate", 12.0),
        "installment": _get(inputs, "installment", 300.0),
        "grade": GRADE_MAP.get(_get(inputs, "grade", "C"), 3),
        "is_joint_app": (
            1 if _get(inputs, "application_type", "individual") == "joint"
            else 0
        ),
        "ever_delinquent": 1 if months_since_delinq is not None else 0,
        "ever_90d_late": 1 if months_since_90d is not None else 0,
        "loan_income_ratio": loan_income_ratio,
        "installment_to_income": installment_to_income,
        "credit_utilization_rate": credit_utilization_rate,
        "credit_age": credit_age,
    }

    homeownership = _get(inputs, "homeownership", "RENT").upper()
    row["homeownership_OWN"] = int(homeownership == "OWN")
    row["homeownership_RENT"] = int(homeownership == "RENT")

    verified = _get(inputs, "verified_income", "Not Verified")
    row["verified_income_Source Verified"] = int(
        verified == "Source Verified")
    row["verified_income_Verified"] = int(verified == "Verified")

    purpose = _get(inputs, "loan_purpose", "other")
    for p in [
        "credit_card", "debt_consolidation", "home_improvement",
        "house", "major_purchase", "medical", "moving", "other",
        "small_business", "vacation",
    ]:
        row[f"loan_purpose_{p}"] = int(purpose == p)

    row["application_type_joint"] = int(
        _get(inputs, "application_type", "individual") == "joint"
    )

    verif_joint = _get(inputs, "verification_income_joint", "Not Applicable")
    row["verification_income_joint_Not Verified"] = int(
        verif_joint == "Not Verified")
    row["verification_income_joint_Source Verified"] = int(
        verif_joint == "Source Verified")
    row["verification_income_joint_Verified"] = int(
        verif_joint == "Verified")

    row["initial_listing_status_whole"] = int(
        _get(inputs, "initial_listing_status", "w") == "whole")
    row["disbursement_method_DirectPay"] = int(
        _get(inputs, "disbursement_method", "Cash") == "DirectPay")

    col_order = NUMERIC_COLUMNS + OHE_COLUMNS
    return pd.DataFrame([row])[col_order]


def predict(inputs: dict) -> dict:
    model = _load_model()
    X = build_feature_row(inputs)
    prob = float(model.predict_proba(X)[0][1])
    label = "High" if prob >= 0.5 else "Medium" if prob >= 0.25 else "Low"
    return {
        "default_probability": round(prob, 4),
        "risk_label": label,
    }
