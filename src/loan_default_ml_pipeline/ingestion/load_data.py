import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data/raw/loans_full_schema.csv"
LARGE_DATA_PATH = (
    PROJECT_ROOT / "data/raw/accepted_2007_to_2018Q4.csv"
)

# Maps LendingClub original column names → pipeline-expected names
COLUMN_MAP = {
    "loan_amnt": "loan_amount",
    "int_rate": "interest_rate",
    "annual_inc": "annual_income",
    "home_ownership": "homeownership",
    "dti": "debt_to_income",
    "verification_status": "verified_income",
    "purpose": "loan_purpose",
    "addr_state": "state",
    "delinq_2yrs": "delinq_2y",
    "mths_since_last_delinq": "months_since_last_delinq",
    "mths_since_last_major_derog": "months_since_90d_late",
    "inq_last_12m": "inquiries_last_12m",
    "total_acc": "total_credit_lines",
    "open_acc": "open_credit_lines",
    "tot_hi_cred_lim": "total_credit_limit",
    "revol_bal": "total_credit_utilized",
    "collections_12_mths_ex_med": "num_collections_last_12m",
    "pub_rec": "num_historical_failed_to_pay",
    "acc_now_delinq": "current_accounts_delinq",
    "tot_coll_amt": "total_collection_amount_ever",
    "open_act_il": "current_installment_accounts",
    "acc_open_past_24mths": "accounts_opened_24m",
    "mths_since_recent_inq": "months_since_last_credit_inquiry",
    "num_sats": "num_satisfactory_accounts",
    "num_accts_ever_120_pd": "num_accounts_120d_past_due",
    "num_tl_30dpd": "num_accounts_30d_past_due",
    "num_actv_bc_tl": "num_active_debit_accounts",
    "total_bal_ex_mort": "total_debit_limit",
    "num_bc_tl": "num_total_cc_accounts",
    "num_bc_sats": "num_open_cc_accounts",
    "num_rev_tl_bal_gt_0": "num_cc_carrying_balance",
    "mort_acc": "num_mort_accounts",
    "pct_tl_nvr_dlq": "account_never_delinq_percent",
    "pub_rec_bankruptcies": "public_record_bankrupt",
    "initial_list_status": "initial_listing_status",
    "annual_inc_joint": "annual_income_joint",
    "verification_status_joint": "verification_income_joint",
    "dti_joint": "debt_to_income_joint",
    "earliest_cr_line": "earliest_credit_line",
    "issue_d": "issue_month",
    # Leakage columns — kept so drop_leakage_cols can remove them
    "total_pymnt": "paid_total",
    "total_rec_prncp": "paid_principal",
    "total_rec_int": "paid_interest",
    "total_rec_late_fee": "paid_late_fees",
    "out_prncp": "balance",
}


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map LendingClub original column names to pipeline-expected names
    and coerce types that differ between the two datasets."""
    df = df.rename(columns=COLUMN_MAP)

    # term: " 36 months" → 36 (int)
    if df["term"].dtype == object:
        df["term"] = df["term"].str.extract(r"(\d+)").astype(float)

    # earliest_cr_line: "Aug-2003" → 2003 (int)
    if df["earliest_credit_line"].dtype == object:
        df["earliest_credit_line"] = pd.to_datetime(
            df["earliest_credit_line"], format="%b-%Y"
        ).dt.year

    # emp_length: "10+ years" / "3 years" / "< 1 year" → numeric float
    if df["emp_length"].dtype == object:
        df["emp_length"] = (
            df["emp_length"]
            .str.replace("< 1", "0", regex=False)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )

    # fico_score: average of high/low bands — strong default signal
    # not present in OpenIntro dataset; added here from Kaggle source
    if "fico_range_high" in df.columns and "fico_range_low" in df.columns:
        df["fico_score"] = (
            df["fico_range_high"] + df["fico_range_low"]
        ) / 2

    return df


# Columns the feature pipeline explicitly uses — everything else dropped
PIPELINE_COLS = [
    "loan_status", "loan_amount", "interest_rate", "term", "installment",
    "grade", "sub_grade", "emp_title", "emp_length", "homeownership",
    "annual_income", "verified_income", "debt_to_income",
    "annual_income_joint", "verification_income_joint", "debt_to_income_joint",
    "application_type", "loan_purpose", "issue_month", "state",
    "delinq_2y", "earliest_credit_line", "inquiries_last_12m",
    "total_credit_lines", "open_credit_lines", "total_credit_limit",
    "total_credit_utilized", "num_collections_last_12m",
    "num_historical_failed_to_pay", "months_since_last_delinq",
    "months_since_90d_late", "current_accounts_delinq",
    "total_collection_amount_ever", "current_installment_accounts",
    "accounts_opened_24m", "months_since_last_credit_inquiry",
    "num_satisfactory_accounts", "num_accounts_120d_past_due",
    "num_accounts_30d_past_due", "num_active_debit_accounts",
    "total_debit_limit", "num_total_cc_accounts", "num_open_cc_accounts",
    "num_cc_carrying_balance", "num_mort_accounts",
    "account_never_delinq_percent", "tax_liens", "public_record_bankrupt",
    "initial_listing_status", "disbursement_method",
    # Leakage cols — kept so drop_leakage_cols can remove them
    "paid_total", "paid_principal", "paid_interest", "paid_late_fees",
    "balance",
    # New from Kaggle — not in OpenIntro
    "fico_score",
]


def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df


def load_large_data():
    df = pd.read_csv(LARGE_DATA_PATH, low_memory=False)
    df = normalize_schema(df)
    keep = [c for c in PIPELINE_COLS if c in df.columns]
    return df[keep]


if __name__ == "__main__":
    df = load_raw_data()
