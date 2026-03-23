import pandas as pd

# -- Constants ----------------------------------
LEAKAGE_COLS = [
    "paid_total",
    "paid_principal",
    "paid_interest",
    "paid_late_fees",
    "balance"
]

DEFAULT_STATUSES = [
    "Charged Off",
    "Late (31-120 days)",
    "Late (16-30 days)"
]

KEEP_STATUSES = [
    "Charged Off",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Fully Paid"
]

# -- Step 1: Drop leakage columns -----------------
def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=LEAKAGE_COLS, errors="ignore")

# -- Step 2: Encode target label ------------------
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    # binary bool testing if loan_status is deemed default
    return df

# -- Step 3: Filter ambiguous rows ----------------
def filter_ambiguous(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["loan_status"].isin(KEEP_STATUSES)]
    return df.drop(columns=["loan_status"])

# -- Step 4: Handle joint application nulls -------
def handle_joint_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df["is_joint_app"] = (df["application_type"] == "joint").astype(int)
    df["annual_income_joint"] = df["annual_income_joint"].fillna(0)
    df["debt_to_income_joint"] = df["debt_to_income_joint"].fillna(0)
    df["verification_income_joint"] = df["verification_income_joint"].fillna(0)
    return df