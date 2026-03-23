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

JOINT_COLUMNS_INT = [
    "annual_income_joint",
    "debt_to_income_joint",
    "verification_income_joint"
]

MONTHS_SINCE_COLS = [
    "months_since_last_delinq",
    "months_since_90d_late",
    "months_since_last_credit_inquiry"
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
    for c in JOINT_COLUMNS_INT:
        df[c] = df[c].fillna(0)
    return df

# -- Step 5: Handle deliquency nulls --------------
def handle_delinq_nulls(df: pd.DataFrame) -> pd.DataFrame:
    # Capture meaning of null BEFORE filling
    df["ever_delinquent"] = df["months_since_last_delinq"].notna().astype(int)
    df["ever_90d_late"] = df["months_since_90d_late"].notna().astype(int)
    
    # Sentinel fill - null means "never happened"
    for c in MONTHS_SINCE_COLS:
        df[c] = df[c].fillna(999)
    return df

# -- Step 6: Encode catigorical columns -----------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Ordinal - grade has natural order A=best, G=worst
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade"] = df["grade"].map(grade_map)

    # Nominal - cols in no natural order, use one-hot encoding
    nominal_cols = [
        "homeownership",
        "verified_income",
        "loan_purpose",
        "application_type",
        "verification_income_joint",
        "initial_listing_status",
        "disbursement_method"
    ]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # Drop high-cardinality or non-predictive columns
    drop_cols = ["emp_title", "sub_grade", "issue_month", "state"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Handlo remianing nulls with median imputation
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())
    df["debt_to_income"] = df["debt_to_income"].fillna(df["debt_to_income"].median())
    df["num_accounts_120d_past_due"] = df["num_accounts_120d_past_due"].fillna(0)

    return df

# —— Master orchestrator —————————————————————————
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # never mutate the raw DataFrame
    df = drop_leakage_cols(df)
    df = encode_target(df)
    df = filter_ambiguous(df)
    df = handle_joint_nulls(df)
    df = handle_delinq_nulls(df)
    df = encode_categoricals(df)
    return df