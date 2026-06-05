from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loan_default_ml_pipeline.inference.predict import predict

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts probability of loan default (PD) given loan "
                "application attributes. Part of EL = PD x LGD x EAD.",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    # Core loan terms
    loan_amount: float = Field(..., gt=0, example=10000)
    interest_rate: float = Field(..., gt=0, le=100, example=12.5)
    term: int = Field(..., ge=12, le=84, example=36)
    grade: str = Field(..., pattern="^[A-G]$", example="C")
    installment: float = Field(..., gt=0, example=333.0)

    # Borrower profile
    emp_length: Optional[float] = Field(None, ge=0, le=10, example=5.0)
    annual_income: float = Field(..., gt=0, example=60000)
    debt_to_income: Optional[float] = Field(None, ge=0, example=20.0)
    homeownership: Optional[str] = Field(
        "RENT", example="RENT",
        description="MORTGAGE, OWN, or RENT"
    )
    verified_income: Optional[str] = Field(
        "Not Verified", example="Verified",
        description="Not Verified, Source Verified, or Verified"
    )
    loan_purpose: Optional[str] = Field(
        "other", example="debt_consolidation"
    )

    # Application type
    application_type: Optional[str] = Field(
        "individual", example="individual",
        description="individual or joint"
    )
    annual_income_joint: Optional[float] = Field(None, example=None)
    debt_to_income_joint: Optional[float] = Field(None, example=None)
    verification_income_joint: Optional[str] = Field(None, example=None)

    # Credit history
    earliest_credit_line: Optional[int] = Field(None, example=2005)
    delinq_2y: Optional[int] = Field(None, ge=0, example=0)
    inquiries_last_12m: Optional[int] = Field(None, ge=0, example=1)
    total_credit_lines: Optional[int] = Field(None, ge=0, example=15)
    open_credit_lines: Optional[int] = Field(None, ge=0, example=8)
    total_credit_limit: Optional[float] = Field(None, ge=0, example=80000)
    total_credit_utilized: Optional[float] = Field(None, ge=0, example=20000)
    months_since_last_delinq: Optional[float] = Field(
        None, ge=0, example=None,
        description="Omit if never delinquent"
    )
    months_since_90d_late: Optional[float] = Field(
        None, ge=0, example=None,
        description="Omit if never 90 days late"
    )
    months_since_last_credit_inquiry: Optional[float] = Field(
        None, ge=0, example=6
    )
    public_record_bankrupt: Optional[int] = Field(None, ge=0, example=0)
    tax_liens: Optional[int] = Field(None, ge=0, example=0)
    account_never_delinq_percent: Optional[float] = Field(
        None, ge=0, le=100, example=95.0
    )

    # Optional bureau detail
    num_collections_last_12m: Optional[int] = Field(None, ge=0, example=0)
    num_historical_failed_to_pay: Optional[int] = Field(None, ge=0, example=0)
    current_accounts_delinq: Optional[int] = Field(None, ge=0, example=0)
    total_collection_amount_ever: Optional[float] = Field(
        None, ge=0, example=0)
    current_installment_accounts: Optional[int] = Field(None, ge=0, example=0)
    accounts_opened_24m: Optional[int] = Field(None, ge=0, example=2)
    num_satisfactory_accounts: Optional[int] = Field(None, ge=0, example=8)
    num_accounts_120d_past_due: Optional[int] = Field(None, ge=0, example=0)
    num_accounts_30d_past_due: Optional[int] = Field(None, ge=0, example=0)
    num_active_debit_accounts: Optional[int] = Field(None, ge=0, example=2)
    total_debit_limit: Optional[float] = Field(None, ge=0, example=10000)
    num_total_cc_accounts: Optional[int] = Field(None, ge=0, example=4)
    num_open_cc_accounts: Optional[int] = Field(None, ge=0, example=3)
    num_cc_carrying_balance: Optional[int] = Field(None, ge=0, example=2)
    num_mort_accounts: Optional[int] = Field(None, ge=0, example=0)
    initial_listing_status: Optional[str] = Field(None, example="w")
    disbursement_method: Optional[str] = Field(None, example="Cash")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_label: str
    inputs_received: dict


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgb_loan_default"}


@app.post("/predict", response_model=PredictionResponse)
def predict_default(application: LoanApplication):
    try:
        inputs = application.model_dump(exclude_none=False)
        result = predict(inputs)
        return {
            "default_probability": result["default_probability"],
            "risk_label": result["risk_label"],
            "inputs_received": {
                k: v for k, v in inputs.items() if v is not None
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
