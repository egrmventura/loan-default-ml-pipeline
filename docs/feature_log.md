# Feature Log

Tracks feature decisions: what was built, why, and what signal it carries.
Each entry should answer: what is it, why does it matter for default prediction, and what did we observe?

---

## Leakage Columns — Dropped

**Columns:** `paid_total`, `paid_principal`, `paid_interest`, `paid_late_fees`, `balance`

**Why dropped:** These are post-outcome fields — they reflect what happened *after* a loan defaulted or was paid off. Including them would let the model "see the future" during training, producing artificially perfect accuracy that collapses to zero at inference time when these values are unknown.

**Decision:** Drop unconditionally before any other step.

---

## loan_status — Filtered and Encoded

**Target:** `default = 1` for `Charged Off`, `Late (31-120 days)`, `Late (16-30 days)`. `default = 0` for `Fully Paid`.

**Dropped statuses:** `Current`, `In Grace Period` — outcome is unknown or ambiguous. These represent >93% of the raw dataset and cannot be labeled reliably.

**Implication:** After filtering, the working dataset shrinks to ~558 rows from 10,000. The model trains on completed loans only.

---

## grade — Ordinal Encoded

**Encoding:** A=1, B=2, C=3, D=4, E=5, F=6, G=7

**Why ordinal:** LendingClub grade has a natural risk order — A is lowest risk, G is highest. One-hot encoding would discard that ordering. The numeric scale preserves it.

---

## is_joint_app — Engineered Flag

**Source:** `application_type == "joint"`

**Why:** Joint applications have different risk profiles and their own income/DTI fields. The flag lets the model use application type as a binary signal without treating the raw categorical as a dummy.

---

## Joint Application Null Fills

**Columns:** `annual_income_joint`, `debt_to_income_joint`, `verification_income_joint`

**Fill value:** 0

**Why:** Nulls in these columns mean the loan was not a joint application. Zero is semantically correct — there is no joint income or joint debt to report. Filling with median would be wrong.

---

## ever_delinquent / ever_90d_late — Engineered Flags

**Source:** `months_since_last_delinq`, `months_since_90d_late`

**Why:** Nulls in these columns mean the borrower has *never* been delinquent — which is informative signal, not missing data. Capturing the null as a binary flag before sentinel-filling preserves that meaning.

**Sentinel fill:** 999 for all `months_since_*` columns — a value outside any realistic range, signaling "never occurred."

---

## High-Cardinality Columns — Dropped

**Columns:** `emp_title`, `sub_grade`, `issue_month`, `state`

**Why dropped:**
- `emp_title`: free text, thousands of unique values, no reliable encoding
- `sub_grade`: redundant with `grade` at higher granularity than the model needs
- `issue_month`: temporal leakage risk; no causal relationship to default
- `state`: high cardinality with thin per-state samples

---

## installment_to_income — Engineered Ratio Feature

**Formula:** `installment / annual_income`

**Why:** Captures monthly payment burden as a fraction of annual income — more precise than `loan_income_ratio` because it reflects actual cash flow pressure rather than total debt size. A borrower spending 20% of their income on a single loan payment is at higher risk than one spending 2%.

**Zero-income handling:** Same guard as `loan_income_ratio` — zero incomes replaced with NaN before division, then median-filled.

---

## credit_utilization_rate — Engineered Ratio Feature

**Formula:** `total_credit_utilized / total_credit_limit`

**Why:** The raw amounts (`total_credit_utilized`, `total_credit_limit`) are already in the pipeline but the *ratio* captures something different. A borrower at 90% utilization is riskier than one at 15%, regardless of absolute dollar amounts. High utilization signals financial stress.

**Zero-limit handling:** Zero total credit limit replaced with NaN before division, then median-filled.

---

## credit_age — Engineered Feature

**Formula:** `2024 - earliest_credit_line`

**Why:** Length of credit history is a standard credit risk signal — longer history means more data points on borrower behavior and generally lower risk. Shorter history means less track record.

---

## loan_income_ratio — Engineered Ratio Feature

**Formula:** `loan_amount / annual_income`

**Why:** Captures affordability relative to income — a borrower taking on a loan that's a large fraction of their annual income is at higher default risk. EDA noted defaults showed ~2.3x higher ratio than non-defaults.

**Zero-income handling:** 23 rows have `annual_income = 0`. These are replaced with `NaN` before division, then median-filled. Dividing by zero would produce `inf`, which corrupts downstream encoding.

**Experiment result:** Adding this feature did not improve performance — AUC-ROC dropped slightly (0.6309 → 0.6072), recall fell from 0.14 to 0.08. Likely explanation: with only 625 rows, the Random Forest doesn't have enough samples to exploit a new continuous feature reliably. The signal may be real but the dataset is too small to surface it with this model. Worth revisiting with XGBoost or a larger dataset.

**Decision:** Keep in pipeline — the feature is domain-valid and the failure is a data volume issue, not a feature logic issue.

---

## Median Imputation

**Columns:** `emp_length`, `debt_to_income`

**Why median over mean:** Both columns have right-skewed distributions. Median is more robust to outliers and better represents the typical borrower.

**`num_accounts_120d_past_due`:** Filled with 0 — null here means no accounts past due, not missing data.
