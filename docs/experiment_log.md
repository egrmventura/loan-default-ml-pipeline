# Experiment Log

Tracks model experiments: what was tried, what metrics resulted, and what was concluded.

Format per entry: model, features used, key metrics, and the decision that followed.

---

## Experiment 1 — Random Forest Baseline

**Date:** 2026-05

**Model:** `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)`

**Feature set:** Full pipeline output from `build_features()` — 61 features after encoding

**Train/test split:** 80/20, stratified on `default` to preserve ~4:1 class ratio

**Why Random Forest:** Chosen as the baseline because it handles tabular data well without heavy tuning, is robust to the mix of ordinal, binary, and one-hot features in this dataset, and provides feature importances for later analysis. `class_weight="balanced"` addresses the 4:1 imbalance without resampling.

**Results:**
- AUC-ROC: 0.6298
- Precision (Default class): 0.50
- Recall (Default class): 0.05
- F1 (Default class): 0.08

**Confusion matrix (test set, n=112):**
```
                      Predicted Not Default  Predicted Default
Actual Not Default          89                     1
Actual Default              21                     1
```

**Conclusion:** Model failed at its primary job. AUC-ROC of 0.63 is above random but well below the 0.75 target. The critical failure is recall on the Default class — the model identified only 1 of 22 actual defaults (recall: 0.05). It is essentially predicting "not default" for almost every loan. `class_weight="balanced"` was not enough to overcome the combination of class imbalance and a 558-row training set.

**Root cause hypothesis:** The working dataset is too small (558 rows, ~112 in test, only 22 defaults in test). The model has too few default examples to learn a reliable decision boundary. This is a data problem before it is a modeling problem.

**Notes:**
- Model artifact saved to `models/rf_loan_default.pkl`
- Working dataset is ~558 rows after filtering ambiguous loan statuses — small sample is the primary constraint

**Next experiment candidates:**
- XGBoost with same feature set (compare AUC-ROC)
- Add `loan_income_ratio` engineered feature (`loan_amnt / annual_income`) — defaults showed ~2.3x higher ratio in EDA
- Logistic Regression as interpretability baseline

---

## Experiment 2 — Random Forest + In Grace Period as default=1

**Date:** 2026-05-22

**Change from Experiment 1:** Added `In Grace Period` to `DEFAULT_STATUSES` and `KEEP_STATUSES` in `build_features.py`.

**Rationale:** In Grace Period loans are 1–15 days past due — consistent with the existing treatment of Late (16–30 days) as default. Adds 67 rows and improves class balance from 4:1 to 2.5:1. More importantly, increases test set defaults from 22 → 36, making evaluation metrics more reliable.

**Model:** Same — `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)`

**Feature set:** Same — 61 features from `build_features()`

**Dataset:** 625 rows (vs 558), default rate 28.5% (vs 19.9%)

**Train/test split:** 500 train / 125 test — 36 defaults in test (vs 22)

**Results:**
- AUC-ROC: 0.6309 (vs 0.6298)
- Precision (Default class): 0.56 (vs 0.50)
- Recall (Default class): 0.14 (vs 0.05)
- F1 (Default class): 0.22 (vs 0.08)

**Confusion matrix (test set, n=125):**
```
                      Predicted Not Default  Predicted Default
Actual Not Default          85                     4
Actual Default              31                     5
```

**Conclusion:** Measurable improvement on the default class — recall jumped from 0.05 to 0.14, F1 from 0.08 to 0.22. AUC-ROC essentially unchanged (0.6298 → 0.6309). The model is still missing most defaults (31 of 36), but the direction is correct. The dataset is still the primary constraint. Next step is feature engineering — add `loan_income_ratio` and compare before moving to a different model class.

---

## Experiment 3 — Random Forest + loan_income_ratio

**Date:** 2026-05-22

**Change from Experiment 2:** Added `engineer_ratio_features()` step to `build_features()` pipeline, computing `loan_income_ratio = loan_amount / annual_income` (zero incomes median-filled).

**Rationale:** EDA noted defaults had ~2.3x higher loan-to-income ratio. Domain-valid affordability signal.

**Model:** Same — `RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)`

**Feature set:** 62 features (61 + `loan_income_ratio`)

**Results:**
- AUC-ROC: 0.6072 (vs 0.6309)
- Precision (Default class): 0.43 (vs 0.56)
- Recall (Default class): 0.08 (vs 0.14)
- F1 (Default class): 0.14 (vs 0.22)

**Confusion matrix (test set, n=125):**
```
                      Predicted Not Default  Predicted Default
Actual Not Default          85                     4
Actual Default              33                     3
```

**Conclusion:** Feature did not improve performance — all metrics declined slightly. With 625 rows, the Random Forest lacks the sample density to exploit a new continuous feature reliably. The signal may be real but the dataset is too small to surface it. Feature kept in the pipeline (domain-valid); failure is a data volume problem, not a feature logic problem.

**Next step:** Try XGBoost with the same feature set — gradient boosting handles small tabular datasets better than Random Forest and may extract more signal from `loan_income_ratio`.
