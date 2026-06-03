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

---

## Experiment 5 — XGBoost + New Engineered Features

**Date:** 2026-05-22

**Change from Experiment 4:** Added three engineered features to `engineer_ratio_features()`:
- `installment_to_income` = `installment / annual_income`
- `credit_utilization_rate` = `total_credit_utilized / total_credit_limit`
- `credit_age` = `2024 - earliest_credit_line`

**Model:** Same XGBClassifier as Experiment 4. Feature count: 65 (vs 62).

**Results:**
- AUC-ROC: 0.6092 (vs 0.6217 Exp 4)
- Precision (Default class): 0.35 (vs 0.41)
- Recall (Default class): 0.33 (vs 0.33)
- F1 (Default class): 0.34 (vs 0.37)

**Confusion matrix (test set, n=125):**
```
                      Predicted Not Default  Predicted Default
Actual Not Default          67                    22
Actual Default              24                    12
```

**Conclusion:** New features did not improve performance — AUC-ROC and F1 declined slightly while recall held flat at 0.33. The model caught the same 12 defaults but generated more false positives (22 vs 17). The features are domain-valid and kept in the pipeline. The pattern across Experiments 3–5 is consistent: adding features to a 625-row dataset with XGBoost produces marginal or negative returns. The bottleneck is data volume, not feature set breadth. Next focus should be either hyperparameter tuning on the current feature set or investigating whether the dataset can be expanded.

---

## Experiment 6 — XGBoost Hyperparameter Tuning

**Date:** 2026-05-22

**Method:** `RandomizedSearchCV` — 50 iterations, 5-fold stratified CV, scoring=`roc_auc`.

**Search space:**
```
n_estimators      : [100, 200, 300, 500]
max_depth         : [3, 4, 5, 6]
learning_rate     : [0.01, 0.05, 0.1, 0.2]
subsample         : [0.7, 0.8, 1.0]
colsample_bytree  : [0.7, 0.8, 1.0]
min_child_weight  : [1, 3, 5]
```

**Best params found:**
```
n_estimators=100, max_depth=4, learning_rate=0.2,
subsample=0.8, colsample_bytree=1.0, min_child_weight=3
```

**Best CV AUC-ROC:** 0.6810 (cross-validated on train set)

**Test set results:**
- AUC-ROC: 0.6092 (vs 0.6092 Exp 5 — identical)
- Precision (Default class): 0.39 (vs 0.35)
- Recall (Default class): 0.33 (vs 0.33)
- F1 (Default class): 0.36 (vs 0.34)

**Confusion matrix (test set, n=125):**
```
                      Predicted Not Default  Predicted Default
Actual Not Default          70                    19
Actual Default              24                    12
```

**Conclusion:** CV AUC-ROC improved meaningfully (0.62 → 0.68), but test set performance is unchanged — same 12 defaults caught, AUC-ROC identical at 0.6092. The gap between CV score (0.68) and test score (0.61) indicates overfitting to the small training set even with regularization. Tuning has extracted what's available from this dataset. The ceiling has been reached with this data volume and approach.

**Key takeaway across all 6 experiments:** Every technique applied — filter changes, feature engineering, model selection, hyperparameter tuning — has produced marginal movement on test metrics. The hard constraint is 625 rows with 178 defaults. No modeling technique overcomes insufficient data. The next meaningful step is either sourcing a larger dataset or implementing cross-validated evaluation as the primary metric rather than a fixed holdout.

---

## Experiment 7 — XGBoost on LendingClub Full Dataset (2.2M rows)

**Date:** 2026-05-26

**Branch:** `feature/large-dataset-test`

**Change:** Replaced OpenIntro dataset (10k rows, 558–625 usable) with LendingClub full historical dataset (2.26M rows, 1.38M usable after filtering). Pipeline extended with:
- `normalize_schema()` in `load_data.py` — maps 50+ Kaggle column names to pipeline-expected names, coerces `term`, `earliest_credit_line`, `emp_length`
- `PIPELINE_COLS` selection — drops ~100 unmapped Kaggle columns before feature engineering
- `impute_remaining_nulls()` — final step in `build_features` handling sparse bureau fields in pre-2012 records (fill 0 for count cols, median for ratio cols)
- `fico_score` added as new feature (average of `fico_range_high` / `fico_range_low`)
- Two new `loan_status` values handled: `"Default"` and `"Does not meet the credit policy. Status:Charged Off"` → default=1

**Dataset:** 1,382,351 rows — 303,612 defaults (22.0%) / 1,078,739 non-defaults. Test set: 276,471 rows.

**Model:** Same XGBoost defaults as Experiment 4 (`n_estimators=200, max_depth=4, learning_rate=0.1, scale_pos_weight=neg/pos`). Training time: 5.6s.

**Results:**
- AUC-ROC: **0.7305** (vs 0.6092 best on small dataset — +12 points)
- Precision (Default class): 0.36
- Recall (Default class): **0.68** (vs 0.33 best on small dataset)
- F1 (Default class): **0.47** (vs 0.37 best on small dataset)

**Confusion matrix (test set, n=276,471):**
```
                      Predicted Not Default  Predicted Default
Actual Not Default       141,773               73,975
Actual Default            19,474               41,249
```

**Conclusion:** Data volume was the constraint all along. AUC-ROC crossed 0.73, recall on defaults more than doubled (0.33 → 0.68), F1 went from 0.37 → 0.47. The model now catches 68% of actual defaults — a usable signal. False positives are high (74k non-defaults flagged) but that is the expected cost of high recall on a 22% minority class. Merge criteria from `data_notes.md` (AUC-ROC > 0.70, recall > 0.40) are both met. Branch is ready to merge.

---

## Experiment 4 — XGBoost Baseline

**Date:** 2026-05-22

**Change from Experiment 3:** Swapped RandomForest for XGBClassifier. `scale_pos_weight = neg/pos (~2.5)` replaces `class_weight="balanced"` for imbalance handling.

**Model:** `XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, scale_pos_weight=2.5, random_state=42)`

**Feature set:** Same 62 features including `loan_income_ratio`. Artifact saved to `models/xgb_loan_default.pkl`.

**Results:**
- AUC-ROC: 0.6217 (vs 0.6072 RF Exp 3)
- Precision (Default class): 0.41 (vs 0.43)
- Recall (Default class): 0.33 (vs 0.08)
- F1 (Default class): 0.37 (vs 0.14)

**Confusion matrix (test set, n=125):**
```
                      Predicted Not Default  Predicted Default
Actual Not Default          72                    17
Actual Default              24                    12
```

**Conclusion:** XGBoost is the best model so far on the metric that matters most — recall on the Default class jumped from 0.08 to 0.33, catching 12 of 36 defaults vs 3 with RF. F1 nearly tripled (0.14 → 0.37). The tradeoff is more false positives (17 non-defaults flagged as default vs 4), which is an acceptable cost in a default prediction context where missing a real default is more expensive than a false alarm. AUC-ROC is comparable to RF (0.62 vs 0.61). XGBoost is now the working model. Next focus: feature engineering to push recall higher.
