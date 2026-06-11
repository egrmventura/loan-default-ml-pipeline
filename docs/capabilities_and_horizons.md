# Capabilities and Horizons

A snapshot of what this pipeline can do today and a menu of where it could go next.
For *how* the pieces fit together, defer to [`CLAUDE.md`](../CLAUDE.md) (architecture,
module table, feature steps) and the docs under `docs/`. This file synthesizes the
*combination* of pieces and lays out concrete next steps.

Last updated: 2026-06-10.

---

## What This Pipeline Can Do Today

The repo is a working end-to-end PD (probability-of-default) pipeline: raw CSV in, a
served prediction out, with validation, experiment tracking, tests, and CI along the way.
Concretely, the combination of pieces enables the following workflows.

### Build a clean, validated training set from raw data — one command

`make data` (small OpenIntro 10k) or `make data-large` (full LendingClub, 2.2M raw →
1.38M usable) runs `pipelines/data_pipeline.py`: load → `build_features()` (the 8-step
pipeline) → `run_validation()` (GO/NO-GO) → write `data/processed/loans_featured.parquet`.
The validation gate (`validation/data_quality.py`) enforces a real schema contract —
required columns and dtypes, zero nulls, value ranges, no duplicates, class-balance
bounds, categorical domains — and `raise_on_fail=True` means a bad build *stops* rather
than silently producing a corrupt parquet. This is the strongest part of the project and
the one most aligned with its stated goal.

### Train, evaluate, and track an experiment

Two model families are wired up in `training/train_model.py`: RandomForest (`train_model()`,
`class_weight="balanced"`) and XGBoost (`train_xgboost_model()`, `scale_pos_weight=neg/pos`),
plus a 50-iter RandomizedSearchCV tuner (`tune_xgboost_model()`). `train_xgboost_model()`
logs params, the four headline metrics (AUC-ROC / recall / precision / F1), and the model
artifact to MLflow; browse runs with `make mlflow-ui`. `validation/validate_model.py`
produces the classification report, AUC-ROC, and confusion matrix for post-training eval.
Current best (Experiment 7, 1.38M rows): AUC-ROC 0.73, recall 0.68, precision 0.36, F1 0.47.

### Serve predictions, in a container

`make run-api` starts the FastAPI app (`inference/api.py`): `GET /health` and
`POST /predict`, with a fully-typed, validated `LoanApplication` Pydantic schema (~45
fields, most optional with sensible server-side defaults). It loads `xgb_loan_default.pkl`
and returns a default probability plus a `Low/Medium/High` risk label. `make docker-build`
/ `make docker-run` package and serve the same API on `:8000` in a container.

### Develop with a safety net

52 pytest tests across features, validation, and training (`make test`), flake8-clean
(`make lint`), and GitHub Actions (`.github/workflows/ci.yml`) runs both on every push/PR
to `main`. Imports are package-style and the project is `pip install -e .` editable.

### Honest caveats a user/extender should know

These aren't bugs to "fix" so much as known shape-of-the-system facts:

- **`make train` trains the RandomForest, not the production model.** The XGBoost path
  (`train_xgboost_model()`) is what's tracked in MLflow and what the API serves, but it
  isn't the default `make train` target. The tuner (`tune_xgboost_model()`) does **not**
  log to MLflow.
- **Training/serving feature logic is duplicated.** `inference/predict.py` hand-reimplements
  the feature transform (one-hot column order, sentinel fills, ratio features) rather than
  calling `build_features()`. It works, but it's a classic train/serve skew risk: any change
  to `build_features.py` must be mirrored by hand in `predict.py`.
- **Validation covers data, not models or live traffic.** There's a GO/NO-GO gate on the
  training parquet, but no check on model outputs, no drift detection, and no monitoring of
  what the deployed API actually sees.
- **The decision threshold is hardcoded at 0.5** in `predict.py`, never tuned — see below.
- **Data and model artifacts are gitignored.** A fresh clone can't train or serve until the
  dataset is downloaded and built locally (see `CLAUDE.md` → Data Setup).

---

## Research Directions and Future Horizons

Organized by theme. Each item notes what it is, why it's valuable *for this project
specifically*, and rough effort. "Quick win" ≈ an afternoon; "rabbit hole" ≈ multi-session.

### Modeling improvements

- **Threshold tuning (quick win, highest ROI).** Precision is 0.36 at the default 0.5 cut,
  which is hardcoded in `predict.py`. Sweep thresholds against a cost model (a missed default
  costs more than a false alarm — exactly the EL framing) and pick an operating point, or
  expose it as config. Touches: `predict.py`, `validate_model.py`. Directly addresses the
  weakest current metric with almost no new machinery.
- **Probability calibration (quick win).** XGBoost with `scale_pos_weight` produces ranked
  but not well-calibrated scores; the API returns a raw `predict_proba` as if it were a true
  PD. Wrap with `CalibratedClassifierCV` (isotonic/Platt) and add a reliability-curve check.
  Valuable because the whole point is *probability* of default feeding `EL = PD × LGD × EAD` —
  miscalibrated PDs make the EL math meaningless.
- **Explainability / SHAP (medium).** Add SHAP values to the training eval and optionally to
  the `/predict` response. The pipeline already preserves interpretable, domain-meaningful
  features (`loan_income_ratio`, `credit_utilization_rate`, grade) so explanations would be
  legible. Lending is a regulated, adverse-action domain — per-decision reason codes are
  table stakes in the real world. Builds genuine ML-eng skill; moderate effort.
- **Fairness / bias analysis (medium, important).** The raw LendingClub data carries proxies
  for protected attributes (geography via `state`, which is currently dropped; income).
  Slice metrics by subgroup and measure disparate impact. High learning value and high
  real-world relevance for credit; mostly analysis, low infra cost.
- **Toward full expected loss — LGD and EAD (deep rabbit hole, but the marquee direction).**
  Today only PD is modeled. The dropped leakage columns (`paid_total`, `paid_principal`,
  `balance`) are exactly the post-outcome fields needed to *derive* LGD and EAD targets for
  the defaulted population. Building even a crude LGD regressor and an EAD model would let the
  pipeline output a dollar EL, not just a probability — the most differentiated thing this
  project could do. Touches ingestion, features, a new model family, and the API contract.
- **Alternative algorithms / feature interactions (low value here).** Experiments 1–6 showed
  the small dataset was the constraint, not the algorithm; on the large set XGBoost already
  clears the bar. Logistic regression as an interpretable baseline is worth one run; broad
  algorithm sweeps are likely low-yield until calibration/threshold/EL work is done.

### MLOps maturity

- **Unify train/serve feature logic (quick win, do this early).** Refactor `predict.py` to
  call `build_features()` on a single-row frame instead of duplicating the transform. Kills
  the train/serve skew risk noted above and is a prerequisite for trusting any of the
  monitoring work below. Touches `predict.py`, `build_features.py` (may need a single-row
  path), tests.
- **Model registry + consistent tracking (quick win).** Promote MLflow from logging-only to
  the MLflow Model Registry, and wire `make train` / the tuner to log the same way the
  XGBoost path does. Then the API loads "the current Production model" by stage rather than a
  hardcoded `.pkl` path. Small change, big maturity jump.
- **Containerize the training/data pipeline (medium).** Only the inference API is dockerized
  today. A training image (or a `docker-compose` with data → train → serve stages) would make
  the whole "run an experiment and redeploy" loop reproducible off one machine. Stated as a
  known gap in `CLAUDE.md`.
- **Drift detection / monitoring (medium → deep).** Compare live `/predict` inputs and score
  distributions against the training parquet (PSI / KS tests). The validation layer already
  defines feature ranges and schema contracts, so the "expected" reference is half-built —
  extend `data_quality.py` style checks to scoring-time inputs. Pairs naturally with logging
  predictions to a store.
- **CD + retraining triggers (deep).** Extend the existing GitHub Actions CI into CD: build
  and push the image, deploy on green, and trigger retraining on a schedule or a drift alarm.
  Infrastructure-as-code (Terraform) for the target environment is the natural companion.
  Highest infra effort; best done after registry + monitoring exist.
- **Shadow / A-B deployment (deep).** Serve a candidate model alongside production and compare
  on live traffic. Mostly meaningful once there's real traffic and monitoring — park it until
  then.

### Data engineering

- **Data versioning with DVC (quick win).** Artifacts are currently gitignored and rebuilt by
  hand. DVC (or `lakeFS`) would version the raw and processed parquet against git commits, so
  an experiment in `experiment_log.md` is reproducible to the exact bytes it ran on. Low
  effort, high reproducibility payoff — well aligned with the project's discipline goal.
- **Feature store (medium → deep).** Formalize the engineered features (Feast or similar) so
  training and serving read from one definition — a heavier-weight, more principled fix for
  the same train/serve skew the `predict.py` refactor addresses. Worth it mainly as a learning
  exercise; arguably overkill for one model.
- **Streaming ingestion / additional sources (deep).** The current ingestion is batch CSV.
  Simulating a stream of new applications (Kafka, or just a polling loop over partitioned
  files) would exercise online-feature and incremental-validation skills. Additional sources
  (macro indicators, rate environment) could sharpen the model. High effort, more "breadth"
  than "depth" given a static historical dataset.

### Engineering-practice / learning value (this is a learning project)

If picking by *skill built per hour*, a sensible order:

1. **Quick wins that also de-risk everything else:** threshold tuning, calibration, unify
   train/serve feature logic, DVC, model registry. Each is an afternoon and each makes the
   next thing safer or more meaningful.
2. **High learning-value analysis:** SHAP explainability and fairness/bias slicing — strong
   resume/portfolio signal for ML-eng roles in a regulated domain, mostly analysis effort.
3. **The differentiated deep dive:** the LGD/EAD → full expected-loss model. It's the one
   direction that turns a generic PD classifier into a credit-risk *system*, and it touches
   every layer of the pipeline — the best demonstration of the engineering discipline this
   project is explicitly about.
4. **Heavier MLOps (CD, drift, IaC, streaming):** highest infra skill-building, but best
   sequenced after the registry/monitoring foundations exist so the work compounds rather
   than gets thrown away.
