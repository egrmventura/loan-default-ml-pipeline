# Loan Default ML Pipeline — Session Summary

## Project Goal

Build a production-style Loan Default Prediction ML Pipeline that demonstrates:

- Data engineering
- ML feature engineering
- MLOps concepts
- Modular Python packaging
- Reproducible workflows
- Model deployment readiness

The repo should resemble a real ML engineering project rather than a Kaggle notebook dump.

---

# Current Repo Direction

## Repository Name

loan-default-ml-pipeline

## High-Level Architecture

raw data
↓
ingestion
↓
validation
↓
feature engineering
↓
training
↓
evaluation
↓
inference/API
↓
monitoring

---

# Current Repo Structure

loan-default-ml-pipeline/

README.md
requirements.txt
Makefile
setup.sh
pyproject.toml
.env.example
.gitignore

data/
    raw/
    processed/

docs/

notebooks/

src/
    loan_default_ml_pipeline/
        __init__.py

        ingestion/
        validation/
        features/
        training/
        inference/
        pipelines/
        config/

tests/

docker/

---

# Python Packaging Setup

## Editable Install

The repo was converted into a proper Python package using:

pip install -e .

This generated:

src/loan_default_ml_pipeline.egg-info/

This is expected behavior for editable installs.

Purpose:
- allows imports directly from package
- avoids PYTHONPATH hacks
- enables module execution

References:
- editable installs create egg-info metadata
- pip/setuptools add package to interpreter path dynamically

Key concept:
Use package imports like:

from loan_default_ml_pipeline.training.train_model import train_model

NOT:

from src.training.train_model import train_model

---

# Makefile Learnings

## Core Problem Encountered

GNU make requires TAB characters before commands.

Spaces caused:

Makefile:4: *** missing separator. Stop.

VS Code was auto-converting tabs to spaces.

## Correct Makefile Pattern

.PHONY: setup train run-api lint test

setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

train:
	./venv/bin/python -m loan_default_ml_pipeline.training.train_model

run-api:
	./venv/bin/uvicorn src.inference.api:app --reload

lint:
	./venv/bin/flake8 src/

test:
	./venv/bin/pytest tests/

## Important Notes

- Use TAB characters, not spaces
- Better to use ./venv/bin/python directly instead of source activate
- VS Code likely auto-converted tabs to spaces

Useful debugging:

cat -vet Makefile

TAB appears as:

^I

---

# Environment / venv Notes

## setup.sh Placement

Place setup.sh at repo root.

## venv Strategy

venv should exist INSIDE repo:

loan-default-ml-pipeline/venv/

Activation:

source venv/bin/activate

No ~/ prefix needed because path is relative to repo root.

## .gitignore

Must include:

venv/
*.egg-info
__pycache__/
.env

---

# .env Strategy

Purpose:
- externalize configs/secrets
- support dev/staging/prod environments

## Example

.env.example

DATA_RAW_PATH=data/raw
MODEL_PATH=models
MLFLOW_TRACKING_URI=http://localhost:5000

## Python Loading

from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv("DATA_RAW_PATH")

Never commit real .env files.

---

# Current Data Loading Code

Current ingestion file:

src/loan_default_ml_pipeline/ingestion/load_data.py

Contains:
- CSV loading
- null validation summaries
- exploratory object column printing

---

# Architectural Refactor Guidance

## ingestion/

Should ONLY load data.

Example:

def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df

No profiling or exploration logic.

---

## validation/

Move data quality checks here.

Suggested file:

src/loan_default_ml_pipeline/validation/data_quality.py

Functions:
- null summaries
- duplicate checks
- schema validation
- type validation
- range checks

Current null_summary logic belongs here.

---

## notebooks/

Purpose:
- EDA
- hypothesis testing
- relationship discovery

NOT:
- production pipeline code

Suggested notebook naming:

01_data_profiling.ipynb
02_feature_exploration.ipynb
03_model_experiments.ipynb

Avoid:
analysis_final_v7.ipynb chaos.

---

# Documentation Strategy (Critical)

Create:

docs/

## Files

docs/data_notes.md
docs/feature_log.md
docs/experiment_log.md

---

## data_notes.md

Tracks:
- schema observations
- missing values
- categorical distributions
- anomalies

---

## feature_log.md

Tracks:
- feature definitions
- business rationale
- observed predictive signal
- decision to keep/remove

Example:

loan_income_ratio
Definition:
loan_amnt / annual_inc

Observation:
Defaults show ~2.3x higher ratio

Decision:
Keep feature

---

## experiment_log.md

Tracks:
- model versions
- metrics
- experiments
- feature additions
- conclusions

Example:

Experiment 1:
Logistic Regression
ROC-AUC: 0.68

Experiment 2:
XGBoost + loan_income_ratio
ROC-AUC: 0.74

---

# Recommended Workflow

Notebook → Discover → Document → Implement → Commit

Example:

1. discover feature in notebook
2. document in feature_log.md
3. move logic into src/features/build_features.py
4. commit clean implementation

---

# Pipeline Direction

Planned pipeline execution:

load_raw_data()
↓
validate_schema()
↓
data_quality_checks()
↓
build_features()
↓
train_model()

---

# Suggested Pipeline Runner

src/loan_default_ml_pipeline/pipelines/data_pipeline.py

Purpose:
- orchestrate stages
- avoid running logic from ingestion modules directly

---

# DataCamp Guidance

Recommended immediate courses:

1. Feature Engineering for Machine Learning in Python
2. ETL and ELT in Python
3. Supervised Learning with scikit-learn

Focus:
- feature engineering
- preprocessing
- ML pipelines

Avoid deep learning/NLP for now.

Priority:
30% course
70% project implementation

---

# Current Strategic Guidance

Key principle:

Exploration ≠ Decisions ≠ Production Code

Notebooks are scratchpads.
Production logic belongs in src/.

Goal:
Transform repo from:
- notebook-heavy toy project

into:
- production-style ML engineering repo

---

# Likely Next Steps

1. Separate validation layer from ingestion
2. Build schema validation module
3. Add feature engineering module
4. Save processed parquet outputs
5. Build baseline model training pipeline
6. Add MLflow experiment tracking
7. Add FastAPI inference endpoint
8. Add Docker support
9. Add CI/CD
10. Add monitoring/drift detection

---

# Important Technical Lessons Learned

## Editable Installs

pip install -e .:
- creates egg-info metadata
- links package into interpreter path
- allows live code changes without reinstall

## Makefile

Commands require literal TAB characters.

## Python Packaging

Run modules using:

python -m loan_default_ml_pipeline.training.train_model

instead of raw script paths.

## Notebooks

Use for:
- exploration only

Do NOT:
- embed production pipeline logic
- maintain feature logic exclusively in notebooks

---

# Current Philosophy

The project should emphasize:

- reproducibility
- modularity
- traceability
- engineering discipline
- production-oriented ML architecture

NOT:
- notebook-only experimentation
- one-off scripts
- ad hoc feature creation
