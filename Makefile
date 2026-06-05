.PHONY: setup data data-large train run-api lint test ingestion_demo

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

data:
	. venv/bin/activate && python -m loan_default_ml_pipeline.pipelines.data_pipeline

data-large:
	. venv/bin/activate && python -m loan_default_ml_pipeline.pipelines.data_pipeline --large

train:
	. venv/bin/activate && python src/loan_default_ml_pipeline/training/train_model.py

run-api:
	. venv/bin/activate && uvicorn loan_default_ml_pipeline.inference.api:app --reload

lint:
	. venv/bin/activate && flake8 src/

test:
	. venv/bin/activate && pytest tests/

ingestion_demo:
	. venv/bin/activate && python src/loan_default_ml_pipeline/ingestion/load_data.py