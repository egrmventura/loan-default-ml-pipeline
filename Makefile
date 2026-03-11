.PHONY: setup train run-api lint test

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt    

train:
	. venv/bin/activate && python src/training/train_model.py

run-api:
	. venv/bin/activate && uvicorn src.inference.api:app --reload

lint:
	. venv/bin/activate && flake8 src/

test:
	. venv/bin/activate && pytest tests/