setup:
    python -m venv venv
    source venv/bin/activate && pip install -r requirements.txt

install:
    pip install -r requirements.txt

train:
   python src/training/train_model.python

run-api:
   uvicorn src.inference.api:app --reload

lint:
   flake8 src/

test:
    pytest tests/