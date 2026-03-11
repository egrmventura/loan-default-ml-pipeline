setup:
    python3 -m venv venv
    source venv/bin/activate && pip install -r requirements.txt

install:
    pip install -r requirements.txt

train:
    python src/training/train_model.py

run-api:
    uvicorn src.inference.api:app --reload

lint:
    flake8 src/

test:
    pytest tests/