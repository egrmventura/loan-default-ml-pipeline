FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/xgb_loan_default.pkl models/xgb_loan_default.pkl

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "loan_default_ml_pipeline.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
