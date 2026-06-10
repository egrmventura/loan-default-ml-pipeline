import pytest
import mlflow
from loan_default_ml_pipeline.training.train_model import (
    train_xgboost_model,
    PROCESSED_DATA_PATH,
)


@pytest.mark.skipif(
    not PROCESSED_DATA_PATH.exists(),
    reason="processed dataset not built locally",
)
def test_train_xgboost_model_logs_mlflow_run(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test_train_xgboost")

    train_xgboost_model()

    runs = mlflow.search_runs()
    assert not runs.empty
