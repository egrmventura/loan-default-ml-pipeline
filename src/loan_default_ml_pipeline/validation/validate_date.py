import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# —— Constants ———————————————————————————————————————————————
MODEL_PATH = Path(__file__).resolve().parents[3] / "models/rf_loan_default.pkl"

# —— Step 1: Loade saved model ———————————————————————————————
def load_model(path=MODEL_PATH):
    return joblib.load(path)

# —— Step 2: classification report ———————————————————————————
def classifcation_summary(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Not Default", "Default"]))
    return y_pred