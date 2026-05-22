import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# -- Constants ------------------------------------
MODEL_PATH = Path(__file__).resolve().parents[3] / "models/rf_loan_default.pkl"


# -- Step 1: Load saved model ---------------------
def load_model(path=MODEL_PATH):
    return joblib.load(path)


# -- Step 2: Classification report ----------------
def classifcation_summary(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(
        y_test, y_pred, target_names=["Not Default", "Default"]
    ))
    return y_pred


# -- Step 3: AUC-ROC score ------------------------
def auc_score(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)
    print(f"=== AUC-ROC Score: {score:.4f} ===")
    print("(0.5 = random guessing | 1.0 = perfect | target: > 0.75)")
    return score


# -- Step 4: Confusion matrix ---------------------
def confusion_summary(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("=== Confusion Matrix ===")
    print("                 Predicted Not Default  Predicted Default")
    print(f"Actual Not Default  {cm[0][0]:>6}                {cm[0][1]:>6}")
    print(f"Actual Default      {cm[1][0]:>6}                {cm[1][1]:>6}")
    return cm


# -- Master orchestrator --------------------------
def evaluate_model(model, X_test, y_test):
    print("\n")
    y_pred = classifcation_summary(model, X_test, y_test)
    print("\n")
    auc_score(model, X_test, y_test)
    print("\n")
    confusion_summary(y_test, y_pred)
