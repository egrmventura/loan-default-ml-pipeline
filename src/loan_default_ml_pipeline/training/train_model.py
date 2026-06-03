import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import joblib

# -- Constants ------------------------------------
PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/processed/loans_featured.parquet"
)
RF_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models/rf_loan_default.pkl"
)
XGB_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models/xgb_loan_default.pkl"
)


# -- Step 1: Load processed data ------------------
def load_processed_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    assert "default" in df.columns, "Target column 'default' missing"
    assert df.isnull().sum().sum() == 0, "Nulls detected in processed data"
    return df


# -- Step 2: Split features and target ------------
def split_data(df: pd.DataFrame):
    X = df.drop("default", axis=1)
    y = df["default"]
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,  # preserve class ratio in both splits
        random_state=42
    )


# -- Step 3a: Train Random Forest -----------------
def train_rf(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",  # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# -- Step 3b: Train XGBoost -----------------------
def train_xgb(X_train, y_train) -> XGBClassifier:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=neg / pos,  # equivalent to class_weight="balanced"
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


# -- Step 4: Save model artifact ------------------
def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


# -- Orchestrators --------------------------------
def train_model():
    print("Loading processed data....")
    df = load_processed_data()
    print(f"Data loaded: {df.shape}")

    print("Splitting data....")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.3f}")
    print(f"Test default rate: {y_test.mean():.3f}")

    print("Training Random Forest....")
    model = train_rf(X_train, y_train)
    print("Training complete.")

    save_model(model, RF_MODEL_PATH)
    return model, X_test, y_test


def train_xgboost_model():
    print("Loading processed data....")
    df = load_processed_data()
    print(f"Data loaded: {df.shape}")

    print("Splitting data....")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.3f}")
    print(f"Test default rate: {y_test.mean():.3f}")

    print("Training XGBoost....")
    model = train_xgb(X_train, y_train)
    print("Training complete.")

    save_model(model, XGB_MODEL_PATH)
    return model, X_test, y_test


# -- Step 3c: Tune XGBoost via random search ------
def tune_xgb(X_train, y_train) -> XGBClassifier:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    base = XGBClassifier(
        scale_pos_weight=neg / pos,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=50,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    print(f"Best CV AUC-ROC : {search.best_score_:.4f}")
    print(f"Best params     : {search.best_params_}")
    return search.best_estimator_


def tune_xgboost_model():
    print("Loading processed data....")
    df = load_processed_data()
    print(f"Data loaded: {df.shape}")

    print("Splitting data....")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    print("Running hyperparameter search (50 iterations, 5-fold CV)....")
    model = tune_xgb(X_train, y_train)

    save_model(model, XGB_MODEL_PATH)
    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
