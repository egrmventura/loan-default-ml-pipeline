import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# –– Constants –––––––––––––––––––––––––––––––––––––
PROCESSED_DATA_PATH = Path(__file__).resolve().parents[3] / "../data/processed/loans_featured.csv"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parents[3] / "../models/rf_loan_default.pkl"
# TODO resolve().parents[3] is a temporary fix to guide to docs

# –– Step 1: Load processed data –––––––––––––––––––
def load_processed_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "default" in df.columns, "Target column 'default' missing"
    assert df.isnull().sum().sum() == 0, "Nulls detected in processed data"
    return df

# –– Step 2: Split features and target –––––––––––––
def split_data(df: pd.DataFrame):
    X = df.drop("default", axis=1)
    y = df["default"]
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y, # preserve 4:1 class ratio in both splits
        random_state=42
    )

# –– Step 3: Train model –––––––––––––––––––––––––––
def train(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced", # handles 4:1 imbalance
        random_state=42,
        n_jobs=-1 # use all available CPU cores
    )
    model.fit(X_train, y_train)
    return model

# –– Step 4: Save model artifact –––––––––––––––––––
def save_model(model, path=MODEL_OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# –– Step 5: Master orchestrator –––––––––––––––––––
def train_model():
    print("Loading processed data....")
    df = load_processed_data()
    print(f"Data loaded: {df.shape}")

    print("Splitting data....")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.3f}")
    print(f"Test default rate: {y_test.mean():.3f}")

    print("Training model....")
    model = train(X_train, y_train)
    print("Training complete.")

    save_model(model)
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()