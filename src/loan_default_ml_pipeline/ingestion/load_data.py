import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data/raw/loans_full_schema.csv"

def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df

if __name__ == "__main__":
    df = load_raw_data()

    # print("Rows: ", len(df))
    # print("Columns: ", df.columns.tolist())
    # print(df.head(10))
