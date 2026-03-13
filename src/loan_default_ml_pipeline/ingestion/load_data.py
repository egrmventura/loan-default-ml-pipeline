import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/loans_full_schema.csv")

def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df

if __name__ == "__main__":
    df = load_raw_data()

    print("Rows: ", len(df))
    print("Columns: ", df.columns.tolist())
    print(df.isnull().sum())
    print(df.head())
