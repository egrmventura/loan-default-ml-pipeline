import pandas as pd

# Tracking null value percentages for data validation
def null_summary(df):
    summary = (
        df.isnull()
        .sum()
        .to_frame("missing_count")
        .assign(
            percent_missing=lambda x: (x["missing_count"] / len(df))* 100,
            dtype=df.dtypes
        )
        .sort_values("missing_count", ascending=False)
        
    )
    return summary