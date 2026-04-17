"""
Generate corrupted versions of iris and titanic datasets for evaluation.

Inserts known errors so the detection → solutions pipeline has clear targets:
  - Missing values
  - Duplicate rows
  - Outliers
  - Mixed types
  - Zero-variance column (titanic only)

Run: python data/generate_corrupted.py
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(__file__)
rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Iris corrupted
# ---------------------------------------------------------------------------

def make_iris_corrupted():
    src = os.path.join(DATA_DIR, "iris.csv")
    if not os.path.exists(src):
        print(f"  ✗ {src} not found — skipping iris_corrupted")
        return

    df = pd.read_csv(src)

    # 15 random rows → missing petal_length and sepal_width
    missing_idx = rng.choice(len(df), size=15, replace=False)
    df.loc[missing_idx[:8],  "petal_length"] = np.nan
    df.loc[missing_idx[8:],  "sepal_width"]  = np.nan

    # 5 duplicate rows appended
    df = pd.concat([df, df.iloc[:5].copy()], ignore_index=True)

    # 3 extreme outliers in sepal_length
    df.loc[10, "sepal_length"] = 99.0
    df.loc[50, "sepal_length"] = 99.0
    df.loc[100, "sepal_length"] = 99.0

    # 2 mixed-type entries in species (numeric string instead of label)
    df.loc[20, "species"] = "1"
    df.loc[40, "species"] = "2"

    out = os.path.join(DATA_DIR, "iris_corrupted.csv")
    df.to_csv(out, index=False)
    print(f"  ✓ iris_corrupted.csv — {len(df)} rows, errors: 15 missing, 5 dupes, 3 outliers, 2 mixed-type")


# ---------------------------------------------------------------------------
# Titanic corrupted
# ---------------------------------------------------------------------------

def make_titanic_corrupted():
    src = os.path.join(DATA_DIR, "..", "Titanic-Dataset.csv")
    if not os.path.exists(src):
        print(f"  ✗ {src} not found — skipping titanic_corrupted")
        return

    df = pd.read_csv(src)

    # 10 duplicate rows appended
    df = pd.concat([df, df.iloc[:10].copy()], ignore_index=True)

    # 5 extreme Fare outliers
    for idx in [5, 20, 50, 100, 200]:
        df.loc[idx, "Fare"] = 5000.0

    # 30 additional missing Fare values
    fare_missing_idx = rng.choice(len(df), size=30, replace=False)
    df.loc[fare_missing_idx, "Fare"] = np.nan

    # Zero-variance column → triggers consistency penalty
    df["DataSource"] = "titanic"

    out = os.path.join(DATA_DIR, "titanic_corrupted.csv")
    df.to_csv(out, index=False)
    print(f"  ✓ titanic_corrupted.csv — {len(df)} rows, errors: 10 dupes, 5 Fare outliers, 30 missing Fare, zero-variance col")


if __name__ == "__main__":
    print("Generating corrupted evaluation datasets...")
    make_iris_corrupted()
    make_titanic_corrupted()
    print("Done.")
