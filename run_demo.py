"""
Housing Price Prediction Pipeline
==================================
Dataset : housing.csv  (California Housing, ~20,640 rows)

          Columns:
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income,
            median_house_value   <- regression target  (auto-detected)
            ocean_proximity      <- categorical string  (triggers CatBoost auto-swap)

  The pipeline infers the target column automatically using name-pattern +
  heuristics — no hard-coded y= needed. "median_house_value" scores a strong
  match on the "value" keyword + float dtype + high cardinality.

Usage:
    python housing_pipeline.py --data /path/to/housing.csv
    python housing_pipeline.py --data ~/repos/my-repo/data/housing.csv
    python housing_pipeline.py --data https://raw.githubusercontent.com/.../housing.csv
    python housing_pipeline.py --data /path/to/housing.csv --sample 3000
    python housing_pipeline.py --data /path/to/housing.csv --raw
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE YOUR PATH HERE  (or pass --data at the command line)
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_DATA_PATH = "D:\\Downloads\\Agents\\Model_Transparency\\Housing.csv"   # <-- change this to your path
# ═══════════════════════════════════════════════════════════════════════════════

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Housing price prediction pipeline")
parser.add_argument(
    "--data", type=str, default=None,
    help=(
        "Path or URL to housing.csv.  Examples:\n"
        "  --data /home/user/datasets/housing.csv\n"
        "  --data ~/repos/my-repo/data/housing.csv\n"
        "  --data https://raw.githubusercontent.com/you/repo/main/housing.csv"
    )
)
parser.add_argument(
    "--raw", action="store_true",
    help="Skip feature engineering — use the raw columns only"
)
parser.add_argument(
    "--sample", type=int, default=None,
    help="Use a random sample of N rows (default: all ~20,640)"
)
args = parser.parse_args()

# ── Resolve data path ──────────────────────────────────────────────────────────
data_path = args.data or DEFAULT_DATA_PATH

print("\n" + "=" * 70)
print("  California Housing — ML Transparency Pipeline")
print("=" * 70 + "\n")
print(f"  Data path : {data_path}")

# ── Load data ──────────────────────────────────────────────────────────────────
print("  Loading ...")
try:
    df = pd.read_csv(data_path)
    print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\n  ERROR: File not found — '{data_path}'")
    print(f"  Set DEFAULT_DATA_PATH at the top of this script, or pass --data <path>")
    sys.exit(1)
except Exception as e:
    print(f"\n  ERROR loading data: {e}")
    sys.exit(1)

# ── Impute missing values ──────────────────────────────────────────────────────
nan_counts = df.isnull().sum()
if nan_counts.any():
    print(f"\n  Missing values detected:")
    for col, n in nan_counts[nan_counts > 0].items():
        median_val = df[col].median()
        df[col]    = df[col].fillna(median_val)
        print(f"    {col:<30}: {n} NaN  -> imputed with median ({median_val:.2f})")

# ── Optional row sampling ──────────────────────────────────────────────────────
if args.sample and args.sample < len(df):
    df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    print(f"\n  Sampled {len(df):,} rows  (--sample {args.sample})")

# ── Feature engineering ────────────────────────────────────────────────────────
if not args.raw and {"total_rooms", "total_bedrooms", "population", "households"}.issubset(df.columns):
    print("\n  Applying feature engineering ...")
    df["rooms_per_household"]      = df["total_rooms"]    / df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"]     / df["households"]
    df = df.drop(columns=["total_rooms", "total_bedrooms", "population"])
    print("    Added   : rooms_per_household, bedrooms_per_room, population_per_household")
    print("    Dropped : total_rooms, total_bedrooms, population  (replaced by ratios)")

# ── Model ──────────────────────────────────────────────────────────────────────
# XGBRegressor is used by default.
# run_pipeline_from_df will auto-swap to CatBoostRegressor if catboost is
# installed, because ocean_proximity is a string categorical column.

try:
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )
    print(f"\n  Supplied model : XGBRegressor")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    print(f"\n  XGBoost not found — using GradientBoostingRegressor as fallback.")
    print(f"  Install with:  pip install xgboost")

print(f"  CatBoost auto-swap active if catboost is installed  (pip install catboost)\n")

# ── Import pipeline ────────────────────────────────────────────────────────────
try:
    from model_transparency import run_pipeline_from_df
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_transparency import run_pipeline_from_df

# ── Run — pass the whole DataFrame, target is auto-detected ───────────────────
#
#   run_pipeline_from_df() will:
#     1. Score every column with name-pattern + heuristics
#     2. Select "median_house_value"  (strong "value" keyword match + float dtype)
#     3. Print a full inference report showing scores for all columns
#     4. Split df into X and y, then run the full transparency pipeline
#
#   To override the inferred target explicitly:
#     run_pipeline_from_df(model, df, target_col="median_house_value", ...)

run_pipeline_from_df(
    model,
    df,                       # full DataFrame — no manual X/y split needed
    task_type     = "regression",
    test_size     = 0.20,
    scale         = True,     # auto-disabled if CatBoost is swapped in
    cv            = 5,
    n_walkthrough = 4,
)