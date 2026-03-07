# Model Transparency Pipeline

This project is a terminal-first ML diagnostics pipeline focused on one thing:
understanding what happened during training, not just final metrics.

It produces a structured, step-by-step report for both classification and regression models.

## What The Pipeline Now Covers

`run_pipeline(...)` runs the following stages:

1. **Step 0 - Dataset Health Report**
- Missing values by feature with severity levels
- Class imbalance / target skew checks
- Constant and low-variance feature detection
- Multicollinearity scan (feature-feature correlation)
- Feature leakage risk scan (very high feature-target correlation)
- Dataset size and samples-per-feature risk checks
- Duplicate row detection
- Outlier detection (z-score based)
- Final health scorecard with critical/warning issues and suggested fixes

2. **Step 1 - Data Analysis**
- Dataset shape and feature summary
- Target distribution details
- Feature stats (mean/std/min/max/NaNs)
- Top target correlations for regression

3. **Step 2 - Preprocessing**
- Train/test split details
- Optional `StandardScaler` diagnostics (means/stds, before/after sample values)
- Clear leakage-safe scaling note (fit on train only)

4. **Step 3 - Training**
- Model class and key parameters
- Fit timing

5. **Step 3.5 - Training Decision Explanations**
- Explains why splits/weights were chosen during learning
- Tree families: split-quality rationale
- Linear families: coefficient and regularization behavior
- Dedicated paths for XGBoost and LightGBM when installed

6. **Step 4 - Model Internals Deep Dive**
- Model-specific internals for linear, tree, forest, GBM, SVM, KNN, XGBoost, LightGBM

7. **Step 5 - Prediction Walkthrough**
- Sample-by-sample prediction decomposition
- Classification confidence/probabilities when available
- Per-sample regression error details

8. **Step 6 - Evaluation Metrics**
- Classification: accuracy, report, confusion matrix
- Regression: MSE, MAE, RMSE, R2

9. **Step 6.5 - Overfitting Diagnosis Engine**
- Train/test gap-based regime detection:
  - `HEALTHY`
  - `MILD_OVERFIT`
  - `OVERFITTING`
  - `UNDERFITTING`
  - `SUSPICIOUS` (test > train)
- Model-aware cause analysis and targeted fix suggestions
- Mini learning-curve simulation across training fractions
- Complexity-vs-performance summary

10. **Step 7 - Smart Validation**
- Auto-selects validation method by model family and dataset size:
  - Forest/Bagging models: OOB scoring path
  - XGBoost/LightGBM: early-stopping style validation path
  - `< 5,000` samples: full k-fold CV
  - `5,000 to < 50,000`: ShuffleSplit
  - `>= 50,000`: holdout-only path
- Stability verdict based on score variance across splits

11. **Step 8 - Permutation Importance**
- Model-agnostic feature importance with uncertainty estimates

12. **Step 9 - Learning Trace**
- Family-specific deep trace of how learning progressed (splits, coefficients, margins, neighbors, boosting rounds)

## Why This Is Useful

- Catches dataset problems before training starts
- Explains model behavior in human terms, not just scores
- Surfaces overfitting causes and concrete remediation ideas
- Adapts validation strategy to scale for faster but reliable diagnostics
- Makes model review reproducible in plain terminal output

## Repository Structure

- `model_transparency.py`: Main pipeline, model explainers, validation engine, Optuna tuning
- `run_demo.py`: Small XGBoost classification demo

## Supported Models

Deep explainers and/or traces are included for:

- Linear: `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDClassifier`, `SGDRegressor`
- Tree/ensemble: `DecisionTree*`, `RandomForest*`, `GradientBoosting*`
- Margin/distance: `SVC`, `SVR`, `KNeighbors*`
- Optional: `XGBClassifier` / `XGBRegressor`, `LGBMClassifier` / `LGBMRegressor`

The pipeline also accepts other sklearn-compatible estimators; unsupported families fall back to generic diagnostics where needed.

## Installation

Base dependencies:

```bash
pip install numpy pandas scikit-learn
```

Optional dependencies:

```bash
pip install xgboost lightgbm optuna
```

## Quick Start

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_transparency import run_pipeline

X = np.array([
    [1.0, 40, 35, 2],
    [2.0, 55, 50, 3],
    [3.0, 70, 60, 5],
    [5.0, 85, 78, 7],
    [7.0, 92, 88, 9],
    [2.5, 60, 55, 4],
])
y = np.array([0, 0, 0, 1, 1, 0])

model = RandomForestClassifier(n_estimators=100, random_state=42)

trained_model, scaler = run_pipeline(
    model,
    X, y,
    feature_names=["hours_studied", "attendance_pct", "prev_score", "assignments_done"],
    task_type="classification",  # optional, auto-inferred if omitted
    test_size=0.25,
    scale=False,                 # tree models usually do not need scaling
    cv=3,
    n_walkthrough=3,
)
```

## API

### `run_pipeline(...)`

```python
run_pipeline(
    model,
    X,
    y,
    feature_names=None,
    task_type=None,
    test_size=0.2,
    scale=True,
    cv=5,
    n_walkthrough=5,
)
```

Returns:

- fitted `model`
- fitted `scaler` (or `None`)

### `tune_model(...)` (Optuna)

Runs Optuna tuning, prints live trial progress, shows best-trial analysis, then hands the best model to `run_pipeline(...)`.

```python
tune_model(
    model,
    X,
    y,
    task_type=None,
    n_trials=50,
    cv_folds=3,
    test_size=0.2,
    scale=True,
    random_state=42,
)
```

Returns:

- `best_model`
- `scaler`
- `study` (`optuna.study.Study`)

## Demos

Run built-in demos:

```bash
python model_transparency.py
```

Run custom XGBoost demo:

```bash
python run_demo.py
```

`run_demo.py` imports `xgboost` directly, so install `xgboost` first.

## Notes

- Output is intentionally verbose and terminal-centric.
- If `task_type` is omitted, it is inferred automatically.
- `tune_model(...)` requires `optuna`.

## License

No license file is currently included.
Add a `LICENSE` file if you plan to distribute this publicly.
