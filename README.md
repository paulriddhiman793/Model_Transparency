# Model Transparency Pipeline

This repository is built to answer one question clearly:

**What happened during model training, and why did the model make these predictions?**

Instead of only giving final metrics, it prints a detailed, step-by-step training narrative in the terminal.

## What This Repo Tells You About Training

When you run `run_pipeline(...)`, you get a structured report across 9 stages:

1. **Data Analysis**
- Dataset shape, feature count, and target profile
- Class balance (for classification) or target distribution stats (for regression)
- Feature summary stats and top correlations with target

2. **Preprocessing**
- Exact train/test split sizes
- Whether scaling was applied
- Per-feature `StandardScaler` means/stds (first features shown)
- Before/after sample values to verify transformations

3. **Training Diagnostics**
- Model class and key hyperparameters
- Fit time
- Quick visibility into what the model was configured to learn

4. **Model Internals (model-specific explainers)**
- Linear/logistic/SGD: coefficients, intercept, equation preview, strongest weights
- Decision tree: node counts, depth, split preview, impurity-based importances
- Random forest: estimator count, depth distribution, OOB stats, aggregated importances
- Gradient boosting: stage-wise loss behavior, learning-rate effects, importances
- SVM: kernel, support vectors, margin intuition
- KNN: neighbor mechanics and distance setup
- XGBoost/LightGBM (if installed): boosted-tree internals and feature influence

5. **Prediction Walkthrough (sample-by-sample)**
- Shows test samples and predicted outputs
- For classification: correctness + class probabilities (if available)
- For regression: per-sample absolute errors

6. **Evaluation Metrics**
- Classification: accuracy, report, confusion matrix
- Regression: MSE, MAE, RMSE, R2
- Train vs test behavior for overfit/underfit signals

7. **Validation Strategy**
- Cross-validation summary
- Adaptive behavior for larger datasets (faster validation strategy)
- Helps judge whether observed test performance is stable

8. **Permutation Feature Importance**
- Global feature influence using permutation on test data
- Works as a model-agnostic check against built-in model importances

9. **Learning Trace (how the model learned)**
- Deeper training-process interpretation by model family
- Examples include split-building logic (trees), round-wise improvements (boosting), coefficient/loss behavior (linear/SGD), neighborhood behavior (KNN), and support vector effects (SVM)

## Why This Is Useful

Use this repo when you need to:

- Debug why a model is overfitting or underperforming
- Explain model behavior to teammates/stakeholders
- Compare model families beyond a single score
- Inspect whether preprocessing and scaling are behaving correctly
- Get interpretable training output without notebooks or dashboards

## Repository Structure

- `model_transparency.py`: Main pipeline, explainers, and optional Optuna tuning
- `run_demo.py`: Minimal example (XGBoost classifier on toy data)

## Supported Models

- Linear: `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDClassifier`, `SGDRegressor`
- Tree/ensemble: `DecisionTree*`, `RandomForest*`, `GradientBoosting*`
- Margin/distance: `SVC`, `SVR`, `KNeighbors*`
- Optional: `XGBClassifier`/`XGBRegressor`, `LGBMClassifier`/`LGBMRegressor`

## Installation

Install base dependencies:

```bash
pip install numpy pandas scikit-learn
```

Optional integrations:

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
    task_type="classification",  # optional; auto-inferred if omitted
    test_size=0.25,
    scale=False,                 # tree models typically do not need scaling
    cv=3,
    n_walkthrough=3,
)
```

## API Reference

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

Parameters:

- `model`: sklearn-compatible estimator instance
- `X`: `np.ndarray` or `pd.DataFrame`
- `y`: `np.ndarray` or `pd.Series`
- `feature_names`: optional names (auto-filled from DataFrame columns)
- `task_type`: `"classification"` or `"regression"` (auto-detected if `None`)
- `test_size`: holdout fraction
- `scale`: apply `StandardScaler`
- `cv`: CV folds in validation stage
- `n_walkthrough`: number of test samples to explain in detail

Returns:

- `model`: fitted model
- `scaler`: fitted scaler or `None`

### `tune_model(...)` (Optuna)

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

What it adds:

- Model-family-specific hyperparameter search spaces
- Trial-by-trial CV optimization
- Final run through the same transparency pipeline with best params

Returns:

- `best_model`
- `scaler`
- `study` (`optuna.study.Study`)

## Running Demos

Run built-in demos:

```bash
python model_transparency.py
```

Run the custom XGBoost demo:

```bash
python run_demo.py
```

`run_demo.py` requires `xgboost` to be installed.

## Practical Interpretation Tips

- If train is much better than test in Step 6, check model complexity and Step 7 CV spread.
- If a feature looks important in built-in importance but weak in Step 8 permutation, treat it as unstable.
- For linear/SGD models, use coefficient magnitudes and signs to inspect directional effects.
- For tree/boosting models, use split/importance outputs plus prediction walkthroughs to confirm behavior.

## Limitations

- Terminal output is intentionally verbose and not optimized for production logging pipelines.
- Explanations are model-family heuristics plus diagnostics, not formal causal inference.
- No visualization dashboard is included; output is text-first.

## License

No license file is currently present. Add a `LICENSE` file before public distribution.
