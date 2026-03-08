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

- `model_transparency.py`  
  Transparency pipeline for tabular regression/classification (sklearn + optional XGBoost/LightGBM/Optuna).

- `Cnn_pipeline.py`  
  13-step CNN transparency pipeline for PyTorch (primary) and Keras (adapter path).

- `cnn_demo.py`  
  End-to-end Dog-vs-Cat demo that can run with:
  - real image folders (`data/train/...`, `data/val/...`), or
  - synthetic generated image data

- `run_demo.py`  
  Small tabular demo using `XGBClassifier`.

---

## Why This Project Exists

You can use this repository when you need answers like:

- "Why is my model overfitting?"
- "Are my inputs healthy enough to trust this training run?"
- "Did the model learn useful structure, or noise?"
- "Which features/channels/regions influence decisions most?"
- "How do I debug confidence, calibration, and misclassifications?"

The output is intentionally terminal-first so it is easy to run in scripts, servers, and CI logs without notebooks.

---

## A) Tabular Transparency Pipeline (`model_transparency.py`)

### Main API

```python
from model_transparency import run_pipeline, tune_model
```

### What `run_pipeline(...)` explains

`run_pipeline(...)` executes a staged narrative:

- **Step 0 - Dataset Health Report**  
  Checks missingness, imbalance/skew, leakage risk, multicollinearity, low variance features, duplicates, outliers, and data-size risk.

- **Step 1 - Data Analysis**  
  Shape, target behavior, feature stats, and correlations.

- **Step 2 - Preprocessing**  
  Train/test split, optional scaling diagnostics, and anti-leakage scaling behavior.

- **Step 3 - Training**  
  Model fit summary and timing.

- **Step 3.5 - Training Decision Explanations**  
  Explains why the model picked certain splits/weights.

- **Step 4 - Model Internals Deep Dive**  
  Family-specific internals (linear coefficients, tree structures, boosting details, etc.).

- **Step 5 - Prediction Walkthrough**  
  Per-sample predictions with detailed interpretation.

- **Step 6 - Evaluation Metrics**  
  Classification or regression metrics with richer context.

- **Step 6.5 - Overfitting Diagnosis Engine**  
  Detects healthy fit vs mild/strong overfit vs underfit, then suggests concrete fixes.

- **Step 7 - Smart Validation**  
  Auto-selects validation strategy by model family + dataset size (CV/ShuffleSplit/OOB/holdout paths).

- **Step 8 - Permutation Importance**  
  Model-agnostic feature impact measurement.

- **Step 9 - Learning Trace**  
  Traces model learning behavior in more detail.

### Supported tabular model families

- Linear: `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDClassifier`, `SGDRegressor`
- Tree/ensemble: `DecisionTree*`, `RandomForest*`, `GradientBoosting*`
- Margin/distance: `SVC`, `SVR`, `KNeighbors*`
- Optional: `XGBClassifier` / `XGBRegressor`, `LGBMClassifier` / `LGBMRegressor`

### Quick start (tabular)

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model_transparency import run_pipeline

X = np.array([[1.0, 40, 35, 2], [2.0, 55, 50, 3], [5.0, 85, 78, 7], [7.0, 92, 88, 9]])
y = np.array([0, 0, 1, 1])

model = RandomForestClassifier(n_estimators=100, random_state=42)
trained_model, scaler = run_pipeline(
    model, X, y,
    feature_names=["hours_studied", "attendance_pct", "prev_score", "assignments_done"],
    task_type="classification",
    scale=False,
)
```

### Hyperparameter tuning (tabular)

`tune_model(...)` uses Optuna and then automatically re-runs the full transparency pipeline with the best parameters.

```python
from model_transparency import tune_model
best_model, scaler, study = tune_model(model, X, y, n_trials=50, cv_folds=3)
```

---

## B) CNN Transparency Pipeline (`Cnn_pipeline.py`)

### Main API

```python
from Cnn_pipeline import run_cnn_pipeline, TrainingTracer
```

### What `run_cnn_pipeline(...)` explains

It provides 13 explainability/debugging steps:

- **Step 0**: image dataset health (shape/range/channel stats/class balance/duplicates)
- **Step 1**: architecture visualizer (layer-by-layer parameter budget and diagnostics)
- **Step 2**: training setup explainer (optimizer, scheduler, loss, augmentation, batch-size implications)
- **Step 3**: training loop trace (`TrainingTracer` history)
- **Step 4**: activation analysis
- **Step 5**: filter visualizer
- **Step 6**: prediction walkthrough (per-sample confidence and errors)
- **Step 7**: evaluation metrics (per-class metrics, confusion matrix, calibration)
- **Step 8**: CNN overfitting diagnosis
- **Step 9**: gradient flow analysis
- **Step 10**: feature map visualizer
- **Step 11**: saliency and attribution maps
- **Step 12**: training decision trace (update magnitudes, BN/dropout behavior, per-sample loss)

You can run all steps or a subset via `run_steps=[...]`.

### Framework support

- PyTorch: primary and most complete path
- TensorFlow/Keras: supported via adapter for core operations

### Quick start (CNN)

```python
from Cnn_pipeline import run_cnn_pipeline, TrainingTracer

tracer = TrainingTracer()

run_cnn_pipeline(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    class_names=["cat", "dog"],
    task="classification",
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    tracer=tracer,
    input_shape=(3, 64, 64),
    n_walkthrough=5,
    run_steps=None,  # e.g. [0,1,2,7,8,11] for focused runs
)
```

---

## `cnn_demo.py` (Dog vs Cat Demo)

`cnn_demo.py` is the easiest way to try the CNN stack.

### Modes

- `auto`: tries real-data mode first; falls back to synthetic mode
- `real`: expects image folders under `data/`
- `synthetic`: generates learnable synthetic cat/dog-style images

### Real data folder layout

```text
data/
  train/
    cats/
    dogs/
  val/
    cats/
    dogs/
```

### Common commands

```bash
python cnn_demo.py
python cnn_demo.py --mode synthetic
python cnn_demo.py --mode real --data-dir data --epochs 20 --batch 32 --img-size 64
python cnn_demo.py --steps 0 1 4 7 9 11
python cnn_demo.py --no-train
```

---

## Installation

### Core tabular dependencies

```bash
pip install numpy pandas scikit-learn
```

### Optional tabular integrations

```bash
pip install xgboost lightgbm optuna
```

### CNN dependencies

```bash
pip install torch torchvision
```

### Optional for real image loading

```bash
pip install Pillow
```

---

## Practical Usage Guidance

- Use the **tabular pipeline** when your input is feature vectors/tables.
- Use the **CNN pipeline** when your input is images and you need architecture + gradient-level inspection.
- Start with full runs once, then use `run_steps` to iterate quickly on specific diagnostics.
- Treat overfitting diagnosis and calibration outputs as decision tools for the next experiment setup.

---

## Notes

- Output is verbose by design and optimized for interpretability.
- `tune_model(...)` requires `optuna`.
- `run_demo.py` requires `xgboost`.
- `cnn_demo.py` expects `Cnn_pipeline.py` in the same directory.

---

## License

No license file is currently present.  
Add a `LICENSE` file before public/open distribution.
