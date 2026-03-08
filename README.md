# Model Transparency Repository

Terminal-first transparency tools for both:

- **Classical ML models** (`model_transparency.py`)
- **CNN models** (`Cnn_pipeline.py`)

The goal is to show what happened during training and prediction, not just final scores.

## Repository Contents

- `model_transparency.py`: end-to-end transparency pipeline for sklearn-style tabular models
- `Cnn_pipeline.py`: 13-step CNN transparency pipeline for PyTorch (and Keras via adapter)
- `cnn_demo.py`: Dog-vs-Cat demo using real image folders or synthetic image generation
- `run_demo.py`: small XGBoost demo for the tabular pipeline

## 1) Tabular Model Transparency Pipeline

Main API:

```python
from model_transparency import run_pipeline, tune_model
```

### What it reports

`run_pipeline(...)` runs a full staged report including:

- Step 0 dataset health report (missing values, imbalance/skew, leakage risk, multicollinearity, duplicates, outliers, scorecard)
- Step 1 data analysis and feature stats
- Step 2 preprocessing and scaling diagnostics
- Step 3 training summary
- Step 3.5 training decision explanations
- Step 4 model internals deep dive
- Step 5 prediction walkthrough
- Step 6 evaluation metrics
- Step 6.5 overfitting diagnosis engine
- Step 7 smart validation strategy
- Step 8 permutation importance
- Step 9 learning trace

### Supported model families

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

## 2) CNN Transparency Pipeline

Main API:

```python
from Cnn_pipeline import run_cnn_pipeline, TrainingTracer
```

### What it reports

`run_cnn_pipeline(...)` supports step filtering (`run_steps`) and includes:

- Step 0: dataset health report for image batches
- Step 1: architecture visualizer and parameter budget
- Step 2: training setup explainer (optimizer/scheduler/loss/augmentation/batch size)
- Step 3: training loop trace (`TrainingTracer`)
- Step 4: activation analysis
- Step 5: filter visualizer
- Step 6: prediction walkthrough
- Step 7: evaluation metrics (per-class metrics, confusion matrix, calibration)
- Step 8: CNN overfitting diagnosis
- Step 9: gradient flow analysis
- Step 10: feature map visualizer
- Step 11: saliency and attribution maps
- Step 12: training decision trace (update magnitudes, BN/dropout behavior, per-sample loss)

### Framework support

- PyTorch: primary path
- TensorFlow/Keras: supported through framework adapter for core pipeline calls

### Quick start (CNN)

```python
from Cnn_pipeline import run_cnn_pipeline, TrainingTracer

# model: torch.nn.Module
# train_loader / val_loader: DataLoader returning (images, labels)
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
    run_steps=None,  # or e.g. [0,1,2,7,8,11]
)
```

## cnn_demo.py (Dog vs Cat Demo)

`cnn_demo.py` auto-detects mode:

- `real`: reads image folders from `data/`
- `synthetic`: generates procedurally learnable cat/dog images
- `auto`: picks `real` if dataset exists, otherwise falls back to `synthetic`

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

### Demo commands

```bash
python cnn_demo.py
python cnn_demo.py --mode synthetic
python cnn_demo.py --mode real --data-dir data --epochs 20 --batch 32 --img-size 64
python cnn_demo.py --steps 0 1 4 7 9 11
python cnn_demo.py --no-train
```

## Installation

Base (tabular pipeline):

```bash
pip install numpy pandas scikit-learn
```

Optional tabular integrations:

```bash
pip install xgboost lightgbm optuna
```

CNN + demo:

```bash
pip install torch torchvision
```

Optional for real image loading:

```bash
pip install Pillow
```

## Notes

- Output is intentionally verbose and terminal-oriented.
- `tune_model(...)` requires `optuna`.
- `run_demo.py` requires `xgboost`.
- `cnn_demo.py` imports `Cnn_pipeline.py` from the same directory.

## License

No license file is currently included.
Add a `LICENSE` file before public distribution.
