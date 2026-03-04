"""
==============================================================================
  ML TRANSPARENCY PIPELINE — See Everything That Happens Inside Your Model
==============================================================================
  Supports: Linear/Logistic Regression, Ridge, Lasso, Decision Tree,
            Random Forest, Gradient Boosting, SVM, KNN, + XGBoost/LightGBM
            (auto-detected if installed)
  Task types: Classification & Regression
  Output: Terminal only — rich, step-by-step internals
==============================================================================
"""

import sys
import time
import warnings
import textwrap
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

# ── Optional heavy libraries ──────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence default optuna logs
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── sklearn ────────────────────────────────────────────────────────────────────
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.inspection import permutation_importance

# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

WIDTH = 80

def banner(title, char="═"):
    pad = (WIDTH - len(title) - 2) // 2
    print(f"\n{char * WIDTH}")
    print(f"{char * pad} {title} {char * (WIDTH - pad - len(title) - 2)}")
    print(f"{char * WIDTH}")

def section(title, char="─"):
    print(f"\n{char * WIDTH}")
    print(f"  ▶  {title}")
    print(f"{char * WIDTH}")

def subsection(title):
    print(f"\n  {'·' * 4}  {title}  {'·' * 4}")

def info(label, value, indent=4):
    label_str = f"{' ' * indent}{label}:"
    print(f"{label_str:<35} {value}")

def row_sep():
    print("  " + "─" * (WIDTH - 4))

def table(headers, rows, col_width=18):
    fmt = ("  " + ("{:<" + str(col_width) + "}") * len(headers))
    print(fmt.format(*headers))
    print("  " + "─" * (col_width * len(headers)))
    for r in rows:
        print(fmt.format(*[str(v) for v in r]))

def bar_chart(label_vals, total=None, width=40, color=True):
    """Print ASCII bar chart. label_vals = list of (label, value)"""
    if not label_vals:
        return
    max_val = max(v for _, v in label_vals) or 1
    if total is None:
        total = max_val
    for label, val in label_vals:
        filled = int((val / max_val) * width)
        bar = "█" * filled + "░" * (width - filled)
        pct = f"{val / total * 100:5.1f}%" if total else ""
        print(f"  {str(label):<20} │{bar}│ {val:>8.4g}  {pct}")

def progress_bar(current, total, label="", width=40):
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100
    print(f"\r  {label:<20} [{bar}] {pct:5.1f}%", end="", flush=True)
    if current == total:
        print()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_data(X, y, feature_names, task_type):
    banner("STEP 1 — DATA ANALYSIS")

    section("Dataset Shape")
    info("Total samples", X.shape[0])
    info("Features", X.shape[1])
    info("Feature names", ", ".join(feature_names[:10]) + ("..." if len(feature_names) > 10 else ""))
    info("Task type", task_type.upper())

    section("Target Variable (y)")
    if task_type == "classification":
        counts = Counter(y)
        info("Classes", sorted(counts.keys()))
        info("Class distribution", "")
        bar_chart(sorted(counts.items()), total=len(y))
        info("Imbalance ratio", f"{max(counts.values()) / min(counts.values()):.2f}x")
    else:
        info("Min", f"{np.min(y):.4f}")
        info("Max", f"{np.max(y):.4f}")
        info("Mean", f"{np.mean(y):.4f}")
        info("Std", f"{np.std(y):.4f}")
        info("Median", f"{np.median(y):.4f}")

    section("Feature Statistics (first 8 features)")
    stats_rows = []
    for i, name in enumerate(feature_names[:8]):
        col = X[:, i]
        stats_rows.append([
            name[:16],
            f"{np.mean(col):.3g}",
            f"{np.std(col):.3g}",
            f"{np.min(col):.3g}",
            f"{np.max(col):.3g}",
            f"{np.sum(np.isnan(col))}",
        ])
    table(["Feature", "Mean", "Std", "Min", "Max", "NaNs"], stats_rows)

    section("Correlation with Target (top 8)")
    if task_type == "regression":
        correlations = []
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations.append((name, corr))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, corr in correlations[:8]:
            bar = "▓" * int(abs(corr) * 30)
            sign = "+" if corr >= 0 else "-"
            print(f"  {name:<20} {sign}{bar} {corr:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(X, y, test_size=0.2, scale=True, random_state=42):
    banner("STEP 2 — PREPROCESSING")

    section("Train/Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    info("Train samples", X_train.shape[0])
    info("Test  samples", X_test.shape[0])
    info("Train ratio", f"{1 - test_size:.0%}")
    info("Test  ratio", f"{test_size:.0%}")

    scaler = None
    if scale:
        section("Feature Scaling (StandardScaler)")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        info("Method", "StandardScaler  →  (x - mean) / std")
        info("Fit on", "Training set only (no data leakage)")

        print("\n  Per-feature scaling params (first 6):")
        feat_rows = []
        for i in range(min(6, X_train.shape[1])):
            feat_rows.append([
                f"feature_{i}" if not hasattr(X, "columns") else X.columns[i],
                f"{scaler.mean_[i]:.4f}",
                f"{scaler.scale_[i]:.4f}",
            ])
        table(["Feature", "Mean (subtracted)", "Std (divided)"], feat_rows, col_width=22)

        print("\n  Before scaling sample (row 0, first 5 features):")
        print(f"  {X_train[0, :5]}")
        print("  After scaling:")
        print(f"  {X_train_s[0, :5]}")
    else:
        X_train_s = X_train
        X_test_s = X_test
        info("Scaling", "Skipped")

    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — MODEL INTERNALS EXPLAINERS
# ══════════════════════════════════════════════════════════════════════════════

def explain_linear(model, feature_names, task_type):
    section("Linear Model Internals")

    if hasattr(model, "coef_"):
        coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        info("Intercept (bias)", f"{np.atleast_1d(model.intercept_)[0]:.6f}")
        info("# Coefficients", len(coefs))

        subsection("Equation")
        terms = []
        for name, c in zip(feature_names[:5], coefs[:5]):
            terms.append(f"({c:.4f} × {name})")
        eq = " + ".join(terms)
        if len(feature_names) > 5:
            eq += " + ..."
        intercept_val = np.atleast_1d(model.intercept_)[0]
        print(f"\n  ŷ = {eq} + {intercept_val:.4f}")

        subsection("Top Coefficients by Magnitude")
        coef_pairs = list(zip(feature_names, coefs))
        coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        bar_chart([(n[:20], abs(c)) for n, c in coef_pairs[:10]])

        if hasattr(model, "alpha"):
            info("\n  Regularization alpha", model.alpha)

    if hasattr(model, "n_iter_"):
        info("Iterations to converge", model.n_iter_)


def explain_tree(model, feature_names, depth=0, max_depth_show=3):
    section("Decision Tree Internals")

    tree = model.tree_
    info("Total nodes", tree.node_count)
    info("Leaf nodes", int(np.sum(tree.children_left == -1)))
    info("Max depth", tree.max_depth)
    info("Features used", len(set(tree.feature[tree.feature >= 0])))

    subsection("Feature Importances (Gini/MSE impurity reduction)")
    importances = model.feature_importances_
    imp_pairs = list(zip(feature_names, importances))
    imp_pairs.sort(key=lambda x: x[1], reverse=True)
    bar_chart([(n[:20], v) for n, v in imp_pairs[:10]], total=1.0)

    subsection(f"Tree Structure Preview (depth ≤ {max_depth_show})")
    _print_tree_nodes(tree, feature_names, node_id=0, depth=0, max_depth=max_depth_show)

def _print_tree_nodes(tree, feature_names, node_id, depth, max_depth):
    if depth > max_depth:
        print("  " + "  " * depth + "...")
        return
    indent = "  " + "  " * depth
    is_leaf = tree.children_left[node_id] == -1
    if is_leaf:
        val = tree.value[node_id].flatten()
        print(f"{indent}[LEAF]  samples={tree.n_node_samples[node_id]}  value={np.round(val, 3)}")
    else:
        feat = feature_names[tree.feature[node_id]]
        thresh = tree.threshold[node_id]
        impurity = tree.impurity[node_id]
        print(f"{indent}[NODE]  if {feat} ≤ {thresh:.4f}  "
              f"(impurity={impurity:.4f}, samples={tree.n_node_samples[node_id]})")
        print(f"{indent}  ├─ TRUE →")
        _print_tree_nodes(tree, feature_names, tree.children_left[node_id], depth + 1, max_depth)
        print(f"{indent}  └─ FALSE →")
        _print_tree_nodes(tree, feature_names, tree.children_right[node_id], depth + 1, max_depth)


def explain_forest(model, feature_names):
    section("Random Forest / Ensemble Internals")

    info("Number of estimators", len(model.estimators_))
    info("Max features per split", model.max_features)
    info("Bootstrap sampling", model.bootstrap)
    info("OOB score available", hasattr(model, "oob_score_") and model.oob_score is not False)

    if hasattr(model, "oob_score_"):
        info("OOB score", f"{model.oob_score_:.4f}")

    subsection("Tree depth distribution across forest")
    depths = [est.tree_.max_depth for est in model.estimators_]
    info("Min tree depth", min(depths))
    info("Max tree depth", max(depths))
    info("Mean tree depth", f"{np.mean(depths):.2f}")

    subsection("Aggregated Feature Importances")
    importances = model.feature_importances_
    std = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
    imp_pairs = list(zip(feature_names, importances, std))
    imp_pairs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {'Feature':<22} {'Importance':>10}  {'±Std':>8}")
    print("  " + "─" * 45)
    for name, imp, s in imp_pairs[:10]:
        bar = "█" * int(imp * 60)
        print(f"  {name:<22} {imp:>10.4f}  ±{s:.4f}  {bar}")


def explain_gbm(model, feature_names):
    section("Gradient Boosting Internals")

    info("n_estimators", model.n_estimators)
    info("Learning rate", model.learning_rate)
    info("Max depth per tree", model.max_depth)
    info("Loss function", model.loss)
    info("Subsample ratio", model.subsample)

    subsection("Stage-wise training loss (every 10 stages)")
    if hasattr(model, "train_score_"):
        scores = model.train_score_
        for i in range(0, len(scores), max(1, len(scores) // 10)):
            progress_bar(i + 1, len(scores), label=f"  Stage {i+1:>4}")
            time.sleep(0.001)
        progress_bar(len(scores), len(scores), label=f"  Stage {len(scores):>4}")
        print(f"\n  Initial loss: {scores[0]:.6f}")
        print(f"  Final  loss: {scores[-1]:.6f}")
        print(f"  Reduction:   {(scores[0] - scores[-1]) / scores[0] * 100:.2f}%")

    subsection("Feature Importances")
    importances = model.feature_importances_
    imp_pairs = list(zip(feature_names, importances))
    imp_pairs.sort(key=lambda x: x[1], reverse=True)
    bar_chart([(n[:20], v) for n, v in imp_pairs[:10]], total=1.0)


def explain_svm(model, feature_names):
    section("SVM Internals")
    info("Kernel", model.kernel)
    info("C (regularization)", model.C)
    if hasattr(model, "gamma"):
        info("Gamma", model.gamma)
    if hasattr(model, "support_vectors_"):
        info("# Support vectors", len(model.support_vectors_))
        if hasattr(model, "n_support_"):
            info("Support vectors per class", model.n_support_.tolist())
    info("Decision function shape", getattr(model, "decision_function_shape", "N/A"))

    subsection("What Support Vectors Mean")
    print(textwrap.fill(
        "  Support vectors are the training samples closest to the decision "
        "boundary. Only these points determine the hyperplane — all other "
        "points are ignored. A larger C = narrower margin, fewer SVs, "
        "potentially overfit. Smaller C = wider margin, more SVs, more robust.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "
    ))


def explain_knn(model, feature_names):
    section("KNN Internals")
    info("n_neighbors (k)", model.n_neighbors)
    info("Distance metric", model.metric)
    info("Weighting", model.weights)
    info("Algorithm for search", model.algorithm)
    info("Training samples stored", model.n_samples_fit_)

    subsection("How KNN Predicts")
    print(textwrap.fill(
        "  For each test point, KNN computes the distance to ALL training "
        "points, selects the k closest, then (for classification) takes a "
        "majority vote or (for regression) averages their target values. "
        "There is no 'training' — the model IS the training data.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "
    ))


def explain_xgboost(model, feature_names):
    section("XGBoost Internals")
    params = model.get_params()
    for k, v in list(params.items())[:12]:
        info(k, v)

    subsection("Feature Importances (weight = # times used in splits)")
    scores = model.get_booster().get_fscore()
    if scores:
        # remap f0, f1... -> real feature names
        remapped = {}
        for k, v in scores.items():
            try:
                idx  = int(k[1:])
                name = feature_names[idx] if idx < len(feature_names) else k
            except (ValueError, IndexError):
                name = k
            remapped[name] = v
        sorted_scores = sorted(remapped.items(), key=lambda x: x[1], reverse=True)
        bar_chart([(k[:20], v) for k, v in sorted_scores[:10]])


def explain_lgbm(model, feature_names):
    section("LightGBM Internals")
    info("num_leaves", model.num_leaves)
    info("learning_rate", model.learning_rate)
    info("n_estimators", model.n_estimators)
    info("max_depth", model.max_depth)

    subsection("Feature Importances (split count)")
    importances = model.feature_importances_
    imp_pairs = list(zip(feature_names, importances))
    imp_pairs.sort(key=lambda x: x[1], reverse=True)
    bar_chart([(n[:20], v) for n, v in imp_pairs[:10]])


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAINING (with live timing)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train):
    banner("STEP 3 — TRAINING")

    model_name = type(model).__name__
    section(f"Training: {model_name}")
    info("Model class", model_name)
    info("Parameters", "")

    params = model.get_params()
    for k, v in list(params.items())[:15]:
        info(f"  └─ {k}", v)

    print(f"\n  ⏳ Fitting model on {X_train.shape[0]} samples × {X_train.shape[1]} features ...")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    print(f"  ✅ Training complete in {elapsed:.4f}s")

    return model, elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — MODEL INTERNALS DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

def explain_model(model, feature_names, task_type):
    banner("STEP 4 — MODEL INTERNALS DEEP DIVE")

    name = type(model).__name__

    if name in ("LinearRegression", "Ridge", "Lasso", "ElasticNet",
                "LogisticRegression", "SGDClassifier", "SGDRegressor"):
        explain_linear(model, feature_names, task_type)

    elif name in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        explain_tree(model, feature_names)

    elif name in ("RandomForestClassifier", "RandomForestRegressor"):
        explain_forest(model, feature_names)

    elif name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        explain_gbm(model, feature_names)

    elif name in ("SVC", "SVR"):
        explain_svm(model, feature_names)

    elif name in ("KNeighborsClassifier", "KNeighborsRegressor"):
        explain_knn(model, feature_names)

    elif HAS_XGB and name in ("XGBClassifier", "XGBRegressor"):
        explain_xgboost(model, feature_names)

    elif HAS_LGB and name in ("LGBMClassifier", "LGBMRegressor"):
        explain_lgbm(model, feature_names)

    else:
        section("Model Internals")
        info("Model", name)
        info("Params", model.get_params())


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — PREDICTION WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════

def walkthrough_predictions(model, X_test, y_test, feature_names, task_type, n=5):
    banner("STEP 5 — PREDICTION WALKTHROUGH (Sample-by-Sample)")

    section(f"Showing first {n} test predictions in detail")

    y_pred = model.predict(X_test)
    has_proba = hasattr(model, "predict_proba")

    for i in range(min(n, len(X_test))):
        print(f"\n  {'─'*60}")
        print(f"  Sample #{i+1}")
        print(f"  {'─'*60}")
        print(f"  Input features:")
        for j, (name, val) in enumerate(zip(feature_names[:8], X_test[i, :8])):
            print(f"    {name:<22}: {val:.4f}")
        if len(feature_names) > 8:
            print(f"    ... ({len(feature_names) - 8} more features)")

        print(f"\n  Actual    target : {y_test[i]}")
        print(f"  Predicted target : {y_pred[i]}")

        if task_type == "regression":
            err = abs(y_pred[i] - y_test[i])
            print(f"  Absolute error   : {err:.4f}")
        else:
            correct = "✅ CORRECT" if y_pred[i] == y_test[i] else "❌ WRONG"
            print(f"  Result           : {correct}")
            if has_proba:
                proba = model.predict_proba(X_test[i:i+1])[0]
                classes = model.classes_
                print(f"  Class probabilities:")
                for cls, p in zip(classes, proba):
                    bar = "█" * int(p * 30)
                    print(f"    Class {cls}: {bar:<30} {p:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_train_s, X_test_s, y_train, y_test, task_type):
    banner("STEP 6 — EVALUATION METRICS")

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    if task_type == "classification":
        section("Classification Metrics")

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        info("Train Accuracy", f"{train_acc:.4f}  ({train_acc*100:.2f}%)")
        info("Test  Accuracy", f"{test_acc:.4f}  ({test_acc*100:.2f}%)")
        gap = train_acc - test_acc
        if gap > 0.1:
            info("⚠  Overfit gap", f"{gap:.4f}  — consider regularization")
        elif gap < -0.02:
            info("ℹ  Note", "Test > Train — may have lucky split")
        else:
            info("✅ Generalization", f"gap={gap:.4f}  looks healthy")

        section("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_test)
        classes = np.unique(y_test)
        header = [""] + [f"Pred {c}" for c in classes]
        rows = [[f"True {classes[i]}"] + list(cm[i]) for i in range(len(classes))]
        table(header, rows, col_width=12)

        section("Per-Class Report")
        report = classification_report(y_test, y_pred_test)
        for line in report.split("\n"):
            print(f"  {line}")

    else:
        section("Regression Metrics")

        metrics = {
            "Train MSE":  mean_squared_error(y_train, y_pred_train),
            "Test  MSE":  mean_squared_error(y_test, y_pred_test),
            "Train RMSE": mean_squared_error(y_train, y_pred_train) ** 0.5,
            "Test  RMSE": mean_squared_error(y_test, y_pred_test) ** 0.5,
            "Train MAE":  mean_absolute_error(y_train, y_pred_train),
            "Test  MAE":  mean_absolute_error(y_test, y_pred_test),
            "Train R²":   r2_score(y_train, y_pred_train),
            "Test  R²":   r2_score(y_test, y_pred_test),
        }
        for k, v in metrics.items():
            info(k, f"{v:.6f}")

        r2_gap = r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test)
        if r2_gap > 0.15:
            info("\n  ⚠  R² gap", f"{r2_gap:.4f}  — possible overfitting")

        section("Residuals (first 10 test samples)")
        print(f"  {'Sample':<10} {'Actual':>12} {'Predicted':>12} {'Residual':>12} {'% Error':>10}")
        print("  " + "─" * 60)
        for i in range(min(10, len(y_test))):
            residual = y_test[i] - y_pred_test[i]
            pct = abs(residual / y_test[i]) * 100 if y_test[i] != 0 else float("nan")
            print(f"  {i:<10} {y_test[i]:>12.4f} {y_pred_test[i]:>12.4f} "
                  f"{residual:>12.4f} {pct:>9.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — SMART VALIDATION  (auto-picks method by model type + dataset size)
# ══════════════════════════════════════════════════════════════════════════════

# Size thresholds
_SMALL  =  5_000    # < 5k   → full CV is fine
_MEDIUM = 50_000    # < 50k  → ShuffleSplit
                    # ≥ 50k  → Holdout only (or OOB / early-stopping if available)

def _print_scores(scores, label="Score"):
    """Print a score array with bars + stats."""
    for i, s in enumerate(scores):
        bar = "█" * int(max(0, s) * 40)
        print(f"  Split {i+1}: {bar:<40} {s:.4f}")
    row_sep()
    info(f"Mean {label}", f"{scores.mean():.4f}")
    info(f"Std  {label}", f"{scores.std():.4f}")
    info(f"Min  {label}", f"{scores.min():.4f}")
    info(f"Max  {label}", f"{scores.max():.4f}")
    info("95% CI", f"[{scores.mean()-2*scores.std():.4f},  {scores.mean()+2*scores.std():.4f}]")


def validate_model(model, X, y, X_train, y_train, X_test, y_test,
                   task_type, cv=5, random_state=42):
    """
    Auto-selects the fastest reliable validation strategy:

    Dataset size     Model type          Method chosen
    ─────────────────────────────────────────────────────────────────
    any              RandomForest*       OOB Score  (free, zero cost)
    any              XGB* / LGB*         Early Stopping on eval_set
    < 5 000          any other           Full k-Fold CV
    5 000 – 50 000   any other           ShuffleSplit (5 random splits)
    ≥ 50 000         any other           Holdout only  (train/test split)
    ─────────────────────────────────────────────────────────────────
    Always appended: Stability check — score variance warning
    """
    banner("STEP 7 — SMART VALIDATION")

    n_samples   = X.shape[0]
    model_name  = type(model).__name__
    scoring     = "accuracy" if task_type == "classification" else "r2"
    model_class = type(model)

    # ── Decide method ──────────────────────────────────────────────────────────
    is_forest  = model_name in ("RandomForestClassifier", "RandomForestRegressor",
                                "ExtraTreesClassifier",   "ExtraTreesRegressor",
                                "BaggingClassifier",      "BaggingRegressor")
    is_xgb     = HAS_XGB and model_name in ("XGBClassifier", "XGBRegressor")
    is_lgb     = HAS_LGB and model_name in ("LGBMClassifier", "LGBMRegressor")
    is_boosted = is_xgb or is_lgb

    section("Strategy Selection")
    info("Model",        model_name)
    info("Dataset size", f"{n_samples:,} samples")
    info("Task",         task_type.upper())

    # ── METHOD 1 — OOB (Random Forest / Bagging) ──────────────────────────────
    if is_forest:
        method = "OOB Score"
        info("Method chosen", f"✦ {method}")
        info("Reason", "Forest models compute OOB for free during training — zero extra cost")

        section(f"Method: {method}")
        print(textwrap.fill(
            "  Every tree is trained on a bootstrap sample (~63% of data). "
            "The remaining ~37% (out-of-bag samples) are used as a natural "
            "test set for that tree. Aggregating these gives an unbiased "
            "estimate without any extra training passes.",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "
        ))

        # Re-fit with oob_score=True if not already set
        params = model.get_params()
        if not params.get("oob_score", False):
            print(f"\n  ⚙  Re-fitting with oob_score=True ...")
            try:
                oob_model = model_class(**{**params, "oob_score": True})
                t0 = time.perf_counter()
                oob_model.fit(X_train, y_train)
                elapsed = time.perf_counter() - t0
                oob_score = oob_model.oob_score_
                print(f"  ✅ Done in {elapsed:.3f}s")
            except Exception as e:
                print(f"  ⚠  OOB refit failed ({e}), falling back to holdout")
                oob_score = None
        else:
            oob_score = getattr(model, "oob_score_", None)

        if oob_score is not None:
            subsection("OOB Result")
            bar = "█" * int(max(0, oob_score) * 40)
            info("OOB Score", f"{oob_score:.4f}  {bar}")

            # Compare with test score
            test_score = model.score(X_test, y_test)
            info("Test Score (holdout)", f"{test_score:.4f}")
            gap = abs(oob_score - test_score)
            if gap < 0.02:
                info("✅ OOB vs Test gap", f"{gap:.4f}  — consistent, trustworthy estimate")
            elif gap < 0.05:
                info("ℹ  OOB vs Test gap", f"{gap:.4f}  — slight variance, normal for small data")
            else:
                info("⚠  OOB vs Test gap", f"{gap:.4f}  — large gap, check for data issues")

        subsection("Why OOB beats CV here")
        print("  CV  : trains model k times  →  k × training cost")
        print("  OOB : trains model 1 time   →  1 × training cost  (same reliability)")
        return

    # ── METHOD 2 — Early Stopping (XGBoost / LightGBM) ───────────────────────
    if is_boosted:
        method = "Early Stopping on Eval Set"
        info("Method chosen", f"✦ {method}")
        info("Reason", "XGB/LGB can watch a validation set live during boosting — no retraining needed")

        section(f"Method: {method}")
        print(textwrap.fill(
            "  Instead of training k separate models, one model is trained "
            "with an eval_set watchdog. After each boosting round the score "
            "on the held-out set is computed. Training stops automatically "
            "when the score stops improving (early_stopping_rounds). "
            "Result: best iteration found at zero extra training cost.",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "
        ))

        params      = model.get_params()
        eval_metric = "logloss" if task_type == "classification" else "rmse"
        es_rounds   = max(10, params.get("n_estimators", 100) // 10)

        try:
            if is_xgb:
                es_model = model_class(
                    **{**params,
                       "n_estimators": params.get("n_estimators", 300),
                       "early_stopping_rounds": es_rounds,
                       "eval_metric": eval_metric}
                )
                print(f"\n  ⏳ Training with early stopping (max {params.get('n_estimators',300)} rounds, "
                      f"stop after {es_rounds} no-improve) ...")
                t0 = time.perf_counter()
                es_model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=False,
                )
                elapsed = time.perf_counter() - t0

            elif is_lgb:
                callbacks = [lgb.early_stopping(es_rounds, verbose=False),
                             lgb.log_evaluation(period=-1)]
                es_model = model_class(**params)
                print(f"\n  ⏳ Training with early stopping ...")
                t0 = time.perf_counter()
                es_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=callbacks,
                )
                elapsed = time.perf_counter() - t0

            print(f"  ✅ Done in {elapsed:.3f}s")

            subsection("Early Stopping Results")
            best_iter = getattr(es_model, "best_iteration_",
                        getattr(es_model, "best_iteration", "N/A"))
            info("Best iteration",     best_iter)
            info("Max rounds set",     params.get("n_estimators", 300))
            if isinstance(best_iter, int) and best_iter < params.get("n_estimators", 300):
                saved = params.get("n_estimators", 300) - best_iter
                info("Rounds saved",   f"{saved}  (stopped early ✅)")
            else:
                info("Note", "Did not stop early — consider increasing n_estimators")

            # Eval curve
            evals = getattr(es_model, "evals_result_", None)
            if evals:
                subsection("Validation Loss Curve (every 10% of rounds)")
                key   = list(evals.keys())[-1]          # last eval set = test
                mkey  = list(evals[key].keys())[0]
                curve = evals[key][mkey]
                step  = max(1, len(curve) // 10)
                best_val = min(curve) if task_type == "regression" else min(curve)
                for i in range(0, len(curve), step):
                    filled = int((1 - curve[i] / (max(curve) + 1e-9)) * 30)
                    bar = "█" * max(0, filled)
                    marker = " ← BEST" if abs(curve[i] - best_val) < 1e-9 else ""
                    print(f"  Round {i+1:>5}: {bar:<30} loss={curve[i]:.5f}{marker}")

            # Final scores
            subsection("Final Scores")
            train_s = es_model.score(X_train, y_train)
            test_s  = es_model.score(X_test,  y_test)
            info("Train score", f"{train_s:.4f}")
            info("Test  score", f"{test_s:.4f}")
            gap = train_s - test_s
            if gap > 0.1:
                info("⚠  Overfit gap", f"{gap:.4f}")
            else:
                info("✅ Overfit gap",  f"{gap:.4f}  — looks healthy")

        except Exception as e:
            print(f"  ⚠  Early stopping failed ({e}), falling back to holdout score")
            test_s = model.score(X_test, y_test)
            info("Holdout score", f"{test_s:.4f}")

        return

    # ── METHOD 3 — Full k-Fold CV  (small datasets < 5k) ─────────────────────
    if n_samples < _SMALL:
        method = f"Full {cv}-Fold Cross-Validation"
        info("Method chosen", f"✦ {method}")
        info("Reason", f"Dataset is small ({n_samples:,} samples) — CV gives reliable estimate cheaply")

        section(f"Method: {method}")
        print(f"  Training {cv} times on {n_samples:,} samples — should be fast.\n")

        t0     = time.perf_counter()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.perf_counter() - t0

        info("Time", f"{elapsed:.3f}s  ({elapsed/cv:.3f}s per fold)")
        _print_scores(scores)

    # ── METHOD 4 — ShuffleSplit  (medium datasets 5k–50k) ────────────────────
    elif n_samples < _MEDIUM:
        n_splits = 5
        method   = f"ShuffleSplit ({n_splits} random splits)"
        info("Method chosen", f"✦ {method}")
        info("Reason", f"Dataset is medium ({n_samples:,} samples) — ShuffleSplit avoids exhaustive folds")

        section(f"Method: {method}")
        print(textwrap.fill(
            "  Instead of exhaustive k-fold, ShuffleSplit does N independent "
            "random train/test splits. Each split trains once on a fresh random "
            "80% and tests on 20%. Gives similar reliability to k-fold at lower cost "
            "because splits can overlap (non-exhaustive).",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "
        ))
        print()

        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)

        t0     = time.perf_counter()
        scores = cross_val_score(model, X, y, cv=ss, scoring=scoring, n_jobs=-1)
        elapsed = time.perf_counter() - t0

        info("Time", f"{elapsed:.3f}s  ({elapsed/n_splits:.3f}s per split)")
        _print_scores(scores)

    # ── METHOD 5 — Holdout only  (large datasets ≥ 50k) ──────────────────────
    else:
        method = "Holdout (single train/test split)"
        info("Method chosen", f"✦ {method}")
        info("Reason", f"Dataset is large ({n_samples:,} samples) — single split is fast and sufficient")

        section(f"Method: {method}")
        print(textwrap.fill(
            "  With large datasets a single holdout split is statistically "
            "reliable because variance of the estimate is low — each split "
            "contains thousands of samples. Re-training k times would cost "
            "k× compute with minimal benefit.",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "
        ))

        t0          = time.perf_counter()
        train_score = model.score(X_train, y_train)
        test_score  = model.score(X_test,  y_test)
        elapsed     = time.perf_counter() - t0

        subsection("Holdout Scores")
        info("Train score", f"{train_score:.4f}")
        info("Test  score", f"{test_score:.4f}")
        gap = train_score - test_score
        bar = "█" * int(max(0, test_score) * 40)
        print(f"\n  Test : {bar:<40} {test_score:.4f}")
        if gap > 0.1:
            info("⚠  Overfit gap", f"{gap:.4f}  — consider regularization or more data")
        elif gap < -0.02:
            info("ℹ  Note", "Test > Train — may have easy test split")
        else:
            info("✅ Gap", f"{gap:.4f}  — healthy generalization")
        info("Time", f"{elapsed:.4f}s")

    # ── Stability check (appended to all non-early-stopping methods) ──────────
    subsection("Stability Verdict")
    try:
        s = scores
        cv_std = s.std()
        if cv_std < 0.02:
            print("  ✅ STABLE    — std < 0.02, model is consistent across splits")
        elif cv_std < 0.05:
            print("  ℹ  MODERATE  — std 0.02–0.05, some variance, acceptable")
        else:
            print("  ⚠  UNSTABLE  — std > 0.05, high variance across splits")
            print("     → Try: more data, stronger regularization, or simpler model")
        info("Std across splits", f"{cv_std:.4f}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def perm_importance(model, X_test, y_test, feature_names, task_type,
                    _print_banner=True):
    if _print_banner:
        banner("STEP 8 — PERMUTATION IMPORTANCE")
    section("What happens when we shuffle each feature?")
    print("  (Measures actual drop in performance — model-agnostic)")

    scoring = "accuracy" if task_type == "classification" else "r2"

    try:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, scoring=scoring, random_state=42
        )
        imp_pairs = list(zip(feature_names, result.importances_mean, result.importances_std))
        imp_pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  {'Feature':<22} {'Importance':>12}  {'±Std':>8}")
        print("  " + "─" * 50)
        for name, imp, std in imp_pairs[:12]:
            bar = "█" * max(0, int(imp * 60))
            neg = "⚠" if imp < 0 else " "
            print(f"  {neg}{name:<21} {imp:>12.4f}  ±{std:.4f}  {bar}")

    except Exception as e:
        print(f"  ⚠  Permutation importance skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

# ── Per-model search spaces ────────────────────────────────────────────────────

def _suggest_params(trial, model_name, task_type):
    """Return a dict of suggested hyperparams for the given model."""

    if model_name in ("RandomForestClassifier", "RandomForestRegressor"):
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "max_depth":         trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    elif model_name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 500),
            "max_depth":     trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    elif model_name in ("XGBClassifier", "XGBRegressor"):
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 500),
            "max_depth":     trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    elif model_name in ("LGBMClassifier", "LGBMRegressor"):
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 500),
            "max_depth":     trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 20, 300),
            "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    elif model_name in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        return {
            "max_depth":         trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    elif model_name in ("LogisticRegression",):
        return {
            "C":        trial.suggest_float("C", 1e-4, 100.0, log=True),
            "solver":   trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": trial.suggest_int("max_iter", 200, 2000),
        }

    elif model_name in ("Ridge",):
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}

    elif model_name in ("Lasso", "ElasticNet"):
        params = {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}
        if model_name == "ElasticNet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return params

    elif model_name in ("SVC", "SVR"):
        return {
            "C":      trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "gamma":  trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    elif model_name in ("KNeighborsClassifier", "KNeighborsRegressor"):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric":      trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        }

    else:
        return {}  # unknown model — no tuning


def tune_model(model, X, y, task_type=None, n_trials=50, cv_folds=3,
               test_size=0.2, scale=True, random_state=42):
    """
    Tune hyperparameters using Optuna, then hand off to run_pipeline().

    Parameters
    ----------
    model        : sklearn-compatible model instance (used to detect model type)
    X            : features (np.ndarray or pd.DataFrame)
    y            : target (np.ndarray or pd.Series)
    task_type    : "classification" or "regression" (auto-detected if None)
    n_trials     : number of Optuna trials (default 50)
    cv_folds     : folds for evaluation inside each trial (default 3, kept low for speed)
    test_size    : final train/test split size
    scale        : apply StandardScaler
    random_state : seed

    Returns
    -------
    best_model   : model fitted with best params
    scaler       : fitted scaler (or None)
    study        : the raw Optuna study object (for further inspection)
    """

    if not HAS_OPTUNA:
        print("\n  ✖  Optuna not installed. Run:  pip install optuna\n")
        return None, None, None

    # ── Coerce inputs ──────────────────────────────────────────────────────────
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if isinstance(y, pd.Series):
        y = y.values

    if task_type is None:
        unique_y = np.unique(y)
        task_type = "classification" if (
            len(unique_y) <= 20 and y.dtype in (int, np.int32, np.int64, object)
        ) else "regression"

    model_name  = type(model).__name__
    model_class = type(model)
    scoring     = "accuracy" if task_type == "classification" else "r2"

    # ── Scale once up front for all trials ────────────────────────────────────
    if scale:
        from sklearn.preprocessing import StandardScaler as _SS
        _scaler = _SS()
        X_scaled = _scaler.fit_transform(X)
    else:
        X_scaled = X
        _scaler  = None

    # ── Optuna objective ───────────────────────────────────────────────────────
    trial_log = []   # store (trial_number, score, params) for live display

    def objective(trial):
        params = _suggest_params(trial, model_name, task_type)

        # Preserve any fixed params the user set that aren't being tuned
        fixed = {k: v for k, v in model.get_params().items()
                 if k not in params and k != "random_state"}
        try:
            candidate = model_class(**params, **fixed, random_state=random_state)
        except TypeError:
            candidate = model_class(**params, **fixed)

        # Use ShuffleSplit for medium/large data inside Optuna too
        if X_scaled.shape[0] >= _MEDIUM:
            from sklearn.model_selection import ShuffleSplit
            cv_strategy = ShuffleSplit(n_splits=3, test_size=0.2, random_state=random_state)
        else:
            cv_strategy = cv_folds

        scores = cross_val_score(
            candidate, X_scaled, y,
            cv=cv_strategy, scoring=scoring, n_jobs=-1
        )
        mean_score = scores.mean()
        trial_log.append((trial.number + 1, mean_score, params))
        return mean_score

    # ── Banner ─────────────────────────────────────────────────────────────────
    banner("OPTUNA HYPERPARAMETER TUNING")

    section("Configuration")
    info("Model",         model_name)
    info("Task",          task_type.upper())
    info("Samples",       X.shape[0])
    info("Features",      X.shape[1])
    info("Trials",        n_trials)
    info("CV folds / trial", cv_folds)
    info("Scoring metric", scoring)
    info("Sampler",       "TPESampler (Tree-structured Parzen Estimator)")
    info("Pruner",        "MedianPruner (kills bad trials early)")

    section("Search Space")
    dummy_trial_params = _suggest_params(
        optuna.trial.FrozenTrial(
            number=0, state=optuna.trial.TrialState.COMPLETE,
            value=0, datetime_start=None, datetime_complete=None,
            params={}, distributions={}, user_attrs={}, system_attrs={},
            intermediate_values={}, trial_id=0,
        ),
        model_name, task_type
    )
    # Just show param names — we can't easily show ranges without running suggest
    # So we create a dummy study to extract one trial's suggested params & show names
    _dummy_study = optuna.create_study(direction="maximize")
    _dummy_trial_obj = _dummy_study.ask()
    _dummy_params = _suggest_params(_dummy_trial_obj, model_name, task_type)
    for k in _dummy_params:
        info(f"  {k}", "(will be searched)")

    section("Live Trial Progress")
    print(f"  {'Trial':>6}  {'Score':>9}  {'Best':>9}  {'Δ Best':>8}  Params (brief)")
    print("  " + "─" * 75)

    # ── Create study and run with live callback ────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    best_so_far = -np.inf

    def live_callback(study, trial):
        nonlocal best_so_far
        score = trial.value
        is_new_best = score > best_so_far
        delta = score - best_so_far if score > best_so_far else 0
        if is_new_best:
            best_so_far = score

        # Brief param string — first 3 params
        p = trial.params
        brief = "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in list(p.items())[:3])

        marker = "★ NEW BEST" if is_new_best else ""
        delta_str = f"+{delta:.4f}" if is_new_best else "       "
        print(f"  {trial.number+1:>6}  {score:>9.4f}  {best_so_far:>9.4f}  "
              f"{delta_str:>8}  {brief}  {marker}")

    study.optimize(objective, n_trials=n_trials, callbacks=[live_callback], show_progress_bar=False)

    # ── Results ────────────────────────────────────────────────────────────────
    section("Tuning Results")

    best_params = study.best_params
    best_score  = study.best_value

    info("Best CV score",  f"{best_score:.6f}")
    info("Best trial #",   study.best_trial.number + 1)
    info("Total trials",   len(study.trials))

    subsection("Best Parameters Found")
    for k, v in best_params.items():
        info(f"  {k}", v)

    subsection("Score Distribution Across All Trials")
    all_scores = [t.value for t in study.trials if t.value is not None]
    info("Mean", f"{np.mean(all_scores):.4f}")
    info("Std",  f"{np.std(all_scores):.4f}")
    info("Min",  f"{np.min(all_scores):.4f}")
    info("Max",  f"{np.max(all_scores):.4f}")

    subsection("Score Progression (best score over trials)")
    best_curve = []
    running_best = -np.inf
    for s in all_scores:
        if s > running_best:
            running_best = s
        best_curve.append(running_best)

    # Show every nth point to fit terminal
    step = max(1, len(best_curve) // 10)
    for i in range(0, len(best_curve), step):
        bar = "█" * int(best_curve[i] * 40)
        print(f"  Trial {i+1:>4}: {bar:<40} {best_curve[i]:.4f}")

    subsection("Top 5 Trials")
    sorted_trials = sorted(study.trials, key=lambda t: t.value or -np.inf, reverse=True)
    print(f"  {'Rank':<6} {'Trial':>6}  {'Score':>9}  Best Parameters")
    print("  " + "─" * 65)
    for rank, t in enumerate(sorted_trials[:5], 1):
        brief = "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in list(t.params.items())[:4])
        print(f"  {rank:<6} {t.number+1:>6}  {t.value:>9.4f}  {brief}")

    subsection("Parameter Importance (which param mattered most?)")
    try:
        importance = optuna.importance.get_param_importances(study)
        bar_chart([(k[:22], v) for k, v in importance.items()], total=1.0)
    except Exception:
        print("  (not enough trials for importance analysis)")

    # ── Build best model and run full pipeline ─────────────────────────────────
    section("Building Best Model → Handing off to run_pipeline()")

    fixed = {k: v for k, v in model.get_params().items()
             if k not in best_params and k != "random_state"}
    try:
        best_model = model_class(**best_params, **fixed, random_state=random_state)
    except TypeError:
        best_model = model_class(**best_params, **fixed)

    print(f"\n  ✅ Best model created: {model_name}({best_params})")
    print(f"  ▶  Now running full transparency pipeline with best params ...\n")

    final_model, final_scaler = run_pipeline(
        best_model, X, y,
        feature_names=feature_names,
        task_type=task_type,
        test_size=test_size,
        scale=scale,
        cv=cv_folds,
        n_walkthrough=3,
    )

    return final_model, final_scaler, study



# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — LEARNING TRACE
#  Shows HOW the model learned: tree-by-tree, split-by-split, sample-by-sample
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def _render_tree_text(lines, feature_names):
    import re
    def replacer(m):
        idx = int(m.group(1))
        return feature_names[idx] if idx < len(feature_names) else m.group(0)
    return [re.sub(r'\bf(\d+)\b', replacer, line) for line in lines]

def _print_xgb_tree(tree_text_lines, feature_names, max_nodes=30):
    lines = _render_tree_text(tree_text_lines, feature_names)
    printed = 0
    for line in lines:
        if printed >= max_nodes:
            print(f"  ... (truncated - {len(lines)-printed} more nodes)")
            break
        depth  = line.count('\t')
        indent = "  " + "  |  " * depth
        clean  = line.strip()
        if 'leaf=' in clean:
            try:
                raw  = float(clean.split('leaf=')[1].split(',')[0])
                prob = _sigmoid(raw)
                tag  = f"  <- raw={raw:.4f}  prob={prob:.4f}  {'PASS' if prob>=0.5 else 'FAIL'}"
            except Exception:
                tag = ""
            print(f"{indent}[LEAF] {clean}{tag}")
        else:
            print(f"{indent}[SPLIT] {clean}")
        printed += 1

def _xgb_learning_trace(model, X_train, y_train, X_test, y_test,
                         feature_names, task_type, n_trees_show=5):
    import re, xgboost as _xgb
    booster   = model.get_booster()
    all_trees = booster.get_dump(with_stats=True)
    n_trees   = len(all_trees)
    is_clf    = task_type == "classification"

    section("A) Tree-by-Tree Structure")
    print(f"  XGBoost built {n_trees} trees sequentially.")
    print(f"  Each tree corrects the RESIDUAL errors of all previous trees.\n")
    print(f"  Showing first {min(n_trees_show, n_trees)} trees:\n")
    for t_idx in range(min(n_trees_show, n_trees)):
        print(f"\n  {'─'*70}")
        print(f"  TREE #{t_idx+1}  --  corrects errors left by trees 1..{t_idx}")
        print(f"  {'─'*70}")
        _print_xgb_tree(all_trees[t_idx].strip().split('\n'), feature_names, max_nodes=25)
    if n_trees > n_trees_show:
        print(f"\n  ... {n_trees - n_trees_show} more trees not shown")

    section("B) Feature Usage Across ALL Trees")
    for label, itype in [("weight  (# splits)", "weight"),
                          ("gain    (avg improvement per split)", "gain"),
                          ("cover   (avg samples per split)", "cover")]:
        subsection(label)
        scores = booster.get_score(importance_type=itype)
        if not scores:
            print("  (none)")
            continue
        remapped = {}
        for k, v in scores.items():
            try:
                idx  = int(k[1:])
                name = feature_names[idx] if idx < len(feature_names) else k
            except Exception:
                name = k
            remapped[name] = v
        bar_chart([(n[:22], v) for n, v in
                   sorted(remapped.items(), key=lambda x: x[1], reverse=True)])

    section("C) How Prediction is Built -- Sample by Sample")
    print("  XGBoost starts from a BASE SCORE, then each tree adds a correction.\n")
    try:
        base_score = float(booster.attr('base_score') or 0.5)
    except Exception:
        base_score = 0.5

    for s_idx in range(min(3, len(X_test))):
        sample     = X_test[s_idx:s_idx+1]
        true_label = y_test[s_idx]
        final_pred = model.predict(sample)[0]
        final_prob = model.predict_proba(sample)[0] if is_clf else None
        print(f"\n  {'─'*65}")
        print(f"  Sample #{s_idx+1}  |  True: {true_label}  |  Pred: {final_pred}"
              + (f"  prob={final_prob[1]:.4f}" if final_prob is not None else ""))
        print(f"  {'─'*65}")
        print("  Feature values:")
        for fn, fv in zip(feature_names, sample[0]):
            print(f"    {fn:<25}: {fv:.4f}")
        print()
        dmat = _xgb.DMatrix(sample, feature_names=[f"f{i}" for i in range(sample.shape[1])])
        margins = [booster.predict(dmat, iteration_range=(0, t), output_margin=True)[0]
                   for t in range(1, n_trees+1)]
        step_t = max(1, n_trees // 10)
        print(f"  {'Tree':>6}  {'Raw Score':>10}  {'Prob':>8}  Delta")
        print("  " + "─" * 45)
        prev = 0.0
        for t in range(0, n_trees, step_t):
            m  = margins[t]
            p  = _sigmoid(m) if is_clf else m
            d  = m - prev
            ds = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
            bar = "█" * int(abs(p if is_clf else 0) * 25)
            print(f"  {t+1:>6}  {m:>10.4f}  {p:>8.4f}  {ds:<10}  {bar}")
            prev = m
        fm = margins[-1]; fp = _sigmoid(fm) if is_clf else fm
        print(f"  {'FINAL':>6}  {fm:>10.4f}  {fp:>8.4f}  --> "
              f"{'PASS' if fp>=0.5 else 'FAIL'}")

    section("D) Decision Boundaries -- All Split Thresholds Learned")
    split_info = {}
    for tree_str in all_trees:
        for line in tree_str.split('\n'):
            m = re.search(r'\[f(\d+)<([\d.\-eE+]+)\]', line)
            if m:
                idx    = int(m.group(1))
                thresh = float(m.group(2))
                name   = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
                split_info.setdefault(name, []).append(thresh)
    for feat, thresholds in sorted(split_info.items()):
        ts = sorted(set(thresholds))
        print(f"\n  Feature: {feat}  ({len(thresholds)} splits, {len(ts)} unique thresholds)")
        if feat in feature_names:
            fi = feature_names.index(feat)
            cmin, cmax = float(np.min(X_train[:,fi])), float(np.max(X_train[:,fi]))
            lw = 50
            print(f"  [{cmin:.2f}" + " "*(lw-10) + f"{cmax:.2f}]")
            nl = [" "]*lw
            for t in ts:
                pos = int((t-cmin)/(cmax-cmin+1e-9)*(lw-1))
                nl[max(0,min(lw-1,pos))] = "^"
            print("  [" + "".join(nl) + "]  (^ = cut point)")
        print(f"  Thresholds: {[round(t,4) for t in ts[:12]]}"
              + (" ..." if len(ts)>12 else ""))

    section("E) Residual Evolution -- How Errors Shrink Each Round")
    print("  Boosting = fitting each tree on the RESIDUALS of the previous ensemble.\n")
    dmat_tr = _xgb.DMatrix(X_train, feature_names=[f"f{i}" for i in range(X_train.shape[1])])
    step_t  = max(1, n_trees // 10)
    print(f"  {'Trees':>8}  {'MSE':>12}  {'MAE':>12}  Residual bar")
    print("  " + "─" * 60)
    mse = 0.0
    for t in range(1, n_trees+1, step_t):
        preds = booster.predict(dmat_tr, iteration_range=(0,t), output_margin=True)
        resid = y_train.astype(float) - (_sigmoid(preds) if is_clf else preds)
        mse   = float(np.mean(resid**2))
        mae   = float(np.mean(np.abs(resid)))
        bar   = "█" * int(min(mse*40, 40))
        print(f"  {t:>8}  {mse:>12.6f}  {mae:>12.6f}  {bar}")
    iv = float(np.mean((y_train.astype(float) - _sigmoid(0))**2)) if is_clf \
         else float(np.var(y_train))
    print(f"\n  Final MSE: {mse:.6f}  |  Baseline: {iv:.6f}  |  "
          f"Reduction: {(iv-mse)/iv*100:.1f}%")


def _gbm_learning_trace(model, X_train, y_train, X_test, y_test,
                         feature_names, task_type, n_trees_show=4):
    from sklearn.tree import export_text
    estimators = model.estimators_
    n_trees    = len(estimators)
    is_clf     = task_type == "classification"

    section("A) Staged Prediction -- Score Evolves Each Round")
    print(f"  GBM builds {n_trees} trees.  update = learning_rate({model.learning_rate}) x tree_output\n")
    staged = list(model.staged_predict(X_train) if not is_clf
                  else model.staged_predict_proba(X_train))
    step_t = max(1, n_trees // 10)
    metric = "MSE" if not is_clf else "Log-Loss"
    print(f"  {'Stage':>7}  {metric:>12}  Progress")
    print("  " + "─" * 50)
    for i in range(0, len(staged), step_t):
        p = staged[i]
        if is_clf:
            from sklearn.metrics import log_loss
            score = log_loss(y_train, p)
        else:
            score = float(np.mean((y_train - p)**2))
        bar = "█" * max(0, int((1 - min(score,1)) * 35))
        print(f"  {i+1:>7}  {score:>12.6f}  {bar}")

    section("B) First Few Tree Structures")
    print("  Each tree learns the NEGATIVE GRADIENT (pseudo-residuals) of the loss.\n")
    for t_idx in range(min(n_trees_show, n_trees)):
        tree = estimators[t_idx][0]
        print(f"\n  {'─'*60}")
        print(f"  TREE #{t_idx+1}  (depth={tree.tree_.max_depth}, nodes={tree.tree_.node_count})")
        text = export_text(tree, feature_names=feature_names, max_depth=3)
        for line in text.split('\n')[:25]:
            print(f"  {line}")

    section("C) Decision Boundaries -- Split Thresholds Per Feature")
    split_info = {}
    for est_arr in estimators:
        t = est_arr[0].tree_
        for node in range(t.node_count):
            if t.children_left[node] != -1:
                split_info.setdefault(feature_names[t.feature[node]], []).append(t.threshold[node])
    for feat, thresholds in sorted(split_info.items()):
        ts = sorted(set(thresholds))
        fi = feature_names.index(feat) if feat in feature_names else -1
        print(f"\n  Feature: {feat}  ({len(thresholds)} splits, {len(ts)} unique)")
        if fi >= 0:
            cmin, cmax = float(np.min(X_train[:,fi])), float(np.max(X_train[:,fi]))
            lw = 50
            print(f"  [{cmin:.2f}" + " "*(lw-10) + f"{cmax:.2f}]")
            nl = [" "]*lw
            for t in ts:
                pos = int((t-cmin)/(cmax-cmin+1e-9)*(lw-1))
                nl[max(0,min(lw-1,pos))] = "^"
            print("  [" + "".join(nl) + "]")
        print(f"  Thresholds: {[round(t,3) for t in ts[:10]]}"
              + (" ..." if len(ts)>10 else ""))


def _tree_learning_trace(model, X_train, y_train, X_test, y_test,
                          feature_names, task_type):
    from sklearn.tree import export_text
    tree   = model.tree_
    is_clf = task_type == "classification"

    section("A) Full Decision Tree -- Every Split and Leaf")
    print(f"  Depth={tree.max_depth}  Nodes={tree.node_count}  "
          f"Leaves={int(np.sum(tree.children_left==-1))}\n")
    for line in export_text(model, feature_names=feature_names,
                             max_depth=6, show_weights=True).split('\n'):
        print(f"  {line}")

    section("B) Decision Path -- How Each Test Sample Travels the Tree")
    node_indicator = model.decision_path(X_test)
    for s_idx in range(min(3, len(X_test))):
        node_ids = node_indicator[s_idx].indices
        print(f"\n  Sample #{s_idx+1}  true={y_test[s_idx]}  "
              f"pred={model.predict(X_test[s_idx:s_idx+1])[0]}")
        print(f"  {'─'*55}")
        for i, node_id in enumerate(node_ids):
            is_leaf = tree.children_left[node_id] == -1
            indent  = "  " + "  "*i
            if is_leaf:
                val = tree.value[node_id].flatten()
                print(f"{indent}[LEAF] node {node_id}  value={np.round(val,3)}  "
                      f"n={tree.n_node_samples[node_id]}")
            else:
                feat  = feature_names[tree.feature[node_id]]
                thr   = tree.threshold[node_id]
                sv    = X_test[s_idx, tree.feature[node_id]]
                went  = "LEFT (TRUE)" if sv <= thr else "RIGHT (FALSE)"
                print(f"{indent}[SPLIT] node {node_id}: {feat}={sv:.4f} <= {thr:.4f}? -> {went}")

    section("C) Feature Impurity Reduction")
    imps  = model.feature_importances_
    pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
    for name, imp in pairs:
        bar = "█" * int(imp * 50)
        print(f"  {name:<25} {imp:>10.4f}  {bar}")


def _forest_learning_trace(model, X_train, y_train, X_test, y_test,
                             feature_names, task_type):
    estimators = model.estimators_
    n_trees    = len(estimators)
    is_clf     = task_type == "classification"

    section("A) Each Tree Sees Different Bootstrap Data")
    print(f"  {n_trees} trees, each trained on a random 63% bootstrap sample.\n")
    print(f"  {'Tree':>6}  {'Depth':>7}  {'Nodes':>7}  {'Leaves':>8}  Score")
    print("  " + "─" * 55)
    for i, est in enumerate(estimators[:20]):
        t        = est.tree_
        n_leaves = int(np.sum(t.children_left == -1))
        pred_i   = est.predict(X_train)
        score    = float(np.mean(pred_i==y_train)) if is_clf else \
                   1 - np.sum((y_train-pred_i)**2) / (np.sum((y_train-np.mean(y_train))**2)+1e-9)
        bar = "█" * min(t.node_count//2, 25)
        print(f"  {i+1:>6}  {t.max_depth:>7}  {t.node_count:>7}  {n_leaves:>8}  "
              f"{score:>7.4f}  {bar}")
    if n_trees > 20:
        print(f"  ... ({n_trees-20} more trees)")

    section("B) Voting / Averaging -- How Trees Combine")
    for s_idx in range(min(3, len(X_test))):
        sample     = X_test[s_idx:s_idx+1]
        final_pred = model.predict(sample)[0]
        print(f"\n  Sample #{s_idx+1}  True={y_test[s_idx]}  Final pred={final_pred}")
        print(f"  {'─'*50}")
        if is_clf:
            votes = [est.predict(sample)[0] for est in estimators]
            vc    = Counter(votes)
            total = len(votes)
            for cls, cnt in sorted(vc.items()):
                bar    = "█" * int(cnt/total*40)
                marker = " <- WINNER" if cls==final_pred else ""
                print(f"    Class {cls}: {bar:<40} {cnt}/{total} ({cnt/total*100:.1f}%){marker}")
        else:
            preds_all = np.array([est.predict(sample)[0] for est in estimators])
            print(f"    min={preds_all.min():.4f}  max={preds_all.max():.4f}  "
                  f"std={preds_all.std():.4f}  mean={preds_all.mean():.4f}")
            hist, edges = np.histogram(preds_all, bins=8)
            for j in range(len(hist)):
                bar = "█" * int(hist[j]/hist.max()*20) if hist.max()>0 else ""
                print(f"    [{edges[j]:6.2f}-{edges[j+1]:6.2f}]: {bar} ({hist[j]} trees)")

    section("C) Decision Boundaries -- Split Thresholds Across All Trees")
    split_info = {}
    for est in estimators:
        t = est.tree_
        for node in range(t.node_count):
            if t.children_left[node] != -1:
                split_info.setdefault(feature_names[t.feature[node]], []).append(t.threshold[node])
    for feat, thresholds in sorted(split_info.items()):
        ts = sorted(set(thresholds))
        fi = feature_names.index(feat) if feat in feature_names else -1
        print(f"\n  Feature: {feat}  ({len(thresholds)} splits, {len(ts)} unique)")
        if fi >= 0:
            cmin, cmax = float(np.min(X_train[:,fi])), float(np.max(X_train[:,fi]))
            lw = 50
            print(f"  [{cmin:.2f}" + " "*(lw-10) + f"{cmax:.2f}]")
            nl = [" "]*lw
            for t in ts:
                pos = int((t-cmin)/(cmax-cmin+1e-9)*(lw-1))
                nl[max(0,min(lw-1,pos))] = "^"
            print("  [" + "".join(nl) + "]")
        print(f"  Cuts: {[round(t,3) for t in ts[:10]]}" + (" ..." if len(ts)>10 else ""))


def _linear_learning_trace(model, X_train, y_train, X_test, y_test,
                             feature_names, task_type):
    is_clf    = task_type == "classification"
    coefs     = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
    intercept = float(np.atleast_1d(model.intercept_)[0])

    section("A) What the Model Learned -- Coefficient Meaning")
    print("  Each coefficient = prediction change per 1-unit increase in that feature.\n")
    print(f"  {'Feature':<25} {'Coef':>12}  {'|Effect|':>10}  Direction")
    print("  " + "─" * 65)
    for name, c in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        bar   = "█" * int(min(abs(c)*10, 35))
        direc = "UP  (pushes positive)" if c > 0 else "DOWN (pushes negative)"
        print(f"  {name:<25} {c:>12.4f}  {abs(c):>10.4f}  {direc}  {bar}")
    print(f"\n  Intercept (bias): {intercept:.4f}")

    section("B) Sample-by-Sample Score Decomposition")
    print("  Watch how each feature contributes to the final score / probability.\n")
    for s_idx in range(min(5, len(X_test))):
        sample    = X_test[s_idx]
        raw_score = intercept
        print(f"  Sample #{s_idx+1}  (true={y_test[s_idx]})")
        print(f"    Intercept:              {intercept:+.4f}")
        for name, c, val in zip(feature_names, coefs, sample):
            contrib   = c * val
            raw_score += contrib
            bar = "█" * int(min(abs(contrib)*5, 25))
            print(f"    {name:<22}: {val:7.3f} x {c:+.4f} = {contrib:+.4f}  {bar}")
        if is_clf:
            prob = _sigmoid(raw_score)
            pred = 1 if prob >= 0.5 else 0
            print(f"    {'─'*45}")
            print(f"    Raw logit: {raw_score:+.4f}  ->  prob={prob:.4f}  "
                  f"-> pred={pred} ({'correct' if pred==y_test[s_idx] else 'wrong'})\n")
        else:
            print(f"    {'─'*45}")
            print(f"    yhat={raw_score:.4f}  true={y_test[s_idx]:.4f}  "
                  f"error={abs(raw_score-y_test[s_idx]):.4f}\n")

    section("C) True Feature Impact = |coef| x std")
    print("  Coefficient alone is misleading if features have different scales.\n")
    stds    = X_train.std(axis=0)
    impacts = sorted(zip(feature_names, coefs, stds),
                     key=lambda x: abs(x[1])*x[2], reverse=True)
    print(f"  {'Feature':<25} {'Impact':>10}  {'coef':>9}  {'std':>9}")
    print("  " + "─" * 60)
    for name, c, s in impacts:
        bar = "█" * int(abs(c)*s*20)
        print(f"  {name:<25} {abs(c)*s:>10.4f}  {c:>9.4f}  {s:>9.4f}  {bar}")



# ══════════════════════════════════════════════════════════════════════════════
#  LEARNING TRACE — SVM
# ══════════════════════════════════════════════════════════════════════════════

def _svm_learning_trace(model, X_train, y_train, X_test, y_test,
                         feature_names, task_type):
    is_clf  = task_type == "classification"
    kernel  = model.kernel
    n_sv    = len(model.support_vectors_)
    n_train = X_train.shape[0]

    # ── A) What SVM learned ───────────────────────────────────────────────────
    section("A) What SVM Learned — The Support Vectors")
    print(textwrap.fill(
        "  SVM ignores most training samples. It only cares about the samples "
        "closest to the decision boundary — the support vectors. Everything "
        "the model knows is encoded in these points alone.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    info("Kernel",             kernel)
    info("C (regularization)", model.C)
    info("Total train samples",n_train)
    info("Support vectors",    n_sv)
    info("SV ratio",           f"{n_sv/n_train*100:.1f}%  of training data")
    if hasattr(model, "n_support_") and is_clf:
        for i, cls in enumerate(model.classes_):
            info(f"  SVs for class {cls}", model.n_support_[i])

    subsection("Support Vector Values (first 5 SVs, all features)")
    svs = model.support_vectors_
    print(f"  {'SV#':<6}", end="")
    for fn in feature_names[:6]:
        print(f"  {fn[:12]:>13}", end="")
    print()
    print("  " + "─" * (6 + 15 * min(6, len(feature_names))))
    for i, sv in enumerate(svs[:5]):
        print(f"  {i+1:<6}", end="")
        for val in sv[:6]:
            print(f"  {val:>13.4f}", end="")
        print()
    if len(svs) > 5:
        print(f"  ... ({len(svs)-5} more support vectors)")

    # ── B) Dual coefficients — how each SV votes ─────────────────────────────
    section("B) Dual Coefficients — How Each Support Vector Votes")
    print(textwrap.fill(
        "  Each support vector has a dual coefficient (alpha * y). "
        "Positive = votes for positive class, negative = votes for negative class. "
        "Magnitude = how strongly this SV influences the boundary.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    dual_coefs = model.dual_coef_.flatten()[:20]  # show first 20
    print(f"  {'SV#':<6} {'Dual Coef':>12}  {'Vote':>10}  Influence bar")
    print("  " + "─" * 55)
    for i, dc in enumerate(dual_coefs):
        bar  = "█" * int(min(abs(dc) * 10, 30))
        vote = "POSITIVE +" if dc > 0 else "NEGATIVE -"
        print(f"  {i+1:<6} {dc:>12.4f}  {vote:>10}  {bar}")
    if len(model.dual_coef_.flatten()) > 20:
        print(f"  ... ({len(model.dual_coef_.flatten())-20} more)")

    # ── C) Decision function — raw scores per sample ──────────────────────────
    section("C) Decision Function — Raw Score for Each Test Sample")
    print(textwrap.fill(
        "  The decision function computes a signed distance from the hyperplane. "
        "Positive = class 1 side, Negative = class 0 side. "
        "Larger magnitude = more confident prediction.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    dec_fn   = model.decision_function(X_test)
    y_pred   = model.predict(X_test)
    max_abs  = max(abs(dec_fn).max(), 1e-9)

    print(f"  {'#':<5} {'Dec Score':>11}  {'Pred':>6}  {'True':>6}  "
          f"{'Result':<10}  Distance bar")
    print("  " + "─" * 70)
    for i, (score, pred, true) in enumerate(zip(dec_fn, y_pred, y_test)):
        correct = "✅ correct" if pred == true else "❌ wrong"
        bar_len = int(abs(score) / max_abs * 30)
        bar     = "█" * bar_len
        side    = "+" if score >= 0 else "-"
        print(f"  {i+1:<5} {score:>11.4f}  {str(pred):>6}  {str(true):>6}  "
              f"{correct:<10}  {side}{bar}")

    # ── D) Margin and boundary ────────────────────────────────────────────────
    section("D) Margin — How Wide is the Decision Boundary?")
    print(textwrap.fill(
        "  The margin is the distance between the two margin hyperplanes "
        "(the gap SVM maximises). A wider margin = better generalisation. "
        "The decision boundary sits exactly in the middle.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    if kernel == "linear":
        w      = model.coef_[0]
        margin = 2.0 / (np.linalg.norm(w) + 1e-12)
        info("Margin width (2/||w||)", f"{margin:.6f}")
        info("||w|| (weight norm)",   f"{np.linalg.norm(w):.6f}")
        print()
        subsection("Linear Weight Vector (= hyperplane normal)")
        pairs = sorted(zip(feature_names, w), key=lambda x: abs(x[1]), reverse=True)
        for name, wi in pairs:
            bar  = "█" * int(min(abs(wi) * 15, 35))
            dirn = "→ class 1" if wi > 0 else "→ class 0"
            print(f"  {name:<25} {wi:>10.4f}  {dirn}  {bar}")
    else:
        info("Kernel",   kernel)
        info("Gamma",    model.gamma)
        info("Note", f"Non-linear kernel — no explicit weight vector. "
                     f"Boundary defined implicitly via {n_sv} support vectors.")
        print()
        subsection("Kernel trick — what each kernel does")
        kernel_desc = {
            "rbf":  "RBF: exp(-gamma*||x-sv||^2)  Gaussian bump around each SV. "
                    "Good for non-linear blobs.",
            "poly": "Poly: (gamma*x·sv + r)^degree  Polynomial surface. "
                    "Good for structured patterns.",
            "sigmoid": "Sigmoid: tanh(gamma*x·sv + r)  Similar to neural net.",
        }
        desc = kernel_desc.get(kernel, f"{kernel} kernel")
        print(textwrap.fill(f"  {desc}", width=WIDTH,
                            initial_indent="  ", subsequent_indent="  "))

    # ── E) Per-sample breakdown — which SVs fired ─────────────────────────────
    section("E) Sample Prediction Breakdown — Which SVs Contributed")
    print("  For each test sample: compute kernel(sample, each_SV) × dual_coef.\n")
    for s_idx in range(min(3, len(X_test))):
        sample = X_test[s_idx:s_idx+1]
        score  = float(model.decision_function(sample)[0])
        pred   = model.predict(sample)[0]
        print(f"\n  Sample #{s_idx+1}  |  true={y_test[s_idx]}  pred={pred}  "
              f"decision_score={score:+.4f}")
        print(f"  {'─'*55}")
        # Kernel values between sample and each SV
        from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
        if kernel == "rbf":
            gamma = (model.gamma if isinstance(model.gamma, float)
                     else 1.0 / X_train.shape[1])
            K = rbf_kernel(sample, model.support_vectors_, gamma=gamma).flatten()
        elif kernel == "linear":
            K = linear_kernel(sample, model.support_vectors_).flatten()
        elif kernel == "poly":
            K = polynomial_kernel(sample, model.support_vectors_,
                                  degree=model.degree, gamma=model.gamma,
                                  coef0=model.coef0).flatten()
        else:
            K = np.ones(len(model.support_vectors_))

        dc        = model.dual_coef_.flatten()
        contribs  = dc * K
        top_idx   = np.argsort(np.abs(contribs))[::-1][:5]
        running   = float(model.intercept_[0]) if hasattr(model,'intercept_') else 0.0
        print(f"  Intercept (bias): {running:+.4f}")
        for rank, sv_i in enumerate(top_idx):
            contrib   = float(contribs[sv_i])
            running  += contrib
            bar       = "█" * int(min(abs(contrib) * 20, 30))
            print(f"  SV#{sv_i+1:<4} K={K[sv_i]:.4f}  α={dc[sv_i]:+.4f}  "
                  f"contrib={contrib:+.6f}  running={running:+.4f}  {bar}")
        print(f"  Final score: {score:+.4f}  →  pred={'1 (PASS)' if score>=0 else '0 (FAIL)'}")


# ══════════════════════════════════════════════════════════════════════════════
#  LEARNING TRACE — KNN
# ══════════════════════════════════════════════════════════════════════════════

def _knn_learning_trace(model, X_train, y_train, X_test, y_test,
                         feature_names, task_type):
    is_clf = task_type == "classification"
    k      = model.n_neighbors
    metric = model.metric

    # ── A) What KNN "learned" ─────────────────────────────────────────────────
    section("A) What KNN Learned — The Entire Training Set IS the Model")
    print(textwrap.fill(
        "  KNN has no training phase. There are no weights, coefficients, or trees. "
        "The model IS the training data. Every prediction is made at query time "
        "by searching the stored samples.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    info("k (neighbors)",     k)
    info("Distance metric",   metric)
    info("Weighting",         model.weights)
    info("Algorithm",         model.algorithm)
    info("Stored samples",    X_train.shape[0])
    info("Stored features",   X_train.shape[1])
    info("Memory for dataset",f"~{X_train.nbytes / 1024:.1f} KB")

    subsection("Training Set Summary (what KNN memorised)")
    print(f"  {'Feature':<25} {'Min':>9}  {'Max':>9}  {'Mean':>9}  {'Std':>9}")
    print("  " + "─" * 60)
    for i, name in enumerate(feature_names):
        col = X_train[:, i]
        print(f"  {name:<25} {col.min():>9.3f}  {col.max():>9.3f}  "
              f"{col.mean():>9.3f}  {col.std():>9.3f}")

    # ── B) Sample-by-sample: who are the neighbours? ──────────────────────────
    section("B) Prediction Walkthrough — Finding the k Nearest Neighbours")
    print(f"  For each test sample: compute distance to ALL {X_train.shape[0]} "
          f"training points, pick k={k} closest, then vote/average.\n")

    from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

    for s_idx in range(min(3, len(X_test))):
        sample     = X_test[s_idx:s_idx+1]
        true_label = y_test[s_idx]
        pred       = model.predict(sample)[0]

        # Get neighbour indices and distances
        dists, indices = model.kneighbors(sample, n_neighbors=min(k, X_train.shape[0]))
        dists   = dists.flatten()
        indices = indices.flatten()

        print(f"\n  {'─'*65}")
        print(f"  Sample #{s_idx+1}  |  True={true_label}  |  Pred={pred}  "
              f"({'✅ correct' if pred==true_label else '❌ wrong'})")
        print(f"  {'─'*65}")
        print(f"  Feature values of query point:")
        for fn, fv in zip(feature_names, sample[0]):
            print(f"    {fn:<25}: {fv:.4f}")

        print(f"\n  The {k} nearest neighbours found:\n")
        print(f"  {'Rank':<6} {'Train#':>7}  {'Distance':>10}  "
              f"{'Label':>7}  Feature values (first 4)")
        print("  " + "─" * 65)

        # Distance bar scale
        max_d = dists.max() if dists.max() > 0 else 1.0
        for rank, (dist, idx) in enumerate(zip(dists, indices)):
            label    = y_train[idx]
            bar_fill = int((1 - dist/max_d) * 25)  # closer = longer bar
            bar      = "█" * max(bar_fill, 1)
            feat_str = "  ".join(f"{v:.2f}" for v in X_train[idx, :4])
            print(f"  {rank+1:<6} {idx:>7}  {dist:>10.4f}  "
                  f"{str(label):>7}  [{feat_str}]  {bar}")

        # Voting detail
        neighbour_labels = y_train[indices]
        print(f"\n  Voting / Averaging:")
        if is_clf:
            vote_counts = Counter(neighbour_labels)
            for cls, cnt in sorted(vote_counts.items()):
                bar    = "█" * int(cnt/k*30)
                winner = " ← WINNER" if cls == pred else ""
                if model.weights == "distance":
                    w_sum = sum(1.0/(d+1e-9) for d, l in zip(dists, neighbour_labels) if l==cls)
                    print(f"    Class {cls}: {cnt}/{k} votes  "
                          f"(weighted sum={w_sum:.4f}){winner}  {bar}")
                else:
                    print(f"    Class {cls}: {cnt}/{k} votes{winner}  {bar}")
        else:
            neighbour_vals = y_train[indices].astype(float)
            if model.weights == "distance":
                weights = 1.0 / (dists + 1e-9)
                weights /= weights.sum()
                wavg = float(np.dot(weights, neighbour_vals))
                print(f"    Neighbour targets: {np.round(neighbour_vals, 3)}")
                print(f"    Weights (1/dist):  {np.round(weights, 4)}")
                print(f"    Weighted average:  {wavg:.4f}  (= prediction)")
            else:
                print(f"    Neighbour targets: {np.round(neighbour_vals, 3)}")
                print(f"    Simple average:    {neighbour_vals.mean():.4f}  (= prediction)")

    # ── C) Distance sensitivity ───────────────────────────────────────────────
    section("C) Distance Sensitivity — How k Affects the Decision")
    print(f"  Testing k = 1, 3, 5, 7 on test set to show k sensitivity:\n")
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    scoring_fn = (lambda m: np.mean(m.predict(X_test) == y_test)) if is_clf \
                 else (lambda m: 1 - np.sum((y_test - m.predict(X_test))**2) /
                                 (np.sum((y_test - np.mean(y_test))**2) + 1e-9))
    cls_knn = KNeighborsClassifier if is_clf else KNeighborsRegressor
    print(f"  {'k':>5}  {'Score':>9}  Confidence bar")
    print("  " + "─" * 45)
    for k_test in [1, 3, 5, 7, 10, 15]:
        if k_test > X_train.shape[0]:
            break
        tmp = cls_knn(n_neighbors=k_test, weights=model.weights, metric=metric)
        tmp.fit(X_train, y_train)
        sc  = scoring_fn(tmp)
        bar = "█" * int(max(sc, 0) * 35)
        tag = " ← current k" if k_test == k else ""
        print(f"  {k_test:>5}  {sc:>9.4f}  {bar}{tag}")

    # ── D) Feature distance contribution ──────────────────────────────────────
    section("D) Feature Distance Contribution — Which Features Drive Similarity")
    print("  For the first test sample: per-feature distance to each neighbour.\n")
    if len(X_test) > 0:
        sample  = X_test[0:1]
        _, idxs = model.kneighbors(sample, n_neighbors=min(k, X_train.shape[0]))
        idxs    = idxs.flatten()
        diffs   = np.abs(X_train[idxs] - sample)   # shape (k, n_features)
        mean_diff = diffs.mean(axis=0)
        total     = mean_diff.sum() + 1e-9
        print(f"  {'Feature':<25} {'Avg |diff|':>12}  {'% of dist':>10}  Contribution")
        print("  " + "─" * 65)
        for name, md in sorted(zip(feature_names, mean_diff),
                                key=lambda x: x[1], reverse=True):
            pct = md / total * 100
            bar = "█" * int(pct * 0.5)
            print(f"  {name:<25} {md:>12.4f}  {pct:>9.1f}%  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  LEARNING TRACE — LightGBM
# ══════════════════════════════════════════════════════════════════════════════

def _lgbm_learning_trace(model, X_train, y_train, X_test, y_test,
                          feature_names, task_type, n_trees_show=5):
    is_clf  = task_type == "classification"

    # ── A) Tree-by-tree dump ───────────────────────────────────────────────────
    section("A) Tree-by-Tree Structure")
    print(textwrap.fill(
        "  LightGBM grows trees LEAF-WISE (best-first), not level-wise like XGBoost. "
        "This means it can grow very deep on one branch while ignoring others, "
        "making it faster but more prone to overfit on small data.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()

    try:
        tree_info = model.booster_.dump_model()
        trees     = tree_info.get("tree_info", [])
        n_trees   = len(trees)
        info("Total trees built", n_trees)
        info("Num leaves",        model.num_leaves)
        info("Max depth",         model.max_depth)
        info("Learning rate",     model.learning_rate)
        print()

        def _print_lgbm_node(node, depth=0, max_depth=4):
            if depth > max_depth:
                print("  " + "  "*depth + "...")
                return
            indent = "  " + "  "*depth
            if "leaf_value" in node:
                lv   = node["leaf_value"]
                prob = _sigmoid(lv) if is_clf else lv
                tag  = f"  prob={prob:.4f} {'PASS' if prob>=0.5 else 'FAIL'}" if is_clf \
                       else f"  value={lv:.4f}"
                n    = node.get("leaf_count", "?")
                print(f"{indent}[LEAF] value={lv:.4f}  samples={n}{tag}")
            else:
                feat   = node.get("split_feature", "?")
                fname  = feature_names[feat] if isinstance(feat,int) and feat<len(feature_names) else str(feat)
                thresh = node.get("threshold", "?")
                gain   = node.get("split_gain", 0)
                n      = node.get("internal_count", "?")
                print(f"{indent}[SPLIT] {fname} <= {thresh}  "
                      f"gain={gain:.4f}  samples={n}")
                if "left_child" in node:
                    print(f"{indent}  TRUE  ->")
                    _print_lgbm_node(node["left_child"], depth+1, max_depth)
                if "right_child" in node:
                    print(f"{indent}  FALSE ->")
                    _print_lgbm_node(node["right_child"], depth+1, max_depth)

        for t_idx in range(min(n_trees_show, n_trees)):
            print(f"\n  {'─'*65}")
            print(f"  TREE #{t_idx+1}  —  leaf-wise growth, corrects residuals of trees 1..{t_idx}")
            print(f"  {'─'*65}")
            tree_struct = trees[t_idx].get("tree_structure", {})
            _print_lgbm_node(tree_struct, depth=0, max_depth=4)

        if n_trees > n_trees_show:
            print(f"\n  ... {n_trees - n_trees_show} more trees not shown")

    except Exception as e:
        print(f"  (Tree dump unavailable: {e})")

    # ── B) Feature importance — three types ───────────────────────────────────
    section("B) Feature Importance — Three Views")
    for itype in ["split", "gain"]:
        subsection(f"Importance by {itype}")
        try:
            scores = model.booster_.feature_importance(importance_type=itype)
            pairs  = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
            bar_chart([(n[:22], v) for n, v in pairs if v > 0])
        except Exception as e:
            print(f"  (unavailable: {e})")

    subsection("Permutation importance (actual performance drop)")
    perm_importance(model, X_test, y_test, feature_names, task_type,
                    _print_banner=False)

    # ── C) Prediction build-up sample by sample ────────────────────────────────
    section("C) How Prediction is Built — Sample by Sample")
    print("  LightGBM accumulates raw scores from each tree then applies sigmoid.\n")

    try:
        raw_scores_all = model.booster_.predict(X_test, raw_score=True)
    except Exception:
        raw_scores_all = None

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if is_clf else None

    for s_idx in range(min(3, len(X_test))):
        sample     = X_test[s_idx:s_idx+1]
        true_label = y_test[s_idx]
        final_pred = y_pred[s_idx]

        print(f"\n  {'─'*65}")
        print(f"  Sample #{s_idx+1}  |  True={true_label}  |  Pred={final_pred}"
              + (f"  prob={y_proba[s_idx][1]:.4f}" if y_proba is not None else ""))
        print(f"  Feature values:")
        for fn, fv in zip(feature_names, sample[0]):
            print(f"    {fn:<25}: {fv:.4f}")

        if raw_scores_all is not None:
            raw = float(raw_scores_all[s_idx]) if raw_scores_all.ndim == 1 \
                  else float(raw_scores_all[s_idx, 1])
            prob = _sigmoid(raw) if is_clf else raw
            print(f"\n  Accumulated raw score:  {raw:+.4f}")
            print(f"  {'sigmoid(raw)' if is_clf else 'Final value'}:   {prob:.4f}")
            print(f"  Decision:               {'PASS ✅' if (prob>=0.5 if is_clf else True) else 'FAIL ❌'}")

        # Stage-wise prediction using staged_predict on booster
        print(f"\n  Score evolution every 10% of trees:")
        print(f"  {'Trees':>7}  {'Raw Score':>11}  {'Prob':>8}  Progress bar")
        print("  " + "─" * 50)
        n_trees = model.n_estimators_
        step    = max(1, n_trees // 10)
        for t in range(step, n_trees+1, step):
            try:
                raw_t = model.booster_.predict(sample, num_iteration=t, raw_score=True)
                raw_t = float(raw_t) if np.ndim(raw_t)==0 else float(raw_t.flat[0])
                prob_t = _sigmoid(raw_t) if is_clf else raw_t
                bar    = "█" * int(abs(prob_t if is_clf else 0) * 25)
                print(f"  {t:>7}  {raw_t:>11.4f}  {prob_t:>8.4f}  {bar}")
            except Exception:
                break

    # ── D) Split thresholds learned ───────────────────────────────────────────
    section("D) Decision Boundaries — All Split Thresholds Per Feature")
    print("  Every value LightGBM chose as a cut point across all trees.\n")
    try:
        split_info = {}
        for tree_dict in trees:
            def _collect_splits(node):
                if "split_feature" in node:
                    feat   = node["split_feature"]
                    thresh = node.get("threshold", None)
                    fname  = feature_names[feat] if isinstance(feat,int) and feat<len(feature_names) else str(feat)
                    if thresh is not None:
                        try:
                            split_info.setdefault(fname, []).append(float(thresh))
                        except (ValueError, TypeError):
                            pass
                    if "left_child" in node:
                        _collect_splits(node["left_child"])
                    if "right_child" in node:
                        _collect_splits(node["right_child"])
            _collect_splits(tree_dict.get("tree_structure", {}))

        for feat, thresholds in sorted(split_info.items()):
            ts = sorted(set(thresholds))
            fi = feature_names.index(feat) if feat in feature_names else -1
            print(f"\n  Feature: {feat}  ({len(thresholds)} splits, {len(ts)} unique thresholds)")
            if fi >= 0:
                cmin = float(np.min(X_train[:, fi]))
                cmax = float(np.max(X_train[:, fi]))
                lw   = 50
                print(f"  [{cmin:.2f}" + " "*(lw-10) + f"{cmax:.2f}]")
                nl = [" "]*lw
                for t in ts:
                    pos = int((t-cmin)/(cmax-cmin+1e-9)*(lw-1))
                    nl[max(0,min(lw-1,pos))] = "^"
                print("  [" + "".join(nl) + "]  (^ = cut point)")
            print(f"  Thresholds: {[round(t,4) for t in ts[:12]]}"
                  + (" ..." if len(ts)>12 else ""))

    except Exception as e:
        print(f"  (split threshold extraction unavailable: {e})")

    # ── E) Residual evolution ─────────────────────────────────────────────────
    section("E) Residual Evolution — How Errors Shrink Round by Round")
    print("  LightGBM is a boosting algorithm — each tree fits the RESIDUALS.\n")
    n_trees = model.n_estimators_
    step    = max(1, n_trees // 10)
    print(f"  {'Trees':>8}  {'MSE':>12}  {'MAE':>12}  Residual bar")
    print("  " + "─" * 55)
    mse = 0.0
    for t in range(step, n_trees+1, step):
        try:
            preds_t = model.booster_.predict(X_train, num_iteration=t,
                                              raw_score=True)
            if preds_t.ndim > 1:
                preds_t = preds_t[:, 1]
            vals    = _sigmoid(preds_t) if is_clf else preds_t
            resid   = y_train.astype(float) - vals
            mse     = float(np.mean(resid**2))
            mae     = float(np.mean(np.abs(resid)))
            bar     = "█" * int(min(mse*40, 40))
            print(f"  {t:>8}  {mse:>12.6f}  {mae:>12.6f}  {bar}")
        except Exception:
            break
    base_var = float(np.var(y_train))
    if base_var > 0:
        print(f"\n  Reduction from baseline variance: {(base_var-mse)/base_var*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  LEARNING TRACE — SGD  (SGDClassifier / SGDRegressor)
# ══════════════════════════════════════════════════════════════════════════════

def _sgd_learning_trace(model, X_train, y_train, X_test, y_test,
                         feature_names, task_type):
    is_clf    = task_type == "classification"
    coefs     = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
    intercept = float(np.atleast_1d(model.intercept_)[0])
    loss      = model.loss
    lr        = model.learning_rate
    eta0      = model.eta0
    alpha     = model.alpha

    # ── A) What SGD learned ───────────────────────────────────────────────────
    section("A) What SGD Learned — Weights Found by Gradient Descent")
    print(textwrap.fill(
        "  SGD finds weights iteratively: for each sample it computes the gradient "
        "of the loss, then nudges the weights in the opposite direction. "
        "Unlike batch gradient descent it uses ONE sample at a time — noisy but fast.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    info("Loss function",    loss)
    info("Learning rate",    lr)
    info("eta0 (initial lr)",eta0)
    info("Regularization",   f"alpha={alpha}  penalty={model.penalty}")
    info("Max iterations",   model.max_iter)
    info("Actual epochs",    getattr(model, "n_iter_", "N/A"))
    info("Weight updates",   getattr(model, "t_", "N/A"))

    subsection("Learned Weights (coefficients)")
    print(f"  {'Feature':<25} {'Weight':>12}  {'|Weight|':>10}  Direction")
    print("  " + "─" * 65)
    pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    for name, c in pairs:
        bar  = "█" * int(min(abs(c)*10, 35))
        dirn = "↑ pushes positive" if c > 0 else "↓ pushes negative"
        print(f"  {name:<25} {c:>12.6f}  {abs(c):>10.6f}  {dirn}  {bar}")
    print(f"\n  Intercept (bias term): {intercept:.6f}")

    # ── B) SGD update rule visualised ────────────────────────────────────────
    section("B) SGD Update Rule — What Happened at Each Step")
    print(textwrap.fill(
        f"  Update rule for '{loss}' loss:  w = w - eta * grad_w(loss(w, x_i, y_i)) "
        f"- alpha * w  (L2 regularization penalty).",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    print("  Simulating 10 SGD steps on training samples to show the update process:\n")

    from sklearn.linear_model import SGDClassifier as _SGDC, SGDRegressor as _SGDR
    # Simulate mini SGD manually for visibility
    w      = np.zeros(X_train.shape[1])
    b      = 0.0
    eta    = float(eta0) if isinstance(eta0, (int, float)) and eta0 > 0 else 0.01
    reg    = float(alpha)
    n_show = min(10, X_train.shape[0])

    print(f"  {'Step':<6}  {'Sample#':>8}  {'Loss':>10}  {'||w||':>9}  "
          f"{'Bias':>9}  Weight update bar")
    print("  " + "─" * 70)

    rng   = np.random.default_rng(42)
    order = rng.permutation(X_train.shape[0])

    for step in range(n_show):
        i       = order[step]
        xi      = X_train[i]
        yi      = float(y_train[i])
        raw     = float(np.dot(w, xi) + b)

        # Compute gradient depending on loss type
        if loss in ("hinge", "squared_hinge"):
            margin  = yi * raw if is_clf else raw - yi
            if loss == "hinge":
                grad_w = -yi * xi if margin < 1 else np.zeros_like(xi)
                grad_b = -yi       if margin < 1 else 0.0
                loss_v = max(0.0, 1.0 - margin)
            else:
                grad_w = -2*yi*(1-margin)*xi if margin < 1 else np.zeros_like(xi)
                grad_b = -2*yi*(1-margin)     if margin < 1 else 0.0
                loss_v = max(0.0, 1.0-margin)**2
        elif loss == "log_loss":
            prob    = _sigmoid(raw)
            err     = prob - yi
            grad_w  = err * xi
            grad_b  = err
            loss_v  = -(yi*np.log(prob+1e-9) + (1-yi)*np.log(1-prob+1e-9))
        elif loss in ("squared_error", "squared_loss"):
            err     = raw - yi
            grad_w  = err * xi
            grad_b  = err
            loss_v  = 0.5 * err**2
        elif loss == "huber":
            err     = raw - yi
            delta   = model.epsilon if hasattr(model, "epsilon") else 0.1
            if abs(err) <= delta:
                grad_w = err * xi; grad_b = err; loss_v = 0.5*err**2
            else:
                grad_w = delta*np.sign(err)*xi; grad_b = delta*np.sign(err)
                loss_v = delta*abs(err) - 0.5*delta**2
        else:
            # fallback: squared error
            err    = raw - yi
            grad_w = err * xi; grad_b = err; loss_v = 0.5*err**2

        # Update
        delta_w = -eta * grad_w - eta * reg * w
        w      += delta_w
        b      += -eta * grad_b
        wn      = np.linalg.norm(w)
        bar     = "█" * int(min(np.linalg.norm(delta_w)*20, 30))
        print(f"  {step+1:<6}  {i:>8}  {loss_v:>10.4f}  {wn:>9.4f}  "
              f"{b:>9.4f}  {bar}")

    print(f"\n  (Simulated weights above are illustrative — model uses optimised internals)")
    print(f"  Final model ||w||: {np.linalg.norm(coefs):.6f}")

    # ── C) Loss landscape for each feature ────────────────────────────────────
    section("C) Loss Landscape — How Loss Changes With Each Weight")
    print("  Perturbing each weight ±10% and measuring loss change.\n")
    from sklearn.metrics import log_loss as _ll

    def _total_loss(w_vec, b_val):
        raw  = X_train @ w_vec + b_val
        if is_clf:
            probs  = _sigmoid(raw)
            probs  = np.clip(probs, 1e-9, 1-1e-9)
            return float(np.mean(-(y_train*np.log(probs)+(1-y_train)*np.log(1-probs))))
        return float(np.mean((y_train - raw)**2))

    base_loss = _total_loss(coefs, intercept)
    print(f"  Current loss: {base_loss:.6f}\n")
    print(f"  {'Feature':<25} {'w':>10}  {'-10%':>10}  {'+10%':>10}  {'Sensitivity':>12}")
    print("  " + "─" * 70)
    for name, c in zip(feature_names, coefs):
        w_lo = coefs.copy(); w_lo[list(feature_names).index(name)] *= 0.9
        w_hi = coefs.copy(); w_hi[list(feature_names).index(name)] *= 1.1
        l_lo = _total_loss(w_lo, intercept)
        l_hi = _total_loss(w_hi, intercept)
        sens = abs(l_hi - l_lo)
        bar  = "█" * int(min(sens * 100, 35))
        print(f"  {name:<25} {c:>10.4f}  {l_lo:>10.4f}  {l_hi:>10.4f}  "
              f"{sens:>12.6f}  {bar}")

    # ── D) Sample-by-sample decomposition ────────────────────────────────────
    section("D) Sample-by-Sample Score Decomposition")
    print("  Each feature's contribution to the final raw score.\n")
    for s_idx in range(min(4, len(X_test))):
        sample    = X_test[s_idx]
        raw_score = intercept
        print(f"  Sample #{s_idx+1}  (true={y_test[s_idx]})")
        print(f"    Intercept:              {intercept:+.6f}")
        for name, c, val in zip(feature_names, coefs, sample):
            contrib   = c * val
            raw_score += contrib
            bar  = "█" * int(min(abs(contrib)*5, 25))
            print(f"    {name:<22}: {val:>8.3f} x {c:>+.4f} = {contrib:>+.6f}  {bar}")
        if is_clf:
            prob = _sigmoid(raw_score)
            pred = 1 if prob >= 0.5 else 0
            print(f"    {'─'*50}")
            print(f"    Logit={raw_score:+.6f}  prob={prob:.4f}  pred={pred} "
                  f"({'✅' if pred==y_test[s_idx] else '❌'})\n")
        else:
            print(f"    {'─'*50}")
            print(f"    yhat={raw_score:.6f}  true={y_test[s_idx]:.4f}  "
                  f"error={abs(raw_score-y_test[s_idx]):.4f}\n")

    # ── E) Convergence check ──────────────────────────────────────────────────
    section("E) Convergence — Did SGD Actually Converge?")
    n_iter  = getattr(model, "n_iter_",  None)
    t_steps = getattr(model, "t_",       None)
    max_it  = model.max_iter
    tol     = model.tol

    info("Max iterations set", max_it)
    info("Actual epochs run",  n_iter if n_iter else "N/A")
    info("Total weight updates", t_steps if t_steps else "N/A")
    info("Tolerance (tol)",    tol)

    if n_iter and n_iter < max_it:
        print(f"\n  ✅ CONVERGED early at epoch {n_iter} (before max {max_it})")
        print(f"     Weight updates saved: ~{(max_it-n_iter)*X_train.shape[0]}")
    elif n_iter and n_iter >= max_it:
        print(f"\n  ⚠  DID NOT CONVERGE — hit max_iter={max_it}")
        print(f"     Try: increase max_iter, reduce learning_rate, or scale features")
    else:
        print(f"\n  ℹ  Convergence info not available (n_iter_ not set)")

    subsection("Weight magnitude distribution")
    print(f"  ||w|| (L2 norm):   {np.linalg.norm(coefs):.6f}")
    print(f"  ||w|| (L1 norm):   {np.sum(np.abs(coefs)):.6f}")
    print(f"  Max |weight|:      {np.max(np.abs(coefs)):.6f}")
    print(f"  Min |weight|:      {np.min(np.abs(coefs)):.6f}")
    pct_zero = np.mean(np.abs(coefs) < 1e-6) * 100
    if pct_zero > 0:
        print(f"  Effectively zero:  {pct_zero:.1f}%  (L1/ElasticNet may have sparsified)")


def learning_trace(model, X_train, y_train, X_test, y_test,
                   feature_names, task_type):
    """Routes to the right tracer based on model type."""
    banner("STEP 9 -- LEARNING TRACE  (How the Model Learned)")
    model_name = type(model).__name__

    section("What This Section Shows")
    print(textwrap.fill(
        "  Traces the actual learning process: which splits were made and why, "
        "how each feature pushed the prediction up or down, where the decision "
        "boundaries are in feature space, and how errors shrank round by round.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "
    ))

    if HAS_XGB and model_name in ("XGBClassifier", "XGBRegressor"):
        _xgb_learning_trace(model, X_train, y_train, X_test, y_test,
                             feature_names, task_type)
    elif model_name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        _gbm_learning_trace(model, X_train, y_train, X_test, y_test,
                             feature_names, task_type)
    elif model_name in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        _tree_learning_trace(model, X_train, y_train, X_test, y_test,
                              feature_names, task_type)
    elif model_name in ("RandomForestClassifier", "RandomForestRegressor"):
        _forest_learning_trace(model, X_train, y_train, X_test, y_test,
                                feature_names, task_type)
    elif model_name in ("LinearRegression", "Ridge", "Lasso", "ElasticNet",
                        "LogisticRegression"):
        _linear_learning_trace(model, X_train, y_train, X_test, y_test,
                                feature_names, task_type)
    elif model_name in ("SVC", "SVR"):
        _svm_learning_trace(model, X_train, y_train, X_test, y_test,
                             feature_names, task_type)
    elif model_name in ("KNeighborsClassifier", "KNeighborsRegressor"):
        _knn_learning_trace(model, X_train, y_train, X_test, y_test,
                             feature_names, task_type)
    elif HAS_LGB and model_name in ("LGBMClassifier", "LGBMRegressor"):
        _lgbm_learning_trace(model, X_train, y_train, X_test, y_test,
                              feature_names, task_type)
    elif model_name in ("SGDClassifier", "SGDRegressor"):
        _sgd_learning_trace(model, X_train, y_train, X_test, y_test,
                             feature_names, task_type)
    else:
        section("Generic Feature Influence")
        print(f"  No deep tracer for {model_name} yet -- showing permutation importance.")
        perm_importance(model, X_test, y_test, feature_names, task_type,
                        _print_banner=False)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(model, X, y, feature_names=None, task_type=None,
                 test_size=0.2, scale=True, cv=5, n_walkthrough=5):
    """
    Main entry point.

    Parameters
    ----------
    model        : Any sklearn-compatible model instance
    X            : np.ndarray or pd.DataFrame  (features)
    y            : np.ndarray or pd.Series     (target)
    feature_names: list of str (auto-detected from DataFrame)
    task_type    : "classification" or "regression" (auto-detected)
    test_size    : fraction for test split (default 0.2)
    scale        : apply StandardScaler (default True)
    cv           : number of cross-val folds (default 5)
    n_walkthrough: samples to walk through in detail (default 5)
    """

    # ── Coerce inputs ──────────────────────────────────────────────────────────
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X = X.values
    elif feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if isinstance(y, pd.Series):
        y = y.values

    if task_type is None:
        unique_y = np.unique(y)
        task_type = "classification" if len(unique_y) <= 20 and y.dtype in (int, np.int32, np.int64, object) else "regression"

    # ── Run all steps ──────────────────────────────────────────────────────────
    banner("ML TRANSPARENCY PIPELINE", "█")
    print(f"  Model   : {type(model).__name__}")
    print(f"  Task    : {task_type.upper()}")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  XGBoost : {'available' if HAS_XGB else 'not installed'}")
    print(f"  LightGBM: {'available' if HAS_LGB else 'not installed'}")
    print(f"  Optuna  : {'available' if HAS_OPTUNA else 'not installed  →  pip install optuna'}")

    # 1. Data analysis
    analyze_data(X, y, feature_names, task_type)

    # 2. Preprocess
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = preprocess(
        X, y, test_size=test_size, scale=scale
    )

    # 3. Train
    model, elapsed = train_model(model, X_train_s, y_train)

    # 4. Internals
    explain_model(model, feature_names, task_type)

    # 5. Walkthrough
    walkthrough_predictions(model, X_test_s, y_test, feature_names, task_type, n=n_walkthrough)

    # 6. Metrics
    evaluate_model(model, X_train_s, X_test_s, y_train, y_test, task_type)

    # 7. Smart validation
    validate_model(model, X, y, X_train_s, y_train, X_test_s, y_test,
                   task_type, cv=cv, random_state=42)

    # 8. Permutation importance
    perm_importance(model, X_test_s, y_test, feature_names, task_type)

    # 9. Learning trace
    learning_trace(model, X_train_s, y_train, X_test_s, y_test,
                   feature_names, task_type)

    # ── Summary ────────────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE", "█")
    print(f"  Model          : {type(model).__name__}")
    print(f"  Training time  : {elapsed:.4f}s")
    print(f"  Train samples  : {len(X_train)}")
    print(f"  Test  samples  : {len(X_test)}")
    print(f"  Scaler used    : {type(scaler).__name__ if scaler else 'None'}")
    print(f"\n  The model object is returned for further use.\n")

    return model, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO — Run if executed directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer

    print("\n" + "═" * WIDTH)
    print("  DEMO MODE — Running 3 pipeline examples + 1 Optuna tuning example")
    print("  1) Classification: Random Forest on Iris")
    print("  2) Regression    : Gradient Boosting on Diabetes")
    print("  3) Classification: Logistic Regression on Breast Cancer")
    print("  4) Optuna Tuning : Random Forest on Breast Cancer  (if optuna installed)")
    print("═" * WIDTH)

    # ── Demo 1: Classification ────────────────────────────────────────────────
    print("\n\n" + "█" * WIDTH)
    print("  DEMO 1 of 4 — Random Forest Classifier — IRIS DATASET")
    print("█" * WIDTH)

    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    feat_iris = iris.feature_names

    model_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    run_pipeline(model_rf, X_iris, y_iris, feature_names=feat_iris,
                 task_type="classification", scale=False, cv=5, n_walkthrough=3)

    # ── Demo 2: Regression ────────────────────────────────────────────────────
    print("\n\n" + "█" * WIDTH)
    print("  DEMO 2 of 4 — Gradient Boosting Regressor — DIABETES DATASET")
    print("█" * WIDTH)

    diabetes = load_diabetes()
    X_db = diabetes.data
    y_db = diabetes.target
    feat_db = diabetes.feature_names

    model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                          max_depth=3, random_state=42)
    run_pipeline(model_gbr, X_db, y_db, feature_names=feat_db,
                 task_type="regression", scale=True, cv=5, n_walkthrough=3)

    # ── Demo 3: Logistic Regression ───────────────────────────────────────────
    print("\n\n" + "█" * WIDTH)
    print("  DEMO 3 of 4 — Logistic Regression — BREAST CANCER DATASET")
    print("█" * WIDTH)

    cancer = load_breast_cancer()
    X_bc = cancer.data
    y_bc = cancer.target
    feat_bc = cancer.feature_names

    model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    run_pipeline(model_lr, X_bc, y_bc, feature_names=list(feat_bc),
                 task_type="classification", scale=True, cv=5, n_walkthrough=3)

    # ── Demo 4: Optuna Tuning ─────────────────────────────────────────────────
    print("\n\n" + "█" * WIDTH)
    print("  DEMO 4 of 4 — Optuna Tuning — Random Forest on Breast Cancer")
    print("█" * WIDTH)

    if HAS_OPTUNA:
        model_to_tune = RandomForestClassifier(random_state=42)
        best_model, best_scaler, study = tune_model(
            model_to_tune,
            X_bc, y_bc,
            task_type="classification",
            n_trials=30,       # increase for better results (50-200 recommended)
            cv_folds=3,        # keep low for speed; 5 for more reliable estimates
            scale=False,
        )
    else:
        print(f"\n  Optuna not installed — skipping demo 4.")
        print(f"  Install with:  pip install optuna")
        print(f"\n  Usage once installed:")
        print(f"    from ml_pipeline import tune_model")
        print(f"    from sklearn.ensemble import RandomForestClassifier")
        print(f"")
        print(f"    model = RandomForestClassifier()")
        print(f"    best_model, scaler, study = tune_model(")
        print(f"        model, X, y,")
        print(f"        task_type='classification',")
        print(f"        n_trials=50,")
        print(f"        cv_folds=3,")
        print(f"    )")