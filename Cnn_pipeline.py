"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CNN TRANSPARENCY PIPELINE  —  cnn_pipeline.py                 ║
║  Terminal-only visibility into every aspect of a CNN: architecture,         ║
║  training dynamics, filters, activations, gradients, saliency maps,         ║
║  overfitting diagnosis, and per-prediction explainability.                  ║
║                                                                              ║
║  Supports: PyTorch (primary)  |  Keras/TF (via adapter)                     ║
║  Usage:                                                                      ║
║    from cnn_pipeline import run_cnn_pipeline                                 ║
║    run_cnn_pipeline(model, train_loader, val_loader,                         ║
║                     class_names=['cat','dog'], task='classification')        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import time
import copy
import textwrap
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")

# ── Optional framework imports ────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

if not HAS_TORCH and not HAS_TF:
    print("WARNING: Neither PyTorch nor TensorFlow found. Install one before use.")

# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

WIDTH = 80

def banner(title, char="═"):
    pad = max(0, WIDTH - len(title) - 4)
    lp  = pad // 2
    rp  = pad - lp
    print()
    print(char * WIDTH)
    print(char * lp + "  " + title + "  " + char * rp)
    print(char * WIDTH)

def section(title):
    print()
    print("─" * WIDTH)
    print(f"  ▶  {title}")
    print("─" * WIDTH)

def subsection(title):
    print(f"\n  ····  {title}  ····")

def info(key, val):
    key_s = f"{key}:"
    print(f"    {key_s:<35} {val}")

def bar_chart(pairs, total=None, width=40):
    if not pairs:
        return
    max_val = max(abs(v) for _, v in pairs) if pairs else 1
    max_val = max_val or 1
    for name, val in pairs:
        filled = int(abs(val) / max_val * width)
        empty  = width - filled
        bar    = "█" * filled + "░" * empty
        pct    = f"{val/total*100:.1f}%" if total else f"{val:.4g}"
        print(f"  {str(name)[:20]:<20} │{bar}│ {pct:>8}")

def ascii_heatmap(matrix, rows=8, cols=16, label=""):
    """Render a 2D numpy array as ASCII art heatmap."""
    if matrix.ndim > 2:
        matrix = matrix.mean(axis=0) if matrix.shape[0] < matrix.shape[-1] \
                  else matrix.mean(axis=-1)
    # Downsample to display size
    from scipy.ndimage import zoom as _zoom
    try:
        zy = rows / matrix.shape[0]
        zx = cols / matrix.shape[1]
        m  = _zoom(matrix.astype(float), (zy, zx), order=1)
    except Exception:
        m = matrix[:rows, :cols]

    mn, mx = m.min(), m.max()
    rng    = mx - mn + 1e-9
    chars  = " ░▒▓█"

    if label:
        print(f"  {label}")
    print("  ┌" + "─" * (m.shape[1] * 2) + "┐")
    for row in m:
        line = "".join(chars[int((v - mn) / rng * (len(chars) - 1))] * 2 for v in row)
        print(f"  │{line}│")
    print("  └" + "─" * (m.shape[1] * 2) + "┘")
    print(f"  min={mn:.4f}  max={mx:.4f}  mean={m.mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  FRAMEWORK ADAPTERS  — uniform interface over PyTorch / Keras
# ══════════════════════════════════════════════════════════════════════════════

class _TorchAdapter:
    """Thin wrapper so the rest of the pipeline is framework-agnostic."""
    def __init__(self, model):
        self.model  = model
        self.device = next(model.parameters()).device

    def get_layers(self):
        """Return list of (name, module) for all named modules."""
        return list(self.model.named_modules())

    def get_conv_layers(self):
        return [(n, m) for n, m in self.model.named_modules()
                if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]

    def get_bn_layers(self):
        return [(n, m) for n, m in self.model.named_modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]

    def count_parameters(self):
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def predict_batch(self, x):
        self.model.eval()
        with torch.no_grad():
            out = self.model(x.to(self.device))
        return out.cpu()

    def predict_proba(self, x):
        logits = self.predict_batch(x)
        if logits.shape[-1] > 1:
            return torch.softmax(logits, dim=-1).numpy()
        return torch.sigmoid(logits).numpy()

    def framework(self):
        return "pytorch"


class _KerasAdapter:
    def __init__(self, model):
        self.model = model

    def get_layers(self):
        return [(l.name, l) for l in self.model.layers]

    def get_conv_layers(self):
        return [(l.name, l) for l in self.model.layers
                if "conv" in l.__class__.__name__.lower()]

    def get_bn_layers(self):
        return [(l.name, l) for l in self.model.layers
                if "batchnorm" in l.__class__.__name__.lower()]

    def count_parameters(self):
        total     = self.model.count_params()
        trainable = sum(tf.size(w).numpy() for w in self.model.trainable_weights)
        return total, trainable

    def predict_batch(self, x):
        if isinstance(x, np.ndarray):
            return self.model(x, training=False).numpy()
        return self.model(x.numpy(), training=False).numpy()

    def predict_proba(self, x):
        out = self.predict_batch(x)
        if out.shape[-1] > 1:
            exp = np.exp(out - out.max(axis=-1, keepdims=True))
            return exp / exp.sum(axis=-1, keepdims=True)
        return 1 / (1 + np.exp(-out))

    def framework(self):
        return "keras"


def _make_adapter(model):
    if HAS_TORCH and isinstance(model, nn.Module):
        return _TorchAdapter(model)
    elif HAS_TF:
        try:
            if isinstance(model, tf.keras.Model):
                return _KerasAdapter(model)
        except Exception:
            pass
    raise ValueError(f"Model type {type(model)} not supported. "
                     "Pass a PyTorch nn.Module or Keras Model.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — DATASET HEALTH REPORT
# ══════════════════════════════════════════════════════════════════════════════

def cnn_dataset_health(loader, class_names=None, task="classification",
                        n_inspect=256):
    """
    Inspect image batches from a DataLoader/dataset for:
    - image shape, channel stats, pixel value range
    - class balance, label distribution
    - pixel intensity distributions per channel
    - near-duplicate detection (via mean hash)
    - normalization check
    """
    banner("STEP 0 — DATASET HEALTH REPORT")
    issues = []

    # ── Collect sample batches ────────────────────────────────────────────────
    images_collected = []
    labels_collected = []
    n_collected = 0

    try:
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imgs, lbls = batch[0], batch[1]
            else:
                imgs, lbls = batch, None

            if HAS_TORCH and isinstance(imgs, torch.Tensor):
                imgs_np = imgs.cpu().numpy()
            else:
                imgs_np = np.array(imgs)

            images_collected.append(imgs_np)
            if lbls is not None:
                if HAS_TORCH and isinstance(lbls, torch.Tensor):
                    labels_collected.extend(lbls.cpu().numpy().tolist())
                else:
                    labels_collected.extend(np.array(lbls).tolist())

            n_collected += len(imgs_np)
            if n_collected >= n_inspect:
                break
    except Exception as e:
        print(f"  ⚠  Could not read loader: {e}")
        return issues

    if not images_collected:
        print("  ⚠  No images collected from loader")
        return issues

    imgs_all = np.concatenate(images_collected, axis=0)[:n_inspect]

    # ── Image shape analysis ──────────────────────────────────────────────────
    section("Image Shape & Format")
    shape = imgs_all.shape
    info("Batch shape (N,C,H,W or N,H,W,C)", str(shape))

    # Detect channel-first vs channel-last
    if len(shape) == 4:
        if shape[1] in (1, 3, 4) and shape[1] < shape[2]:
            fmt     = "channel-first (N,C,H,W)  [PyTorch style]"
            n, c, h, w = shape
        elif shape[3] in (1, 3, 4):
            fmt     = "channel-last  (N,H,W,C)  [TF/Keras style]"
            n, h, w, c = shape
            imgs_all = np.transpose(imgs_all, (0, 3, 1, 2))  # normalise to NCHW
        else:
            fmt = "ambiguous — assuming channel-first"
            n, c, h, w = shape
    elif len(shape) == 3:
        fmt = "grayscale (N,H,W)"
        n, h, w = shape
        c = 1
        imgs_all = imgs_all[:, np.newaxis, :, :]
    else:
        fmt = f"unexpected shape: {shape}"
        n, c, h, w = len(imgs_all), 1, shape[-2] if len(shape) > 2 else 1, shape[-1]

    info("Format detected", fmt)
    info("Num samples inspected", n)
    info("Channels", c)
    info("Height × Width", f"{h} × {w}")
    info("Pixels per image", f"{c * h * w:,}")
    info("Aspect ratio", f"{w/h:.2f}")

    if h != w:
        print(f"  🟡 Non-square images ({h}×{w}) — ensure your model handles this")
        issues.append(("WARNING", f"Non-square images {h}×{w}",
                        "Verify model input size matches"))
    if h < 32 or w < 32:
        print(f"  🟡 Very small images ({h}×{w}) — limited spatial information")
        issues.append(("WARNING", "Very small images", "May limit model capacity"))

    # ── Pixel value range & normalization check ────────────────────────────────
    section("Pixel Value Range & Normalization")
    flat = imgs_all.reshape(n, c, -1)
    p_min  = float(flat.min())
    p_max  = float(flat.max())
    p_mean = float(flat.mean())
    p_std  = float(flat.std())

    info("Global min", f"{p_min:.4f}")
    info("Global max", f"{p_max:.4f}")
    info("Global mean", f"{p_mean:.4f}")
    info("Global std", f"{p_std:.4f}")

    if p_min >= 0 and p_max <= 1.01:
        norm_status = "✅ [0, 1] range — looks normalized"
    elif p_min >= -2.5 and p_max <= 2.5 and abs(p_mean) < 0.5:
        norm_status = "✅ ~[-2.5, 2.5] range — likely standardized (mean≈0, std≈1)"
    elif p_min >= 0 and p_max <= 255:
        norm_status = "🟡 [0, 255] raw pixel range — normalize before training!"
        issues.append(("WARNING", "Pixels not normalized (0-255 range)",
                        "Divide by 255 or use transforms.Normalize()"))
    else:
        norm_status = f"🟡 Unusual range [{p_min:.2f}, {p_max:.2f}] — verify normalization"
        issues.append(("WARNING", f"Unusual pixel range [{p_min:.2f},{p_max:.2f}]",
                        "Check preprocessing pipeline"))
    print(f"  {norm_status}")

    subsection("Per-Channel Statistics")
    ch_names = ["R", "G", "B", "A"] if c <= 4 else [f"Ch{i}" for i in range(c)]
    print(f"  {'Channel':<10} {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  "
          f"{'Saturation%':>12}")
    print("  " + "─" * 60)
    for ci in range(min(c, 8)):
        ch_data  = flat[:, ci, :]
        saturated = float(np.mean((ch_data <= p_min + 1e-4) | (ch_data >= p_max - 1e-4)) * 100)
        print(f"  {ch_names[ci]:<10} {ch_data.mean():>9.4f}  {ch_data.std():>9.4f}  "
              f"{ch_data.min():>9.4f}  {ch_data.max():>9.4f}  {saturated:>11.1f}%")
        if saturated > 20:
            issues.append(("WARNING", f"Channel {ch_names[ci]} has {saturated:.0f}% saturated pixels",
                            "Check augmentation or clipping"))

    # ── Class distribution ────────────────────────────────────────────────────
    section("Class Distribution")
    if labels_collected:
        labels_arr = np.array(labels_collected)
        if task == "classification":
            counts = Counter(labels_arr.astype(int).tolist())
            total  = sum(counts.values())
            sorted_counts = sorted(counts.items())
            if class_names and len(class_names) >= len(counts):
                display = [(class_names[k], v) for k, v in sorted_counts]
            else:
                display = [(f"Class {k}", v) for k, v in sorted_counts]

            bar_chart(display, total=total)

            majority = max(counts.values())
            minority = min(counts.values())
            ratio    = majority / minority
            info("Imbalance ratio", f"{ratio:.2f}x")
            if ratio > 10:
                print(f"  🔴 SEVERE IMBALANCE — use class weights or oversampling")
                issues.append(("CRITICAL", f"Severe class imbalance ({ratio:.1f}x)",
                                "Use weighted loss: nn.CrossEntropyLoss(weight=class_weights)"))
            elif ratio > 3:
                print(f"  🟡 MODERATE IMBALANCE — accuracy metric will be misleading")
                issues.append(("WARNING", f"Class imbalance ({ratio:.1f}x)",
                                "Monitor per-class F1, not just accuracy"))
            else:
                print(f"  ✅ Reasonably balanced")
        else:
            info("Target mean", f"{labels_arr.mean():.4f}")
            info("Target std",  f"{labels_arr.std():.4f}")
    else:
        print("  (No labels found in loader)")

    # ── Pixel intensity histogram (ASCII) ─────────────────────────────────────
    section("Pixel Intensity Distribution (grayscale histogram)")
    gray = imgs_all.mean(axis=1).flatten()
    hist, edges = np.histogram(gray, bins=20, range=(gray.min(), gray.max()))
    max_h = hist.max() + 1
    print(f"  Range: [{gray.min():.3f}, {gray.max():.3f}]\n")
    for i in range(len(hist)):
        bar  = "█" * int(hist[i] / max_h * 35)
        print(f"  {edges[i]:>7.3f} │{bar:<35}│ {hist[i]:>6}")

    if hist[0] > hist.sum() * 0.4 or hist[-1] > hist.sum() * 0.4:
        print(f"\n  🟡 Distribution heavily skewed toward one extreme — check normalization")
        issues.append(("WARNING", "Skewed pixel intensity distribution",
                        "Verify normalization is applied correctly"))
    elif hist[9] + hist[10] < hist.sum() * 0.05:
        print(f"\n  🟡 Bimodal distribution — images may have very different exposures")
    else:
        print(f"\n  ✅ Pixel distribution looks reasonable")

    # ── Dataset size ──────────────────────────────────────────────────────────
    section("Dataset Size Recommendations")
    try:
        total_samples = len(loader.dataset)
    except Exception:
        total_samples = n_collected
    info("Total samples in dataset", total_samples)

    if task == "classification" and labels_collected:
        n_classes  = len(set(labels_collected))
        per_class  = total_samples / n_classes
        info("Estimated samples per class", f"{per_class:.0f}")
        if per_class < 100:
            print(f"  🔴 Very few samples per class ({per_class:.0f})")
            print(f"     → Use heavy augmentation, transfer learning, or few-shot methods")
            issues.append(("CRITICAL", f"Only {per_class:.0f} samples/class",
                            "Use transfer learning + strong augmentation"))
        elif per_class < 500:
            print(f"  🟡 Small per-class count ({per_class:.0f}) — transfer learning recommended")
            issues.append(("WARNING", f"Small dataset ({per_class:.0f} samples/class)",
                            "Consider transfer learning from pretrained model"))
        else:
            print(f"  ✅ Dataset size looks adequate ({per_class:.0f} samples/class)")

    # ── Near-duplicate detection ──────────────────────────────────────────────
    section("Near-Duplicate Detection (mean-hash)")
    print("  Computing 8×8 mean hash for sampled images...")
    def _mean_hash(img):
        """8x8 mean hash — fast approximate duplicate detector."""
        try:
            from scipy.ndimage import zoom as _z
            small = _z(img.mean(axis=0), (8/img.shape[-2], 8/img.shape[-1]), order=1)
        except Exception:
            small = img.mean(axis=0)[:8, :8]
        return (small > small.mean()).flatten()

    hashes   = [_mean_hash(imgs_all[i]) for i in range(min(n, 200))]
    hash_strs = ["".join(h.astype(int).astype(str)) for h in hashes]
    dup_count = len(hash_strs) - len(set(hash_strs))

    if dup_count > 0:
        pct = dup_count / len(hash_strs) * 100
        print(f"  🟡 ~{dup_count} near-duplicate image(s) found in sample ({pct:.1f}%)")
        issues.append(("WARNING", f"~{dup_count} near-duplicate images in sample",
                        "Deduplicate dataset to avoid inflated val scores"))
    else:
        print(f"  ✅ No near-duplicates found in {min(n, 200)}-sample check")

    # ── Health scorecard ──────────────────────────────────────────────────────
    section("Health Scorecard")
    critical = [i for i in issues if i[0] == "CRITICAL"]
    warnings_ = [i for i in issues if i[0] == "WARNING"]
    if not issues:
        print("  ✅ HEALTHY — Dataset looks ready for CNN training")
    else:
        if critical:
            print(f"  🔴 {len(critical)} CRITICAL issue(s):\n")
            for _, msg, fix in critical:
                print(f"     • {msg}")
                print(f"       Fix: {fix}")
        if warnings_:
            print(f"\n  🟡 {len(warnings_)} WARNING(s):\n")
            for _, msg, fix in warnings_:
                print(f"     • {msg}")
                print(f"       Fix: {fix}")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — ARCHITECTURE VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

def architecture_visualizer(adapter, input_shape=None):
    """
    Full architecture breakdown:
    - Every layer with type, output shape, params, role
    - Parameter budget waterfall
    - Receptive field growth per conv layer
    - Bottleneck / capacity warnings
    """
    banner("STEP 1 — ARCHITECTURE VISUALIZER")
    model_name = type(adapter.model).__name__
    total_p, trainable_p = adapter.count_parameters()

    section("Model Overview")
    info("Model class", model_name)
    info("Framework", adapter.framework())
    info("Total parameters", f"{total_p:,}")
    info("Trainable parameters", f"{trainable_p:,}")
    info("Non-trainable parameters", f"{total_p - trainable_p:,}")
    info("Memory (float32)", f"~{total_p * 4 / 1024**2:.2f} MB")

    # ── Layer-by-layer breakdown ───────────────────────────────────────────────
    section("Layer-by-Layer Breakdown")

    if adapter.framework() == "pytorch":
        _arch_pytorch(adapter, input_shape, total_p)
    else:
        _arch_keras(adapter, total_p)


def _arch_pytorch(adapter, input_shape, total_p):
    print(f"  {'Layer':<35} {'Type':<22} {'Params':>10}  {'% Budget':>9}  Role")
    print("  " + "─" * 85)

    layer_info  = []
    cum_params  = 0
    conv_idx    = 0
    rf          = 1    # receptive field
    rf_stride   = 1

    for name, module in adapter.model.named_modules():
        if name == "":
            continue
        # Count params for THIS module only (not children)
        own_params = sum(p.numel() for p in module.parameters(recurse=False))
        if own_params == 0 and len(list(module.children())) > 0:
            continue  # skip container modules

        mtype   = type(module).__name__
        cum_params += own_params
        pct     = own_params / (total_p + 1e-9) * 100

        # Determine role
        role = _layer_role(module)

        # Receptive field tracking
        rf_info = ""
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) \
                else module.kernel_size
            s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            d = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
            rf         = rf + (k - 1) * d * rf_stride
            rf_stride *= s
            rf_info    = f"  [RF={rf}]"
            conv_idx  += 1

        bar = "█" * int(pct * 0.5) if pct >= 0.1 else "·"
        print(f"  {name[:34]:<35} {mtype[:21]:<22} {own_params:>10,}  "
              f"{pct:>8.2f}%  {role}{rf_info}")

        layer_info.append((name, mtype, own_params, pct, role))

    # ── Parameter budget waterfall ────────────────────────────────────────────
    subsection("Parameter Budget (top layers by count)")
    top = sorted(layer_info, key=lambda x: x[2], reverse=True)[:10]
    for name, mtype, params, pct, role in top:
        bar = "█" * int(pct * 0.8)
        print(f"  {name[:28]:<28} {params:>10,}  {pct:>6.1f}%  {bar}")

    # ── Architecture warnings ─────────────────────────────────────────────────
    subsection("Architecture Diagnostics")
    n_conv   = sum(1 for _, t, _, _, _ in layer_info if "Conv" in t)
    n_bn     = sum(1 for _, t, _, _, _ in layer_info if "BatchNorm" in t)
    n_drop   = sum(1 for _, t, _, _, _ in layer_info if "Dropout" in t)
    n_linear = sum(1 for _, t, _, _, _ in layer_info if "Linear" in t)

    info("Conv layers", n_conv)
    info("BatchNorm layers", n_bn)
    info("Dropout layers", n_drop)
    info("Linear (FC) layers", n_linear)
    info("Final receptive field", rf)

    print()
    if n_conv > 0 and n_bn == 0:
        print(f"  🟡 No BatchNorm layers found — training may be unstable")
        print(f"     Add nn.BatchNorm2d after conv layers")
    else:
        print(f"  ✅ BatchNorm present ({n_bn} layers)")

    if n_drop == 0 and total_p > 100_000:
        print(f"  🟡 No Dropout layers — model with {total_p:,} params may overfit")
        print(f"     Consider adding Dropout(0.3-0.5) before FC layers")
    elif n_drop > 0:
        print(f"  ✅ Dropout present ({n_drop} layers)")

    if total_p > 10_000_000:
        print(f"  🟡 Large model ({total_p/1e6:.1f}M params) — needs substantial data")
    elif total_p < 1_000:
        print(f"  🟡 Very small model ({total_p} params) — may underfit complex images")
    else:
        print(f"  ✅ Model size: {total_p:,} parameters")


def _arch_keras(adapter, total_p):
    print(f"  {'Layer':<35} {'Type':<22} {'Params':>10}  {'% Budget':>9}  Output Shape")
    print("  " + "─" * 85)
    for layer in adapter.model.layers:
        mtype      = type(layer).__name__
        own_params = layer.count_params()
        pct        = own_params / (total_p + 1e-9) * 100
        try:
            out_shape = str(layer.output_shape)
        except Exception:
            out_shape = "?"
        print(f"  {layer.name[:34]:<35} {mtype[:21]:<22} {own_params:>10,}  "
              f"{pct:>8.2f}%  {out_shape}")


def _layer_role(module):
    name = type(module).__name__
    roles = {
        "Conv2d":          "Feature extraction",
        "Conv1d":          "Sequence feature extraction",
        "ConvTranspose2d": "Upsampling / decoder",
        "BatchNorm2d":     "Stabilize activations",
        "BatchNorm1d":     "Stabilize activations",
        "ReLU":            "Non-linearity",
        "LeakyReLU":       "Non-linearity (leak)",
        "GELU":            "Non-linearity (smooth)",
        "MaxPool2d":       "Spatial downsampling",
        "AvgPool2d":       "Spatial downsampling",
        "AdaptiveAvgPool2d": "Global spatial pooling",
        "Dropout":         "Regularization",
        "Dropout2d":       "Channel regularization",
        "Linear":          "Classification / regression head",
        "Embedding":       "Learned input encoding",
        "MultiheadAttention": "Self-attention",
        "LayerNorm":       "Normalize across features",
        "Sigmoid":         "Binary output",
        "Softmax":         "Multiclass output",
        "Flatten":         "Reshape → FC",
    }
    return roles.get(name, name)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TRAINING SETUP EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════

def training_setup_explainer(optimizer=None, scheduler=None, criterion=None,
                               train_loader=None, augmentation_info=None):
    """
    Explains the full training setup:
    - Optimizer type, learning rate, momentum, weight decay
    - LR scheduler type and effect
    - Loss function and its implications
    - Augmentation pipeline
    - Batch size and its effect on training dynamics
    """
    banner("STEP 2 — TRAINING SETUP EXPLAINER")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    section("Optimizer")
    if optimizer is None:
        print("  (No optimizer provided — pass optimizer= to see details)")
    elif HAS_TORCH and isinstance(optimizer, torch.optim.Optimizer):
        opt_name = type(optimizer).__name__
        info("Type", opt_name)
        for pg_idx, pg in enumerate(optimizer.param_groups):
            print(f"\n  Param group {pg_idx}:")
            for k, v in pg.items():
                if k != "params":
                    info(f"  {k}", v)

        # Explain optimizer choice
        print()
        explanations = {
            "SGD":      "Stochastic Gradient Descent. Needs tuned lr+momentum. "
                        "Often best final accuracy with careful scheduling.",
            "Adam":     "Adaptive lr per parameter. Fast convergence. "
                        "May generalize slightly worse than SGD on some tasks.",
            "AdamW":    "Adam + decoupled weight decay. Preferred over Adam for CNNs.",
            "RMSprop":  "Adaptive lr, good for non-stationary problems.",
            "Adagrad":  "Adapts lr per parameter. Lr decays over time — can stop early.",
        }
        expl = explanations.get(opt_name, f"{opt_name} — check documentation")
        print(textwrap.fill(f"  {expl}", width=WIDTH,
                            initial_indent="  ", subsequent_indent="  "))

        # Weight decay check
        wd = optimizer.param_groups[0].get("weight_decay", 0)
        if wd == 0:
            print(f"\n  🟡 weight_decay=0 — no L2 regularization applied")
            print(f"     Consider weight_decay=1e-4 to reduce overfitting")
        else:
            print(f"\n  ✅ weight_decay={wd}")

    # ── Learning rate scheduler ────────────────────────────────────────────────
    section("Learning Rate Scheduler")
    if scheduler is None:
        print("  (No scheduler provided — constant lr throughout training)")
        print("  🟡 Consider CosineAnnealingLR or OneCycleLR for better convergence")
    elif HAS_TORCH:
        sched_name = type(scheduler).__name__
        info("Type", sched_name)
        sched_explanations = {
            "StepLR":             f"Reduces lr by gamma every step_size epochs. "
                                   f"Simple but discontinuous.",
            "MultiStepLR":        "Reduces lr at specific milestones. Manual but effective.",
            "CosineAnnealingLR":  "Smooth cosine decay. Often best final accuracy.",
            "OneCycleLR":         "Warm up then cool down. Excellent for fast training.",
            "ReduceLROnPlateau":  "Reduces lr when metric stops improving. Adaptive.",
            "ExponentialLR":      "Exponential decay every epoch. Can decay too fast.",
            "CyclicLR":           "Cycles lr between bounds. Can escape local minima.",
            "WarmupScheduler":    "Linear warmup then decay. Prevents early instability.",
        }
        expl = sched_explanations.get(sched_name, f"{sched_name} scheduler")
        print(textwrap.fill(f"  {expl}", width=WIDTH,
                            initial_indent="  ", subsequent_indent="  "))

        # Show lr trajectory
        subsection("Projected LR Trajectory (first 20 epochs)")
        if hasattr(scheduler, "get_last_lr") or hasattr(scheduler, "base_lrs"):
            try:
                import copy as _copy
                sched_copy = _copy.deepcopy(scheduler)
                opt_copy   = _copy.deepcopy(optimizer) if optimizer else None
                lrs = []
                for ep in range(20):
                    lr = sched_copy.get_last_lr()[0] if hasattr(sched_copy, "get_last_lr") \
                         else sched_copy.base_lrs[0]
                    lrs.append(lr)
                    sched_copy.step()
                max_lr = max(lrs) + 1e-10
                print(f"  {'Epoch':>7}  {'LR':>12}  LR bar")
                print("  " + "─" * 45)
                for ep, lr in enumerate(lrs):
                    bar = "█" * int(lr / max_lr * 30)
                    print(f"  {ep+1:>7}  {lr:>12.2e}  {bar}")
            except Exception:
                print("  (could not simulate lr trajectory)")

    # ── Loss function ─────────────────────────────────────────────────────────
    section("Loss Function")
    if criterion is None:
        print("  (No criterion provided — pass criterion= to see details)")
    elif HAS_TORCH:
        loss_name = type(criterion).__name__
        info("Type", loss_name)
        loss_explanations = {
            "CrossEntropyLoss":  "Standard for multiclass classification. "
                                  "Applies softmax internally — don't apply softmax in model.",
            "BCELoss":           "Binary cross-entropy. Expects probabilities [0,1]. "
                                  "Apply sigmoid before loss.",
            "BCEWithLogitsLoss": "BCE + sigmoid in one step. More numerically stable "
                                  "than BCELoss. Preferred for binary tasks.",
            "MSELoss":           "Mean squared error. Standard for regression. "
                                  "Sensitive to outliers.",
            "L1Loss":            "Mean absolute error. Robust to outliers. "
                                  "Less smooth gradient near 0.",
            "NLLLoss":           "Negative log-likelihood. Use with log_softmax output.",
            "HuberLoss":         "Combination of MSE and L1. Robust regression loss.",
            "FocalLoss":         "Downweights easy examples. Good for imbalanced datasets.",
        }
        expl = loss_explanations.get(loss_name, f"{loss_name} — check documentation")
        print(textwrap.fill(f"  {expl}", width=WIDTH,
                            initial_indent="  ", subsequent_indent="  "))

        # Weight check
        if hasattr(criterion, "weight") and criterion.weight is not None:
            print(f"\n  ✅ Class weights set: {criterion.weight.tolist()}")
        elif loss_name in ("CrossEntropyLoss", "BCEWithLogitsLoss"):
            print(f"\n  🟡 No class weights — if imbalanced, set weight=class_weights")

    # ── Batch size ─────────────────────────────────────────────────────────────
    section("Batch Size & Training Dynamics")
    if train_loader is not None:
        try:
            bs = train_loader.batch_size
        except Exception:
            bs = None
        if bs:
            info("Batch size", bs)
            if bs < 16:
                print(f"  🟡 Small batch ({bs}) — noisy gradients, slow training")
                print(f"     Increase if GPU memory allows")
            elif bs > 512:
                print(f"  🟡 Large batch ({bs}) — may converge to sharp minima")
                print(f"     May need linear LR scaling: lr = base_lr × batch_size/256")
            else:
                print(f"  ✅ Batch size looks reasonable")

            try:
                n_samples = len(train_loader.dataset)
                n_batches = len(train_loader)
                info("Training samples", n_samples)
                info("Batches per epoch", n_batches)
                info("Steps for 100 epochs", f"{n_batches * 100:,}")
            except Exception:
                pass

    # ── Augmentation ──────────────────────────────────────────────────────────
    section("Data Augmentation")
    if augmentation_info:
        print(f"  {augmentation_info}")
    elif train_loader is not None:
        try:
            transforms = train_loader.dataset.transform
            if transforms is not None:
                print(f"  Detected transform pipeline:\n")
                _print_transforms(transforms)
            else:
                print(f"  🟡 No transforms detected — raw images fed to model")
                print(f"     For CNNs, standard augmentations significantly improve generalization:")
                print(f"     RandomHorizontalFlip, RandomCrop, ColorJitter, Normalize")
        except Exception:
            print("  (Could not inspect transforms from loader.dataset.transform)")


def _print_transforms(t):
    """Recursively print transform pipeline."""
    t_name = type(t).__name__
    if hasattr(t, "transforms"):
        print(f"  Compose(")
        for sub in t.transforms:
            _print_transforms(sub)
        print(f"  )")
    else:
        params = {k: v for k, v in vars(t).items()
                  if not k.startswith("_") and not callable(v)}
        param_str = "  ".join(f"{k}={v}" for k, v in list(params.items())[:4])
        print(f"    {t_name}({param_str})")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — TRAINING LOOP TRACE  (called during / after training)
# ══════════════════════════════════════════════════════════════════════════════

class TrainingTracer:
    """
    Attach to your training loop. Records per-epoch:
    - train/val loss and accuracy
    - gradient norms
    - learning rate
    - dead neuron ratio
    - batch norm stats

    Usage:
        tracer = TrainingTracer()
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(...)
            val_loss, val_acc     = validate(...)
            tracer.record(epoch, train_loss, train_acc, val_loss, val_acc,
                          model=model, optimizer=optimizer)
        tracer.report()
    """

    def __init__(self):
        self.history = []

    def record(self, epoch, train_loss, train_acc=None,
               val_loss=None, val_acc=None,
               model=None, optimizer=None):
        rec = {
            "epoch":      epoch + 1,
            "train_loss": float(train_loss),
            "train_acc":  float(train_acc) if train_acc is not None else None,
            "val_loss":   float(val_loss)  if val_loss  is not None else None,
            "val_acc":    float(val_acc)   if val_acc   is not None else None,
            "lr":         None,
            "grad_norm":  None,
            "dead_ratio": None,
        }

        if optimizer is not None and HAS_TORCH:
            rec["lr"] = optimizer.param_groups[0]["lr"]

        if model is not None and HAS_TORCH:
            # Gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            rec["grad_norm"] = total_norm ** 0.5

            # Dead ReLU ratio
            dead = 0; total_act = 0
            for m in model.modules():
                if isinstance(m, nn.ReLU):
                    if m.inplace:
                        pass  # can't hook inplace
                    # Approximate: check weight norms of following conv
            # We'll compute this properly in the activation step
            rec["dead_ratio"] = None

        self.history.append(rec)

    def report(self):
        banner("STEP 3 — TRAINING LOOP TRACE")

        if not self.history:
            print("  No training history recorded. Call tracer.record() each epoch.")
            return

        epochs = [r["epoch"] for r in self.history]
        n_ep   = len(epochs)

        # ── Loss curves ───────────────────────────────────────────────────────
        section("Loss Curves — Train vs Validation")
        train_losses = [r["train_loss"] for r in self.history]
        val_losses   = [r["val_loss"]   for r in self.history if r["val_loss"] is not None]
        has_val      = len(val_losses) == n_ep

        max_loss = max(max(train_losses), max(val_losses) if has_val else 0) + 1e-9
        print(f"  {'Epoch':>7}  {'Train Loss':>12}  {'Val Loss':>12}  {'Gap':>8}  "
              f"Train bar  Val bar")
        print("  " + "─" * 75)
        step = max(1, n_ep // 20)
        for r in self.history[::step]:
            ep  = r["epoch"]
            tl  = r["train_loss"]
            vl  = r["val_loss"]
            gap = (tl - vl) if vl is not None else 0.0
            t_bar = "█" * int((1 - tl/max_loss) * 15)
            v_bar = "░" * int((1 - vl/max_loss) * 15) if vl is not None else ""
            vl_s  = f"{vl:>12.4f}" if vl is not None else f"{'N/A':>12}"
            gap_s = f"{gap:>+8.4f}" if vl is not None else f"{'N/A':>8}"
            print(f"  {ep:>7}  {tl:>12.4f}  {vl_s}  {gap_s}  {t_bar:<15}  {v_bar}")

        # ── Accuracy curves ────────────────────────────────────────────────────
        train_accs = [r["train_acc"] for r in self.history if r["train_acc"] is not None]
        val_accs   = [r["val_acc"]   for r in self.history if r["val_acc"]   is not None]
        if train_accs:
            section("Accuracy Curves")
            print(f"  {'Epoch':>7}  {'Train Acc':>10}  {'Val Acc':>10}  Progress")
            print("  " + "─" * 55)
            for r in self.history[::step]:
                ta = r["train_acc"]
                va = r["val_acc"]
                if ta is None:
                    continue
                bar_t = "█" * int(ta * 25) if ta else ""
                bar_v = "░" * int(va * 25) if va else ""
                va_s  = f"{va:>10.4f}" if va is not None else f"{'N/A':>10}"
                print(f"  {r['epoch']:>7}  {ta:>10.4f}  {va_s}  {bar_t}{bar_v}")

        # ── Learning rate trace ────────────────────────────────────────────────
        lrs = [r["lr"] for r in self.history if r["lr"] is not None]
        if lrs:
            section("Learning Rate Schedule (actual)")
            max_lr = max(lrs) + 1e-10
            print(f"  {'Epoch':>7}  {'LR':>14}  LR bar")
            print("  " + "─" * 45)
            for r in self.history[::step]:
                if r["lr"] is None:
                    continue
                bar = "█" * int(r["lr"] / max_lr * 30)
                print(f"  {r['epoch']:>7}  {r['lr']:>14.2e}  {bar}")

        # ── Gradient norms ─────────────────────────────────────────────────────
        gnorms = [r["grad_norm"] for r in self.history if r["grad_norm"] is not None]
        if gnorms:
            section("Gradient Norms — Vanishing / Exploding Detection")
            max_gn = max(gnorms) + 1e-10
            print(f"  {'Epoch':>7}  {'Grad Norm':>12}  Status    Bar")
            print("  " + "─" * 55)
            for r in self.history[::step]:
                gn = r["grad_norm"]
                if gn is None:
                    continue
                if gn < 1e-4:
                    status = "🔴 VANISHING"
                elif gn > 100:
                    status = "🔴 EXPLODING"
                elif gn < 0.01:
                    status = "🟡 very small"
                else:
                    status = "✅ healthy"
                bar = "█" * int(min(gn / max_gn, 1.0) * 25)
                print(f"  {r['epoch']:>7}  {gn:>12.4f}  {status:<12}  {bar}")

        # ── Training event detection ───────────────────────────────────────────
        section("Training Events — Spikes, Plateaus, Divergence")
        events = []
        for i in range(1, len(self.history)):
            prev = self.history[i-1]
            curr = self.history[i]
            ep   = curr["epoch"]

            # Loss spike
            if curr["train_loss"] > prev["train_loss"] * 1.5:
                events.append(f"  🔴 Epoch {ep:>4}: LOSS SPIKE  "
                               f"{prev['train_loss']:.4f} → {curr['train_loss']:.4f}")
            # Plateau detection
            if i >= 5:
                recent = [r["train_loss"] for r in self.history[i-5:i+1]]
                if max(recent) - min(recent) < 0.001:
                    events.append(f"  🟡 Epoch {ep:>4}: PLATEAU detected "
                                   f"(loss flat for 5 epochs: ~{np.mean(recent):.4f})")
            # Val loss divergence
            if (curr["val_loss"] is not None and prev["val_loss"] is not None
                    and curr["val_loss"] > prev["val_loss"] * 1.3
                    and curr["train_loss"] < prev["train_loss"]):
                events.append(f"  🔴 Epoch {ep:>4}: OVERFIT SIGNAL — "
                               f"val_loss rising while train_loss falling")

        if not events:
            print("  ✅ No anomalous training events detected")
        else:
            # Deduplicate plateau messages
            seen = set()
            for ev in events:
                key = ev[:40]
                if key not in seen:
                    print(ev)
                    seen.add(key)

        # ── Best epoch summary ─────────────────────────────────────────────────
        section("Best Epoch Summary")
        best_val_loss = min((r for r in self.history if r["val_loss"] is not None),
                            key=lambda r: r["val_loss"], default=None)
        best_val_acc  = max((r for r in self.history if r["val_acc"]  is not None),
                            key=lambda r: r["val_acc"],  default=None)

        if best_val_loss:
            info("Best val loss epoch", f"{best_val_loss['epoch']}  "
                 f"(loss={best_val_loss['val_loss']:.4f})")
        if best_val_acc:
            info("Best val acc epoch",  f"{best_val_acc['epoch']}  "
                 f"(acc={best_val_acc['val_acc']:.4f})")
        if train_accs and val_accs:
            final_gap = train_accs[-1] - val_accs[-1]
            info("Final train/val acc gap", f"{final_gap:+.4f}")
            if final_gap > 0.15:
                print(f"  🔴 Large gap — model is overfitting")
            elif final_gap > 0.05:
                print(f"  🟡 Moderate gap — some overfitting")
            else:
                print(f"  ✅ Gap is healthy")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — LAYER-BY-LAYER ACTIVATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def activation_analysis(adapter, sample_batch, class_names=None):
    """
    Hook into every layer and capture activation statistics:
    - mean, std, min, max per layer
    - sparsity (% zeros / near-zero)
    - dead ReLU detection
    - activation collapse detection
    - per-layer ASCII distribution
    """
    banner("STEP 4 — LAYER-BY-LAYER ACTIVATION ANALYSIS")

    if not HAS_TORCH:
        print("  PyTorch required for activation analysis")
        return {}

    model   = adapter.model
    device  = adapter.device

    if isinstance(sample_batch, (list, tuple)):
        x = sample_batch[0]
    else:
        x = sample_batch

    if isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        x = torch.tensor(np.array(x), dtype=torch.float32).to(device)

    if x.dim() == 3:
        x = x.unsqueeze(0)

    # ── Register forward hooks ─────────────────────────────────────────────────
    activation_store = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, output):
            if isinstance(output, torch.Tensor):
                activation_store[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(x[:min(4, len(x))])

    for h in hooks:
        h.remove()

    # ── Report per layer ───────────────────────────────────────────────────────
    section("Activation Statistics — Every Layer")
    print(f"  {'Layer':<35} {'Type':<16} {'Mean':>8}  {'Std':>8}  "
          f"{'Min':>8}  {'Max':>8}  {'Sparsity':>9}  Status")
    print("  " + "─" * 100)

    issues     = []
    layer_stats = {}

    for name, module in model.named_modules():
        if name not in activation_store:
            continue
        act   = activation_store[name].float()
        mtype = type(module).__name__

        mn   = float(act.mean())
        std  = float(act.std())
        mn_v = float(act.min())
        mx_v = float(act.max())
        sparsity = float((act.abs() < 1e-5).float().mean() * 100)

        status = "✅"
        notes  = ""
        if sparsity > 90:
            status = "🔴 DEAD"
            notes  = f"  → {sparsity:.0f}% zero activations — dead neurons"
            issues.append(("CRITICAL", f"Layer '{name}' has {sparsity:.0f}% dead activations",
                           "Use LeakyReLU, reduce lr, check weight init"))
        elif sparsity > 60:
            status = "🟡 sparse"
            issues.append(("WARNING", f"Layer '{name}' is {sparsity:.0f}% sparse",
                           "High sparsity may limit expressiveness"))
        elif std < 1e-4:
            status = "🔴 COLLAPSED"
            notes  = "  → near-zero std — activations have collapsed"
            issues.append(("CRITICAL", f"Layer '{name}' activations collapsed (std={std:.2e})",
                           "Check BatchNorm, weight init, learning rate"))
        elif mx_v > 1e4:
            status = "🟡 large"
            notes  = f"  → max={mx_v:.1e} — activations exploding"
            issues.append(("WARNING", f"Layer '{name}' has large activations",
                           "Add gradient clipping or reduce learning rate"))

        print(f"  {name[:34]:<35} {mtype[:15]:<16} {mn:>8.4f}  {std:>8.4f}  "
              f"{mn_v:>8.4f}  {mx_v:>8.4f}  {sparsity:>8.1f}%  {status}")
        if notes:
            print(f"  {' '*90}{notes}")

        layer_stats[name] = {"mean": mn, "std": std, "min": mn_v,
                              "max": mx_v, "sparsity": sparsity}

    # ── Dead ReLU summary ──────────────────────────────────────────────────────
    section("Dead ReLU Summary")
    relu_layers = [(n, layer_stats[n]) for n, m in model.named_modules()
                   if isinstance(m, nn.ReLU) and n in layer_stats]
    if relu_layers:
        print(f"  {'Layer':<35} {'Sparsity':>10}  Dead bar")
        print("  " + "─" * 60)
        for n, s in relu_layers:
            sp  = s["sparsity"]
            bar = "█" * int(sp * 0.3)
            flag = "🔴" if sp > 90 else "🟡" if sp > 60 else "✅"
            print(f"  {n:<35} {sp:>9.1f}%  {flag} {bar}")
    else:
        print("  No ReLU layers tracked (may use other activations)")

    # ── Activation flow: std across depth ─────────────────────────────────────
    section("Activation Flow — Signal Propagation Through Depth")
    print("  Std of activations layer by layer (healthy = roughly stable):\n")
    conv_bn_layers = [(n, s) for n, m in model.named_modules()
                      if n in layer_stats
                      and isinstance(m, (nn.Conv2d, nn.BatchNorm2d,
                                         nn.Linear, nn.ReLU))]
    max_std = max((s["std"] for _, s in conv_bn_layers), default=1.0) + 1e-9
    for n, s in conv_bn_layers[:20]:
        std = s["std"]
        bar = "█" * int(std / max_std * 35)
        flag = "🔴" if std < 1e-4 or std > 100 else "✅"
        print(f"  {flag} {n[:32]:<32}  std={std:.4f}  {bar}")

    # ── Issues summary ─────────────────────────────────────────────────────────
    if issues:
        section("Activation Issues Found")
        for sev, msg, fix in issues:
            icon = "🔴" if sev == "CRITICAL" else "🟡"
            print(f"  {icon} {msg}")
            print(f"     Fix: {fix}")
    else:
        section("Activation Health")
        print("  ✅ All activation layers look healthy")

    return layer_stats


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — FILTER / WEIGHT VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

def filter_visualizer(adapter):
    """
    Inspect what each convolutional filter has learned:
    - Weight statistics per filter
    - Dead filter detection (near-zero weights)
    - Filter diversity (are filters learning different things?)
    - ASCII visualization of first-layer filters
    - Weight distribution histograms per layer
    """
    banner("STEP 5 — FILTER & WEIGHT VISUALIZER")

    conv_layers = adapter.get_conv_layers()
    if not conv_layers:
        print("  No convolutional layers found")
        return

    section("Convolutional Filter Statistics")
    print(f"  {'Layer':<35} {'Filters':>8}  {'Mean W':>9}  {'Std W':>9}  "
          f"{'Dead%':>7}  {'Diversity':>10}  Status")
    print("  " + "─" * 90)

    filter_summaries = []

    for name, module in conv_layers:
        if not HAS_TORCH:
            break
        w       = module.weight.data.cpu().numpy()  # (out_ch, in_ch, kH, kW)
        n_filt  = w.shape[0]
        w_flat  = w.reshape(n_filt, -1)

        mean_w  = float(w.mean())
        std_w   = float(w.std())
        dead_pct = float(np.mean(np.abs(w_flat).max(axis=1) < 1e-5) * 100)

        # Filter diversity: mean pairwise cosine distance (sample 50 pairs)
        np.random.seed(42)
        n_sample = min(n_filt, 50)
        idx  = np.random.choice(n_filt, n_sample, replace=False)
        samp = w_flat[idx]
        norms = np.linalg.norm(samp, axis=1, keepdims=True) + 1e-9
        normed = samp / norms
        cos_sim = normed @ normed.T
        np.fill_diagonal(cos_sim, 0)
        diversity = float(1 - np.abs(cos_sim).mean())  # 0=all same, 1=all different

        status = "✅"
        if dead_pct > 50:
            status = "🔴 DEAD"
        elif dead_pct > 20:
            status = "🟡 sparse"
        elif diversity < 0.1:
            status = "🟡 low diversity"

        print(f"  {name:<35} {n_filt:>8}  {mean_w:>9.4f}  {std_w:>9.4f}  "
              f"{dead_pct:>6.1f}%  {diversity:>10.4f}  {status}")
        filter_summaries.append((name, module, w, n_filt, dead_pct, diversity))

    # ── Weight distribution histograms per layer ───────────────────────────────
    section("Weight Distribution Histograms")
    print(textwrap.fill(
        "  Healthy weights: roughly Gaussian, centered near 0. "
        "Heavy tails = exploding. All near 0 = not learning.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))

    for name, module, w, n_filt, dead_pct, diversity in filter_summaries[:4]:
        w_flat = w.flatten()
        print(f"\n  Layer: {name}  ({n_filt} filters × {w.shape[1]} in-ch × "
              f"{w.shape[2]}×{w.shape[3]} kernel)")
        hist, edges = np.histogram(w_flat, bins=16)
        max_h = hist.max() + 1
        for i in range(len(hist)):
            bar  = "█" * int(hist[i] / max_h * 30)
            sign = "+" if edges[i] >= 0 else "-"
            print(f"    {edges[i]:>8.4f} │{bar:<30}│ {hist[i]:>5}")

    # ── First-layer filter visualization ──────────────────────────────────────
    section("First-Layer Filters — ASCII Visualization")
    print(textwrap.fill(
        "  First conv layer learns low-level detectors: edges, blobs, "
        "color gradients. Each filter below is one learned detector.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))

    if filter_summaries:
        first_name, first_mod, first_w, n_filt, _, _ = filter_summaries[0]
        n_show = min(n_filt, 8)
        print(f"\n  Showing {n_show}/{n_filt} filters from '{first_name}':")
        for fi in range(n_show):
            filt = first_w[fi]  # (in_ch, kH, kW)
            if filt.shape[0] > 1:
                filt_2d = filt.mean(axis=0)  # average across channels
            else:
                filt_2d = filt[0]
            kH, kW = filt_2d.shape
            mn, mx = filt_2d.min(), filt_2d.max()
            rng    = mx - mn + 1e-9
            chars  = " ░▒▓█"
            print(f"\n  Filter #{fi+1}  (min={mn:.4f}  max={mx:.4f})")
            print("  ┌" + "──" * kW + "┐")
            for row in filt_2d:
                line = "".join(chars[int((v - mn) / rng * (len(chars)-1))] * 2
                               for v in row)
                print(f"  │{line}│")
            print("  └" + "──" * kW + "┘")

    # ── BatchNorm statistics ───────────────────────────────────────────────────
    bn_layers = adapter.get_bn_layers()
    if bn_layers:
        section("BatchNorm Running Statistics")
        print(f"  {'Layer':<35} {'Run Mean':>10}  {'Run Var':>10}  "
              f"{'γ (scale)':>10}  {'β (bias)':>10}")
        print("  " + "─" * 75)
        for name, module in bn_layers[:10]:
            if HAS_TORCH:
                rm  = module.running_mean.mean().item() if module.running_mean is not None else 0
                rv  = module.running_var.mean().item()  if module.running_var  is not None else 1
                gam = module.weight.mean().item()        if module.weight       is not None else 1
                bet = module.bias.mean().item()          if module.bias         is not None else 0
                print(f"  {name:<35} {rm:>10.4f}  {rv:>10.4f}  {gam:>10.4f}  {bet:>10.4f}")

    # ── Gradient-free weight change monitor (if old weights available) ─────────
    section("Weight Magnitude Summary")
    print("  Layer-wise L2 norms — should grow during early training:\n")
    for name, module in conv_layers[:8]:
        if HAS_TORCH:
            w_norm = module.weight.data.norm().item()
            bar    = "█" * int(min(w_norm / 2.0, 1.0) * 30)
            print(f"  {name:<35}  ||W||₂ = {w_norm:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — PREDICTION WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════

def prediction_walkthrough(adapter, val_loader, class_names=None,
                             task="classification", n_show=5):
    """
    Per-image prediction breakdown:
    - top-k probabilities with confidence bars
    - correct / wrong
    - margin between top-2 predictions (confidence)
    - worst predictions highlighted
    """
    banner("STEP 6 — PREDICTION WALKTHROUGH")

    batches   = []
    collected = 0
    for batch in val_loader:
        batches.append(batch)
        if isinstance(batch, (list, tuple)):
            collected += len(batch[0])
        else:
            collected += len(batch)
        if collected >= n_show * 4:
            break

    if not batches:
        print("  No data available")
        return

    # Combine batches
    all_imgs, all_labels = [], []
    for batch in batches:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, lbls = batch[0], batch[1]
        else:
            imgs, lbls = batch, None

        if HAS_TORCH and isinstance(imgs, torch.Tensor):
            all_imgs.append(imgs.cpu())
            if lbls is not None:
                all_labels.append(lbls.cpu())
        else:
            all_imgs.append(torch.tensor(np.array(imgs)))
            if lbls is not None:
                all_labels.append(torch.tensor(np.array(lbls)))

    X      = torch.cat(all_imgs)[:n_show * 4]
    has_y  = len(all_labels) > 0
    y      = torch.cat(all_labels)[:n_show * 4] if has_y else None

    probas = adapter.predict_proba(X)
    preds  = probas.argmax(axis=-1)

    section(f"Showing {min(n_show, len(X))} Predictions in Detail")

    correct_count  = 0
    wrong_indices  = []

    for i in range(min(n_show, len(X))):
        print(f"\n  {'─'*60}")
        print(f"  Sample #{i+1}")
        print(f"  {'─'*60}")

        img_shape = tuple(X[i].shape)
        print(f"  Input shape: {img_shape}")

        if has_y:
            true_label = int(y[i].item()) if hasattr(y[i], "item") else int(y[i])
            true_name  = class_names[true_label] if class_names and true_label < len(class_names) \
                         else f"Class {true_label}"
            print(f"  True label:  {true_name}  (idx={true_label})")

        pred_label = int(preds[i])
        pred_name  = class_names[pred_label] if class_names and pred_label < len(class_names) \
                     else f"Class {pred_label}"

        p = probas[i]
        top_k = min(5, len(p))
        top_idx = np.argsort(p)[::-1][:top_k]

        print(f"\n  Top-{top_k} predictions:")
        for rank, ci in enumerate(top_idx):
            cname = class_names[ci] if class_names and ci < len(class_names) else f"Class {ci}"
            bar   = "█" * int(p[ci] * 30)
            mark  = ""
            if has_y and ci == true_label:
                mark = " ← TRUE"
            if rank == 0:
                mark += " ← PREDICTED"
            print(f"    #{rank+1}  {cname:<20} {bar:<30} {p[ci]:.4f}{mark}")

        if has_y:
            correct = (pred_label == true_label)
            if correct:
                print(f"\n  Result: ✅ CORRECT")
                correct_count += 1
            else:
                print(f"\n  Result: ❌ WRONG  (predicted '{pred_name}', true was '{true_name}')")
                wrong_indices.append(i)

        # Confidence margin
        top2 = np.sort(p)[::-1][:2]
        margin = top2[0] - top2[1] if len(top2) > 1 else top2[0]
        conf_level = "HIGH" if margin > 0.5 else "MEDIUM" if margin > 0.2 else "LOW"
        bar = "█" * int(margin * 30)
        print(f"  Confidence margin (top-1 vs top-2): {margin:.4f}  {bar}  [{conf_level}]")

    # ── Summary ────────────────────────────────────────────────────────────────
    section("Walkthrough Summary")
    n_shown = min(n_show, len(X))
    if has_y:
        info("Correct predictions", f"{correct_count}/{n_shown}")
        info("Wrong predictions",   f"{n_shown - correct_count}/{n_shown}")
        if wrong_indices:
            print(f"\n  Wrong on samples: {wrong_indices}")
            print(f"  Investigate these images manually for label errors or edge cases")

    # Overall confidence distribution
    all_margins = np.sort(probas, axis=-1)[:, -1] - np.sort(probas, axis=-1)[:, -2]
    print(f"\n  Confidence margin distribution (all {len(X)} samples):")
    hist, edges = np.histogram(all_margins, bins=10)
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / (hist.max() + 1) * 25)
        print(f"    {edges[i]:.2f}-{edges[i+1]:.2f} │{bar:<25}│ {hist[i]:>4}")

    low_conf = np.sum(all_margins < 0.2)
    print(f"\n  Low-confidence predictions (<0.2 margin): {low_conf}/{len(X)}")
    if low_conf / len(X) > 0.3:
        print(f"  🟡 Many uncertain predictions — model may be underfit or miscalibrated")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluation_metrics(adapter, val_loader, class_names=None, task="classification"):
    """
    Full evaluation suite:
    - Accuracy, Top-5 accuracy
    - Per-class precision, recall, F1
    - Confusion matrix (ASCII, scales to any number of classes)
    - Calibration check (are probabilities trustworthy?)
    - Worst-performing classes
    """
    banner("STEP 7 — EVALUATION METRICS")

    all_preds, all_labels, all_probs = [], [], []

    for batch in val_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, lbls = batch[0], batch[1]
        else:
            print("  No labels in loader — skipping evaluation")
            return

        probs = adapter.predict_proba(imgs)
        preds = probs.argmax(axis=-1)

        if HAS_TORCH and isinstance(lbls, torch.Tensor):
            lbls = lbls.cpu().numpy()
        else:
            lbls = np.array(lbls)

        all_preds.append(preds)
        all_labels.append(lbls)
        all_probs.append(probs)

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels).astype(int)
    probs  = np.concatenate(all_probs)
    n      = len(labels)

    if task == "classification":
        classes = sorted(set(labels.tolist()))
        n_cls   = len(classes)

        # ── Overall accuracy ───────────────────────────────────────────────────
        section("Overall Performance")
        acc   = float(np.mean(preds == labels))
        info("Samples evaluated", n)
        info("Overall accuracy",  f"{acc:.4f}  ({acc*100:.2f}%)")
        bar = "█" * int(acc * 40)
        print(f"  Accuracy: {bar:<40} {acc:.4f}")

        # Top-5 accuracy
        if probs.shape[1] >= 5:
            top5_correct = 0
            for i in range(n):
                top5 = np.argsort(probs[i])[::-1][:5]
                if labels[i] in top5:
                    top5_correct += 1
            top5_acc = top5_correct / n
            info("Top-5 accuracy", f"{top5_acc:.4f}  ({top5_acc*100:.2f}%)")

        # ── Per-class metrics ──────────────────────────────────────────────────
        section("Per-Class Metrics")
        print(f"  {'Class':<25} {'Precision':>10}  {'Recall':>8}  "
              f"{'F1':>8}  {'Support':>8}  F1 bar")
        print("  " + "─" * 72)

        class_metrics = []
        for ci in classes:
            cname   = class_names[ci] if class_names and ci < len(class_names) \
                      else f"Class {ci}"
            tp      = int(np.sum((preds == ci) & (labels == ci)))
            fp      = int(np.sum((preds == ci) & (labels != ci)))
            fn      = int(np.sum((preds != ci) & (labels == ci)))
            support = int(np.sum(labels == ci))
            prec    = tp / (tp + fp + 1e-9)
            rec     = tp / (tp + fn + 1e-9)
            f1      = 2 * prec * rec / (prec + rec + 1e-9)
            bar     = "█" * int(f1 * 25)
            print(f"  {cname:<25} {prec:>10.4f}  {rec:>8.4f}  "
                  f"{f1:>8.4f}  {support:>8}  {bar}")
            class_metrics.append((cname, prec, rec, f1, support))

        macro_f1 = np.mean([m[3] for m in class_metrics])
        info("\n  Macro F1", f"{macro_f1:.4f}")

        # ── Confusion matrix ───────────────────────────────────────────────────
        section("Confusion Matrix")
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p_ in zip(labels, preds):
            cm[t, p_] += 1

        if n_cls <= 15:
            cw = max(8, max(len(class_names[c]) if class_names and c < len(class_names)
                            else len(f"Class {c}") for c in classes) + 2)
            header = " " * 18 + "  ".join(
                (class_names[c][:6] if class_names and c < len(class_names)
                 else f"P{c}")[:6].center(6)
                for c in classes)
            print(f"  {header}")
            print("  " + "─" * len(header))
            for i, ci in enumerate(classes):
                cname = (class_names[ci][:14] if class_names and ci < len(class_names)
                         else f"Class {ci}")[:14]
                row_vals = "  ".join(f"{cm[i,j]:>6}" for j in range(n_cls))
                correct_flag = "✅" if cm[i, i] == cm[i].sum() and cm[i].sum() > 0 else \
                               "❌" if cm[i, i] == 0 else "  "
                print(f"  {cname:<16} │ {row_vals} │ {correct_flag}")
        else:
            # Compact view for many classes
            diag_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-9)
            print("  (Matrix too large to display — showing per-class recall instead)\n")
            for i, ci in enumerate(classes[:20]):
                cname = class_names[ci] if class_names and ci < len(class_names) \
                        else f"Class {ci}"
                bar   = "█" * int(diag_acc[i] * 25)
                print(f"  {cname:<25} recall={diag_acc[i]:.4f}  {bar}")

        # ── Worst classes ──────────────────────────────────────────────────────
        section("Worst-Performing Classes")
        worst = sorted(class_metrics, key=lambda x: x[3])[:5]
        print(f"  {'Class':<25} {'F1':>8}  {'Prec':>8}  {'Rec':>8}  {'Support':>8}")
        print("  " + "─" * 62)
        for cname, prec, rec, f1, support in worst:
            flag = "🔴" if f1 < 0.5 else "🟡" if f1 < 0.75 else "✅"
            print(f"  {flag} {cname:<23} {f1:>8.4f}  {prec:>8.4f}  {rec:>8.4f}  {support:>8}")

        # ── Calibration check ──────────────────────────────────────────────────
        section("Calibration Check — Are Probabilities Trustworthy?")
        print(textwrap.fill(
            "  A calibrated model: when it says 80% confidence, it should be right ~80% "
            "of the time. Overconfident models say 99% but are right 70%.",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "))
        print()

        bins = np.arange(0, 1.1, 0.1)
        top_probs = probs.max(axis=1)
        print(f"  {'Conf bin':<15} {'Predicted':>10}  {'Actual acc':>12}  "
              f"{'Samples':>8}  Calibration")
        print("  " + "─" * 60)
        for i in range(len(bins) - 1):
            lo, hi    = bins[i], bins[i+1]
            mask      = (top_probs >= lo) & (top_probs < hi)
            n_bin     = int(mask.sum())
            if n_bin == 0:
                continue
            bin_acc   = float(np.mean(preds[mask] == labels[mask]))
            bin_conf  = float(top_probs[mask].mean())
            gap       = abs(bin_conf - bin_acc)
            flag      = "✅" if gap < 0.1 else "🟡" if gap < 0.2 else "🔴 OVER-CONF"
            bar_a     = "█" * int(bin_acc  * 20)
            bar_c     = "░" * int(bin_conf * 20)
            print(f"  [{lo:.1f}-{hi:.1f}]        {bin_conf:>10.3f}  {bin_acc:>12.3f}  "
                  f"{n_bin:>8}  {flag}")

    else:
        # Regression metrics
        section("Regression Metrics")
        preds_float = probs.flatten()
        if HAS_TORCH:
            preds_float = adapter.predict_batch(
                torch.cat([b[0] for b in val_loader])
            ).numpy().flatten()
        mse  = float(np.mean((preds_float - labels)**2))
        mae  = float(np.mean(np.abs(preds_float - labels)))
        rmse = mse ** 0.5
        ss_res = np.sum((labels - preds_float)**2)
        ss_tot = np.sum((labels - labels.mean())**2)
        r2   = 1 - ss_res / (ss_tot + 1e-9)
        info("MSE",  f"{mse:.6f}")
        info("RMSE", f"{rmse:.6f}")
        info("MAE",  f"{mae:.6f}")
        info("R²",   f"{r2:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — OVERFITTING DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════

def cnn_overfitting_diagnosis(adapter, train_loader, val_loader,
                               tracer=None, class_names=None, task="classification"):
    """
    CNN-specific overfitting diagnosis:
    - Train/val gap analysis
    - Capacity vs data ratio
    - Regularization inventory
    - Model-specific fixes
    - Mini learning curve if tracer available
    """
    banner("STEP 8 — OVERFITTING DIAGNOSIS ENGINE")

    model      = adapter.model
    total_p, _ = adapter.count_parameters()
    model_name = type(model).__name__

    try:
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
    except Exception:
        n_train, n_val = 0, 0

    # ── Compute current scores ─────────────────────────────────────────────────
    section("Train vs Validation Accuracy")
    def _compute_acc(loader):
        correct, total = 0, 0
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                return None
            imgs, lbls = batch[0], batch[1]
            probs = adapter.predict_proba(imgs)
            preds = probs.argmax(axis=-1)
            if HAS_TORCH and isinstance(lbls, torch.Tensor):
                lbls = lbls.numpy()
            else:
                lbls = np.array(lbls)
            correct += int(np.sum(preds == lbls.astype(int)))
            total   += len(lbls)
        return correct / total if total > 0 else None

    train_acc = _compute_acc(train_loader)
    val_acc   = _compute_acc(val_loader)

    if train_acc is None or val_acc is None:
        print("  Could not compute accuracy (loader has no labels)")
    else:
        bar_t = "█" * int(train_acc * 40)
        bar_v = "░" * int(val_acc   * 40)
        print(f"  Train accuracy: {train_acc:.4f}  {bar_t}")
        print(f"  Val   accuracy: {val_acc:.4f}  {bar_v}")
        gap  = train_acc - val_acc
        print(f"  Gap (train-val): {gap:+.4f}\n")

        if gap > 0.15:
            regime = "OVERFITTING"
            print(f"  Diagnosis: 🔴 OVERFITTING DETECTED")
        elif gap > 0.05:
            regime = "MILD_OVERFIT"
            print(f"  Diagnosis: 🟡 MILD OVERFITTING")
        elif train_acc < 0.6 and val_acc < 0.6:
            regime = "UNDERFITTING"
            print(f"  Diagnosis: 🔴 UNDERFITTING")
        else:
            regime = "HEALTHY"
            print(f"  Diagnosis: ✅ HEALTHY")

    # ── Regularization inventory ───────────────────────────────────────────────
    section("Regularization Inventory")
    bn_count   = sum(1 for _, m in adapter.get_bn_layers())
    drop_count = 0
    drop_rates = []
    if HAS_TORCH:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                drop_count += 1
                drop_rates.append(m.p)

    print(f"  {'BatchNorm layers:':<35} {bn_count}")
    print(f"  {'Dropout layers:':<35} {drop_count}"
          + (f"  (rates: {[f'{r:.2f}' for r in drop_rates]})" if drop_rates else ""))

    # Check optimizer weight decay
    wd = None
    if HAS_TORCH:
        try:
            for m in model.modules():
                pass  # can't easily get optimizer here
        except Exception:
            pass

    reg_score = 0
    if bn_count > 0:
        reg_score += 1
        print(f"  ✅ BatchNorm present — stabilizes and mildly regularizes")
    else:
        print(f"  🟡 No BatchNorm — consider adding for stability + regularization")

    if drop_count > 0:
        reg_score += 1
        print(f"  ✅ Dropout present")
    else:
        print(f"  🟡 No Dropout — add before FC layers (rate=0.3-0.5)")

    # ── Capacity vs data ratio ─────────────────────────────────────────────────
    section("Model Capacity vs Dataset Size")
    info("Total parameters", f"{total_p:,}")
    info("Training samples", f"{n_train:,}")
    if n_train > 0:
        ratio = total_p / n_train
        info("Param/sample ratio", f"{ratio:.1f}x")
        print()
        if ratio > 100:
            print(f"  🔴 {ratio:.0f}x more parameters than training samples")
            print(f"     This model almost certainly needs:")
            print(f"     → Transfer learning from pretrained weights")
            print(f"     → Heavy augmentation (random crops, flips, color jitter)")
            print(f"     → Strong regularization (dropout ≥ 0.5, weight decay ≥ 1e-4)")
        elif ratio > 10:
            print(f"  🟡 {ratio:.1f}x param/sample ratio — moderate overfit risk")
            print(f"     → Use data augmentation")
            print(f"     → Consider weight decay = 1e-4")
        else:
            print(f"  ✅ Reasonable capacity/data ratio ({ratio:.1f}x)")

    # ── CNN-specific causes and fixes ──────────────────────────────────────────
    section("CNN-Specific Diagnosis")

    causes = []
    fixes  = []

    if HAS_TORCH and train_acc and val_acc:
        gap = train_acc - val_acc
        if gap > 0.05:
            if n_train < 1000:
                causes.append("Very small training set for a CNN")
                fixes.append("Use pretrained model (torchvision.models) + fine-tune")
            if drop_count == 0:
                causes.append("No dropout regularization")
                fixes.append("Add nn.Dropout(0.4) before classification head")
            if bn_count == 0:
                causes.append("No BatchNorm — less stable, weaker regularization")
                fixes.append("Add nn.BatchNorm2d after every conv layer")
            try:
                first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
                if first_conv.kernel_size[0] > 7:
                    causes.append(f"Large first-layer kernel ({first_conv.kernel_size}) — may overfit")
                    fixes.append("Use 3×3 or 5×5 kernels; larger kernels rarely help")
            except StopIteration:
                pass

    if causes:
        print(f"  Likely causes:\n")
        for c in causes:
            print(f"    • {c}")
        print(f"\n  Recommended fixes:\n")
        for f_ in fixes:
            print(f"    → {f_}")
    elif train_acc and val_acc:
        gap = train_acc - val_acc
        if gap <= 0.05:
            print(f"  ✅ No significant overfitting. Model generalizes well.")

    # ── Training curve from tracer ─────────────────────────────────────────────
    if tracer and tracer.history:
        section("Training Curve (from TrainingTracer)")
        train_losses = [r["train_loss"] for r in tracer.history]
        val_losses   = [r["val_loss"] for r in tracer.history if r["val_loss"] is not None]
        if train_losses and val_losses and len(val_losses) == len(train_losses):
            gaps = [t - v for t, v in zip(train_losses, val_losses)]
            max_gap = max(abs(g) for g in gaps) + 1e-9
            print(f"  {'Epoch':>7}  {'Train L':>10}  {'Val L':>10}  {'Gap':>10}  Gap bar")
            print("  " + "─" * 55)
            step = max(1, len(tracer.history) // 10)
            for r in tracer.history[::step]:
                ep, tl, vl = r["epoch"], r["train_loss"], r["val_loss"]
                g = tl - vl if vl is not None else 0
                bar = "█" * int(abs(g) / max_gap * 20)
                sign = "↑" if g > 0.01 else "↓" if g < -0.01 else "~"
                print(f"  {ep:>7}  {tl:>10.4f}  {vl:>10.4f}  {g:>+10.4f}  {sign}{bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — GRADIENT FLOW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def gradient_flow_analysis(adapter, train_batch, criterion=None):
    """
    Per-layer gradient analysis after a single forward+backward pass:
    - Gradient norm per layer
    - Vanishing gradient detection
    - Exploding gradient detection
    - Layer-wise gradient ratio (gradient vs weight magnitude)
    - Dead layer identification
    """
    banner("STEP 9 — GRADIENT FLOW ANALYSIS")

    if not HAS_TORCH:
        print("  PyTorch required for gradient flow analysis")
        return

    model   = adapter.model
    device  = adapter.device

    if isinstance(train_batch, (list, tuple)) and len(train_batch) >= 2:
        x, y = train_batch[0], train_batch[1]
    else:
        print("  Need (images, labels) batch for gradient analysis")
        return

    x = x.to(device)
    y = y.to(device) if HAS_TORCH and isinstance(y, torch.Tensor) else \
        torch.tensor(np.array(y)).to(device)

    # Single forward + backward
    model.train()
    model.zero_grad()

    try:
        out = model(x)
        if criterion is not None:
            loss = criterion(out, y)
        else:
            # Default: cross-entropy or MSE
            if out.shape[-1] > 1:
                loss = nn.CrossEntropyLoss()(out, y.long())
            else:
                loss = nn.MSELoss()(out.squeeze(), y.float())
        loss.backward()
    except Exception as e:
        print(f"  Could not compute gradients: {e}")
        model.eval()
        return

    model.eval()

    # ── Collect gradient norms ─────────────────────────────────────────────────
    section("Per-Layer Gradient Norms")
    print(f"  {'Layer':<40} {'Param Norm':>12}  {'Grad Norm':>12}  "
          f"{'Ratio':>10}  Status")
    print("  " + "─" * 85)

    layer_grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        w_norm  = float(param.data.norm(2))
        g_norm  = float(param.grad.norm(2))
        ratio   = g_norm / (w_norm + 1e-9)
        layer_grads.append((name, w_norm, g_norm, ratio))

    max_g = max(g for _, _, g, _ in layer_grads) + 1e-9
    for name, wn, gn, ratio in layer_grads:
        if gn < 1e-7:
            status = "🔴 VANISHING"
        elif gn > 100:
            status = "🔴 EXPLODING"
        elif gn < 1e-4:
            status = "🟡 very small"
        elif ratio > 0.1:
            status = "🟡 large ratio"
        else:
            status = "✅"
        bar = "█" * int(min(gn / max_g, 1.0) * 25)
        print(f"  {name[:39]:<40} {wn:>12.4f}  {gn:>12.4f}  {ratio:>10.4f}  {status}  {bar}")

    # ── Gradient flow visualization ────────────────────────────────────────────
    section("Gradient Flow — Visual (per named layer)")
    print("  Shows gradient magnitude flowing back through the network.")
    print("  Healthy: roughly uniform bars. Vanishing: bars shrink to nothing early.\n")

    # Group by layer (combine weight and bias)
    layer_grad_map = {}
    for name, _, gn, _ in layer_grads:
        base = name.rsplit(".", 1)[0] if "." in name else name
        layer_grad_map[base] = layer_grad_map.get(base, 0.0) + gn

    items = list(layer_grad_map.items())
    max_v = max(v for _, v in items) + 1e-9
    for lname, gn in items:
        bar  = "█" * int(gn / max_v * 40)
        flag = "🔴" if gn < 1e-6 else "🟡" if gn < 1e-4 else "✅"
        print(f"  {flag} {lname[:35]:<35}  {gn:.2e}  {bar}")

    # ── Diagnosis ──────────────────────────────────────────────────────────────
    section("Gradient Health Diagnosis")
    vanishing = [(n, gn) for n, _, gn, _ in layer_grads if gn < 1e-6]
    exploding = [(n, gn) for n, _, gn, _ in layer_grads if gn > 100]
    total_g   = sum(gn for _, _, gn, _ in layer_grads)
    mean_g    = total_g / len(layer_grads) if layer_grads else 0

    info("Total gradient norm (all layers)", f"{total_g:.4f}")
    info("Mean  gradient norm per param",    f"{mean_g:.4f}")

    print()
    if vanishing:
        print(f"  🔴 VANISHING GRADIENTS detected in {len(vanishing)} parameter(s)")
        for n, gn in vanishing[:5]:
            print(f"     {n}: {gn:.2e}")
        print(f"\n  Fixes for vanishing gradients:")
        print(f"    → Use BatchNorm after each conv layer")
        print(f"    → Use residual connections (ResNet-style)")
        print(f"    → Use GELU or LeakyReLU instead of ReLU/tanh")
        print(f"    → Use He initialization: nn.init.kaiming_normal_")
        print(f"    → Reduce depth or use gradient clipping")

    if exploding:
        print(f"  🔴 EXPLODING GRADIENTS detected in {len(exploding)} parameter(s)")
        for n, gn in exploding[:5]:
            print(f"     {n}: {gn:.2e}")
        print(f"\n  Fixes for exploding gradients:")
        print(f"    → Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
        print(f"    → Reduce learning rate")
        print(f"    → Add BatchNorm")

    if not vanishing and not exploding:
        print(f"  ✅ Gradients look healthy — no vanishing or exploding detected")
        print(f"     Mean gradient norm: {mean_g:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 — FEATURE MAP VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

def feature_map_visualizer(adapter, sample_image, layer_names=None, n_maps=4):
    """
    Show what each conv layer actually sees when processing an image.
    Renders activation maps as ASCII heatmaps.
    """
    banner("STEP 10 — FEATURE MAP VISUALIZER")

    if not HAS_TORCH:
        print("  PyTorch required for feature map visualization")
        return

    model  = adapter.model
    device = adapter.device

    if isinstance(sample_image, (list, tuple)):
        x = sample_image[0]
    else:
        x = sample_image

    if isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        x = torch.tensor(np.array(x), dtype=torch.float32).to(device)

    if x.dim() == 3:
        x = x.unsqueeze(0)  # add batch dim

    # ── Get all conv layers if none specified ──────────────────────────────────
    conv_layers_list = adapter.get_conv_layers()
    if not conv_layers_list:
        print("  No conv layers found")
        return

    if layer_names:
        target_layers = [(n, m) for n, m in conv_layers_list if n in layer_names]
    else:
        target_layers = conv_layers_list[:4]  # show first 4 conv layers

    # ── Hook activations ───────────────────────────────────────────────────────
    feat_maps = {}
    hooks = []

    def make_hook(n):
        def h(mod, inp, out):
            feat_maps[n] = out.detach().cpu()
        return h

    for name, module in target_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        out = model(x[:1])

    for h in hooks:
        h.remove()

    # ── Render each layer's feature maps ──────────────────────────────────────
    section("What the Network Sees — Layer by Layer")
    print(textwrap.fill(
        "  Each heatmap shows one feature map from a conv layer. "
        "Bright areas = high activation = the filter detected its pattern there. "
        "Early layers detect edges; deeper layers detect textures, parts, objects.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))

    for name, _ in target_layers:
        if name not in feat_maps:
            continue
        fmap = feat_maps[name]  # (1, C, H, W)
        C, H, W = fmap.shape[1], fmap.shape[2], fmap.shape[3]

        print(f"\n  ═══ Layer: {name}  "
              f"({C} feature maps, spatial size {H}×{W}) ═══")

        n_show = min(n_maps, C)
        for fi in range(n_show):
            fm = fmap[0, fi].numpy()
            ascii_heatmap(fm, rows=8, cols=20,
                          label=f"  Filter #{fi+1}  "
                                f"(mean={fm.mean():.3f}  std={fm.std():.3f}  "
                                f"sparsity={float((fm < 1e-5).mean())*100:.0f}%)")

        # Summary stats across all maps in this layer
        all_maps = fmap[0].numpy()  # (C, H, W)
        mean_act  = float(all_maps.mean())
        sparsity  = float((all_maps < 1e-5).mean() * 100)
        top_maps  = np.argsort(all_maps.reshape(C, -1).mean(axis=1))[::-1][:3]
        print(f"\n  Layer summary: mean={mean_act:.4f}  sparsity={sparsity:.1f}%")
        print(f"  Most active filter indices: {list(top_maps)}")
        if sparsity > 80:
            print(f"  🟡 High sparsity ({sparsity:.0f}%) — many filters inactive on this input")
        else:
            print(f"  ✅ Activations look diverse")

    # ── Input image heatmap ────────────────────────────────────────────────────
    section("Input Image (ASCII preview)")
    img_np = x[0].cpu().numpy()
    ascii_heatmap(img_np, rows=10, cols=24, label="  Input image (channel-averaged)")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11 — SALIENCY / ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def saliency_analysis(adapter, sample_batch, class_names=None, n_samples=3):
    """
    Gradient-based saliency maps — which pixels drive the prediction?
    - Vanilla gradient saliency
    - Gradient × Input attribution
    - Per-image ASCII saliency heatmap
    - Most / least salient regions summary
    No extra libraries required (pure autograd).
    """
    banner("STEP 11 — SALIENCY & ATTRIBUTION MAPS")

    if not HAS_TORCH:
        print("  PyTorch required for saliency analysis")
        return

    model  = adapter.model
    device = adapter.device

    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
        imgs, labels = sample_batch[0], sample_batch[1]
    else:
        imgs   = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        labels = None

    imgs = imgs.to(device)
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device)

    section("What This Shows")
    print(textwrap.fill(
        "  Saliency maps answer: which pixels most influenced this prediction? "
        "Computed by backpropagating the predicted class score to the input. "
        "High-saliency pixels = the model 'looked here' to decide.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))

    model.eval()

    for si in range(min(n_samples, len(imgs))):
        x     = imgs[si:si+1].clone().requires_grad_(True)
        label = int(labels[si].item()) if labels is not None else None

        out     = model(x)
        pred_cl = int(out.argmax(dim=-1).item())
        pred_nm = class_names[pred_cl] if class_names and pred_cl < len(class_names) \
                  else f"Class {pred_cl}"

        model.zero_grad()
        score = out[0, pred_cl]
        score.backward()

        # Vanilla gradient saliency
        grad  = x.grad.data.abs()  # (1, C, H, W)
        sal   = grad[0].mean(dim=0).cpu().numpy()  # (H, W)

        # Gradient × Input
        gxi   = (x.grad.data * x.data).abs()
        gxi_m = gxi[0].mean(dim=0).cpu().numpy()

        print(f"\n  {'─'*60}")
        print(f"  Sample #{si+1}  |  Predicted: {pred_nm}  (class {pred_cl})")
        if label is not None:
            true_nm = class_names[label] if class_names and label < len(class_names) \
                      else f"Class {label}"
            correct = "✅" if pred_cl == label else "❌"
            print(f"  True label: {true_nm}  {correct}")
        print(f"  Confidence: {float(torch.softmax(out, dim=-1)[0, pred_cl]):.4f}")

        # ASCII saliency heatmap
        print(f"\n  Gradient Saliency Map (bright = model focused here):")
        ascii_heatmap(sal, rows=10, cols=24)

        print(f"\n  Gradient × Input Attribution:")
        ascii_heatmap(gxi_m, rows=10, cols=24)

        # Spatial statistics
        H, W = sal.shape
        quad = {
            "top-left":     sal[:H//2,  :W//2 ].mean(),
            "top-right":    sal[:H//2,   W//2:].mean(),
            "bottom-left":  sal[H//2:,  :W//2 ].mean(),
            "bottom-right": sal[H//2:,   W//2:].mean(),
            "center":       sal[H//4:3*H//4, W//4:3*W//4].mean(),
        }
        print(f"\n  Saliency by region:")
        max_q = max(quad.values()) + 1e-9
        for region, val_ in sorted(quad.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(val_ / max_q * 25)
            print(f"    {region:<15} {val_:.4f}  {bar}")

        # Top salient pixels
        flat_idx = np.argsort(sal.flatten())[::-1][:5]
        rows_top = flat_idx // W
        cols_top = flat_idx % W
        print(f"\n  Top-5 most salient pixel locations (row, col):")
        for r, c in zip(rows_top, cols_top):
            print(f"    ({r:>3}, {c:>3})  saliency={sal[r,c]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 12 — TRAINING DECISION TRACE
# ══════════════════════════════════════════════════════════════════════════════

def training_decision_trace(adapter, train_batch, optimizer=None,
                              criterion=None, epoch=None):
    """
    Explains WHY the model updated the way it did:
    - Which layers received the largest updates
    - BatchNorm statistics (running mean/var stability)
    - Dropout effect (how much is being dropped)
    - Loss decomposition per sample
    - LR × gradient = actual update magnitude per layer
    """
    banner("STEP 12 — TRAINING DECISION TRACE")

    if not HAS_TORCH:
        print("  PyTorch required")
        return

    model  = adapter.model
    device = adapter.device

    if isinstance(train_batch, (list, tuple)) and len(train_batch) >= 2:
        x, y = train_batch[0], train_batch[1]
    else:
        print("  Need (images, labels) batch")
        return

    x = x.to(device)
    y = y.to(device) if isinstance(y, torch.Tensor) else \
        torch.tensor(np.array(y)).to(device)

    if epoch is not None:
        section(f"Training Step Analysis  (epoch {epoch})")
    else:
        section("Training Step Analysis")

    # ── Per-sample loss decomposition ──────────────────────────────────────────
    section("Per-Sample Loss — Who Is Hard to Learn?")
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    model.train()

    n_show = min(10, len(x))
    if out_eval.shape[-1] > 1:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = loss_fn(out_eval, y.long()).cpu().numpy()
    else:
        loss_fn = nn.MSELoss(reduction="none")
        per_sample_loss = loss_fn(out_eval.squeeze(), y.float()).cpu().numpy()

    preds   = out_eval.argmax(dim=-1).cpu().numpy()
    labels  = y.cpu().numpy()

    print(f"  {'Sample':>8}  {'Loss':>10}  {'Pred':>8}  {'True':>8}  Loss bar")
    print("  " + "─" * 55)
    max_l   = per_sample_loss.max() + 1e-9
    for i in range(n_show):
        bar    = "█" * int(per_sample_loss[i] / max_l * 25)
        correct = "✅" if preds[i] == int(labels[i]) else "❌"
        print(f"  {i:>8}  {per_sample_loss[i]:>10.4f}  {preds[i]:>8}  "
              f"{int(labels[i]):>8}  {correct} {bar}")

    mean_l = per_sample_loss.mean()
    hard   = per_sample_loss > mean_l * 2
    print(f"\n  Mean loss: {mean_l:.4f}  |  "
          f"Hard samples (>2× mean): {int(hard.sum())}/{n_show}")

    # ── Actual weight update magnitudes ────────────────────────────────────────
    section("Weight Update Magnitudes  (lr × grad)")
    model.train()
    model.zero_grad()

    out_train = model(x)
    if out_train.shape[-1] > 1:
        loss = nn.CrossEntropyLoss()(out_train, y.long())
    else:
        loss = nn.MSELoss()(out_train.squeeze(), y.float())
    loss.backward()

    lr = optimizer.param_groups[0]["lr"] if optimizer else 0.001

    print(f"  Loss: {loss.item():.4f}  |  Learning rate: {lr:.2e}\n")
    print(f"  {'Parameter':<40} {'Update Mag':>12}  {'Relative':>10}  Update bar")
    print("  " + "─" * 72)

    updates = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            update_mag = float(param.grad.norm(2) * lr)
            weight_mag = float(param.data.norm(2))
            relative   = update_mag / (weight_mag + 1e-9)
            updates.append((name, update_mag, relative))

    max_u = max(u for _, u, _ in updates) + 1e-9
    for name, um, rel in sorted(updates, key=lambda x: x[1], reverse=True)[:15]:
        bar  = "█" * int(um / max_u * 30)
        flag = "🔴" if rel > 0.1 else "🟡" if rel > 0.01 else "✅"
        print(f"  {flag} {name[:39]:<40} {um:>12.2e}  {rel:>10.4f}  {bar}")

    # ── BatchNorm current statistics ───────────────────────────────────────────
    bn_layers = adapter.get_bn_layers()
    if bn_layers:
        section("BatchNorm Statistics — Current State")
        print(f"  {'Layer':<35} {'Run Mean':>10}  {'Run Var':>10}  "
              f"{'γ':>8}  {'β':>8}  Status")
        print("  " + "─" * 75)
        for name, module in bn_layers[:8]:
            rm  = module.running_mean.mean().item() if module.running_mean is not None else 0
            rv  = module.running_var.mean().item()  if module.running_var  is not None else 1
            gam = module.weight.mean().item()        if module.weight       is not None else 1
            bet = module.bias.mean().item()          if module.bias         is not None else 0
            status = "✅"
            if rv < 1e-6:
                status = "🔴 var≈0"
            elif rv > 100:
                status = "🟡 high var"
            elif abs(rm) > 10:
                status = "🟡 high mean"
            print(f"  {name:<35} {rm:>10.4f}  {rv:>10.4f}  {gam:>8.4f}  {bet:>8.4f}  {status}")

    # ── Dropout effect ─────────────────────────────────────────────────────────
    if HAS_TORCH:
        drop_layers = [(n, m) for n, m in model.named_modules()
                       if isinstance(m, (nn.Dropout, nn.Dropout2d))]
        if drop_layers:
            section("Dropout Effect")
            print(f"  {'Layer':<35} {'Rate':>8}  Expected % dropped")
            print("  " + "─" * 55)
            for name, m in drop_layers:
                rate = m.p
                bar  = "█" * int(rate * 30)
                print(f"  {name:<35} {rate:>8.2f}  {bar}  {rate*100:.0f}% of activations zeroed")

    # ── Summary ────────────────────────────────────────────────────────────────
    section("Training Step Summary")
    total_update = sum(u for _, u, _ in updates)
    max_single   = max(updates, key=lambda x: x[1]) if updates else None
    info("Total update magnitude", f"{total_update:.4f}")
    if max_single:
        info("Largest single update", f"{max_single[0]}  ({max_single[1]:.4f})")
    large_rel = [(n, r) for n, _, r in updates if r > 0.1]
    if large_rel:
        print(f"\n  🟡 {len(large_rel)} parameter(s) with large relative updates (>10% of weight):")
        for n, r in large_rel[:3]:
            print(f"     {n}: {r:.4f}")
        print(f"     → Consider gradient clipping or lower lr")
    else:
        print(f"\n  ✅ All parameter updates are proportionally small — stable training")




# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — run_cnn_pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_cnn_pipeline(
    model,
    train_loader,
    val_loader,
    class_names     = None,
    task            = "classification",
    optimizer       = None,
    scheduler       = None,
    criterion       = None,
    tracer          = None,
    input_shape     = None,
    augmentation_info = None,
    n_walkthrough   = 5,
    run_steps       = None,      # None = all steps; or list like [0,1,4,7]
    sample_batch    = None,      # Override sample batch for activation/saliency
):
    """
    Run the full CNN Transparency Pipeline.

    Parameters
    ----------
    model           : PyTorch nn.Module or Keras Model
    train_loader    : DataLoader with (images, labels) batches
    val_loader      : DataLoader with (images, labels) batches
    class_names     : list of str, e.g. ['cat', 'dog']
    task            : 'classification' or 'regression'
    optimizer       : torch.optim.Optimizer (optional, for setup/gradient steps)
    scheduler       : LR scheduler (optional)
    criterion       : Loss function (optional)
    tracer          : TrainingTracer instance (optional, for training curves)
    input_shape     : (C, H, W) tuple for architecture analysis
    augmentation_info : str describing augmentations (optional)
    n_walkthrough   : number of samples to show in prediction walkthrough
    run_steps       : list of step indices to run (default: all)
                      Steps: 0=health 1=arch 2=setup 3=training 4=activations
                             5=filters 6=walkthrough 7=metrics 8=overfit
                             9=gradients 10=featuremaps 11=saliency 12=decisions
    sample_batch    : optional (imgs, labels) tuple to use for activation/saliency/gradient steps
    """

    adapter = _make_adapter(model)
    all_steps = set(range(13)) if run_steps is None else set(run_steps)

    # Get a sample batch for steps that need one
    _sample = sample_batch
    if _sample is None:
        try:
            _sample = next(iter(train_loader))
        except Exception:
            _sample = next(iter(val_loader))

    # ── Banner ─────────────────────────────────────────────────────────────────
    print()
    print("█" * WIDTH)
    print("█" * 22 + "  CNN TRANSPARENCY PIPELINE  " + "█" * 29)
    print("█" * WIDTH)
    print(f"  Model       : {type(model).__name__}")
    print(f"  Framework   : {adapter.framework()}")
    print(f"  Task        : {task.upper()}")
    total_p, trainable_p = adapter.count_parameters()
    print(f"  Parameters  : {total_p:,}  ({trainable_p:,} trainable)")
    if class_names:
        print(f"  Classes     : {len(class_names)}  ({', '.join(class_names[:6])}"
              f"{'...' if len(class_names) > 6 else ''})")
    try:
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        print(f"  Train size  : {n_train:,}")
        print(f"  Val size    : {n_val:,}")
    except Exception:
        pass
    print(f"  PyTorch     : {'available' if HAS_TORCH else 'not installed'}")
    print(f"  TensorFlow  : {'available' if HAS_TF    else 'not installed'}")

    # ── Run steps ──────────────────────────────────────────────────────────────

    if 0 in all_steps:
        health_issues = cnn_dataset_health(
            train_loader, class_names=class_names, task=task)
    else:
        health_issues = []

    if 1 in all_steps:
        architecture_visualizer(adapter, input_shape=input_shape)

    if 2 in all_steps:
        training_setup_explainer(
            optimizer=optimizer, scheduler=scheduler,
            criterion=criterion, train_loader=train_loader,
            augmentation_info=augmentation_info)

    if 3 in all_steps:
        if tracer and tracer.history:
            tracer.report()
        else:
            banner("STEP 3 — TRAINING LOOP TRACE")
            print("  No training history yet.")
            print("  Usage:")
            print("    tracer = TrainingTracer()")
            print("    for epoch in range(epochs):")
            print("        train_loss, train_acc = train_one_epoch(...)")
            print("        val_loss, val_acc     = validate(...)")
            print("        tracer.record(epoch, train_loss, train_acc,")
            print("                      val_loss, val_acc, model=model, optimizer=opt)")
            print("    run_cnn_pipeline(..., tracer=tracer)")

    if 4 in all_steps:
        activation_analysis(adapter, _sample, class_names=class_names)

    if 5 in all_steps:
        filter_visualizer(adapter)

    if 6 in all_steps:
        prediction_walkthrough(adapter, val_loader, class_names=class_names,
                                task=task, n_show=n_walkthrough)

    if 7 in all_steps:
        evaluation_metrics(adapter, val_loader, class_names=class_names, task=task)

    if 8 in all_steps:
        cnn_overfitting_diagnosis(adapter, train_loader, val_loader,
                                   tracer=tracer, class_names=class_names, task=task)

    if 9 in all_steps:
        gradient_flow_analysis(adapter, _sample, criterion=criterion)

    if 10 in all_steps:
        if _sample is not None:
            img = _sample[0] if isinstance(_sample, (list, tuple)) else _sample
            feature_map_visualizer(adapter, img)

    if 11 in all_steps:
        saliency_analysis(adapter, _sample, class_names=class_names, n_samples=3)

    if 12 in all_steps:
        training_decision_trace(adapter, _sample, optimizer=optimizer,
                                 criterion=criterion)

    # ── Final summary ──────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE", "█")
    print(f"  Model       : {type(model).__name__}")
    print(f"  Parameters  : {total_p:,}")
    if class_names:
        print(f"  Classes     : {len(class_names)}")
    print()
    print("  Steps completed:")
    step_names = {
        0: "Dataset Health Report",      1: "Architecture Visualizer",
        2: "Training Setup Explainer",   3: "Training Loop Trace",
        4: "Activation Analysis",        5: "Filter Visualizer",
        6: "Prediction Walkthrough",     7: "Evaluation Metrics",
        8: "Overfitting Diagnosis",      9: "Gradient Flow Analysis",
        10: "Feature Map Visualizer",    11: "Saliency & Attribution",
        12: "Training Decision Trace",
    }
    for s in sorted(all_steps):
        print(f"    ✅ Step {s:>2}: {step_names.get(s, '?')}")

    print()
    print("  The model object is returned for further use.")
    return model