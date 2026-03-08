"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            DOG vs CAT — CNN TRANSPARENCY PIPELINE DEMO                      ║
║                                                                              ║
║  Two modes (auto-detected):                                                  ║
║    A) REAL DATA  — put images in:                                            ║
║         data/train/cats/  *.jpg / *.png                                     ║
║         data/train/dogs/  *.jpg / *.png                                     ║
║         data/val/cats/    *.jpg / *.png                                     ║
║         data/val/dogs/    *.jpg / *.png                                     ║
║       → Trains a real CNN, then runs the full pipeline                      ║
║                                                                              ║
║    B) SYNTHETIC DATA  — no dataset needed                                   ║
║       → Generates random "images", trains quickly, runs pipeline            ║
║       → All 13 steps still run and all output is meaningful                 ║
║                                                                              ║
║  Usage:                                                                      ║
║    python dog_cat_demo.py                        # auto-detect mode          ║
║    python dog_cat_demo.py --mode synthetic       # force synthetic          ║
║    python dog_cat_demo.py --mode real            # force real data          ║
║    python dog_cat_demo.py --epochs 20            # change epoch count       ║
║    python dog_cat_demo.py --steps 0 1 4 7 9 11  # run only specific steps  ║
║                                                                              ║
║  Requirements:                                                               ║
║    pip install torch torchvision                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import warnings
import time
warnings.filterwarnings("ignore")

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Dog vs Cat CNN Pipeline Demo")
parser.add_argument("--mode",   choices=["auto", "real", "synthetic"], default="auto")
parser.add_argument("--epochs", type=int, default=10,
                    help="Training epochs (default: 10)")
parser.add_argument("--batch",  type=int, default=32,
                    help="Batch size (default: 32)")
parser.add_argument("--img-size", type=int, default=64,
                    help="Image size H=W (default: 64)")
parser.add_argument("--steps", type=int, nargs="*", default=None,
                    help="Pipeline steps to run (default: all 0-12)")
parser.add_argument("--data-dir", type=str, default="data",
                    help="Root data directory for real mode")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 0.001)")
parser.add_argument("--no-train", action="store_true",
                    help="Skip training (pipeline on untrained model)")
args = parser.parse_args()

# ── Check PyTorch ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
    import torchvision.transforms as T
    import numpy as np
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Install with:  pip install torch torchvision")
    sys.exit(1)

# ── Check for PIL (needed only for real image loading) ────────────────────────
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Check for torchvision datasets ───────────────────────────────────────────
try:
    from torchvision.datasets import ImageFolder
    HAS_IMAGEFOLDER = True
except ImportError:
    HAS_IMAGEFOLDER = False


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION — DogCatCNN
#  A clean, well-commented CNN with:
#    - 4 conv blocks (conv → BN → ReLU → MaxPool)
#    - Global average pooling (no large FC bottleneck)
#    - Dropout for regularization
#    - Binary classification head
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d"""
    def __init__(self, in_ch, out_ch, kernel=3, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),  # NOT inplace — pipeline hooks need the output
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DogCatCNN(nn.Module):
    """
    4-block CNN for binary Dog vs Cat classification.

    Input:  (N, 3, H, W)  — RGB image, H=W=64 by default
    Output: (N, 2)         — logits for [cat, dog]

    Architecture rationale:
      Block 1: 3 → 32   — learns edges and colours
      Block 2: 32 → 64  — learns textures and simple patterns
      Block 3: 64 → 128 — learns parts (ears, eyes, snout)
      Block 4: 128 → 256 — learns high-level features (face shape)
      GAP: collapses spatial dims without a huge FC layer
      Head: 256 → 128 → 2 with dropout
    """
    def __init__(self, img_size=64, n_classes=2):
        super().__init__()
        self.img_size  = img_size
        self.n_classes = n_classes

        # Feature extraction
        self.block1 = ConvBlock(3,   32,  kernel=3, pool=True)   # → (32, H/2, W/2)
        self.block2 = ConvBlock(32,  64,  kernel=3, pool=True)   # → (64, H/4, W/4)
        self.block3 = ConvBlock(64,  128, kernel=3, pool=True)   # → (128, H/8, W/8)
        self.block4 = ConvBlock(128, 256, kernel=3, pool=True)   # → (256, H/16, W/16)

        # Global average pooling — (256, H/16, W/16) → (256,)
        self.gap     = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.dropout = nn.Dropout(0.4)
        self.fc1     = nn.Linear(256, 128)
        self.relu_fc = nn.ReLU(inplace=False)
        self.fc2     = nn.Linear(128, n_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATASET  — generates "dog" and "cat" images procedurally
#  Dogs: warm-toned images (more R/G)
#  Cats: cool-toned images (more B/G)
#  With realistic noise, shapes, and variation so the model can actually learn
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticDogCatDataset(Dataset):
    """
    Procedurally generated dog/cat images.

    Class 0 = cat: cool tones, circular central patch (round face)
    Class 1 = dog: warm tones, elongated central patch (snout shape)

    Not realistic, but gives the CNN a genuine learnable signal so all
    pipeline diagnostics produce meaningful output.
    """
    def __init__(self, n_samples=1000, img_size=64, split="train", seed=42):
        self.n       = n_samples
        self.size    = img_size
        self.split   = split
        rng          = np.random.default_rng(seed)

        imgs, labels = [], []
        for i in range(n_samples):
            label = i % 2  # perfectly balanced
            img   = self._make_image(label, rng, img_size)
            imgs.append(img)
            labels.append(label)

        self.images = torch.tensor(np.stack(imgs), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Normalise to mean≈0.5, std≈0.5
        self.images = (self.images - self.images.mean()) / (self.images.std() + 1e-5)
        # Clamp to [-2.5, 2.5]
        self.images = self.images.clamp(-2.5, 2.5)

        # Augmentation transforms for training
        if split == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            ])
        else:
            self.transform = None

    def _make_image(self, label, rng, size):
        """Generate one synthetic image (C, H, W) in [0, 1]."""
        img = np.zeros((3, size, size), dtype=np.float32)
        cx, cy = size // 2, size // 2

        # Background gradient (different per class)
        for r in range(size):
            for c in range(size):
                if label == 0:  # cat: cool blue-green background
                    img[0, r, c] = 0.2 + 0.15 * rng.random()
                    img[1, r, c] = 0.4 + 0.15 * rng.random()
                    img[2, r, c] = 0.6 + 0.15 * rng.random()
                else:            # dog: warm orange-brown background
                    img[0, r, c] = 0.6 + 0.15 * rng.random()
                    img[1, r, c] = 0.4 + 0.15 * rng.random()
                    img[2, r, c] = 0.2 + 0.15 * rng.random()

        # Central "face" patch
        radius = size // 4
        for r in range(size):
            for c in range(size):
                dy = r - cy
                dx = c - cx
                if label == 0:  # cat: circular face
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radius:
                        # Light grey cat face
                        alpha = 1.0 - dist / radius
                        img[0, r, c] = img[0, r, c] * (1-alpha) + 0.75 * alpha
                        img[1, r, c] = img[1, r, c] * (1-alpha) + 0.75 * alpha
                        img[2, r, c] = img[2, r, c] * (1-alpha) + 0.80 * alpha
                else:            # dog: wider elliptical face + snout
                    rx, ry = radius * 1.3, radius * 0.9
                    if (dx/rx)**2 + (dy/ry)**2 < 1:
                        alpha = 1.0 - np.sqrt((dx/rx)**2 + (dy/ry)**2)
                        alpha = max(0, alpha)
                        img[0, r, c] = img[0, r, c] * (1-alpha) + 0.65 * alpha
                        img[1, r, c] = img[1, r, c] * (1-alpha) + 0.45 * alpha
                        img[2, r, c] = img[2, r, c] * (1-alpha) + 0.25 * alpha

        # Add "ears" — triangular dark patches
        ear_size = size // 8
        if label == 0:  # cat: pointy triangular ears
            for er in range(ear_size):
                for ec in range(er + 1):
                    # left ear
                    rr = cy - radius - er
                    cc = cx - radius // 2 + ec
                    if 0 <= rr < size and 0 <= cc < size:
                        img[:, rr, cc] = 0.15
                    # right ear
                    cc2 = cx + radius // 2 - ec
                    if 0 <= rr < size and 0 <= cc2 < size:
                        img[:, rr, cc2] = 0.15
        else:           # dog: rounded floppy ears
            for er in range(ear_size * 2):
                for ec in range(ear_size):
                    rr = cy - radius // 2 + er
                    cc_l = cx - radius - ec
                    cc_r = cx + radius + ec
                    if 0 <= rr < size and 0 <= cc_l < size:
                        img[0, rr, cc_l] = 0.35
                        img[1, rr, cc_l] = 0.22
                        img[2, rr, cc_l] = 0.10
                    if 0 <= rr < size and 0 <= cc_r < size:
                        img[0, rr, cc_r] = 0.35
                        img[1, rr, cc_r] = 0.22
                        img[2, rr, cc_r] = 0.10

        # Add eyes (two dark circles)
        eye_r = max(2, size // 16)
        for eye_cx in [cx - radius // 3, cx + radius // 3]:
            eye_cy = cy - radius // 4
            for r in range(size):
                for c in range(size):
                    if (r - eye_cy)**2 + (c - eye_cx)**2 < eye_r**2:
                        img[:, r, c] = 0.05  # dark eyes

        # Gaussian noise
        img += rng.normal(0, 0.04, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            # Apply augmentation
            img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════════════════════════
#  REAL IMAGE DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_real_dataset(data_dir, img_size, batch_size):
    """
    Expects structure:
      data_dir/train/cats/*.jpg
      data_dir/train/dogs/*.jpg
      data_dir/val/cats/*.jpg
      data_dir/val/dogs/*.jpg

    Returns (train_loader, val_loader, class_names)
    """
    if not HAS_PIL:
        raise RuntimeError("Pillow required for real images: pip install Pillow")
    if not HAS_IMAGEFOLDER:
        raise RuntimeError("torchvision required: pip install torchvision")

    train_transforms = T.Compose([
        T.Resize((img_size + 8, img_size + 8)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225]),
    ])
    val_transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")

    if not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"Train directory not found: {train_path}\n"
            f"Expected structure:\n"
            f"  {data_dir}/train/cats/  (*.jpg)\n"
            f"  {data_dir}/train/dogs/  (*.jpg)\n"
            f"  {data_dir}/val/cats/    (*.jpg)\n"
            f"  {data_dir}/val/dogs/    (*.jpg)")

    train_ds = ImageFolder(train_path, transform=train_transforms)
    val_ds   = ImageFolder(val_path,   transform=val_transforms)

    # Ensure consistent class order: cats=0, dogs=1
    class_to_idx = train_ds.class_to_idx
    print(f"  Class mapping: {class_to_idx}")
    class_names = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        class_names[idx] = name

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(imgs)
        preds       = out.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += len(imgs)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            out    = model(imgs)
            loss   = criterion(out, labels)
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(dim=1) == labels).sum().item()
            total      += len(imgs)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, epochs, tracer):
    """Full training loop with live progress and TrainingTracer integration."""
    print(f"\n  Training for {epochs} epoch(s) on {device}")
    print(f"  {'─'*65}")
    print(f"  {'Epoch':>7}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}  Time")
    print(f"  {'─'*65}")

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc     = validate(
            model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Record in tracer
        tracer.record(epoch, train_loss, train_acc,
                      val_loss, val_acc,
                      model=model, optimizer=optimizer)

        # Print row
        gap_flag = "⚠" if train_acc - val_acc > 0.15 else " "
        print(f"  {epoch+1:>7}  {train_loss:>12.4f}  {train_acc:>10.4f}  "
              f"{val_loss:>10.4f}  {val_acc:>9.4f}  {lr:>10.2e}  "
              f"{elapsed:.1f}s  {gap_flag}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  ✅ Restored best weights (val_acc={best_val_acc:.4f})")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("█" * 70)
    print("█" * 18 + "  DOG vs CAT — DEMO  " + "█" * 31)
    print("█" * 70)

    # ── Detect mode ───────────────────────────────────────────────────────────
    IMG_SIZE   = args.img_size
    BATCH_SIZE = args.batch
    EPOCHS     = args.epochs
    LR         = args.lr
    DATA_DIR   = args.data_dir

    mode = args.mode
    if mode == "auto":
        real_exists = (
            HAS_PIL and HAS_IMAGEFOLDER
            and os.path.isdir(os.path.join(DATA_DIR, "train"))
        )
        mode = "real" if real_exists else "synthetic"

    print(f"\n  Mode       : {mode.upper()}")
    print(f"  Image size : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  LR         : {LR}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"  Device     : {device}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    CLASS_NAMES = ["cat", "dog"]

    if mode == "synthetic":
        print("\n  Building synthetic Dog vs Cat dataset...")
        print("  (cats=cool tones + round face,  dogs=warm tones + wide face + floppy ears)")

        n_train = max(BATCH_SIZE * 20, 400)
        n_val   = max(BATCH_SIZE * 5,  100)

        train_ds   = SyntheticDogCatDataset(n_train, IMG_SIZE, split="train", seed=42)
        val_ds     = SyntheticDogCatDataset(n_val,   IMG_SIZE, split="val",   seed=99)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        print(f"  Train samples : {n_train}")
        print(f"  Val   samples : {n_val}")

    else:
        print(f"\n  Loading real images from {DATA_DIR}/ ...")
        try:
            train_loader, val_loader, CLASS_NAMES = load_real_dataset(
                DATA_DIR, IMG_SIZE, BATCH_SIZE)
            print(f"  Train: {len(train_loader.dataset)}  "
                  f"Val: {len(val_loader.dataset)}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n  ERROR: {e}")
            print(f"\n  Falling back to SYNTHETIC mode.")
            mode = "synthetic"
            n_train = max(BATCH_SIZE * 20, 400)
            n_val   = max(BATCH_SIZE * 5,  100)
            train_ds     = SyntheticDogCatDataset(n_train, IMG_SIZE, split="train")
            val_ds       = SyntheticDogCatDataset(n_val,   IMG_SIZE, split="val")
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\n  Building DogCatCNN ({IMG_SIZE}×{IMG_SIZE} input)...")
    model = DogCatCNN(img_size=IMG_SIZE, n_classes=len(CLASS_NAMES)).to(device)

    total_p  = sum(p.numel() for p in model.parameters())
    train_p  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {total_p:,}  ({train_p:,} trainable)")

    # ── Optimizer, scheduler, loss ────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 100)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── TrainingTracer ────────────────────────────────────────────────────────
    # Import the pipeline
    try:
        from Cnn_pipeline import run_cnn_pipeline, TrainingTracer
    except ImportError:
        print("\n  ERROR: cnn_pipeline.py not found.")
        print("  Make sure cnn_pipeline.py is in the same directory as this script.")
        sys.exit(1)

    tracer = TrainingTracer()

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.no_train and EPOCHS > 0:
        print()
        model = train_model(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion,
            device, EPOCHS, tracer,
        )
    else:
        print(f"\n  Skipping training (--no-train flag set or epochs=0)")
        print(f"  Running pipeline on untrained model")

    # ── Reset optimizer for gradient step in pipeline ─────────────────────────
    # Rebuild optimizer pointing to same model (no state needed for grad step)
    optimizer_for_pipeline = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4)

    # ── Run CNN Pipeline ──────────────────────────────────────────────────────
    print("\n  Launching CNN Transparency Pipeline...\n")

    augmentation_description = (
        "RandomHorizontalFlip(p=0.5)  "
        "RandomAffine(degrees=15, translate=0.1)  "
        "[Synthetic mode — no ColorJitter or real crop]"
        if mode == "synthetic" else
        "Resize→RandomCrop→RandomHorizontalFlip→ColorJitter→Normalize(ImageNet)"
    )

    run_cnn_pipeline(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        class_names    = CLASS_NAMES,
        task           = "classification",
        optimizer      = optimizer_for_pipeline,
        scheduler      = scheduler,
        criterion      = criterion,
        tracer         = tracer,
        input_shape    = (3, IMG_SIZE, IMG_SIZE),
        augmentation_info = augmentation_description,
        n_walkthrough  = 5,
        run_steps      = args.steps,
    )


if __name__ == "__main__":
    main()