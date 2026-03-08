"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SENTIMENT ANALYSIS — NLP TRANSPARENCY PIPELINE DEMO               ║
║                                                                              ║
║  Two modes (auto-detected):                                                  ║
║    A) REAL DATA  — put plain-text files in:                                 ║
║         data/train/positive.txt  (one review per line)                     ║
║         data/train/negative.txt  (one review per line)                     ║
║         data/val/positive.txt                                               ║
║         data/val/negative.txt                                               ║
║                                                                              ║
║    B) SYNTHETIC  — no dataset needed, runs immediately                      ║
║       Generates realistic-looking positive/negative movie reviews           ║
║                                                                              ║
║  Two model architectures (choose with --model):                             ║
║    lstm        — Bidirectional LSTM (default)                               ║
║    transformer — Mini Transformer encoder                                   ║
║                                                                              ║
║  Usage:                                                                      ║
║    python nlp_demo.py                          # auto mode, LSTM            ║
║    python nlp_demo.py --model transformer      # mini transformer           ║
║    python nlp_demo.py --mode synthetic         # force synthetic            ║
║    python nlp_demo.py --epochs 15 --lr 3e-4   # custom training            ║
║    python nlp_demo.py --steps 0 1 4 5 10      # run specific steps         ║
║    python nlp_demo.py --no-train               # pipeline on fresh model    ║
║                                                                              ║
║  Requirements:  pip install torch                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, re, time, math, random, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sentiment NLP Pipeline Demo")
parser.add_argument("--model",    choices=["lstm","transformer"], default="lstm")
parser.add_argument("--mode",     choices=["auto","real","synthetic"], default="auto")
parser.add_argument("--epochs",   type=int,   default=10)
parser.add_argument("--batch",    type=int,   default=32)
parser.add_argument("--max-len",  type=int,   default=128)
parser.add_argument("--lr",       type=float, default=3e-4)
parser.add_argument("--data-dir", type=str,   default="data")
parser.add_argument("--steps",    type=int,   nargs="*", default=None)
parser.add_argument("--no-train", action="store_true")
parser.add_argument("--vocab-size",type=int,  default=5000)
args = parser.parse_args()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch not installed.\n  pip install torch")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  TOKENIZER — simple word-level tokenizer, no external deps
# ══════════════════════════════════════════════════════════════════════════════
class SimpleTokenizer:
    """
    Word-level tokenizer.
    - Lowercases and strips punctuation
    - Builds vocab from training corpus
    - Encodes/decodes with PAD=0, UNK=1, [CLS]=2, [SEP]=3
    """
    PAD_ID = 0; UNK_ID = 1; CLS_ID = 2; SEP_ID = 3
    SPECIAL = ["[PAD]","[UNK]","[CLS]","[SEP]"]

    def __init__(self, vocab_size=5000, max_length=128):
        self.vocab_size  = vocab_size
        self.max_length  = max_length
        self.pad_token_id = self.PAD_ID
        self.unk_token_id = self.UNK_ID
        self.special_tokens = self.SPECIAL
        self.word2id = {w:i for i,w in enumerate(self.SPECIAL)}
        self.id2word = {i:w for i,w in enumerate(self.SPECIAL)}

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def build_vocab(self, texts):
        freq = {}
        for text in texts:
            for w in self._tokenize(text):
                freq[w] = freq.get(w, 0) + 1
        # Most frequent words up to vocab_size
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top = top[:self.vocab_size - len(self.SPECIAL)]
        for w, _ in top:
            if w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w] = idx
                self.id2word[idx] = w
        self.vocab_size = len(self.word2id)
        print(f"  Vocabulary built: {self.vocab_size:,} tokens")

    def encode(self, text, add_special=False):
        tokens = self._tokenize(text)
        max_t  = self.max_length - (2 if add_special else 0)
        ids    = [self.word2id.get(w, self.UNK_ID) for w in tokens[:max_t]]
        if add_special:
            ids = [self.CLS_ID] + ids + [self.SEP_ID]
        # Pad
        ids = ids[:self.max_length]
        ids = ids + [self.PAD_ID] * (self.max_length - len(ids))
        return ids

    def decode(self, ids):
        return self.id2word.get(int(ids[0]), "[UNK]") if ids else "[PAD]"

    def decode_sequence(self, ids):
        return " ".join(self.id2word.get(int(i),"[UNK]") for i in ids if i!=0)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATASET
# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary pools for generating reviews
_POS_OPENERS = [
    "This film is absolutely", "I absolutely loved", "A masterpiece of",
    "One of the best movies", "Brilliant and",
    "The director delivers a", "Stunning performances and",
    "An emotionally powerful", "Genuinely moving and", "Outstanding cinema,",
]
_NEG_OPENERS = [
    "This film is absolutely", "I absolutely hated", "A disaster of",
    "One of the worst movies", "Dull and",
    "The director fails to", "Wooden performances and",
    "An emotionally empty", "Painfully boring and", "Terrible cinema,",
]
_POS_MIDS = [
    "wonderful storytelling", "breathtaking visuals", "superb acting",
    "a gripping narrative", "outstanding character development",
    "a perfectly crafted script", "mesmerising direction",
    "moving and thought-provoking", "layered and nuanced performances",
    "a heartfelt emotional journey",
]
_NEG_MIDS = [
    "awful storytelling", "terrible visuals", "wooden acting",
    "a boring narrative", "no character development",
    "a poorly written script", "directionless editing",
    "cold and unengaging", "flat and forgettable performances",
    "a pointless emotional void",
]
_POS_ENDS = [
    "I highly recommend it.", "Do not miss this film.",
    "A must-see for everyone.", "Will watch again for sure.",
    "A true cinematic gem.", "Five stars without hesitation.",
    "Left me speechless.", "This deserves every award.",
    "My favourite film of the year.", "Absolutely unforgettable.",
]
_NEG_ENDS = [
    "I strongly advise skipping it.", "Do not waste your time.",
    "A complete waste of money.", "I walked out halfway through.",
    "One of the worst I have seen.", "Zero stars if possible.",
    "Left me deeply disappointed.", "This deserves no awards.",
    "My least favourite film ever.", "Completely forgettable.",
]
_FILLER = [
    "The movie", "This production", "The story", "The whole experience",
    "The cast", "The screenplay", "The direction", "Every scene",
]

def _make_review(label, rng, min_words=15, max_words=60):
    """Generate a synthetic positive (1) or negative (0) review."""
    openers  = _POS_OPENERS  if label else _NEG_OPENERS
    mids     = _POS_MIDS     if label else _NEG_MIDS
    ends     = _POS_ENDS     if label else _NEG_ENDS
    n_sents  = rng.randint(2, 5)
    parts    = [rng.choice(openers) + " " + rng.choice(mids) + "."]
    for _ in range(n_sents - 2):
        filler = rng.choice(_FILLER)
        mid    = rng.choice(mids)
        parts.append(f"{filler} shows {mid}.")
    parts.append(rng.choice(ends))
    return " ".join(parts)


class SyntheticSentimentDataset(Dataset):
    """Balanced positive/negative sentiment dataset."""
    def __init__(self, n_samples=1000, tokenizer=None, split="train", seed=42):
        rng = random.Random(seed)
        texts, labels = [], []
        for i in range(n_samples):
            lbl = i % 2
            texts.append(_make_review(lbl, rng))
            labels.append(lbl)

        # Shuffle
        combined = list(zip(texts, labels))
        rng.shuffle(combined)
        texts, labels = zip(*combined)

        if tokenizer is None:
            raise ValueError("tokenizer required")

        self.ids    = torch.tensor([tokenizer.encode(t) for t in texts],
                                    dtype=torch.long)
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        self.texts  = list(texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  REAL DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.ids    = torch.tensor([tokenizer.encode(t) for t in texts], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.texts  = texts
    def __len__(self):   return len(self.labels)
    def __getitem__(self, i): return self.ids[i], self.labels[i]


def load_real_data(data_dir, split, tokenizer):
    texts, labels = [], []
    for lbl_idx, fname in enumerate(["negative.txt","positive.txt"]):
        path = os.path.join(data_dir, split, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected: {path}\n"
                f"  Structure:\n"
                f"    {data_dir}/{split}/positive.txt  (one review per line)\n"
                f"    {data_dir}/{split}/negative.txt  (one review per line)")
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(lbl_idx)
    return TextDataset(texts, labels, tokenizer)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — BIDIRECTIONAL LSTM SENTIMENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment classification.

    Architecture:
      Embedding(V, D) → Dropout → BiLSTM(D, H, L layers)
                      → [last hidden fwd + last hidden bwd]
                      → Dropout → FC(2H, H) → ReLU → FC(H, n_classes)

    The bidirectional design lets the model read context from both
    directions — important for sentences like
    "not bad at all" (negation before the positive word).
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 num_layers=2, n_classes=2, dropout=0.4, pad_idx=0):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim,
                                        padding_idx=pad_idx)
        self.emb_drop   = nn.Dropout(dropout)
        self.lstm       = nn.LSTM(embed_dim, hidden_size, num_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if num_layers>1 else 0.0)
        self.drop2      = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size*2, hidden_size)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(hidden_size, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        self.embedding.weight.data[0].zero_()  # pad = zero vector
        for name, param in self.lstm.named_parameters():
            if "weight" in name: nn.init.orthogonal_(param)
            if "bias"   in name: nn.init.zeros_(param)

    def forward(self, x):
        # x: (B, T)
        emb    = self.emb_drop(self.embedding(x))           # (B, T, D)
        out, (hn, _) = self.lstm(emb)                       # out:(B,T,2H)
        # Use last real hidden state (not padding)
        # hn: (num_layers*2, B, H)
        # Take last layer fwd + bwd
        fwd = hn[-2]; bwd = hn[-1]                          # each (B, H)
        h   = torch.cat([fwd, bwd], dim=1)                  # (B, 2H)
        h   = self.drop2(h)
        h   = self.relu(self.fc1(h))
        return self.fc2(h)                                   # (B, n_classes)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — MINI TRANSFORMER ENCODER
# ══════════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SentimentTransformer(nn.Module):
    """
    Mini Transformer encoder for sentiment classification.

    Architecture:
      Embedding(V, D) + PositionalEncoding
      → N × TransformerEncoderLayer(D, heads, FF, dropout)
      → Mean pooling over non-padding positions
      → FC(D, D//2) → GELU → Dropout → FC(D//2, n_classes)

    Uses mean pooling rather than [CLS] so every token contributes
    to the final prediction — better for short sequences.
    """
    def __init__(self, vocab_size, embed_dim=128, n_heads=4,
                 ff_dim=256, n_layers=2, n_classes=2,
                 dropout=0.3, max_len=256, pad_idx=0):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len, dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True)  # pre-LN for stability
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.drop       = nn.Dropout(dropout)
        self.fc1        = nn.Linear(embed_dim, embed_dim//2)
        self.act        = nn.GELU()
        self.fc2        = nn.Linear(embed_dim//2, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, 0, 0.02)
        self.embedding.weight.data[self.pad_idx].zero_()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T)
        pad_mask = (x == self.pad_idx)                          # (B, T) True=padding
        emb      = self.pos_enc(self.embedding(x))              # (B, T, D)
        enc      = self.encoder(emb, src_key_padding_mask=pad_mask)  # (B, T, D)
        # Mean pool over real tokens
        mask_f   = (~pad_mask).unsqueeze(-1).float()            # (B, T, 1)
        pooled   = (enc * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        out      = self.drop(pooled)
        out      = self.act(self.fc1(out))
        return self.fc2(out)                                     # (B, n_classes)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device,
                    grad_clip=1.0, grad_accum=1):
    model.train()
    total_loss = total_correct = total = 0
    # Accumulate squared gradient norms across all optimizer steps this epoch
    # so we can report the mean pre-clip norm to the tracer.
    sum_sq_norm = 0.0
    n_opt_steps = 0
    optimizer.zero_grad()
    for step, (ids, lbls) in enumerate(loader):
        ids  = ids.to(device)
        lbls = lbls.to(device)
        out  = model(ids)
        loss = criterion(out, lbls) / grad_accum
        loss.backward()
        if (step+1) % grad_accum == 0:
            # ── Capture norm AFTER backward, BEFORE clip+zero ─────────────────
            raw_sq = sum(
                p.grad.data.norm(2).item()**2
                for p in model.parameters() if p.grad is not None
            )
            sum_sq_norm += raw_sq
            n_opt_steps += 1
            # ─────────────────────────────────────────────────────────────────
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        total_loss    += loss.item() * grad_accum * len(lbls)
        total_correct += (out.argmax(dim=1)==lbls).sum().item()
        total         += len(lbls)
    mean_grad_norm = (sum_sq_norm / max(n_opt_steps, 1)) ** 0.5
    return total_loss/total, total_correct/total, mean_grad_norm


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for ids, lbls in loader:
            ids  = ids.to(device)
            lbls = lbls.to(device)
            out  = model(ids)
            total_loss    += criterion(out, lbls).item() * len(lbls)
            total_correct += (out.argmax(dim=1)==lbls).sum().item()
            total         += len(lbls)
    return total_loss/total, total_correct/total


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, epochs, tracer):
    print(f"\n  Training {type(model).__name__} for {epochs} epoch(s) on {device}")
    print(f"  {'─'*67}")
    print(f"  {'Epoch':>7}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>9}  {'LR':>10}  Time")
    print(f"  {'─'*67}")

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(epochs):
        t0 = time.time()
        tl, ta, grad_norm = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        tracer.record(epoch, tl, ta, vl, va,
                      optimizer=optimizer, grad_norm=grad_norm)

        flag = " ⚠" if ta-va > 0.15 else " ★" if va > best_val_acc else "  "
        print(f"  {epoch+1:>7}  {tl:>12.4f}  {ta:>10.4f}  "
              f"{vl:>10.4f}  {va:>9.4f}  {lr:>10.2e}  {time.time()-t0:.1f}s{flag}")

        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k:v.clone() for k,v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  ✅ Restored best weights (val_acc={best_val_acc:.4f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  WARMUP + COSINE LR SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing — standard for NLP."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale    = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale    = max(scale, self.min_lr / self.base_lrs[0])
        return [lr * scale for lr in self.base_lrs]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("█"*70)
    print("█"*13 + "  SENTIMENT ANALYSIS — NLP PIPELINE DEMO  " + "█"*14)
    print("█"*70)

    VOCAB_SIZE = args.vocab_size
    MAX_LEN    = args.max_len
    BATCH      = args.batch
    EPOCHS     = args.epochs
    LR         = args.lr
    DATA_DIR   = args.data_dir
    MODEL_TYPE = args.model

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Model arch : {MODEL_TYPE.upper()}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH}")
    print(f"  Max length : {MAX_LEN}")
    print(f"  LR         : {LR}")

    CLASS_NAMES = ["negative", "positive"]

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\n  Building tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE, max_length=MAX_LEN)

    # ── Detect mode ───────────────────────────────────────────────────────────
    mode = args.mode
    if mode == "auto":
        mode = "real" if (
            os.path.exists(os.path.join(DATA_DIR,"train","positive.txt")) and
            os.path.exists(os.path.join(DATA_DIR,"train","negative.txt"))
        ) else "synthetic"
    print(f"  Mode       : {mode.upper()}")

    # ── Build datasets ────────────────────────────────────────────────────────
    if mode == "real":
        print(f"\n  Loading real reviews from {DATA_DIR}/ ...")
        try:
            # Build vocab from training texts first
            train_path_pos = os.path.join(DATA_DIR,"train","positive.txt")
            train_path_neg = os.path.join(DATA_DIR,"train","negative.txt")
            all_train_texts = []
            for p in [train_path_pos, train_path_neg]:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    all_train_texts.extend(f.read().splitlines())
            tokenizer.build_vocab(all_train_texts)

            train_ds   = load_real_data(DATA_DIR, "train", tokenizer)
            val_ds     = load_real_data(DATA_DIR, "val",   tokenizer)
            sample_texts = all_train_texts[:10]
            print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
        except (FileNotFoundError, Exception) as e:
            print(f"\n  ⚠  Real data failed: {e}")
            print(f"  Falling back to SYNTHETIC mode.\n")
            mode = "synthetic"

    if mode == "synthetic":
        print(f"\n  Generating synthetic sentiment reviews...")
        n_train = max(BATCH*20, 600)
        n_val   = max(BATCH*5,  150)
        # Generate texts first to build vocab
        rng = random.Random(42)
        raw_train = [_make_review(i%2, rng) for i in range(n_train)]
        raw_val   = [_make_review(i%2, random.Random(99)) for i in range(n_val)]
        tokenizer.build_vocab(raw_train)
        train_ds  = SyntheticSentimentDataset(n_train, tokenizer, "train", seed=42)
        val_ds    = SyntheticSentimentDataset(n_val,   tokenizer, "val",   seed=99)
        sample_texts = raw_train[:8]
        print(f"  Train: {n_train}  Val: {n_val}")

    VOCAB_SIZE = tokenizer.vocab_size  # actual size after vocab build

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\n  Building {MODEL_TYPE.upper()} model...")
    if MODEL_TYPE == "lstm":
        model = SentimentLSTM(
            vocab_size   = VOCAB_SIZE,
            embed_dim    = 128,
            hidden_size  = 256,
            num_layers   = 2,
            n_classes    = len(CLASS_NAMES),
            dropout      = 0.4,
            pad_idx      = tokenizer.PAD_ID,
        ).to(device)
    else:  # transformer
        model = SentimentTransformer(
            vocab_size = VOCAB_SIZE,
            embed_dim  = 128,
            n_heads    = 4,
            ff_dim     = 256,
            n_layers   = 2,
            n_classes  = len(CLASS_NAMES),
            dropout    = 0.3,
            max_len    = MAX_LEN + 10,
            pad_idx    = tokenizer.PAD_ID,
        ).to(device)

    total_p  = sum(p.numel() for p in model.parameters())
    train_p  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {total_p:,}  ({train_p:,} trainable)")

    # ── Optimizer / scheduler / loss ──────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                             betas=(0.9, 0.999))
    total_steps   = EPOCHS * len(train_loader)
    warmup_steps  = max(1, total_steps // 10)
    scheduler     = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    criterion     = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── NLPTrainingTracer ──────────────────────────────────────────────────────
    try:
        from Nlp_pipeline import run_nlp_pipeline, NLPTrainingTracer
    except ImportError:
        print("\n  ERROR: nlp_pipeline.py not found in the same directory.")
        sys.exit(1)

    tracer = NLPTrainingTracer()

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.no_train and EPOCHS > 0:
        model = train_model(model, train_loader, val_loader,
                            optimizer, scheduler, criterion,
                            device, EPOCHS, tracer)
    else:
        print(f"\n  Skipping training (--no-train)")

    # ── Rebuild optimizer/scheduler for pipeline (grad step needs clean state) ─
    opt_pipeline  = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched_pipeline= WarmupCosineScheduler(opt_pipeline, warmup_steps, total_steps)

    # ── Top words for embedding nearest-neighbour analysis ───────────────────
    top_words = ["good","bad","great","terrible","love","hate",
                 "boring","exciting","beautiful","awful"]
    top_words = [w for w in top_words if w in tokenizer.word2id]

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    print(f"\n  Launching NLP Transparency Pipeline...\n")
    run_nlp_pipeline(
        model             = model,
        train_loader      = train_loader,
        val_loader        = val_loader,
        tokenizer         = tokenizer,
        class_names       = CLASS_NAMES,
        task              = "classification",
        optimizer         = opt_pipeline,
        scheduler         = sched_pipeline,
        criterion         = criterion,
        tracer            = tracer,
        grad_accumulation = 1,
        sample_texts      = sample_texts,
        top_words         = top_words,
        run_steps         = args.steps,
    )


if __name__ == "__main__":
    main()