"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NLP TRANSPARENCY PIPELINE  —  nlp_pipeline.py                 ║
║  Terminal-only visibility into every aspect of an NLP model:                ║
║  dataset health, architecture, embeddings, attention, token attribution,    ║
║  gradient flow, neuron analysis, training dynamics, overfitting diagnosis.  ║
║                                                                              ║
║  Supports: PyTorch (LSTM, GRU, Transformer, custom)                         ║
║  Tasks:    classification | generation | sequence-labelling                 ║
║                                                                              ║
║  Usage:                                                                      ║
║    from nlp_pipeline import run_nlp_pipeline, NLPTrainingTracer             ║
║    run_nlp_pipeline(model, train_loader, val_loader, tokenizer,             ║
║                     class_names=['neg','pos'], task='classification')       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, time, copy, textwrap, warnings, math
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found.  pip install torch")

WIDTH = 80

# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def banner(title, char="═"):
    pad = max(0, WIDTH - len(title) - 4)
    print(); print(char*WIDTH)
    print(char*(pad//2) + "  " + title + "  " + char*(pad - pad//2))
    print(char*WIDTH)

def section(title):
    print(); print("─"*WIDTH); print(f"  ▶  {title}"); print("─"*WIDTH)

def subsection(title):
    print(f"\n  ····  {title}  ····")

def info(key, val):
    print(f"    {key+':':<35} {val}")

def hbar(val, width=35, char="█"):
    filled = max(0, int(abs(val) * width))
    return char * filled

def bar_chart(pairs, total=None, width=38):
    if not pairs: return
    mx = max(abs(v) for _, v in pairs) or 1
    for name, val in pairs:
        filled = int(abs(val)/mx*width); empty = width - filled
        bar = "█"*filled + "░"*empty
        pct = f"{val/total*100:.1f}%" if total else f"{val:.4g}"
        print(f"  {str(name)[:22]:<22} │{bar}│ {pct:>8}")

def ascii_bar(val, max_val, width=30, char="█"):
    return char * int(min(abs(val)/(max_val+1e-9), 1.0) * width)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — DATASET HEALTH REPORT
# ══════════════════════════════════════════════════════════════════════════════
def nlp_dataset_health(loader, tokenizer=None, class_names=None,
                        task="classification", n_inspect=512):
    banner("STEP 0 — DATASET HEALTH REPORT")
    issues = []

    # ── Collect samples ───────────────────────────────────────────────────────
    texts_raw, ids_all, labels_all = [], [], []
    collected = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            ids, lbls = batch[0], batch[1]
        else:
            ids, lbls = batch, None
        if HAS_TORCH and isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        ids_all.append(ids)
        if lbls is not None:
            if HAS_TORCH and isinstance(lbls, torch.Tensor):
                lbls = lbls.cpu().numpy()
            labels_all.extend(np.array(lbls).tolist())
        collected += len(ids)
        if collected >= n_inspect: break

    if not ids_all:
        print("  ⚠  No data collected from loader"); return issues

    ids_all = np.concatenate(ids_all, axis=0)[:n_inspect]

    # ── Sequence length distribution ──────────────────────────────────────────
    section("Sequence Length Distribution")
    lengths = (ids_all != 0).sum(axis=1) if ids_all.ndim == 2 else \
              np.array([len(s) for s in ids_all])
    info("Min length",    int(lengths.min()))
    info("Max length",    int(lengths.max()))
    info("Mean length",   f"{lengths.mean():.1f}")
    info("Median length", f"{float(np.median(lengths)):.1f}")
    info("Std length",    f"{lengths.std():.1f}")

    # ASCII histogram
    hist, edges = np.histogram(lengths, bins=12)
    max_h = hist.max() + 1
    print(f"\n  {'Len range':<14}  Distribution")
    print("  " + "─"*55)
    for i in range(len(hist)):
        bar = "█" * int(hist[i]/max_h*35)
        print(f"  [{int(edges[i]):>5}-{int(edges[i+1]):>5}]  │{bar:<35}│ {hist[i]:>5}")

    very_short = int((lengths < 5).sum())
    very_long  = int((lengths >= lengths.max() * 0.95).sum())
    if very_short > collected * 0.1:
        print(f"\n  🟡 {very_short} very short sequences (<5 tokens) — check tokenizer")
        issues.append(("WARNING","Many very short sequences","Check tokenizer / data cleaning"))
    if very_long > collected * 0.3:
        print(f"  🟡 {very_long} sequences at max length — many may be truncated")
        issues.append(("WARNING","Many truncated sequences",
                        "Increase max_length or use hierarchical model"))
    else:
        print(f"\n  ✅ Length distribution looks reasonable")

    # ── Padding ratio ─────────────────────────────────────────────────────────
    section("Padding Analysis")
    if ids_all.ndim == 2:
        total_tokens   = ids_all.size
        padding_tokens = int((ids_all == 0).sum())
        pad_pct        = padding_tokens / total_tokens * 100
        info("Total token slots",   f"{total_tokens:,}")
        info("Padding tokens",      f"{padding_tokens:,}  ({pad_pct:.1f}%)")
        if pad_pct > 50:
            print(f"  🟡 {pad_pct:.0f}% of tokens are padding — batch sorting or dynamic padding could help")
            issues.append(("WARNING",f"High padding ratio ({pad_pct:.0f}%)",
                            "Sort batches by length or use dynamic padding"))
        else:
            print(f"  ✅ Padding ratio is acceptable ({pad_pct:.1f}%)")
    else:
        print("  (variable-length sequences — no fixed padding analysis)")

    # ── Vocabulary coverage ───────────────────────────────────────────────────
    section("Vocabulary Coverage")
    all_token_ids = ids_all.flatten() if ids_all.ndim == 2 else \
                    np.concatenate([np.array(s) for s in ids_all])
    all_token_ids = all_token_ids[all_token_ids != 0]  # remove padding

    unique_ids = len(np.unique(all_token_ids))
    total_tok  = len(all_token_ids)
    info("Total tokens (excl. padding)", f"{total_tok:,}")
    info("Unique token IDs used",        f"{unique_ids:,}")
    info("Token ID range",               f"[{int(all_token_ids.min())}, {int(all_token_ids.max())}]")

    freq = Counter(all_token_ids.tolist())
    top10 = freq.most_common(10)

    subsection("Top 10 Most Frequent Token IDs")
    print(f"  {'Token ID':>10}  {'Count':>8}  {'%':>7}  Frequency bar")
    print("  " + "─"*52)
    max_freq = top10[0][1] if top10 else 1
    for tid, cnt in top10:
        decoded = ""
        if tokenizer and hasattr(tokenizer, "decode"):
            try: decoded = f"  '{tokenizer.decode([int(tid)])[:12]}'"
            except: pass
        bar = "█" * int(cnt/max_freq*25)
        print(f"  {int(tid):>10}  {cnt:>8}  {cnt/total_tok*100:>6.2f}%  {bar}{decoded}")

    # Zipf law check — token frequencies should follow power law
    sorted_freqs = sorted(freq.values(), reverse=True)[:200]
    if len(sorted_freqs) > 10:
        ranks  = np.arange(1, len(sorted_freqs)+1, dtype=float)
        freqs  = np.array(sorted_freqs, dtype=float)
        log_r  = np.log(ranks); log_f = np.log(freqs)
        slope  = np.polyfit(log_r, log_f, 1)[0]
        info("Zipf slope (log-log)",
             f"{slope:.3f}  (natural language ≈ -1.0)")
        if abs(slope + 1.0) < 0.3:
            print(f"  ✅ Token distribution follows Zipf's law — looks like real text")
        else:
            print(f"  🟡 Token distribution deviates from Zipf — check tokenizer/data")
            issues.append(("WARNING","Non-Zipfian token distribution",
                            "Verify tokenizer is correct and data is real text"))

    # Rare token analysis
    rare   = sum(1 for c in freq.values() if c == 1)
    rare_p = rare / unique_ids * 100
    info("Hapax legomena (seen once)", f"{rare:,}  ({rare_p:.1f}% of unique tokens)")
    if rare_p > 60:
        print(f"  🟡 Very high hapax rate — many rare tokens may hurt generalisation")
        issues.append(("WARNING",f"High hapax rate ({rare_p:.0f}%)",
                        "Consider subword tokenization (BPE/WordPiece)"))

    # ── Class distribution ────────────────────────────────────────────────────
    section("Label Distribution")
    if labels_all:
        counts = Counter(int(l) for l in labels_all)
        total  = sum(counts.values())
        pairs  = sorted(counts.items())
        display = [(class_names[k] if class_names and k < len(class_names)
                    else f"Class {k}", v) for k, v in pairs]
        bar_chart(display, total=total)
        maj = max(counts.values()); mn = min(counts.values())
        ratio = maj/mn
        info("Imbalance ratio", f"{ratio:.2f}x")
        if ratio > 5:
            print(f"  🔴 SEVERE IMBALANCE — use weighted loss")
            issues.append(("CRITICAL",f"Class imbalance {ratio:.1f}x",
                            "Use nn.CrossEntropyLoss(weight=...) or oversample"))
        elif ratio > 2:
            print(f"  🟡 MODERATE IMBALANCE — monitor per-class F1")
            issues.append(("WARNING",f"Class imbalance {ratio:.1f}x","Monitor per-class F1"))
        else:
            print(f"  ✅ Reasonably balanced")
    else:
        print("  (No labels found in loader)")

    # ── Duplicate detection ────────────────────────────────────────────────────
    section("Duplicate Sample Detection")
    if ids_all.ndim == 2:
        rows = [tuple(r.tolist()) for r in ids_all[:200]]
        dups = len(rows) - len(set(rows))
        if dups:
            print(f"  🟡 {dups} duplicate sequences found in sample")
            issues.append(("WARNING",f"{dups} duplicate sequences",
                            "Deduplicate before training"))
        else:
            print(f"  ✅ No duplicates in {min(len(ids_all),200)}-sample check")

    # ── Dataset size ───────────────────────────────────────────────────────────
    section("Dataset Size")
    try:    n_total = len(loader.dataset)
    except: n_total = collected
    info("Total samples", f"{n_total:,}")
    if n_total < 500:
        print(f"  🔴 Very small NLP dataset ({n_total}) — use pretrained embeddings")
        issues.append(("CRITICAL",f"Only {n_total} samples",
                        "Use pretrained word vectors or fine-tune BERT"))
    elif n_total < 5000:
        print(f"  🟡 Small dataset ({n_total}) — transfer learning recommended")
        issues.append(("WARNING",f"Small dataset ({n_total})",
                        "Consider pretrained embeddings (GloVe/FastText/BERT)"))
    else:
        print(f"  ✅ Dataset size looks adequate")

    # ── Scorecard ─────────────────────────────────────────────────────────────
    section("Health Scorecard")
    crits = [i for i in issues if i[0]=="CRITICAL"]
    warns = [i for i in issues if i[0]=="WARNING"]
    if not issues:
        print("  ✅ HEALTHY — Dataset looks ready for NLP training")
    else:
        if crits:
            print(f"  🔴 {len(crits)} CRITICAL issue(s):\n")
            for _,msg,fix in crits: print(f"     • {msg}\n       Fix: {fix}")
        if warns:
            print(f"\n  🟡 {len(warns)} WARNING(s):\n")
            for _,msg,fix in warns: print(f"     • {msg}\n       Fix: {fix}")
    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — ARCHITECTURE VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
def nlp_architecture_visualizer(model, tokenizer=None, sample_input=None):
    banner("STEP 1 — ARCHITECTURE VISUALIZER")
    if not HAS_TORCH: print("  PyTorch required"); return

    model_name = type(model).__name__
    total_p    = sum(p.numel() for p in model.parameters())
    train_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    section("Model Overview")
    info("Model class",           model_name)
    info("Total parameters",      f"{total_p:,}")
    info("Trainable parameters",  f"{train_p:,}")
    info("Frozen parameters",     f"{total_p - train_p:,}")
    info("Memory (float32)",      f"~{total_p*4/1024**2:.2f} MB")

    # ── Layer table ───────────────────────────────────────────────────────────
    section("Layer-by-Layer Breakdown")
    layer_info = []
    print(f"  {'Layer':<38} {'Type':<22} {'Params':>10}  {'%Bgt':>6}  Role")
    print("  " + "─"*85)

    for name, module in model.named_modules():
        if name == "": continue
        own_p = sum(p.numel() for p in module.parameters(recurse=False))
        if own_p == 0 and len(list(module.children())) > 0: continue
        mtype = type(module).__name__
        pct   = own_p / (total_p+1e-9) * 100
        role  = _nlp_layer_role(module)
        print(f"  {name[:37]:<38} {mtype[:21]:<22} {own_p:>10,}  {pct:>5.1f}%  {role}")
        layer_info.append((name, mtype, own_p, pct, role))

    # ── Parameter budget waterfall ────────────────────────────────────────────
    subsection("Parameter Budget (top layers)")
    top = sorted(layer_info, key=lambda x: x[2], reverse=True)[:10]
    for name, mtype, p, pct, _ in top:
        bar = "█" * int(pct*0.7)
        print(f"  {name[:30]:<30} {p:>10,}  {pct:>6.1f}%  {bar}")

    # ── NLP-specific architecture features ───────────────────────────────────
    section("NLP Architecture Features")

    # Embedding layers
    emb_layers = [(n,m) for n,m in model.named_modules()
                  if isinstance(m, nn.Embedding)]
    if emb_layers:
        print(f"  Embedding tables:")
        for n,m in emb_layers:
            print(f"    {n:<35}  vocab={m.num_embeddings:,}  dim={m.embedding_dim}"
                  f"  size={m.num_embeddings*m.embedding_dim*4/1024:.0f}KB")
        info("  Total embedding params",
             f"{sum(m.num_embeddings*m.embedding_dim for _,m in emb_layers):,}")

    # LSTM/GRU
    rnn_layers = [(n,m) for n,m in model.named_modules()
                  if isinstance(m,(nn.LSTM,nn.GRU,nn.RNN))]
    if rnn_layers:
        print(f"\n  Recurrent layers:")
        for n,m in rnn_layers:
            bidi = "bidirectional" if m.bidirectional else "unidirectional"
            print(f"    {n:<35}  {type(m).__name__}  "
                  f"hidden={m.hidden_size}  layers={m.num_layers}  {bidi}")

    # Transformer attention
    attn_layers = [(n,m) for n,m in model.named_modules()
                   if isinstance(m, nn.MultiheadAttention)]
    if attn_layers:
        print(f"\n  Attention layers:")
        for n,m in attn_layers:
            print(f"    {n:<35}  heads={m.num_heads}  "
                  f"d_model={m.embed_dim}  "
                  f"d_head={m.embed_dim//m.num_heads}")

    # Linear heads
    linear_layers = [(n,m) for n,m in model.named_modules()
                     if isinstance(m, nn.Linear)]
    if linear_layers:
        print(f"\n  Linear layers: {len(linear_layers)}")
        for n,m in linear_layers[-3:]:  # show last 3 (output head)
            print(f"    {n:<35}  {m.in_features} → {m.out_features}")

    # Dropout
    drop_layers = [(n,m) for n,m in model.named_modules()
                   if isinstance(m, nn.Dropout)]
    if drop_layers:
        rates = [m.p for _,m in drop_layers]
        print(f"\n  Dropout: {len(drop_layers)} layer(s), "
              f"rates={[f'{r:.2f}' for r in rates[:5]]}")

    # Diagnostics
    section("Architecture Diagnostics")
    if not emb_layers:
        print("  🟡 No Embedding layer found — verify input is pre-embedded")
    else:
        print(f"  ✅ Embedding layer(s) present ({len(emb_layers)})")

    if not drop_layers:
        print(f"  🟡 No Dropout — add dropout for regularization")
    else:
        print(f"  ✅ Dropout present ({len(drop_layers)} layers)")

    if total_p > 10_000_000:
        print(f"  🟡 Large model ({total_p/1e6:.1f}M params) — needs substantial data or pretrained init")
    elif total_p < 1000:
        print(f"  🟡 Very small model — may underfit complex language patterns")
    else:
        print(f"  ✅ Model size: {total_p:,} parameters")

    # Forward pass shape trace
    if sample_input is not None:
        section("Forward Pass Shape Trace")
        try:
            model.eval()
            hooks = {}; handles = []
            def make_hook(n):
                def h(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        hooks[n] = tuple(out.shape)
                    elif isinstance(out, (tuple,list)) and isinstance(out[0], torch.Tensor):
                        hooks[n] = tuple(out[0].shape)
                return h
            for n,m in model.named_modules():
                if n and len(list(m.children())) == 0:
                    handles.append(m.register_forward_hook(make_hook(n)))
            x = sample_input[0] if isinstance(sample_input,(list,tuple)) else sample_input
            if isinstance(x, torch.Tensor):
                with torch.no_grad(): model(x[:1])
            for h in handles: h.remove()
            print(f"  {'Layer':<40} Output shape")
            print("  " + "─"*60)
            for n, shp in list(hooks.items())[:15]:
                print(f"  {n[:39]:<40} {shp}")
        except Exception as e:
            print(f"  (shape trace failed: {e})")


def _nlp_layer_role(m):
    n = type(m).__name__
    return {
        "Embedding":          "Token → dense vector lookup",
        "LSTM":               "Sequential context encoder",
        "GRU":                "Sequential context encoder (lightweight)",
        "RNN":                "Sequential context encoder (basic)",
        "MultiheadAttention": "Self/cross attention",
        "TransformerEncoder": "Stack of attention + FFN blocks",
        "TransformerDecoder": "Stack of masked + cross attention blocks",
        "LayerNorm":          "Normalize across features",
        "BatchNorm1d":        "Normalize across batch",
        "Linear":             "Projection / classification head",
        "Dropout":            "Regularization",
        "ReLU":               "Non-linearity",
        "GELU":               "Smooth non-linearity (Transformer standard)",
        "Softmax":            "Output probabilities",
        "LogSoftmax":         "Log probabilities for NLLLoss",
        "Flatten":            "Reshape for FC layer",
        "Conv1d":             "Local n-gram feature extractor",
    }.get(n, n)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TRAINING SETUP EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
def nlp_training_setup(optimizer=None, scheduler=None, criterion=None,
                        tokenizer=None, train_loader=None,
                        grad_accumulation=1):
    banner("STEP 2 — TRAINING SETUP EXPLAINER")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    section("Optimizer")
    if optimizer is None:
        print("  (No optimizer provided)")
    elif HAS_TORCH:
        oname = type(optimizer).__name__
        info("Type", oname)
        pg = optimizer.param_groups[0]
        for k,v in pg.items():
            if k != "params": info(f"  {k}", v)

        expls = {
            "AdamW":  "Adam + decoupled weight decay. Standard for Transformers.",
            "Adam":   "Adaptive lr. Fast but may overfit small NLP datasets.",
            "SGD":    "Needs careful lr tuning. Rarely used for NLP from scratch.",
            "Adagrad":"Good for sparse gradients (large vocab). Lr decays over time.",
            "RMSprop":"Adaptive. Less common for NLP.",
        }
        print(f"\n  {expls.get(oname, oname)}")
        wd = pg.get("weight_decay", 0)
        if wd == 0:
            print(f"\n  🟡 weight_decay=0 — NLP models benefit from wd=0.01-0.1")
        else:
            print(f"\n  ✅ weight_decay={wd}")

    # ── Scheduler + warmup ────────────────────────────────────────────────────
    section("Learning Rate Scheduler")
    if scheduler is None:
        print("  (No scheduler provided)")
        print("  🟡 NLP models benefit greatly from linear warmup + decay")
    elif HAS_TORCH:
        sname = type(scheduler).__name__
        info("Type", sname)
        expls = {
            "LinearLR":           "Linear decay. Simple and effective.",
            "CosineAnnealingLR":  "Smooth cosine decay. Good for long training.",
            "OneCycleLR":         "Warmup + decay. Excellent for transformers.",
            "WarmupScheduler":    "Warmup then constant/decay. Essential for BERT-style.",
            "ReduceLROnPlateau":  "Reduces lr on plateau. Good for LSTM/GRU.",
            "StepLR":             "Step decay. Works but less smooth than cosine.",
        }
        print(f"  {expls.get(sname, sname)}")

        # Project LR for 20 steps
        subsection("Projected LR Trajectory (20 steps)")
        try:
            sc2 = copy.deepcopy(scheduler)
            lrs = []
            for _ in range(20):
                lrs.append(sc2.get_last_lr()[0]
                            if hasattr(sc2,"get_last_lr") else
                            sc2.base_lrs[0])
                sc2.step()
            maxlr = max(lrs)+1e-10
            print(f"  {'Step':>6}  {'LR':>12}  LR bar")
            print("  " + "─"*42)
            for i,lr in enumerate(lrs):
                bar = "█"*int(lr/maxlr*28)
                print(f"  {i+1:>6}  {lr:>12.2e}  {bar}")
        except:
            print("  (could not simulate lr trajectory)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    section("Loss Function")
    if criterion is None:
        print("  (No criterion provided)")
    elif HAS_TORCH:
        lname = type(criterion).__name__
        info("Type", lname)
        expls = {
            "CrossEntropyLoss":    "Standard multiclass. Applies softmax internally.",
            "BCEWithLogitsLoss":   "Binary classification. More stable than BCELoss.",
            "BCELoss":             "Binary. Expects sigmoid output.",
            "NLLLoss":             "Negative log-likelihood. Use with LogSoftmax.",
            "MSELoss":             "Regression. Less common for NLP.",
            "CTCLoss":             "Sequence-to-sequence without explicit alignment.",
            "LabelSmoothingLoss":  "Regularizes by softening hard targets.",
        }
        print(f"  {expls.get(lname, lname)}")
        if hasattr(criterion,"label_smoothing") and criterion.label_smoothing > 0:
            print(f"  ✅ Label smoothing={criterion.label_smoothing} — good NLP regularizer")
        if hasattr(criterion,"weight") and criterion.weight is not None:
            print(f"  ✅ Class weights set: {criterion.weight.tolist()}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    section("Tokenizer")
    if tokenizer is None:
        print("  (No tokenizer provided)")
    else:
        tname = type(tokenizer).__name__
        info("Type", tname)
        if hasattr(tokenizer,"vocab_size"):
            info("Vocab size", f"{tokenizer.vocab_size:,}")
        if hasattr(tokenizer,"max_length"):
            info("Max length", tokenizer.max_length)
        if hasattr(tokenizer,"pad_token_id"):
            info("Pad token ID", tokenizer.pad_token_id)
        if hasattr(tokenizer,"unk_token_id"):
            info("UNK token ID", tokenizer.unk_token_id)

        # Sample tokenizations
        samples = [
            "The movie was absolutely fantastic and I loved every moment.",
            "This film is terrible, boring, and a complete waste of time.",
            "An interesting but somewhat confusing plot with great acting.",
        ]
        subsection("Sample Tokenizations")
        for s in samples:
            try:
                ids = tokenizer.encode(s)
                toks = [tokenizer.decode([i]) for i in ids[:15]] \
                       if hasattr(tokenizer,"decode") else [str(i) for i in ids[:15]]
                trunc = "..." if len(ids) > 15 else ""
                print(f"  Input:  {s[:60]}")
                print(f"  Tokens: {' | '.join(toks)}{trunc}")
                print(f"  Length: {len(ids)} tokens\n")
            except Exception as e:
                print(f"  (tokenization failed: {e})")

    # ── Batch size & gradient accumulation ───────────────────────────────────
    section("Batch / Gradient Accumulation")
    if train_loader is not None:
        try:
            bs = train_loader.batch_size
            info("Batch size", bs)
            info("Gradient accumulation steps", grad_accumulation)
            info("Effective batch size", bs * grad_accumulation)
            n = len(train_loader.dataset)
            info("Training samples", f"{n:,}")
            info("Steps per epoch", len(train_loader))
        except: pass

        if grad_accumulation > 1:
            print(f"  ✅ Gradient accumulation × {grad_accumulation} "
                  f"simulates larger batch without extra memory")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — TRAINING LOOP TRACE
# ══════════════════════════════════════════════════════════════════════════════
class NLPTrainingTracer:
    """
    Record per-epoch metrics during training.

    Usage:
        tracer = NLPTrainingTracer()
        for epoch in range(epochs):
            ...
            tracer.record(epoch, train_loss, train_acc, val_loss, val_acc,
                          model=model, optimizer=optimizer, perplexity=ppl)
        # Pass tracer to run_nlp_pipeline
    """
    def __init__(self):
        self.history = []

    def record(self, epoch, train_loss, train_acc=None,
               val_loss=None, val_acc=None,
               model=None, optimizer=None, perplexity=None):
        rec = {
            "epoch":      epoch+1,
            "train_loss": float(train_loss),
            "train_acc":  float(train_acc) if train_acc is not None else None,
            "val_loss":   float(val_loss)  if val_loss  is not None else None,
            "val_acc":    float(val_acc)   if val_acc   is not None else None,
            "ppl":        float(perplexity) if perplexity is not None else None,
            "lr":         None, "grad_norm": None,
        }
        if optimizer and HAS_TORCH:
            rec["lr"] = optimizer.param_groups[0]["lr"]
        if model and HAS_TORCH:
            gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += p.grad.data.norm(2).item()**2
            rec["grad_norm"] = gn**0.5
        self.history.append(rec)

    def report(self):
        banner("STEP 3 — TRAINING LOOP TRACE")
        if not self.history:
            print("  No training history. Call tracer.record() each epoch.")
            return
        n = len(self.history)
        step = max(1, n//20)

        # ── Loss curves ───────────────────────────────────────────────────────
        section("Loss Curves — Train vs Validation")
        tl = [r["train_loss"] for r in self.history]
        vl = [r["val_loss"]   for r in self.history if r["val_loss"] is not None]
        has_v = len(vl) == n
        maxl  = max(max(tl), max(vl) if vl else 0) + 1e-9
        print(f"  {'Epoch':>7}  {'Train L':>10}  {'Val L':>10}  {'Gap':>9}  Bars (█=train ░=val)")
        print("  " + "─"*68)
        for r in self.history[::step]:
            ep=r["epoch"]; t=r["train_loss"]; v=r["val_loss"]
            g=(t-v) if v else 0.0
            bt = "█"*int((1-t/maxl)*14); bv = "░"*int((1-v/maxl)*14) if v else ""
            vs = f"{v:>10.4f}" if v else f"{'N/A':>10}"
            gs = f"{g:>+9.4f}" if v else f"{'N/A':>9}"
            print(f"  {ep:>7}  {t:>10.4f}  {vs}  {gs}  {bt}{bv}")

        # ── Accuracy curves ───────────────────────────────────────────────────
        ta = [r["train_acc"] for r in self.history if r["train_acc"] is not None]
        va = [r["val_acc"]   for r in self.history if r["val_acc"]   is not None]
        if ta:
            section("Accuracy Curves")
            print(f"  {'Epoch':>7}  {'Train Acc':>10}  {'Val Acc':>9}  Progress bar")
            print("  " + "─"*55)
            for r in self.history[::step]:
                t=r["train_acc"]; v=r["val_acc"]
                if t is None: continue
                bt="█"*int(t*22); bv="░"*int(v*22) if v else ""
                vs=f"{v:>9.4f}" if v else f"{'N/A':>9}"
                print(f"  {r['epoch']:>7}  {t:>10.4f}  {vs}  {bt}{bv}")

        # ── Perplexity ────────────────────────────────────────────────────────
        ppls = [r["ppl"] for r in self.history if r["ppl"] is not None]
        if ppls:
            section("Perplexity  (lower = better, random ≈ vocab_size)")
            maxp = max(ppls)+1e-9
            print(f"  {'Epoch':>7}  {'Train PPL':>10}  PPL bar")
            print("  " + "─"*48)
            for r in self.history[::step]:
                p = r["ppl"]
                if p is None: continue
                bar = "█"*int(p/maxp*30)
                print(f"  {r['epoch']:>7}  {p:>10.2f}  {bar}")

        # ── LR schedule ───────────────────────────────────────────────────────
        lrs = [r["lr"] for r in self.history if r["lr"] is not None]
        if lrs:
            section("Actual Learning Rate Schedule")
            maxlr = max(lrs)+1e-10
            print(f"  {'Epoch':>7}  {'LR':>14}  LR bar")
            print("  " + "─"*44)
            for r in self.history[::step]:
                if r["lr"] is None: continue
                bar = "█"*int(r["lr"]/maxlr*28)
                print(f"  {r['epoch']:>7}  {r['lr']:>14.2e}  {bar}")

        # ── Gradient norms ────────────────────────────────────────────────────
        gns = [r["grad_norm"] for r in self.history if r["grad_norm"] is not None]
        if gns:
            section("Gradient Norms")
            maxg = max(gns)+1e-10
            print(f"  {'Epoch':>7}  {'GradNorm':>12}  Status    Bar")
            print("  " + "─"*52)
            for r in self.history[::step]:
                gn = r["grad_norm"]
                if gn is None: continue
                st = "🔴 VANISHING" if gn<1e-4 else "🔴 EXPLODING" if gn>100 else "✅"
                bar = "█"*int(gn/maxg*24)
                print(f"  {r['epoch']:>7}  {gn:>12.4f}  {st:<12}  {bar}")

        # ── Event detection ───────────────────────────────────────────────────
        section("Training Events")
        events = []
        for i in range(1, len(self.history)):
            prev=self.history[i-1]; curr=self.history[i]; ep=curr["epoch"]
            if curr["train_loss"] > prev["train_loss"]*1.5:
                events.append(f"  🔴 Epoch {ep}: LOSS SPIKE  "
                               f"{prev['train_loss']:.4f}→{curr['train_loss']:.4f}")
            if (curr["val_loss"] and prev["val_loss"] and
                curr["val_loss"] > prev["val_loss"]*1.2 and
                curr["train_loss"] < prev["train_loss"]):
                events.append(f"  🔴 Epoch {ep}: OVERFIT — val_loss rising, train_loss falling")
            if i >= 5:
                rec = [r["train_loss"] for r in self.history[i-5:i+1]]
                if max(rec)-min(rec) < 0.001:
                    events.append(f"  🟡 Epoch {ep}: PLATEAU (5-epoch flat: ~{np.mean(rec):.4f})")
        seen = set()
        [print(e) or seen.add(e[:40]) for e in events if e[:40] not in seen] \
            if events else print("  ✅ No anomalous events detected")

        # ── Best epoch ────────────────────────────────────────────────────────
        section("Best Epoch Summary")
        bvl = min((r for r in self.history if r["val_loss"]),
                  key=lambda r: r["val_loss"], default=None)
        bva = max((r for r in self.history if r["val_acc"]),
                  key=lambda r: r["val_acc"],  default=None)
        if bvl: info("Best val loss epoch", f"{bvl['epoch']}  (loss={bvl['val_loss']:.4f})")
        if bva: info("Best val acc epoch",  f"{bva['epoch']}  (acc={bva['val_acc']:.4f})")
        if ta and va:
            gap = ta[-1] - va[-1]
            info("Final train/val acc gap", f"{gap:+.4f}")
            print(f"  {'🔴 OVERFITTING' if gap>0.15 else '🟡 mild overfit' if gap>0.05 else '✅ healthy'}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — EMBEDDING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def embedding_analysis(model, tokenizer=None, top_words=None):
    banner("STEP 4 — EMBEDDING ANALYSIS")
    if not HAS_TORCH: print("  PyTorch required"); return

    emb_layers = [(n,m) for n,m in model.named_modules()
                  if isinstance(m, nn.Embedding)]
    if not emb_layers:
        print("  No Embedding layers found"); return

    for emb_name, emb in emb_layers:
        W = emb.weight.data.cpu().numpy()  # (V, D)
        V, D = W.shape
        section(f"Embedding Layer: {emb_name}  ({V:,} tokens × {D} dims)")

        # Basic stats
        info("Vocabulary size", f"{V:,}")
        info("Embedding dimension", D)
        info("Weight mean",  f"{W.mean():.6f}")
        info("Weight std",   f"{W.std():.6f}")
        info("Weight min",   f"{W.min():.4f}")
        info("Weight max",   f"{W.max():.4f}")

        # Dead dimensions (near-zero variance across all tokens)
        dim_vars  = W.var(axis=0)
        dead_dims = int((dim_vars < 1e-6).sum())
        info("Dead dimensions (var<1e-6)", f"{dead_dims}/{D}  ({dead_dims/D*100:.1f}%)")
        if dead_dims > D*0.1:
            print(f"  🟡 {dead_dims} dead embedding dimensions — wasted capacity")
        else:
            print(f"  ✅ No significant dead dimensions")

        # Isotropy — are all directions used equally?
        # Measure: mean cosine similarity between random pairs
        subsection("Embedding Isotropy")
        print(textwrap.fill(
            "  Isotropic embeddings use all directions equally (good). "
            "Anisotropic embeddings collapse into a narrow cone (bad — "
            "common in trained transformers, hurts similarity tasks).",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "))
        np.random.seed(42)
        n_sample = min(V, 500)
        idx = np.random.choice(V, n_sample, replace=False)
        samp = W[idx]
        norms = np.linalg.norm(samp, axis=1, keepdims=True)+1e-9
        normed = samp / norms
        cos_sim = (normed @ normed.T)
        np.fill_diagonal(cos_sim, 0)
        mean_cos = float(np.abs(cos_sim).mean())
        isotropy = 1.0 - mean_cos
        bar = "█"*int(isotropy*35)
        print(f"\n  Isotropy score: {isotropy:.4f}  {bar}  (1.0 = perfectly isotropic)")
        if isotropy < 0.5:
            print(f"  🟡 Low isotropy — embeddings are anisotropic")
            print(f"     Fix: whitening, BERT-flow, or train longer")
        else:
            print(f"  ✅ Reasonable isotropy")

        # Embedding norm distribution
        subsection("Token Embedding Norm Distribution")
        norms_all = np.linalg.norm(W, axis=1)
        hist, edges = np.histogram(norms_all, bins=10)
        maxh = hist.max()+1
        for i in range(len(hist)):
            bar = "█"*int(hist[i]/maxh*30)
            print(f"  [{edges[i]:>6.2f}-{edges[i+1]:>6.2f}]  │{bar:<30}│ {hist[i]:>5}")
        info("Mean norm",   f"{norms_all.mean():.4f}")
        info("Std  norm",   f"{norms_all.std():.4f}")

        # Nearest neighbours for sample tokens
        if tokenizer is not None and top_words:
            subsection("Nearest Neighbour Tokens (cosine similarity)")
            print("  Shows which tokens the model considers semantically similar.\n")
            for word in top_words[:5]:
                try:
                    wid = tokenizer.encode(word)
                    if isinstance(wid, (list,tuple)): wid = wid[0]
                    if wid >= V: continue
                    q = W[wid]; q_norm = q/(np.linalg.norm(q)+1e-9)
                    W_norm = W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9)
                    sims = W_norm @ q_norm
                    top_idx = np.argsort(sims)[::-1][1:6]
                    neighbours = []
                    for ni in top_idx:
                        dec = tokenizer.decode([int(ni)]) \
                              if hasattr(tokenizer,"decode") else str(ni)
                        neighbours.append(f"'{dec}'({sims[ni]:.3f})")
                    print(f"  '{word:>15}' → {', '.join(neighbours)}")
                except: pass

        # 2D PCA projection — ASCII scatter plot
        subsection("2D PCA Projection of Embeddings (ASCII)")
        print("  Top 200 most frequent token embeddings projected onto 2 principal components.")
        print("  Clusters = semantically related tokens.\n")
        try:
            n_plot = min(200, V)
            # Frequency-based sampling (first n_plot IDs as proxy)
            W_plot = W[:n_plot]
            # Manual PCA (no sklearn needed)
            Wc = W_plot - W_plot.mean(axis=0)
            U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
            coords = Wc @ Vt[:2].T  # (n_plot, 2)
            cx, cy = coords[:,0], coords[:,1]
            # Normalize to grid
            GRID_W, GRID_H = 60, 18
            cx_n = (cx - cx.min())/(cx.max()-cx.min()+1e-9)
            cy_n = (cy - cy.min())/(cy.max()-cy.min()+1e-9)
            grid = [[" " for _ in range(GRID_W)] for _ in range(GRID_H)]
            for i,(x,y) in enumerate(zip(cx_n, cy_n)):
                gx = min(int(x*(GRID_W-1)), GRID_W-1)
                gy = min(int((1-y)*(GRID_H-1)), GRID_H-1)
                grid[gy][gx] = "·"
            print("  ┌" + "─"*GRID_W + "┐")
            for row in grid:
                print("  │" + "".join(row) + "│")
            print("  └" + "─"*GRID_W + "┘")
            info("PC1 variance explained", f"{S[0]**2/((S**2).sum())*100:.1f}%")
            info("PC2 variance explained", f"{S[1]**2/((S**2).sum())*100:.1f}%")
        except Exception as e:
            print(f"  (PCA failed: {e})")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — ATTENTION VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
def attention_visualizer(model, sample_batch, tokenizer=None, n_samples=2):
    banner("STEP 5 — ATTENTION VISUALIZER")
    if not HAS_TORCH:
        print("  PyTorch required"); return

    attn_layers = [(n,m) for n,m in model.named_modules()
                   if isinstance(m, nn.MultiheadAttention)]
    rnn_layers  = [(n,m) for n,m in model.named_modules()
                   if isinstance(m,(nn.LSTM,nn.GRU,nn.RNN))]

    if not attn_layers and not rnn_layers:
        section("No Attention or RNN Layers")
        print("  This model has no MultiheadAttention or RNN layers to visualize.")
        print("  (For custom attention, hook it manually and pass weights here)")
        return

    # ── Transformer: hook attention weights ───────────────────────────────────
    if attn_layers:
        section(f"Transformer Attention  ({len(attn_layers)} layer(s))")
        attn_store = {}

        def make_attn_hook(name):
            def hook(mod, inp, out):
                # out = (attn_output, attn_weights)
                if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                    attn_store[name] = out[1].detach().cpu()
            return hook

        handles = [m.register_forward_hook(make_attn_hook(n))
                   for n,m in attn_layers]

        x = sample_batch[0] if isinstance(sample_batch,(list,tuple)) else sample_batch
        if isinstance(x, torch.Tensor):
            model.eval()
            with torch.no_grad():
                try:
                    model(x[:min(2,len(x))])
                except Exception as e:
                    print(f"  (forward pass failed: {e})")
        for h in handles: h.remove()

        if not attn_store:
            print("  No attention weights captured.")
            print("  Set need_weights=True in MultiheadAttention forward call.")
            return

        for layer_name, weights in attn_store.items():
            # weights shape: (batch, heads, seq, seq) OR (batch, seq, seq)
            if weights.dim() == 3:
                weights = weights.unsqueeze(1)  # add head dim
            B, H, S, _ = weights.shape

            print(f"\n  Layer: {layer_name}  "
                  f"({H} head(s), seq_len={S})")

            # Decode tokens if possible
            x_ids = (sample_batch[0] if isinstance(sample_batch,(list,tuple))
                     else sample_batch)
            if isinstance(x_ids, torch.Tensor):
                x_ids = x_ids[0].cpu().numpy()
            if tokenizer and hasattr(tokenizer,"decode"):
                try:
                    tok_labels = [tokenizer.decode([int(i)])[:6]
                                  for i in x_ids[:S]]
                except:
                    tok_labels = [str(i) for i in range(S)]
            else:
                tok_labels = [str(i) for i in range(S)]

            # Show first sample, all heads
            w_sample = weights[0]  # (H, S, S)

            subsection("Attention Heatmaps (row = query token, col = attended token)")
            print("  Bright = high attention. Each row sums to 1.0.\n")
            chars = " ░▒▓█"
            n_show_heads = min(H, 4)
            n_show_seq   = min(S, 16)

            for h_idx in range(n_show_heads):
                aw = w_sample[h_idx, :n_show_seq, :n_show_seq].numpy()
                print(f"  Head #{h_idx+1}:")
                # Column headers
                col_hdr = "        " + "".join(f"{tok_labels[j][:4]:>5}" for j in range(n_show_seq))
                print(f"  {col_hdr}")
                for i in range(n_show_seq):
                    row_label = tok_labels[i][:5]
                    row = "".join(chars[int(aw[i,j]*(len(chars)-1))] * 2
                                  for j in range(n_show_seq))
                    max_attn = aw[i].argmax()
                    print(f"  {row_label:>5} │{row}│ → '{tok_labels[min(max_attn,len(tok_labels)-1)][:6]}'")
                print()

            # Attention entropy per head (how focused vs diffuse)
            subsection("Head Entropy Analysis")
            print("  Low entropy = head focuses sharply on few tokens (specialised)")
            print("  High entropy = head attends broadly (diffuse)")
            print()
            max_entropy = math.log(S+1e-9)
            for h_idx in range(H):
                aw = w_sample[h_idx].numpy() + 1e-9
                ent = float(-np.sum(aw * np.log(aw), axis=-1).mean())
                rel = ent / max_entropy
                bar = "█"*int(rel*25)
                spec = "diffuse" if rel > 0.7 else "focused" if rel < 0.3 else "balanced"
                print(f"  Head #{h_idx+1:<2}  entropy={ent:.3f}  "
                      f"(rel={rel:.2f})  {bar}  [{spec}]")

            # Attention sink detection (many tokens attend to position 0)
            subsection("Attention Sink Detection")
            print("  Some models route attention to a 'sink' token (e.g. [CLS] or first token)\n")
            first_tok_attn = w_sample[:, :, 0].mean(dim=-1)  # mean over query dim per head
            for h_idx in range(H):
                sink = float(first_tok_attn[h_idx])
                bar  = "█"*int(sink*30)
                flag = "🟡 SINK" if sink > 0.5 else "✅"
                print(f"  Head #{h_idx+1:<2}  sink_attn={sink:.3f}  {bar}  {flag}")

    # ── LSTM/GRU: show hidden state magnitude as proxy for "attention" ────────
    if rnn_layers and not attn_layers:
        section("RNN Hidden State Analysis")
        print(textwrap.fill(
            "  LSTMs and GRUs don't have explicit attention. "
            "We visualise the hidden state magnitudes at each timestep, "
            "which indicates where the model 'cares' most.",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "))

        x = sample_batch[0] if isinstance(sample_batch,(list,tuple)) else sample_batch
        if not isinstance(x, torch.Tensor): return

        hidden_store = {}
        def rnn_hook(name):
            def h(mod, inp, out):
                if isinstance(out, tuple):
                    hidden_store[name] = out[0].detach().cpu()  # output states
                else:
                    hidden_store[name] = out.detach().cpu()
            return h

        handles = [m.register_forward_hook(rnn_hook(n)) for n,m in rnn_layers]
        model.eval()
        with torch.no_grad():
            try: model(x[:1])
            except: pass
        for h in handles: h.remove()

        for lname, hstates in hidden_store.items():
            # hstates: (batch, seq, hidden) or (seq, batch, hidden)
            if hstates.dim() == 3:
                hs = hstates[0]  # first sample: (seq, hidden)
                T, H = hs.shape
                magnitudes = hs.abs().mean(dim=-1).numpy()  # (T,)
                chars = " ░▒▓█"
                print(f"\n  Layer: {lname}  (seq_len={T}, hidden={H})")
                print(f"  Hidden state magnitude at each timestep:\n")
                if tokenizer and hasattr(tokenizer,"decode"):
                    x_ids = x[0].cpu().numpy()
                    try:
                        tok_labels = [tokenizer.decode([int(i)])[:8]
                                      for i in x_ids[:T]]
                    except:
                        tok_labels = [str(i) for i in range(T)]
                else:
                    tok_labels = [str(i) for i in range(T)]

                max_m = magnitudes.max()+1e-9
                for t in range(min(T, 20)):
                    bar  = chars[int(magnitudes[t]/max_m*4)] * int(magnitudes[t]/max_m*30)
                    mark = " ← PEAK" if magnitudes[t] == magnitudes.max() else ""
                    print(f"  t={t:<3}  {tok_labels[t][:8]:<10}  "
                          f"|h|={magnitudes[t]:.4f}  {bar}{mark}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — TOKEN-LEVEL PREDICTION WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════
def token_prediction_walkthrough(model, val_loader, tokenizer=None,
                                  class_names=None, task="classification",
                                  n_show=5):
    banner("STEP 6 — PREDICTION WALKTHROUGH")
    if not HAS_TORCH: print("  PyTorch required"); return

    all_imgs, all_labels = [], []
    for batch in val_loader:
        if isinstance(batch,(list,tuple)) and len(batch)>=2:
            ids, lbls = batch[0], batch[1]
        else:
            print("  Need (ids, labels) batches"); return
        all_imgs.append(ids.cpu() if isinstance(ids,torch.Tensor) else torch.tensor(np.array(ids)))
        if isinstance(lbls,torch.Tensor): all_labels.append(lbls.cpu())
        else: all_labels.append(torch.tensor(np.array(lbls)))
        if sum(len(x) for x in all_imgs) >= n_show*4: break

    X = torch.cat(all_imgs)[:n_show*4]
    Y = torch.cat(all_labels)[:n_show*4]
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        out = model(X.to(device))
    logits = out.cpu()
    if logits.dim()==1: logits = logits.unsqueeze(0)
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = probs.argmax(axis=-1)
    labels = Y.numpy().astype(int)

    section(f"Sample Predictions  (showing {min(n_show,len(X))})")
    correct_count = 0
    wrong_idxs    = []

    for i in range(min(n_show, len(X))):
        print(f"\n  {'─'*62}")
        print(f"  Sample #{i+1}")

        # Decode the input text
        ids = X[i].numpy()
        ids_clean = ids[ids != 0].tolist()
        if tokenizer and hasattr(tokenizer,"decode_sequence"):
            text = tokenizer.decode_sequence(ids_clean)[:100]
        elif tokenizer and hasattr(tokenizer,"decode"):
            text = " ".join(tokenizer.decode([int(t)]) for t in ids_clean[:20])
            if len(ids_clean) > 20: text += " ..."
        else:
            text = str(ids_clean[:15])
        print(f"  Text: {text}")

        true_lbl  = labels[i]
        true_name = class_names[true_lbl] if class_names and true_lbl<len(class_names) \
                    else f"Class {true_lbl}"
        print(f"  True label: {true_name}  (idx={true_lbl})")

        p = probs[i]
        top_k = min(len(p), 5)
        top_idx = np.argsort(p)[::-1][:top_k]
        print(f"\n  Top-{top_k} predictions:")
        for rank, ci in enumerate(top_idx):
            cname = class_names[ci] if class_names and ci<len(class_names) else f"Class {ci}"
            bar   = "█"*int(p[ci]*30)
            marks = ""
            if ci == true_lbl:  marks += " ← TRUE"
            if rank == 0:       marks += " ← PREDICTED"
            print(f"    #{rank+1}  {cname:<18}  {bar:<30}  {p[ci]:.4f}{marks}")

        correct = (preds[i] == true_lbl)
        if correct: correct_count += 1
        else: wrong_idxs.append(i)
        print(f"\n  Result: {'✅ CORRECT' if correct else '❌ WRONG'}")

        top2 = np.sort(p)[::-1][:2]
        margin = top2[0]-top2[1] if len(top2)>1 else top2[0]
        conf   = "HIGH" if margin>0.5 else "MEDIUM" if margin>0.2 else "LOW"
        print(f"  Confidence margin: {margin:.4f}  {'█'*int(margin*25)}  [{conf}]")

    section("Walkthrough Summary")
    n_shown = min(n_show,len(X))
    info("Correct", f"{correct_count}/{n_shown}")
    info("Wrong",   f"{n_shown-correct_count}/{n_shown}")
    if wrong_idxs:
        print(f"\n  Wrong on samples: {wrong_idxs}")

    all_margins = np.sort(probs,axis=-1)[:,-1] - np.sort(probs,axis=-1)[:,-2]
    print(f"\n  Confidence distribution (all {len(X)} samples):")
    hist, edges = np.histogram(all_margins, bins=8)
    for i in range(len(hist)):
        bar = "█"*int(hist[i]/(hist.max()+1)*25)
        print(f"    {edges[i]:.2f}-{edges[i+1]:.2f}  │{bar:<25}│ {hist[i]:>4}")
    low_conf = int((all_margins<0.2).sum())
    print(f"\n  Low-confidence (<0.2 margin): {low_conf}/{len(X)}")
    if low_conf/len(X) > 0.3:
        print(f"  🟡 Many uncertain predictions — model may be underfitting or miscalibrated")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
def nlp_evaluation_metrics(model, val_loader, class_names=None, task="classification"):
    banner("STEP 7 — EVALUATION METRICS")
    if not HAS_TORCH: print("  PyTorch required"); return

    device = next(model.parameters()).device
    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if not (isinstance(batch,(list,tuple)) and len(batch)>=2):
                print("  No labels — skipping"); return
            ids, lbls = batch[0], batch[1]
            out   = model(ids.to(device))
            probs = torch.softmax(out, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            if isinstance(lbls, torch.Tensor): lbls = lbls.numpy()
            all_preds.append(preds); all_labels.append(np.array(lbls))
            all_probs.append(probs)

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels).astype(int)
    probs  = np.concatenate(all_probs)
    n      = len(labels)
    classes = sorted(set(labels.tolist()))
    n_cls   = len(classes)

    section("Overall Accuracy")
    acc = float(np.mean(preds==labels))
    info("Samples evaluated", n)
    info("Overall accuracy",  f"{acc:.4f}  ({acc*100:.2f}%)")
    bar = "█"*int(acc*40)
    print(f"  Accuracy:  {bar:<40}  {acc:.4f}")

    # Per-class
    section("Per-Class Metrics  (Precision / Recall / F1)")
    print(f"  {'Class':<22} {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Support':>8}  F1 bar")
    print("  "+"─"*72)
    class_metrics = []
    for ci in classes:
        cn  = class_names[ci] if class_names and ci<len(class_names) else f"Class {ci}"
        tp  = int(np.sum((preds==ci)&(labels==ci)))
        fp  = int(np.sum((preds==ci)&(labels!=ci)))
        fn  = int(np.sum((preds!=ci)&(labels==ci)))
        sup = int(np.sum(labels==ci))
        p_  = tp/(tp+fp+1e-9); r_=tp/(tp+fn+1e-9); f1=2*p_*r_/(p_+r_+1e-9)
        bar = "█"*int(f1*25)
        print(f"  {cn:<22} {p_:>8.4f}  {r_:>8.4f}  {f1:>8.4f}  {sup:>8}  {bar}")
        class_metrics.append((cn,p_,r_,f1,sup))
    macro_f1 = np.mean([m[3] for m in class_metrics])
    print(f"\n    {'Macro F1:':<35} {macro_f1:.4f}")

    # Confusion matrix
    section("Confusion Matrix")
    cm = np.zeros((n_cls,n_cls),dtype=int)
    for t,p in zip(labels,preds): cm[t,p] += 1
    if n_cls <= 12:
        w = 7
        hdr = " "*20 + "  ".join(
            (class_names[c][:5] if class_names and c<len(class_names) else f"P{c}")[:5].center(5)
            for c in classes)
        print(f"  {hdr}"); print("  "+"─"*len(hdr))
        for i,ci in enumerate(classes):
            cn = (class_names[ci][:14] if class_names and ci<len(class_names)
                  else f"Class {ci}")[:14]
            row = "  ".join(f"{cm[i,j]:>5}" for j in range(n_cls))
            flag = "✅" if cm[i,i]==cm[i].sum()>0 else "❌" if cm[i,i]==0 else "  "
            print(f"  {cn:<16} │ {row} │ {flag}")
    else:
        diag = np.diag(cm)/(cm.sum(axis=1)+1e-9)
        print("  (Large matrix — showing per-class recall)\n")
        for i,ci in enumerate(classes[:20]):
            cn = class_names[ci] if class_names and ci<len(class_names) else f"C{ci}"
            bar = "█"*int(diag[i]*25)
            print(f"  {cn:<22} recall={diag[i]:.4f}  {bar}")

    # Worst classes
    section("Worst-Performing Classes")
    worst = sorted(class_metrics, key=lambda x:x[3])[:5]
    print(f"  {'Class':<22} {'F1':>8}  {'Prec':>8}  {'Rec':>8}  {'Support':>8}")
    print("  "+"─"*60)
    for cn,p_,r_,f1,sup in worst:
        flag = "🔴" if f1<0.5 else "🟡" if f1<0.75 else "✅"
        print(f"  {flag} {cn:<20} {f1:>8.4f}  {p_:>8.4f}  {r_:>8.4f}  {sup:>8}")

    # Calibration
    section("Calibration Check")
    top_probs = probs.max(axis=1)
    bins = np.arange(0,1.1,0.1)
    print(f"  {'Bin':<12} {'Pred conf':>10}  {'Actual acc':>11}  {'N':>6}  Status")
    print("  "+"─"*52)
    for i in range(len(bins)-1):
        lo,hi = bins[i],bins[i+1]
        mask  = (top_probs>=lo)&(top_probs<hi)
        nb    = int(mask.sum())
        if nb==0: continue
        ba  = float(np.mean(preds[mask]==labels[mask]))
        bc  = float(top_probs[mask].mean())
        gap = abs(bc-ba)
        fl  = "✅" if gap<0.1 else "🟡" if gap<0.2 else "🔴"
        print(f"  [{lo:.1f}-{hi:.1f}]       {bc:>10.3f}  {ba:>11.3f}  {nb:>6}  {fl}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — OVERFITTING DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════
def nlp_overfitting_diagnosis(model, train_loader, val_loader,
                               tracer=None, class_names=None, task="classification"):
    banner("STEP 8 — OVERFITTING DIAGNOSIS ENGINE")
    if not HAS_TORCH: print("  PyTorch required"); return

    device = next(model.parameters()).device
    total_p,_ = sum(p.numel() for p in model.parameters()), None
    total_p    = sum(p.numel() for p in model.parameters())
    try:
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
    except: n_train=n_val=0

    def _acc(loader):
        c,t = 0,0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if not (isinstance(batch,(list,tuple)) and len(batch)>=2): return None
                ids,lbls = batch[0],batch[1]
                out  = model(ids.to(device))
                pred = out.argmax(dim=-1).cpu().numpy()
                if isinstance(lbls,torch.Tensor): lbls=lbls.numpy()
                c += int(np.sum(pred==np.array(lbls).astype(int)))
                t += len(lbls)
        return c/t if t>0 else None

    section("Train vs Validation Accuracy")
    ta = _acc(train_loader); va = _acc(val_loader)
    if ta is None or va is None:
        print("  Could not compute accuracy (loader missing labels)"); return

    bt = "█"*int(ta*40); bv = "░"*int(va*40)
    print(f"  Train : {ta:.4f}  {bt}")
    print(f"  Val   : {va:.4f}  {bv}")
    gap = ta - va
    print(f"  Gap   : {gap:+.4f}\n")

    if   gap > 0.15: regime = "OVERFITTING";    print("  Diagnosis: 🔴 OVERFITTING")
    elif gap > 0.05: regime = "MILD_OVERFIT";   print("  Diagnosis: 🟡 MILD OVERFITTING")
    elif ta<0.6 and va<0.6: regime="UNDERFITTING"; print("  Diagnosis: 🔴 UNDERFITTING")
    else:            regime = "HEALTHY";         print("  Diagnosis: ✅ HEALTHY")

    # Regularisation inventory
    section("Regularisation Inventory")
    drop_layers = [(n,m) for n,m in model.named_modules() if isinstance(m,nn.Dropout)]
    emb_layers  = [(n,m) for n,m in model.named_modules() if isinstance(m,nn.Embedding)]
    ln_layers   = [(n,m) for n,m in model.named_modules() if isinstance(m,nn.LayerNorm)]

    print(f"  {'Dropout layers:':<35} {len(drop_layers)}"
          + (f"  (rates: {[f'{m.p:.2f}' for _,m in drop_layers[:5]]})" if drop_layers else ""))
    print(f"  {'LayerNorm layers:':<35} {len(ln_layers)}")
    print(f"  {'Embedding layers:':<35} {len(emb_layers)}")

    if not drop_layers: print("  🟡 No Dropout — add dropout for NLP regularisation")
    else:               print("  ✅ Dropout present")
    if not ln_layers:   print("  🟡 No LayerNorm — consider adding for stability")
    else:               print("  ✅ LayerNorm present")

    # Capacity vs data
    section("Model Capacity vs Dataset Size")
    info("Total parameters",   f"{total_p:,}")
    info("Training samples",   f"{n_train:,}")
    if n_train>0:
        ratio = total_p/n_train
        info("Param/sample ratio", f"{ratio:.1f}x")
        print()
        if ratio > 1000:
            print(f"  🔴 {ratio:.0f}× more params than training samples")
            print(f"     → Use pretrained embeddings (GloVe/FastText/BERT)")
            print(f"     → Add strong dropout (≥0.5)")
            print(f"     → Reduce model size or add more data")
        elif ratio > 100:
            print(f"  🟡 High param/sample ratio ({ratio:.0f}×)")
            print(f"     → Use weight decay ≥ 0.01 and dropout ≥ 0.3")
        else:
            print(f"  ✅ Reasonable capacity/data ratio ({ratio:.1f}×)")

    # NLP-specific causes
    section("NLP-Specific Diagnosis")
    causes=[]; fixes=[]
    if regime in ("OVERFITTING","MILD_OVERFIT"):
        if not drop_layers:
            causes.append("No dropout regularisation")
            fixes.append("Add nn.Dropout(0.3-0.5) after embedding and before output")
        if n_train < 5000:
            causes.append(f"Small dataset ({n_train} samples)")
            fixes.append("Use pretrained embeddings (torchtext GloVe / HuggingFace)")
        rnn_layers = [(n,m) for n,m in model.named_modules()
                      if isinstance(m,(nn.LSTM,nn.GRU))]
        for _,m in rnn_layers:
            if m.num_layers > 2:
                causes.append(f"Deep RNN ({m.num_layers} layers) easy to overfit")
                fixes.append("Reduce RNN layers to 1-2 or add inter-layer dropout")
                break
    if regime == "UNDERFITTING":
        causes.append("Model may be too small or lr too low")
        fixes.append("Increase hidden size, add layers, or raise learning rate")

    if causes:
        print("  Likely causes:\n")
        for c in causes: print(f"    • {c}")
        print("\n  Recommended fixes:\n")
        for f in fixes:  print(f"    → {f}")
    else:
        print("  ✅ No significant fitting issues detected")

    # Training curve from tracer
    if tracer and tracer.history:
        section("Loss Gap Over Training Epochs")
        tls = [r["train_loss"] for r in tracer.history]
        vls = [r["val_loss"]   for r in tracer.history if r["val_loss"] is not None]
        if len(vls)==len(tls):
            max_g = max(abs(t-v) for t,v in zip(tls,vls))+1e-9
            step  = max(1,len(tls)//12)
            print(f"  {'Epoch':>7}  {'Train L':>10}  {'Val L':>10}  {'Gap':>10}  Gap bar")
            print("  "+"─"*55)
            for r in tracer.history[::step]:
                ep=r["epoch"]; t=r["train_loss"]; v=r["val_loss"]
                g=(t-v) if v else 0
                bar="█"*int(abs(g)/max_g*20)
                sign="↑" if g>0.01 else "↓" if g<-0.01 else "~"
                print(f"  {ep:>7}  {t:>10.4f}  {v:>10.4f}  {g:>+10.4f}  {sign}{bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — GRADIENT FLOW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def nlp_gradient_flow(model, train_batch, criterion=None):
    banner("STEP 9 — GRADIENT FLOW ANALYSIS")
    if not HAS_TORCH: print("  PyTorch required"); return

    device = next(model.parameters()).device
    if not (isinstance(train_batch,(list,tuple)) and len(train_batch)>=2):
        print("  Need (ids, labels) batch"); return

    ids, lbls = train_batch[0], train_batch[1]
    ids  = ids.to(device)
    lbls = lbls.to(device) if isinstance(lbls,torch.Tensor) else \
           torch.tensor(np.array(lbls)).to(device)

    model.train(); model.zero_grad()
    try:
        out  = model(ids)
        loss = criterion(out, lbls.long()) if criterion else \
               nn.CrossEntropyLoss()(out, lbls.long())
        loss.backward()
    except Exception as e:
        print(f"  Could not compute gradients: {e}"); model.eval(); return
    model.eval()

    section("Per-Parameter Gradient Norms")
    print(f"  {'Parameter':<42} {'||W||':>10}  {'||G||':>10}  {'G/W':>8}  Status")
    print("  "+"─"*82)
    layer_grads=[]
    for name,param in model.named_parameters():
        if param.grad is None: continue
        wn = float(param.data.norm(2))
        gn = float(param.grad.norm(2))
        rt = gn/(wn+1e-9)
        layer_grads.append((name,wn,gn,rt))

    maxg = max(g for _,_,g,_ in layer_grads)+1e-9
    for name,wn,gn,rt in layer_grads:
        st = "🔴 VANISHING" if gn<1e-7 else "🔴 EXPLODING" if gn>100 \
             else "🟡 tiny" if gn<1e-4 else "🟡 large G/W" if rt>0.1 else "✅"
        bar = "█"*int(gn/maxg*22)
        print(f"  {name[:41]:<42} {wn:>10.4f}  {gn:>10.4f}  {rt:>8.4f}  {st}  {bar}")

    # Group by component
    section("Gradient Flow by Component")
    print("  Groups parameter gradients into model components.\n")
    groups = defaultdict(float)
    for name,_,gn,_ in layer_grads:
        part = name.split(".")[0]
        groups[part] += gn
    maxg2 = max(groups.values())+1e-9
    for part, gn in sorted(groups.items(), key=lambda x:x[1], reverse=True):
        bar  = "█"*int(gn/maxg2*35)
        flag = "🔴" if gn<1e-5 else "✅"
        print(f"  {flag} {part:<30}  {gn:.4f}  {bar}")

    section("Gradient Health Diagnosis")
    vanish = [(n,gn) for n,_,gn,_ in layer_grads if gn<1e-7]
    explod = [(n,gn) for n,_,gn,_ in layer_grads if gn>100]
    tg     = sum(gn for _,_,gn,_ in layer_grads)
    mg     = tg/len(layer_grads) if layer_grads else 0
    info("Total gradient norm", f"{tg:.4f}")
    info("Mean  gradient norm", f"{mg:.4f}")
    print()
    if vanish:
        print(f"  🔴 VANISHING in {len(vanish)} param(s) — check embedding layer and deep RNNs")
        print(f"     Fix: gradient clipping, LSTM over RNN, LayerNorm, shorter sequences")
        for n,g in vanish[:3]: print(f"     {n}: {g:.2e}")
    if explod:
        print(f"  🔴 EXPLODING in {len(explod)} param(s)")
        print(f"     Fix: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
        for n,g in explod[:3]: print(f"     {n}: {g:.2e}")
    if not vanish and not explod:
        print(f"  ✅ Gradients healthy — no vanishing or exploding detected")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 — TOKEN ATTRIBUTION / SALIENCY
# ══════════════════════════════════════════════════════════════════════════════
def token_attribution(model, sample_batch, tokenizer=None,
                       class_names=None, n_samples=3):
    banner("STEP 10 — TOKEN ATTRIBUTION & SALIENCY")
    if not HAS_TORCH: print("  PyTorch required"); return

    device = next(model.parameters()).device
    if isinstance(sample_batch,(list,tuple)) and len(sample_batch)>=2:
        ids, labels = sample_batch[0], sample_batch[1]
    else:
        print("  Need (ids, labels) batch"); return

    if not isinstance(ids, torch.Tensor): ids = torch.tensor(np.array(ids))
    if not isinstance(labels,torch.Tensor): labels=torch.tensor(np.array(labels))

    # Find embedding layer
    emb_module = None
    for n,m in model.named_modules():
        if isinstance(m, nn.Embedding): emb_module=m; break
    if emb_module is None:
        print("  No Embedding layer found — cannot compute token attribution"); return

    section("What This Shows")
    print(textwrap.fill(
        "  Which input tokens most influenced the model's prediction? "
        "Computed via gradient × embedding (input attribution). "
        "High magnitude = that token had the biggest impact on this prediction.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))

    model.eval()
    for si in range(min(n_samples, len(ids))):
        x     = ids[si:si+1].to(device)      # (1, seq)
        y_true= int(labels[si].item())

        # Forward with gradient through embeddings
        emb_module.weight.requires_grad_(True)
        x_emb = emb_module(x)                # (1, seq, D)
        x_emb.retain_grad()

        # Pass embedded input through rest of model
        # We need to hook the embedding and re-route forward
        emb_out_store = {}
        def emb_hook(mod, inp, out):
            emb_out_store['emb'] = out
        hndl = emb_module.register_forward_hook(emb_hook)

        try:
            out  = model(x)
            pred = int(out.argmax(dim=-1).item())
        except Exception as e:
            hndl.remove(); print(f"  Sample {si+1}: forward failed ({e})"); continue
        hndl.remove()

        if 'emb' not in emb_out_store:
            print(f"  Sample {si+1}: could not capture embedding output"); continue

        emb_out = emb_out_store['emb']        # (1, seq, D) — has grad
        model.zero_grad()
        score = out[0, pred]
        try:
            score.backward()
        except Exception as e:
            print(f"  Sample {si+1}: backward failed ({e})"); continue

        if emb_out.grad is None:
            # Try gradient through weight directly
            if emb_module.weight.grad is not None:
                # Gather gradients for tokens in this sequence
                tok_ids = x[0].cpu().numpy()
                grads   = emb_module.weight.grad[tok_ids].abs().mean(dim=-1).numpy()
            else:
                print(f"  Sample {si+1}: no gradient captured"); continue
        else:
            grads = emb_out.grad[0].abs().mean(dim=-1).cpu().numpy()  # (seq,)

        # Decode tokens
        tok_ids   = x[0].cpu().numpy()
        real_mask = tok_ids != 0
        tok_ids_r = tok_ids[real_mask]
        grads_r   = grads[:len(real_mask)][real_mask]

        if tokenizer and hasattr(tokenizer,"decode"):
            tok_strs = [tokenizer.decode([int(t)])[:10] for t in tok_ids_r[:30]]
        else:
            tok_strs = [str(t) for t in tok_ids_r[:30]]

        pred_name = class_names[pred]   if class_names and pred<len(class_names)   else f"Class {pred}"
        true_name = class_names[y_true] if class_names and y_true<len(class_names) else f"Class {y_true}"
        correct   = "✅" if pred==y_true else "❌"

        print(f"\n  {'─'*62}")
        print(f"  Sample #{si+1}  |  Predicted: {pred_name}  "
              f"True: {true_name}  {correct}")
        print(f"  Confidence: {float(torch.softmax(out,dim=-1)[0,pred]):.4f}")
        print(f"\n  Token Attribution (gradient × embedding magnitude):\n")

        max_g = grads_r.max()+1e-9
        chars = " ░▒▓█"
        for t,(tok,g) in enumerate(zip(tok_strs[:25], grads_r[:25])):
            bar  = chars[int(g/max_g*4)] * int(g/max_g*20)
            mark = " ← PEAK" if g==grads_r.max() else ""
            print(f"  [{t:>2}] {tok:<14}  {g:.6f}  {bar}{mark}")

        # Top-5 most influential tokens
        top5 = np.argsort(grads_r)[::-1][:5]
        print(f"\n  Top-5 most influential tokens:")
        for rank,t in enumerate(top5):
            tok  = tok_strs[t] if t<len(tok_strs) else "?"
            print(f"    #{rank+1}  pos={t:<3}  '{tok:<14}'  attribution={grads_r[t]:.6f}")

        # Attribution heatmap by position category
        T = len(grads_r)
        if T >= 6:
            thirds = {"beginning (0-33%)": grads_r[:T//3].mean(),
                      "middle (33-66%)":   grads_r[T//3:2*T//3].mean(),
                      "end (66-100%)":     grads_r[2*T//3:].mean()}
            print(f"\n  Attribution by position:")
            mg = max(thirds.values())+1e-9
            for region, val in thirds.items():
                bar = "█"*int(val/mg*25)
                print(f"    {region:<22}  {val:.6f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11 — VOCABULARY & TOKENIZATION INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def vocabulary_insights(tokenizer, val_loader=None, sample_texts=None):
    banner("STEP 11 — VOCABULARY & TOKENIZATION INSIGHTS")
    if tokenizer is None:
        print("  No tokenizer provided"); return

    section("Tokenizer Properties")
    info("Type", type(tokenizer).__name__)
    if hasattr(tokenizer,"vocab_size"):    info("Vocab size",   f"{tokenizer.vocab_size:,}")
    if hasattr(tokenizer,"max_length"):    info("Max length",   tokenizer.max_length)
    if hasattr(tokenizer,"pad_token_id"):  info("PAD token ID", tokenizer.pad_token_id)
    if hasattr(tokenizer,"unk_token_id"):  info("UNK token ID", tokenizer.unk_token_id)
    if hasattr(tokenizer,"special_tokens"):info("Special tokens", tokenizer.special_tokens)

    # Token fertility = chars per token (higher = more subword splitting)
    if sample_texts:
        section("Token Fertility Analysis")
        print(textwrap.fill(
            "  Token fertility = average characters per token. "
            "Whole-word tokenizers: ~5. BPE: ~3-4. Char-level: ~1. "
            "Very low fertility means excessive fragmentation.",
            width=WIDTH, initial_indent="  ", subsequent_indent="  "))
        print()
        fertilities = []
        for text in sample_texts[:20]:
            try:
                ids = tokenizer.encode(text)
                if not isinstance(ids,(list,np.ndarray)): ids=[ids]
                fertility = len(text)/max(len(ids),1)
                fertilities.append(fertility)
            except: pass
        if fertilities:
            info("Mean fertility (chars/token)", f"{np.mean(fertilities):.2f}")
            info("Min  fertility",               f"{min(fertilities):.2f}")
            info("Max  fertility",               f"{max(fertilities):.2f}")
            if np.mean(fertilities) < 2:
                print("  🟡 Very low fertility — char-level or over-fragmented tokenizer")
            elif np.mean(fertilities) > 8:
                print("  🟡 High fertility — rare words may not be split (OOV risk)")
            else:
                print("  ✅ Fertility looks reasonable")

        section("Sample Tokenizations — Detailed")
        for text in sample_texts[:4]:
            try:
                ids = tokenizer.encode(text)
                if not isinstance(ids,(list,np.ndarray)): ids=list(ids)
                print(f"\n  Input:  \"{text[:70]}\"")
                print(f"  Length: {len(ids)} tokens")
                toks = []
                for i in ids[:20]:
                    dec = tokenizer.decode([int(i)]) \
                          if hasattr(tokenizer,"decode") else str(i)
                    toks.append(f"'{dec}'")
                trunc = " ..." if len(ids)>20 else ""
                print(f"  Tokens: {' | '.join(toks)}{trunc}")

                # UNK analysis
                unk_id  = getattr(tokenizer,"unk_token_id",1)
                n_unk   = sum(1 for i in ids if int(i)==unk_id)
                unk_pct = n_unk/len(ids)*100
                if n_unk > 0:
                    print(f"  ⚠  UNK tokens: {n_unk}/{len(ids)} ({unk_pct:.1f}%)")
                else:
                    print(f"  ✅ No UNK tokens")
            except Exception as e:
                print(f"  (tokenization failed: {e})")

    # Vocabulary frequency analysis from loader
    if val_loader is not None:
        section("Token Frequency Distribution")
        all_ids = []
        for batch in val_loader:
            ids = batch[0] if isinstance(batch,(list,tuple)) else batch
            if isinstance(ids,torch.Tensor): ids=ids.cpu().numpy()
            all_ids.append(ids.flatten())
            if len(all_ids) >= 10: break
        if all_ids:
            all_flat = np.concatenate(all_ids)
            all_flat = all_flat[all_flat!=0]
            freq = Counter(all_flat.tolist())
            total = len(all_flat)
            singletons = sum(1 for c in freq.values() if c==1)
            info("Unique tokens in val set",  f"{len(freq):,}")
            info("Singleton tokens (count=1)", f"{singletons:,}  ({singletons/len(freq)*100:.1f}%)")
            print("\n  Top 15 most frequent token IDs:")
            print(f"  {'Token ID':>10}  {'Count':>8}  {'%':>7}  Bar")
            print("  "+"─"*50)
            maxf = freq.most_common(1)[0][1]
            for tid,cnt in freq.most_common(15):
                dec = tokenizer.decode([int(tid)]) \
                      if hasattr(tokenizer,"decode") else ""
                bar = "█"*int(cnt/maxf*25)
                print(f"  {int(tid):>10}  {cnt:>8}  {cnt/total*100:>6.2f}%  {bar}  '{dec}'")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 12 — HIDDEN STATE / NEURON ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def hidden_state_analysis(model, sample_batch, tokenizer=None):
    banner("STEP 12 — HIDDEN STATE & NEURON ANALYSIS")
    if not HAS_TORCH: print("  PyTorch required"); return

    device = next(model.parameters()).device
    x = sample_batch[0] if isinstance(sample_batch,(list,tuple)) else sample_batch
    if not isinstance(x,torch.Tensor): x=torch.tensor(np.array(x))
    x = x[:min(4,len(x))].to(device)

    # Hook all leaf modules
    act_store = {}
    handles   = []
    def make_hook(n):
        def h(mod,inp,out):
            if isinstance(out,torch.Tensor):
                act_store[n] = out.detach().cpu()
            elif isinstance(out,(list,tuple)) and isinstance(out[0],torch.Tensor):
                act_store[n] = out[0].detach().cpu()
        return h
    for n,m in model.named_modules():
        if n and not list(m.children()):
            handles.append(m.register_forward_hook(make_hook(n)))
    model.eval()
    with torch.no_grad():
        try: model(x)
        except: pass
    for h in handles: h.remove()

    section("Activation Statistics — Every Layer")
    print(f"  {'Layer':<40} {'Type':<16} {'Mean':>8}  {'Std':>8}  {'Sparsity':>9}  Status")
    print("  "+"─"*90)

    issues = []
    for name,module in model.named_modules():
        if name not in act_store: continue
        act   = act_store[name].float()
        mtype = type(module).__name__
        mn    = float(act.mean())
        std   = float(act.std())
        spars = float((act.abs()<1e-5).float().mean()*100)
        mx    = float(act.max()); minn=float(act.min())

        st = "✅"
        if spars > 90:
            st="🔴 DEAD"; issues.append(("CRITICAL",f"'{name}' {spars:.0f}% dead","Check init/lr"))
        elif spars > 60:
            st="🟡 sparse"; issues.append(("WARNING",f"'{name}' {spars:.0f}% sparse","May limit capacity"))
        elif std < 1e-5:
            st="🔴 COLLAPSE"; issues.append(("CRITICAL",f"'{name}' collapsed (std≈0)","Check LayerNorm/init"))
        elif mx > 1e4:
            st="🟡 large vals"

        print(f"  {name[:39]:<40} {mtype[:15]:<16} {mn:>8.4f}  {std:>8.4f}  "
              f"{spars:>8.1f}%  {st}")

    # LSTM hidden state evolution across layers
    rnn_acts = [(n, act_store[n]) for n,m in model.named_modules()
                if isinstance(m,(nn.LSTM,nn.GRU,nn.RNN)) and n in act_store]
    if rnn_acts:
        section("RNN Hidden State Statistics Across Layers")
        print(f"  {'Layer':<40} {'Mean |h|':>10}  {'Std |h|':>10}  Magnitude bar")
        print("  "+"─"*65)
        for n,hs in rnn_acts:
            if hs.dim()==3:
                mag = hs.abs().mean()
                std_h = hs.abs().std()
                bar   = "█"*int(min(float(mag)*10,1)*25)
                print(f"  {n:<40} {float(mag):>10.4f}  {float(std_h):>10.4f}  {bar}")

    # Representation collapse check
    section("Representation Collapse Check")
    print(textwrap.fill(
        "  Checks whether the model's intermediate representations are diverse. "
        "If all hidden vectors are nearly identical (low variance across samples), "
        "the model has collapsed and is not using its capacity.",
        width=WIDTH, initial_indent="  ", subsequent_indent="  "))
    print()
    collapse_checked = 0
    for name,module in model.named_modules():
        if name not in act_store: continue
        if not isinstance(module,(nn.Linear,nn.LSTM,nn.GRU)): continue
        act = act_store[name].float()
        if act.dim() < 2: continue
        # Flatten to (samples, features)
        flat = act.reshape(act.shape[0],-1)
        if flat.shape[0] < 2: continue
        # Pairwise cosine similarity between samples
        norms = flat.norm(dim=1,keepdim=True)+1e-9
        normed= flat/norms
        cos   = float((normed@normed.T).fill_diagonal_(0).abs().mean())
        div   = 1.0-cos
        bar   = "█"*int(div*25)
        flag  = "🔴 COLLAPSED" if div<0.1 else "🟡 low" if div<0.3 else "✅"
        print(f"  {name[:38]:<38}  diversity={div:.4f}  {bar}  {flag}")
        collapse_checked += 1
        if collapse_checked >= 6: break

    if issues:
        section("Neuron Issues Found")
        for sev,msg,fix in issues:
            icon = "🔴" if sev=="CRITICAL" else "🟡"
            print(f"  {icon} {msg}")
            print(f"     Fix: {fix}")
    else:
        print("\n  ✅ All hidden states look healthy")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 13 — TRAINING DECISION TRACE
# ══════════════════════════════════════════════════════════════════════════════
def nlp_training_decision_trace(model, train_batch, optimizer=None,
                                  criterion=None, tokenizer=None):
    banner("STEP 13 — TRAINING DECISION TRACE")
    if not HAS_TORCH: print("  PyTorch required"); return

    device = next(model.parameters()).device
    if not (isinstance(train_batch,(list,tuple)) and len(train_batch)>=2):
        print("  Need (ids, labels) batch"); return

    ids, lbls = train_batch[0], train_batch[1]
    ids  = ids.to(device)
    lbls = lbls.to(device) if isinstance(lbls,torch.Tensor) else \
           torch.tensor(np.array(lbls)).to(device)

    # Per-sample loss
    section("Per-Sample Loss — Which Samples Are Hard?")
    model.eval()
    with torch.no_grad():
        out_eval = model(ids)
    crit = criterion or nn.CrossEntropyLoss(reduction="none")
    try:
        per_loss = nn.CrossEntropyLoss(reduction="none")(out_eval, lbls.long()).cpu().numpy()
    except: per_loss = np.zeros(len(ids))

    preds  = out_eval.argmax(dim=-1).cpu().numpy()
    labels = lbls.cpu().numpy().astype(int)
    n_show = min(12, len(ids))
    print(f"  {'Sample':>8}  {'Loss':>10}  {'Pred':>6}  {'True':>6}  Loss bar")
    print("  "+"─"*52)
    maxl = per_loss.max()+1e-9
    for i in range(n_show):
        bar = "█"*int(per_loss[i]/maxl*25)
        ok  = "✅" if preds[i]==labels[i] else "❌"
        # Decode short version of input
        tok_preview = ""
        if tokenizer and hasattr(tokenizer,"decode"):
            try:
                raw = ids[i].cpu().numpy()
                tok_preview = tokenizer.decode([int(raw[0])])[:6]+"..."
            except: pass
        print(f"  {i:>8}  {per_loss[i]:>10.4f}  {preds[i]:>6}  {labels[i]:>6}  {ok} {bar}  {tok_preview}")

    mean_l = per_loss.mean()
    hard   = int((per_loss > mean_l*2).sum())
    print(f"\n  Mean loss: {mean_l:.4f}  |  Hard samples (>2× mean): {hard}/{n_show}")

    # Actual weight updates
    section("Weight Update Magnitudes  (lr × ||grad||)")
    model.train(); model.zero_grad()
    out_train = model(ids)
    loss = (criterion(out_train,lbls.long()) if criterion else
            nn.CrossEntropyLoss()(out_train, lbls.long()))
    loss.backward()
    lr = optimizer.param_groups[0]["lr"] if optimizer else 1e-3
    print(f"  Loss: {loss.item():.4f}  |  LR: {lr:.2e}\n")
    print(f"  {'Parameter':<42} {'Update':>12}  {'Rel':>9}  Update bar")
    print("  "+"─"*72)
    updates=[]
    for name,param in model.named_parameters():
        if param.grad is None: continue
        um  = float(param.grad.norm(2)*lr)
        wm  = float(param.data.norm(2))
        rel = um/(wm+1e-9)
        updates.append((name,um,rel))
    maxu = max(u for _,u,_ in updates)+1e-9
    for name,um,rel in sorted(updates,key=lambda x:x[1],reverse=True)[:15]:
        bar  = "█"*int(um/maxu*28)
        flag = "🔴" if rel>0.1 else "🟡" if rel>0.01 else "✅"
        print(f"  {flag} {name[:41]:<42} {um:>12.2e}  {rel:>9.4f}  {bar}")

    # LayerNorm stats
    ln_layers = [(n,m) for n,m in model.named_modules() if isinstance(m,nn.LayerNorm)]
    if ln_layers:
        section("LayerNorm Statistics")
        print(f"  {'Layer':<38} {'γ mean':>10}  {'β mean':>10}  Status")
        print("  "+"─"*62)
        for n,m in ln_layers[:8]:
            gm = float(m.weight.mean()) if m.weight is not None else 1.0
            bm = float(m.bias.mean())   if m.bias   is not None else 0.0
            st = "✅"
            if abs(gm-1.0)>0.5: st="🟡 γ shifted"
            if abs(bm)>1.0:     st="🟡 β large"
            print(f"  {n[:37]:<38} {gm:>10.4f}  {bm:>10.4f}  {st}")

    # Summary
    section("Training Step Summary")
    total_u = sum(u for _,u,_ in updates)
    maxu_item = max(updates,key=lambda x:x[1]) if updates else None
    info("Total update magnitude", f"{total_u:.6f}")
    if maxu_item: info("Largest single update", f"{maxu_item[0]}  ({maxu_item[1]:.4f})")
    large = [(n,r) for n,_,r in updates if r>0.1]
    if large:
        print(f"\n  🟡 {len(large)} param(s) with large relative update (>10% weight):")
        for n,r in large[:3]: print(f"     {n}: {r:.4f}")
        print(f"     → Lower lr or add gradient clipping")
    else:
        print(f"\n  ✅ All updates proportionally small — stable training step")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — run_nlp_pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_nlp_pipeline(
    model,
    train_loader,
    val_loader,
    tokenizer         = None,
    class_names       = None,
    task              = "classification",
    optimizer         = None,
    scheduler         = None,
    criterion         = None,
    tracer            = None,
    grad_accumulation = 1,
    sample_batch      = None,
    sample_texts      = None,
    top_words         = None,
    run_steps         = None,    # None = all; or list e.g. [0,1,4,7,10]
):
    """
    Run the full NLP Transparency Pipeline.

    Parameters
    ----------
    model             : PyTorch nn.Module
    train_loader      : DataLoader yielding (token_ids, labels)
    val_loader        : DataLoader yielding (token_ids, labels)
    tokenizer         : object with .encode(text) and .decode([id]) methods
    class_names       : list of str, e.g. ['negative','positive']
    task              : 'classification' | 'generation' | 'seq_labelling'
    optimizer         : torch.optim.Optimizer (optional)
    scheduler         : LR scheduler (optional)
    criterion         : loss function (optional)
    tracer            : NLPTrainingTracer instance (optional)
    grad_accumulation : gradient accumulation steps (for display only)
    sample_batch      : (ids, labels) tuple override for analysis steps
    sample_texts      : list of raw strings for tokenisation examples
    top_words         : list of words for nearest-neighbour embedding analysis
    run_steps         : list of step ints to run (default: all 0-13)
                        0=health  1=arch     2=setup    3=training
                        4=embed   5=attention 6=walkthrough 7=metrics
                        8=overfit 9=gradients 10=attribution 11=vocab
                        12=hidden  13=decisions
    """
    all_steps = set(range(14)) if run_steps is None else set(run_steps)

    # grab a sample batch
    _sample = sample_batch
    if _sample is None:
        try:    _sample = next(iter(train_loader))
        except: _sample = next(iter(val_loader))

    total_p = sum(p.numel() for p in model.parameters())

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("█"*WIDTH)
    print("█"*21 + "  NLP TRANSPARENCY PIPELINE  " + "█"*30)
    print("█"*WIDTH)
    print(f"  Model       : {type(model).__name__}")
    print(f"  Task        : {task.upper()}")
    print(f"  Parameters  : {total_p:,}")
    if class_names: print(f"  Classes     : {len(class_names)}  ({', '.join(class_names[:6])})")
    if tokenizer and hasattr(tokenizer,"vocab_size"):
        print(f"  Vocab size  : {tokenizer.vocab_size:,}")
    try:
        print(f"  Train size  : {len(train_loader.dataset):,}")
        print(f"  Val size    : {len(val_loader.dataset):,}")
    except: pass
    print(f"  PyTorch     : {'available' if HAS_TORCH else 'NOT INSTALLED'}")

    step_names = {
        0:"Dataset Health Report",    1:"Architecture Visualizer",
        2:"Training Setup Explainer", 3:"Training Loop Trace",
        4:"Embedding Analysis",       5:"Attention Visualizer",
        6:"Prediction Walkthrough",   7:"Evaluation Metrics",
        8:"Overfitting Diagnosis",    9:"Gradient Flow Analysis",
        10:"Token Attribution",       11:"Vocabulary Insights",
        12:"Hidden State Analysis",   13:"Training Decision Trace",
    }

    # ── Run steps ─────────────────────────────────────────────────────────────
    if  0 in all_steps: nlp_dataset_health(train_loader, tokenizer, class_names, task)
    if  1 in all_steps: nlp_architecture_visualizer(model, tokenizer, _sample)
    if  2 in all_steps: nlp_training_setup(optimizer, scheduler, criterion,
                                            tokenizer, train_loader, grad_accumulation)
    if  3 in all_steps:
        if tracer and tracer.history: tracer.report()
        else:
            banner("STEP 3 — TRAINING LOOP TRACE")
            print("  No training history. Attach NLPTrainingTracer to your loop:")
            print("    tracer = NLPTrainingTracer()")
            print("    tracer.record(epoch, train_loss, train_acc, val_loss, val_acc,")
            print("                  model=model, optimizer=optimizer)")
    if  4 in all_steps: embedding_analysis(model, tokenizer, top_words)
    if  5 in all_steps: attention_visualizer(model, _sample, tokenizer)
    if  6 in all_steps: token_prediction_walkthrough(model, val_loader, tokenizer,
                                                       class_names, task)
    if  7 in all_steps: nlp_evaluation_metrics(model, val_loader, class_names, task)
    if  8 in all_steps: nlp_overfitting_diagnosis(model, train_loader, val_loader,
                                                    tracer, class_names, task)
    if  9 in all_steps: nlp_gradient_flow(model, _sample, criterion)
    if 10 in all_steps: token_attribution(model, _sample, tokenizer, class_names)
    if 11 in all_steps: vocabulary_insights(tokenizer, val_loader, sample_texts)
    if 12 in all_steps: hidden_state_analysis(model, _sample, tokenizer)
    if 13 in all_steps: nlp_training_decision_trace(model, _sample, optimizer,
                                                      criterion, tokenizer)

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE", "█")
    print(f"  Model      : {type(model).__name__}")
    print(f"  Parameters : {total_p:,}")
    if class_names: print(f"  Classes    : {len(class_names)}")
    print(f"\n  Steps completed:")
    for s in sorted(all_steps):
        print(f"    ✅ Step {s:>2}: {step_names.get(s,'?')}")
    return model