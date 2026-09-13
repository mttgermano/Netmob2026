"""
contrastive_simclr.py — SimCLR contrastive representation as an alternative to
the autoencoder latent (follow-up to R01).

R01 (baseline_raw.py) showed the AE latent buys nothing over the original
features for trees, and only linear separability for LR. Contrastive learning
optimises exactly that property directly, instead of getting it as a by-product
of reconstruction. This script swaps the pretext task and holds everything else
fixed, so the numbers are paired per fold with:
    artifacts_ae_cnn/results.csv, artifacts_ae_lstm/results.csv  (AE latent)
    artifacts_baseline/results.csv                               (no latent)

Held fixed: StratifiedGroupKFold(5, shuffle, seed=42) by service date, the
ConvEncoder architecture (copied verbatim from ML_autoencoder_cnn.ipynb), the
MLP/XGBoost/RF/LR hyper-parameters, compute_metrics and the results.csv schema.
Changed: decoder -> projection head, MSE -> NT-Xent on two augmented views.

Phase A: delay classification on the frozen embedding.
Phase B: out-of-fold anomaly score = mean cosine distance to the k nearest
         training-split neighbours, compared (Pearson + Spearman, R17) against
         the AE reconstruction-error scores.

Usage:  python contrastive_simclr.py               # full run (~2h, RTX 3050)
        NETMOB_SMOKE=1 python contrastive_simclr.py
        NETMOB_SHORTCUT_GATE=1 python contrastive_simclr.py   # gate only
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from baseline_raw import (METRIC_NAMES, classifiers, compute_metrics,
                          scale_sequences)

# ── config (mirrors ML_autoencoder_cnn.ipynb) ──────────────────────────────
FEATURE_COLS = ["speed_mps", "accel", "heading_change", "progress_frac",
                "progress_rate", "cross_track_m", "dwell_frac",
                "tod_sin", "tod_cos", "is_weekend", "rain_mm", "headway_min"]
# channels that are constant (or near-constant) within a trip: the shortcut risk
STATIC_CH  = [FEATURE_COLS.index(c) for c in
              ["tod_sin", "tod_cos", "is_weekend", "rain_mm", "headway_min"]]
DYNAMIC_CH = [i for i in range(len(FEATURE_COLS)) if i not in STATIC_CH]

RANDOM_SEED = 42
SMOKE = os.environ.get("NETMOB_SMOKE", "0") == "1"
GATE_ONLY = os.environ.get("NETMOB_SHORTCUT_GATE", "0") == "1"

T_MAX      = 64
LATENT_DIM = 64
AE_HIDDEN  = 128
CNN_KERNEL_SIZE, CNN_NUM_LAYERS = 5, 3

SIMCLR_EPOCHS = 3 if SMOKE else 30
SIMCLR_BATCH  = 256
SIMCLR_LR     = 1e-3
TEMPERATURE   = 0.1
PROJ_DIM      = 64

CLS_EPOCHS = 3 if SMOKE else 12
CLS_BATCH  = 128
FOLDS      = 2 if SMOKE else 5
KNN_K      = 20

dataset_in   = "./cache/sequences.npz"
artifact_dir = "./artifacts_simclr"
os.makedirs(artifact_dir, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = ["SimCLR+MLP", "SimCLR+XGBoost", "SimCLR+RF", "SimCLR+LR"]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"{artifact_dir}/progress.log", "a") as fh:
        fh.write(line + "\n")


# ── encoder: verbatim from ML_autoencoder_cnn.ipynb (cell 7) ───────────────
class ConvEncoder(nn.Module):
    """(N, T, F) -> z_seq (N, T, LATENT_DIM) and length-masked pooled z."""

    def __init__(self, input_size, hidden_size=AE_HIDDEN, latent_dim=LATENT_DIM,
                 num_layers=CNN_NUM_LAYERS, kernel_size=CNN_KERNEL_SIZE,
                 dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        blocks, in_ch = [], input_size
        for _ in range(num_layers):
            blocks += [nn.Conv1d(in_ch, hidden_size, kernel_size, padding=pad),
                       nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = hidden_size
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(hidden_size, latent_dim, kernel_size=1)

    def forward(self, x, lengths=None):
        h = self.conv(x.transpose(1, 2))
        z_seq = self.proj(h).transpose(1, 2)
        if lengths is not None:
            t_idx = torch.arange(z_seq.size(1), device=z_seq.device).unsqueeze(0)
            mask = (t_idx < lengths.unsqueeze(1).to(z_seq.device)).float().unsqueeze(2)
            z = (z_seq * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            z = z_seq.mean(dim=1)
        return z, z_seq


class SimCLRNet(nn.Module):
    """Encoder + projection head. The head exists only for the NT-Xent loss and
    is discarded before the downstream classifiers (standard SimCLR)."""

    def __init__(self, input_size):
        super().__init__()
        self.encoder = ConvEncoder(input_size)
        self.head = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(),
                                  nn.Linear(128, PROJ_DIM))

    def forward(self, x, lengths=None):
        z, _ = self.encoder(x, lengths)
        return z, self.head(z)


# ── augmentations ──────────────────────────────────────────────────────────
def _crop_resize(x, lens, gen, min_frac=0.5):
    """Random crop over the REAL timesteps, linearly resized back to T.
    Views are therefore full-length (no padding), which is why augmented
    batches are encoded with lengths=T while extraction uses the true lengths.
    """
    N, T, F = x.shape
    lens_f = lens.clamp(min=2).float()
    frac = min_frac + (1.0 - min_frac) * torch.rand(N, device=x.device, generator=gen)
    span = (lens_f * frac).clamp(min=2.0)
    start = torch.rand(N, device=x.device, generator=gen) * (lens_f - span)
    grid = torch.linspace(0, 1, T, device=x.device).unsqueeze(0)
    pos = start.unsqueeze(1) + grid * (span - 1).unsqueeze(1)   # (N, T) floats
    lo = pos.floor().long().clamp(0, T - 1)
    hi = (lo + 1).clamp(max=T - 1)
    w = (pos - lo.float()).unsqueeze(2)
    g_lo = torch.gather(x, 1, lo.unsqueeze(2).expand(-1, -1, F))
    g_hi = torch.gather(x, 1, hi.unsqueeze(2).expand(-1, -1, F))
    return g_lo * (1 - w) + g_hi * w


def augment(x, lens, gen, jitter=0.1, static_jitter=0.3, p_drop=0.1,
            mask_frac=(0.1, 0.2)):
    """One SimCLR view. static_jitter is the anti-shortcut measure: the five
    trip-constant metadata channels would otherwise be an exact per-trip
    fingerprint that NT-Xent can memorise instead of learning motion."""
    N, T, F = x.shape
    v = _crop_resize(x, lens, gen)
    noise = torch.randn(v.shape, device=v.device, generator=gen)
    scale = torch.full((F,), jitter, device=v.device)
    scale[STATIC_CH] = static_jitter
    v = v + noise * scale
    keep = (torch.rand(N, 1, F, device=v.device, generator=gen) >= p_drop).float()
    v = v * keep
    span = int(T * (mask_frac[0] + (mask_frac[1] - mask_frac[0]) * float(
        torch.rand(1, device=v.device, generator=gen))))
    if span > 0:
        s0 = torch.randint(0, max(1, T - span), (N,), device=v.device, generator=gen)
        t_idx = torch.arange(T, device=v.device).unsqueeze(0)
        m = ((t_idx >= s0.unsqueeze(1)) & (t_idx < (s0 + span).unsqueeze(1)))
        v = v * (~m).float().unsqueeze(2)
    return v


def nt_xent(p1, p2, temperature=TEMPERATURE):
    """NT-Xent over 2N projections; positives are the (i, i+N) pairs."""
    p = Fn.normalize(torch.cat([p1, p2], dim=0), dim=1)
    n = p1.size(0)
    sim = p @ p.t() / temperature
    sim.fill_diagonal_(-torch.inf)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(p.device)
    return Fn.cross_entropy(sim, target)


# ── pretraining ────────────────────────────────────────────────────────────
def pretrain_simclr(X_tr, lens_tr, epochs=SIMCLR_EPOCHS, batch_size=SIMCLR_BATCH):
    """Self-supervised: labels are never touched."""
    model = SimCLRNet(X_tr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=SIMCLR_LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=device).manual_seed(RANDOM_SEED)
    ds = TensorDataset(torch.tensor(X_tr).to(device),
                       torch.tensor(lens_tr, dtype=torch.long).to(device))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    full_len = torch.full((batch_size,), T_MAX, dtype=torch.long, device=device)
    losses = []
    for epoch in range(epochs):
        model.train()
        ep, nb = 0.0, 0
        for xb, lb in loader:
            opt.zero_grad()
            v1, v2 = augment(xb, lb, gen), augment(xb, lb, gen)
            _, q1 = model(v1, full_len[:len(xb)])
            _, q2 = model(v2, full_len[:len(xb)])
            loss = nt_xent(q1, q2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep += loss.item(); nb += 1
        sched.step()
        losses.append(ep / max(1, nb))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            chance = float(np.log(2 * batch_size - 1))
            log(f"    SimCLR epoch {epoch+1:3d}/{epochs} | NT-Xent "
                f"{losses[-1]:.4f} (chance floor {chance:.3f})")
    return model, losses


def extract_z(model, X, lens, batch_size=256):
    """Frozen encoder over the ORIGINAL (padded) sequences with true lengths —
    identical to extract_latent() in the notebook."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size]).to(device)
            lb = torch.tensor(lens[i:i + batch_size], dtype=torch.long).to(device)
            z, _ = model.encoder(xb, lb)
            out.append(z.cpu().numpy())
    return np.vstack(out)


# ── downstream MLP (same architecture/schedule as the notebook) ────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=LATENT_DIM, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp_on_z(z_tr, y_tr, z_va, y_va, epochs=CLS_EPOCHS, pos_weight_val=1.0):
    model = MLPClassifier().to(device)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       patience=5, factor=0.5)
    z_tr_t = torch.tensor(z_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    z_va_t = torch.tensor(z_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(z_tr_t, y_tr_t), batch_size=CLS_BATCH,
                        shuffle=True)
    best_val, best_wts = float("inf"), None
    for _ in range(epochs):
        model.train()
        for zb, yb in loader:
            opt.zero_grad()
            loss = crit(model(zb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(z_va_t), y_va_t).item()
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_wts = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_wts)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(z_va_t)).cpu().numpy()


# ── shortcut gate ──────────────────────────────────────────────────────────
def shortcut_gate(z_tr, z_te, X_tr_raw, X_te_raw, lens_tr, lens_te):
    """Can the frozen embedding recover the trip-constant metadata channels?
    Near-perfect recovery means NT-Xent memorised the per-trip fingerprint
    instead of learning motion. Reported either way."""
    def const_ch(X, lens, name):
        i = FEATURE_COLS.index(name)
        return X[np.arange(len(X)), np.maximum(lens - 1, 0), i]

    wk_tr, wk_te = (const_ch(X, l, "is_weekend")
                    for X, l in ((X_tr_raw, lens_tr), (X_te_raw, lens_te)))
    hw_tr, hw_te = (const_ch(X, l, "headway_min")
                    for X, l in ((X_tr_raw, lens_tr), (X_te_raw, lens_te)))
    out = {}
    if len(np.unique(wk_tr)) > 1 and len(np.unique(wk_te)) > 1:
        clf = LogisticRegression(max_iter=1000).fit(z_tr, wk_tr > 0.5)
        out["is_weekend_auc"] = roc_auc_score(wk_te > 0.5,
                                              clf.predict_proba(z_te)[:, 1])
    out["headway_r2"] = r2_score(hw_te, LinearRegression().fit(z_tr, hw_tr).predict(z_te))
    log(f"  [shortcut gate] {out}")
    return out


# ── Phase A ────────────────────────────────────────────────────────────────
def run_cv(X, y, trip_ids, lens, groups, n_folds=FOLDS):
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=RANDOM_SEED)
    results = {m: [] for m in MODEL_NAMES}
    latents, curves, gates = {}, [], []
    log(f"CV start — {len(np.unique(groups))} service days, {n_folds} folds, N={len(X)}")

    for k, (tr, te) in enumerate(splitter.split(X.reshape(len(X), -1), y,
                                                groups=np.asarray(groups))):
        ckpt = f"{artifact_dir}/cv_fold{k+1}.pkl"
        if os.path.exists(ckpt):
            with open(ckpt, "rb") as fh:
                fold = pickle.load(fh)
            log(f"FOLD {k+1}/{n_folds} restored from checkpoint")
        else:
            log(f"FOLD {k+1}/{n_folds} start | test days "
                f"{sorted(set(np.asarray(groups)[te]))}")
            fold = _run_fold(X, y, trip_ids, lens, tr, te)
            with open(ckpt, "wb") as fh:
                pickle.dump(fold, fh)
            log(f"FOLD {k+1}/{n_folds} done -> {ckpt}")
        for m in MODEL_NAMES:
            results[m].append(fold["metrics"][m])
        latents[k] = fold["latents"]
        curves.append(fold["pretrain_loss"])
        gates.append(fold["gate"])
    log("CV complete")
    return results, latents, curves, gates


def _run_fold(X, y, trip_ids, lens, tr, te):
    X_tr, X_te = scale_sequences(X[tr], X[te])
    y_tr, y_te = y[tr], y[te]
    lens_tr, lens_te = lens[tr], lens[te]
    pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    log("  [Phase 1] SimCLR pretraining (no labels)...")
    model, losses = pretrain_simclr(X_tr, lens_tr)

    z_tr = extract_z(model, X_tr, lens_tr)
    z_te = extract_z(model, X_te, lens_te)
    gate = shortcut_gate(z_tr, z_te, X[tr], X[te], lens_tr, lens_te)

    metrics = {}
    log("  training SimCLR+MLP...")
    metrics["SimCLR+MLP"] = compute_metrics(
        y_te, train_mlp_on_z(z_tr, y_tr, z_te, y_te, pos_weight_val=pos_weight))
    for name, clf in classifiers(pos_weight).items():
        log(f"  training SimCLR+{name}...")
        clf.fit(z_tr, y_tr)
        metrics[f"SimCLR+{name}"] = compute_metrics(
            y_te, clf.predict_proba(z_te)[:, 1])
    for m in MODEL_NAMES:
        log(f"    {m:<15} F1 {metrics[m]['f1']:.3f} "
            f"ROC-AUC {metrics[m]['roc_auc']:.3f}")

    return {"metrics": metrics, "pretrain_loss": losses, "gate": gate,
            "latents": {"z_train": z_tr, "y_train": y_tr, "z_test": z_te,
                        "y_test": y_te, "trip_ids_test": trip_ids[te]}}


# ── Phase B: anomaly scores from the embedding ─────────────────────────────
def knn_anomaly_scores(latents, k=KNN_K):
    """Out-of-fold: mean cosine distance to the k nearest TRAIN-split neighbours
    in the frozen embedding of the fold where the trip was held out."""
    rows = []
    for fold, d in latents.items():
        a = torch.tensor(d["z_test"], dtype=torch.float32, device=device)
        b = torch.tensor(d["z_train"], dtype=torch.float32, device=device)
        a, b = Fn.normalize(a, dim=1), Fn.normalize(b, dim=1)
        parts = []
        for i in range(0, len(a), 512):                    # chunked: N_train ~18k
            sim = a[i:i + 512] @ b.t()
            parts.append((1.0 - sim.topk(k, dim=1).values).mean(1).cpu().numpy())
        score = np.concatenate(parts)
        rows += [{"trip_instance_id": t, "knn_dist": float(s), "y": int(yy)}
                 for t, s, yy in zip(d["trip_ids_test"], score, d["y_test"])]
    return pd.DataFrame(rows)


def compare_anomaly(df):
    """Pearson AND Spearman vs both AE reconstruction-error series (R17)."""
    for ae in ("artifacts_ae_cnn", "artifacts_ae_lstm"):
        path = f"{ae}/anomaly_scores.csv"
        if not os.path.exists(path):
            continue
        m = df.merge(pd.read_csv(path), on="trip_instance_id", suffixes=("", "_ae"))
        if len(m) < 10:
            continue
        r, pr = pearsonr(m["knn_dist"], m["recon_mse"])
        rho, ps = spearmanr(m["knn_dist"], m["recon_mse"])
        log(f"  vs {ae}: n={len(m):,} Pearson r={r:.3f} (p={pr:.1e})  "
            f"Spearman rho={rho:.3f} (p={ps:.1e})")


# ── reporting ──────────────────────────────────────────────────────────────
def summarize(results):
    out = []
    for model, folds in results.items():
        for i, m in enumerate(folds):
            out.append({"model": model, "fold": i + 1, **m})
        out.append({"model": model, "fold": "mean",
                    **{m: float(np.mean([f[m] for f in folds])) for m in METRIC_NAMES},
                    **{f"{m}_std": float(np.std([f[m] for f in folds]))
                       for m in METRIC_NAMES}})
    return pd.DataFrame(out)


def compare_to_baselines(df):
    """Per-fold paired deltas vs the AE latent and vs the raw features."""
    sim = df[df.fold != "mean"]
    others = {"AE-cnn": ("artifacts_ae_cnn/results.csv", "AE+{}"),
              "AE-lstm": ("artifacts_ae_lstm/results.csv", "AE+{}"),
              "RAW-FLAT": ("artifacts_baseline/results.csv", "RAW-FLAT+{}"),
              "RAW-STATS": ("artifacts_baseline/results.csv", "RAW-STATS+{}")}
    print("\n===== SimCLR vs alternatives (per-fold paired, mean +/- std) =====")
    for clf in ["XGBoost", "RF", "LR", "MLP"]:
        a = sim[sim.model == f"SimCLR+{clf}"].sort_values("fold")
        if a.empty:
            continue
        for tag, (path, pat) in others.items():
            if not os.path.exists(path):
                continue
            o = pd.read_csv(path)
            o = o[(o.fold != "mean") & (o.model == pat.format(clf))]
            if o.empty or len(o) != len(a):
                continue
            o = o.sort_values("fold")
            for m in ["f1", "roc_auc", "pr_auc"]:
                d = a[m].values - o[m].astype(float).values
                print(f"  {clf:<8} {m:<8} SimCLR {a[m].mean():.3f}  "
                      f"{tag:<10} {o[m].astype(float).mean():.3f}   "
                      f"delta {d.mean():+.3f} +/- {d.std(ddof=0):.3f}"
                      f"{'  [consistent]' if np.all(np.sign(d) == np.sign(d[0])) else ''}")
        print()


# ── self-tests ─────────────────────────────────────────────────────────────
def _selftest():
    g = torch.Generator(device="cpu").manual_seed(0)

    # NT-Xent: identical positives + orthogonal negatives -> ~0 loss;
    # all-orthogonal (no signal) -> the log(2N-1) chance floor.
    n, d = 8, 64
    e = torch.eye(n, d)
    assert nt_xent(e, e.clone(), temperature=0.1).item() < 1e-3
    rnd = torch.eye(2 * n, d)
    chance = nt_xent(rnd[:n], rnd[n:], temperature=1.0).item()
    assert abs(chance - float(np.log(2 * n - 1))) < 0.5, chance

    # crop_resize samples only real timesteps: a sequence that is 1.0 up to len
    # and 0.0 in the padding must come back with no zeros.
    x = torch.zeros(4, T_MAX, 1)
    lens = torch.tensor([10, 20, 40, 64])
    for i, L in enumerate(lens):
        x[i, :L, 0] = 1.0
    v = _crop_resize(x, lens, g)
    assert v.min().item() > 0.99, v.min().item()

    # augment keeps shape and actually perturbs the static channels
    xb = torch.randn(4, T_MAX, len(FEATURE_COLS))
    v1, v2 = augment(xb, lens, g), augment(xb, lens, g)
    assert v1.shape == xb.shape
    assert not torch.allclose(v1[:, :, STATIC_CH], v2[:, :, STATIC_CH])
    print("selftest ok")


def _assert_folds_match(X, y, groups, trip_ids):
    """The comparison is only legitimate if these are the AE's exact folds."""
    ref = "artifacts_ae_cnn/latent_vectors.npz"
    if SMOKE or not os.path.exists(ref):
        return
    z = np.load(ref, allow_pickle=True)
    sp = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for k, (_, te) in enumerate(sp.split(X.reshape(len(X), -1), y,
                                         groups=np.asarray(groups))):
        key = f"fold{k}_trip_ids_test"
        assert np.array_equal(trip_ids[te], z[key]), f"fold {k+1} split drifted"
    print(f"fold split matches {ref} (all {FOLDS} folds)")


def main():
    _selftest()
    npz = np.load(dataset_in, allow_pickle=True)
    X, y = npz["X"].astype(np.float32), npz["y"].astype(int)
    lens, trip_ids, groups = (npz["lengths"].astype(int), npz["trip_instance_id"],
                              npz["groups"])
    _assert_folds_match(X, y, groups, trip_ids)
    if SMOKE:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(y), 3000, False)
        X, y, lens, trip_ids, groups = (X[idx], y[idx], lens[idx],
                                        trip_ids[idx], groups[idx])
    log(f"device={device} N={len(y):,} delayed={y.mean():.1%} folds={FOLDS} smoke={SMOKE}")

    if GATE_ONLY:
        sp = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_SEED)
        tr, te = next(sp.split(X.reshape(len(X), -1), y, groups=np.asarray(groups)))
        X_tr, X_te = scale_sequences(X[tr], X[te])
        model, _ = pretrain_simclr(X_tr, lens[tr])
        shortcut_gate(extract_z(model, X_tr, lens[tr]),
                      extract_z(model, X_te, lens[te]),
                      X[tr], X[te], lens[tr], lens[te])
        return

    results, latents, curves, gates = run_cv(X, y, trip_ids, lens, groups)

    suffix = "_smoke" if SMOKE else ""
    summary = summarize(results)
    summary.to_csv(f"{artifact_dir}/results{suffix}.csv", index=False)
    pd.DataFrame(curves).T.to_csv(f"{artifact_dir}/pretrain_curves{suffix}.csv",
                                  index_label="epoch")
    pd.DataFrame(gates).to_csv(f"{artifact_dir}/shortcut_gate{suffix}.csv",
                               index_label="fold")
    np.savez_compressed(f"{artifact_dir}/latent_vectors{suffix}.npz",
                        **{f"fold{k}_{key}": v for k, d in latents.items()
                           for key, v in d.items()})

    print("\n" + summary[summary.fold == "mean"][["model"] + METRIC_NAMES]
          .to_string(index=False))
    compare_to_baselines(summary)

    log("[Phase B] kNN anomaly scores from the contrastive embedding")
    anom = knn_anomaly_scores(latents)
    anom.to_csv(f"{artifact_dir}/anomaly_scores{suffix}.csv", index=False)
    compare_anomaly(anom)
    log(f"artifacts -> {artifact_dir}/")


if __name__ == "__main__":
    main()
