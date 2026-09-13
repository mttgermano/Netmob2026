"""
baseline_raw.py — R01: classifiers trained DIRECTLY on the original features,
without the autoencoder latent representation.

Same data, same StratifiedGroupKFold(5, seed=42) split by service date, and the
same XGBoost / RandomForest / LogisticRegression hyper-parameters as
ML_autoencoder_{cnn,lstm}.ipynb, so the per-fold metrics are paired with the
AE+* rows in artifacts_ae_*/results.csv (feeds the paired test asked in R08).

Two input representations replace the latent z:
  RAW-FLAT  : the scaled (64, 12) sequence flattened to 768 dims (all the
              information the AE encoder sees).
  RAW-STATS : per-channel mean/std/min/max over the REAL timesteps only
              (48 dims) — the same length-masked pooling the AE does, but
              with no learning.

Usage:  python baseline_raw.py            # full run
        NETMOB_SMOKE=1 python baseline_raw.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
import xgboost as xgb

RANDOM_SEED = 42
SMOKE       = os.environ.get("NETMOB_SMOKE", "0") == "1"
FOLDS       = 2 if SMOKE else 5
dataset_in  = os.environ.get("NETMOB_SEQ_CACHE", "./cache/sequences.npz")
artifact_dir = "./artifacts_baseline"
os.makedirs(artifact_dir, exist_ok=True)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Identical to the notebook's — f1/precision/recall are average='binary'."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_true, y_prob),
        "pr_auc"   : average_precision_score(y_true, y_prob),
    }


def scale_sequences(X_tr, X_te):
    """Per-feature StandardScaler fitted on train only (notebook's version)."""
    N_tr, T, F = X_tr.shape
    sc = StandardScaler()
    a = sc.fit_transform(X_tr.reshape(N_tr * T, F)).reshape(N_tr, T, F)
    b = sc.transform(X_te.reshape(len(X_te) * T, F)).reshape(X_te.shape)
    return a.astype(np.float32), b.astype(np.float32)


def masked_stats(X, lens):
    """Per-channel mean/std/min/max over the real (non-padded) timesteps."""
    T = X.shape[1]
    mask = (np.arange(T)[None, :] < np.asarray(lens)[:, None])[:, :, None]
    n = np.maximum(mask.sum(axis=1), 1)                       # (N, 1)
    mean = (X * mask).sum(axis=1) / n
    var = ((X - mean[:, None, :]) ** 2 * mask).sum(axis=1) / n
    big = np.where(mask, X, np.inf)
    small = np.where(mask, X, -np.inf)
    return np.hstack([mean, np.sqrt(var),
                      big.min(axis=1), small.max(axis=1)]).astype(np.float32)


def build_features(X_tr, X_te, lens_tr, lens_te):
    X_tr, X_te = scale_sequences(X_tr, X_te)
    return {
        "RAW-FLAT":  (X_tr.reshape(len(X_tr), -1), X_te.reshape(len(X_te), -1)),
        "RAW-STATS": (masked_stats(X_tr, lens_tr), masked_stats(X_te, lens_te)),
    }


def classifiers(pos_weight):
    cw = {0: 1.0, 1: pos_weight}
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0),
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight=cw,
            random_state=RANDOM_SEED, n_jobs=-1),
        "LR": LogisticRegression(
            C=1.0, class_weight=cw, solver="lbfgs", max_iter=1000,
            random_state=RANDOM_SEED),
    }


def run():
    z = np.load(dataset_in, allow_pickle=True)
    X, y, lens, groups = (z["X"].astype(np.float32), z["y"].astype(int),
                          z["lengths"].astype(int), z["groups"])
    if SMOKE:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(y), 3000, False)
        X, y, lens, groups = X[idx], y[idx], lens[idx], groups[idx]
    print(f"N={len(y):,}  delayed={y.mean():.1%}  folds={FOLDS}  smoke={SMOKE}")

    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True,
                                    random_state=RANDOM_SEED)
    rows = []
    for k, (tr, te) in enumerate(
            splitter.split(X.reshape(len(X), -1), y, groups=np.asarray(groups))):
        pos_weight = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        feats = build_features(X[tr], X[te], lens[tr], lens[te])
        for rep, (F_tr, F_te) in feats.items():
            for name, clf in classifiers(pos_weight).items():
                clf.fit(F_tr, y[tr])
                m = compute_metrics(y[te], clf.predict_proba(F_te)[:, 1])
                rows.append({"model": f"{rep}+{name}", "fold": k + 1, **m})
                print(f"  fold{k+1} {rep}+{name:<8} "
                      f"F1 {m['f1']:.3f}  ROC-AUC {m['roc_auc']:.3f}")

    df = pd.DataFrame(rows)
    means = (df.groupby("model")[METRIC_NAMES].agg(["mean", "std"]))
    out = []
    for model, g in df.groupby("model", sort=False):
        out.append(g)
        row = {"model": model, "fold": "mean"}
        row.update({m: g[m].mean() for m in METRIC_NAMES})
        row.update({f"{m}_std": g[m].std(ddof=0) for m in METRIC_NAMES})
        out.append(pd.DataFrame([row]))
    summary = pd.concat(out, ignore_index=True)
    path = f"{artifact_dir}/results{'_smoke' if SMOKE else ''}.csv"
    summary.to_csv(path, index=False)
    print(f"\n{means.round(4)}\n\nsaved -> {path}")
    return summary


def _selftest():
    """masked_stats must ignore padding: constant signal + zero padding."""
    X = np.zeros((2, 4, 1), np.float32)
    X[0, :3, 0] = [1.0, 1.0, 1.0]        # len 3, padded tail
    X[1, :, 0] = [2.0, 4.0, 6.0, 8.0]    # len 4, no padding
    s = masked_stats(X, [3, 4])
    mean, std, mn, mx = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    assert np.allclose(mean, [1.0, 5.0]), mean
    assert np.allclose(std, [0.0, np.std([2, 4, 6, 8])]), std
    assert np.allclose(mn, [1.0, 2.0]) and np.allclose(mx, [1.0, 8.0])
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
    run()
