"""E7 — one global autoencoder instead of one per fold (reviewer item R07 and
the advisor's proposal).

The report's main pipeline trains a separate AE inside every cross-validation
fold. That gives five encoders, five error scales that have to be standardised
before they can be pooled, and the five artificial lobes in the t-SNE. The
alternative is to train a single AE once on a historical slice of the data and
then cross-validate only the classifiers on the remaining days.

Split design, which is the part that has to be right:
  * the AE trains on the EARLIEST service dates only (a fixed historical
    partition), never on any day used to score a classifier;
  * the classifiers cross-validate over the LATER days, still grouped by
    service date, so no classifier is scored on a day it trained on;
  * the two sets of days are disjoint, so nothing the AE saw leaks into a
    classifier test fold.

This is a temporally honest arrangement and is close to how the system would
actually be deployed: fit the representation on history, apply it going forward.

Outputs -> artifacts_e7/
Usage:  python exp_e7_global_ae.py   (NETMOB_SMOKE=1 for a fast pass)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from baseline_raw import scale_sequences, compute_metrics, METRIC_NAMES

RANDOM_SEED = 42
SMOKE = os.environ.get("NETMOB_SMOKE", "0") == "1"
FOLDS = 2 if SMOKE else 5
AE_EPOCHS = 3 if SMOKE else 20
LATENT_DIM, AE_HIDDEN, KERNEL, LAYERS = 64, 128, 5, 3
HIST_FRACTION = 0.4                      # earliest 40% of service dates -> AE
OUT = "./artifacts_e7"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count())


class ConvAE(nn.Module):
    """Same architecture as ML_autoencoder_cnn.ipynb, trained once globally."""

    def __init__(self, n_feat, T=64):
        super().__init__()
        self.T = T
        enc, c = [], n_feat
        for _ in range(LAYERS):
            enc += [nn.Conv1d(c, AE_HIDDEN, KERNEL, padding=KERNEL // 2),
                    nn.BatchNorm1d(AE_HIDDEN), nn.ReLU()]
            c = AE_HIDDEN
        self.enc = nn.Sequential(*enc)
        self.to_latent = nn.Conv1d(AE_HIDDEN, LATENT_DIM, 1)
        self.pos = nn.Parameter(torch.zeros(1, LATENT_DIM, T))
        dec, c = [], LATENT_DIM
        for _ in range(LAYERS):
            dec += [nn.Conv1d(c, AE_HIDDEN, KERNEL, padding=KERNEL // 2),
                    nn.BatchNorm1d(AE_HIDDEN), nn.ReLU()]
            c = AE_HIDDEN
        self.dec = nn.Sequential(*dec, nn.Conv1d(AE_HIDDEN, n_feat, 1))

    def encode(self, x, mask):
        h = self.to_latent(self.enc(x.transpose(1, 2)))       # (N, D, T)
        m = mask.unsqueeze(1)
        z = (h * m).sum(-1) / m.sum(-1).clamp(min=1)
        return z, h

    def forward(self, x, mask):
        z, _ = self.encode(x, mask)
        h = z.unsqueeze(-1).expand(-1, -1, self.T) + self.pos
        return self.dec(h).transpose(1, 2), z


def masked_mse(recon, x, mask):
    m = mask.unsqueeze(-1)
    return (((recon - x) ** 2) * m).sum() / m.sum().clamp(min=1) / x.shape[-1]


def train_global_ae(X, lens, epochs=AE_EPOCHS):
    n_feat, T = X.shape[2], X.shape[1]
    ae = ConvAE(n_feat, T).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    mask = (torch.arange(T)[None, :] < torch.as_tensor(lens)[:, None]).float()
    dl = DataLoader(TensorDataset(torch.as_tensor(X), mask), batch_size=64,
                    shuffle=True)
    ae.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, mb in dl:
            xb, mb = xb.to(device), mb.to(device)
            opt.zero_grad()
            recon, _ = ae(xb, mb)
            loss = masked_mse(recon, xb, mb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        print(f"    AE epoch {ep+1:>2}/{epochs}  masked MSE {tot/len(X):.4f}")
    return ae


@torch.no_grad()
def embed(ae, X, lens):
    T = X.shape[1]
    mask = (torch.arange(T)[None, :] < torch.as_tensor(lens)[:, None]).float()
    ae.eval()
    zs, errs = [], []
    for i in range(0, len(X), 512):
        xb = torch.as_tensor(X[i:i + 512]).to(device)
        mb = mask[i:i + 512].to(device)
        recon, z = ae(xb, mb)
        m = mb.unsqueeze(-1)
        e = (((recon - xb) ** 2) * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1) / xb.shape[-1]
        zs.append(z.cpu().numpy()); errs.append(e.cpu().numpy())
    return np.vstack(zs), np.concatenate(errs)


def main():
    z = np.load("./cache/sequences.npz", allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(int)
    lens, groups = z["lengths"].astype(int), np.asarray(z["groups"])
    ids = z["trip_instance_id"]

    days = np.array(sorted(np.unique(groups)))
    n_hist = max(2, int(round(len(days) * HIST_FRACTION)))
    hist_days, cv_days = set(days[:n_hist]), set(days[n_hist:])
    hist = np.array([g in hist_days for g in groups])
    cv = ~hist
    assert not (hist & cv).any()
    print(f"AE history : {len(hist_days)} days, {hist.sum():,} trips "
          f"({days[0]} .. {days[n_hist-1]})")
    print(f"classifier : {len(cv_days)} days, {cv.sum():,} trips "
          f"({days[n_hist]} .. {days[-1]})")

    # scale on the AE-history split only, then apply everywhere
    Xh, Xc = scale_sequences(X[hist], X[cv])
    ae = train_global_ae(Xh, lens[hist])

    Zc, err_c = embed(ae, Xc, lens[cv])
    Zh, err_h = embed(ae, Xh, lens[hist])
    yc, gc = y[cv], groups[cv]

    # one global error scale: no per-fold standardisation needed
    pd.DataFrame({"trip_instance_id": np.concatenate([ids[hist], ids[cv]]),
                  "recon_mse": np.concatenate([err_h, err_c]),
                  "split": ["history"] * hist.sum() + ["cv"] * cv.sum(),
                  "y": np.concatenate([y[hist], yc])}
                 ).to_csv(f"{OUT}/anomaly_scores.csv", index=False)

    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for k, (tr, te) in enumerate(splitter.split(Zc, yc, groups=gc)):
        pw = (yc[tr] == 0).sum() / max(1, (yc[tr] == 1).sum())
        models = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss",
                random_state=RANDOM_SEED, verbosity=0),
            "RF": RandomForestClassifier(n_estimators=300, max_depth=12,
                                         class_weight={0: 1.0, 1: pw},
                                         random_state=RANDOM_SEED, n_jobs=-1),
            "LR": LogisticRegression(C=1.0, class_weight={0: 1.0, 1: pw},
                                     solver="lbfgs", max_iter=1000,
                                     random_state=RANDOM_SEED),
        }
        for name, clf in models.items():
            clf.fit(Zc[tr], yc[tr])
            m = compute_metrics(yc[te], clf.predict_proba(Zc[te])[:, 1])
            rows.append({"model": f"GLOBAL-AE+{name}", "fold": k + 1, **m})
            print(f"  fold{k+1} GLOBAL-AE+{name:<8} ROC-AUC {m['roc_auc']:.3f} "
                  f"PR-AUC {m['pr_auc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/results.csv", index=False)
    print("\n=== global AE, mean over folds ===")
    print(df.groupby("model")[METRIC_NAMES].agg(["mean", "std"]).round(3).to_string())
    print(f"\nreconstruction error is on ONE scale "
          f"(history mean {err_h.mean():.4f}, cv mean {err_c.mean():.4f}); "
          f"no per-fold standardisation required.")
    print(f"saved -> {OUT}/")


if __name__ == "__main__":
    main()
