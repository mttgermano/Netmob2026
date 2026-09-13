"""E3 — which factors are associated with delay (reviewer items R15, R16).

Three analyses, all on the same day-grouped folds as every other experiment so
the numbers stay comparable with artifacts_ae_*/results.csv:

  1. Permutation importance and SHAP over RAW-STATS+XGBoost, the best delay
     model in the report. RAW-STATS is 48 features: mean/std/min/max of each of
     the 12 input channels over the real (non-padded) timesteps, so every
     feature has a direct physical reading.
  2. A logistic-regression baseline on the same features with coefficients,
     standard errors and Wald p-values, which is what R15 explicitly asked for.
  3. Latent-dimension probing (R16): correlation of each of the 64 CNN-AE latent
     dimensions with the delay label and with the input features, plus a
     single-dimension linear probe, to see whether any axis is individually
     interpretable.

Outputs -> artifacts_e3/
Usage:  python exp_e3_attribution.py   (NETMOB_SMOKE=1 for a fast pass)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from scipy import stats
import xgboost as xgb
import shap

from baseline_raw import masked_stats, scale_sequences, compute_metrics

RANDOM_SEED = 42
SMOKE = os.environ.get("NETMOB_SMOKE", "0") == "1"
FOLDS = 2 if SMOKE else 5
OUT = "./artifacts_e3"
os.makedirs(OUT, exist_ok=True)

CHANNELS = ["speed_mps", "accel", "heading_change", "progress_frac",
            "progress_rate", "cross_track_m", "dwell_frac",
            "tod_sin", "tod_cos", "is_weekend", "rain_mm", "headway_min"]
STATS = ["mean", "std", "min", "max"]
FEATNAMES = [f"{s}({c})" for s in STATS for c in CHANNELS]


def main():
    z = np.load("./cache/sequences.npz", allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(int)
    lens, groups = z["lengths"].astype(int), np.asarray(z["groups"])
    if SMOKE:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(y), 3000, False)
        X, y, lens, groups = X[idx], y[idx], lens[idx], groups[idx]
    assert len(FEATNAMES) == 4 * X.shape[2], (len(FEATNAMES), X.shape)
    print(f"N={len(y):,}  delayed={y.mean():.1%}  folds={FOLDS}")

    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=RANDOM_SEED)
    perm_rows, shap_rows, coef_rows = [], [], []

    for k, (tr, te) in enumerate(splitter.split(X.reshape(len(X), -1), y, groups=groups)):
        Xtr, Xte = scale_sequences(X[tr], X[te])
        Ftr = masked_stats(Xtr, lens[tr])
        Fte = masked_stats(Xte, lens[te])
        pw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())

        # ---- XGBoost: permutation importance + SHAP ----------------------
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=pw, eval_metric="logloss",
                                random_state=RANDOM_SEED, verbosity=0)
        clf.fit(Ftr, y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(Fte)[:, 1])

        pi = permutation_importance(clf, Fte, y[te], n_repeats=10,
                                    random_state=RANDOM_SEED, n_jobs=-1,
                                    scoring="roc_auc")
        for name, m in zip(FEATNAMES, pi.importances_mean):
            perm_rows.append({"fold": k + 1, "feature": name, "drop_roc_auc": m})

        sv = shap.TreeExplainer(clf).shap_values(Fte)
        for name, m in zip(FEATNAMES, np.abs(sv).mean(axis=0)):
            shap_rows.append({"fold": k + 1, "feature": name, "mean_abs_shap": m})

        # ---- logistic regression with Wald tests -------------------------
        sc = StandardScaler().fit(Ftr)
        Ztr, Zte = sc.transform(Ftr), sc.transform(Fte)
        lr = LogisticRegression(C=1.0, class_weight={0: 1.0, 1: pw},
                                solver="lbfgs", max_iter=2000,
                                random_state=RANDOM_SEED).fit(Ztr, y[tr])
        # Wald SEs from the observed information matrix of the fitted model
        p = lr.predict_proba(Ztr)[:, 1]
        W = p * (1 - p)
        Zi = np.hstack([np.ones((len(Ztr), 1)), Ztr])
        cov = np.linalg.pinv(Zi.T * W @ Zi)
        se = np.sqrt(np.diag(cov))[1:]
        beta = lr.coef_[0]
        wald = beta / np.where(se > 0, se, np.nan)
        pval = 2 * (1 - stats.norm.cdf(np.abs(wald)))
        for name, b, s_, w, pv in zip(FEATNAMES, beta, se, wald, pval):
            coef_rows.append({"fold": k + 1, "feature": name, "beta": b,
                              "se": s_, "wald_z": w, "p": pv})
        print(f"  fold{k+1}  XGB ROC-AUC {auc:.3f}   LR ROC-AUC "
              f"{roc_auc_score(y[te], lr.predict_proba(Zte)[:, 1]):.3f}")

    perm = pd.DataFrame(perm_rows); shp = pd.DataFrame(shap_rows)
    coef = pd.DataFrame(coef_rows)
    perm.to_csv(f"{OUT}/permutation_importance.csv", index=False)
    shp.to_csv(f"{OUT}/shap_importance.csv", index=False)
    coef.to_csv(f"{OUT}/logreg_coefficients.csv", index=False)

    agg = (perm.groupby("feature")["drop_roc_auc"].agg(["mean", "std"])
           .join(shp.groupby("feature")["mean_abs_shap"].mean().rename("shap"))
           .join(coef.groupby("feature")["beta"].mean().rename("beta"))
           .join(coef.groupby("feature")["p"].max().rename("p_worst_fold"))
           .sort_values("mean", ascending=False))
    agg.to_csv(f"{OUT}/attribution_summary.csv")
    print("\n=== top 15 features by permutation importance (ROC-AUC drop) ===")
    print(agg.head(15).round(4).to_string())
    print(f"\nfeatures significant (p<0.05) in every fold: "
          f"{(coef.groupby('feature')['p'].max() < 0.05).sum()} / {len(FEATNAMES)}")

    # ---- R16: latent-dimension probing ----------------------------------
    lat = np.load("./artifacts_ae_cnn/latent_vectors.npz", allow_pickle=True)
    fids = sorted({int(k.split("_")[0][4:]) for k in lat.files})
    Z = np.vstack([lat[f"fold{f}_z_test"] for f in fids])
    yz = np.concatenate([lat[f"fold{f}_y_test"] for f in fids])
    rows = []
    for d in range(Z.shape[1]):
        r_pb, p_pb = stats.pointbiserialr(yz, Z[:, d])
        probe = LogisticRegression(max_iter=1000).fit(Z[:, [d]], yz)
        rows.append({"dim": d, "point_biserial_r": r_pb, "p": p_pb,
                     "probe_roc_auc": roc_auc_score(yz, probe.predict_proba(Z[:, [d]])[:, 1])})
    probe = pd.DataFrame(rows).sort_values("probe_roc_auc", ascending=False)
    probe.to_csv(f"{OUT}/latent_probe.csv", index=False)
    full = LogisticRegression(max_iter=2000).fit(Z, yz)
    print("\n=== latent probing (R16) ===")
    print(probe.head(8).round(3).to_string(index=False))
    print(f"best single dim ROC-AUC : {probe['probe_roc_auc'].max():.3f}")
    print(f"all 64 dims together    : {roc_auc_score(yz, full.predict_proba(Z)[:, 1]):.3f}")
    print(f"dims with |r| > 0.20    : {(probe['point_biserial_r'].abs() > 0.20).sum()} / 64")
    print(f"\nsaved -> {OUT}/")


if __name__ == "__main__":
    main()
