"""E9 — hyperparameter search with nested-in-fold selection (reviewer R11).

Every number in the report comes from literature defaults, so the comparison
between representations is a comparison of untuned models. This checks whether
tuning changes the ranking.

Selection is nested inside each outer fold: the grid is scored on an inner
split carved out of that fold's TRAINING days only, the winner is refit on the
full training split, and only then is it scored on the outer test fold. No
configuration ever sees an outer test day before being chosen, which is the
failure mode that would trade one reviewer objection for a worse one.

Scope: the search runs over the classifier side (XGBoost, random forest,
logistic regression) on RAW-STATS, the best-performing representation in the
report. Searching the AE architecture as well would multiply the cost by the
grid size, and Section E7 already shows the AE is not where the performance
comes from; that limitation is stated rather than hidden.

Outputs -> artifacts_e9/
Usage:  python exp_e9_hparam.py   (NETMOB_SMOKE=1 for a fast pass)
"""
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import xgboost as xgb

from baseline_raw import masked_stats, scale_sequences, compute_metrics, METRIC_NAMES

RANDOM_SEED = 42
SMOKE = os.environ.get("NETMOB_SMOKE", "0") == "1"
FOLDS = 2 if SMOKE else 5
OUT = "./artifacts_e9"
os.makedirs(OUT, exist_ok=True)

# Grid kept deliberately small. Nested selection refits every point on the
# inner split and then twice more on the outer fold, so cost grows fast: a
# 34-point grid did not finish in 15 minutes even on the smoke subset.
GRIDS = {
    "XGBoost": {"max_depth": [4, 6, 10], "learning_rate": [0.05, 0.1]},
    "RF": {"max_depth": [12, 20], "min_samples_leaf": [1, 5]},
    "LR": {"C": [0.05, 0.25, 1.0, 4.0]},
}
DEFAULTS = {"XGBoost": {"max_depth": 6, "learning_rate": 0.05},
            "RF": {"max_depth": 12, "min_samples_leaf": 1},
            "LR": {"C": 1.0}}


def build(name, params, pw):
    cw = {0: 1.0, 1: pw}
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=pw, eval_metric="logloss",
                                 random_state=RANDOM_SEED, verbosity=0, **params)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, class_weight=cw,
                                      random_state=RANDOM_SEED, n_jobs=-1, **params)
    return LogisticRegression(class_weight=cw, solver="lbfgs", max_iter=2000,
                              random_state=RANDOM_SEED, **params)


def grid_points(g):
    keys = list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, vals))


def main():
    z = np.load("./cache/sequences.npz", allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(int)
    lens, groups = z["lengths"].astype(int), np.asarray(z["groups"])
    if SMOKE:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(y), 3000, False)
        X, y, lens, groups = X[idx], y[idx], lens[idx], groups[idx]
    print(f"N={len(y):,}  folds={FOLDS}  "
          f"grid sizes: {[(k, len(list(grid_points(v)))) for k, v in GRIDS.items()]}")

    outer = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows, picks = [], []
    for k, (tr, te) in enumerate(outer.split(X.reshape(len(X), -1), y, groups=groups)):
        Xtr, Xte = scale_sequences(X[tr], X[te])
        Ftr, Fte = masked_stats(Xtr, lens[tr]), masked_stats(Xte, lens[te])
        ytr, yte, gtr = y[tr], y[te], groups[tr]
        pw = (ytr == 0).sum() / max(1, (ytr == 1).sum())

        # inner split over TRAINING days only
        inner = StratifiedGroupKFold(2, shuffle=True, random_state=RANDOM_SEED)
        i_tr, i_va = next(inner.split(Ftr, ytr, groups=gtr))
        assert not (set(np.unique(gtr[i_tr])) & set(np.unique(gtr[i_va])))
        pw_i = (ytr[i_tr] == 0).sum() / max(1, (ytr[i_tr] == 1).sum())

        for name, grid in GRIDS.items():
            scored = []
            for params in grid_points(grid):
                clf = build(name, params, pw_i).fit(Ftr[i_tr], ytr[i_tr])
                s = average_precision_score(ytr[i_va],
                                            clf.predict_proba(Ftr[i_va])[:, 1])
                scored.append((s, params))
            best_s, best_p = max(scored, key=lambda t: t[0])
            picks.append({"fold": k + 1, "model": name, "inner_pr_auc": best_s,
                          **{f"best_{a}": b for a, b in best_p.items()}})

            for tag, params in [("tuned", best_p), ("default", DEFAULTS[name])]:
                clf = build(name, params, pw).fit(Ftr, ytr)
                m = compute_metrics(yte, clf.predict_proba(Fte)[:, 1])
                rows.append({"model": f"RAW-STATS+{name}", "variant": tag,
                             "fold": k + 1, **m})
            print(f"  fold{k+1} {name:<8} picked {best_p}")

    df = pd.DataFrame(rows); pk = pd.DataFrame(picks)
    df.to_csv(f"{OUT}/results.csv", index=False)
    pk.to_csv(f"{OUT}/selected_params.csv", index=False)

    piv = df.groupby(["model", "variant"])[["roc_auc", "pr_auc", "f1"]].agg(["mean", "std"])
    print("\n=== tuned vs default (nested selection) ===")
    print(piv.round(4).to_string())

    from scipy import stats as st
    print("\n=== paired test over folds, tuned minus default ===")
    for model in sorted(df.model.unique()):
        for met in ["roc_auc", "pr_auc"]:
            a = df[(df.model == model) & (df.variant == "tuned")].sort_values("fold")[met].values
            b = df[(df.model == model) & (df.variant == "default")].sort_values("fold")[met].values
            d = np.mean(a - b)
            p = st.ttest_rel(a, b).pvalue if np.ptp(a - b) > 0 else 1.0
            print(f"  {model:20s} {met:8s} diff={d:+.4f}  p_t={p:.3f}")
    print(f"\nsaved -> {OUT}/")


if __name__ == "__main__":
    main()
