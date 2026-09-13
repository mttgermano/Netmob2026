"""E8 — early prediction from trajectory prefixes (reviewer item R12, and the
operational argument in the introduction).

The deterministic GTFS comparison needs a finished trip before it can say
anything. A model over telemetry does not. This measures how much of the
retrospective performance survives when only the first 25%, 50% or 75% of each
trip is visible.

Truncation is applied to each trip's own real length, not to a fixed bin count,
so "50%" means half of that trip's route regardless of how long it ran. The
tail is zeroed and the recorded length shortened, exactly the padding
convention the full-trip pipeline already uses.

Same day-grouped folds and the same models as baseline_raw.py, so the 100% row
reproduces the RAW-STATS/RAW-FLAT numbers in the report.

Outputs -> artifacts_e8/results.csv
Usage:  python exp_e8_prefix.py   (NETMOB_SMOKE=1 for a fast pass)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from baseline_raw import (masked_stats, scale_sequences, compute_metrics,
                          classifiers, METRIC_NAMES)

RANDOM_SEED = 42
SMOKE = os.environ.get("NETMOB_SMOKE", "0") == "1"
FOLDS = 2 if SMOKE else 5
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
OUT = "./artifacts_e8"
os.makedirs(OUT, exist_ok=True)


def truncate(X, lens, frac):
    """Keep the first `frac` of each trip's real length; zero the rest.

    At least one bin is always kept so masked_stats never sees an empty trip.
    """
    new_lens = np.maximum(1, np.floor(lens * frac).astype(int))
    Xc = X.copy()
    T = X.shape[1]
    keep = np.arange(T)[None, :] < new_lens[:, None]
    Xc[~keep] = 0.0
    return Xc, new_lens


def _selftest():
    X = np.ones((2, 10, 3), np.float32)
    lens = np.array([10, 4])
    Xc, nl = truncate(X, lens, 0.5)
    assert (nl == [5, 2]).all(), nl
    assert Xc[0, :5].all() and not Xc[0, 5:].any()
    assert Xc[1, :2].all() and not Xc[1, 2:].any()
    # a 25% prefix of a 2-bin trip must still keep one bin
    assert truncate(X, np.array([2, 2]), 0.25)[1].tolist() == [1, 1]
    print("selftest ok")


def main():
    z = np.load("./cache/sequences.npz", allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(int)
    lens, groups = z["lengths"].astype(int), np.asarray(z["groups"])
    if SMOKE:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(y), 3000, False)
        X, y, lens, groups = X[idx], y[idx], lens[idx], groups[idx]
    print(f"N={len(y):,}  delayed={y.mean():.1%}  folds={FOLDS}")

    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for frac in FRACTIONS:
        Xf, lf = truncate(X, lens, frac)
        for k, (tr, te) in enumerate(
                splitter.split(Xf.reshape(len(Xf), -1), y, groups=groups)):
            Xtr, Xte = scale_sequences(Xf[tr], Xf[te])
            reps = {"RAW-STATS": (masked_stats(Xtr, lf[tr]), masked_stats(Xte, lf[te])),
                    "RAW-FLAT": (Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1))}
            pw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
            for rep, (Ftr, Fte) in reps.items():
                for name, clf in classifiers(pw).items():
                    if name == "RF" and rep == "RAW-FLAT":
                        continue                      # slow and never competitive
                    clf.fit(Ftr, y[tr])
                    m = compute_metrics(y[te], clf.predict_proba(Fte)[:, 1])
                    rows.append({"frac": frac, "model": f"{rep}+{name}",
                                 "fold": k + 1, **m})
        best = pd.DataFrame(rows)
        best = best[(best.frac == frac) & (best.model == "RAW-STATS+XGBoost")]
        print(f"  {int(frac*100):>3}% of route : RAW-STATS+XGBoost "
              f"ROC-AUC {best.roc_auc.mean():.3f}  PR-AUC {best.pr_auc.mean():.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/results.csv", index=False)
    piv = df.groupby(["model", "frac"])[["roc_auc", "pr_auc", "f1"]].mean().round(3)
    print("\n=== mean over folds ===")
    print(piv.to_string())

    ref = df[df.frac == 1.00].groupby("model")["roc_auc"].mean()
    print("\n=== fraction of full-trip ROC-AUC retained (above 0.5 baseline) ===")
    for model in sorted(df.model.unique()):
        full = ref[model] - 0.5
        line = [f"{model:20s}"]
        for f in FRACTIONS:
            v = df[(df.model == model) & (df.frac == f)]["roc_auc"].mean() - 0.5
            line.append(f"{int(f*100):>3}%: {v / full:5.1%}")
        print("  ".join(line))
    print(f"\nsaved -> {OUT}/results.csv")


if __name__ == "__main__":
    _selftest()
    main()
