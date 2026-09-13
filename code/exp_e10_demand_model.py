"""E10 Part 1b — does ticketing demand improve delay detection?

Takes the best model in the report (RAW-STATS + XGBoost, 0.846 ROC-AUC) and
appends nine demand features joined on (service_date, line, hour-of-start):
raw boardings on that line in that hour, distinct riders, transfer / free-pass
/ cash shares, network-wide boardings in that hour, the line's share of the
network in that hour, the ratio of this cell's boardings to the line's own
median for that hour-of-day (a "busier than usual" signal that is comparable
across lines), and a flag for whether the join found anything at all.

Everything else is held fixed: the same sequences.npz, the same
StratifiedGroupKFold(5, shuffle, seed 42) grouped by service_date, the same
XGBoost hyper-parameters. The comparison is therefore paired fold-by-fold and
a paired t-test over the 5 folds is the right test, with the caveat that n=5
gives it very little power — a null result here means "no large effect", not
"no effect", and we report the per-fold deltas so the reader can judge.

Judgment calls. The ticket schema has no direction field and its vehicle ids do
not overlap the AVL vehicle ids, so demand can only be attached at line x hour
granularity; a trip inherits the demand of every trip on its line in its hour,
including the opposite direction. 6.7% of trips are on lines absent from the
ticket feed and get the missing flag with zero-filled counts rather than being
dropped, so the sample stays identical to the baseline. The hour aggregate
spans the whole clock hour, so it includes boardings after the trip started;
this is fine for an operational "was this an unusually busy hour" feature but
it is not a strictly causal ex-ante predictor.

The decisive control is RAW-STATS+LINEID: a single integer line code. Demand
volume on a line is largely a fingerprint of which line it is, and Part 2 shows
line identity alone separates delay rates from 1.4% to 48.8%. If the demand
gain disappears once line identity is in the model, the ticketing data is
adding nothing the operator does not already know from the timetable.

Outputs -> artifacts_e10/part1_*.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from baseline_raw import (masked_stats, scale_sequences, compute_metrics,
                          classifiers, METRIC_NAMES)

RANDOM_SEED = 42
SMOKE = os.environ.get("NETMOB_SMOKE", "0") == "1"
FOLDS = 2 if SMOKE else 5
OUT = "./artifacts_e10"
DEMAND_COLS = ["boardings", "riders", "transfer_share", "free_share",
               "cash_share", "net_boardings", "line_share", "load_ratio",
               "has_demand"]


def build_demand_table():
    """One row per trip_instance_id, aligned to trips_meta order."""
    trips = pd.read_parquet("./cache/trips_meta.parquet")
    line = pd.read_csv(f"{OUT}/demand_line_hour.csv")
    net = pd.read_csv(f"{OUT}/demand_net_hour.csv")
    line["line_id"] = line["line_id"].astype(str)

    # per-line median boardings for each hour-of-day, over the whole month
    med = (line.groupby(["line_id", "hour"])["boardings"].median()
           .rename("line_hour_median").reset_index())
    line = line.merge(med, on=["line_id", "hour"], how="left").merge(
        net, on=["service_date", "hour"], how="left")
    line["line_share"] = line["boardings"] / line["net_boardings"]
    line["load_ratio"] = line["boardings"] / line["line_hour_median"].replace(0, np.nan)

    t = trips[["trip_instance_id", "service_date", "line_id", "start_time"]].copy()
    t["line_id"] = (t["line_id"].astype(str).str.strip().str.upper()
                    .str.replace(r"\.0$", "", regex=True))
    t["hour"] = pd.to_datetime(t["start_time"]).dt.hour
    m = t.merge(line, on=["service_date", "line_id", "hour"], how="left")
    m["has_demand"] = m["boardings"].notna().astype(float)
    for c in DEMAND_COLS[:-1]:
        m[c] = m[c].fillna(0.0)
    assert len(m) == len(trips)
    return trips["trip_instance_id"].values, m.set_index("trip_instance_id")[DEMAND_COLS]


def _pairs(a, b, base, alt):
    out = []
    for m in ["roc_auc", "pr_auc", "f1"]:
        d = b[m].values - a[m].values
        t, p = stats.ttest_rel(b[m].values, a[m].values)
        out.append({"base": base, "alt": alt, "metric": m,
                    "base_mean": a[m].mean(), "base_sd": a[m].std(ddof=1),
                    "alt_mean": b[m].mean(), "alt_sd": b[m].std(ddof=1),
                    "mean_delta": d.mean(), "sd_delta": d.std(ddof=1),
                    "t": t, "p": p, "folds_improved": int((d > 0).sum())})
        print(f"  {m:<8} {a[m].mean():.4f}+-{a[m].std(ddof=1):.4f} -> "
              f"{b[m].mean():.4f}+-{b[m].std(ddof=1):.4f}   "
              f"delta {d.mean():+.4f}+-{d.std(ddof=1):.4f}  t={t:+.3f}  p={p:.3f}  "
              f"({int((d > 0).sum())}/{len(d)} folds up)")
    return out


def main():
    z = np.load("./cache/sequences.npz", allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(int)
    lens, groups = z["lengths"].astype(int), np.asarray(z["groups"])
    tids = np.asarray(z["trip_instance_id"]) if "trip_instance_id" in z.files else None
    order, dem = build_demand_table()
    meta = pd.read_parquet("./cache/trips_meta.parquet").set_index("trip_instance_id")
    if tids is None:
        tids = order            # sequences are stored in trips_meta order
        assert len(tids) == len(y)
    D = dem.reindex(pd.Index(tids)).values.astype(np.float32)
    assert not np.isnan(D).any()
    print(f"N={len(y):,}  delayed={y.mean():.1%}  demand-joined={D[:, -1].mean():.1%}")

    # line identity as an integer code, to test whether the demand features are
    # merely a fingerprint for "which line is this" (Part 2 shows line identity
    # alone carries a lot of signal).
    L = meta["line_id"].reindex(pd.Index(tids)).astype("category").cat.codes.values
    L = L.reshape(-1, 1).astype(np.float32)

    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for k, (tr, te) in enumerate(splitter.split(X.reshape(len(X), -1), y, groups=groups)):
        Xtr, Xte = scale_sequences(X[tr], X[te])
        Ftr, Fte = masked_stats(Xtr, lens[tr]), masked_stats(Xte, lens[te])
        reps = {
            "RAW-STATS": (Ftr, Fte),
            "RAW-STATS+DEMAND": (np.hstack([Ftr, D[tr]]), np.hstack([Fte, D[te]])),
            "DEMAND-ONLY": (D[tr], D[te]),
            "RAW-STATS+LINEID": (np.hstack([Ftr, L[tr]]), np.hstack([Fte, L[te]])),
            "RAW-STATS+LINEID+DEMAND": (np.hstack([Ftr, L[tr], D[tr]]),
                                        np.hstack([Fte, L[te], D[te]])),
            "LINEID-ONLY": (L[tr], L[te]),
        }
        pw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        for rep, (A, B) in reps.items():
            clf = classifiers(pw)["XGBoost"]
            clf.fit(A, y[tr])
            m = compute_metrics(y[te], clf.predict_proba(B)[:, 1])
            rows.append({"model": rep, "fold": k + 1, **m})
            print(f"  fold{k+1} {rep:<18} ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/part1_folds.csv", index=False)
    summ = df.groupby("model")[METRIC_NAMES].agg(["mean", "std"]).round(4)
    summ.to_csv(f"{OUT}/part1_summary.csv")
    print("\n" + summ.to_string())

    tt = []
    for base, alt in [("RAW-STATS", "RAW-STATS+DEMAND"),
                      ("RAW-STATS", "RAW-STATS+LINEID"),
                      ("RAW-STATS+LINEID", "RAW-STATS+LINEID+DEMAND")]:
        a = df[df.model == base].sort_values("fold")
        b = df[df.model == alt].sort_values("fold")
        print(f"\n=== paired t-test, {alt} vs {base} (n=5 folds) ===")
        tt += _pairs(a, b, base, alt)
    pd.DataFrame(tt).to_csv(f"{OUT}/part1_ttest.csv", index=False)

    # marginal association of each demand feature with the label, for context
    ass = [{"feature": c,
            "auc_alone": roc_auc_score(y, D[:, i]),
            "r_pointbiserial": stats.pointbiserialr(y, D[:, i])[0]}
           for i, c in enumerate(DEMAND_COLS)]
    ass = pd.DataFrame(ass).sort_values("auc_alone", ascending=False)
    ass.to_csv(f"{OUT}/part1_feature_association.csv", index=False)
    print("\n=== univariate association of each demand feature with delay ===")
    print(ass.round(4).to_string(index=False))
    print(f"\nsaved -> {OUT}/")


if __name__ == "__main__":
    main()
