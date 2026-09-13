"""Does the stop-coverage gate change the conclusions?

The rule "at least 5 matched stops and at least 50% coverage" removes about a
fifth of all candidate trips, the largest single rejection in the pipeline. It
exists because the delay label is a MEDIAN over matched stops, and a median
taken over two or three stops is not trustworthy. That is a reasonable belief,
but the report never tested it.

This rebuilds the dataset with the rule relaxed to "at least 1 matched stop",
which is the weakest rule that still allows a label to exist at all, and then
asks three questions:

  1. How many trips come back, and does the delayed share move?
  2. Does a model trained on the strict data still work on the recovered
     trips? If their labels were mostly noise, performance there should fall
     apart.
  3. Does the headline result change when the model is trained and scored on
     the relaxed dataset instead?

Question 2 is the decisive one. Question 3 alone could hide a problem, because
adding noisy labels to both train and test can leave the average metric looking
similar while the model has quietly become worse.

Stage 1 rebuilds the cache (re-parses the raw GPS, several minutes).
Stage 2 trains. Pass --skip-build to reuse an existing relaxed cache.

Outputs -> artifacts_coverage/
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

RANDOM_SEED = 42
OUT = "./artifacts_coverage"
RELAXED_DIR = "./cache_relaxed"
RELAXED_NPZ = f"{RELAXED_DIR}/sequences.npz"
RELAXED_META = f"{RELAXED_DIR}/trips_meta.parquet"
os.makedirs(OUT, exist_ok=True)


def build_relaxed():
    """Re-run preprocessing with the stop rule relaxed to a single stop.

    build_dataset() writes sequences.npz and trips_meta.parquet into whatever
    cache_dir it is given, so it MUST be pointed at a separate directory. With
    the default it would overwrite the primary cache that every other
    experiment in this report depends on.
    """
    os.environ["NETMOB_MIN_STOPS"] = "1"
    os.environ["NETMOB_MIN_COVERAGE"] = "0.0"
    import importlib
    import netmob_prep as prep
    importlib.reload(prep)
    assert prep.MIN_STOPS_MATCHED == 1 and prep.MIN_STOP_COVERAGE == 0.0
    assert os.path.abspath(RELAXED_DIR) != os.path.abspath(prep.CACHE_DIR), \
        "relaxed cache must not point at the primary cache"

    gtfs = prep.build_gtfs_index(verbose=False)
    rain = prep.load_rain_by_utc_hour()
    dates = prep.mobility_dates()
    print(f"rebuilding with MIN_STOPS=1, MIN_COVERAGE=0.0 over {len(dates)} dates")
    meta, counters = prep.build_dataset(dates, gtfs, rain, cache_dir=RELAXED_DIR)
    print(f"relaxed dataset: {len(meta):,} trips -> {RELAXED_DIR}/")


def main():
    if "--skip-build" not in sys.argv:
        build_relaxed()

    from baseline_raw import masked_stats, scale_sequences

    strict = np.load("./cache/sequences.npz", allow_pickle=True)
    relax = np.load(RELAXED_NPZ, allow_pickle=True)
    s_ids = set(strict["trip_instance_id"].tolist())
    r_ids = relax["trip_instance_id"]
    Xr, yr = relax["X"].astype(np.float32), relax["y"].astype(int)
    lr, gr = relax["lengths"].astype(int), np.asarray(relax["groups"])
    recovered = np.array([i not in s_ids for i in r_ids])

    rmeta = pd.read_parquet(RELAXED_META)
    cov = rmeta.set_index("trip_instance_id")["stop_coverage"].reindex(r_ids).to_numpy()

    print("\n" + "=" * 66)
    print(f"strict dataset     : {len(s_ids):>7,} trips   "
          f"delayed {strict['y'].astype(int).mean():6.2%}")
    print(f"relaxed dataset    : {len(yr):>7,} trips   delayed {yr.mean():6.2%}")
    print(f"recovered by relax : {recovered.sum():>7,} trips   "
          f"delayed {yr[recovered].mean():6.2%}   "
          f"({100*recovered.sum()/len(yr):.1f}% of relaxed)")
    print(f"recovered coverage : median {np.nanmedian(cov[recovered]):.3f}   "
          f"kept coverage median {np.nanmedian(cov[~recovered]):.3f}")
    print("=" * 66)

    # ---- Q2/Q3: train strict, score on both; then train relaxed ----------
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for k, (tr, te) in enumerate(splitter.split(Xr.reshape(len(Xr), -1), yr, groups=gr)):
        # strict-only training rows within this fold's training days
        tr_strict = tr[~recovered[tr]]
        Xtr_s, Xte = scale_sequences(Xr[tr_strict], Xr[te])
        Ftr_s, Fte = masked_stats(Xtr_s, lr[tr_strict]), masked_stats(Xte, lr[te])
        pw = (yr[tr_strict] == 0).sum() / max(1, (yr[tr_strict] == 1).sum())
        m_s = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=pw, eval_metric="logloss",
                                random_state=RANDOM_SEED, verbosity=0)
        m_s.fit(Ftr_s, yr[tr_strict])
        p = m_s.predict_proba(Fte)[:, 1]

        te_keep, te_rec = ~recovered[te], recovered[te]
        for tag, mask in [("strict-train / strict-test", te_keep),
                          ("strict-train / RECOVERED-test", te_rec),
                          ("strict-train / all-test", np.ones(len(te), bool))]:
            if mask.sum() > 30 and len(np.unique(yr[te][mask])) > 1:
                rows.append({"setting": tag, "fold": k + 1, "n": int(mask.sum()),
                             "roc_auc": roc_auc_score(yr[te][mask], p[mask]),
                             "pr_auc": average_precision_score(yr[te][mask], p[mask]),
                             "prevalence": float(yr[te][mask].mean())})

        # relaxed training for comparison
        Xtr_r, Xte2 = scale_sequences(Xr[tr], Xr[te])
        Ftr_r, Fte2 = masked_stats(Xtr_r, lr[tr]), masked_stats(Xte2, lr[te])
        pw2 = (yr[tr] == 0).sum() / max(1, (yr[tr] == 1).sum())
        m_r = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=pw2, eval_metric="logloss",
                                random_state=RANDOM_SEED, verbosity=0)
        m_r.fit(Ftr_r, yr[tr])
        p2 = m_r.predict_proba(Fte2)[:, 1]
        for tag, mask in [("relaxed-train / strict-test", te_keep),
                          ("relaxed-train / all-test", np.ones(len(te), bool))]:
            if mask.sum() > 30 and len(np.unique(yr[te][mask])) > 1:
                rows.append({"setting": tag, "fold": k + 1, "n": int(mask.sum()),
                             "roc_auc": roc_auc_score(yr[te][mask], p2[mask]),
                             "pr_auc": average_precision_score(yr[te][mask], p2[mask]),
                             "prevalence": float(yr[te][mask].mean())})
        print(f"  fold{k+1} done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/coverage_sensitivity.csv", index=False)
    print("\n=== RAW-STATS+XGBoost, mean over 5 folds ===")
    print(df.groupby("setting")[["n", "prevalence", "roc_auc", "pr_auc"]]
          .mean().round(4).to_string())

    # ---- is the label noisier at low coverage? --------------------------
    band = pd.cut(cov, [0, .25, .5, .75, 1.01],
                  labels=["<0.25", "0.25-0.50", "0.50-0.75", ">0.75"])
    prof = pd.DataFrame({"coverage": band, "delayed": yr,
                         "n_matched": rmeta.set_index("trip_instance_id")
                         ["n_stops_matched"].reindex(r_ids).to_numpy()})
    print("\n=== by coverage band (relaxed dataset) ===")
    print(prof.groupby("coverage", observed=True)
          .agg(trips=("delayed", "size"), delayed_share=("delayed", "mean"),
               median_matched_stops=("n_matched", "median")).round(4).to_string())
    print(f"\nsaved -> {OUT}/coverage_sensitivity.csv")


if __name__ == "__main__":
    main()
