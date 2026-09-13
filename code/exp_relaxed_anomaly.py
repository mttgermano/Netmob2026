"""Do the anomaly and per-line findings survive on the relaxed dataset?

Everything in the anomaly sections of the report was computed on the strict
22,605 trips. This repeats the load-bearing checks on the 27,494-trip relaxed
dataset, where the trips the stop rule discarded have been put back:

  * the day-level ordering and the weekend effect;
  * degraded telemetry days, stratified by day type so the weekend confound
    cannot carry the result;
  * the threshold comparison, global percentile against per-line calibration;
  * the chronic line-directions from E10.

The rain ablation is not here because it needs a second autoencoder trained
without the weather channel on the relaxed data; that runs separately.

Usage:  python exp_relaxed_anomaly.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = "./artifacts_relaxed_anomaly"
os.makedirs(OUT, exist_ok=True)
K = 1.645
DEGRADED = {"2026-03-20", "2026-03-22", "2026-03-28"}


def load(scores_csv, meta_path):
    m = pd.read_parquet(meta_path).set_index("trip_instance_id")
    s = pd.read_csv(scores_csv).set_index("trip_instance_id")["recon_mse"]
    l = np.log(s.reindex(m.index))
    m = m.assign(z=(l - l.mean()) / l.std()).dropna(subset=["z"])
    m["date"] = m["file_date"].astype(str)
    m["wknd"] = pd.to_datetime(m["date"]).dt.dayofweek >= 5
    m["deg"] = m["date"].isin(DEGRADED)
    m["hour"] = pd.to_datetime(m["start_time"]).dt.hour
    return m


def report(tag, m):
    print(f"\n{'='*64}\n{tag}   n = {len(m):,}\n{'='*64}")

    day = m.groupby("date").agg(z=("z", "median"), n=("z", "size"),
                                deg=("deg", "max"), wknd=("wknd", "max"))
    day = day.sort_values("z", ascending=False)
    top7 = day.head(7)
    print(f"weekend days in the top 7 by day-median score : "
          f"{int(top7.wknd.sum())} of 7   (6 weekends among {len(day)} days)")
    a, b = m.loc[m.wknd, "z"], m.loc[~m.wknd, "z"]
    p = stats.mannwhitneyu(a, b, alternative="greater").pvalue
    print(f"weekend vs weekday, trip level : {a.median():+.3f} vs {b.median():+.3f}"
          f"   p = {p:.2e}")

    print("\ndegraded vs normal, stratified by day type:")
    for lbl, sub in (("weekday", m[~m.wknd]), ("weekend", m[m.wknd])):
        d, n = sub[sub.deg], sub[~sub.deg]
        if len(d) and len(n):
            pv = stats.mannwhitneyu(d.z, n.z, alternative="greater").pvalue
            print(f"  {lbl:8s} degraded {d.z.median():+.3f} (n={len(d):,})"
                  f"  normal {n.z.median():+.3f} (n={len(n):,})   p = {pv:.2e}")
    sun = m[pd.to_datetime(m.date).dt.dayofweek == 6]
    if len(sun):
        x, y = sun[sun.date == "2026-03-22"], sun[sun.date != "2026-03-22"]
        if len(x) and len(y):
            pv = stats.mannwhitneyu(x.z, y.z, alternative="greater").pvalue
            print(f"  22 Mar vs other Sundays {x.z.median():+.3f} vs "
                  f"{y.z.median():+.3f}   p = {pv:.2e}")

    print("\nthreshold rules, alarm rate on 22 March vs overall:")
    flags = {"global-p95": m.z > np.percentile(m.z, 95)}
    thr = {d: (lambda o: o.mean() + K * o.std())(m.loc[m.date != d, "z"])
           for d in m.date.unique()}
    flags["dynamic"] = m.z > m["date"].map(thr)
    for name, key in (("per-line", "line_id"), ("per-hour", "hour")):
        g = m.groupby(key)["z"]
        zz = (m.z - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        flags[name] = zz.fillna(0) > K
    rows = []
    for name, f in flags.items():
        overall, mar22 = f.mean(), f[m.date == "2026-03-22"].mean()
        rows.append({"rule": name, "overall": overall, "mar22": mar22,
                     "lift": mar22 / max(overall, 1e-9)})
        print(f"  {name:11s} overall {overall:6.1%}   22 Mar {mar22:6.1%}"
              f"   lift {mar22/max(overall,1e-9):.1f}x")
    pd.DataFrame(rows).to_csv(f"{OUT}/thresholds_{tag.split()[0].lower()}.csv", index=False)

    print("\nchronic line-directions (>=50 trips, Wilson lower bound above base):")
    base = m["delayed"].mean()
    g = m.groupby(["line_id", "direction"])["delayed"].agg(["size", "mean", "sum"])
    g = g[g["size"] >= 50]
    n, ph = g["size"], g["mean"]
    z = 1.96
    den = 1 + z**2 / n
    centre = (ph + z**2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    g["wilson_lo"] = centre - half
    chronic = g[g.wilson_lo > base]
    key = set(chronic.index)
    sel = pd.Series(list(zip(m.line_id, m.direction)), index=m.index).isin(key)
    print(f"  base rate {base:.3%} | {len(chronic)} chronic of {len(g)} eligible")
    print(f"  they carry {sel.mean():.1%} of trips and "
          f"{m.loc[sel,'delayed'].sum()/m['delayed'].sum():.1%} of delayed trips "
          f"({(m.loc[sel,'delayed'].sum()/m['delayed'].sum())/sel.mean():.2f}x)")
    chronic.sort_values("mean", ascending=False).to_csv(
        f"{OUT}/chronic_{tag.split()[0].lower()}.csv")
    return day


strict = load("./artifacts_ae_cnn/anomaly_scores.csv", "./cache/trips_meta.parquet")
relax = load("./artifacts_ae_cnn_relaxed/anomaly_scores.csv",
             "./cache_relaxed/trips_meta.parquet")
ds = report("STRICT (reference)", strict)
dr = report("RELAXED (+17% restored)", relax)

common = ds.index.intersection(dr.index)
rho = stats.spearmanr(ds.loc[common, "z"], dr.loc[common, "z"]).statistic
print(f"\n{'='*64}")
print(f"day-median ordering, strict vs relaxed: Spearman rho = {rho:.3f} "
      f"over {len(common)} shared dates")
print(f"saved -> {OUT}/")
