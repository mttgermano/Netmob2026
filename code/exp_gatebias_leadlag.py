"""Two checks quoted in the report that previously lived only in a session.

  1. Gate bias: do days that lose more trips to the quality gates look more
     punctual among their survivors?  (Section 3.B)
  2. Lead-lag: do lines move together in the same hour, or does one lead
     another by an hour?  (Section 8.C)

Usage: python exp_gatebias_leadlag.py   ->  artifacts_e10/part3_leadlag.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

m = pd.read_parquet("./cache/trips_meta.parquet")
m["date"] = m["file_date"].astype(str)
m["hour"] = pd.to_datetime(m["start_time"]).dt.hour

# ---- 1. gate bias ------------------------------------------------------
c = pd.read_csv("./cache/preprocess_counters.csv")
c["date"] = c["date"].astype(str)
c["drop"] = 1 - c["kept"] / c["instances_raw"]
s = pd.read_csv("./artifacts_ae_cnn_norain/anomaly_scores.csv").set_index("trip_instance_id")["recon_mse"]
l = np.log(s.reindex(m["trip_instance_id"]).to_numpy())
m["z"] = (l - np.nanmean(l)) / np.nanstd(l)
day = m.groupby("date").agg(delay=("delayed", "mean"), z=("z", "median"), deg=("degraded_day", "max"))
j = c.set_index("date").join(day)
r1 = stats.spearmanr(j["drop"], j["delay"])
r2 = stats.spearmanr(j["drop"], j["z"])
d, n = j.loc[j.deg == 1, "drop"], j.loc[j.deg == 0, "drop"]
print(f"drop rate vs retained delay rate : rho = {r1.statistic:+.3f}  p = {r1.pvalue:.3f}")
print(f"drop rate vs day-median score    : rho = {r2.statistic:+.3f}  p = {r2.pvalue:.3f}")
print(f"degraded days drop {d.mean():.1%} vs normal {n.mean():.1%}  "
      f"Welch t p = {stats.ttest_ind(d, n, equal_var=False).pvalue:.3f}")

# ---- 2. lead-lag over line-hour cells ---------------------------------
cell = m.groupby(["line_id", "date", "hour"])["delayed"].agg(["mean", "size"])
cell = cell[cell["size"] >= 3]
piv = cell["mean"].unstack("line_id").sort_index()
lines = [k for k in piv.columns if piv[k].notna().sum() >= 50]
rows = []
for a in lines:
    for b in lines:
        if a == b:
            continue
        x = piv[[a, b]].copy()
        x["b_next"] = x.groupby(level=0)[b].shift(-1)      # b one hour later, same date
        same, lag = x[[a, b]].dropna(), x[[a, "b_next"]].dropna()
        if len(same) >= 20 and len(lag) >= 20:
            rows.append({"lead": a, "follow": b, "r_same_hour": same[a].corr(same[b]),
                         "r_lag1": lag[a].corr(lag["b_next"])})
df = pd.DataFrame(rows)
diff = df["r_same_hour"] - df["r_lag1"]
t = stats.ttest_1samp(diff, 0)
print(f"\n{len(cell):,} line-hour cells, {len(lines)} lines, {len(df)} ordered pairs")
print(f"same hour r = {df.r_same_hour.mean():+.3f}   lag-1 r = {df.r_lag1.mean():+.3f}   "
      f"drop = {diff.mean():.3f}  (paired t p = {t.pvalue:.1e})")
print(f"minimum detectable drop at 80% power ~ {2.8 * diff.std() / np.sqrt(len(diff)):.3f}")
df.to_csv("./artifacts_e10/part3_leadlag.csv", index=False)
