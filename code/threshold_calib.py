"""R06 — is the fixed 95th-percentile threshold the right alarm rule?

Compares four rules on the existing out-of-fold CNN-AE scores (no retraining):
  global-p95   the paper's rule: one pooled 95th percentile
  dynamic      absolute cut at mu + 1.645*sigma, calibrated leave-one-day-out
  per-line     score re-standardized within each line, then the same cut
  per-hour     score re-standardized within each hour of day, then the same cut

The question is whether the daily alarm count can respond to conditions at all:
a quantile rule flags 5% every day by construction, an absolute rule need not.
"""
import numpy as np
import pandas as pd

K = 1.645  # normal quantile matching a 5% upper tail

df = pd.read_parquet("./cache/trips_meta.parquet").set_index("trip_instance_id")
s = pd.read_csv("./artifacts_ae_cnn/anomaly_scores.csv").set_index("trip_instance_id")
logs = np.log(s["recon_mse"].reindex(df.index))
df["score"] = (logs - logs.mean()) / logs.std()
df = df.dropna(subset=["score"])
df["date"] = df["file_date"].astype(str)
df["hour"] = pd.to_datetime(df["start_time"]).dt.hour

df["global_p95"] = df["score"] > np.percentile(df["score"], 95)

# leave-one-day-out absolute threshold: calibrate on every other day
thr = {d: (lambda o: o.mean() + K * o.std())(df.loc[df["date"] != d, "score"])
       for d in df["date"].unique()}
df["dynamic"] = df["score"] > df["date"].map(thr)

for name, key in [("per_line", "line_id"), ("per_hour", "hour")]:
    g = df.groupby(key)["score"]
    z = (df["score"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df[name] = z.fillna(0) > K

rules = ["global_p95", "dynamic", "per_line", "per_hour"]
print(f"\noverall alarm rate: " +
      "  ".join(f"{r} {df[r].mean():.1%}" for r in rules))

by_day = df.groupby("date")[rules].mean() * 100
by_day["n"] = df.groupby("date").size()
by_day["degraded"] = df.groupby("date")["degraded_day"].max()
print("\nalarm rate per day (%)")
print(by_day.round(1).sort_values("dynamic", ascending=False).to_string())

print("\nspread across days (max - min, percentage points):")
for r in rules:
    print(f"  {r:<11} {by_day[r].max() - by_day[r].min():5.1f}"
          f"   (22 Mar {by_day.loc['2026-03-22', r]:.1f}%)")
