"""R05 — does the anomaly signal survive removing rainfall from the AE input?

Compares the paper's CNN-AE (12 channels, rain_mm included) against the same
architecture trained without rain_mm, on the three external validations of
03_anomaly_analysis.ipynb: degraded-telemetry days, rain, and the 23 March
disruption. Circularity would show up as the rain effect collapsing while the
operational signals stay put.

    python ablation_norain.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import netmob_prep as prep

ANOM_PCT = 95
VARIANTS = {"with rain": "./artifacts_ae_cnn", "no rain": "./artifacts_ae_cnn_norain"}

meta = pd.read_parquet("./cache/trips_meta.parquet").set_index("trip_instance_id")
df = meta.copy()
for tag, adir in VARIANTS.items():
    s = pd.read_csv(f"{adir}/anomaly_scores.csv").set_index("trip_instance_id")
    logs = np.log(s["recon_mse"].reindex(df.index))
    df[f"score::{tag}"] = (logs - logs.mean()) / logs.std()
df = df.dropna(subset=[f"score::{t}" for t in VARIANTS])
for tag in VARIANTS:
    df[f"anom::{tag}"] = df[f"score::{tag}"] > np.nanpercentile(df[f"score::{tag}"], ANOM_PCT)

# `code/` is a symlink, so netmob_prep's relative WEATHER_CSV default resolves
# into the wrong tree; anchor it to this file's real location instead.
weather_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "..", "..", "..", "data", "auxiliar_data",
                           "meteorological_data.csv")
rain_utc = prep.load_rain_by_utc_hour(os.path.normpath(weather_csv))
starts = (pd.to_datetime(df["start_time"]) + pd.Timedelta(hours=3)).dt.floor("h")
df["rain_mm_trip"] = rain_utc.reindex(starts).fillna(0.0).to_numpy()
df["rainy"] = df["rain_mm_trip"] > 0

a, b = list(VARIANTS)
print(f"\nn = {len(df):,} trips scored out-of-fold by both variants")
print(f"score agreement: Pearson r = {df[f'score::{a}'].corr(df[f'score::{b}']):.3f}, "
      f"Spearman rho = {df[f'score::{a}'].corr(df[f'score::{b}'], method='spearman'):.3f}")
both = (df[f"anom::{a}"] & df[f"anom::{b}"]).sum()
either = (df[f"anom::{a}"] | df[f"anom::{b}"]).sum()
print(f"flagged sets: {df[f'anom::{a}'].sum()} / {df[f'anom::{b}'].sum()}, "
      f"shared {both}, Jaccard {both / max(1, either):.2f}")

for tag in VARIANTS:
    sc, an = f"score::{tag}", f"anom::{tag}"
    deg = df.loc[df["degraded_day"] == 1, sc]
    nor = df.loc[df["degraded_day"] == 0, sc]
    p_deg = stats.mannwhitneyu(deg, nor, alternative="greater").pvalue
    p_rain = stats.mannwhitneyu(df.loc[df["rainy"], sc], df.loc[~df["rainy"], sc],
                                alternative="greater").pvalue
    top20 = df.nlargest(20, sc)
    mar23 = (pd.to_datetime(top20["start_time"]).dt.strftime("%Y-%m-%d") == "2026-03-23").sum()
    print(f"\n── CNN-AE, {tag} " + "─" * 40)
    print(f"  degraded days : median {deg.median():+.2f} vs {nor.median():+.2f} "
          f"(p={p_deg:.1e}) | flag rate {df.loc[df['degraded_day']==1, an].mean():.1%} "
          f"vs {df.loc[df['degraded_day']==0, an].mean():.1%}")
    print(f"  rain          : flag rate {df.loc[df['rainy'], an].mean():.1%} wet vs "
          f"{df.loc[~df['rainy'], an].mean():.1%} dry (p={p_rain:.1e})")
    print(f"  top-20 trips  : {mar23}/20 on 23 March | "
          f"{top20['line_id'].nunique()} distinct lines")
