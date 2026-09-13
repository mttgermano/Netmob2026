"""Figures for the full report that the short-paper artifacts do not cover.

fig_daylevel.png  day-median anomaly score for both CNN-AE variants, with
                  weekends and known-degraded days marked. This is the figure
                  behind the report's central anomaly claim: day type, not
                  disruption, drives the day-level ordering.
fig_threshold.png alarm rate per day under the four threshold rules of E6.
fig_rain.png      anomaly score by rain condition for the weather-inclusive
                  models, restyled from the notebook version so the whole
                  figure set shares one font and palette.

Styling follows the figures already in the report (t-SNE, rain_vs_score,
score_agreement) so the whole figure set reads as one family:
  * seaborn whitegrid, same as 03_anomaly_analysis.ipynb
  * blue #275DBD / red #D91A26 as the primary pair, with green #1D9E75 and
    grey ink for supporting series
  * Glacial Indifference (OFL, installed under ~/.local/share/fonts), the same
    geometric sans the LaTeX tables use.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# code/ is often reached through a symlink, so start from the real file
# location and walk up for an existing full_report/figures. Set
# NETMOB_FIG_DIR to override.
def _figure_dir():
    env = os.environ.get("NETMOB_FIG_DIR")
    if env:
        return env
    d = os.path.dirname(os.path.realpath(__file__))
    while d != "/":
        cand = os.path.join(d, "full_report", "figures")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    # fresh clone: create it next to the repo root
    repo = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    cand = os.path.join(repo, "full_report", "figures")
    os.makedirs(cand, exist_ok=True)
    return cand


OUT = _figure_dir()
K = 1.645

BLUE, RED, GREEN = "#275DBD", "#D91A26", "#1D9E75"
ORANGE = RED  # the report's second series is now red, not orange
INK = "#2A2A2A"

sns.set_theme(style="whitegrid", font="Glacial Indifference")
plt.rcParams.update({
    "font.family": "Glacial Indifference",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titleweight": "normal",
    "grid.color": "#D6D6D6",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

df = pd.read_parquet("./cache/trips_meta.parquet").set_index("trip_instance_id")
for tag, d in [("rain", "artifacts_ae_cnn"), ("norain", "artifacts_ae_cnn_norain")]:
    s = pd.read_csv(f"./{d}/anomaly_scores.csv").set_index("trip_instance_id")["recon_mse"]
    l = np.log(s.reindex(df.index))
    df[tag] = (l - l.mean()) / l.std()
df = df.dropna(subset=["rain", "norain"])
df["date"] = df["file_date"].astype(str)
df["hour"] = pd.to_datetime(df["start_time"]).dt.hour

# ── Figure: day-level score, both variants ────────────────────────────────
g = df.groupby("date")[["rain", "norain"]].median()
g["deg"] = df.groupby("date")["degraded_day"].max()
g["wknd"] = pd.to_datetime(g.index).dayofweek >= 5
g = g.sort_values("norain", ascending=False)

fig, ax = plt.subplots(figsize=(11, 3.1))
x = np.arange(len(g))
ax.bar(x - 0.205, g["rain"], 0.41, label="CNN-AE, rainfall in input",
       color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
ax.bar(x + 0.205, g["norain"], 0.41, label="CNN-AE, weather-free",
       color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)

lo = g[["rain", "norain"]].min().min() - 0.10
hi = g[["rain", "norain"]].max().max() + 0.34
for i, (_, r) in enumerate(g.iterrows()):
    if r["wknd"]:
        ax.axvspan(i - 0.5, i + 0.5, color=INK, alpha=0.055, zorder=0, linewidth=0)
    if r["deg"]:
        ax.plot(i, hi - 0.035, marker="v", ms=7, color=RED, zorder=5, clip_on=False)

ax.axhline(0, color=INK, lw=0.9, zorder=4)
ax.set_ylim(lo, hi)
ax.set_xlim(-0.6, len(g) - 0.4)
ax.set_xticks(x)
ax.set_xticklabels([d[5:] for d in g.index], rotation=45, ha="right")
ax.set_ylabel("day-median anomaly score")
ax.set_xlabel("service date in March 2026 (shaded band = weekend)")
deg_handle, = ax.plot([], [], marker="v", ls="", ms=7, color=RED,
                      label="known degraded telemetry")
bars = ax.containers[0], ax.containers[1]
# explicit order: matplotlib otherwise lists the Line2D handle before the bars
ax.legend([bars[0], bars[1], deg_handle],
          [bars[0].get_label(), bars[1].get_label(), deg_handle.get_label()],
          frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.90),
          ncol=1, handlelength=1.4)
sns.despine(ax=ax, left=True, bottom=True)
fig.savefig(f"{OUT}/fig_daylevel.png", dpi=200)
plt.close(fig)

# ── Figure: alarm rate under four threshold rules ─────────────────────────
df["global_p95"] = df["norain"] > np.percentile(df["norain"], 95)
thr = {d: (lambda o: o.mean() + K * o.std())(df.loc[df.date != d, "norain"])
       for d in df.date.unique()}
df["dynamic"] = df["norain"] > df["date"].map(thr)
for name, key in [("per_line", "line_id"), ("per_hour", "hour")]:
    gr = df.groupby(key)["norain"]
    z = (df["norain"] - gr.transform("mean")) / gr.transform("std").replace(0, np.nan)
    df[name] = z.fillna(0) > K

rules = [("global_p95", "global p95", INK, "o", "--"),
         ("dynamic", "dynamic mean/SD", GREEN, "s", "-"),
         ("per_line", "per line", ORANGE, "D", "-"),
         ("per_hour", "per hour", BLUE, "^", "-")]
bd = df.groupby("date")[[r[0] for r in rules]].mean() * 100
bd["deg"] = df.groupby("date")["degraded_day"].max()
bd = bd.sort_values("per_line", ascending=False)

fig, ax = plt.subplots(figsize=(11, 3.1))
for i, (_, r) in enumerate(bd.iterrows()):
    if r["deg"]:
        ax.axvspan(i - 0.5, i + 0.5, color=RED, alpha=0.11, zorder=0, linewidth=0)
for key, label, color, marker, ls in rules:
    lw = 2.0 if key == "per_line" else 1.3
    ax.plot(range(len(bd)), bd[key], marker=marker, ms=4.5, lw=lw, ls=ls,
            color=color, label=label, zorder=3,
            markeredgecolor="white", markeredgewidth=0.5)

ax.set_xticks(range(len(bd)))
ax.set_xticklabels([d[5:] for d in bd.index], rotation=45, ha="right")
ax.set_xlim(-0.6, len(bd) - 0.4)
ax.set_ylabel("alarm rate (% of trips)")
ax.set_xlabel("service date in March 2026 (shaded band = known degraded telemetry)")
ax.legend(frameon=False, ncol=4, loc="upper right", handlelength=2.0)
sns.despine(ax=ax, left=True, bottom=True)
fig.savefig(f"{OUT}/fig_threshold.png", dpi=200)
plt.close(fig)

print("wrote fig_daylevel.png and fig_threshold.png")

# ── Figure: anomaly score by rain condition (weather-inclusive models) ────
import netmob_prep as prep
from scipy import stats

# OUT is <root>/full_report/figures, so the data tree sits two levels up.
WEATHER = os.environ.get(
    "NETMOB_WEATHER_CSV",
    os.path.join(os.path.dirname(os.path.dirname(OUT)),
                 "data", "auxiliar_data", "meteorological_data.csv"))
rain_utc = prep.load_rain_by_utc_hour(WEATHER)
starts = (pd.to_datetime(df["start_time"]) + pd.Timedelta(hours=3)).dt.floor("h")
df["rain_mm_trip"] = rain_utc.reindex(starts).fillna(0.0).to_numpy()
df["rainy"] = df["rain_mm_trip"] > 0

lstm = pd.read_csv("./artifacts_ae_lstm/anomaly_scores.csv").set_index("trip_instance_id")["recon_mse"]
l = np.log(lstm.reindex(df.index))
df["lstm"] = (l - l.mean()) / l.std()

# one compact panel instead of two, so the figure fits a single column and
# does not stack with the other full-width floats on the same page
long = []
for tag, name in [("rain", "Conv1D AE"), ("lstm", "LSTM AE")]:
    sub = df.dropna(subset=[tag])
    long.append(pd.DataFrame({"model": name, "score": sub[tag],
                              "cond": np.where(sub["rainy"], "rain hour", "dry hour")}))
long = pd.concat(long)

fig, ax = plt.subplots(figsize=(3.5, 2.8))
sns.boxplot(data=long, x="model", y="score", hue="cond", ax=ax,
            showfliers=False, width=0.62, gap=0.18, palette=[BLUE, ORANGE],
            linecolor=INK, linewidth=0.9)
ax.set_xlabel(""); ax.set_ylabel("anomaly score")
ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.1,
          columnspacing=1.0, fontsize=9, bbox_to_anchor=(0.5, 1.16))
sns.despine(ax=ax, left=True, bottom=True)
fig.savefig(f"{OUT}/fig_rain.png", dpi=220)
plt.close(fig)
print("wrote fig_rain.png")

# ── Figure: AE reconstruction curves per fold, both variants ─────────────
# Fold 5 holds out 23 March, the 11.8 mm/h rain day. With rainfall as an input
# channel its validation error is ~4x the other folds and never settles; drop
# the channel and it behaves like every other fold. Showing both variants makes
# that mechanism visible instead of leaving fold 5 looking like a glitch.
rc_a = pd.read_csv("./artifacts_ae_cnn/ae_reconstruction.csv")
rc_b = pd.read_csv("./artifacts_ae_cnn_norain/ae_reconstruction.csv")
folds = sorted(rc_a["fold"].unique())
fig, axes = plt.subplots(1, len(folds), figsize=(11, 2.3), sharey=True)
for ax, f in zip(np.atleast_1d(axes), folds):
    for rc, color, ls, lbl in [(rc_a, BLUE, "-", "rainfall in input"),
                               (rc_b, RED, "--", "weather-free")]:
        sub = rc[(rc["fold"] == f) & (rc["split"] == "val")].sort_values("epoch")
        ax.plot(sub["epoch"], sub["mse"], color=color, ls=ls, lw=1.6, label=lbl)
    ax.set_title(f"fold {f}", fontsize=11)
    ax.set_xlabel("epoch")
    sns.despine(ax=ax, left=True, bottom=True)
np.atleast_1d(axes)[0].set_ylabel("validation MSE")
np.atleast_1d(axes)[0].legend(frameon=False, handlelength=1.6, fontsize=9,
                              loc="upper left")
fig.savefig(f"{OUT}/fig_recon.png", dpi=220)
plt.close(fig)

# ── Figure: CNN vs LSTM anomaly score agreement ───────────────────────────
sub = df.dropna(subset=["rain", "lstm"])
tx = np.percentile(sub["rain"], 95)
ty = np.percentile(sub["lstm"], 95)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(sub["rain"], sub["lstm"], s=3.5, alpha=0.30, color=BLUE,
           linewidths=0)
ax.axvline(tx, color=RED, ls="--", lw=1.1)
ax.axhline(ty, color=RED, ls="--", lw=1.1)
ax.set_xlabel("CNN-AE anomaly score")
ax.set_ylabel("LSTM-AE anomaly score")
ax.text(0.97, 0.03, f"Pearson $r$ = {sub['rain'].corr(sub['lstm']):.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5)
sns.despine(ax=ax, left=True, bottom=True)
fig.savefig(f"{OUT}/fig_agreement.png", dpi=400)
plt.close(fig)
print("wrote fig_recon.png and fig_agreement.png")

# t-SNE figure reverted to the notebook original (artifacts_anomaly).

# ── Figure: what drives delay (E3 feature attribution) ───────────────────
# Two panels because one number dominates: permutation importance puts
# mean(progress_frac) about eight times above everything else, so a single
# linear axis would flatten the rest into invisibility. Left panel keeps that
# dominance visible; right panel gives the direction of each effect.
att = pd.read_csv("./artifacts_e3/attribution_summary.csv")
coef = pd.read_csv("./artifacts_e3/logreg_coefficients.csv")
TOPN = 12
top = att.nlargest(TOPN, "mean").iloc[::-1]          # smallest at bottom
sig = coef.groupby("feature")["p"].max() < 0.05      # significant in every fold

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"wspace": 0.42})

ax = axes[0]
ypos = np.arange(len(top))
ax.barh(ypos, top["mean"], xerr=top["std"], color=BLUE, height=0.68,
        error_kw=dict(ecolor=INK, lw=0.9, capsize=2.5), zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels(top["feature"])
ax.set_xlabel("drop in ROC-AUC when the feature is permuted")
ax.set_title("Permutation importance", fontsize=11.5)
sns.despine(ax=ax, left=True, bottom=True)

ax = axes[1]
b = top["beta"].to_numpy()
colors = [RED if v > 0 else BLUE for v in b]
ax.barh(ypos, b, color=colors, height=0.68, zorder=3)
ax.axvline(0, color=INK, lw=0.9, zorder=4)
for i, f in enumerate(top["feature"]):
    if sig.get(f, False):
        ax.plot(0, i, marker="o", ms=3.2, color=INK, zorder=5)
ax.set_yticks(ypos); ax.set_yticklabels([])
ax.set_xlabel("logistic coefficient (standardised features)")
ax.set_title("Direction of the effect", fontsize=11.5)
ax.plot([], [], marker="o", ls="", ms=3.2, color=INK,
        label="significant in every fold")
ax.bar(0, 0, color=RED, label="raises delay risk")
ax.bar(0, 0, color=BLUE, label="lowers delay risk")
ax.legend(frameon=False, fontsize=8.5, loc="lower left", handlelength=1.1)
sns.despine(ax=ax, left=True, bottom=True)

fig.savefig(f"{OUT}/fig_importance.png", dpi=220)
plt.close(fig)
print("wrote fig_importance.png")
