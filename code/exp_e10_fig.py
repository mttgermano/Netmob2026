"""E10 figure — the two Part-2/Part-3 results worth a picture.

Left: delay rate with Wilson 95% intervals for the lines that are chronically
worse than the network base rate, split by direction, which makes the point
that the unit of the problem is the line-direction and not the line.
Right: delay rate against scheduled headway, which is U-shaped — the worst
service is both the most frequent (<5 min, bunching) and the least frequent
(45-60 min).

Styling copied verbatim from make_report_figs.py so the figure set stays one
family.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from exp_e10_chronic import wilson

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
BLUE, RED, GREEN = "#275DBD", "#D91A26", "#1D9E75"
INK = "#2A2A2A"

sns.set_theme(style="whitegrid", font="Glacial Indifference")
plt.rcParams.update({
    "font.family": "Glacial Indifference", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "normal",
    "grid.color": "#D6D6D6", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

df = pd.read_parquet("./cache/trips_meta.parquet")
base = df["delayed"].mean()
ld = pd.read_csv("./artifacts_e10/part2_by_line_direction.csv")
ld = ld[ld["enough"]]
worst = ld.sort_values("delay_rate", ascending=False).head(14)
worst = worst.iloc[::-1]
lbl = [f"{r.line_id} dir {int(r.direction)}" for r in worst.itertuples()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0),
                               gridspec_kw={"width_ratios": [1.25, 1]})

ypos = np.arange(len(worst))
err = np.vstack([worst["delay_rate"] - worst["wilson_lo"],
                 worst["wilson_hi"] - worst["delay_rate"]])
cols = [RED if d == 1 else BLUE for d in worst["direction"]]
ax1.barh(ypos, worst["delay_rate"], color=cols, height=0.68, zorder=2)
ax1.errorbar(worst["delay_rate"], ypos, xerr=err, fmt="none",
             ecolor=INK, elinewidth=1.1, capsize=2.5, zorder=3)
ax1.axvline(base, color=GREEN, lw=1.6, ls="--", zorder=4,
            label=f"network base rate {base:.1%}")
ax1.set_yticks(ypos); ax1.set_yticklabels(lbl)
ax1.set_xlabel("delay rate (Wilson 95% CI)")
ax1.set_title("Chronically late line-directions")
ax1.set_xlim(0, 0.92)
h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)]
ax1.legend(h + ax1.get_legend_handles_labels()[0],
           ["direction 0", "direction 1"] + ax1.get_legend_handles_labels()[1],
           loc="lower right", frameon=True)

hb = pd.read_csv("./artifacts_e10/part3_headway.csv")
x = np.arange(len(hb))
lo, hi = zip(*[wilson(int(round(r.delay_rate * r.n)), int(r.n)) for r in hb.itertuples()])
ax2.errorbar(x, hb["delay_rate"], yerr=[hb["delay_rate"] - np.array(lo),
                                        np.array(hi) - hb["delay_rate"]],
             fmt="o-", color=BLUE, ecolor=INK, elinewidth=1.0, capsize=2.5,
             ms=6, lw=1.8, zorder=3)
ax2.axhline(base, color=GREEN, lw=1.6, ls="--", zorder=2,
            label=f"network base rate {base:.1%}")
for xi, r in zip(x, hb.itertuples()):
    ax2.annotate(f"n={r.n:,}", (xi, r.delay_rate), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=8.5, color=INK,
                 bbox=dict(fc="white", ec="none", pad=1.0), zorder=4)
ax2.set_xticks(x); ax2.set_xticklabels(hb["bin"])
ax2.set_xlabel("scheduled headway (min)")
ax2.set_ylabel("delay rate")
ax2.set_ylim(0.09, 0.40)
ax2.set_title("Delay rate is U-shaped in headway")
ax2.legend(loc="upper center", frameon=True)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_e10_chronic.png", dpi=220, bbox_inches="tight")
print(f"saved -> {OUT}/fig_e10_chronic.png")
