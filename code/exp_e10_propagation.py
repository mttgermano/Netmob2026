"""E10 Part 3 — does delay propagate through the network?

Three questions, each with its confound stated up front, because the honest
answer to "does delay propagate" from AVL data alone is mostly "we cannot tell
propagation from shared conditions".

(1) Consecutive trips. Within a (service_date, line, direction) block, sort by
    start time and ask whether the previous trip being delayed predicts this
    one being delayed. The naive contingency table is worthless on its own:
    trips on the same line on the same day share weather, traffic and crew, so
    a positive association is guaranteed even with no propagation whatsoever.
    We therefore report (a) the raw conditional rates, (b) a Mantel-Haenszel
    common odds ratio stratified by (line, direction, service_date), which
    removes any effect that is constant within a line-day, and (c) a
    permutation null in which the delay labels are shuffled WITHIN each
    line-day block, which preserves every line-day marginal exactly and
    destroys only the time ordering. Only (b) and (c) speak to propagation.

    We further split by vehicle continuity. When the next trip is operated by
    the same bus, a late arrival mechanically delays the next departure — that
    is genuine propagation and the one mechanism this data can actually
    identify. When it is a different bus, any remaining association is shared
    conditions, not hand-over.

(2) Headway. Delay rate and median delay against the scheduled headway already
    stored in trips_meta, plus a Spearman correlation. Short headways are a
    marker of peak service, so this is descriptive, not causal.

(3) Corridors. Stop sets per (line, direction) are read from the GTFS snapshots
    and turned into pairwise Jaccard overlaps. For every (service_date, hour)
    we compute each line-direction's delay-rate residual (observed minus its
    own month-long mean) and correlate residuals pairwise. If delay spreads
    along shared road space, corridor-sharing pairs should co-move more than
    disjoint pairs. Both groups are exposed to the same city-wide weather, so
    a positive baseline correlation is expected for ALL pairs; the test is the
    difference between the groups, by Mann-Whitney, not the level.

Outputs -> artifacts_e10/part3_*.csv
"""
import glob
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import netmob_prep

OUT = "./artifacts_e10"
MAX_GAP_MIN = 120      # beyond this the "previous trip" is not a neighbour
N_PERM = 2000
SEED = 42


def consecutive(df):
    d = df.sort_values(["service_date", "line_id", "direction", "start_sec"]).copy()
    key = ["service_date", "line_id", "direction"]
    g = d.groupby(key, sort=False)
    d["prev_delayed"] = g["delayed"].shift(1)
    d["prev_veh"] = g["vehicle"].shift(1)
    d["gap_min"] = g["start_sec"].diff() / 60.0
    d["prev_med_delay"] = g["median_delay_sec"].shift(1)
    d = d[d["prev_delayed"].notna() & (d["gap_min"] <= MAX_GAP_MIN)].copy()
    d["prev_delayed"] = d["prev_delayed"].astype(int)
    d["same_vehicle"] = (d["vehicle"] == d["prev_veh"]).astype(int)
    d["block"] = d["service_date"] + "|" + d["line_id"].astype(str) + "|" + d["direction"].astype(str)
    # finer stratum: also fix the hour, so that a traffic peak that moves
    # through the day cannot masquerade as trip-to-trip propagation.
    d["block_h"] = d["block"] + "|" + (d["start_sec"] // 3600).astype(int).astype(str)
    return d


def mh_odds(d, col="block"):
    """Mantel-Haenszel common OR of (delayed | prev_delayed), stratified by `col`."""
    num = den = 0.0
    a_s = b_s = c_s = e_s = 0.0
    for _, g in d.groupby(col, sort=False):
        a = ((g.prev_delayed == 1) & (g.delayed == 1)).sum()
        b = ((g.prev_delayed == 1) & (g.delayed == 0)).sum()
        c = ((g.prev_delayed == 0) & (g.delayed == 1)).sum()
        e = ((g.prev_delayed == 0) & (g.delayed == 0)).sum()
        n = a + b + c + e
        if n == 0 or (a + b) == 0 or (c + e) == 0 or (a + c) == 0 or (b + e) == 0:
            continue
        num += a * e / n
        den += b * c / n
        a_s += a; b_s += b; c_s += c; e_s += e
    return (num / den if den > 0 else np.nan), int(a_s + b_s + c_s + e_s)


def perm_test(d, rng, col="block"):
    """Null: shuffle `delayed` within each block, preserving block marginals."""
    obs = d.loc[d.prev_delayed == 1, "delayed"].mean() - d.loc[d.prev_delayed == 0, "delayed"].mean()
    y = d["delayed"].values.copy()
    prev = d["prev_delayed"].values
    blocks = d[col].values
    idx = {b: np.where(blocks == b)[0] for b in np.unique(blocks)}
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        yp = y.copy()
        for b, ii in idx.items():
            yp[ii] = rng.permutation(y[ii])
        null[i] = yp[prev == 1].mean() - yp[prev == 0].mean()
    p = (np.abs(null) >= abs(obs)).mean()
    return obs, null.mean(), null.std(), p


def gtfs_stop_sets():
    """(route_short_name, direction_id) -> set of stop_ids, unioned over snapshots."""
    sets = {}
    for snap in sorted(glob.glob(os.path.join(netmob_prep.DATA_DIR, "GTFS_data", "*"))):
        try:
            routes = pd.read_csv(f"{snap}/routes.txt", dtype=str)
            trips = pd.read_csv(f"{snap}/trips.txt", dtype=str)
            st = pd.read_csv(f"{snap}/stop_times.txt", dtype=str,
                             usecols=["trip_id", "stop_id"])
        except (FileNotFoundError, NotADirectoryError):
            continue
        m = (st.merge(trips[["trip_id", "route_id", "direction_id"]], on="trip_id")
               .merge(routes[["route_id", "route_short_name"]], on="route_id"))
        for (ln, dr), g in m.groupby(["route_short_name", "direction_id"]):
            k = (str(ln).strip().upper(), int(dr))
            sets.setdefault(k, set()).update(g["stop_id"].unique())
    return sets


def main():
    df = pd.read_parquet("./cache/trips_meta.parquet")
    rng = np.random.default_rng(SEED)
    base = df["delayed"].mean()

    # ---------------- (1) consecutive trips -------------------------------
    d = consecutive(df)
    print(f"pairs with a previous trip within {MAX_GAP_MIN} min: {len(d):,} "
          f"({len(d)/len(df):.1%} of trips); base rate {base:.3%}")
    rows = []
    for label, sub in [("all", d),
                       ("same vehicle", d[d.same_vehicle == 1]),
                       ("different vehicle", d[d.same_vehicle == 0])]:
        p1 = sub.loc[sub.prev_delayed == 1, "delayed"].mean()
        p0 = sub.loc[sub.prev_delayed == 0, "delayed"].mean()
        or_mh, n_mh = mh_odds(sub)
        or_h, n_h = mh_odds(sub, "block_h")
        obs, nmu, nsd, pperm = perm_test(sub, np.random.default_rng(SEED))
        obsh, nmuh, nsdh, pph = perm_test(sub, np.random.default_rng(SEED), "block_h")
        rows.append({"subset": label, "n_pairs": len(sub),
                     "n_prev_delayed": int((sub.prev_delayed == 1).sum()),
                     "p_delay_given_prev_delayed": p1,
                     "p_delay_given_prev_ontime": p0,
                     "raw_lift": p1 - p0, "raw_risk_ratio": p1 / p0 if p0 else np.nan,
                     "mh_odds_ratio": or_mh, "n_in_mh_strata": n_mh,
                     "mh_odds_ratio_hour": or_h, "n_in_mh_strata_hour": n_h,
                     "perm_hour_obs": obsh, "perm_hour_null_mean": nmuh,
                     "perm_hour_null_sd": nsdh, "perm_hour_p": pph,
                     "perm_hour_z": (obsh - nmuh) / nsdh if nsdh else np.nan,
                     "perm_obs_diff": obs, "perm_null_mean": nmu,
                     "perm_null_sd": nsd, "perm_p": pperm,
                     "perm_z": (obs - nmu) / nsd if nsd else np.nan})
        print(f"\n--- {label} (n={len(sub):,}) ---")
        print(f"  P(delay | prev delayed) = {p1:.3%}   P(delay | prev on time) = {p0:.3%}"
              f"   lift {p1-p0:+.3%}  RR {p1/p0 if p0 else float('nan'):.2f}")
        print(f"  Mantel-Haenszel OR (stratified by line x direction x day) = {or_mh:.3f}"
              f"  over {n_mh:,} trips in usable strata")
        print(f"  within-line-day permutation: obs {obs:+.4f} vs null "
              f"{nmu:+.4f}+-{nsd:.4f}  z={(obs-nmu)/nsd:+.2f}  p={pperm:.4f}")
        print(f"  adding hour to the stratum -> MH OR {or_h:.3f} over {n_h:,} trips; "
              f"permutation obs {obsh:+.4f} vs null {nmuh:+.4f}+-{nsdh:.4f}  "
              f"z={(obsh-nmuh)/nsdh if nsdh else float('nan'):+.2f}  p={pph:.4f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/part3_consecutive.csv", index=False)

    # continuous version: previous trip's median delay vs this trip's
    for label, sub in [("all", d), ("same vehicle", d[d.same_vehicle == 1]),
                       ("different vehicle", d[d.same_vehicle == 0])]:
        r, p = stats.spearmanr(sub["prev_med_delay"], sub["median_delay_sec"])
        # residualised on the line-day mean, which is the confound
        res_prev = sub["prev_med_delay"] - sub.groupby("block")["prev_med_delay"].transform("mean")
        res_cur = sub["median_delay_sec"] - sub.groupby("block")["median_delay_sec"].transform("mean")
        rr, pp = stats.spearmanr(res_prev, res_cur)
        print(f"  [{label}] spearman(prev median delay, this median delay) = {r:+.3f} "
              f"(p={p:.1e});  after removing the line-day mean: {rr:+.3f} (p={pp:.1e})")

    # ---------------- (2) headway ------------------------------------------
    h = df[df["headway_min"].notna() & (df["headway_min"] > 0)].copy()
    h["bin"] = pd.cut(h["headway_min"], [0, 5, 10, 15, 20, 30, 45, 60, 1e9],
                      labels=["<5", "5-10", "10-15", "15-20", "20-30", "30-45", "45-60", ">60"])
    hb = h.groupby("bin", observed=True).agg(
        n=("delayed", "size"), delay_rate=("delayed", "mean"),
        median_delay_sec=("median_delay_sec", "median")).reset_index()
    hb.to_csv(f"{OUT}/part3_headway.csv", index=False)
    rs, ps = stats.spearmanr(h["headway_min"], h["delayed"])
    rs2, ps2 = stats.spearmanr(h["headway_min"], h["median_delay_sec"])
    print(f"\n=== headway vs delay (n={len(h):,}) ===")
    print(hb.round(4).to_string(index=False))
    print(f"  spearman(headway, delayed)          = {rs:+.4f}  p={ps:.1e}")
    print(f"  spearman(headway, median_delay_sec) = {rs2:+.4f}  p={ps2:.1e}")

    # ---------------- (3) corridors ----------------------------------------
    sets = gtfs_stop_sets()
    df["_ln"] = df["line_id"].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    df["ld"] = df["_ln"] + "|" + df["gtfs_direction"].astype(str)
    have = {f"{a}|{b}": s for (a, b), s in sets.items()}
    df["hour"] = pd.to_datetime(df["start_time"]).dt.hour
    cell = (df.groupby(["service_date", "hour", "ld"])["delayed"]
            .agg(["mean", "size"]).reset_index())
    cell = cell[cell["size"] >= 2]
    overall = cell.groupby("ld")["mean"].transform("mean")
    cell["resid"] = cell["mean"] - overall
    piv = cell.pivot_table(index=["service_date", "hour"], columns="ld", values="resid")
    keep = [c for c in piv.columns if c in have and piv[c].notna().sum() >= 30]
    piv = piv[keep]
    print(f"\n=== corridors: {len(keep)} line-directions with GTFS stops and >=30 "
          f"day-hour cells, over {len(piv)} day-hour cells ===")
    prows = []
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            both = piv[[a, b]].dropna()
            if len(both) < 30:
                continue
            j = len(have[a] & have[b]) / max(1, len(have[a] | have[b]))
            r, p = stats.pearsonr(both[a], both[b])
            prows.append({"a": a, "b": b, "jaccard": j, "n_cells": len(both),
                          "r": r, "p": p})
    pr = pd.DataFrame(prows).dropna(subset=["r"])
    pr.to_csv(f"{OUT}/part3_corridor_pairs.csv", index=False)
    if len(pr):
        sh = pr[pr.jaccard >= 0.10]["r"]
        ds = pr[pr.jaccard == 0.0]["r"]
        u, pu = stats.mannwhitneyu(sh, ds) if len(sh) > 5 and len(ds) > 5 else (np.nan, np.nan)
        print(f"  pairs compared: {len(pr):,}   sharing >=10% of stops: {len(sh)}   "
              f"disjoint: {len(ds)}")
        print(f"  mean residual correlation, sharing pairs : {sh.mean():+.4f} "
              f"(sd {sh.std():.4f}, median {sh.median():+.4f})")
        print(f"  mean residual correlation, disjoint pairs: {ds.mean():+.4f} "
              f"(sd {ds.std():.4f}, median {ds.median():+.4f})")
        print(f"  Mann-Whitney U p = {pu:.4f}")
        rj, pj = stats.spearmanr(pr["jaccard"], pr["r"])
        print(f"  spearman(jaccard, residual correlation) = {rj:+.4f}  p={pj:.2e}")
        pd.DataFrame([{"n_pairs": len(pr), "n_sharing": len(sh), "n_disjoint": len(ds),
                       "mean_r_sharing": sh.mean(), "mean_r_disjoint": ds.mean(),
                       "mannwhitney_p": pu, "spearman_jaccard_r": rj,
                       "spearman_p": pj}]).to_csv(f"{OUT}/part3_corridor_summary.csv",
                                                  index=False)
    print(f"\nsaved -> {OUT}/")


def _selftest():
    """MH OR must be 1 when the association is purely between-strata: two blocks
    with very different delay rates but no within-block link to prev_delayed."""
    recs = []
    for b, k in [("A", 80), ("B", 10)]:          # k delayed out of 100, each arm
        for prev in (0, 1):
            recs += [{"block": b, "prev_delayed": prev, "delayed": 1}] * k
            recs += [{"block": b, "prev_delayed": prev, "delayed": 0}] * (100 - k)
    orr, n = mh_odds(pd.DataFrame(recs))
    assert abs(orr - 1.0) < 1e-9 and n == 400, (orr, n)
    print(f"selftest ok (MH OR under pure confounding = {orr:.3f})")


if __name__ == "__main__":
    _selftest()
    main()
