"""E10 Part 2 — chronic offenders: which lines are persistently late.

Pure descriptive statistics over cache/trips_meta.parquet. For every line, and
for every line x direction, we report the number of trips, the delay rate with
a Wilson 95% interval (the normal approximation is unusable at the small end
where some line-directions have only a few dozen trips), and the median of
median_delay_sec with a bootstrap 95% interval.

Two judgment calls. First, a minimum of 50 trips is required before a
line-direction is allowed into the "chronic" ranking: the raw delay rate of a
20-trip cell is noise, and ranking on it produces a leaderboard of small cells.
The cells below the threshold are still written to the CSV with a flag so that
nothing is silently dropped. Second, "chronic" is defined as a Wilson LOWER
bound above the network-wide delay rate, i.e. we only call a line chronically
late when we can be confident it is worse than average, not merely when its
point estimate is.

The concentration figure (share of all delayed trips coming from the top N
lines) is reported against the share of trips those lines contribute, because
a line with many trips will supply many delayed trips even at the base rate;
the interesting quantity is the excess.

Outputs -> artifacts_e10/
"""
import numpy as np
import pandas as pd

OUT = "./artifacts_e10"
MIN_TRIPS = 50
Z = 1.959963984540054


def wilson(k, n, z=Z):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def boot_median_ci(x, n_boot=2000, seed=42):
    x = np.asarray(x, float)
    if len(x) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.median(x[rng.integers(0, len(x), (n_boot, len(x)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(df, keys):
    rows = []
    for key, g in df.groupby(keys, sort=False):
        n, k = len(g), int(g["delayed"].sum())
        lo, hi = wilson(k, n)
        mlo, mhi = boot_median_ci(g["median_delay_sec"].values)
        rec = dict(zip(keys if isinstance(keys, list) else [keys],
                       key if isinstance(key, tuple) else (key,)))
        rec.update({"n_trips": n, "n_delayed": k, "delay_rate": k / n,
                    "wilson_lo": lo, "wilson_hi": hi,
                    "median_delay_sec": float(np.median(g["median_delay_sec"])),
                    "median_ci_lo": mlo, "median_ci_hi": mhi,
                    "p90_delay_sec": float(np.percentile(g["median_delay_sec"], 90)),
                    "n_days": g["service_date"].nunique(),
                    "enough": n >= MIN_TRIPS})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("delay_rate", ascending=False)


def main():
    df = pd.read_parquet("./cache/trips_meta.parquet")
    base = df["delayed"].mean()
    print(f"N={len(df):,} trips  {df.line_id.nunique()} lines  "
          f"{df.service_date.nunique()} days  base delay rate {base:.3%}")

    by_line = summarise(df, ["line_id"])
    by_ld = summarise(df, ["line_id", "direction"])
    by_line.to_csv(f"{OUT}/part2_by_line.csv", index=False)
    by_ld.to_csv(f"{OUT}/part2_by_line_direction.csv", index=False)

    # --- chronic: Wilson lower bound above the network base rate ----------
    chronic = by_line[by_line["enough"] & (by_line["wilson_lo"] > base)]
    print(f"\n=== chronically late lines (>={MIN_TRIPS} trips, Wilson lo > {base:.3f}) ===")
    print(chronic[["line_id", "n_trips", "delay_rate", "wilson_lo", "wilson_hi",
                   "median_delay_sec", "n_days"]].round(3).to_string(index=False))
    chronic.to_csv(f"{OUT}/part2_chronic_lines.csv", index=False)

    chronic_ld = by_ld[by_ld["enough"] & (by_ld["wilson_lo"] > base)]
    print(f"\n=== chronic line x direction (top 15 of {len(chronic_ld)}) ===")
    print(chronic_ld.head(15)[["line_id", "direction", "n_trips", "delay_rate",
                               "wilson_lo", "median_delay_sec"]].round(3).to_string(index=False))
    chronic_ld.to_csv(f"{OUT}/part2_chronic_line_direction.csv", index=False)

    # --- concentration -----------------------------------------------------
    order = by_line.sort_values("n_delayed", ascending=False)
    tot_d, tot_t = order["n_delayed"].sum(), order["n_trips"].sum()
    conc = []
    for n in [1, 3, 5, 10, 15, 20]:
        top = order.head(n)
        conc.append({"top_n_lines": n,
                     "share_of_delayed": top["n_delayed"].sum() / tot_d,
                     "share_of_trips": top["n_trips"].sum() / tot_t,
                     "lines": ",".join(top["line_id"].astype(str))})
    conc = pd.DataFrame(conc)
    conc.to_csv(f"{OUT}/part2_concentration.csv", index=False)
    print("\n=== concentration of delayed trips ===")
    print(conc.drop(columns="lines").round(3).to_string(index=False))
    for r in conc.itertuples():
        print(f"  top {r.top_n_lines:>2}: {r.lines}")

    # how many lines to cover half the delayed trips
    cum = order["n_delayed"].cumsum() / tot_d
    n_half = int((cum < 0.5).sum() + 1)
    print(f"\nlines needed to cover 50% of all delayed trips: {n_half} of {len(order)}")

    # spread between best and worst adequately-sampled line
    ok = by_line[by_line["enough"]]
    print(f"delay rate range over {len(ok)} lines with >={MIN_TRIPS} trips: "
          f"{ok.delay_rate.min():.3f} ({ok.loc[ok.delay_rate.idxmin(),'line_id']}) .. "
          f"{ok.delay_rate.max():.3f} ({ok.loc[ok.delay_rate.idxmax(),'line_id']})")

    # within-line direction asymmetry
    piv = by_ld[by_ld["enough"]].pivot_table(index="line_id", columns="direction",
                                             values="delay_rate")
    piv = piv.dropna()
    piv["gap"] = (piv[0] - piv[1]).abs()
    print(f"\n=== largest direction asymmetry ({len(piv)} lines with both directions "
          f">={MIN_TRIPS} trips) ===")
    print(piv.sort_values("gap", ascending=False).head(8).round(3).to_string())
    piv.to_csv(f"{OUT}/part2_direction_asymmetry.csv")
    print(f"\nsaved -> {OUT}/")
    return by_line


def _selftest():
    lo, hi = wilson(5, 100)
    assert 0.01 < lo < 0.05 < hi < 0.13, (lo, hi)
    assert wilson(0, 10)[0] == 0.0
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
    main()
