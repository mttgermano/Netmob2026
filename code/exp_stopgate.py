"""How much does the stop-coverage gate actually remove?

The label rule in the report says a trip is only trusted if it has at least
5 matched stops AND at least 50% stop coverage. Those are two separate clauses
recorded under one counter, so this re-runs the per-day pass to split them
apart and reports what each one costs.

What "50% coverage" means, from netmob_prep.process_day:

    lo, hi   = first and last observed along-route position of the GPS trace
    in_span  = scheduled stops whose distance along the shape falls in [lo, hi]
    coverage = len(in_span) / total scheduled stops on that trip

So coverage is the share of the route's scheduled stops that lie between where
the GPS trace starts and where it ends. It measures how much of the route the
trace spans, not how close the GPS points came to the stop positions. A trace
that only covers the middle of a route scores low even if its points are
perfectly accurate.

Outputs -> artifacts_stopgate/
Usage:  python exp_stopgate.py
"""
import os
import numpy as np
import pandas as pd

import netmob_prep as prep

OUT = "./artifacts_stopgate"
os.makedirs(OUT, exist_ok=True)


def main():
    gtfs = prep.build_gtfs_index(verbose=False)
    rain = prep.load_rain_by_utc_hour()
    dates = prep.mobility_dates()
    print(f"re-running the gate over {len(dates)} service dates")

    rows = []
    for d in dates:
        _, c = prep.process_day(d, gtfs, rain)
        rows.append({"date": d, **c})
        print(f"  {d}  candidates {c['instances_raw']:>5,}  "
              f"gate drops {c['low_stop_coverage']:>5,}  kept {c['kept']:>5,}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/stopgate_counters.csv", index=False)
    t = df.sum(numeric_only=True)

    cand = t["instances_raw"]
    gate = t["low_stop_coverage"]
    print("\n" + "=" * 62)
    print(f"candidate trip instances reaching the gate : {cand:>8,.0f}")
    print(f"removed by the stop rule (either clause)   : {gate:>8,.0f}"
          f"   {100*gate/cand:5.2f}% of candidates")
    print(f"  fewer than 5 matched stops only          : {t['fail_min_stops_only']:>8,.0f}"
          f"   {100*t['fail_min_stops_only']/cand:5.2f}%")
    print(f"  below 50% coverage only                  : {t['fail_coverage_only']:>8,.0f}"
          f"   {100*t['fail_coverage_only']/cand:5.2f}%")
    print(f"  failed both clauses                      : {t['fail_both']:>8,.0f}"
          f"   {100*t['fail_both']/cand:5.2f}%")
    print(f"retained overall                           : {t['kept']:>8,.0f}"
          f"   {100*t['kept']/cand:5.2f}%")
    print("=" * 62)
    print(f"\nThe stop rule is {100*gate/(cand-t['kept']):.1f}% of all rejections "
          f"({gate:,.0f} of {cand-t['kept']:,.0f} dropped trips).")

    # where the surviving trips sit relative to the threshold
    meta = pd.read_parquet("./cache/trips_meta.parquet")
    cov = meta["stop_coverage"]
    print(f"\nretained trips, stop_coverage distribution "
          f"(n={len(cov):,}, threshold 0.50):")
    print(f"  min {cov.min():.3f}  p05 {cov.quantile(.05):.3f}  "
          f"median {cov.median():.3f}  mean {cov.mean():.3f}  max {cov.max():.3f}")
    for edge in (0.55, 0.60, 0.75):
        print(f"  within [0.50, {edge:.2f}) : {((cov < edge).sum()):>6,} "
              f"({100*(cov < edge).mean():5.2f}% of retained)")
    print(f"  matched stops: median {meta['n_stops_matched'].median():.0f} "
          f"of {meta['n_stops'].median():.0f} scheduled")
    print(f"\nsaved -> {OUT}/stopgate_counters.csv")


if __name__ == "__main__":
    main()
