"""Complete analysis of multi-vehicle trip ids.

In process_day, GPS rows are grouped by (tripId, vehicle id). When one tripId
appears under several vehicles, the pipeline keeps only the segment with the
most points and discards the rest:

    for trip_id, veh, g in instances:
        cur = by_trip.get(trip_id)
        if cur is None or len(g) > len(cur[1]):
            by_trip[trip_id] = (veh, g)
        counters["multi_vehicle_tripids"] += int(cur is not None)

So this is not a rejection like the other gates. The trip survives; what is
discarded is every other vehicle's view of it. The counter reports superseded
INSTANCES, not lost trips, which is worth stating plainly because the report
currently lists it beside gates that do remove trips outright.

That raises questions the report never asked:

  * How often does it happen, and with how many vehicles?
  * Do the vehicles overlap in time, which would mean duplicate or reassigned
    identifiers, or do they run one after another, which would mean a genuine
    vehicle swap or relief driver mid-route?
  * How much of the route does the kept vehicle actually see compared with all
    vehicles combined? If the kept segment is a fraction of the whole, the
    trajectory fed to the model is truncated.
  * Would merging the segments instead of keeping the largest recover trips
    that currently fail the stop-coverage rule? This is the link between the
    two questions: a trip may look low-coverage only because most of its route
    was recorded under a different vehicle id.

Outputs -> artifacts_multivehicle/
"""
import os
import numpy as np
import pandas as pd

import netmob_prep as prep

OUT = "./artifacts_multivehicle"
os.makedirs(OUT, exist_ok=True)
MAX_GAP_SEC = prep.MAX_GAP_SEC


def analyse_day(date_str):
    """Recreate the (tripId, vehicle) grouping and describe the collisions."""
    path = os.path.join(prep.MOBILITY_DIR, f"{date_str}.csv")
    df = pd.read_csv(path, dtype={"lineId": "str"})
    df = df[df["lineId"].notna() & (df["lineId"].astype(str).str.strip() != "")]
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    rows = []
    for trip_id, gt in df.groupby("tripId", sort=False):
        vehs = gt["id"].unique()
        if len(vehs) < 2:
            continue
        segs = []
        for veh, g in gt.groupby("id", sort=False):
            g = g.sort_values("timestamp")
            ts = g["timestamp"].to_numpy()
            gaps = np.diff(ts).astype("timedelta64[s]").astype(float)
            seg_id = np.r_[0, np.cumsum(gaps > MAX_GAP_SEC)]
            if seg_id[-1] > 0:                    # same longest-segment rule
                sizes = np.bincount(seg_id)
                g = g.iloc[seg_id == sizes.argmax()]
            segs.append({"veh": veh, "n": len(g),
                         "t0": g["timestamp"].iloc[0], "t1": g["timestamp"].iloc[-1]})
        segs.sort(key=lambda s: -s["n"])
        kept, others = segs[0], segs[1:]

        # temporal overlap between the kept segment and the rest
        overlap = 0.0
        for o in others:
            lo = max(kept["t0"], o["t0"]); hi = min(kept["t1"], o["t1"])
            overlap = max(overlap, max(0.0, (hi - lo).total_seconds()))
        kept_span = max(1.0, (kept["t1"] - kept["t0"]).total_seconds())
        union_lo = min(s["t0"] for s in segs); union_hi = max(s["t1"] for s in segs)
        union_span = max(1.0, (union_hi - union_lo).total_seconds())

        rows.append({
            "date": date_str, "trip_id": trip_id, "n_vehicles": len(segs),
            "points_kept": kept["n"], "points_total": sum(s["n"] for s in segs),
            "kept_span_sec": kept_span, "union_span_sec": union_span,
            "span_ratio": kept_span / union_span,
            "overlap_sec": overlap,
            "overlap_frac_of_kept": overlap / kept_span,
            "line_id": str(gt["lineId"].iloc[0]),
            "hour": int(kept["t0"].hour),
        })
    return rows


def main():
    dates = prep.mobility_dates()
    print(f"scanning {len(dates)} service dates for multi-vehicle trip ids")
    allrows = []
    for d in dates:
        r = analyse_day(d)
        allrows += r
        print(f"  {d}: {len(r):>4,} multi-vehicle trip ids")
    df = pd.DataFrame(allrows)
    df.to_csv(f"{OUT}/multivehicle_trips.csv", index=False)

    counters = pd.read_csv("./cache/preprocess_counters.csv")
    cand = counters["instances_raw"].sum()
    superseded = counters["multi_vehicle_tripids"].sum()

    print("\n" + "=" * 68)
    print(f"trip ids served by more than one vehicle : {len(df):>7,}")
    print(f"superseded (tripId, vehicle) instances   : {superseded:>7,}"
          f"   {100*superseded/cand:5.2f}% of {cand:,} candidates")
    print(f"GPS points discarded with them           : "
          f"{(df.points_total - df.points_kept).sum():>7,.0f}"
          f"   {100*(df.points_total-df.points_kept).sum()/df.points_total.sum():5.2f}%"
          f" of points on these trips")
    print("=" * 68)

    print("\n=== vehicles per affected trip id ===")
    print(df.n_vehicles.value_counts().sort_index().to_string())

    print("\n=== simultaneous or sequential? ===")
    simul = df.overlap_frac_of_kept > 0.5
    partial = (df.overlap_frac_of_kept > 0.05) & ~simul
    seq = df.overlap_frac_of_kept <= 0.05
    for tag, m in [("mostly simultaneous (>50% overlap)", simul),
                   ("partial overlap (5-50%)", partial),
                   ("sequential, no real overlap (<5%)", seq)]:
        print(f"  {tag:<38} {m.sum():>6,}  {100*m.mean():5.1f}%")
    print("  simultaneous points to duplicated or reassigned vehicle ids;")
    print("  sequential points to a genuine vehicle swap mid-service.")

    print("\n=== how much of the trip does the kept vehicle see? ===")
    q = df.span_ratio.quantile([.05, .25, .5, .75, .95]).round(3)
    print(f"  span of kept segment / span of all vehicles combined")
    print(f"  p05 {q[.05]}  p25 {q[.25]}  median {q[.5]}  p75 {q[.75]}  p95 {q[.95]}")
    for thr in (0.5, 0.8):
        print(f"  kept segment covers <{thr:.0%} of the combined span : "
              f"{(df.span_ratio < thr).sum():,} trips "
              f"({100*(df.span_ratio < thr).mean():.1f}%)")

    print("\n=== concentration ===")
    top = df.line_id.value_counts().head(8)
    print("  most affected lines:")
    for line, n in top.items():
        print(f"    line {line:<8} {n:>5,}  ({100*n/len(df):4.1f}%)")
    print("  by hour of day (top 6):")
    for hr, n in df.hour.value_counts().head(6).sort_index().items():
        print(f"    {hr:02d}:00  {n:>5,}")

    # link to the stop-coverage question
    meta = pd.read_parquet("./cache/trips_meta.parquet")
    affected = set(df.trip_id.astype(str))
    meta["multi_veh"] = meta["trip_id"].astype(str).isin(affected)
    print("\n=== retained trips: coverage, multi-vehicle vs not ===")
    print(meta.groupby("multi_veh")
          .agg(trips=("stop_coverage", "size"),
               median_coverage=("stop_coverage", "median"),
               mean_coverage=("stop_coverage", "mean"),
               delayed_share=("delayed", "mean")).round(4).to_string())
    print("\nIf multi-vehicle trips show clearly lower coverage, then part of")
    print("the stop-coverage loss is caused by this deduplication rather than")
    print("by genuinely short GPS traces, and merging segments would recover it.")
    print(f"\nsaved -> {OUT}/multivehicle_trips.csv")


if __name__ == "__main__":
    main()
