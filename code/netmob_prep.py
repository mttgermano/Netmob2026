"""
netmob_prep.py
---
Preprocessing for the NetMob 2026 challenge (Niterói bus system).

Turns raw GPS telemetry (data/mobility_data/*.csv) + GTFS schedules
(data/GTFS_data/*) + weather (data/auxiliar_data/meteorological_data.csv)
into fixed-length per-trip sequences for latent feature representation
learning, plus per-trip metadata with schedule-delay labels.

Pipeline (driven by 00_preprocess.ipynb):
  build_gtfs_index()  -> shape polylines in local meters, stop distances
                         along shape, scheduled stop times
  process_day(date)   -> clean one day of GPS, segment into trip instances,
                         map-match to GTFS shape, infer stop arrivals,
                         compute delays + 1-minute-bin feature sequences
  build_dataset()     -> loop days, add headway, write cache/sequences.npz
                         + cache/trips_meta.parquet

Coordinates: equirectangular local-meter projection around Niterói
(city is ~15 km across; error < 0.1%, so no pyproj needed).
Times: mobility timestamps are BRT (UTC-3); GTFS times are seconds after
service-day midnight (values >= 86400 = trip crosses midnight); weather is
hourly UTC (= BRT + 3 h).
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── paths / constants ──────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
MOBILITY_DIR = os.path.join(DATA_DIR, "mobility_data")
GTFS_DIR     = os.path.join(DATA_DIR, "GTFS_data")
WEATHER_CSV  = os.path.join(DATA_DIR, "auxiliar_data", "meteorological_data.csv")
CACHE_DIR    = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")

LAT0, LON0 = -22.90, -43.06                  # local projection origin (Niterói)
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))

# trip-instance segmentation / quality gates
MAX_GAP_SEC        = 30 * 60      # split instance on internal gap > 30 min
MIN_POINTS         = 10
MIN_DURATION_SEC   = 8 * 60
MAX_DURATION_SEC   = 3 * 3600
MAX_CROSS_TRACK_M  = 200.0        # drop GPS points further than this off-route
MAX_OFFROUTE_FRAC  = 0.40         # discard instance if more points dropped
MIN_STOPS_MATCHED  = 5
MIN_STOP_COVERAGE  = 0.50

# label
DELAY_THRESHOLD_SEC = 300         # median stop delay >= 5 min -> "delayed"

# sequences
BIN_SEC   = 60
T_MAX     = 64
FEATURE_COLS = [
    "speed_mps", "accel", "heading_change", "progress_frac",
    "progress_rate", "cross_track_m", "dwell_frac",
    "tod_sin", "tod_cos", "is_weekend", "rain_mm", "headway_min",
]
DWELL_SPEED_MPS = 0.5
HEADWAY_CAP_MIN = 60.0

DEGRADED_DAYS = {"2026-03-20", "2026-03-22", "2026-03-28"}


def to_local_xy(lat, lng):
    """WGS84 -> local meters (equirectangular around Niterói)."""
    x = (np.asarray(lng, dtype=np.float64) - LON0) * M_PER_DEG_LON
    y = (np.asarray(lat, dtype=np.float64) - LAT0) * M_PER_DEG_LAT
    return np.stack([x, y], axis=-1)


def day_letter(date: pd.Timestamp) -> str:
    """GTFS service-type letter encoded in tripIds: U weekday, S Sat, D Sun."""
    return {5: "S", 6: "D"}.get(date.dayofweek, "U")


# ── GTFS index ─────────────────────────────────────────────────────────────

@dataclass
class TripSched:
    """Schedule info for one trip (keyed by mobility tripId)."""
    snapshot: str
    route_id: str
    direction_id: int
    shape_xy: np.ndarray          # (M, 2) local meters
    cum_len: np.ndarray           # (M,) cumulative arc length
    stop_dist: np.ndarray         # (S,) distance of each stop along shape
    stop_sched: np.ndarray        # (S,) scheduled arrival, sec after service midnight
    n_stops: int = field(init=False)

    def __post_init__(self):
        self.n_stops = len(self.stop_dist)

    @property
    def shape_len(self):
        return float(self.cum_len[-1])

    @property
    def sched_start(self):
        return float(self.stop_sched[0])


def _parse_gtfs_seconds(series: pd.Series) -> np.ndarray:
    """'HH:MM:SS' (HH may exceed 23) -> seconds after service midnight."""
    parts = series.str.split(":", expand=True).astype(int)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()


def _project_onto_polyline(pts, verts, cum_len, chunk=512):
    """
    Project points onto a polyline (both in local meters), vectorized.
    Returns (progress_m, cross_track_m) per point.
    """
    A = verts[:-1]                                  # (M-1, 2)
    AB = verts[1:] - A
    seg_len2 = np.maximum((AB ** 2).sum(1), 1e-9)
    seg_len = np.sqrt(seg_len2)

    prog = np.empty(len(pts))
    cross = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        P = pts[i:i + chunk]                        # (n, 2)
        d = P[:, None, :] - A[None]                 # (n, M-1, 2)
        t = np.clip((d * AB[None]).sum(-1) / seg_len2, 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        dist2 = ((P[:, None, :] - proj) ** 2).sum(-1)
        j = dist2.argmin(1)
        r = np.arange(len(P))
        prog[i:i + chunk] = cum_len[j] + t[r, j] * seg_len[j]
        cross[i:i + chunk] = np.sqrt(dist2[r, j])
    return prog, cross


def build_gtfs_index(gtfs_dir: str = GTFS_DIR, verbose: bool = True) -> dict:
    """
    Returns {mobility_tripId: {snapshot_name: {direction_id: TripSched}}}.
    mobility tripId = GTFS trip_id minus trailing '_<direction>' suffix;
    the same stripped key can exist for BOTH directions, so schedules are
    kept per direction and matched against the mobility `direction` column.
    """
    index: dict[str, dict[str, dict[int, TripSched]]] = {}
    for snap_dir in sorted(glob.glob(os.path.join(gtfs_dir, "*/"))):
        snap = os.path.basename(snap_dir.rstrip("/"))
        trips = pd.read_csv(os.path.join(snap_dir, "trips.txt"))
        shapes = pd.read_csv(os.path.join(snap_dir, "shapes.txt"))
        stop_times = pd.read_csv(os.path.join(snap_dir, "stop_times.txt"))
        stops = pd.read_csv(os.path.join(snap_dir, "stops.txt"))

        # shape polylines in local meters + cumulative arc length
        shape_geo = {}
        for sid, g in shapes.sort_values("shape_pt_sequence").groupby("shape_id"):
            xy = to_local_xy(g["shape_pt_lat"].to_numpy(), g["shape_pt_lon"].to_numpy())
            keep = np.r_[True, (np.diff(xy, axis=0) ** 2).sum(1) > 1e-6]  # drop dup verts
            xy = xy[keep]
            seg = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
            shape_geo[sid] = (xy, np.r_[0.0, np.cumsum(seg)])

        stop_xy = dict(zip(stops["stop_id"].astype(str),
                           to_local_xy(stops["stop_lat"].to_numpy(),
                                       stops["stop_lon"].to_numpy())))

        st_sec = _parse_gtfs_seconds(stop_times["arrival_time"])
        stop_times = stop_times.assign(sched_sec=st_sec).sort_values(
            ["trip_id", "stop_sequence"])
        st_groups = {k: g for k, g in stop_times.groupby("trip_id")}

        n_added = 0
        for row in trips.itertuples(index=False):
            key = row.trip_id.rsplit("_", 1)[0]     # strip direction suffix
            if row.shape_id not in shape_geo or row.trip_id not in st_groups:
                continue
            xy, cum = shape_geo[row.shape_id]
            g = st_groups[row.trip_id]
            s_xy = np.array([stop_xy[s] for s in g["stop_id"].astype(str)
                             if s in stop_xy])
            in_stops = g["stop_id"].astype(str).isin(stop_xy).to_numpy()
            if in_stops.sum() < 2:
                continue
            g = g[in_stops]
            s_dist, _ = _project_onto_polyline(s_xy, xy, cum)
            # enforce monotone stop order along the shape (loops/GPS quirks)
            s_dist = np.maximum.accumulate(s_dist)
            ts = TripSched(snapshot=snap, route_id=str(row.route_id),
                           direction_id=int(row.direction_id),
                           shape_xy=xy, cum_len=cum,
                           stop_dist=s_dist,
                           stop_sched=g["sched_sec"].to_numpy().astype(float))
            index.setdefault(key, {}).setdefault(snap, {})[int(row.direction_id)] = ts
            n_added += 1
        if verbose:
            print(f"  {snap}: {n_added} trips indexed")
    if verbose:
        print(f"  total distinct trip keys: {len(index)}")
    return index


def snapshot_preference(date_str: str) -> list[str]:
    """Dated-snapshot rule: TransOceânico changed schedules on Mar 29."""
    if date_str <= "2026-03-28":
        to_order = ["TransOceanico_20260315", "TransOceanico_20260329"]
    else:
        to_order = ["TransOceanico_20260329", "TransOceanico_20260315"]
    return ["TransNit_20260320"] + to_order


def lookup_sched(index: dict, trip_key: str, date_str: str,
                 direction: int) -> TripSched | None:
    """Prefer the dated snapshot AND the schedule whose GTFS direction_id
    matches the mobility `direction` column; fall back to any direction."""
    snaps = index.get(trip_key)
    if not snaps:
        return None
    for snap in snapshot_preference(date_str):
        if snap in snaps:
            by_dir = snaps[snap]
            if direction in by_dir:
                return by_dir[direction]
    for snap in snapshot_preference(date_str):
        if snap in snaps:
            return next(iter(snaps[snap].values()))
    return None


# ── weather ────────────────────────────────────────────────────────────────

def load_rain_by_utc_hour(path: str = WEATHER_CSV) -> pd.Series:
    """Hourly rain (mm) indexed by UTC hour timestamp."""
    w = pd.read_csv(path)
    idx = pd.to_datetime(w["Timestamp (UTC)"])
    rain = pd.Series(w["Rain (mm)"].to_numpy(), index=idx).astype(float)
    return rain.ffill().fillna(0.0)


# ── per-day processing ─────────────────────────────────────────────────────

def _wrap_heading_diff(h):
    """|Δheading| wrapped to [0, 180]."""
    d = np.abs(np.diff(h))
    return np.minimum(d, 360.0 - d)


def process_day(date_str: str, gtfs_index: dict, rain_utc: pd.Series,
                mobility_dir: str = MOBILITY_DIR):
    """
    One day of GPS -> list of trip-instance dicts:
      meta fields + 'X' (L, F) float32 feature sequence (headway filled later).
    Also returns a counters dict for the verification report.
    """
    counters = dict(raw_rows=0, service_rows=0, instances_raw=0,
                    no_gtfs=0, too_few_points=0, bad_duration=0,
                    offroute_discard=0, low_stop_coverage=0,
                    multi_vehicle_tripids=0, midnight_flips=0,
                    day_letter_mismatch=0, sched_mismatch=0, kept=0)

    path = os.path.join(mobility_dir, f"{date_str}.csv")
    df = pd.read_csv(path, dtype={"lineId": "str"})
    counters["raw_rows"] = len(df)
    df = df[df["lineId"].notna() & (df["lineId"].astype(str).str.strip() != "")]
    df = df.dropna(subset=["tripId", "lat", "lng", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["id", "timestamp"]).sort_values(
        ["tripId", "id", "timestamp"])
    counters["service_rows"] = len(df)

    file_date = pd.Timestamp(date_str)

    instances = []
    for (trip_id, veh), g in df.groupby(["tripId", "id"], sort=False):
        counters["instances_raw"] += 1
        ts = g["timestamp"].to_numpy()
        # split on internal gaps > MAX_GAP_SEC, keep longest segment
        gaps = np.diff(ts).astype("timedelta64[s]").astype(float)
        seg_id = np.r_[0, np.cumsum(gaps > MAX_GAP_SEC)]
        if seg_id[-1] > 0:
            sizes = np.bincount(seg_id)
            g = g.iloc[seg_id == sizes.argmax()]
        instances.append((trip_id, veh, g))

    # multi-vehicle same tripId: keep the segment with the most points
    by_trip: dict[str, tuple] = {}
    for trip_id, veh, g in instances:
        cur = by_trip.get(trip_id)
        if cur is None or len(g) > len(cur[1]):
            by_trip[trip_id] = (veh, g)
        counters["multi_vehicle_tripids"] += int(cur is not None)

    out = []
    for trip_id, (veh, g) in by_trip.items():
        if len(g) < MIN_POINTS:
            counters["too_few_points"] += 1
            continue
        dur = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds()
        if not (MIN_DURATION_SEC <= dur <= MAX_DURATION_SEC):
            counters["bad_duration"] += 1
            continue

        direction = int(g["direction"].iloc[0])
        sched = lookup_sched(gtfs_index, trip_id, date_str, direction)
        if sched is None:
            counters["no_gtfs"] += 1
            continue

        # ── service date: pick the candidate day (matching the trip's
        # day-type letter) that minimizes |start deviation| — handles
        # trips crossing midnight in either direction ────────────────────
        trip_letter = trip_id.split("_")[1] if "_" in trip_id else "U"
        start_local = g["timestamp"].iloc[0]
        candidates = [file_date + pd.Timedelta(days=off) for off in (-1, 0, 1)
                      if day_letter(file_date + pd.Timedelta(days=off)) == trip_letter]
        if not candidates:
            counters["day_letter_mismatch"] += 1
            candidates = [file_date]
        devs = [abs((start_local - c.normalize()).total_seconds() - sched.sched_start)
                for c in candidates]
        service_date = candidates[int(np.argmin(devs))]
        if service_date != file_date:
            counters["midnight_flips"] += 1
        midnight = service_date.normalize()
        t_sec = (g["timestamp"] - midnight).dt.total_seconds().to_numpy()

        # ── map matching ────────────────────────────────────────────────
        pts = to_local_xy(g["lat"].to_numpy(), g["lng"].to_numpy())
        prog, cross = _project_onto_polyline(pts, sched.shape_xy, sched.cum_len)
        on_route = cross <= MAX_CROSS_TRACK_M
        if on_route.mean() < (1.0 - MAX_OFFROUTE_FRAC):
            counters["offroute_discard"] += 1
            continue
        t_sec_f, prog_f, cross_f = t_sec[on_route], prog[on_route], cross[on_route]
        heading_f = g["heading"].to_numpy()[on_route]
        if len(t_sec_f) < MIN_POINTS:
            counters["too_few_points"] += 1
            continue
        prog_f = np.maximum.accumulate(prog_f)      # kill backward GPS jitter

        # ── stop arrivals + delay label ────────────────────────────────
        lo, hi = prog_f[0], prog_f[-1]
        in_span = (sched.stop_dist >= lo) & (sched.stop_dist <= hi)
        n_matched = int(in_span.sum())
        coverage = n_matched / max(1, sched.n_stops)
        if n_matched < MIN_STOPS_MATCHED or coverage < MIN_STOP_COVERAGE:
            counters["low_stop_coverage"] += 1
            continue
        # first time progress crosses each stop's distance (prog_f is monotone)
        arr_sec = np.interp(sched.stop_dist[in_span], prog_f, t_sec_f)
        delays = arr_sec - sched.stop_sched[in_span]
        median_delay = float(np.median(delays))
        # |median delay| beyond 2 h means the vehicle is transmitting a
        # tripId it is not actually serving — a schedule mismatch, not a
        # delayed run; excluding keeps labels meaningful
        if abs(median_delay) > 7200:
            counters["sched_mismatch"] += 1
            continue
        label = int(median_delay >= DELAY_THRESHOLD_SEC)

        X, length = _build_sequence(t_sec_f, prog_f, cross_f, heading_f,
                                    sched.shape_len, service_date, rain_utc)

        line_id = str(g["lineId"].iloc[0])
        out.append(dict(
            trip_instance_id=f"{date_str}_{trip_id}",
            file_date=date_str, service_date=str(service_date.date()),
            trip_id=trip_id, vehicle=int(veh), line_id=line_id,
            direction=direction, gtfs_direction=sched.direction_id,
            snapshot=sched.snapshot, start_time=str(start_local),
            start_sec=float(t_sec_f[0]), duration_sec=float(dur),
            n_points=int(len(g)), n_points_onroute=int(on_route.sum()),
            n_stops=sched.n_stops, n_stops_matched=n_matched,
            stop_coverage=float(coverage),
            median_delay_sec=median_delay,
            mean_delay_sec=float(np.mean(delays)),
            final_stop_delay_sec=float(delays[-1]),
            start_deviation_sec=float(t_sec_f[0] - sched.sched_start),
            shape_len_m=sched.shape_len,
            progress_span_m=float(hi - lo),
            degraded_day=int(date_str in DEGRADED_DAYS),
            delayed=label, seq_len=length, X=X,
        ))
        counters["kept"] += 1
    return out, counters


def _build_sequence(t_sec, prog, cross, heading, shape_len,
                    service_date, rain_utc):
    """1-minute-bin feature sequence (L, F) float32; headway filled later."""
    dur = t_sec[-1] - t_sec[0]
    L = int(min(max(1, np.ceil(dur / BIN_SEC)), T_MAX))
    edges = t_sec[0] + BIN_SEC * np.arange(L + 1)
    p_edges = np.interp(edges, t_sec, prog)

    speed = np.diff(p_edges) / BIN_SEC                       # m/s
    accel = np.r_[0.0, np.diff(speed)]
    progress_frac = p_edges[1:] / max(shape_len, 1.0)
    progress_rate = np.diff(p_edges) / max(shape_len, 1.0)

    # per-raw-point quantities aggregated into bins
    bin_of = np.clip(((t_sec - t_sec[0]) // BIN_SEC).astype(int), 0, L - 1)
    hd = np.r_[0.0, _wrap_heading_diff(heading)]
    inst_speed = np.r_[0.0, np.diff(prog) / np.maximum(np.diff(t_sec), 1.0)]
    heading_change = np.zeros(L)
    cross_mean = np.zeros(L)
    dwell = np.zeros(L)
    counts = np.bincount(bin_of, minlength=L).astype(float)
    has = counts > 0
    heading_change[has] = np.bincount(bin_of, weights=hd, minlength=L)[has] / counts[has]
    cross_mean[has] = np.bincount(bin_of, weights=cross, minlength=L)[has] / counts[has]
    dwell[has] = np.bincount(bin_of, weights=(inst_speed < DWELL_SPEED_MPS).astype(float),
                             minlength=L)[has] / counts[has]
    # bins without raw points: no observed movement -> dwell if interp speed low
    dwell[~has] = (speed[~has] < DWELL_SPEED_MPS).astype(float)
    if (~has).any():                                # carry last seen residual
        idx = np.where(has, np.arange(L), -1)
        np.maximum.accumulate(idx, out=idx)
        cross_mean = np.where(idx >= 0, cross_mean[np.maximum(idx, 0)], 0.0)

    mid_sec = (edges[:-1] + edges[1:]) / 2.0        # sec after service midnight
    tod = 2 * np.pi * (mid_sec % 86400) / 86400
    is_weekend = float(service_date.dayofweek >= 5)

    mid_utc = (service_date.normalize() + pd.to_timedelta(mid_sec, unit="s")
               + pd.Timedelta(hours=3)).floor("h")
    rain = rain_utc.reindex(mid_utc).ffill().fillna(0.0).to_numpy()

    X = np.column_stack([
        speed, accel, heading_change, progress_frac, progress_rate,
        cross_mean, dwell, np.sin(tod), np.cos(tod),
        np.full(L, is_weekend), rain, np.zeros(L),   # headway_min filled later
    ]).astype(np.float32)
    return X, L


# ── dataset assembly ───────────────────────────────────────────────────────

def add_headways(records: list[dict]):
    """headway_min = minutes since previous observed trip start on the same
    (service_date, line_id, direction); first trip of the day gets the cap."""
    df = pd.DataFrame([{k: r[k] for k in
                        ("trip_instance_id", "service_date", "line_id",
                         "direction", "start_sec")} for r in records])
    df = df.sort_values(["service_date", "line_id", "direction", "start_sec"])
    prev = df.groupby(["service_date", "line_id", "direction"])["start_sec"].shift(1)
    hw = ((df["start_sec"] - prev) / 60.0).fillna(HEADWAY_CAP_MIN).clip(0, HEADWAY_CAP_MIN)
    hw_map = dict(zip(df["trip_instance_id"], hw))
    for r in records:
        h = float(hw_map[r["trip_instance_id"]])
        r["headway_min"] = h
        r["X"][:, FEATURE_COLS.index("headway_min")] = h


def build_dataset(dates: list[str], gtfs_index: dict, rain_utc: pd.Series,
                  cache_dir: str = CACHE_DIR, max_trips: int = 60_000,
                  seed: int = 42, verbose: bool = True):
    """Run all days, assemble padded arrays, write cache files.
    Returns (meta_df, counters_df)."""
    os.makedirs(cache_dir, exist_ok=True)
    all_records, counter_rows = [], []
    for d in dates:
        recs, counters = process_day(d, gtfs_index, rain_utc)
        counters["date"] = d
        counter_rows.append(counters)
        all_records.extend(recs)
        if verbose:
            print(f"  {d}: kept {counters['kept']:5d} instances "
                  f"(raw rows {counters['raw_rows']:,})")

    add_headways(all_records)

    if len(all_records) > max_trips:               # stratified by (day, line)
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"i": np.arange(len(all_records)),
                           "g": [r["service_date"] + "|" + r["line_id"]
                                 for r in all_records]})
        frac = max_trips / len(all_records)
        keep_idx = (df.groupby("g")["i"]
                    .apply(lambda s: s.sample(max(1, int(round(len(s) * frac))),
                                              random_state=seed))
                    .to_numpy())
        all_records = [all_records[i] for i in sorted(keep_idx)][:max_trips]
        print(f"  subsampled to {len(all_records)} instances")

    N = len(all_records)
    F = len(FEATURE_COLS)
    X = np.zeros((N, T_MAX, F), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int64)
    y = np.zeros(N, dtype=np.int64)
    for i, r in enumerate(all_records):
        L = r["seq_len"]
        X[i, :L] = r["X"]
        lengths[i] = L
        y[i] = r["delayed"]

    meta = pd.DataFrame([{k: v for k, v in r.items() if k != "X"}
                         for r in all_records])
    trip_ids = meta["trip_instance_id"].to_numpy()
    groups = meta["service_date"].to_numpy()

    np.savez_compressed(os.path.join(cache_dir, "sequences.npz"),
                        X=X, lengths=lengths, y=y,
                        trip_instance_id=trip_ids, groups=groups,
                        feature_cols=np.array(FEATURE_COLS))
    meta.to_parquet(os.path.join(cache_dir, "trips_meta.parquet"), index=False)
    counters_df = pd.DataFrame(counter_rows).set_index("date")
    counters_df.to_csv(os.path.join(cache_dir, "preprocess_counters.csv"))
    if verbose:
        print(f"\n  dataset: X {X.shape}, positives {y.mean():.1%}, "
              f"saved to {cache_dir}")
    return meta, counters_df


def mobility_dates(mobility_dir: str = MOBILITY_DIR) -> list[str]:
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(mobility_dir, "*.csv")))
