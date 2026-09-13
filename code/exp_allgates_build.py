"""Rebuild with every preprocessing gate relaxed at once.

The report tests the two largest gates individually. The other five, off-route
removal, implausible duration, schedule mismatch, midnight rollover and
too-few-points, rest on the argument that each removes a trace we cannot
interpret. Relaxing all of them together is the strongest version of the test:
if the conclusions hold on the most permissive dataset the pipeline can produce,
the individual thresholds cannot be carrying them.

Thresholds are loosened, not removed, since a trip with two GPS points has no
trajectory to model at all.
"""
import os
for k, v in {"NETMOB_MIN_STOPS": "1", "NETMOB_MIN_COVERAGE": "0.0",
             "NETMOB_MIN_POINTS": "4", "NETMOB_MIN_DUR": "120",
             "NETMOB_MAX_DUR": "21600", "NETMOB_CROSS_TRACK_M": "500.0",
             "NETMOB_OFFROUTE_FRAC": "0.80"}.items():
    os.environ[k] = v
import netmob_prep as prep
OUT = "./cache_allrelaxed"
assert os.path.abspath(OUT) != os.path.abspath(prep.CACHE_DIR)
print(f"gates: pts>={prep.MIN_POINTS} dur={prep.MIN_DURATION_SEC}-{prep.MAX_DURATION_SEC}s "
      f"xtrack={prep.MAX_CROSS_TRACK_M}m offroute<{prep.MAX_OFFROUTE_FRAC} "
      f"stops>={prep.MIN_STOPS_MATCHED} cov>={prep.MIN_STOP_COVERAGE}")
gtfs = prep.build_gtfs_index(verbose=False)
meta, _ = prep.build_dataset(prep.mobility_dates(), gtfs,
                             prep.load_rain_by_utc_hour(), cache_dir=OUT)
print(f"all-relaxed dataset: {len(meta):,} trips, delayed {meta.delayed.mean():.2%}")
