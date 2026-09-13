"""Rebuild the dataset with multi-vehicle segments stitched together.

The pipeline normally keeps only the vehicle that contributed the most points
to a tripId and discards the rest, losing 71.4% of the GPS points on those
trips. NETMOB_MERGE_MULTIVEH=1 concatenates the segments in time order instead.
This rebuilds the cache that way so the models can be rerun on trajectories
that are whole rather than truncated.
"""
import os
os.environ["NETMOB_MERGE_MULTIVEH"] = "1"
import netmob_prep as prep

OUT = "./cache_merged"
assert os.path.abspath(OUT) != os.path.abspath(prep.CACHE_DIR)
gtfs = prep.build_gtfs_index(verbose=False)
rain = prep.load_rain_by_utc_hour()
meta, counters = prep.build_dataset(prep.mobility_dates(), gtfs, rain, cache_dir=OUT)
print(f"merged dataset: {len(meta):,} trips, delayed {meta.delayed.mean():.2%} -> {OUT}")
