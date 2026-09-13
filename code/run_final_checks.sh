#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
b(){ echo; echo "##### $* #####"; date '+%H:%M:%S'; }

b "weather-free AE on RELAXED data (rain ablation)"
NETMOB_SEQ_CACHE=./cache_relaxed/sequences.npz \
NETMOB_META_CACHE=./cache_relaxed/trips_meta.parquet \
NETMOB_DROP_FEATURES=rain_mm NETMOB_ARTIFACT_DIR=./artifacts_ae_cnn_relaxed_norain \
  .venv/bin/python -u run_notebook.py ML_autoencoder_cnn.ipynb 2>&1 | tail -6

b "rebuild with multi-vehicle segments MERGED"
.venv/bin/python -u exp_merge_build.py 2>&1 | tail -6

b "raw baselines on MERGED data"
NETMOB_SEQ_CACHE=./cache_merged/sequences.npz .venv/bin/python -u baseline_raw.py 2>&1 | tail -10
mv -f artifacts_baseline/results.csv artifacts_merged_baseline.csv 2>/dev/null

b "CNN-AE on MERGED data"
NETMOB_SEQ_CACHE=./cache_merged/sequences.npz \
NETMOB_META_CACHE=./cache_merged/trips_meta.parquet \
NETMOB_ARTIFACT_DIR=./artifacts_ae_cnn_merged \
  .venv/bin/python -u run_notebook.py ML_autoencoder_cnn.ipynb 2>&1 | tail -8

b "FINAL CHECKS DONE"
