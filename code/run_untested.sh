#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
b(){ echo; echo "##### $* #####"; date '+%H:%M:%S'; }

b "A: rebuild with ALL gates relaxed"
.venv/bin/python -u exp_allgates_build.py 2>&1 | tail -5

b "A: raw vs latent on ALL-RELAXED data"
NETMOB_SEQ_CACHE=./cache_allrelaxed/sequences.npz .venv/bin/python -u baseline_raw.py 2>&1 | tail -9
mv -f artifacts_baseline/results.csv artifacts_allrelaxed_baseline.csv 2>/dev/null
NETMOB_SEQ_CACHE=./cache_allrelaxed/sequences.npz \
NETMOB_META_CACHE=./cache_allrelaxed/trips_meta.parquet \
NETMOB_ARTIFACT_DIR=./artifacts_ae_cnn_allrelaxed \
  .venv/bin/python -u run_notebook.py ML_autoencoder_cnn.ipynb 2>&1 | tail -8

b "B: autoencoder architecture sweep"
.venv/bin/python -u exp_ae_arch.py 2>&1 | tail -20

b "UNTESTED-POINTS RUN DONE"
