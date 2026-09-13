#!/bin/bash
# Does the paper's central claim (raw features >= latent representation) still
# hold when the 17% the stop rule discards is put back? Same folds, same models,
# only the dataset changes.
set -u
cd "$(dirname "$(readlink -f "$0")")"
export NETMOB_SEQ_CACHE=./cache_relaxed/sequences.npz
export NETMOB_META_CACHE=./cache_relaxed/trips_meta.parquet
b(){ echo; echo "##### $* #####"; date '+%H:%M:%S'; }
b "RAW baselines on relaxed data"
.venv/bin/python -u baseline_raw.py 2>&1 | tail -12
mv -f artifacts_baseline/results.csv artifacts_relaxed_baseline.csv 2>/dev/null
b "CNN-AE on relaxed data"
NETMOB_ARTIFACT_DIR=./artifacts_ae_cnn_relaxed .venv/bin/python -u run_notebook.py ML_autoencoder_cnn.ipynb 2>&1 | tail -10
b "RELAXED CHECK DONE"
