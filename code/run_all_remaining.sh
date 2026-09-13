#!/bin/bash
# Everything still outstanding, run strictly one after another in a single
# process. The earlier version used separate waiter scripts that polled with
# `pgrep -f exp_e8_prefix.py`; each waiter's own command line contained that
# string, so pgrep matched the waiter itself and the loop never exited. E8
# finished and the whole queue sat deadlocked. Sequencing in one script removes
# the need to poll at all.
#
# Order matters: the AE runs (E5a, E5b, E7) never overlap, because this VM has
# ~5 GB usable and concurrent training has crashed it before.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=.venv/bin/python

banner () { echo; echo "############ $* ############"; date '+%H:%M:%S'; }

banner "E5a — schedule deviation at dispatch (start_deviation_sec)"
NETMOB_EXTRA_CHANNEL=start_deviation_sec NETMOB_ARTIFACT_DIR=./artifacts_e5_startdev \
  $PY -u run_notebook.py ML_autoencoder_cnn.ipynb 2>&1 | tail -8

banner "E5b — the label source itself (median_delay_sec)"
NETMOB_EXTRA_CHANNEL=median_delay_sec NETMOB_ARTIFACT_DIR=./artifacts_e5_meddelay \
  $PY -u run_notebook.py ML_autoencoder_cnn.ipynb 2>&1 | tail -8

banner "E7 — one global AE on a historical partition"
$PY -u exp_e7_global_ae.py 2>&1 | tail -25

banner "E9 — nested hyperparameter search"
$PY -u exp_e9_hparam.py 2>&1 | tail -30

banner "STOP-GATE — decomposing the 5-stop / 50%-coverage rule"
$PY -u exp_stopgate.py 2>&1 | tail -32

banner "MULTI-VEHICLE — trips served by several vehicles"
$PY -u exp_multivehicle.py 2>&1 | tail -60

banner "COVERAGE SENSITIVITY — does relaxing the stop rule move the result?"
$PY -u exp_coverage_sensitivity.py 2>&1 | tail -45

banner "ALL REMAINING EXPERIMENTS DONE"
