#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=.venv/bin/python
b(){ echo; echo "############ $* ############"; date '+%H:%M:%S'; }
b "STOP-GATE"; $PY -u exp_stopgate.py 2>&1 | tail -32
b "MULTI-VEHICLE"; $PY -u exp_multivehicle.py 2>&1 | tail -55
b "COVERAGE SENSITIVITY"; $PY -u exp_coverage_sensitivity.py 2>&1 | tail -42
b "FIXED3 DONE"
