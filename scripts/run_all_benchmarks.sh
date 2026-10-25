#!/bin/sh
# Runs every benchmark campaign sequentially (about 2-3 hours on the reported machine). Resumable.
# Usage: sh scripts/run_all_benchmarks.sh [BS]
BS=${1:-96}
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python3}
$PY scripts/env_info.py > /dev/null
$PY scripts/run_benchmarks.py --campaign bsweep --resume
$PY scripts/run_benchmarks.py --campaign bsweep --tag 2 --resume
$PY scripts/run_benchmarks.py --campaign dense --bs "$BS" --resume
$PY scripts/run_benchmarks.py --campaign flags --bs "$BS" --resume
for k in 1 2 3 4 5; do $PY scripts/run_benchmarks.py --campaign procvar --bs "$BS" --tag $k --resume; done
$PY scripts/run_benchmarks.py --campaign sparse --bs "$BS" --resume
$PY scripts/run_benchmarks.py --campaign scale --resume
echo ALL_CAMPAIGNS_FINISHED
