#!/bin/bash
# Slab-certificate batch runner.  Splits the slab list into 4 strided
# chunks (each spans the whole kappa range, so every region reports
# early) and runs them on niced workers.
#
# Adapt the three variables below to the worker host.  Concurrency
# drops to 2 when the health file (if present) reports an active
# interactive user or a degraded state; the full 4-worker slice plus an
# active user tripped a load sentinel three times during the first
# strip build, and one batch at a time is enforced for the same reason.
#
# Usage: bash run_batch.sh slabs_pos.json pos [certify_nak.py]
set -e
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}
VENV=${VENV:-python3}
HEALTH=${HEALTH:-/var/local/compute_health.json}
cd "$WORKDIR"
SLABS=${1:-slabs_pos.json}
TAG=${2:-pos}
SCRIPT=${3:-certify_km_slab.py}
# One batch at a time.
if pgrep -f "certify_.*.py chunk_" > /dev/null 2>&1; then
  echo "refusing: certify workers already running (one batch at a time)"
  exit 3
fi
NW=4
if [ -r "$HEALTH" ]; then
  if grep -q '"owner_rdp_active": true' "$HEALTH" 2>/dev/null; then NW=2; fi
  if ! grep -q '"status": "ok"' "$HEALTH" 2>/dev/null; then NW=2; fi
fi
$VENV - "$SLABS" "$TAG" <<'EOF'
import json, sys
slabs = json.load(open(sys.argv[1]))
n = 4
for i in range(n):
    json.dump(slabs[i::n], open(f'chunk_{sys.argv[2]}_{i}.json', 'w'))
print(f'{len(slabs)} slabs -> 4 chunks')
EOF
launch() {
  nohup nice -n 12 $VENV $SCRIPT \
    chunk_${TAG}_$1.json result_${TAG}_$1.json \
    > chunk_${TAG}_$1.log 2>&1 &
}
if [ "$NW" = "4" ]; then
  for i in 0 1 2 3; do launch $i; done
  echo "launched 4 workers for $SLABS"
else
  launch 0; launch 1
  export TAG VENV SCRIPT
  nohup bash -c 'while pgrep -f "$SCRIPT chunk_${TAG}_0.json" > /dev/null 2>&1 || pgrep -f "$SCRIPT chunk_${TAG}_1.json" > /dev/null 2>&1; do sleep 30; done
    nice -n 12 $VENV $SCRIPT chunk_${TAG}_2.json result_${TAG}_2.json > chunk_${TAG}_2.log 2>&1 &
    nice -n 12 $VENV $SCRIPT chunk_${TAG}_3.json result_${TAG}_3.json > chunk_${TAG}_3.log 2>&1 &
    wait' > /dev/null 2>&1 &
  echo "launched 2 workers for $SLABS (interactive user or degraded host; chunks 2,3 follow)"
fi
