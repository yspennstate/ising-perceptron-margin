#!/bin/bash
# Fresh-clone consumer test: run the release tree exactly as a
# reproducer would, from a cold copy.  Everything here must pass from
# the clone alone (core.py vendored, no environment variables, no
# sibling checkouts).
#
# Usage (from the clone's verification/ directory):
#   bash consumer_test.sh [python]
set -e
set -o pipefail   # a failure anywhere in a piped step must fail the test
V=${1:-python3}
echo "== selfcheck"
$V selfcheck.py
echo "== import every certificate module"
$V - <<'EOF'
import importlib
for m in ('core', 'block3bc_exact', 'km', 'certify_km_slab',
          'certify_nak', 'cert_hessian', 'hessian_kappa', 'huang2var',
          'huang_cert_np', 'huang_cert_grid', 'huang_cert_sweep',
          'huang_cert_region1', 'huang_cert_sweep2',
          'huang_cert_star_interior'):
    importlib.import_module(m)
    print('  ok', m)
EOF
echo "== past-wall replay (kappa = 0.13, ~10 s)"
$V past_wall_0p13.py | tail -3
echo "== single-slab replay (Nakajima lane, ~70 s)"
$V - <<'EOF'
import json
slabs = json.load(open('slabs_pos2_main.json'))
json.dump(slabs[:1], open('_consumer_one_slab.json', 'w'))
EOF
$V certify_nak.py _consumer_one_slab.json _consumer_one_result.json \
  | tail -2
rm -f _consumer_one_slab.json _consumer_one_result.json
echo "== portable connection checker (stdlib only)"
$V portable_check.py | tail -2
echo "== two-variable assembly verdict"
$V assemble_2varfn_0p05.py
echo "== symbolic identity battery"
$V sympy_identities.py | tail -2
echo "CONSUMER TEST PASSED"
