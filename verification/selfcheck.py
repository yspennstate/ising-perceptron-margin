"""Repo self-check: everything the documentation tells a reproducer
to touch must exist, parse, and compile.

Catches documentation drift of the kind found on 2026-07-14 (the
replay list named batch tags instead of the slab filenames).  Run
from the repo root; exits nonzero on any failure.
"""
import json
import os
import py_compile
import sys

SLABS = [
    'slabs_pos.json', 'slabs_neg.json', 'slabs_pos_ext.json',
    'slabs_neg_ext.json', 'slabs_neg2_ext.json', 'slabs_neg3b.json',
    'slabs_neg4.json', 'slabs_neg5.json', 'slabs_neg6.json',
    'slabs_neg7.json', 'slabs_pos2_main.json', 'slabs_pos2_tail.json',
    'slabs_pos3_a.json', 'slabs_pos3_b.json',
]
SCRIPTS = [
    'certify_km_slab.py', 'certify_nak.py', 'collect_slabs.py',
    'gen_slabs.py', 'locate_alpha.py', 'cert_hessian.py',
    'past_wall_0p13.py', 'km.py', 'scan_kappa.py', 'huang2var.py',
    'scan2var.py', 'hessian_kappa.py', 'plot_kappa.py', 'selfcheck.py',
]
RESULTS = [
    'results/certified_intervals.csv',
    'results/certified_intervals_nak.csv',
    'results/lambda_margin_by_kappa.csv',
    'results/hessian_kappa.csv',
    'results/past_wall_0p13.log',
    'results/cert_hessian_0p05.log', 'results/cert_hessian_n0p05.log',
    'results/cert_hessian_0p0995.log', 'results/cert_hessian_n0p45.log',
]


def main():
    fails = []
    for f in SLABS:
        if not os.path.exists(f):
            fails.append(f'missing slab file: {f}')
            continue
        try:
            d = json.load(open(f))
            assert d and all('kappa_lo' in s for s in d)
        except Exception as e:
            fails.append(f'{f}: {e}')
    for f in SCRIPTS:
        if not os.path.exists(f):
            fails.append(f'missing script: {f}')
            continue
        try:
            py_compile.compile(f, doraise=True)
        except Exception as e:
            fails.append(f'{f}: compile failure: {e}')
    for f in RESULTS:
        if not os.path.exists(f):
            fails.append(f'missing result: {f}')
    if fails:
        for f in fails:
            print('FAIL', f)
        raise SystemExit(1)
    print(f'selfcheck: {len(SLABS)} slab files, {len(SCRIPTS)} scripts, '
          f'{len(RESULTS)} result artifacts - all present')


if __name__ == '__main__':
    main()
