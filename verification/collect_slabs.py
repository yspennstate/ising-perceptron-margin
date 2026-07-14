"""Collect slab-certificate results into one table.

Reads result_*.json files (from run_batch.sh workers or local runs),
checks every slab passed together with its global section, and writes
results/certified_intervals.csv: one row per kappa slab with the
certified alpha_*(kappa) interval and the located fixed-point boxes.

Run:  python collect_slabs.py result_pos_*.json -o results/certified_intervals.csv
"""

import argparse
import csv
import glob
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='+')
    ap.add_argument('-o', '--out', default='results/certified_intervals.csv')
    args = ap.parse_args()

    files = []
    for p in args.patterns:
        files += glob.glob(p)
    if not files:
        print('no result files matched', file=sys.stderr)
        raise SystemExit(2)

    rows = []
    n_fail = 0
    n_globals = 0
    globals_ok = True
    for f in sorted(files):
        with open(f) as fh:
            recs = json.load(fh)
        for rec in recs:
            if 'global' in rec:
                n_globals += 1
                if not rec['global']['ok']:
                    globals_ok = False
                    print(f"{f}: GLOBAL FAIL", file=sys.stderr)
                continue
            s = rec['slab']
            if not rec['ok']:
                n_fail += 1
                bad = [c['name'] for c in rec['checks'] if not c['ok']]
                print(f"{f}: slab [{s['kappa_lo']}, {s['kappa_hi']}] "
                      f"FAIL: {bad}", file=sys.stderr)
                continue
            rows.append({
                'kappa_lo': s['kappa_lo'],
                'kappa_hi': s['kappa_hi'],
                'alpha_lb': s['alpha_lb'],
                'alpha_ub': s['alpha_ub'],
                'q_lb': s['q_lb'],
                'q_ub': s['q_ub'],
                'source': f,
            })
    rows.sort(key=lambda r: float(r['kappa_lo']))

    # continuity of the certified strip: consecutive slabs must share
    # their kappa boundary or the union is not an interval
    gaps = []
    for a, b in zip(rows[:-1], rows[1:]):
        if a['kappa_hi'] != b['kappa_lo']:
            gaps.append((a['kappa_hi'], b['kappa_lo']))

    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lo = rows[0]['kappa_lo'] if rows else '-'
    hi = rows[-1]['kappa_hi'] if rows else '-'
    print(f"{len(rows)} slabs certified, {n_fail} failed; "
          f"{n_globals} global record(s), "
          f"{'ok' if globals_ok else 'FAILED'}; "
          f"kappa strip [{lo}, {hi}]"
          + (f"; GAPS: {gaps}" if gaps else "; contiguous"))
    print(f"wrote {args.out}")
    if n_fail or not globals_ok or n_globals == 0 or gaps:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
