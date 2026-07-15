"""Per-slab thin-rectangle contraction for the Nakajima lane.

For each slab of certified_intervals_nak.csv, discharge the rectangle
clause of Huang's Condition 3.1 on a thin located rectangle, the way
rect_cond31.py does at a single margin: locate the fixed point at
both alpha endpoints (mean-value Picard, depth 4000 - sufficient for
inwardness and contraction; the G signs are already certified per
slab by the lane itself), take q-hull +- delta, check endpoint
inwardness with the thin adaptive evaluators, and bound the
contraction product on the rectangle at the larger alpha.

Appends one JSON line per slab to results/rect_strip_nak.jsonl
(checkpointed - rerun resumes after the last completed slab).

Run: python rect_strip.py [--intervals results/certified_intervals_nak.csv]
"""
import argparse
import csv
import json
import os

import certify_km_slab as C
from core import dec, iv, endpoints
from locate_alpha import locate_deep_n

N_LOC = 4000
OUT = 'results/rect_strip_nak.jsonl'


N_SUB = 8  # kappa sub-balls per slab, matching the lane's own design


def one_piece(kball, alo, ahi, seed, delta):
    """The rectangle checks on one kappa sub-ball.  The sub-ball width
    (slab width / N_SUB ~ 6e-5) keeps the image spread from the kappa
    enclosure below the inward margin delta*(1-c); the full-slab ball
    (5e-4) spread it past the margin and every slab failed inwardness
    at the drift-facing edges."""
    loc_lo = locate_deep_n(alo, kball, seed, N_LOC)
    loc_hi = locate_deep_n(ahi, kball, seed, N_LOC)
    if loc_lo is None or loc_hi is None:
        return {'ok': False, 'why': 'locate failed'}
    ql1, qh1 = endpoints(loc_lo[0])
    ql2, qh2 = endpoints(loc_hi[0])
    qlo = min(ql1, ql2) - delta
    qhi = max(qh1, qh2) + delta
    qbox = iv(qlo, qhi)
    checks = {}
    for aname, a in (('lb', alo), ('ub', ahi)):
        v_lo = C.P_of(C.R_of(qlo, a, kball)) - qlo
        v_hi = qhi - C.P_of(C.R_of(qhi, a, kball))
        checks['inward_%s_lo' % aname] = float(endpoints(v_lo)[0]) > 0
        checks['inward_%s_hi' % aname] = float(endpoints(v_hi)[0]) > 0
    R_a = C.R_of_wide(qbox, alo, kball)
    R_b = C.R_of_wide(qbox, ahi, kball)
    plo = min(endpoints(R_a)[0], endpoints(R_b)[0])
    phi_ = max(endpoints(R_a)[1], endpoints(R_b)[1])
    prod = C.Pprime_bound(plo, phi_) * C.dRdq_bound(qbox, ahi, kball)
    _, prod_hi = endpoints(prod)
    checks['contraction'] = float(prod_hi) < 1.0
    return {'ok': all(checks.values()), 'q_lb': str(qlo),
            'q_ub': str(qhi), 'contraction_hi': str(prod_hi),
            'checks': checks}


def one_slab(row, delta):
    kap = dec(row['kappa_lo'])
    kap_hi = dec(row['kappa_hi'])
    alo, ahi = dec(row['alpha_lb']), dec(row['alpha_ub'])
    seed = iv(dec(row['q_lb']) - dec('0.002'),
              dec(row['q_ub']) + dec('0.002'))
    out = {'kappa_lo': row['kappa_lo'], 'kappa_hi': row['kappa_hi'],
           'alpha_lb': row['alpha_lb'], 'alpha_ub': row['alpha_ub'],
           'n_sub': N_SUB}
    step = (kap_hi - kap) / N_SUB
    pieces = []
    worst = None
    for j in range(N_SUB):
        kb = iv(kap + step * j, kap + step * (j + 1))
        pc = one_piece(kb, alo, ahi, seed, delta)
        pieces.append(pc)
        c = pc.get('contraction_hi')
        if c is not None and (worst is None or c > worst):
            worst = c
        if not pc['ok']:
            break                       # fail fast; resume rework later
    out['pieces_ok'] = sum(1 for p in pieces if p['ok'])
    out['contraction_worst'] = worst
    out['checks'] = pieces[-1].get('checks', {})
    out['ok'] = (len(pieces) == N_SUB
                 and all(p['ok'] for p in pieces))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intervals',
                    default='results/certified_intervals_nak.csv')
    ap.add_argument('--delta', default='0.002')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    delta = dec(args.delta)
    rows = list(csv.DictReader(open(args.intervals, newline='')))
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r['kappa_lo'], r['kappa_hi']))
    n_ok = n_fail = 0
    with open(OUT, 'a') as fh:
        for row in rows:
            key = (row['kappa_lo'], row['kappa_hi'])
            if key in done:
                continue
            out = one_slab(row, delta)
            fh.write(json.dumps(out) + '\n')
            fh.flush()
            n_ok += out['ok']
            n_fail += not out['ok']
            print('slab [%s, %s] %s  contraction %s'
                  % (row['kappa_lo'], row['kappa_hi'],
                     'OK  ' if out['ok'] else 'FAIL',
                     str(out.get('contraction_worst', '-'))[:24]),
                  flush=True)
    print('strip rectangles: %d ok, %d fail (this run)' % (n_ok, n_fail))


if __name__ == '__main__':
    main()
