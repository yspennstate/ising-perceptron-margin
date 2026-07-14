"""Certified bisection of alpha_*(kappa) at a fixed kappa.

The slab certificates pin alpha_*(kappa) to intervals of width about
3.5e-3 (the alpha pads).  This driver tightens single points: starting
from a slab's [alpha_lb, alpha_ub], it bisects on the certified sign of
G at the located fixed point (the same locate_fp + G_mean_value
machinery as the slabs; G_* is strictly decreasing, so a certified sign
at the midpoint halves the interval).  A midpoint whose G enclosure
straddles zero stops the bisection - that is the honest resolution
floor of the current G evaluator (~6e-4 enclosure width, so the floor
sits near 1.5e-3 of alpha; tightening further needs finer Riemann
grids, which the -n flag exposes).

This is the prerequisite for certifying the Hessian at the
distinguished point (NOTES section 8): det M ~ 1e-4 needs (q0, psi0)
boxes ~1e-5 wide, hence alpha_* to ~1e-5.

Run:  python locate_alpha.py 0.05 [--n 8000] [--tol 2e-5]
"""

import argparse
import json
import sys

import certify_km_slab as C
from core import dec, iv, endpoints


def locate_deep_n(alpha, kappa, q_ball, n_loc, n_iter=40):
    """locate_fp_deep with explicit evaluator depth."""
    lo, hi = endpoints(q_ball)
    v_lo = C.P_of(C.R_of(lo, alpha, kappa)) - lo
    v_hi = hi - C.P_of(C.R_of(hi, alpha, kappa))
    if not (v_lo > 0 and v_hi > 0):
        return None
    qb = q_ball
    for _ in range(n_iter):
        img = C.P_of_mv(C.R_of_mv(qb, alpha, kappa, n=n_loc), n=n_loc)
        nb = C._intersect(img, qb)
        if nb is None:
            break
        lo_n, hi_n = endpoints(nb)
        lo_o, hi_o = endpoints(qb)
        if hi_n - lo_n > (hi_o - lo_o) * dec('0.98'):
            qb = nb
            break
        qb = nb
    return qb, C.R_of_mv(qb, alpha, kappa, n=n_loc)


def locate_alpha_star(kappa_str, a_lo_str, a_hi_str, q_lb_str, q_ub_str,
                      tol=2e-5, max_iter=24, gn=4000, ln=4000):
    """Bisect alpha_* at thin kappa.  Returns (a_lo, a_hi, history).
    gn: G0 quadrature cells; ln: locate-evaluator cells (the two floors
    measured in NOTES - raise both to go below ~1.4e-5)."""
    kappa = dec(kappa_str)
    a_lo, a_hi = dec(a_lo_str), dec(a_hi_str)
    q_ball = iv(dec(q_lb_str), dec(q_ub_str))
    hist = []
    for it in range(max_iter):
        width = a_hi - a_lo
        lo_w, hi_w = endpoints(width)
        if hi_w < tol:
            break
        mid = (a_lo + a_hi) / 2
        loc = locate_deep_n(mid, kappa, q_ball, ln)
        if loc is None:
            hist.append((str(mid), 'no-locate'))
            break
        qb, pb = loc
        G = C.G_mean_value_deep(mid, qb, pb, kappa, n=gn)
        g_lo, g_hi = endpoints(G)
        if g_lo > 0:
            a_lo = mid
            hist.append((str(mid), '+'))
        elif g_hi < 0:
            a_hi = mid
            hist.append((str(mid), '-'))
        else:
            hist.append((str(mid), 'straddle'))
            break
    return a_lo, a_hi, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kappa')
    ap.add_argument('--intervals', default='results/certified_intervals.csv')
    ap.add_argument('--tol', type=float, default=2e-5)
    ap.add_argument('--gn', type=int, default=4000)
    ap.add_argument('--ln', type=int, default=4000)
    args = ap.parse_args()

    import csv
    k = float(args.kappa)
    row = None
    with open(args.intervals, newline='') as fh:
        for r in csv.DictReader(fh):
            if float(r['kappa_lo']) <= k <= float(r['kappa_hi']):
                row = r
                break
    if row is None:
        print('kappa not inside the certified strip', file=sys.stderr)
        raise SystemExit(2)

    a_lo, a_hi, hist = locate_alpha_star(
        args.kappa, row['alpha_lb'], row['alpha_ub'],
        row['q_lb'], row['q_ub'], tol=args.tol, gn=args.gn, ln=args.ln)
    lo_l, _ = endpoints(a_lo)
    _, hi_h = endpoints(a_hi)
    f_lo, f_hi = float(lo_l), float(hi_h)
    print(f"kappa = {args.kappa}: alpha_* in [{f_lo:.9f}, {f_hi:.9f}]")
    print(f"width = {f_hi - f_lo:.3g}, {len(hist)} bisections "
          f"(last: {hist[-1][1] if hist else '-'})")
    out = {'kappa': args.kappa, 'alpha_lo': str(lo_l),
           'alpha_hi': str(hi_h), 'alpha_lo_f': f_lo, 'alpha_hi_f': f_hi,
           'history': hist}
    with open(f"results/alpha_star_{args.kappa.replace('.', 'p')}.json",
              'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
