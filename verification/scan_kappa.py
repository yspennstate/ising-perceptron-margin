"""Continuation of the RS capacity candidate alpha_cand(kappa).

Walks a kappa range monotonically away from 0, seeding each solve from
its neighbor (the fixed point moves continuously, so warm starts keep the
iteration on the physical branch; cold solves far from a seed escape to
the frozen branch q -> 1).  Appends one CSV row per kappa as soon as it
is computed, with the full diagnostic set from km.alpha_star plus two
external anchors:

  alpha_annealed = log 2 / (-log Psi(kappa)),  the first-moment bound
                   (alpha_cand must stay strictly below it);
  alpha_sph      = 1 / E[(kappa + Z)_+^2],     the Gardner RS spherical
                   capacity, for context.

Run (two directions as separate processes if wanted):
  python scan_kappa.py --kmin 0 --kmax 1.5 --step 0.05 --out results/kappa_pos.csv
  python scan_kappa.py --kmin -1.0 --kmax -0.05 --step 0.05 --out results/kappa_neg.csv
"""

import argparse
import csv
import os
import sys
import time

from mpmath import mp, mpf, quad

import km


def be_polite():
    if sys.platform == 'win32':
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(h, 0x4000)
        except Exception:
            pass


def alpha_spherical(kappa):
    """Gardner: 1 / int_{-kappa}^inf (kappa + t)^2 phi(t) dt."""
    val = quad(lambda t: (kappa + t) ** 2 * km.phi(t), [-kappa, 20])
    return 1 / val


FIELDS = ['kappa', 'alpha', 'q', 'psi', 'gamma', 'kb', 'ElogPsi',
          'PRprime', 'amp_works', 'lambda0', 'z0',
          'fp_iters', 'newton_iters', 'dG_dq', 'dG_dpsi',
          'alpha_annealed', 'alpha_sph', 'seconds']

MP_FIELDS = ('alpha', 'q', 'psi', 'gamma', 'kb', 'ElogPsi', 'PRprime',
             'amp_works', 'lambda0', 'z0', 'dG_dq', 'dG_dpsi',
             'alpha_annealed', 'alpha_sph')


def walk(kappas, out_path, alpha0=None, q0=None):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    new = not os.path.exists(out_path)
    with open(out_path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
            fh.flush()
        for k in kappas:
            t0 = time.time()
            try:
                r = km.alpha_star(mpf(str(k)), alpha0=alpha0, q0=q0)
            except (km.BranchEscape, RuntimeError) as e:
                print(f"kappa={k:+.3f}  STOPPED: {e}", flush=True)
                break
            r['alpha_sph'] = alpha_spherical(mpf(str(k)))
            r['seconds'] = round(time.time() - t0, 1)
            alpha0, q0 = r['alpha'], r['q']
            w.writerow({f: (mp.nstr(r[f], 25) if f in MP_FIELDS
                            else r[f]) for f in FIELDS})
            fh.flush()
            print(f"kappa={k:+.3f}  alpha={mp.nstr(r['alpha'], 12)}  "
                  f"q={mp.nstr(r['q'], 8)}  PR'={mp.nstr(r['PRprime'], 4)}  "
                  f"amp={mp.nstr(r['amp_works'], 4)}  "
                  f"lam0={mp.nstr(r['lambda0'], 4)}  "
                  f"dG/dq={mp.nstr(r['dG_dq'], 2)}  ({r['seconds']}s)",
                  flush=True)
    print(f"done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmin', type=float, default=0.0)
    ap.add_argument('--kmax', type=float, default=1.5)
    ap.add_argument('--step', type=float, default=0.05)
    ap.add_argument('--dps', type=int, default=30)
    ap.add_argument('--alpha0', type=str, default=None,
                    help='seed alpha for the first point (restart aid)')
    ap.add_argument('--q0', type=str, default=None)
    ap.add_argument('--out', default=os.path.join('results',
                                                  'kappa_scan.csv'))
    args = ap.parse_args()
    be_polite()
    mp.dps = args.dps
    km.mp.dps = args.dps

    n = int(round((args.kmax - args.kmin) / args.step))
    grid = [round(args.kmin + i * args.step, 10) for i in range(n + 1)]
    if args.kmax <= 0:
        grid = sorted(grid, reverse=True)   # walk away from 0
    alpha0 = mpf(args.alpha0) if args.alpha0 else None
    q0 = mpf(args.q0) if args.q0 else None
    walk(grid, args.out, alpha0=alpha0, q0=q0)


if __name__ == '__main__':
    main()
