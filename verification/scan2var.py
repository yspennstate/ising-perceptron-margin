"""Global-condition diagnostics for S_*(l1, l2) along the kappa curve.

For each requested kappa, takes (alpha, q0, psi0) from the continuation
CSVs (results/kappa_pos.csv, kappa_neg.csv) and runs the battery:

  anchor    : S_*(1,0) (must be ~0) and s_min vs sqrt(1-q0)
  hessian   : FD Hessian of S_* at (1,0) in (l1,l2); negative definite
  clearance : moment-body clearance of the distinguished point
  grid      : coarse scan over [-L,L]^2; largest value outside a ball
              around (1,0); any positive cell is a condition violation
  rays      : S_* far along 16 rays (must stay negative)

Writes one JSON per kappa to results/ plus the grid as .npz.

Run:  python scan2var.py --kappas 0,0.05,-0.05,0.25,-0.25,0.5,-0.5
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from huang2var import STwoVar


def be_polite():
    if sys.platform == 'win32':
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(h, 0x4000)
        except Exception:
            pass


def load_curve(paths):
    rows = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='') as fh:
            for row in csv.DictReader(fh):
                rows[round(float(row['kappa']), 10)] = row
    return rows


def run_one(kappa, row, L, n, outdir):
    S = STwoVar(float(row['alpha']), float(row['q']), float(row['psi']),
                kappa)
    anchor, s_min, s_pred = S.anchor()
    H = S.hessian_center()
    evals = np.linalg.eigvalsh(H)
    clear, th_min = S.clearance()
    ls, Sv = S.grid(L=L, n=n)
    # off-center statistic outside radius 1.25 of the distinguished
    # point (the 0.4-spaced shoulder of the S = 0 peak is not a
    # competing maximizer); -inf cells are the s-unbounded directions,
    # trivially <= 0, counted separately
    L1, L2 = np.meshgrid(ls, ls, indexing='ij')
    off = np.sqrt((L1 - 1) ** 2 + L2 ** 2) > 1.25
    finite = np.isfinite(Sv)
    n_unbounded = int(np.isneginf(Sv).sum())
    n_nan = int(np.isnan(Sv).sum())
    sel = np.where(off & finite, Sv, -np.inf)
    i, j = np.unravel_index(int(np.argmax(sel)), Sv.shape)
    off_max = float(sel[i, j])
    rays = S.rays()
    ray_max = max(max(r) for _, r in rays)
    rec = {
        'kappa': kappa,
        'alpha': float(row['alpha']),
        'q0': float(row['q']),
        'psi0': float(row['psi']),
        'anchor_S10': anchor,
        's_min': s_min,
        's_pred': s_pred,
        'hessian': H.tolist(),
        'hess_eigs': evals.tolist(),
        'clearance': clear,
        'clearance_theta': th_min,
        'offcenter_max': off_max,
        'offcenter_at': [float(ls[i]), float(ls[j])],
        'n_unbounded': n_unbounded,
        'n_nan': n_nan,
        'ray_max': float(ray_max),
        'grid_L': L,
        'grid_n': n,
    }
    tag = f"{kappa:+.3f}".replace('.', 'p')
    np.savez_compressed(os.path.join(outdir, f's2v_grid_{tag}.npz'),
                        ls=ls, S=Sv)
    with open(os.path.join(outdir, f's2v_{tag}.json'), 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f"kappa={kappa:+.3f}  S(1,0)={anchor:+.2e}  "
          f"eigs=({evals[0]:+.4f},{evals[1]:+.4f})  clear={clear:.5f}  "
          f"offmax={off_max:+.5f} at ({ls[i]:+.1f},{ls[j]:+.1f})  "
          f"raymax={ray_max:+.3f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappas', default='0')
    ap.add_argument('--L', type=float, default=8.0)
    ap.add_argument('--n', type=int, default=41)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()
    be_polite()
    curve = load_curve([os.path.join('results', 'kappa_pos.csv'),
                        os.path.join('results', 'kappa_neg.csv'),
                        os.path.join('results', 'kappa_neg2.csv')])
    os.makedirs(args.outdir, exist_ok=True)
    for ks in args.kappas.split(','):
        k = round(float(ks), 10)
        if k not in curve:
            print(f"kappa={k}: no curve row yet, skipping", flush=True)
            continue
        run_one(k, curve[k], args.L, args.n, args.outdir)


if __name__ == '__main__':
    main()
