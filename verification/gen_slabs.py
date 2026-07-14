"""Generate kappa sub-slab rectangles for certify_km_slab.py.

Walks kappa from 0 to kmax in steps of the slab width, solving the RS
system at each boundary (dps 20 is plenty: the pads are ~1e-3), and
emits one padded rectangle per sub-slab.  The pads must cover the
fixed-point drift across the sub-slab plus the interval-arithmetic
spread; the certificate itself is the proof, so a too-tight seed simply
fails and gets regenerated wider.

Run:  python gen_slabs.py --kmax 0.05 --width 0.001 --out slabs.json
      python gen_slabs.py --kmax -0.05 --width 0.001 --out slabs_neg.json
"""

import argparse
import json
import sys

from mpmath import mp, mpf

import km


def be_polite():
    if sys.platform == 'win32':
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(h, 0x4000)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmax', type=float, default=0.05)
    ap.add_argument('--kstart', type=float, default=0.0,
                    help='start the walk here instead of 0 (same sign '
                         'as kmax); pass --alpha0/--q0 seeds')
    ap.add_argument('--alpha0', type=str, default=None)
    ap.add_argument('--q0', type=str, default=None)
    ap.add_argument('--width', type=float, default=0.001)
    ap.add_argument('--pad-alpha', type=float, default=2.5e-3)
    ap.add_argument('--pad-q', type=float, default=5e-4)
    ap.add_argument('--pad-psi', type=float, default=5e-2)
    ap.add_argument('--dps', type=int, default=20)
    ap.add_argument('--out', default='slabs.json')
    args = ap.parse_args()
    be_polite()
    mp.dps = args.dps
    km.mp.dps = args.dps

    sign = 1 if args.kmax >= 0 else -1
    n0 = int(round(abs(args.kstart) / args.width))
    n = int(round(abs(args.kmax) / args.width))
    bounds = [sign * i * args.width for i in range(n0, n + 1)]

    pts = []
    alpha0 = mpf(args.alpha0) if args.alpha0 else None
    q0 = mpf(args.q0) if args.q0 else None
    for k in bounds:
        r = km.alpha_star(mpf(str(k)), alpha0=alpha0, q0=q0,
                          conditions=False)
        alpha0, q0 = r['alpha'], r['q']
        pts.append((k, r['alpha'], r['q'], r['psi']))
        print(f"kappa={k:+.4f} alpha={mp.nstr(r['alpha'], 12)} "
              f"q={mp.nstr(r['q'], 10)} psi={mp.nstr(r['psi'], 10)}",
              flush=True)

    slabs = []
    for (k0, a0, qq0, p0), (k1, a1, qq1, p1) in zip(pts[:-1], pts[1:]):
        lo_k, hi_k = sorted((k0, k1))
        slabs.append({
            'kappa_lo': f"{lo_k:.6f}",
            'kappa_hi': f"{hi_k:.6f}",
            'alpha_lb': mp.nstr(min(a0, a1) - mpf(str(args.pad_alpha)), 15),
            'alpha_ub': mp.nstr(max(a0, a1) + mpf(str(args.pad_alpha)), 15),
            'q_lb': mp.nstr(min(qq0, qq1) - mpf(str(args.pad_q)), 15),
            'q_ub': mp.nstr(max(qq0, qq1) + mpf(str(args.pad_q)), 15),
            'psi_lb': mp.nstr(min(p0, p1) - mpf(str(args.pad_psi)), 15),
            'psi_ub': mp.nstr(max(p0, p1) + mpf(str(args.pad_psi)), 15),
        })
    with open(args.out, 'w') as fh:
        json.dump(slabs, fh, indent=1)
    print(f"wrote {len(slabs)} slabs -> {args.out}")


if __name__ == '__main__':
    main()
