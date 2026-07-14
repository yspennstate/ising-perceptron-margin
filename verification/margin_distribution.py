"""The replica-symmetric margin-gap distribution among solutions.

Under the RS cavity picture at (alpha, kappa) on the fixed-point branch,
the local field of a constraint on a typical solution decomposes as
sqrt(q) Z + sqrt(1-q) W with Z the quenched overlap direction and W the
fluctuation, conditioned on satisfaction (field >= kappa).  The margin
density among satisfied constraints is therefore the mixture

    p(h) = E_Z[ phi_{1-q}(h - sqrt(q) Z) 1{h >= kappa} / Psi(u(Z)) ],
    u(z) = (kappa - sqrt(q) z)/sqrt(1-q),

normalized per constraint.  This is the object that connects the storage
geometry to classical margin-type generalization bounds: it says how much
margin beyond the demanded kappa a typical solution actually carries.

Nonrigorous (float diagnostics, labeled as such); q(alpha, kappa) from
the km continuation.  Writes results/margin_gap_distribution.csv and a
figure.

Run: python margin_distribution.py
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import km
from mpmath import mpf


def be_polite():
    if sys.platform == 'win32':
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(h, 0x4000)
        except Exception:
            pass


def gap_density(alpha, kappa, q, hs):
    """p(h) for h >= kappa, and the mean gap E[h] - kappa."""
    z = np.linspace(-10, 10, 4001)
    dz = z[1] - z[0]
    wz = norm.pdf(z) * dz
    sq, s1q = np.sqrt(q), np.sqrt(1 - q)
    u = (kappa - sq * z) / s1q
    Psi_u = norm.sf(u)
    dens = np.zeros_like(hs)
    for i, h in enumerate(hs):
        if h < kappa:
            continue
        cond = norm.pdf((h - sq * z) / s1q) / s1q
        dens[i] = np.sum(wz * cond / Psi_u)
    # each conditional density integrates to one over h >= kappa by
    # construction; the numerical mass differs from 1 only by grid and
    # tail truncation, and renormalizing keeps that honest
    mass = np.trapezoid(dens, hs)
    dens = dens / mass
    mean_gap = np.trapezoid(hs * dens, hs) - kappa
    return dens, mean_gap


def main():
    be_polite()
    kappa = 0.05
    rows = []
    hs = np.linspace(kappa, kappa + 6.0, 1201)
    curves = []
    for frac in (0.5, 0.8, 0.95, 0.999):
        r = km.alpha_star(mpf(str(kappa)), conditions=False)
        a_star = float(r['alpha'])
        alpha = frac * a_star
        fp = km.fixed_point(mpf(str(alpha)), mpf(str(kappa)),
                            q0=float(r['q']))
        q = float(fp[0])
        dens, mean_gap = gap_density(alpha, kappa, q, hs)
        curves.append((frac, alpha, q, dens, mean_gap))
        print(f'alpha/alpha* = {frac:.3f}  alpha = {alpha:.5f}  '
              f'q = {q:.5f}  mean gap above kappa = {mean_gap:.4f}',
              flush=True)
        for h, d in zip(hs[::10], dens[::10]):
            rows.append({'alpha_frac': frac, 'alpha': alpha, 'q': q,
                         'h': h, 'density': d})
    os.makedirs('results', exist_ok=True)
    with open('results/margin_gap_distribution.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    colors = ['#2a78d6', '#1baf7a', '#eda100', '#b0413e']
    for (frac, alpha, q, dens, mean_gap), c in zip(curves, colors):
        ax.plot(hs, dens, color=c, lw=1.8,
                label=f'$\\alpha/\\alpha_\\star={frac:g}$ '
                      f'(mean gap {mean_gap:.2f})')
    ax.axvline(kappa, color='#52514e', lw=0.8, ls='--')
    ax.set_xlabel('constraint field $h$')
    ax.set_ylabel('density among solutions')
    ax.set_title('RS margin distribution at $\\kappa=0.05$ '
                 '(diagnostic, nonrigorous)', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig('results/margin_gap_distribution.png')
    print('wrote results/margin_gap_distribution.csv and .png')


if __name__ == '__main__':
    main()
