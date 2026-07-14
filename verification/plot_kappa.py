"""Figures for the margin continuation: the capacity curve against its
reference bounds, and the condition diagnostics along kappa.

Reads results/kappa_*.csv; writes results/alpha_curve.png and
results/conditions.png.
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

C_BLUE = '#2a78d6'
C_AQUA = '#1baf7a'
C_YELLOW = '#eda100'
C_GREEN = '#008300'
INK = '#0b0b0b'
INK2 = '#52514e'


def load_rows():
    rows = []
    for name in ('kappa_pos.csv', 'kappa_neg.csv', 'kappa_neg2.csv'):
        p = os.path.join('results', name)
        if not os.path.exists(p):
            continue
        with open(p, newline='') as fh:
            rows += [r for r in csv.DictReader(fh)]
    rows.sort(key=lambda r: float(r['kappa']))
    return rows


def alpha_c_nakajima(k):
    """Saddle-existence bound 2/(pi E[(kappa - Z)_+^2]) (Nakajima,
    arXiv 2512.23195)."""
    val = quad(lambda z: (k - z) ** 2 * norm.pdf(z), -np.inf, k)[0]
    return 2 / (np.pi * val) if val > 0 else np.inf


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)


def main():
    rows = load_rows()
    k = np.array([float(r['kappa']) for r in rows])
    a = np.array([float(r['alpha']) for r in rows])
    ann = np.array([float(r['alpha_annealed']) for r in rows])
    sph = np.array([float(r['alpha_sph']) for r in rows])
    prp = np.array([float(r['PRprime']) for r in rows])
    amp = np.array([float(r['amp_works']) for r in rows])
    lam = np.array([float(r['lambda0']) for r in rows])
    nak = np.array([alpha_c_nakajima(x) if x >= 0 else np.nan for x in k])

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(k, a, color=C_BLUE, lw=2, label='RS capacity candidate')
    ax.plot(k, ann, color=C_AQUA, lw=2, label='annealed (first moment)')
    ax.plot(k, sph, color=C_YELLOW, lw=2, label='spherical (Gardner)')
    ax.plot(k, nak, color=C_GREEN, lw=2,
            label='saddle bound (Nakajima)')
    ax.plot([0], [0.8330786], marker='o', ms=5, color=C_BLUE)
    ax.set_yscale('log')
    ax.set_xlabel(r'margin $\kappa$', color=INK)
    ax.set_ylabel(r'constraint density $\alpha$', color=INK)
    ax.set_title('Ising perceptron: RS capacity candidate at margin '
                 r'$\kappa$', color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    style_axis(ax)

    # inset: the certified strip (invisible at full scale); the two
    # uniqueness lanes (contraction / Nakajima) share one band
    crows = []
    for name in ('certified_intervals.csv', 'certified_intervals_nak.csv'):
        cert = os.path.join('results', name)
        if os.path.exists(cert):
            with open(cert, newline='') as fh:
                crows += list(csv.DictReader(fh))
    crows.sort(key=lambda r: float(r['kappa_lo']))
    if crows:
        ck = [(float(r['kappa_lo']) + float(r['kappa_hi'])) / 2
              for r in crows]
        clb = [float(r['alpha_lb']) for r in crows]
        cub = [float(r['alpha_ub']) for r in crows]
        axi = ax.inset_axes([0.08, 0.08, 0.38, 0.38])
        axi.fill_between(ck, clb, cub, color=C_BLUE, alpha=0.22, lw=0)
        m = (k >= min(ck) - 0.01) & (k <= max(ck) + 0.01)
        axi.plot(k[m], a[m], color=C_BLUE, lw=1.6)
        axi.set_xlim(min(ck) - 0.005, max(ck) + 0.005)
        axi.set_title('certified interval per slab', fontsize=7,
                      color=INK2)
        axi.tick_params(labelsize=6, colors=INK2)
        axi.grid(True, alpha=0.2, linewidth=0.5)
        for s in axi.spines.values():
            s.set_color(INK2)
            s.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(os.path.join('results', 'alpha_curve.png'))

    hess = os.path.join('results', 'hessian_kappa.csv')
    n_panels = 4 if os.path.exists(hess) else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.4),
                             dpi=150)
    axes[0].plot(k, prp, color=C_BLUE, lw=2)
    axes[0].axhline(1.0, color=INK2, lw=0.8, ls='--')
    axes[0].set_title(r"contraction $(P\circ R_\alpha)'(q_*)$  (< 1)",
                      fontsize=9, color=INK)
    axes[1].plot(k, amp, color=C_BLUE, lw=2)
    axes[1].axhline(1.0, color=INK2, lw=0.8, ls='--')
    axes[1].set_title('Condition amp-works LHS  (< 1)', fontsize=9,
                      color=INK)
    axes[2].plot(k, lam, color=C_BLUE, lw=2)
    axes[2].axhline(0.0, color=INK2, lw=0.8, ls='--')
    axes[2].set_title(r'Condition local-concavity $\lambda_0$  (< 0)',
                      fontsize=9, color=INK)
    if n_panels == 4:
        with open(hess, newline='') as fh:
            hrows = list(csv.DictReader(fh))
        hk = np.array([float(r['kappa']) for r in hrows])
        hd = np.array([float(r['det']) for r in hrows])
        axes[3].plot(hk, hd * 1e4, color=C_BLUE, lw=2)
        axes[3].axhline(0.0, color=INK2, lw=0.8, ls='--')
        axes[3].set_title(r'fixed-tilt $\det M \times 10^4$  (> 0)',
                          fontsize=9, color=INK)
    cert = os.path.join('results', 'certified_intervals.csv')
    crows = []
    for name in ('certified_intervals.csv', 'certified_intervals_nak.csv'):
        cert = os.path.join('results', name)
        if os.path.exists(cert):
            with open(cert, newline='') as fh:
                crows += list(csv.DictReader(fh))
    strip = None
    if crows:
        strip = (min(float(r['kappa_lo']) for r in crows),
                 max(float(r['kappa_hi']) for r in crows))
    for ax in axes:
        if strip:
            ax.axvspan(strip[0], strip[1], color=C_BLUE, alpha=0.08,
                       lw=0)
        ax.set_xlabel(r'$\kappa$', color=INK)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join('results', 'conditions.png'))
    print('wrote results/alpha_curve.png, results/conditions.png')


if __name__ == '__main__':
    main()
