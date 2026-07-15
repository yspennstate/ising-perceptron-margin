"""Overlay the finite-size experiment on the RS theory: success rate
of the annealer against the certified alpha*(kappa), and the mean
margin-gap of found solutions against the typical-solution RS mean.

Reads results/finite_size_N<N>.csv; writes
results/finite_size_overlay_N<N>.png.  Floats, diagnostics.

Usage: python overlay_finite_size.py [N]
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from margin_distribution import gap_density, be_polite
import km
from mpmath import mpf

C = {0.0: '#2a78d6', 0.05: '#1baf7a', 0.1: '#eda100'}
# authoritative references (kappa 0, 0.05 certified digits; 0.1 is the
# RS float at exactly 0.1 - the certified margin sits at 0.0995), used
# instead of whatever the experiment CSV recorded
ASTAR = {0.0: 0.8330786, 0.05: 0.7810743, 0.1: 0.7330167}


def rs_mean_gap(alpha, kappa, q_seed):
    fp = km.fixed_point(mpf(str(alpha)), mpf(str(kappa)), q0=q_seed)
    q = float(fp[0])
    hs = np.linspace(kappa, kappa + 6.0, 1201)
    _, mean_gap = gap_density(alpha, kappa, q, hs)
    return mean_gap, q


def main():
    be_polite()
    N = sys.argv[1] if len(sys.argv) > 1 else '101'
    rows = list(csv.DictReader(open('results/finite_size_N%s.csv' % N,
                                    newline='')))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150)
    for kappa in (0.0, 0.05, 0.1):
        sub = [r for r in rows if abs(float(r['kappa']) - kappa) < 1e-9]
        a = np.array([float(r['alpha']) for r in sub])
        s = np.array([float(r['success_rate']) for r in sub])
        astar = ASTAR[kappa]
        axes[0].plot(a, s, 'o-', color=C[kappa], lw=1.6, ms=4,
                     label=r'$\kappa=%g$' % kappa)
        axes[0].axvline(astar, color=C[kappa], lw=1.0, ls='--',
                        alpha=0.7)
        # RS mean gap along alpha, and the experimental points
        q_seed = 0.35
        exp_a, exp_g, rs_g = [], [], []
        for r in sub:
            if r['mean_gap'] == '':
                continue
            alpha = float(r['alpha'])
            mg, q_seed = rs_mean_gap(alpha, kappa, q_seed)
            exp_a.append(alpha)
            exp_g.append(float(r['mean_gap']))
            rs_g.append(mg)
        axes[1].plot(exp_a, rs_g, '-', color=C[kappa], lw=1.6,
                     label=r'RS typical, $\kappa=%g$' % kappa)
        axes[1].plot(exp_a, exp_g, 'o', color=C[kappa], ms=5)
    axes[0].set_xlabel(r'$\alpha$')
    axes[0].set_ylabel('annealer success rate')
    axes[0].set_title('where algorithms stop vs where solutions end\n'
                      '(dashed: certified/RS $\\alpha_\\star$)',
                      fontsize=9)
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel(r'mean gap above $\kappa$')
    axes[1].set_title('found solutions (dots) vs typical solutions\n'
                      '(lines): the atypical-width effect', fontsize=9)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('finite-size contact at N=%s (diagnostic, '
                 'nonrigorous)' % N, fontsize=10)
    fig.tight_layout()
    out = 'results/finite_size_overlay_N%s.png' % N
    fig.savefig(out)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
