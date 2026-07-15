"""Histogram the finite-size found-solution gaps against the RS
typical-solution density p(h): tracking at low load, the wide-solution
bias near the algorithmic wall.

Reads results/finite_size_gaps_N<N>.csv; writes
results/gap_histograms_N<N>.png.  Floats, diagnostics.

Usage: python gap_histogram_figure.py [N]
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

PANELS_BY_N = {
    '301': [(0.0, 0.50), (0.0, 0.56), (0.05, 0.56)],
}
PANELS_DEFAULT = [(0.0, 0.50), (0.0, 0.61), (0.05, 0.67)]
C_EXP = '#2a78d6'
C_RS = '#b0413e'


def main():
    be_polite()
    N = sys.argv[1] if len(sys.argv) > 1 else '201'
    panels = PANELS_BY_N.get(N, PANELS_DEFAULT)
    rows = list(csv.DictReader(
        open('results/finite_size_gaps_N%s.csv' % N, newline='')))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels),
                                                      3.6), dpi=150)
    for ax, (kappa, alpha) in zip(axes, panels):
        gaps = np.array([float(r['gap']) for r in rows
                         if abs(float(r['kappa']) - kappa) < 1e-9
                         and abs(float(r['alpha']) - alpha) < 1e-9])
        if len(gaps) == 0:
            ax.set_title('no solutions at kappa=%g alpha=%g'
                         % (kappa, alpha), fontsize=9)
            continue
        fp = km.fixed_point(mpf(str(alpha)), mpf(str(kappa)), q0=0.4)
        q = float(fp[0])
        hs = np.linspace(kappa, kappa + 4.0, 801)
        dens, _ = gap_density(alpha, kappa, q, hs)
        ax.hist(gaps, bins=40, density=True, color=C_EXP, alpha=0.55,
                label='found (N=%s)' % N)
        ax.plot(hs - kappa, dens, color=C_RS, lw=1.8,
                label='RS typical')
        ax.set_xlim(0, 3.2)
        ax.set_xlabel(r'gap above $\kappa$')
        ax.set_title(r'$\kappa=%g,\ \alpha=%g$' % (kappa, alpha),
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('density')
    fig.suptitle('found-solution gaps vs the typical-solution law '
                 '(diagnostic)', fontsize=10)
    fig.tight_layout()
    out = 'results/gap_histograms_N%s.png' % N
    fig.savefig(out)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
