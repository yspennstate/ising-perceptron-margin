"""Figure: the algorithmic fraction wall/alpha_* vs kappa.

One panel.  For each run (N, budget) the three margins' walls are
plotted as a fraction of the certified capacity; the flat lines are
the point.  Reads results/finite_size_N*.csv, writes
results/algorithmic_fraction.png.
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from algorithmic_fraction import ASTAR, RUNS, wall

COLORS = {'N=101, 400 sweeps/spin': '#4477aa',
          'N=201, 400 sweeps/spin': '#cc6677',
          'N=201, 1600 sweeps/spin': '#228833'}


def main():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, path in RUNS:
        rows = list(csv.DictReader(open(path, newline='')))
        ks, fs = [], []
        for kappa in (0.0, 0.05, 0.1):
            w = wall(rows, kappa)
            if w is None:
                continue
            ks.append(kappa)
            fs.append(w / ASTAR[kappa])
        c = COLORS[label]
        ax.plot(ks, fs, 'o-', color=c, lw=2, ms=7)
        mean = sum(fs) / len(fs)
        ax.axhline(mean, color=c, lw=0.8, ls=':', alpha=0.6)
        ax.annotate(label, (ks[-1], fs[-1]),
                    textcoords='offset points', xytext=(10, 0),
                    va='center', fontsize=9, color=c)
    ax.set_xlabel(r'margin $\kappa$')
    ax.set_ylabel(r'wall / $\alpha_\star(\kappa)$')
    ax.set_title('The annealing wall as a fraction of certified capacity')
    ax.set_xlim(-0.012, 0.155)
    ax.set_ylim(0.6, 1.0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', lw=0.4, alpha=0.35)
    fig.tight_layout()
    out = 'results/algorithmic_fraction.png'
    fig.savefig(out, dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    main()
