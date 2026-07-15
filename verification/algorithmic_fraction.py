"""The algorithmic fraction: where the annealing wall sits as a
fraction of the certified capacity, across margin, size, and budget.

For each experiment table, the wall is the 50-percent success
crossing (linear interpolation between the bracketing alpha grid
points).  The output tests the simplest law: wall(kappa; N, budget)
= f(N, budget) * alpha_*(kappa) with f independent of kappa.

Floats, diagnostics.  Reads results/finite_size_N*.csv.
"""
import csv
import os

import numpy as np

ASTAR = {0.0: 0.8330786, 0.05: 0.7810743, 0.1: 0.7330167}
RUNS = [(label, path) for label, path in
        [('N=101, 400 sweeps/spin', 'results/finite_size_N101.csv'),
         ('N=201, 400 sweeps/spin', 'results/finite_size_N201.csv'),
         ('N=201, 1600 sweeps/spin', 'results/finite_size_N201_s1600.csv'),
         ('N=201, 6400 sweeps/spin', 'results/finite_size_N201_s6400.csv'),
         ('N=301, 400 sweeps/spin', 'results/finite_size_N301.csv')]
        if os.path.exists(path)]


def wall(rows, kappa):
    pts = sorted((float(r['alpha']), float(r['success_rate']))
                 for r in rows if abs(float(r['kappa']) - kappa) < 1e-9)
    for (a0, s0), (a1, s1) in zip(pts[:-1], pts[1:]):
        if s0 >= 0.5 > s1:
            return a0 + (a1 - a0) * (s0 - 0.5) / (s0 - s1)
    return None


def main():
    print('%-26s %8s %10s %10s %10s' %
          ('run', 'kappa', 'wall', 'alpha*', 'fraction'))
    for label, path in RUNS:
        rows = list(csv.DictReader(open(path, newline='')))
        fr = []
        for kappa in (0.0, 0.05, 0.1):
            w = wall(rows, kappa)
            if w is None:
                print('%-26s %8.2f %10s' % (label, kappa, 'n/a'))
                continue
            f = w / ASTAR[kappa]
            fr.append(f)
            print('%-26s %8.2f %10.3f %10.4f %10.3f'
                  % (label, kappa, w, ASTAR[kappa], f))
        if fr:
            print('%-26s %8s %10s %10s %10s  (spread %.3f)'
                  % ('', 'mean', '', '', '%.3f' % np.mean(fr),
                     max(fr) - min(fr)))
    print()
    print('reading: if the fraction is flat in kappa within a run,')
    print('the demanded margin costs the ALGORITHM the same relative')
    print('capacity it costs the STORAGE bound - the wall tracks the')
    print('certified curve.')


if __name__ == '__main__':
    main()
