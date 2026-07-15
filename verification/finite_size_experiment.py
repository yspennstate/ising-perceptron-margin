"""Finite-size contact for the margin theory: train +-1-weight
perceptrons on Gaussian patterns at demanded margin kappa and compare
against the certified curve.

Three measurements per (kappa, alpha):
  - success rate of simulated annealing (population of chains, hinge
    energy on violated margins): where ALGORITHMS stop finding
    solutions, against where solutions provably end (alpha*(kappa));
  - the empirical margin-gap distribution of found solutions, against
    the RS conditional law p(h) (Section 9 of NOTES);
  - the mean gap above kappa, against the RS mean.

The known physics makes the honest reading explicit: heuristics stall
below alpha* (frozen-1RSB clustering), and found solutions are
atypical - wider margins than the typical-solution law.  Both effects
are part of the ML story, not artifacts.

Floats, diagnostics; the certified objects remain the alpha* bounds.
Run: python finite_size_experiment.py [N] [seed]
"""
import csv
import os
import sys
import time

import numpy as np


def anneal_population(xi, kappa, n_steps, n_chains, rng, beta0=1.0,
                      beta1=60.0):
    # the hinge energy moves in ~2/sqrt(N) units per active constraint,
    # so beta must reach O(sqrt(N)) for the endgame to be greedy; a
    # ramp topping out at O(1) leaves chains warm and success rates
    # collapse far below the known algorithmic threshold
    """Parallel simulated annealing on E = sum_mu max(0, kappa - h_mu)
    with h = W xi^T / sqrt(N).  Single-spin-flip Metropolis, linear
    beta ramp, incremental field updates.  Returns (W, success) per
    chain."""
    M, N = xi.shape
    W = rng.choice(np.array([-1.0, 1.0]), size=(n_chains, N))
    fields = (W @ xi.T) / np.sqrt(N)              # n_chains x M
    xi_scaled = 2.0 * xi.T / np.sqrt(N)           # N x M
    gaps = kappa - fields
    energy = np.maximum(0.0, gaps).sum(axis=1)
    betas = beta0 * (beta1 / beta0) ** (np.arange(n_steps) / n_steps)
    chain_idx = np.arange(n_chains)
    for step in range(n_steps):
        j = rng.integers(0, N)
        # flipping W[:, j]: h -> h - 2 W[:, j] xi[:, j] / sqrt(N)
        delta_f = -W[:, j, None] * xi_scaled[j][None, :]
        new_fields = fields + delta_f
        new_energy = np.maximum(0.0, kappa - new_fields).sum(axis=1)
        dE = new_energy - energy
        accept = (dE <= 0) | (rng.random(n_chains)
                              < np.exp(-betas[step] * np.minimum(dE, 50)))
        W[accept, j] *= -1.0
        fields[accept] = new_fields[accept]
        energy[accept] = new_energy[accept]
        # stop at the first solve: later steps could walk a solved
        # chain off its solution while the ramp is still warm
        if step % 250 == 0 and (energy == 0.0).any():
            break
    success = energy == 0.0
    return W, fields, success


def run_point(N, alpha, kappa, rng, n_samples=6, n_chains=24,
              steps_per_spin=400):
    """Fraction of disorder samples where ANY chain finds a solution,
    plus the pooled margin gaps of found solutions."""
    M = max(2, int(round(alpha * N)))
    n_steps = steps_per_spin * N
    hits = 0
    gaps = []
    for _ in range(n_samples):
        xi = rng.standard_normal((M, N))
        _, fields, success = anneal_population(xi, kappa, n_steps,
                                               n_chains, rng)
        if success.any():
            hits += 1
            k = int(np.flatnonzero(success)[0])
            gaps.extend((fields[k] - kappa).tolist())
    return hits / n_samples, gaps


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260715
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    tag = '%d' % N if steps == 400 else '%d_s%d' % (N, steps)
    rng = np.random.default_rng(seed)
    # reference points: kappa 0 and 0.05 are certified digits; 0.1 is
    # the RS float at exactly 0.1 (the certified margin sits at 0.0995)
    alpha_star = {0.0: 0.8330786, 0.05: 0.7810743, 0.1: 0.7330167}
    rows = []
    gap_rows = []
    t0 = time.time()
    for kappa in (0.0, 0.05, 0.1):
        for alpha in (0.40, 0.50, 0.55, 0.61, 0.67, 0.73, 0.79):
            rate, gaps = run_point(N, alpha, kappa, rng, steps_per_spin=steps)
            gap_rows.extend((kappa, alpha, round(float(g), 5))
                            for g in gaps)
            g = np.array(gaps)
            rows.append(dict(N=N, kappa=kappa, alpha=alpha,
                             success_rate=round(rate, 3),
                             n_gaps=len(gaps),
                             mean_gap=(round(float(g.mean()), 4)
                                       if len(gaps) else ''),
                             alpha_star=alpha_star[kappa]))
            print('N=%d kappa=%.2f alpha=%.2f  success=%.2f  '
                  'mean_gap=%s  (%.0fs)'
                  % (N, kappa, alpha, rate,
                     rows[-1]['mean_gap'], time.time() - t0),
                  flush=True)
    os.makedirs('results', exist_ok=True)
    out = 'results/finite_size_N%s.csv' % tag
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    gout = 'results/finite_size_gaps_N%s.csv' % tag
    with open(gout, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['kappa', 'alpha', 'gap'])
        w.writerows(gap_rows)
    print('wrote %s and %s' % (out, gout), flush=True)


if __name__ == '__main__':
    main()
