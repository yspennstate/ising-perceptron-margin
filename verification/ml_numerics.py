"""Quantitative readings of the margin curve for the learning-theory
discussion: the storage price of robustness, binary-vs-spherical
efficiency, and the annealed gap, from the continuation CSVs.

Writes results/robustness_price.csv and prints the paper table.
"""
import csv
import math
import os

from mpmath import mp, mpf, quad, exp, log, sqrt, pi, erfc

mp.dps = 20


def phi(z):
    return exp(-z * z / 2) / sqrt(2 * pi)


def Psi(u):
    return erfc(u / sqrt(2)) / 2


def alpha_spherical(k):
    """Gardner: 1 / E[(kappa + Z)_+^2]."""
    val = quad(lambda t: (k + t) ** 2 * phi(t), [-k, 25])
    return 1 / val


def alpha_annealed(k):
    return math.log(2) / float(-log(Psi(mpf(k))))


def load_curve():
    rows = {}
    for name in ('kappa_pos.csv', 'kappa_neg.csv', 'kappa_neg2.csv'):
        p = os.path.join('results', name)
        if not os.path.exists(p):
            continue
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                rows[round(float(r['kappa']), 10)] = float(r['alpha'])
    return dict(sorted(rows.items()))


def main():
    curve = load_curve()
    out = []
    for k, a in curve.items():
        sph = float(alpha_spherical(mpf(k))) if k > -20 else float('nan')
        ann = alpha_annealed(k)
        out.append({
            'kappa': k,
            'alpha_star': a,
            'params_per_robust_label': 1 / a,
            'alpha_spherical': sph,
            'binary_over_spherical': a / sph,
            'alpha_annealed': ann,
            'alpha_over_annealed': a / ann,
        })
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'robustness_price.csv'), 'w',
              newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    # decay rate of alpha_star on the certified strip and beyond
    ks = [r['kappa'] for r in out]
    las = [math.log(r['alpha_star']) for r in out]
    sel = [(k, la) for k, la in zip(ks, las) if -0.65 <= k <= 0.19]
    n = len(sel)
    sx = sum(k for k, _ in sel)
    sy = sum(la for _, la in sel)
    sxx = sum(k * k for k, _ in sel)
    sxy = sum(k * la for k, la in sel)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    print(f'log-linear decay rate of alpha_star on the certified strip: '
          f'{slope:.4f} per unit kappa '
          f'(halving margin every {math.log(2)/abs(slope):.3f})')
    print()
    print('kappa   alpha*    1/alpha*  a*/spherical  a*/annealed')
    for r in out:
        if r['kappa'] in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.05, 0.25, 0.5,
                          1.0, 1.5):
            print(f"{r['kappa']:+.2f}  {r['alpha_star']:8.5f}  "
                  f"{r['params_per_robust_label']:8.4f}  "
                  f"{r['binary_over_spherical']:10.4f}  "
                  f"{r['alpha_over_annealed']:9.4f}")


if __name__ == '__main__':
    main()
