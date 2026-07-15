"""Figure: ternary versus binary capacity and per-bit storage.

Left panel: the RS capacity curves alpha_*(kappa) for {-1,0,+1} and
{-1,+1} weights.  Right panel: the same curves per stored bit
(log2(3) bits for a ternary weight), where the order reverses.

Ternary points are read from results/ternary_curve.out (the box run
of exploration/ternary_km.py); binary points are recomputed here by
continuation with km.alpha_star.  Floats, diagnostics.
"""
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import km

LOG2_3 = math.log2(3.0)
TERN_RE = re.compile(r'kappa=([+-]?\d+\.\d+)\s+alpha\*=(\d+\.\d+)')

# the run's argv list; the log line prints kappa rounded to two
# decimals (0.0995 shows as +0.10), so pair results back to the
# requested margins by order
KAPPAS = (0.0, 0.05, 0.0995, 0.13, 0.2, 0.25, 0.3, 0.35)


def ternary_points(path='results/ternary_curve.out'):
    pts = []
    for line in open(path):
        m = TERN_RE.search(line)
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
    assert len(pts) == len(KAPPAS) and all(
        abs(shown - true) < 0.006 for (shown, _), true in zip(pts, KAPPAS))
    return [(true, a) for (_, a), true in zip(pts, KAPPAS)]


def binary_points(kappas):
    pts = []
    alpha0 = 0.8330786
    for kappa in sorted(kappas):
        a = float(km.alpha_star(kappa, alpha0=alpha0)['alpha'])
        pts.append((kappa, a))
        alpha0 = a
    return pts


def main():
    tern = ternary_points()
    bina = binary_points([k for k, _ in tern])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, per_bit in ((ax1, False), (ax2, True)):
        div_t = LOG2_3 if per_bit else 1.0
        kt, at = zip(*tern)
        kb, ab = zip(*bina)
        ax.plot(kt, [a / div_t for a in at], 'o-', color='#cc6677',
                lw=2, ms=5, label='ternary')
        ax.plot(kb, list(ab), 's-', color='#4477aa',
                lw=2, ms=5, label='binary')
        ax.set_xlabel(r'margin $\kappa$')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', lw=0.4, alpha=0.35)
    ax1.set_ylabel(r'$\alpha_\star(\kappa)$  (constraints per weight)')
    ax2.set_ylabel(r'$\alpha_\star(\kappa)/b$  (constraints per bit)')
    ax1.legend(frameon=False)
    ax1.set_title('capacity per weight')
    ax2.set_title('capacity per bit')
    fig.tight_layout()
    out = 'results/ternary_comparison.png'
    fig.savefig(out, dpi=160)
    print('wrote', out)
    print('%8s %12s %12s %10s %10s' %
          ('kappa', 'ternary', 'binary', 'tern/bit', 'bin/bit'))
    for (k, at_), (_, ab_) in zip(tern, bina):
        print('%8.4f %12.6f %12.6f %10.4f %10.4f'
              % (k, at_, ab_, at_ / LOG2_3, ab_))


if __name__ == '__main__':
    main()
