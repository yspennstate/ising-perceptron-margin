"""Figure: the general-margin Ding--Sun lower-bound condition.

Plots S_{kappa, alpha_*(kappa)}(lambda) at the six margins from
results/ds_condition_scan.json.  The condition asks S < 0 off
{0, 1}; the curves collapse nearly on top of one another, so a
single panel with an inset near lambda = 0 suffices.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'-0.4500': '#332288', '-0.0500': '#88ccee', '+0.0000': '#44aa99',
          '+0.0500': '#999933', '+0.0995': '#cc6677', '+0.1300': '#882255'}


def main():
    d = json.load(open('results/ds_condition_scan.json'))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for key in sorted(d, key=float):
        rows = d[key]['rows']
        ls, ss = zip(*rows)
        ax.plot(ls, ss, '-', lw=1.6, color=COLORS[key],
                label=r'$\kappa=%g$' % float(key))
    ax.axhline(0.0, color='0.4', lw=0.7)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$S_{\kappa,\alpha_\star(\kappa)}(\lambda)$')
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig('results/ds_condition.png', dpi=160)
    print('wrote results/ds_condition.png')


if __name__ == '__main__':
    main()
