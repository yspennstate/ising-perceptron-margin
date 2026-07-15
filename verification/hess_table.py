"""Emit the certified Hessian enclosure table from the endpoint runs.

Reads results/hess_endpoints_<tag>.out (the finish() endpoint lines)
and prints LaTeX rows: kappa, upper endpoint of M11 (negative), lower
endpoint of det M (positive).  One-sided bounds are the quotable
objects; the short arb display can round a positive ball's midpoint
to zero.
"""
import re
import sys

TAGS = [('0p0', '0'), ('0p05', '0.05'), ('n0p05', '-0.05'),
        ('0p0995', '0.0995'), ('0p13', '0.13'), ('n0p45', '-0.45')]

BALL = r'\[([-0-9.e+]+) \+/- [0-9.e-]+\]'
PAT_M11 = re.compile(r'M11 endpoints = \[' + BALL + ', ' + BALL)
PAT_DET = re.compile(r'det endpoints = \[' + BALL + ', ' + BALL)


def main():
    rows = []
    for tag, kappa in TAGS:
        try:
            txt = open('results/hess_endpoints_%s.out' % tag).read()
        except OSError:
            print('%% %s: missing' % tag)
            continue
        m11 = PAT_M11.search(txt)
        det = PAT_DET.search(txt)
        cert = 'CERTIFIED' in txt
        if not (m11 and det and cert):
            print('%% %s: incomplete or not certified' % tag)
            continue
        m11_hi = float(m11.group(2))
        det_lo = float(det.group(1))
        rows.append((float(kappa), kappa, m11_hi, det_lo))
    rows.sort()
    for _, kappa, m11_hi, det_lo in rows:
        print('$%s$ & $M_{11}\\le %.5f$ & $\\det M\\ge %.3g$\\\\'
              % (kappa, m11_hi, det_lo))


if __name__ == '__main__':
    main()
