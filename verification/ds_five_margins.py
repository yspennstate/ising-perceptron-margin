"""Scan the general-kappa Ding--Sun condition at the five certified
margins.  Writes results/ds_condition_scan.json with the S(lambda)
curves and summary lines to stdout.  BelowNormal priority per the
compute doctrine."""
import ctypes
import json

import ds_condition

BELOW_NORMAL = 0x4000
ctypes.windll.kernel32.SetPriorityClass(
    ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)

MARGINS = [(-0.45, 1.557653118), (-0.05, 0.889408907), (0.0, 0.8330786),
           (0.05, 0.781073922), (0.0995, 0.733479359), (0.13, 0.705933058)]

out = {}
for kappa, alpha in MARGINS:
    ds, rows = ds_condition.scan(kappa, alpha)
    checks = ds.checks()
    bad = {k: v for k, v in checks.items()
           if 'lambda_min' not in k and abs(v) > 1e-8}
    worst = max(s for l, s in rows if abs(l) > 1e-3 and abs(l - 1) > 1e-3)
    interior = max(s for l, s in rows if 0.02 < l < 0.98)
    print('kappa=%+.4f alpha*=%.7f  lambda_min=%+.4f  '
          'max S off {0,1}: %+.6f  interior max: %+.6f  %s'
          % (kappa, ds.alpha, ds.lambda_min(), worst, interior,
             'CHECKS-FAIL: %s' % bad if bad else 'checks ok'),
          flush=True)
    out['%+.4f' % kappa] = {
        'alpha': ds.alpha, 'q': ds.q, 'psi': ds.psi,
        'lambda_min': ds.lambda_min(),
        'checks': {k: float(v) for k, v in checks.items()},
        'rows': [(float(l), float(s)) for l, s in rows]}

with open('results/ds_condition_scan.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print('wrote results/ds_condition_scan.json')
