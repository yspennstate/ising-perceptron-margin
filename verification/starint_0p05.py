"""Star-interior certificate at kappa = 0.05: prove the Region-I star
(plus its symbolic-origin ball) lies in the interior of the moment body.

Usage: python starint_0p05.py
"""
import os

import platform
platform.node = lambda: 'research-worker'   # release manifests carry no machine names
import sys
if sys.platform.startswith('linux'):
    sys.executable = 'python3'   # manifests carry no machine paths (fork-safe on the worker hosts)
import certify_km_slab as C
import huang_cert_region1 as R1
import huang_cert_star_interior as SI


def main():
    kappa = C.iv(C.dec('0.05'), C.dec('0.05'))
    alpha = C.iv(C.dec('0.781073068'), C.dec('0.781074776'))
    q_seed = C.iv(C.dec('0.565'), C.dec('0.578'))
    la = C.locate_fp_bisect(C.dec('0.781073068'), kappa, q_seed)
    lb = C.locate_fp_bisect(C.dec('0.781074776'), kappa, q_seed)
    if la is None or lb is None:
        raise SystemExit('locate failed')
    R1.configure_region1(C.arb(C.dec('0.05')), alpha,
                         C.hull(la[0], lb[0]), C.hull(la[1], lb[1]))
    out = os.path.join(R1.RESULTS_DIR, 'huang_star_interior_0p05.json')
    payload = SI.write_certificate(out)
    b = payload['bounds']
    for name in ('inradius', 'required_radius', 'clearance'):
        p = b[name]
        print('%s: %se%s (+/- %se%s)' % (name, p['mid10'], p['exp10'],
                                         p.get('rad10'), p['exp10']),
              flush=True)
    print('PASS star interior 0p05; certificate:',
          payload['certificate_sha256'], out, flush=True)


if __name__ == '__main__':
    main()
