"""Stage-2 sweep at kappa = 0.05: locate the certified (q0, psi0) boxes,
configure the general-kappa modules, and certify the stage-1 exclusion
box minus the Region-I star.

Usage: python stage2_0p05.py [nworkers]
"""
import os
os.environ.setdefault('HUANG_GRID_N', '3600')   # before hg import

import platform
platform.node = lambda: 'research-worker'   # release manifests carry no machine names
import sys
if sys.platform.startswith('linux'):
    sys.executable = 'python3'   # manifests carry no machine paths (fork-safe on the worker hosts)
import certify_km_slab as C
import huang_cert_sweep as SW
import huang_cert_region1 as R1
import huang_cert_sweep2 as S2


def main():
    kappa = C.iv(C.dec('0.05'), C.dec('0.05'))
    alpha = C.iv(C.dec('0.781073068'), C.dec('0.781074776'))
    q_seed = C.iv(C.dec('0.565'), C.dec('0.578'))
    la = C.locate_fp_bisect(C.dec('0.781073068'), kappa, q_seed)
    lb = C.locate_fp_bisect(C.dec('0.781074776'), kappa, q_seed)
    if la is None or lb is None:
        raise SystemExit('locate failed')
    q = C.hull(la[0], lb[0])
    psi = C.hull(la[1], lb[1])
    SW.configure_sweep('0p05', C.arb(C.dec('0.05')), alpha, q, psi)
    R1.configure_region1(C.arb(C.dec('0.05')), alpha, q, psi)
    S2.configure_sweep2(
        os.path.join(R1.RESULTS_DIR, 'huang_region1_0p05.json'),
        supplement_path=os.path.join(R1.RESULTS_DIR,
                                     'huang_region1_supp_0p05.json'))
    print(f'EXCL_OLD = {S2.EXCL_OLD}, RB = {S2.RB:.2e}, '
          f'W_ANG = {R1.W_ANG:.4f}, manifest = {S2.MANIFEST_SHA[:16]}',
          flush=True)
    S2.main()


if __name__ == '__main__':
    main()
