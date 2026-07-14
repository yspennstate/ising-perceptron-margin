"""Region II sweep at kappa = 0.05: locate the certified (q0, psi0)
boxes at the deep alpha interval, configure the general-kappa two-variable
machinery, and run the full moment-rectangle sweep.

Usage: python sweep_0p05.py [nworkers] [coarse_n]
"""
import os
import sys

os.environ.setdefault('HUANG_GRID_N', '3600')   # before hg import: the
# T enclosures at the sweep's far tilts need the fine grid (measured:
# radius 0.205 at n = 900 vs 0.070 at n = 3600)

import platform
platform.node = lambda: 'research-worker'   # release manifests carry no machine names
if sys.platform.startswith('linux'):
    sys.executable = 'python3'   # manifests carry no machine paths (fork-safe on the worker hosts)
import certify_km_slab as C
import huang_cert_sweep as SW


def main():
    kappa = C.iv(C.dec('0.05'), C.dec('0.05'))
    alpha = C.iv(C.dec('0.781073068'), C.dec('0.781074776'))
    q_seed = C.iv(C.dec('0.565'), C.dec('0.578'))
    loc_lb = C.locate_fp_bisect(C.dec('0.781073068'), kappa, q_seed)
    loc_ub = C.locate_fp_bisect(C.dec('0.781074776'), kappa, q_seed)
    if loc_lb is None or loc_ub is None:
        raise SystemExit('locate failed')
    q_all = C.hull(loc_lb[0], loc_ub[0])
    p_all = C.hull(loc_lb[1], loc_ub[1])
    print('located q:', q_all, ' psi:', p_all, flush=True)
    SW.configure_sweep('0p05', C.arb(C.dec('0.05')), alpha, q_all, p_all)
    print(f'rect a1 +-{SW.A1MAX}, a2 +-{SW.A2MAX}; EXCL {SW.EXCL}',
          flush=True)
    SW.main()


if __name__ == '__main__':
    main()
