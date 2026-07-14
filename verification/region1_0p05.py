"""Region I (the star) at kappa = 0.05: locate the certified (q0, psi0)
boxes, configure the general-kappa ray machinery, and run the banded
ray-concavity certificates.

Usage: python region1_0p05.py [nworkers]
"""
import platform
platform.node = lambda: 'research-worker'   # release manifests carry no machine names
import sys
if sys.platform.startswith('linux'):
    sys.executable = 'python3'   # manifests carry no machine paths (fork-safe on the worker hosts)
import certify_km_slab as C
import huang_cert_region1 as R1


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
    print(f'a* = ({R1.A1S:.8f}, {R1.A2S:.8f}), s0 = {R1.S0F:.8f}, '
          f'W_ANG = {R1.W_ANG:.4f}, '
          f'a1 ball rad = {R1._ball_rad(R1.A1B):.2e}',
          flush=True)
    R1.main()


if __name__ == '__main__':
    main()
