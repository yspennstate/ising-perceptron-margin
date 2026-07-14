"""Stage-2 smoke at kappa = 0.05: certify the top cell containing a*
(hardest: in_star + bisection + eval_cell interplay) and one far cell."""
import os
os.environ.setdefault('HUANG_GRID_N', '3600')

import time

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
    q = C.hull(la[0], lb[0])
    psi = C.hull(la[1], lb[1])
    SW.configure_sweep('0p05', C.arb(C.dec('0.05')), alpha, q, psi)
    R1.configure_region1(C.arb(C.dec('0.05')), alpha, q, psi)
    S2.configure_sweep2(os.path.join(R1.RESULTS_DIR,
                                     'huang_region1_0p05.json'))
    print(f'EXCL_OLD = {S2.EXCL_OLD}, RB = {S2.RB:.2e}', flush=True)
    jobs, n1, n2 = S2.build_jobs()
    center = [j for j in jobs
              if j[0] <= R1.A1S <= j[1] and j[2] <= R1.A2S <= j[3]]
    far = jobs[0]
    for tag, j in (('center', center[0]), ('far', far)):
        t0 = time.time()
        ok, nc, worst = S2.cert_cell(j)
        print('%s cell %s: ok=%s leaves=%d worst=%s (%.0fs)'
              % (tag, j, ok, nc, worst, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
