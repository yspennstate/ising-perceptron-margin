"""Radius ladder at kappa = 0.05: run one weak-direction chunk per radius
band to find the edge_check pass floor.  Diagnostic only."""
import sys
import time

from core import set_prec
set_prec(50)
import certify_km_slab as C
import huang_cert_region1 as R1


def main():
    kappa = C.iv(C.dec('0.05'), C.dec('0.05'))
    alpha = C.iv(C.dec('0.781073068'), C.dec('0.781074776'))
    q_seed = C.iv(C.dec('0.565'), C.dec('0.578'))
    la = C.locate_fp_bisect(C.dec('0.781073068'), kappa, q_seed)
    lb = C.locate_fp_bisect(C.dec('0.781074776'), kappa, q_seed)
    R1.configure_region1(C.arb(C.dec('0.05')), alpha,
                         C.hull(la[0], lb[0]), C.hull(la[1], lb[1]))
    jobs = R1.bands()
    ch = [j for j in jobs if j[1] > 4.5e-4]
    t1s = sorted(set(round(j[1], 9) for j in ch))
    for t1 in t1s:
        cand = [j for j in ch if round(j[1], 9) == t1
                and j[2] <= R1.W_ANG < j[3]]
        if not cand:
            cand = [j for j in ch if round(j[1], 9) == t1]
        j = cand[0]
        t0 = time.time()
        r = R1.band_job(j)
        print('t1=%.6f ok=%s why=%s worst=%s (%.0fs)'
              % (t1, r['ok'], r.get('why', '-'), r.get('worst'),
                 time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
