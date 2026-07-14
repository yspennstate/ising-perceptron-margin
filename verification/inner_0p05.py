"""Innermost bands at kappa = 0.05: every full-circle band (t1 <= 4.5e-4),
including the final [0, T_IN] origin band -- the regime the old decimal
origin inflation could not certify."""
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
    jobs = [j for j in R1.bands() if j[1] <= 4.5e-4]
    print('full-circle bands:', len(jobs), flush=True)
    nfail = 0
    for j in jobs:
        t0 = time.time()
        r = R1.band_job(j)
        print('t=[%.2e,%.2e] ok=%s why=%s worst=%s (%.0fs)'
              % (j[0], j[1], r['ok'], r.get('why', '-'), r.get('worst'),
                 time.time() - t0), flush=True)
        if not r['ok']:
            nfail += 1
    print('RESULT:', 'PASS' if nfail == 0 else 'FAIL %d' % nfail)


if __name__ == '__main__':
    main()
