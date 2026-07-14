"""Final pre-relaunch validation at kappa = 0.05: Bregman consistency,
both ex-failing bands, the innermost + origin bands, and one weak-wedge
ladder rung -- all with the final code state (stable Mills kernels,
exponential-split evaluators, 5-radius fans)."""
import time

from core import set_prec, endpoints
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
    bad = 0
    # 1. Bregman anchor identity + consistency vs the uncorrelated path
    tol = R1.arb(R1.dec('0.000000001'))
    breg0 = R1.Phi_breg0_acb(R1.dec('1.0'), R1.dec('0.0'), tol=tol)
    lo, hi = endpoints(breg0)
    print('Breg0(1,0) = [%g, %g]' % (float(lo), float(hi)), flush=True)
    if not (float(lo) <= 0.0 <= float(hi)):
        bad += 1
    p0 = R1.Phi_acb(R1.dec('1.0'), R1.dec('0.0'), tol=tol)
    g10, g20 = R1.gradPhi_acb(R1.dec('1.0'), R1.dec('0.0'), tol=tol)
    for s1, s2 in (('0.694', '-0.178'), ('1.0846', '0.5582'),
                   ('0.9965', '0.0148')):
        l1, l2 = R1.dec(s1), R1.dec(s2)
        b = R1.Phi_breg0_acb(l1, l2, tol=tol)
        pl = (R1.Phi_acb(l1, l2, tol=tol) - p0
              - (R1.arb(l1) - 1) * g10 - R1.arb(l2) * g20)
        blo, bhi = endpoints(b)
        plo, phi_ = endpoints(pl)
        ok = (float(blo) <= float(phi_)) and (float(plo) <= float(bhi))
        print('breg(%s,%s) w=%.2e overlap=%s' % (
            s1, s2, float(bhi) - float(blo), ok), flush=True)
        if not ok:
            bad += 1
    # 2. band gates
    gates = [
        ('minus-wedge', (0.00864, 0.012, 3.6054389046534583,
                         3.6214389046534583)),
        ('plus-wedge', (0.00864, 0.012, 0.7678462510636656,
                        0.7838462510636653)),
        ('cone-plus', (0.006220799999999999, 0.00864,
                       0.8898462510636657, 0.9398462510636657)),
        ('wedge-small', (0.00044926874911493833, 0.0006239843737707477,
                         0.6100000000000001, 0.6260000000000001)),
        ('innermost', (1.6871692432026063e-06, 2.3432906155591756e-06,
                       0.0, 6.283185307179586)),
        ('origin', (0.0, 1e-06, 0.0, 6.283185307179586)),
    ]
    for tag, band in gates:
        t0 = time.time()
        r = R1.band_job(band)
        print('%s: ok=%s why=%s cells=%s worst=%s nfail=%s (%.0fs)'
              % (tag, r['ok'], r.get('why', '-'), r.get('cells'),
                 r.get('worst'), r.get('nfail'), time.time() - t0),
              flush=True)
        if not r['ok']:
            bad += 1
    print('VALIDATION:', 'PASS' if bad == 0 else 'FAIL %d' % bad, flush=True)


if __name__ == '__main__':
    main()
