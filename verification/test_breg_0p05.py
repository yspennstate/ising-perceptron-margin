"""Consistency test for the anchored-Bregman evaluators at kappa = 0.05.

Phi_breg0_acb(l) and  Phi_acb(l) - Phi_acb(1,0) - (l-(1,0)).gradPhi_acb(1,0)
enclose the SAME quantity, so the enclosures must intersect; at l = (1,0)
the Bregman gap is identically 0.  Also compares widths (the point of the
correlated form) and checks grad_diff_acb the same way."""
from core import set_prec, endpoints
set_prec(50)
import certify_km_slab as C
import huang_cert_region1 as R1


def ep(b):
    lo, hi = endpoints(b)
    return float(lo), float(hi)


def main():
    kappa = C.iv(C.dec('0.05'), C.dec('0.05'))
    alpha = C.iv(C.dec('0.781073068'), C.dec('0.781074776'))
    q_seed = C.iv(C.dec('0.565'), C.dec('0.578'))
    la = C.locate_fp_bisect(C.dec('0.781073068'), kappa, q_seed)
    lb = C.locate_fp_bisect(C.dec('0.781074776'), kappa, q_seed)
    R1.configure_region1(C.arb(C.dec('0.05')), alpha,
                         C.hull(la[0], lb[0]), C.hull(la[1], lb[1]))
    tol = R1.arb(R1.dec('0.000000001'))
    pts = [('anchor', '1.0', '0.0'),
           ('near', '1.003463', '-0.009671'),
           ('mid', '0.9965', '0.0148'),
           ('far', '1.0286', '-0.0842')]
    p0 = R1.Phi_acb(R1._dec(1.0, 10), R1._dec(0.0, 10), tol=tol)
    g10, g20 = R1.gradPhi_acb(R1._dec(1.0, 10), R1._dec(0.0, 10), tol=tol)
    nfail = 0
    for name, s1, s2 in pts:
        l1 = R1.dec(s1)
        l2 = R1.dec(s2)
        breg = R1.Phi_breg0_acb(l1, l2, tol=tol)
        plain = (R1.Phi_acb(l1, l2, tol=tol) - p0
                 - (R1.arb(l1) - 1) * g10 - R1.arb(l2) * g20)
        blo, bhi = ep(breg)
        plo, phi_ = ep(plain)
        overlap = (blo <= phi_) and (plo <= bhi)
        wb = bhi - blo
        wp = phi_ - plo
        print('%-6s breg=[%.3e,%.3e] w=%.1e  plain w=%.1e  overlap=%s'
              % (name, blo, bhi, wb, wp, overlap), flush=True)
        if not overlap:
            nfail += 1
        if name == 'anchor' and not (blo <= 0.0 <= bhi):
            print('  FAIL: Breg0(1,0) does not contain 0')
            nfail += 1
        gd1, gd2 = R1.grad_diff_acb(l1, l2, tol=tol)
        pg1, pg2 = R1.gradPhi_acb(l1, l2, tol=tol)
        d1lo, d1hi = ep(gd1)
        q1lo, q1hi = ep(pg1 - g10)
        d2lo, d2hi = ep(gd2)
        q2lo, q2hi = ep(pg2 - g20)
        ok1 = (d1lo <= q1hi) and (q1lo <= d1hi)
        ok2 = (d2lo <= q2hi) and (q2lo <= d2hi)
        print('       gdiff w=(%.1e,%.1e) plain w=(%.1e,%.1e) overlap=%s,%s'
              % (d1hi - d1lo, d2hi - d2lo, q1hi - q1lo, q2hi - q2lo,
                 ok1, ok2), flush=True)
        if not (ok1 and ok2):
            nfail += 1
        if name == 'anchor' and not (d1lo <= 0.0 <= d1hi
                                     and d2lo <= 0.0 <= d2hi):
            print('  FAIL: grad_diff(1,0) does not contain (0,0)')
            nfail += 1
    print('RESULT:', 'PASS' if nfail == 0 else 'FAIL %d' % nfail)


if __name__ == '__main__':
    main()
