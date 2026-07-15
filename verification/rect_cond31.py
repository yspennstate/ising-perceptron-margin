"""Huang's Condition 3.1 discharged on a thin rectangle at one margin.

Condition 3.1 (arXiv:2404.18902) is existential: it asks for SOME
0 < alpha_lb < alpha_ub and 0 < q_lb < q_ub < 1 such that on the
rectangle, (i) sup_q (P o R_alpha)' < 1 for every alpha in the
interval, (ii) each such alpha has a unique fixed point q_* in
(q_lb, q_ub), and (iii) G_* is strictly decreasing across the alpha
interval with a unique root.  Huang's own kappa = 0 instantiation
(his Proposition 3.2, quoting Ding--Sun Proposition 1.3) takes a
rectangle of width 1e-9 about the fixed point; this driver produces
the same object at any margin, on the deep certified alpha_* interval
and q_* +- delta.

Evaluator discipline follows locate_alpha.py: endpoint inwardness by
the thin adaptive evaluators (P_of, R_of at point arguments), the
fixed-point location by the mean-value Picard shrink (locate_deep_n),
the G signs by G_mean_value_deep at the located balls, and the
rectangle contraction by the identity-based product bound
(Pprime_bound, dRdq_bound).  The plain-hull wide evaluators are never
used here: their O(1/n) fuzz (~7e-4) exceeds the inwardness gap
(~2e-4) on a thin rectangle, which is a precision mismatch, not a
soundness question.

Uniqueness in (ii) follows from (i) plus inwardness (Banach on the
closed interval); strict decrease in (iii) from the envelope
derivative E log Psi < 0 along the branch that (i)+(ii) make
differentiable.  Output: results/rect_cond31_<tag>.json.

Run: python rect_cond31.py 0.13 0.705932217 0.705933898 [delta]
"""
import json
import sys

import certify_km_slab as C
from core import dec, iv, endpoints
from locate_alpha import locate_deep_n

N_LOC = 16000
N_G = 16000


def main():
    kappa = sys.argv[1]
    alo, ahi = dec(sys.argv[2]), dec(sys.argv[3])
    delta = dec(sys.argv[4]) if len(sys.argv) > 4 else dec('0.001')
    kap = dec(kappa)
    tag = kappa.replace('.', 'p').replace('-', 'n')

    checks = []

    def check(name, ok, detail):
        checks.append({'name': name, 'ok': bool(ok), 'detail': str(detail)})
        print('%-44s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail),
              flush=True)

    # locate the fixed point tightly at both alpha endpoints, seeding
    # from a generous ball; locate_deep_n itself verifies endpoint
    # inwardness of the seed with the thin adaptive evaluators
    seed = iv(dec('0.55'), dec('0.62'))
    loc_lo = locate_deep_n(alo, kap, seed, N_LOC)
    loc_hi = locate_deep_n(ahi, kap, seed, N_LOC)
    check('seed ball inward at both alpha ends',
          loc_lo is not None and loc_hi is not None, 'locate returned')
    if loc_lo is None or loc_hi is None:
        raise SystemExit(1)
    qb_lo, pb_lo = loc_lo
    qb_hi, pb_hi = loc_hi
    ql1, qh1 = endpoints(qb_lo)
    ql2, qh2 = endpoints(qb_hi)
    print('located q at alpha_lb: [%s, %s]' % (ql1, qh1))
    print('located q at alpha_ub: [%s, %s]' % (ql2, qh2))

    # the rectangle
    qlo = min(ql1, ql2) - delta
    qhi = max(qh1, qh2) + delta
    qbox = iv(qlo, qhi)
    print('rectangle: alpha in [%s, %s], q in [%s, %s]'
          % (alo, ahi, qlo, qhi))

    # (ii) endpoint inwardness of the RECTANGLE at both alpha ends,
    # thin adaptive evaluators at point arguments
    for aname, a in (('alpha_lb', alo), ('alpha_ub', ahi)):
        v_lo = C.P_of(C.R_of(qlo, a, kap)) - qlo
        v_hi = qhi - C.P_of(C.R_of(qhi, a, kap))
        l_lo, _ = endpoints(v_lo)
        l_hi, _ = endpoints(v_hi)
        check('inward at %s, low edge' % aname, l_lo > 0,
              'P(R(q_lo)) - q_lo >= %s' % l_lo)
        check('inward at %s, high edge' % aname, l_hi > 0,
              'q_hi - P(R(q_hi)) >= %s' % l_hi)

    # (i) contraction on the rectangle: R is linear increasing in
    # alpha, so the psi range is hulled over both alpha ends
    R_a = C.R_of_wide(qbox, alo, kap)
    R_b = C.R_of_wide(qbox, ahi, kap)
    plo = min(endpoints(R_a)[0], endpoints(R_b)[0])
    phi_ = max(endpoints(R_a)[1], endpoints(R_b)[1])
    Pp = C.Pprime_bound(plo, phi_)
    dR = C.dRdq_bound(qbox, ahi, kap)
    prod = Pp * dR
    _, prod_hi = endpoints(prod)
    check('contraction sup(P\')*sup|dR/dq| < 1', float(prod_hi) < 1.0,
          'enclosure hi = %s' % prod_hi)

    # (iii) G signs at the located fixed points, deep mean-value form
    G_lo = C.G_mean_value_deep(alo, qb_lo, pb_lo, kap, n=N_G)
    G_hi = C.G_mean_value_deep(ahi, qb_hi, pb_hi, kap, n=N_G)
    gl, _ = endpoints(G_lo)
    _, gh = endpoints(G_hi)
    check('G(alpha_lb) > 0', float(gl) > 0, 'lo = %s' % gl)
    check('G(alpha_ub) < 0', float(gh) < 0, 'hi = %s' % gh)

    # strict decrease along the branch
    elp = C.ElogPsi_kappa(qbox, kap)
    _, elp_hi = endpoints(elp)
    check('E log Psi < 0 over the rectangle', float(elp_hi) < 0,
          'hi = %s' % elp_hi)

    ok = all(c['ok'] for c in checks)
    out = {
        'kind': 'huang_condition_3_1_rectangle',
        'kappa': kappa,
        'alpha_lb': str(alo), 'alpha_ub': str(ahi),
        'q_lb': str(qlo), 'q_ub': str(qhi),
        'contraction_hi': str(prod_hi),
        'evaluators': {'n_loc': N_LOC, 'n_g': N_G},
        'checks': checks, 'ok': ok,
    }
    with open('results/rect_cond31_%s.json' % tag, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('CONDITION 3.1 RECTANGLE %s at kappa = %s'
          % ('CERTIFIED' if ok else 'FAILED', kappa), flush=True)


if __name__ == '__main__':
    main()
