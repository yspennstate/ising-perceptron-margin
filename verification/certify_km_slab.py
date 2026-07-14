"""Certified verification of the Krauth-Mezard fixed-point condition
(Huang's Condition km-well-defd) on kappa slabs.

Batch structure (ball arithmetic throughout):

  global (once per batch, over the hull of all slab rectangles):
    psi range    derived from the R enclosure over the whole rectangle
                 (contains every fixed-point psi_*; domain for the
                 amp-works / local-concavity checks);
    diagnostic   the product bound over the batch hull is recorded but
                 not claimed there (the wide kappa ball poisons it);
                 the CLAIMED contraction bound is per slab, below.

  per slab (kappa ball):
    contraction  sup_q (P o R_alpha)'(q) <= sup_psi P' * sup_q dR/dq
                 < 1 on the slab's own rectangle (dR/dq via its
                 one-term integral identity);
    sweep        on each alpha sub-interval the q-interval endpoints map
                 strictly inward: existence of a fixed point inside for
                 every alpha, by the intermediate value theorem;
    crossing     at the two thin alpha endpoints, a certified Picard
                 shrink locates a fixed-point box and the sign of G
                 resolves through a mean-value form that keeps the
                 near-zero saddle derivatives (q - P(psi))/2 and
                 (psi - R(q))/2 explicit;
    conditions   Huang's amp-works product < 1 and local-concavity
                 lambda(z_hat) < 0 at the fixed witness.

Together with E log Psi < 0 pointwise (so G_* strictly decreases; see
NOTES.md for the stationarity lemma behind the envelope), each passing
slab discharges Condition km-well-defd verbatim: a unique fixed point
q_*(alpha, kappa) in (q_lb, q_ub) for every alpha in the interval, and
a unique crossing alpha_*(kappa) in [alpha_lb, alpha_ub].  Nakajima's
uniqueness theorem (kappa >= 0) is an independent cross-check on the
positive side.

The kappa = 0 corner argument of Ding-Sun Proposition 1.3 (Block 1 of
ising-perceptron-capacity) relies on their analytic monotonicity Lemma
7.2, proved at kappa = 0; the full-rectangle evaluation here does not
need it, at the price of thin slabs.

Uses the certified primitives (arb integration, tails) of
ising-perceptron-capacity/verification/core.py; point the environment
variable ISING_CAPACITY_VERIFICATION there if the default relative path
does not hold.

Run:  python certify_km_slab.py slabs.json [results.json]
"""

import json
import os
import sys

_CAP = os.environ.get(
    'ISING_CAPACITY_VERIFICATION',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'ising-perceptron-capacity', 'verification'))
sys.path.insert(0, _CAP)

from flint import arb, acb  # noqa: E402
from core import (set_prec, dec, iv, hull, endpoints, phi, Psi, mills,  # noqa: E402
                  c_phi, c_Psi, c_logPsi, c_mills, c_log2cosh,
                  gauss_tail_mass, z1_tail, z2_tail, report)
import core  # noqa: E402


_TOL_BITS = 34


def integrate(f, a, b, pieces=None):
    """Rigorous adaptive integral for THIN-parameter integrands:
    tolerance matched to the certified margins (~1e-4 scale), generous
    budget (thin integrands converge fast, the budget is headroom).
    Wide-parameter-ball integrals use riemann_E instead - complex
    probes cannot resolve parameter-driven width at any budget."""
    tol = arb(2) ** (-_TOL_BITS)
    pts = [arb(a)] + [arb(p) for p in (pieces or [])] + [arb(b)]
    total = acb(0)
    for lo, hi in zip(pts[:-1], pts[1:]):
        total += acb.integral(f, acb(lo), acb(hi), abs_tol=tol,
                              depth_limit=2000, eval_limit=2000000)
    assert total.imag.contains(arb(0)), "integral drifted off the real axis"
    return total.real


def _packet_or_none(val):
    """Exact outward packet of a ball for the Lean skeleton; None when
    the value is non-finite (a failed tail bound) - the display strings
    stay for reading, the packet is the machine-checkable claim."""
    try:
        import block3bc_exact as exact
        return exact.arb_packet(val)
    except Exception:
        return None


def verify_global(slabs):
    """Once per batch, over the hull of all slab rectangles: the uniform
    contraction bound.

    Wide-ball integrals go through real-cell Riemann sums (complex
    probes fail structurally on parameter-driven width).  The psi range
    on which P' is bounded is DERIVED from the R enclosure over the
    whole rectangle - since psi_*(alpha, kappa) = R(q_*) and q_* lies in
    the q rectangle (per-slab sweep checks), the derived range contains
    every fixed-point psi; it is returned for the per-slab amp-works and
    local-concavity domains.

    E log Psi(u) < 0 needs no numerics: Psi(u) is in (0,1) pointwise, so
    the envelope derivative dG_*/dalpha = E log Psi(u) is strictly
    negative and G_* strictly decreases; see NOTES.md for the
    stationarity lemma behind the envelope."""
    kappa = iv(dec(min(s['kappa_lo'] for s in slabs)),
               dec(max(s['kappa_hi'] for s in slabs)))
    alpha = iv(dec(min(s['alpha_lb'] for s in slabs)),
               dec(max(s['alpha_ub'] for s in slabs)))
    q_ball = iv(dec(min(s['q_lb'] for s in slabs)),
                dec(max(s['q_ub'] for s in slabs)))

    checks = []

    def chk(name, val, want):
        ok = report(f"  {name}", val, want)
        lo, hi = endpoints(val)
        checks.append({'name': name, 'want': want,
                       'lo': str(lo), 'hi': str(hi), 'ok': ok,
                       'packet': _packet_or_none(val)})
        return ok

    R_glob = R_of_wide(q_ball, alpha, kappa)
    lo_R, hi_R = endpoints(R_glob)
    psi_range = (lo_R - dec('0.01')).union(hi_R + dec('0.01'))
    lo_pr, hi_pr = endpoints(psi_range)

    ok = True
    ok &= chk("R(rect) finite and positive", R_glob, '>0')

    # The contraction product sup P' * sup |dR/dq| is recorded as a
    # diagnostic, not a claim: the product-of-sups bound loses ~2x
    # against the true sup of the product (P' and dR/dq peak at
    # different points), and with slab-scale parameter boxes it lands
    # above 1 even where the true contraction is ~0.75.  Uniqueness of
    # the fixed point for kappa >= 0 below alpha_c is Nakajima's
    # analytic theorem (arXiv 2512.23195); the certificate's own claims
    # are existence (inward sweep), the crossing, and the amp-works /
    # local-concavity conditions.
    pp = Pprime_bound(lo_pr, hi_pr)
    dr = dRdq_bound(q_ball, alpha, kappa)
    _, pp_hi = endpoints(pp)
    _, dr_hi = endpoints(abs(dr))
    contraction = pp_hi * dr_hi

    rec = {'contraction_product_diagnostic': str(contraction),
           'Pprime_ub': str(pp_hi), 'dRdq_enclosure': str(dr),
           'psi_range': [str(lo_pr), str(hi_pr)],
           'R_global': [str(lo_R), str(hi_R)],
           'checks': checks, 'ok': bool(ok)}
    return ok, rec, psi_range

# 80 bits is deliberate: every certified margin here is >= 1e-4 while
# ball radii at this precision run ~1e-16; the adaptive integrator's
# cost grows superlinearly in precision, and 50 slabs must be tractable.
set_prec(int(os.environ.get('KM_SLAB_PREC', '80')))

L = arb(8)


def c_u(z, q, kappa):
    """u(z) = (kappa - sqrt(q) z)/sqrt(1-q) on acb; q, kappa arb balls."""
    return (acb(kappa) - acb(q).sqrt() * z) / acb(1 - q).sqrt()


def mills_ball(u):
    """M(u) = phi(u)/Psi(u) as a real-ball enclosure via monotone
    endpoints.

    M is strictly increasing (M' = M(M-u) > 0), so the enclosure over a
    ball is exactly [M(lo), M(hi)] with two THIN evaluations - which are
    well-conditioned at any |u| (arb tracks tiny phi and Psi exactly).
    Evaluating phi(ball)/Psi(ball) directly instead NaNs: erfc on a wide
    ball in the steep region returns an enclosure spanning zero."""
    lo, hi = endpoints(u)
    return hull(phi(lo) / Psi(lo), phi(hi) / Psi(hi))


def riemann_mv(w_mid, dw_cell, n=2000):
    """Mean-value Riemann enclosure of int_{-L}^{L} w(z) phi(z) dz:
    per cell, w(midpoint) * mass plus the derivative correction (the
    dsfun.frH_upper pattern from the capacity verification).  O(h^2)
    where the plain hull is O(h) - this is what makes the G signs
    resolve against their ~6e-4 margins.

    w_mid: thin-argument evaluation of w; dw_cell: enclosure of w' on a
    cell ball."""
    h = 2 * L / n
    total = arb(0)
    for j in range(n):
        lo = -L + h * j
        hi = -L + h * (j + 1)
        mid = (lo + hi) / 2
        cell = lo.union(hi)
        mass = Psi(lo) - Psi(hi)
        corr_signed = phi(lo) - phi(hi) - mid * mass
        mass_lo = Psi(lo) - Psi(mid)
        mass_hi = Psi(mid) - Psi(hi)
        corr_abs = (mid * mass_lo - (phi(lo) - phi(mid))
                    + (phi(mid) - phi(hi)) - mid * mass_hi)
        d = dw_cell(cell)
        d_mid = arb(d.mid())
        d_rad = arb(d.rad())
        slack = d_rad * corr_abs
        total += w_mid(mid) * mass + d_mid * corr_signed \
            + slack.union(-slack)
    return total


def riemann_E(w, n=20000):
    """Plain-hull Riemann enclosure of int_{-L}^{L} w(z) phi(z) dz on
    real cells: sum of w(cell) times the exact Gaussian cell mass.

    This is the wide-parameter-ball workhorse: adaptive complex
    quadrature cannot resolve integrand width that comes from parameter
    balls (subdividing z does not shrink it) and burns its budget to
    NaN; real cells have no probes and no analyticity guards.  The
    O(h) hull error at n = 20000 is ~1e-3 against ~1e-1 margins."""
    h = 2 * L / n
    total = arb(0)
    for j in range(n):
        lo = -L + h * j
        hi = -L + h * (j + 1)
        cell = lo.union(hi)
        mass = Psi(lo) - Psi(hi)
        total += w(cell) * mass
    return total


def u_arb(z, q, kappa):
    return (kappa - q.sqrt() * z) / (1 - q).sqrt()


def P_of(psi):
    """E tanh^2(sqrt(psi) Z); tail tanh^2 <= 1.  Adaptive path - use
    only with thin psi."""
    s = psi.sqrt()
    main = integrate(lambda z, an: (s * z).tanh() ** 2 * c_phi(z), -L, L)
    return hull(main, main + gauss_tail_mass(L))


def P_of_wide(psi, n=8000):
    """P on a psi ball: P is increasing in psi, so the enclosure is the
    hull of two thin real-cell Riemann evaluations at the endpoints."""
    lo_p, hi_p = endpoints(psi)

    def at(pv):
        s = pv.sqrt()
        v = riemann_E(lambda z: (s * z).tanh() ** 2, n)
        return hull(v, v + gauss_tail_mass(L))

    return hull(at(arb(lo_p)), at(arb(hi_p)))


def P_of_mv(psi, n=4000):
    """P via mean-value Riemann (O(h^2) instead of the plain hull's
    O(h)): w = tanh^2(s z), w' = 2 s tanh (1 - tanh^2).  Monotone
    endpoints in psi.  This is what lets Picard shrink located boxes
    to ~1e-5 for the deep alpha_* bisections."""
    lo_p, hi_p = endpoints(psi)

    def at(pv):
        s = pv.sqrt()

        def dw(cell):
            t = (s * cell).tanh()
            return 2 * s * t * (1 - t * t)

        v = riemann_mv(lambda z: (s * z).tanh() ** 2, dw, n)
        return hull(v, v + gauss_tail_mass(L))

    return hull(at(arb(lo_p)), at(arb(hi_p)))


def R_of_mv(q, alpha, kappa, n=4000):
    """R via mean-value Riemann: w = M(u)^2, w' = 2 M M' u'(z) with
    u' = -sqrt(q)/sqrt(1-q) constant.  Parameter width stays (it is
    genuine); only the O(h) cell-hull term is removed."""
    up = -q.sqrt() / (1 - q).sqrt()

    def dw(cell):
        u = u_arb(cell, q, kappa)
        return 2 * mills_ball(u) * Mprime_ball(u) * up

    main = riemann_mv(lambda z: mills_ball(u_arb(z, q, kappa)) ** 2,
                      dw, n)
    body = hull(main, main + _R_tails(q, kappa))
    return alpha * body / (1 - q)


def _R_tails(q, kappa):
    """Two-sided tails for E[M(u(Z))^2] outside [-L, L].

    z >= L : u <= u(L) << 0 and M is increasing, so M(u)^2 <= M(u(L))^2.
    z <= -L: M(u)^2 <= (1 + max(0, u))^2 <= (1 + (kappa + sqrt(q)|z|)_+
             / sqrt(1-q))^2, a quadratic in |z| against the Gaussian tail.
    """
    uL = u_arb(L, q, kappa)
    lo, hi = endpoints(uL)
    tail_pos = mills_ball(arb(hi)) ** 2 * gauss_tail_mass(L) / 2 \
        if hi < 0 else arb(0, arb('inf'))
    kp = abs(kappa)
    s1q = (1 - q).sqrt()
    c0 = 1 + kp / s1q
    c1 = q.sqrt() / s1q
    m0 = gauss_tail_mass(L) / 2
    m1 = phi(L)
    m2 = L * phi(L) + Psi(L)
    tail_neg = c0 ** 2 * m0 + 2 * c0 * c1 * m1 + c1 ** 2 * m2
    return tail_pos + tail_neg


def R_of(q, alpha, kappa):
    """(alpha/(1-q)) E[M(u(Z))^2], adaptive quadrature (thin parameters)."""
    main = integrate(
        lambda z, an: c_mills(c_u(z, q, kappa)) ** 2 * c_phi(z), -L, L)
    body = hull(main, main + _R_tails(q, kappa))
    return alpha * body / (1 - q)


def R_of_wide(q, alpha, kappa, n=20000):
    """R for wide parameter balls, via real-cell Riemann."""
    main = riemann_E(lambda z: mills_ball(u_arb(z, q, kappa)) ** 2, n)
    body = hull(main, main + _R_tails(q, kappa))
    return alpha * body / (1 - q)


def ent2_of(x):
    """log(2 cosh x) - stable: |x| + log1p(exp(-2|x|))."""
    a = abs(x)
    return a + (-2 * a).exp().log1p()


def Elog2cosh(psi, n=2000):
    """E log 2cosh(sqrt(psi) Z), mean-value Riemann (thin psi):
    w = log 2cosh(s z), w' = s tanh(s z)."""
    s = psi.sqrt()
    main = riemann_mv(lambda z: ent2_of(s * z),
                      lambda cell: s * (s * cell).tanh(), n)
    lo = s * z1_tail(L)
    hi = lo + arb(2).log() * gauss_tail_mass(L)
    return main + hull(lo, hi)


def logPsi_ball(u):
    """log Psi on a real ball via monotone endpoints (log Psi is
    strictly decreasing; thin evaluations are stable at any u)."""
    lo, hi = endpoints(u)
    return hull(Psi(hi).log(), Psi(lo).log())


def ElogPsi_kappa(q, kappa, n=2000):
    """E log Psi(u(Z)), mean-value Riemann with rigorous tails.

    w = log Psi(u(z)) (kappa ball enters the thin-midpoint evaluation
    as genuine width), w' = M(u) sqrt(q)/sqrt(1-q).

    z >= L : 0 >= log Psi(u) >= -2 Psi(-u) >= -2 Psi(-u(L)).
    z <= -L: 0 >= log Psi(u) >= -(u^2/2 + log(2pi)/2 + u + 2 Psi(u)),
             u = (kappa + sqrt(q)|z|)/sqrt(1-q) >= 11-ish; expand u^2 and
             u against Gaussian tail moments of |z|.
    """
    coef = q.sqrt() / (1 - q).sqrt()
    main = riemann_mv(
        lambda z: logPsi_ball(u_arb(z, q, kappa)),
        lambda cell: mills_ball(u_arb(cell, q, kappa)) * coef, n)
    uL = u_arb(L, q, kappa)          # value at z = +L (most negative u)
    lo_uL, hi_uL = endpoints(uL)
    if not (hi_uL < 0):
        return arb(0, arb('inf'))
    tail_pos_lo = -2 * Psi(-hi_uL) * gauss_tail_mass(L) / 2

    kp = abs(kappa)
    s1q = (1 - q).sqrt()
    m0 = gauss_tail_mass(L) / 2
    m1 = phi(L)
    m2 = L * phi(L) + Psi(L)
    # u^2/2 <= (kappa^2 + 2 kp sqrt(q)|z| + q z^2) / (2(1-q))
    quad = (kp ** 2 * m0 + 2 * kp * q.sqrt() * m1 + q * m2) / (2 * (1 - q))
    lin = (kp * m0 + q.sqrt() * m1) / s1q
    u_at = (kp + q.sqrt() * L) / s1q
    tail_neg_lo = -(quad + (2 * arb.pi()).log() / 2 * m0 + lin
                    + 2 * Psi(u_at) * m0)
    return main + hull(tail_pos_lo + tail_neg_lo, arb(0))


def G_of(alpha, q, psi, kappa):
    return -psi * (1 - q) / 2 + Elog2cosh(psi) + alpha * ElogPsi_kappa(q, kappa)


def Pprime_bound(psi_lb, psi_ub, n=20000):
    """Upper bound for P'(psi) on [psi_lb, psi_ub] (Ding-Sun Lemma 7.4
    argument, prior side only, kappa-free): both monotone pieces
    evaluated at their extreme psi.  Real-cell Riemann - the 1/cosh^4
    integrand has complex poles at i(k+1/2)pi/s that trip adaptive
    probes at reduced budgets.  Dropping term2's tail (positive
    integrand) keeps the difference an upper bound."""
    s_lb = psi_lb.sqrt()
    s_ub = psi_ub.sqrt()
    term1 = riemann_E(lambda z: 2 / (s_lb * z).cosh() ** 4, n) \
        + 2 * gauss_tail_mass(L)
    term2 = riemann_E(lambda z: (2 * s_ub * z).cosh()
                      / (s_ub * z).cosh() ** 4, n)
    return term1 - term2


def dRdq_bound(q, alpha, kappa, n=20000):
    """Enclosure of dR/dq via the Stein-collapsed identity

        dR/dq = alpha E[ M'(u)^2 + 2 M(u)^2 M'(u) ] / (1-q)^2,

    derived by integrating the naive R/(1-q) + cross-term form by parts
    (the R/(1-q) piece cancels exactly; verified against a finite
    difference to 1e-15 in mpmath).  The integrand is positive, so R is
    increasing in q at every kappa - the general-kappa form of Ding-Sun
    Lemma 7.2 falls out for free.

    M' = M(M-u) in (0,1).  The per-cell M' enclosure intersects the
    ball form M(M-u) with [0,1].  Tails: integrand <= 1 + 2 M^2 with
    M <= 1 + u_+, a quadratic in |z| against Gaussian tail moments on
    the left; on the right M is exponentially small and the integrand
    is <= 1 + 2 M(u(L))^2 <= 2.
    """
    one = arb(1)

    def w(z):
        u = u_arb(z, q, kappa)
        m = mills_ball(u)
        mp = m * (m - u)
        # M' in (0,1) always; intersect to kill ball-dependency blowup
        lo_mp, hi_mp = endpoints(mp)
        lo_c = lo_mp if lo_mp > 0 else arb(0)
        hi_c = hi_mp if hi_mp < 1 else one
        if not (lo_c < hi_c):
            lo_c, hi_c = arb(0), one
        mp = lo_c.union(hi_c)
        return mp ** 2 + 2 * m ** 2 * mp

    main = riemann_E(w, n)
    kp = abs(kappa)
    s1q = (1 - q).sqrt()
    a0 = 1 + kp / s1q
    a1 = q.sqrt() / s1q
    # left tail: 1 + 2 (a0 + a1|z|)^2 against one-sided moments
    m0 = gauss_tail_mass(L) / 2
    m1 = phi(L)
    m2 = L * phi(L) + Psi(L)
    tail_left = ((1 + 2 * a0 ** 2) * m0 + 4 * a0 * a1 * m1
                 + 2 * a1 ** 2 * m2)
    uL = u_arb(L, q, kappa)
    _, hi_uL = endpoints(uL)
    mR = mills_ball(arb(hi_uL)) if hi_uL < 0 else arb(1)
    tail_right = (1 + 2 * mR ** 2) * m0
    body = hull(main, main + tail_left + tail_right)
    return alpha * body / (1 - q) ** 2


def _intersect(a, b):
    lo_a, hi_a = endpoints(a)
    lo_b, hi_b = endpoints(b)
    lo = lo_a if lo_a > lo_b else lo_b
    hi = hi_a if hi_a < hi_b else hi_b
    if not (lo < hi):
        return None
    return lo.union(hi)


def locate_fp(alpha, kappa, q_ball, n_iter=10):
    """Certified location of the fixed point of q -> P(R(q)) inside
    q_ball, for thin alpha and a kappa ball.

    First verifies the endpoints map inward (existence on the interval by
    the intermediate value theorem; with the slab's certified contraction
    the fixed point is also unique).  Then Picard-shrinks: each image
    intersected with the current ball still contains the fixed point, and
    the width contracts down to the kappa-driven floor.  Returns
    (q_located, psi_located) or None if the endpoint check fails."""
    lo, hi = endpoints(q_ball)
    v_lo = P_of(R_of(lo, alpha, kappa)) - lo
    v_hi = hi - P_of(R_of(hi, alpha, kappa))
    if not (v_lo > 0 and v_hi > 0):
        return None
    qb = q_ball
    for _ in range(n_iter):
        img = P_of_wide(R_of_wide(qb, alpha, kappa))
        nb = _intersect(img, qb)
        if nb is None:
            break
        lo_n, hi_n = endpoints(nb)
        lo_o, hi_o = endpoints(qb)
        if hi_n - lo_n > (hi_o - lo_o) * dec('0.9'):
            qb = nb
            break
        qb = nb
    return qb, R_of_wide(qb, alpha, kappa)


def locate_fp_deep(alpha, kappa, q_ball, n_iter=30):
    """locate_fp with the mean-value evaluators: the Picard floor drops
    from the plain-hull ~1e-3 to the mv ~1e-5, which the deep alpha_*
    bisections need.  Slab certificates keep the original locate_fp."""
    lo, hi = endpoints(q_ball)
    v_lo = P_of(R_of(lo, alpha, kappa)) - lo
    v_hi = hi - P_of(R_of(hi, alpha, kappa))
    if not (v_lo > 0 and v_hi > 0):
        return None
    qb = q_ball
    for _ in range(n_iter):
        img = P_of_mv(R_of_mv(qb, alpha, kappa))
        nb = _intersect(img, qb)
        if nb is None:
            break
        lo_n, hi_n = endpoints(nb)
        lo_o, hi_o = endpoints(qb)
        if hi_n - lo_n > (hi_o - lo_o) * dec('0.95'):
            qb = nb
            break
        qb = nb
    return qb, R_of_mv(qb, alpha, kappa)


def alpha_c_ball(kappa):
    """Certified Nakajima saddle-existence bound
    alpha_c(kappa) = 2/(pi E[(kappa-Z)_+^2]) over a kappa ball, via
    the closed form E[(kappa-Z)_+^2] = (1+kappa^2) Phi(kappa)
    + kappa phi(kappa) (integrate (kappa-z)^2 phi(z) by parts twice)."""
    Phi = (-kappa / arb(2).sqrt()).erfc() / 2
    phi_k = (-(kappa * kappa) / 2).exp() / (2 * arb.pi()).sqrt()
    return 2 / (arb.pi() * ((1 + kappa * kappa) * Phi + kappa * phi_k))


def locate_fp_bisect(alpha, kappa, q_ball, n_bis=40):
    """Contraction-free certified location of a fixed point of
    q -> P(R(q)) inside q_ball, for thin alpha and a kappa ball.

    The displacement d(q) = P(R(q)) - q is evaluated at thin q points
    and its certified sign steers a bisection.  The invariant is pure
    intermediate-value: d(lo) > 0 > d(hi) at all times, so [lo, hi]
    always contains a fixed point; bisection stops when a sign fails
    to resolve (the thin-evaluation floor) or at n_bis.  No
    contraction input anywhere - this is the locate step that survives
    past the rectangle product bound's wall near kappa = 0.12.
    Uniqueness is NOT provided here; supply it externally (the
    per-slab contraction where it holds, Nakajima's theorem for
    kappa >= 0 beyond it).  Returns (q_located, psi_located) or None
    if the endpoint check fails."""
    lo, hi = endpoints(q_ball)
    if not (P_of(R_of(lo, alpha, kappa)) - lo > 0):
        return None
    if not (hi - P_of(R_of(hi, alpha, kappa)) > 0):
        return None

    def sign_at(q):
        d = P_of(R_of(arb(q), alpha, kappa)) - q
        if d > 0:
            return 1
        if d < 0:
            return -1
        return 0

    for _ in range(n_bis):
        mid = (lo + hi) / 2
        s = sign_at(mid)
        if s > 0:
            lo = mid
            continue
        if s < 0:
            hi = mid
            continue
        # midpoint unresolved (the kappa-width floor): the quartiles
        # may still resolve and shrink the bracket by a quarter
        q1 = (3 * lo + hi) / 4
        q3 = (lo + 3 * hi) / 4
        s1 = sign_at(q1)
        s3 = sign_at(q3)
        if s3 > 0:
            lo = q3
            continue
        if s1 < 0:
            hi = q1
            continue
        moved = False
        if s1 > 0:
            lo = q1
            moved = True
        if s3 < 0:
            hi = q3
            moved = True
        if not moved:
            break
    qb = hull(arb(lo), arb(hi))
    return qb, R_of_mv(qb, alpha, kappa)


def G_mean_value_deep(alpha, qb, pb, kappa, n=4000):
    """G_mean_value with mv coefficients and finer G0 quadrature."""
    lo_q, hi_q = endpoints(qb)
    lo_p, hi_p = endpoints(pb)
    q_mid = (lo_q + hi_q) / 2
    p_mid = (lo_p + hi_p) / 2
    G0 = (-arb(p_mid) * (1 - arb(q_mid)) / 2
          + Elog2cosh(arb(p_mid), n=n)
          + alpha * ElogPsi_kappa(arb(q_mid), kappa, n=n))
    dGdpsi = (qb - P_of_mv(pb)) / 2
    dGdq = (arb(p_mid) - R_of_mv(qb, alpha, kappa)) / 2
    return G0 + dGdpsi * (pb - p_mid) + dGdq * (qb - q_mid)


def G_mean_value(alpha, qb, pb, kappa):
    """Enclosure of G over (qb x pb) that keeps the near-cancellation
    dG/dpsi = (q - P(psi))/2, dG/dq = (psi - R(q))/2 explicit: G at the
    midpoint plus box-coefficient mean-value terms.  Direct in kappa and
    alpha (their variation is genuine signal, not dependency loss)."""
    lo_q, hi_q = endpoints(qb)
    lo_p, hi_p = endpoints(pb)
    q_mid = (lo_q + hi_q) / 2
    p_mid = (lo_p + hi_p) / 2
    # Sequential mean-value split: G(q,psi) - G(qm,pm)
    #   = dG/dpsi(q, psi~)(psi - pm) + dG/dq(q~, pm)(q - qm),
    # so the psi coefficient ranges over the FULL q box (q, not q_mid;
    # using the midpoint understates the enclosure by ~1e-5) and the q
    # coefficient is evaluated at thin pm (pb would also be sound, just
    # wider).
    G0 = G_of(alpha, q_mid, p_mid, kappa)
    dGdpsi = (qb - P_of_wide(pb)) / 2
    dGdq = (arb(p_mid) - R_of_wide(qb, alpha, kappa)) / 2
    return G0 + dGdpsi * (pb - p_mid) + dGdq * (qb - q_mid)


def Mprime_ball(u):
    """M'(u) = M(u)(M(u) - u) in (0,1), as a real ball intersected with
    [0,1] (the analytic range) to kill ball-dependency blowup."""
    m = mills_ball(u)
    mp = m * (m - u)
    lo, hi = endpoints(mp)
    lo = lo if lo > 0 else arb(0)
    hi = hi if hi < 1 else arb(1)
    if not (lo < hi):
        return iv(arb(0), arb(1))
    return lo.union(hi)


def cond_amp_works_cert(alpha, qb, pb, kappa, n=8000):
    """Enclosure of Huang's amp-works LHS
    alpha E[sech^4(sqrt(psi) Z)] E[M'(u(Z))^2]/(1-q)^2.

    E[sech^4] is monotone decreasing in psi, so the psi-range enters as
    two thin Riemann evaluations at its endpoints; the M' factor is a
    plain Riemann with clamped balls.  Tails: sech^4 <= 1, M'^2 <= 1."""
    lo_p, hi_p = endpoints(pb)

    def e_th_at(pv):
        s = pv.sqrt()
        v = riemann_E(lambda z: (1 / (s * z).cosh()) ** 4, n)
        return hull(v, v + gauss_tail_mass(L))

    e_th = hull(e_th_at(arb(lo_p)), e_th_at(arb(hi_p)))
    e_f = riemann_E(lambda z: Mprime_ball(u_arb(z, qb, kappa)) ** 2, n)
    e_f = hull(e_f, e_f + gauss_tail_mass(L))
    return alpha * e_th * e_f / (1 - qb) ** 2


Z_WITNESS = '-0.6693'


def cond_local_concavity_cert(alpha, qb, pb, kappa, n=8000):
    """Certified lambda(z_hat) < 0 at the fixed witness z_hat = -0.6693
    (Huang's kappa = 0 witness; the lambda(z) < 0 basin is wide and moves
    slowly with kappa, and the condition needs only one witness).

    lambda(z) = z - alpha E[hf/(1 + m(z) hf)] - d0 with
    hf = M'/((1-q)(1-M')), m(z) = E[(z + cosh^2(sqrt(psi)Z))^-1] and
    d0 = -alpha E[M'(u)]/(1-q).  The two E-terms nearly cancel, so
    evaluating them as separate Riemann sums doubles the q-box width and
    the enclosure drowns for kappa below about -0.35.  Combine them
    inside one integrand: with t = M'(u), A = 1 - q, b = m - A,

      lambda(z) = z + alpha E[g],   g(t) = b t^2 / (A (A + b t)),

    since hf/(1+m hf) - t/A = -g.  For every fixed (m, A) the map
    t -> g is monotone (dg/dt = b t (2A + b t)/(A (A + b t)^2), and
    A + b t = A(1-t) + m t > 0, 2A + b t = A + (A + b t) > 0, so the
    sign of dg/dt is the sign of b), hence per z-cell g lies between its
    values at the clamped-M'-ball endpoints; the hull of the two
    endpoint balls covers the cell for either sign of b.  g is regular
    on the whole closed t in [0,1] -- g(0) = 0, g(1) = b/(A m) -- so no
    cap branch is needed, and |g| <= |b|/(A m) bounds the tails.  The m
    integrand is monotone decreasing in psi, so the psi-range enters as
    two thin evaluations; m integrand tail in (0, 1/(z_hat+1))."""
    zh = dec(Z_WITNESS)
    lo_p, hi_p = endpoints(pb)

    def m_at(pv):
        s = pv.sqrt()
        v = riemann_E(lambda z: 1 / (zh + (s * z).cosh() ** 2), n)
        return hull(v, v + gauss_tail_mass(L) / (zh + 1))

    m_z = hull(m_at(arb(lo_p)), m_at(arb(hi_p)))
    A = 1 - qb
    b = m_z - A

    def g_of(t):
        return b * t * t / (A * (A + b * t))

    def g_w(z):
        t = Mprime_ball(u_arb(z, qb, kappa))
        lo_t, hi_t = endpoints(t)
        e_lo = g_of(arb(lo_t))
        e_hi = g_of(arb(hi_t))
        return e_lo.union(e_hi)

    g_int = riemann_E(g_w, n)
    g_cap = endpoints(abs(b) / (A * m_z))[1]
    g_int = hull(g_int,
                 g_int + gauss_tail_mass(L) * g_cap.union(-g_cap))
    return zh + alpha * g_int


def verify_slab(slab, n_alpha_sub=8):
    """Certify Condition km-well-defd on one kappa slab.

    Structure (all quantified over the kappa ball):
      contraction  sup P' * sup |dR/dq| < 1 on the full rectangle, so
                   q -> P(R(q, alpha)) is a uniform contraction there;
      sweep        for each of n_alpha_sub alpha sub-intervals, the
                   endpoints of the q rectangle map strictly inward
                   (existence inside for every alpha; uniqueness from
                   the contraction);
      crossing     at the thin alpha endpoints, locate the fixed point
                   by certified Picard shrink and resolve the sign of G
                   through the mean-value form: G(alpha_lb) > 0 and
                   G(alpha_ub) < 0.  With dG_*/dalpha = E log Psi < 0
                   (strict), alpha_*(kappa) is the unique root and lies
                   in [alpha_lb, alpha_ub] for every kappa in the slab.
    """
    kappa = iv(dec(slab['kappa_lo']), dec(slab['kappa_hi']))
    a_lb, a_ub = dec(slab['alpha_lb']), dec(slab['alpha_ub'])
    alpha = iv(a_lb, a_ub)
    q_lb, q_ub = dec(slab['q_lb']), dec(slab['q_ub'])
    q_ball = iv(q_lb, q_ub)

    checks = []

    def chk(name, val, want):
        ok = report(f"  {name}", val, want)
        lo, hi = endpoints(val)
        checks.append({'name': name, 'want': want,
                       'lo': str(lo), 'hi': str(hi), 'ok': ok,
                       'packet': _packet_or_none(val)})
        return ok

    ok = True

    # Contraction on this slab's own rectangle: sup_q (P o R)'(q) <=
    # sup_{psi in R(rect)} P'(psi) * sup_q dR/dq.  This is Huang's
    # condition verbatim (his sup is over the same q interval).  The
    # product bound clears here because the slab's kappa width (5e-4)
    # keeps the dR/dq enclosure tight - the batch-hull version fails
    # only through its 100x wider kappa ball.
    R_slab = R_of_wide(q_ball, alpha, kappa)
    lo_Rs, hi_Rs = endpoints(R_slab)
    pp_s = Pprime_bound(lo_Rs - dec('0.005'), hi_Rs + dec('0.005'))
    dr_s = dRdq_bound(q_ball, alpha, kappa)
    _, pp_s_hi = endpoints(pp_s)
    _, dr_s_hi = endpoints(abs(dr_s))
    contraction_slab = pp_s_hi * dr_s_hi
    ok &= chk("1 - contraction (slab)", 1 - contraction_slab, '>0')

    # endpoint-inward sweep over alpha sub-intervals
    lo_q, hi_q = endpoints(q_ball)
    for j in range(n_alpha_sub):
        a_j = a_lb + (a_ub - a_lb) * j / n_alpha_sub
        a_j1 = a_lb + (a_ub - a_lb) * (j + 1) / n_alpha_sub
        aj = iv(a_j, a_j1)
        v_lo = P_of(R_of(lo_q, aj, kappa)) - lo_q
        v_hi = hi_q - P_of(R_of(hi_q, aj, kappa))
        ok &= chk(f"inward lo alpha[{j}]", v_lo, '>0')
        ok &= chk(f"inward hi alpha[{j}]", v_hi, '>0')

    # crossing at the thin alpha endpoints
    loc_lb = locate_fp(a_lb, kappa, q_ball)
    loc_ub = locate_fp(a_ub, kappa, q_ball)
    if loc_lb is None or loc_ub is None:
        chk("fixed point located", arb(-1), '>0')
        rec = {'slab': slab, 'checks': checks, 'ok': False}
        return False, rec
    qb_lb, pb_lb = loc_lb
    qb_ub, pb_ub = loc_ub
    G_lo = G_mean_value(a_lb, qb_lb, pb_lb, kappa)
    ok &= chk("G(alpha_lb)", G_lo, '>0')
    G_hi = G_mean_value(a_ub, qb_ub, pb_ub, kappa)
    ok &= chk("G(alpha_ub)", G_hi, '<0')

    # Huang's amp-works and local-concavity at the fixed point of
    # alpha_*(kappa).  The sweep checks put q_*(alpha) inside the q
    # rectangle for every alpha, and psi_* = R(q_*, alpha_*) lies in
    # R_slab, the slab's own derived range computed above -- the sound
    # domain, and far thinner than the batch-hull psi range (whose
    # width drowned the lambda enclosure below kappa = -0.35).
    psi_slab = hull(arb(lo_Rs), arb(hi_Rs))
    amp = cond_amp_works_cert(alpha, q_ball, psi_slab, kappa)
    ok &= chk("1 - amp-works", 1 - amp, '>0')
    lam_w = cond_local_concavity_cert(alpha, q_ball, psi_slab, kappa)
    ok &= chk("lambda(z_hat)", lam_w, '<0')

    lq1, hq1 = endpoints(qb_lb)
    lq2, hq2 = endpoints(qb_ub)
    lp1, hp1 = endpoints(pb_lb)
    lp2, hp2 = endpoints(pb_ub)
    rec = {'slab': slab,
           'contraction_slab_ub': str(contraction_slab),
           'q_located_alb': [str(lq1), str(hq1)],
           'q_located_aub': [str(lq2), str(hq2)],
           'psi_located_alb': [str(lp1), str(hp1)],
           'psi_located_aub': [str(lp2), str(hp2)],
           'checks': checks, 'ok': bool(ok)}
    return ok, rec


def verify_slab_nak(slab, n_alpha_sub=8):
    """Certify the fixed-point condition on a kappa slab past the
    rectangle product bound's wall (kappa >= 0 only).

    The box contraction is replaced, not weakened:
      premise      alpha_ub < alpha_c(kappa) certified over the kappa
                   ball (closed-form ball evaluation of Nakajima's
                   bound), so his saddle-uniqueness theorem covers
                   every (alpha, kappa) in the slab;
      sweep        endpoint-inward on n_alpha_sub alpha sub-intervals
                   (existence; already contraction-free);
      crossing     locate_fp_bisect at the thin alpha endpoints (sign
                   bisection, no contraction input) and the mean-value
                   G signs;
      contraction  sup (P o R)' < 1 on the LOCATED intervals only --
                   the local form Huang's perturbation argument
                   consumes, which survives past the box wall;
      conditions   amp-works and lambda(z_hat) on the slab rectangle
                   with the slab-local psi range, unchanged.
    """
    kappa = iv(dec(slab['kappa_lo']), dec(slab['kappa_hi']))
    a_lb, a_ub = dec(slab['alpha_lb']), dec(slab['alpha_ub'])
    alpha = iv(a_lb, a_ub)
    q_lb, q_ub = dec(slab['q_lb']), dec(slab['q_ub'])
    q_ball = iv(q_lb, q_ub)

    checks = []

    def chk(name, val, want):
        ok = report(f"  {name}", val, want)
        lo, hi = endpoints(val)
        checks.append({'name': name, 'want': want,
                       'lo': str(lo), 'hi': str(hi), 'ok': ok,
                       'packet': _packet_or_none(val)})
        return ok

    ok = True

    ac = alpha_c_ball(kappa)
    ok &= chk("alpha_c - alpha_ub (Nakajima premise)", ac - arb(a_ub),
              '>0')

    lo_q, hi_q = endpoints(q_ball)
    for j in range(n_alpha_sub):
        a_j = a_lb + (a_ub - a_lb) * j / n_alpha_sub
        a_j1 = a_lb + (a_ub - a_lb) * (j + 1) / n_alpha_sub
        aj = iv(a_j, a_j1)
        v_lo = P_of(R_of(lo_q, aj, kappa)) - lo_q
        v_hi = hi_q - P_of(R_of(hi_q, aj, kappa))
        ok &= chk(f"inward lo alpha[{j}]", v_lo, '>0')
        ok &= chk(f"inward hi alpha[{j}]", v_hi, '>0')

    # The locate / located-contraction / G block runs per kappa
    # sub-ball: the kappa width floors the displacement's sign
    # resolution (located width ~ 20x the kappa width), and each check
    # quantifies over its sub-ball, so the union covers the slab.
    k_lo_d, k_hi_d = dec(slab['kappa_lo']), dec(slab['kappa_hi'])
    n_k = 8
    q_loc_rec = {}
    for j in range(n_k):
        ks = iv(k_lo_d + (k_hi_d - k_lo_d) * j / n_k,
                k_lo_d + (k_hi_d - k_lo_d) * (j + 1) / n_k)
        loc_lb = locate_fp_bisect(a_lb, ks, q_ball)
        loc_ub = locate_fp_bisect(a_ub, ks, q_ball)
        if loc_lb is None or loc_ub is None:
            chk(f"fixed point located k[{j}]", arb(-1), '>0')
            rec = {'slab': slab, 'lane': 'nakajima', 'checks': checks,
                   'ok': False}
            return False, rec
        qb_lb, pb_lb = loc_lb
        qb_ub, pb_ub = loc_ub
        if j in (0, n_k - 1):
            q_loc_rec[f'k{j}_alb'] = [str(x) for x in endpoints(qb_lb)]
            q_loc_rec[f'k{j}_aub'] = [str(x) for x in endpoints(qb_ub)]
        for tag, a_thin, qb in (('alb', a_lb, qb_lb),
                                ('aub', a_ub, qb_ub)):
            R_loc = R_of_wide(qb, arb(a_thin), ks)
            lo_Rl, hi_Rl = endpoints(R_loc)
            pp_l = Pprime_bound(lo_Rl - dec('0.002'),
                                hi_Rl + dec('0.002'))
            dr_l = dRdq_bound(qb, arb(a_thin), ks)
            prod = endpoints(pp_l)[1] * endpoints(abs(dr_l))[1]
            ok &= chk(f"1 - contraction (located, {tag}, k[{j}])",
                      1 - prod, '>0')
        G_lo = G_mean_value(a_lb, qb_lb, pb_lb, ks)
        ok &= chk(f"G(alpha_lb) k[{j}]", G_lo, '>0')
        G_hi = G_mean_value(a_ub, qb_ub, pb_ub, ks)
        ok &= chk(f"G(alpha_ub) k[{j}]", G_hi, '<0')

    R_slab = R_of_wide(q_ball, alpha, kappa)
    lo_Rs, hi_Rs = endpoints(R_slab)
    psi_slab = hull(arb(lo_Rs), arb(hi_Rs))
    amp = cond_amp_works_cert(alpha, q_ball, psi_slab, kappa)
    ok &= chk("1 - amp-works", 1 - amp, '>0')
    lam_w = cond_local_concavity_cert(alpha, q_ball, psi_slab, kappa)
    ok &= chk("lambda(z_hat)", lam_w, '<0')

    rec = {'slab': slab, 'lane': 'nakajima',
           'q_located_sample': q_loc_rec,
           'checks': checks, 'ok': bool(ok)}
    return ok, rec


def main():
    slabs_path = sys.argv[1] if len(sys.argv) > 1 else 'slabs.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'slab_results.json'
    with open(slabs_path) as fh:
        slabs = json.load(fh)
    print("global checks over the batch hull:", flush=True)
    g_ok, g_rec, psi_range = verify_global(slabs)
    print(f"  -> {'PASS' if g_ok else 'FAIL'}", flush=True)
    results = [{'global': g_rec}]
    n_ok = 0
    for i, slab in enumerate(slabs):
        print(f"slab {i}: kappa [{slab['kappa_lo']}, {slab['kappa_hi']}]",
              flush=True)
        ok, rec = verify_slab(slab)
        results.append(rec)
        n_ok += ok
        print(f"  -> {'PASS' if ok else 'FAIL'}", flush=True)
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=1)
    print(f"global {'PASS' if g_ok else 'FAIL'}; "
          f"{n_ok}/{len(slabs)} slabs verified -> {out_path}")
    if n_ok < len(slabs) or not g_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
