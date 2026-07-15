"""Ball-arithmetic enclosure of the fixed-tilt Hessian at the
distinguished point, at a fixed kappa.

Takes tight parameter boxes (alpha from a deep bisection, (q0, psi0)
from locate_fp_deep at the interval midpoint) and evaluates the
general-kappa M-matrix of hessian_kappa.py in arb: the prior-side
integrals via p4 = E tanh^4 (mean-value Riemann), the constraint-side
constants via real-cell Riemann with monotone Mills enclosures, and
the negativity verdict M11 < 0, det M > 0.

Tails: the hH-side integrands are bounded by polynomials in |z| times
N-powers; on the right tail (z >> 0) u is large negative so N and F'
vanish (bounded by their values at the cut via monotone endpoints); on
the left tail N grows linearly, F' stays in (-1/(1-q0), 0), and the
integrands are bounded by explicit quadratics against Gaussian
moments.  For the first version the z-cut is pushed to L = 10 and the
crude bound max(integrand at cut) * tail mass is used; every discarded
term is enclosed, not dropped.

Run: python cert_hessian.py 0.05 alpha_lo alpha_hi
     (alpha bounds from results/alpha_star_*.json)
"""

import sys

import certify_km_slab as C
from certify_km_slab import (arb, dec, iv, hull, endpoints, phi, Psi,
                             mills_ball, Mprime_ball, riemann_E,
                             riemann_mv, gauss_tail_mass, u_arb, report)

L = C.L


def p4_of(psi, n=4000):
    """E tanh^4(sqrt(psi) Z), mean-value Riemann, monotone in psi via
    thin endpoints."""
    lo_p, hi_p = endpoints(psi)

    def at(pv):
        s = pv.sqrt()

        def dw(cell):
            t = (s * cell).tanh()
            return 4 * s * t ** 3 * (1 - t * t)

        v = riemann_mv(lambda z: (s * z).tanh() ** 4, dw, n)
        return hull(v, v + gauss_tail_mass(L))

    return hull(at(arb(lo_p)), at(arb(hi_p)))


def w4_of(psi, q0, n=4000):
    """E[1 - 4 tanh^2 + 3 tanh^4](sqrt(psi) Z) - (1 - 4 q0 + 3 p4) as a
    SINGLE integrand, so the near-cancellation (value ~0.066 from O(1)
    pieces) happens pointwise instead of between independent balls.
    Not monotone in psi pointwise; hull over thin psi endpoints with a
    derivative-in-psi correction is overkill - the psi width enters
    linearly with slope |dE/dpsi| <= E[|z (...)'|]/(2 sqrt(psi)),
    bounded here crudely by evaluating at both endpoints and hulling
    (the function is smooth and the endpoints bracket to first order;
    for the certificate the hull of two enclosures plus their spread is
    sound because E(psi) is continuous and the interval is tiny).

    Soundness note: hulling two thin evaluations encloses E only at the
    endpoints; to cover the interior the hull is widened by the maximal
    slope times the width, with slope bound 1 (|integrand'| <= 8|z|
    against phi gives E-slope well under 1 for psi ~ 2.6)."""
    lo_p, hi_p = endpoints(psi)

    def at(pv):
        s = pv.sqrt()

        def w(z):
            t = (s * z).tanh()
            t2 = t * t
            return 1 - 4 * t2 + 3 * t2 * t2

        def dw(cell):
            t = (s * cell).tanh()
            return s * (1 - t * t) * (-8 * t + 12 * t ** 3)

        v = riemann_mv(w, dw, n)
        t = gauss_tail_mass(L) * 4
        return v + t.union(-t)

    # subdivide psi: hull of thin evaluations covers the interval up to
    # slope * sub-width.  Slope bound 1 with a factor-2 margin:
    # |dE/dpsi| <= max_t |w'(t)|(1-t^2) * E|z| / (2 sqrt(psi))
    #           <= 2.1 * 0.7979 / (2 * 1.6) < 0.55.
    n_sub = 32
    lo_a, hi_a = arb(lo_p), arb(hi_p)
    vals = None
    for j in range(n_sub + 1):
        pv = lo_a + (hi_a - lo_a) * j / n_sub
        v = at(pv)
        vals = v if vals is None else hull(vals, v)
    slack = (hi_a - lo_a) / n_sub
    return vals + slack.union(-slack)


def w22_of(psi, n=4000):
    """E[t^2 (1 - t^2)](sqrt(psi) Z) = q0 - p4 as one integrand
    (correlated), psi-subdivided like w4_of."""
    lo_p, hi_p = endpoints(psi)

    def at(pv):
        s = pv.sqrt()

        def w(z):
            t = (s * z).tanh()
            return t * t * (1 - t * t)

        def dw(cell):
            t = (s * cell).tanh()
            return s * (1 - t * t) * (2 * t - 4 * t ** 3)

        v = riemann_mv(w, dw, n)
        t = gauss_tail_mass(L)
        return v + t.union(-t)

    n_sub = 32
    lo_a, hi_a = arb(lo_p), arb(hi_p)
    vals = None
    for j in range(n_sub + 1):
        pv = lo_a + (hi_a - lo_a) * j / n_sub
        v = at(pv)
        vals = v if vals is None else hull(vals, v)
    slack = (hi_a - lo_a) / n_sub
    return vals + slack.union(-slack)


def hessian_cert(alpha, q0, psi0, kappa, n=8000):
    """Enclosures of (I1, I2, I3, C1, C2, C3, M11, M12, M22, det)."""
    s1q = (1 - q0).sqrt()
    sq = q0.sqrt()

    # I's with the near-cancellations evaluated as single integrands
    # (q0 = P(psi0) at the fixed point, so 1 - 4 q0 + 3 p4 and q0 - p4
    # are E[1 - 4 t^2 + 3 t^4] and E[t^2(1 - t^2)] over the psi0 box)
    w4 = w4_of(psi0, q0, n=max(2000, n // 2))
    I1 = psi0 * (1 - q0) - 2 * psi0 ** 2 * w4
    I2 = psi0 * w4
    I3 = w22_of(psi0, n=max(2000, n // 2))

    def parts(z):
        hH = sq * z
        u = u_arb(z, q0, kappa)          # (kappa - sq z)/s1q
        N = mills_ball(u) / s1q
        Fp = -Mprime_ball(u) / (1 - q0)
        G0 = (kappa - hH - (1 - q0) * N) / (1 - q0)
        Q = -hH / q0 + G0
        return hH, N, Fp, Q

    up = -sq / s1q                       # du/dz

    def parts_d(cell):
        """(hH, N, Fp, Q) plus their z-derivatives on a cell ball."""
        hH = sq * cell
        u = u_arb(cell, q0, kappa)
        m = mills_ball(u)
        mp = Mprime_ball(u)
        mpp = mp * (m - u) + m * (mp - 1)     # M'' enclosure
        N = m / s1q
        dN = mp * up / s1q
        Fp = -mp / (1 - q0)
        dFp = -mpp * up / (1 - q0)
        G0 = (kappa - hH - (1 - q0) * N) / (1 - q0)
        Q = -hH / q0 + G0
        dG0 = (-sq - (1 - q0) * dN) / (1 - q0)
        dQ = -sq / q0 + dG0
        return hH, N, Fp, Q, dN, dFp, dQ

    # Rigorous tails.  Every integrand is degree <= 2 in |z| with
    # bounded prefactors: on the left tail (z <= -L, u >= 0)
    #   N <= n0 + n1|z|,  n0 = (1 + kappa_+/s1q)/s1q, n1 = sq/s1q^2
    # (via M(u) <= 1 + u for u >= 0), and
    #   |Q| <= qc0 + qc1|z|
    # with qc0 = |kappa|/(1-q0) + n0, qc1 = sq/q0 + sq/(1-q0) + n1;
    # F' in (0, 1/(1-q0)); |hH| = sq|z|.  On the right tail u < 0 and
    # N there is below its left-tail bound, so the same coefficients
    # cover both sides.  Quadratics fold against the two-sided moments
    # m0 = 2 Psi(L), m1 = 2 phi(L), m2 = 2(L phi(L) + Psi(L)).
    kp = abs(kappa)
    n0 = (1 + kp / s1q) / s1q + 1 / s1q
    n1 = sq / s1q ** 2
    qc0 = kp / (1 - q0) + n0
    qc1 = sq / q0 + sq / (1 - q0) + n1
    Fp_hi = 1 / (1 - q0)
    m0 = 2 * Psi(arb(L))
    m1 = 2 * phi(arb(L))
    m2 = 2 * (arb(L) * phi(arb(L)) + Psi(arb(L)))

    def quad_tail(c0, c1, c2):
        t = c0 * m0 + c1 * m1 + c2 * m2
        return t.union(-t)

    def E_mv(w_mid, dw_cell, c0, c1, c2):
        return riemann_mv(w_mid, dw_cell, n) + quad_tail(c0, c1, c2)

    E_FpN2 = E_mv(
        lambda z: parts(z)[2] * parts(z)[1] ** 2,
        lambda cell: (lambda hH, N, Fp, Q, dN, dFp, dQ:
                      dFp * N * N + Fp * 2 * N * dN)(*parts_d(cell)),
        Fp_hi * n0 ** 2, Fp_hi * 2 * n0 * n1, Fp_hi * n1 ** 2)
    E_FpNQ = E_mv(
        lambda z: (lambda hH, N, Fp, Q: Fp * N * Q)(*parts(z)),
        lambda cell: (lambda hH, N, Fp, Q, dN, dFp, dQ:
                      dFp * N * Q + Fp * (dN * Q + N * dQ)
                      )(*parts_d(cell)),
        Fp_hi * n0 * qc0, Fp_hi * (n0 * qc1 + n1 * qc0),
        Fp_hi * n1 * qc1)
    E_FpQ2 = E_mv(
        lambda z: (lambda hH, N, Fp, Q: Fp * Q * Q)(*parts(z)),
        lambda cell: (lambda hH, N, Fp, Q, dN, dFp, dQ:
                      dFp * Q * Q + Fp * 2 * Q * dQ)(*parts_d(cell)),
        Fp_hi * qc0 ** 2, Fp_hi * 2 * qc0 * qc1, Fp_hi * qc1 ** 2)
    E_NH = E_mv(
        lambda z: (lambda hH, N, Fp, Q: N * hH)(*parts(z)),
        lambda cell: (lambda hH, N, Fp, Q, dN, dFp, dQ:
                      dN * hH + N * sq)(*parts_d(cell)),
        arb(0), n0 * sq, n1 * sq)
    # |N(kappa - hH - (1-q0)N)| <= N(kp + sq|z| + (1-q0)N):
    # degree 2 with coefficients below
    E_Nk = E_mv(
        lambda z: (lambda hH, N, Fp, Q:
                   N * (kappa - hH - (1 - q0) * N))(*parts(z)),
        lambda cell: (lambda hH, N, Fp, Q, dN, dFp, dQ:
                      dN * (kappa - hH - (1 - q0) * N)
                      + N * (-sq - (1 - q0) * dN))(*parts_d(cell)),
        n0 * kp + (1 - q0) * n0 ** 2,
        n0 * sq + n1 * kp + (1 - q0) * 2 * n0 * n1,
        n1 * sq + (1 - q0) * n1 ** 2)

    C1 = alpha / psi0 ** 2 * E_FpN2
    C2 = -(2 * alpha / psi0) * E_FpNQ + 2 / (1 - q0)
    C3 = (alpha * E_FpQ2 + 2 * alpha * E_NH / (q0 * (1 - q0))
          - alpha * E_Nk * (3 / (1 - q0) ** 2 + 1 / (q0 * (1 - q0))))

    def form(u1, u2):
        a = I1 * u1 + I2 * u2
        b = I2 * u1 + I3 * u2
        prior = I1 * u1 ** 2 + 2 * I2 * u1 * u2 + I3 * u2 ** 2
        return -prior + C1 * a * a + C2 * a * b + C3 * b * b

    M11 = form(1, 0)
    M22 = form(0, 1)
    M12 = (form(arb(1), arb(1)) - M11 - M22) / 2
    det = M11 * M22 - M12 ** 2
    return dict(I=(I1, I2, I3), C=(C1, C2, C3),
                M11=M11, M12=M12, M22=M22, det=det)


def main():
    kappa = dec(sys.argv[1])
    a_lo, a_hi = dec(sys.argv[2]), dec(sys.argv[3])
    alpha = iv(a_lo, a_hi)

    k = float(sys.argv[1])
    if k == 0.0:
        # the Ding-Sun certified rectangle IS the tight box at kappa=0
        from core import Q_LB, Q_UB, PSI_LB, PSI_UB
        qb = iv(Q_LB, Q_UB)
        pb = iv(PSI_LB, PSI_UB)
        lo_q, hi_q = endpoints(qb)
        lo_p, hi_p = endpoints(pb)
        print(f"boxes (DS rectangle): q width {float(hi_q - lo_q):.2g}, "
              f"psi width {float(hi_p - lo_p):.2g}")
        h = hessian_cert(alpha, qb, pb, kappa)
        finish(h, sys.argv[1])
        return

    # locate (q0, psi0) at the alpha interval, deep; the certified
    # q bracket comes from the strip CSV (optional argv[5] selects the
    # lane file, e.g. certified_intervals_nak.csv past the wall)
    import csv
    intervals = sys.argv[5] if len(sys.argv) > 5 else \
        'results/certified_intervals.csv'
    row = None
    with open(intervals, newline='') as fh:
        for r in csv.DictReader(fh):
            if float(r['kappa_lo']) <= k <= float(r['kappa_hi']):
                row = r
                break
    if row is None:
        print(f"kappa {k} not inside {intervals}")
        raise SystemExit(2)
    n_loc = int(sys.argv[4]) if len(sys.argv) > 4 else 16000
    q_seed = iv(dec(row['q_lb']), dec(row['q_ub']))
    qb = q_seed
    for _ in range(40):
        img = C.P_of_mv(C.R_of_mv(qb, alpha, kappa, n=n_loc), n=n_loc)
        nb = C._intersect(img, qb)
        if nb is None:
            break
        lo_n, hi_n = endpoints(nb)
        lo_o, hi_o = endpoints(qb)
        if hi_n - lo_n > (hi_o - lo_o) * dec('0.98'):
            qb = nb
            break
        qb = nb
    pb = C.R_of_mv(qb, alpha, kappa, n=n_loc)
    lo_q, hi_q = endpoints(qb)
    lo_p, hi_p = endpoints(pb)
    print(f"boxes: q width {float(hi_q - lo_q):.2g}, "
          f"psi width {float(hi_p - lo_p):.2g}")

    h = hessian_cert(alpha, qb, pb, kappa)
    finish(h, sys.argv[1])


def finish(h, kappa_str):
    ok = True
    ok &= report("M11", h['M11'], '<0')
    ok &= report("det M", h['det'], '>0')
    for nm in ('I', 'C'):
        print(' ', nm, '=', ', '.join(str(x) for x in h[nm]))
    print(' M12 =', h['M12'], ' M22 =', h['M22'])
    # explicit endpoints, quotable in the paper (the short arb display
    # can round a strictly positive ball's midpoint to zero)
    for nm in ('M11', 'M12', 'M22', 'det'):
        lo, hi = endpoints(h[nm])
        print(' %s endpoints = [%s, %s]' % (nm, lo, hi))
    print('fixed-tilt local maximality at kappa =', kappa_str,
          ':', 'CERTIFIED' if ok else 'NOT RESOLVED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
