"""The Hessian of Huang's first-moment functional at the distinguished
point (1, 0), at general kappa.

His Lemma sS-2deriv-formula states the kappa = 0 case; the display it
is specialized from (the second-derivative expansion at (1,0), before
"specializing further to kappa = 0") gives, at any kappa, the same
quadratic form

  <M, u^{x2}> = -(I_1 u1^2 + 2 I_2 u1 u2 + I_3 u2^2)
                + C_1 a^2 + C_2 a b + C_3 b^2,
  a = I_1 u1 + I_2 u2,   b = I_2 u1 + I_3 u2,

with prior-side integrals (dH ~ N(0, psi0), M = tanh(dH), p4 = E M^4)

  I_1 = psi0(1-q0) - 2 psi0^2 (1 - 4 q0 + 3 p4),
  I_2 = psi0 (1 - 4 q0 + 3 p4),
  I_3 = q0 - p4,

and constraint-side constants (hH ~ N(0, q0), u = (kappa - hH)/s,
s = sqrt(1-q0), N = M_mills(u)/s, F' = -M'_mills(u)/(1-q0),
G0 = (kappa - hH - (1-q0) N)/(1-q0), Q = -hH/q0 + G0)

  C_1 = (alpha/psi0^2) E[F' N^2]
  C_2 = -(2 alpha/psi0) E[F' N Q] + 2/(1-q0)
  C_3 = alpha E[F' Q^2] + 2 alpha E[N hH]/(q0(1-q0))
        - alpha E[N (kappa - hH - (1-q0) N)] (3/(1-q0)^2 + 1/(q0(1-q0)))

Both kappa = 0 reductions check out symbolically: the 2/(1-q0) term
uses alpha E[N^2] = psi0 (true at every kappa - the same identity
behind the s-unbounded directions), and the psi0/q0 constant in his
C_3 comes from alpha E[N hH] = -q0(1-q0)psi0, which holds only at
kappa = 0 and is why his closed forms do not transfer.

Anchors: at kappa = 0 this must land in his certified intervals
(C_1 in [-0.7193, -0.7165], C_2 in [5.0439, 5.0568],
C_3 in [1.1345, 1.1526]) and the assembled M must match the
finite-difference Hessian of S_*(l1, l2) from huang2var.py
(-0.1571, -0.1011; -0.1011, -0.0714) - the same object computed two
ways.
"""

from mpmath import mp, mpf, exp, log, sqrt, tanh, erfc, pi, quad

mp.dps = 25

ZCUT = mpf(12)
SPLIT = [-ZCUT, -6, -2, 0, 2, 6, ZCUT]


def phi(z):
    return exp(-z * z / 2) / sqrt(2 * pi)


def Psi(u):
    return erfc(u / sqrt(2)) / 2


def M_mills(u):
    return phi(u) / Psi(u)


def Mp_mills(u):
    m = M_mills(u)
    return m * (m - u)


def gauss_E(f):
    return quad(lambda z: f(z) * phi(z), SPLIT)


def hessian(alpha, q0, psi0, kappa):
    alpha, q0, psi0, kappa = (mpf(str(x)) for x in (alpha, q0, psi0,
                                                    kappa))
    s = sqrt(1 - q0)
    sq = sqrt(q0)
    spsi = sqrt(psi0)

    p4 = gauss_E(lambda z: tanh(spsi * z) ** 4)
    I1 = psi0 * (1 - q0) - 2 * psi0 ** 2 * (1 - 4 * q0 + 3 * p4)
    I2 = psi0 * (1 - 4 * q0 + 3 * p4)
    I3 = q0 - p4

    def parts(z):
        hH = sq * z
        u = (kappa - hH) / s
        N = M_mills(u) / s
        Fp = -Mp_mills(u) / (1 - q0)
        G0 = (kappa - hH - (1 - q0) * N) / (1 - q0)
        Q = -hH / q0 + G0
        return hH, N, Fp, Q

    E_FpN2 = gauss_E(lambda z: (lambda hH, N, Fp, Q: Fp * N * N)(*parts(z)))
    E_FpNQ = gauss_E(lambda z: (lambda hH, N, Fp, Q: Fp * N * Q)(*parts(z)))
    E_FpQ2 = gauss_E(lambda z: (lambda hH, N, Fp, Q: Fp * Q * Q)(*parts(z)))
    E_NH = gauss_E(lambda z: (lambda hH, N, Fp, Q: N * hH)(*parts(z)))
    E_Nk = gauss_E(
        lambda z: (lambda hH, N, Fp, Q:
                   N * (kappa - hH - (1 - q0) * N))(*parts(z)))
    E_N2 = gauss_E(lambda z: (lambda hH, N, Fp, Q: N * N)(*parts(z)))

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
    M12 = (form(1, 1) - M11 - M22) / 2
    return {
        'I': (I1, I2, I3), 'C': (C1, C2, C3),
        'M11': M11, 'M12': M12, 'M22': M22,
        'det': M11 * M22 - M12 ** 2,
        'tilt_identity': alpha * E_N2 - psi0,
    }


if __name__ == '__main__':
    r = hessian('0.83307859973957923369', '0.56394908008456163626',
                '2.5763513190756745721', '0')
    print('kappa = 0 anchors:')
    print('  I1, I2, I3 =', *(mp.nstr(x, 10) for x in r['I']))
    print('  C1, C2, C3 =', *(mp.nstr(x, 8) for x in r['C']))
    print('  certified   C1 [-0.7193,-0.7165]  C2 [5.0439,5.0568]  '
          'C3 [1.1345,1.1526]')
    print('  M11, M12, M22 =', mp.nstr(r['M11'], 8),
          mp.nstr(r['M12'], 8), mp.nstr(r['M22'], 8))
    print('  FD Hessian      -0.15708   -0.10112   -0.071446')
    print('  det =', mp.nstr(r['det'], 8), ' (certified >= 0.0002246)')
    print('  alpha E[N^2] - psi0 =', mp.nstr(r['tilt_identity'], 3))
