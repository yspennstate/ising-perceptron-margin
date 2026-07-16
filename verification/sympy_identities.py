"""Symbolic verification of every algebraic identity the certificates
lean on.  Each check reduces a difference to zero (or a factorization
to an exactly matching form) in sympy; a nonzero residue fails the
battery.  These are the identities whose derivations appear in the
paper and in code comments --- the battery keeps them tied to the code.

Run: python sympy_identities.py
"""
import sympy as sp


def check(name, expr):
    ok = sp.simplify(expr) == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main():
    ok = True
    u, x, z, t, A, b, m, q, kap, a1, a2, s, psi, D = sp.symbols(
        'u x z t A b m q kappa a1 a2 s psi D', real=True)
    M = sp.Function('M')

    # 1. Mills derivative: M' = M(M - u) for M = phi/Psi (hazard form),
    #    from phi' = -u phi and Psi' = -phi.
    phi = sp.exp(-u ** 2 / 2) / sp.sqrt(2 * sp.pi)
    Psi = sp.Rational(1, 2) * sp.erfc(u / sp.sqrt(2))
    Mills = phi / Psi
    ok &= check('mills_derivative',
                sp.diff(Mills, u) - Mills * (Mills - u))

    # 2. M'' from the audit's enclosure: M'' = M'(M-u) + M(M'-1),
    #    substituting M' = M(M-u).
    Mp = Mills * (Mills - u)
    ok &= check('mills_second_derivative',
                sp.diff(Mills, u, 2) - (Mp * (Mills - u)
                                        + Mills * (Mp - 1)))

    # 3. The lambda reformulation: hf/(1 + m hf) - t/A = -g with
    #    g = b t^2/(A(A + b t)), hf = t/(A(1-t)), b = m - A.
    hf = t / (A * (1 - t))
    g = b * t ** 2 / (A * (A + b * t))
    ok &= check('lambda_identity',
                sp.together(hf / (1 + m * hf) - t / A
                            + g.subs(b, m - A)))

    # 4. g(1) = b/(A m) with b = m - A.
    ok &= check('lambda_g_at_one',
                (g.subs(t, 1) - b / (A * (A + b))).subs(b, m - A)
                * (A * m))          # clears the denominator exactly

    # 5. dg/dt factors as t b (2A + b t) / (A (A + b t)^2): the sign of
    #    dg/dt is the sign of b since A + b t = A(1-t) + m t > 0.
    dg = sp.diff(g, t)
    target = t * b * (2 * A + b * t) / (A * (A + b * t) ** 2)
    ok &= check('lambda_monotonicity_factorization', sp.together(dg - target))

    # 6. Nakajima's closed form: E[(kappa - Z)_+^2]
    #    = (1 + kappa^2) Phi(kappa) + kappa phi(kappa) =: F(kappa).
    #    Fully symbolic chain (differentiation under the integral is
    #    standard for these tails): d/dk E[(k-Z)_+^2] = 2 E[(k-Z)_+] and
    #    d/dk E[(k-Z)_+] = Phi(k), so it suffices that F'' = 2 Phi with
    #    F and F' vanishing at k -> -infinity, which pins F to the
    #    integral.  (sympy's blackbox antiderivative of the direct
    #    integral carries a removable NaN at kappa = 0.)
    Phi_k = sp.Rational(1, 2) * (1 + sp.erf(kap / sp.sqrt(2)))
    phi_k = sp.exp(-kap ** 2 / 2) / sp.sqrt(2 * sp.pi)
    F = (1 + kap ** 2) * Phi_k + kap * phi_k
    G = kap * Phi_k + phi_k          # E[(kappa - Z)_+]
    ok &= check('nakajima_Fprime_is_2G', sp.diff(F, kap) - 2 * G)
    ok &= check('nakajima_Gprime_is_Phi', sp.diff(G, kap) - Phi_k)
    ok &= check('nakajima_F_vanishes',
                sp.limit(F, kap, -sp.oo))
    ok &= check('nakajima_G_vanishes',
                sp.limit(G, kap, -sp.oo))

    # 7. The general-kappa constant in dV/da2: with D = sqrt(1 - a2^2/q),
    #    d(kappa/D)/da2 = kappa a2 / (q D^3).
    Dexpr = sp.sqrt(1 - a2 ** 2 / q)
    ok &= check('dV_da2_constant',
                sp.diff(kap / Dexpr, a2) - kap * a2 / (q * Dexpr ** 3))

    # 8. ... and its a2-derivative, the second-derivative constant:
    #    kappa (1/(q D^3) + 3 a2^2/(q^2 D^5)).
    ok &= check('d2V_da2_constant',
                sp.diff(kap / Dexpr, a2, 2)
                - kap * (1 / (q * Dexpr ** 3)
                         + 3 * a2 ** 2 / (q ** 2 * Dexpr ** 5)))

    # 9. The anchored-Bregman collapse: with theta = X + Delta and
    #    Mx = tanh X,
    #    log2cosh(theta) - log2cosh(X) - Mx Delta
    #      = log(cosh Delta + Mx sinh Delta) - Mx Delta.
    X, Dl = sp.symbols('X Delta', real=True)
    Mx = sp.tanh(X)
    lhs9 = (sp.log(2 * sp.cosh(X + Dl)) - sp.log(2 * sp.cosh(X)))
    rhs9 = sp.log(sp.cosh(Dl) + Mx * sp.sinh(Dl))
    ok &= check('bregman_collapse',
                sp.simplify(sp.expand_trig(sp.exp(lhs9) - sp.exp(rhs9))))

    # 10. The exponential split of the same factor:
    #     cosh D + Mx sinh D = e^D (1+Mx)/2 + e^{-D} (1-Mx)/2.
    ok &= check('exponential_split',
                sp.expand((sp.cosh(Dl) + Mx * sp.sinh(Dl)
                           - (sp.exp(Dl) * (1 + Mx) / 2
                              + sp.exp(-Dl) * (1 - Mx) / 2)
                           ).rewrite(sp.exp)))

    # 11. The gradient form: tanh(X + D) - tanh X
    #     = (1 - Mx^2) tanh D / (1 + Mx tanh D).
    ok &= check('grad_diff_identity',
                sp.simplify(sp.tanh(X + Dl) - Mx
                            - (1 - Mx ** 2) * sp.tanh(Dl)
                            / (1 + Mx * sp.tanh(Dl))))

    # 12. The Epp regrouping: (2E - V)(E - V) - 1 = (E + t)t - 1 at
    #     t = E - V.
    E, V = sp.symbols('E V', real=True)
    ok &= check('epp_regrouping',
                (2 * E - V) * (E - V) - 1
                - ((E + (E - V)) * (E - V) - 1))

    # 13. E'' < E - V rearrangement: E'' = E'(2E - V) - E and E' < 1
    #     give E'' - (E - V) = (E' - 1)(2E - V) + (E - V) - E + ...;
    #     verify the exact identity E'' - t = (E' - 1)(E + t) with
    #     E' = E t, t = E - V (so E' < 1 makes the bound strict when
    #     E + t > 0).
    tt = E - V
    Ep_ = E * tt
    Epp_ = E * ((E + tt) * tt - 1)
    ok &= check('hazard_convexity_bound',
                Epp_ - tt - (Ep_ - 1) * (E + tt))

    # 14. The E'' formula itself: differentiating E' = E^2 - V E and
    #     substituting E' back gives E'' = E (t(E+t) - 1), the form the
    #     code evaluates and the convexity lemma signs.
    ok &= check('epp_formula',
                2 * E * Ep_ - E - V * Ep_ - Epp_)

    # 15. The convexity lemma's moment algebra: with E[W] = t and
    #     E[W^2] = 1 - V E + V^2 for the truncated residual W,
    #     2 (E[W])^2 - E[W^2] = t(E+t) - 1 exactly, so E'' > 0 is
    #     equivalent to E[W^2] < 2 (E[W])^2.
    ok &= check('epp_positivity_equivalence',
                2 * tt ** 2 - (1 - V * E + V ** 2)
                - (tt * (E + tt) - 1))

    # 16. Truncated first and second moments as total derivatives:
    #     z phi = -(phi)' and z^2 phi = phi - (z phi)', which integrate
    #     over (V, inf) to E[Z 1_{Z>V}] = phi(V) and
    #     E[Z^2 1_{Z>V}] = Psi(V) + V phi(V).
    phiz = sp.exp(-z ** 2 / 2) / sp.sqrt(2 * sp.pi)
    ok &= check('trunc_first_moment', z * phiz + sp.diff(phiz, z))
    ok &= check('trunc_second_moment',
                z ** 2 * phiz - phiz + sp.diff(z * phiz, z))

    # 17. Lemma 1 (stationarity), psi part, as integrand identities:
    #     d/dpsi log 2cosh(sqrt(psi) z) = z tanh(sqrt(psi) z)/(2 sqrt(psi)),
    #     and the Gaussian IBP z tanh(c z) phi = c sech^2(c z) phi
    #     - (tanh(c z) phi)'.
    c = sp.symbols('c', positive=True)
    ok &= check('stationarity_psi_derivative',
                sp.diff(sp.log(2 * sp.cosh(sp.sqrt(psi) * z)), psi)
                - z * sp.tanh(sp.sqrt(psi) * z) / (2 * sp.sqrt(psi)))
    ok &= check('stationarity_psi_ibp',
                z * sp.tanh(c * z) * phiz
                - c * sp.sech(c * z) ** 2 * phiz
                + sp.diff(sp.tanh(c * z) * phiz, z))

    # 18. Lemma 1, q part, assembled: with the calculus inputs
    #     u_z = -sqrt(q/(1-q)), u_q = (:-z/sqrt(q) + u/sqrt(1-q):)/(2 sqrt(1-q)),
    #     phi'/phi = -z, and the Mills rule M' = M(M-u), the Stein
    #     rearrangement of the proof of Lemma 1 is the pointwise identity
    #     M u_q - M^2/(2(1-q))
    #       = (1/(2 sqrt(q(1-q)))) (M' u_z - z M),
    #     whose right side is (d/dz)[M(u(z)) phi(z)]/phi times the constant.
    #     (Boundary terms vanish: |M(u)| <= |u| + M(0) and phi kills a
    #     linear factor.)
    #     Radicals are cleared by hand (sympy will not identify
    #     sqrt(q(1-q)) with sqrt(q) sqrt(1-q) without sign knowledge):
    #     a = sqrt(q), b = sqrt(1-q) as independent positive symbols.
    Msym = sp.symbols('Mv', positive=True)
    a_, b_ = sp.symbols('a b', positive=True)
    u_z = -a_ / b_
    u_q = (-z / a_ + u / b_) / (2 * b_)
    Mp_rule = Msym * (Msym - u)
    ok &= check('stationarity_q_assembly',
                Msym * u_q - Msym ** 2 / (2 * b_ ** 2)
                - (Mp_rule * u_z - z * Msym) / (2 * a_ * b_))
    #     ... and the u_q formula itself for u = (kappa - sqrt(q) z)/sqrt(1-q):
    u_expr = (kap - sp.sqrt(q) * z) / sp.sqrt(1 - q)
    ok &= check('u_q_formula',
                sp.diff(u_expr, q)
                - (-z / sp.sqrt(q) + u_expr / sp.sqrt(1 - q))
                / (2 * sp.sqrt(1 - q)))

    # 19. Lemma 2 (dR/dq), assembled the same way: with
    #     (M M')' = M'^2 + M M'' and M'' = M'(M-u) + M(M'-1), the proof's
    #     Stein rearrangement is the pointwise identity
    #     2 M M' u_q - (M'^2 + 2 M^2 M' - M^2)/(1-q)
    #       = (1/sqrt(q(1-q))) ((M M')' u_z - z M M').
    Mpp_rule = Mp_rule * (Msym - u) + Msym * (Mp_rule - 1)
    MMp_prime = Mp_rule ** 2 + Msym * Mpp_rule
    ok &= check('dRdq_assembly',
                2 * Msym * Mp_rule * u_q
                - (Mp_rule ** 2 + 2 * Msym ** 2 * Mp_rule - Msym ** 2)
                / b_ ** 2
                - (MMp_prime * u_z - z * Msym * Mp_rule)
                / (a_ * b_))

    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
