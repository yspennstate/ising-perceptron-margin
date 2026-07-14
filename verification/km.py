"""Replica-symmetric Krauth-Mezard system for the Ising perceptron at
general margin kappa.

Conventions follow Ding-Sun sections 7-9 and Huang section 3 (his
Theorem "Main result, general kappa" is conditional on four numerical
conditions; this module computes their content along the kappa curve):

  P(psi)         = E tanh^2(sqrt(psi) Z)
  u(z; q, kappa) = (kappa - sqrt(q) z) / sqrt(1-q)
  R(q, alpha)    = (alpha/(1-q)) E[ M(u(Z))^2 ],  M = phi/Psi (inverse Mills)
  G(alpha,q,psi) = -psi(1-q)/2 + E log(2 cosh(sqrt(psi) Z))
                   + alpha E log Psi(u(Z))

At kappa = 0, u(z) = -gamma(q) z with gamma = sqrt(q/(1-q)) and the system
is exactly the one whose fixed-point rectangle and zero crossing are
certified in ising-perceptron-capacity (alpha_*(0) = 0.833078599...).
(Huang's displayed RS free energy carries a sign typo on the alpha term;
the fixed point requires +alpha E log Psi, which is what Ding-Sun and the
certified verification use, and what reproduces the rectangle.)

This module is the nonrigorous continuation in kappa.  Cold solves far
from a seed are unreliable by construction: for alpha above the fold the
inner fixed point escapes to the frozen branch q -> 1.  The iteration
therefore guards the branch (BranchEscape) and alpha_star damps its
Newton steps; walk the curve by continuation from kappa = 0 outward.

Conditions computed per point (Huang's numbering):
  km-well-defd : contraction (P o R_alpha)'(q_*) < 1, G_* crossing
  amp-works    : alpha E[th'(sqrt(psi0)Z)^2] E[F'_{1-q0}(sqrt(q0)Z)^2] < 1
  local-concavity : lambda_0 = inf_{z > -1} lambda(z) < 0
The global two-variable condition (his Condition on S_*) is separate
moment-body machinery, handled elsewhere.
"""

from mpmath import mp, mpf, exp, log, sqrt, tanh, cosh, sech, erfc, pi, quad

mp.dps = 30

ZCUT = mpf(14)
SPLIT = [-ZCUT, -6, -2, 0, 2, 6, ZCUT]
Q_CEIL = mpf('0.995')


class BranchEscape(RuntimeError):
    """The inner iteration left the physical q < 1 branch (alpha is above
    the fold, or the RS fixed point has disappeared at this kappa)."""


def phi(z):
    return exp(-z * z / 2) / sqrt(2 * pi)


def Psi(u):
    """Gaussian upper tail P(Z >= u)."""
    return erfc(u / sqrt(2)) / 2


def logPsi(u):
    return log(Psi(u))


def M(u):
    """Inverse Mills ratio phi(u)/Psi(u)."""
    return phi(u) / Psi(u)


def Mprime(u):
    """M'(u) = M(u)(M(u) - u), in (0,1)."""
    m = M(u)
    return m * (m - u)


def gauss_E(f):
    """E[f(Z)] over the standard Gaussian, |z| <= ZCUT (tail < 1e-43)."""
    return quad(lambda z: f(z) * phi(z), SPLIT)


def P_of(psi):
    s = sqrt(psi)
    return gauss_E(lambda z: tanh(s * z) ** 2)


def u_of(z, q, kappa):
    return (kappa - sqrt(q) * z) / sqrt(1 - q)


def R_of(q, alpha, kappa):
    e2 = gauss_E(lambda z: M(u_of(z, q, kappa)) ** 2)
    return alpha * e2 / (1 - q)


def Elog2cosh(psi):
    s = sqrt(psi)
    return gauss_E(lambda z: log(2 * cosh(s * z)))


def ElogPsi(q, kappa):
    return gauss_E(lambda z: logPsi(u_of(z, q, kappa)))


def G_of(alpha, q, psi, kappa):
    if not (0 < q < 1) or psi <= 0:
        raise BranchEscape(f"G_of outside domain: q={q}, psi={psi}")
    return (-psi * (1 - q) / 2 + Elog2cosh(psi)
            + alpha * ElogPsi(q, kappa))


def _PR(q, alpha, kappa):
    if not (0 < q < Q_CEIL):
        raise BranchEscape(f"q={q} outside (0, {Q_CEIL})")
    return P_of(R_of(q, alpha, kappa))


def fixed_point(alpha, kappa, q0=None, tol=None, maxit=200):
    """Solve q = P(R_alpha(q)) on the physical branch.

    Plain iteration for a few steps (gives the empirical contraction),
    then secant on g(q) = P(R(q)) - q.  Returns (q, psi, iters, PRprime)
    where PRprime is the central-difference (P o R_alpha)'(q_*) -- the
    quantity Huang's Condition km-well-defd bounds below 1.
    """
    if tol is None:
        tol = mpf(10) ** (-(mp.dps - 6))
    q = mpf('0.5639') if q0 is None else mpf(q0)
    it = 0
    for _ in range(6):
        it += 1
        qn = _PR(q, alpha, kappa)
        if abs(qn - q) < tol:
            q = qn
            break
        q_prev, q = q, qn
    else:
        # secant on g(q) = PR(q) - q, seeded by the last two iterates
        g_prev = _PR(q_prev, alpha, kappa) - q_prev
        g_cur = _PR(q, alpha, kappa) - q
        it += 2
        while abs(g_cur) > tol and it < maxit:
            denom = g_cur - g_prev
            if denom == 0:
                break
            q_new = q - g_cur * (q - q_prev) / denom
            if not (0 < q_new < Q_CEIL):
                raise BranchEscape(f"secant step to q={q_new}")
            q_prev, g_prev = q, g_cur
            q = q_new
            g_cur = _PR(q, alpha, kappa) - q
            it += 1
        if abs(g_cur) > tol:
            raise RuntimeError(f"fixed point not converged: alpha={alpha}, "
                               f"kappa={kappa}, residual {g_cur}")
    h = mpf(10) ** (-(mp.dps // 3))
    prp = (_PR(q + h, alpha, kappa) - _PR(q - h, alpha, kappa)) / (2 * h)
    return q, R_of(q, alpha, kappa), it, prp


def stationarity_residuals(alpha, q, psi, kappa, h=None):
    """Central-difference dG/dq and dG/dpsi at the point; both must be ~0
    at a genuine saddle.  Nonzero dG/dq means the kappa-generalized R is
    NOT the stationarity condition and the continuation would be wrong."""
    if h is None:
        h = mpf(10) ** (-(mp.dps // 3))
    dGdq = (G_of(alpha, q + h, psi, kappa)
            - G_of(alpha, q - h, psi, kappa)) / (2 * h)
    dGdpsi = (G_of(alpha, q, psi + h, kappa)
              - G_of(alpha, q, psi - h, kappa)) / (2 * h)
    return dGdq, dGdpsi


def cond_amp_works(alpha, q, psi, kappa):
    """Huang Condition amp-works, LHS (must be < 1):
    alpha E[th'(sqrt(psi)Z)^2] E[F'_{1-q}(sqrt(q)Z)^2],
    with th' = sech^2 and F'_{1-q}(x) = -M'((kappa-x)/sqrt(1-q))/(1-q)."""
    s = sqrt(psi)
    e_th = gauss_E(lambda z: sech(s * z) ** 4)
    e_f = gauss_E(lambda z: Mprime(u_of(z, q, kappa)) ** 2) / (1 - q) ** 2
    return alpha * e_th * e_f


def cond_local_concavity(alpha, q, psi, kappa):
    """Huang Condition local-concavity: lambda_0 = inf_{z>-1} lambda(z),
    required < 0.  lambda(z) = z - alpha E[hf/(1+m(z) hf)] - d0 with
      hf(x)  = M'(u)/((1-q)(1 - M'(u))),   u = (kappa - x)/sqrt(1-q)
      m(z)   = E[(z + cosh^2(sqrt(psi) Z))^{-1}]
      d0     = alpha E[F'_{1-q}(sqrt(q) Z)] = -alpha E[M'(u)]/(1-q)
    lambda'(z) = 1 - alpha theta(z), theta strictly decreasing, so the
    minimizer z0 solves theta(z0) = 1/alpha; solved here by bisection.
    Returns (lambda_0, z0)."""
    s = sqrt(psi)
    sq = sqrt(q)

    def hf(z):
        mp_u = Mprime(u_of(z, q, kappa))
        return mp_u / ((1 - q) * (1 - mp_u))

    def m_of(zz):
        return gauss_E(lambda t: 1 / (zz + cosh(s * t) ** 2))

    def theta(zz):
        e1 = gauss_E(lambda t: 1 / (zz + cosh(s * t) ** 2) ** 2)
        mz = m_of(zz)
        e2 = gauss_E(lambda t: (hf(t) / (1 + mz * hf(t))) ** 2)
        return e1 * e2

    target = 1 / alpha
    lo, hi = mpf('-0.999'), mpf(1)
    while theta(hi) > target:
        hi *= 2
        if hi > 1e6:
            raise RuntimeError("theta minimizer out of range")
    # lambda is flat at its minimum (lambda'(z0) = 0), so z0 to ~1e-14
    # already gives lambda0 to ~1e-28; no need to bisect to full precision.
    for _ in range(48):
        mid = (lo + hi) / 2
        if theta(mid) > target:
            lo = mid
        else:
            hi = mid
    z0 = (lo + hi) / 2

    d0 = -alpha * gauss_E(lambda t: Mprime(u_of(t, q, kappa))) / (1 - q)
    mz0 = m_of(z0)
    lam0 = z0 - alpha * gauss_E(
        lambda t: hf(t) / (1 + mz0 * hf(t))) - d0
    return lam0, z0


def alpha_star(kappa, alpha0=None, tol=None, maxit=80, q0=None,
               conditions=True):
    """Zero of alpha -> G_*(alpha) at fixed kappa, by damped Newton with
    the envelope derivative dG_*/dalpha = E log Psi(u) < 0.  On
    BranchEscape the step is halved toward the last good alpha.  Walk the
    kappa curve by continuation (seed alpha0, q0 from the neighbor)."""
    if tol is None:
        tol = mpf(10) ** (-(mp.dps - 8))
    alpha = mpf('0.8330786') if alpha0 is None else mpf(alpha0)
    q_seed = q0
    prev_alpha = None
    for it in range(1, maxit + 1):
        try:
            q, psi, fp_iters, prp = fixed_point(alpha, kappa, q0=q_seed)
        except BranchEscape:
            if prev_alpha is None:
                raise
            alpha = (alpha + prev_alpha) / 2
            continue
        q_seed = q
        elp = ElogPsi(q, kappa)
        G = -psi * (1 - q) / 2 + Elog2cosh(psi) + alpha * elp
        step = -G / elp
        prev_alpha = alpha
        alpha = alpha + step
        if abs(step) < tol * max(1, abs(alpha)):
            break
    else:
        raise RuntimeError(f"alpha_star not converged at kappa={kappa}")
    dGdq, dGdpsi = stationarity_residuals(alpha, q, psi, kappa)
    out = {
        'kappa': kappa,
        'alpha': alpha,
        'q': q,
        'psi': psi,
        'gamma': sqrt(q / (1 - q)),
        'kb': kappa / sqrt(1 - q),
        'ElogPsi': elp,
        'PRprime': prp,
        'fp_iters': fp_iters,
        'newton_iters': it,
        'dG_dq': dGdq,
        'dG_dpsi': dGdpsi,
        'alpha_annealed': log(2) / (-logPsi(mpf(kappa))),
    }
    if conditions:
        out['amp_works'] = cond_amp_works(alpha, q, psi, kappa)
        lam0, z0 = cond_local_concavity(alpha, q, psi, kappa)
        out['lambda0'] = lam0
        out['z0'] = z0
    return out


if __name__ == '__main__':
    r = alpha_star(mpf(0))
    for k in ('alpha', 'q', 'psi', 'gamma', 'PRprime', 'amp_works',
              'lambda0', 'z0', 'dG_dq', 'dG_dpsi', 'alpha_annealed'):
        print(f"{k:>14} = {mp.nstr(r[k], 20)}")
    print("certified     alpha in [0.833078599, 0.833078600]")
    print("certified     q     in [0.56394907949, 0.56394908030]")
    print("certified     psi   in [2.5763513100, 2.5763513224]")
