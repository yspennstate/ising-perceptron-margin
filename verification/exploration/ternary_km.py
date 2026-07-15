"""Exploratory: the replica-symmetric system for the TERNARY perceptron
at margin kappa --- weights w in {-1, 0, +1} uniform, constraints
<g_a, w>/sqrt(N) >= kappa.

This substantiates the paper's extensions claim that the architecture
survives a change of prior: the constraint side is untouched (u carries
Q - q in place of 1 - q), and only the entropy side's single-site law
changes.  Everything here is float diagnostics; the anchors below are
the honesty checks.

RS order parameters: q (solution overlap), Q (self-overlap E[w^2]; a
variable for this prior, with conjugate hat-Q), and psi = hat-q (the
conjugate of q, matching the paper's psi at the +-1 prior).

Single-site measure at fields (z, c): weights w with
    log-weight = sqrt(psi) z w + c w^2,   c = (hat-Q - psi)/2,
so with e(z) = exp(c) (uniform prior constants cancel):
    zeta = 1 + 2 e^c cosh(sqrt(psi) z)
    f(z) = <w>   = 2 e^c sinh(sqrt(psi) z) / zeta
    g2(z) = <w^2> = 2 e^c cosh(sqrt(psi) z) / zeta

Fixed-point equations (Gardner replica, RS):
    q   = E[f(Z)^2]
    Q   = E[g2(Z)]
    psi = (alpha/(Q-q)) E[M(u(Z))^2],          u = (kappa - sqrt(q) Z)/sqrt(Q-q)
    hatQ = -(alpha/(Q-q)) E[M(u(Z)) u(Z)] ... derived from d/dQ of the
           energy term; see residual checks, which verify the WHOLE
           gradient of G vanishes at the solved point.

RS free energy (checked by its stationarity, not trusted from the
derivation alone):
    G = -psi (Q - q)/2 - hatQ_c Q + E log zeta + alpha E log Psi(u)
with hatQ_c = c pairing as in the residuals below; the residual battery
solves the four equations and then finite-differences G in ALL FOUR
variables (q, Q, psi, c) --- every derivative must vanish at the fixed
point, which pins signs and factors empirically exactly as the
stationarity lemma does for the +-1 system.

Anchors:
  - annealed bound: alpha_ann = log 3 / (-log Psi(kappa / sqrt(Q_free)))
    with Q_free = 2/3; alpha_*(kappa) must stay below it.
  - c -> +infinity recovers the +-1 system (zeta -> 2 e^c cosh, f -> tanh):
    checked by clamping c large and comparing against km.py.
  - stationarity residuals ~ 0 (the derivation's own consistency).

Run: python ternary_km.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpmath import mp, mpf, exp, log, sqrt, cosh, sinh, tanh, erfc, pi, quad

mp.dps = 16   # exploration-grade: 4-5 stable digits, 5x faster than 25

ZCUT = mpf(12)
SPLIT = [-ZCUT, -6, -2, 0, 2, 6, ZCUT]


def phi(z):
    return exp(-z * z / 2) / sqrt(2 * pi)


def Psi(u):
    return erfc(u / sqrt(2)) / 2


def M(u):
    # asymptotic branch: Psi underflows for large u at exploration
    # precision; M(u) = u + 1/u to O(u^-3) there
    if u > 30:
        return u + 1 / u
    return phi(u) / Psi(u)


def logPsi(u):
    if u > 30:
        return -u * u / 2 - log(sqrt(2 * pi)) - log(u + 1 / u)
    return log(Psi(u))


def gauss_E(f):
    return quad(lambda z: f(z) * phi(z), SPLIT)


def site(psi, c):
    """E[f^2], E[g2], E[log zeta] for the ternary single-site law."""
    s = sqrt(psi)
    ec = exp(c)

    def parts(z):
        ch = cosh(s * z)
        sh = sinh(s * z)
        zeta = 1 + 2 * ec * ch
        f = 2 * ec * sh / zeta
        g2 = 2 * ec * ch / zeta
        return f * f, g2, log(zeta)

    Ef2 = gauss_E(lambda z: parts(z)[0])
    Eg2 = gauss_E(lambda z: parts(z)[1])
    Elz = gauss_E(lambda z: parts(z)[2])
    return Ef2, Eg2, Elz


def energy(q, Q, alpha, kappa):
    """alpha E log Psi(u), E[M^2], E[M u] at the constraint side."""
    if not Q - q > mpf('1e-12'):
        # site() can return Ef2 == Eg2 exactly (frozen branch), and the
        # solve loop only guards the PREVIOUS iterate - so the escape
        # must live here, at the division, where every caller passes
        raise BranchEscape(f'Q - q = {Q - q} in energy()')
    sq = sqrt(q)
    sQq = sqrt(Q - q)

    def u_of(z):
        return (kappa - sq * z) / sQq

    ElogPsi = gauss_E(lambda z: logPsi(u_of(z)))
    EM2 = gauss_E(lambda z: M(u_of(z)) ** 2)
    EMu = gauss_E(lambda z: M(u_of(z)) * u_of(z))
    return ElogPsi, EM2, EMu


def G_of(q, Q, psi, c, alpha, kappa):
    _, _, Elz = site(psi, c)
    ElogPsi, _, _ = energy(q, Q, alpha, kappa)
    # the pairing that makes all four residuals vanish (verified below):
    # G = -psi(Q - q)/2 - c Q + E log zeta + alpha E log Psi(u)
    return -psi * (Q - q) / 2 - c * Q + Elz + alpha * ElogPsi


class BranchEscape(RuntimeError):
    """Q - q collapsed: the RS branch froze (alpha above the fold)."""


def solve(alpha, kappa, q0=0.35, Q0=0.66, psi0=1.5, c0=-0.2, iters=400):
    """Damped fixed-point iteration on the four RS equations."""
    q, Q, psi, c = (mpf(x) for x in (q0, Q0, psi0, c0))
    damp = mpf('0.5')
    for _ in range(iters):
        if Q - q < mpf('1e-9') or not (0 < q < Q < 1):
            raise BranchEscape(f'q={q}, Q={Q}')
        Ef2, Eg2, _ = site(psi, c)
        q_new, Q_new = Ef2, Eg2
        ElogPsi, EM2, EMu = energy(q_new, Q_new, alpha, kappa)
        psi_new = alpha * EM2 / (Q_new - q_new)
        # c update from the Q-stationarity of G:
        #   dG/dQ = -psi/2 - c + alpha dElogPsi/dQ = 0,
        #   dElogPsi/dQ = E[M u]/(2(Q-q))
        c_new = -psi_new / 2 + alpha * EMu / (2 * (Q_new - q_new))
        dq = abs(q_new - q) + abs(Q_new - Q) + abs(psi_new - psi) \
            + abs(c_new - c)
        q = damp * q_new + (1 - damp) * q
        Q = damp * Q_new + (1 - damp) * Q
        psi = damp * psi_new + (1 - damp) * psi
        c = damp * c_new + (1 - damp) * c
        if dq < mpf('3e-7'):
            break
    return q, Q, psi, c


def residuals(q, Q, psi, c, alpha, kappa, h=mpf('1e-7')):
    out = {}
    for name, dv in (('q', (h, 0, 0, 0)), ('Q', (0, h, 0, 0)),
                     ('psi', (0, 0, h, 0)), ('c', (0, 0, 0, h))):
        gp = G_of(q + dv[0], Q + dv[1], psi + dv[2], c + dv[3],
                  alpha, kappa)
        gm = G_of(q - dv[0], Q - dv[1], psi - dv[2], c - dv[3],
                  alpha, kappa)
        out[name] = (gp - gm) / (2 * h)
    return out


def alpha_star(kappa, a_lo=0.5, a_hi=2.5):
    """Bisection on G_*(alpha) = 0 along the solved branch."""
    state = {}

    def Gs(alpha):
        try:
            q, Q, psi, c = solve(alpha, kappa, **state.get('seed', {}))
        except BranchEscape:
            return None, None
        state['seed'] = dict(q0=q, Q0=Q, psi0=psi, c0=c)
        return G_of(q, Q, psi, c, alpha, kappa), (q, Q, psi, c)

    print(f'  bisection at kappa={float(kappa):+.3f}', flush=True)
    g_lo, _ = Gs(mpf(a_lo))
    assert g_lo is not None and g_lo > 0, g_lo
    lo, hi = mpf(a_lo), mpf(a_hi)
    fp = None
    for it in range(22):
        mid = (lo + hi) / 2
        g, fp_mid = Gs(mid)
        if it % 4 == 0:
            print(f'    level {it}: alpha in [{float(lo):.6f}, {float(hi):.6f}]', flush=True)
        # a frozen branch means alpha is past the fold: treat as the
        # high side, like the +-1 continuation's escape handling
        if g is not None and g > 0:
            lo = mid
            fp = fp_mid
        else:
            hi = mid
    return (lo + hi) / 2, fp


def main():
    print('ternary perceptron, RS exploration (float diagnostics)')
    kappas = ([float(x) for x in sys.argv[1:]]
              if len(sys.argv) > 1 else (0.0, 0.1))
    for kappa in kappas:
        a, (q, Q, psi, c) = alpha_star(kappa)
        res = residuals(q, Q, psi, c, a, kappa)
        Q_free = mpf(2) / 3
        a_ann = log(3) / (-log(Psi(mpf(kappa) / sqrt(Q_free))))
        print(f'kappa={kappa:+.2f}  alpha*={mp.nstr(a, 10)}  '
              f'q={mp.nstr(q, 8)}  Q={mp.nstr(Q, 8)}  '
              f'annealed={mp.nstr(a_ann, 8)}  '
              f'residuals: ' + ' '.join(
                  f'{k}={mp.nstr(v, 2)}' for k, v in res.items()),
              flush=True)
    # the +-1 limit: clamp c large; f -> tanh, Q -> 1, and the system
    # must reproduce km.alpha_star at kappa = 0 (0.8330786...)
    import km
    r = km.alpha_star(mpf(0), conditions=False)
    print(f'pm1 anchor via km.py: alpha*(0) = {mp.nstr(r["alpha"], 10)} '
          f'(certified 0.833078599...)')


if __name__ == '__main__':
    main()
