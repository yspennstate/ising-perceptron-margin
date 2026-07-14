"""Fast fixed-grid certified quadrature for Huang's first-moment functional,
in moment coordinates (a1, a2).

Background. Huang's Condition 1.3 asks that
    S_*(l1, l2) = inf_{s>=0}[ s^2 psi/2 + ent(l) + alpha T(a1(l), a2(l), s) ]
be <= 0 for all (l1, l2), where the profile Lam = tanh(l1 X + l2 M),
M = tanh(X), X ~ N(0, psi), and a1 = E[X Lam], a2 = E[M Lam]. Because Lam is
the max-entropy profile for its own moments, ent(l) = H(a1, a2) with
    H(a1,a2) = max{ E ent2((1+Lam)/2) : E[X Lam]=a1, E[M Lam]=a2, |Lam|<=1 }.
So sup_l S_*(l) = sup_{(a1,a2)} [ H(a1,a2) + G(a1,a2) ], G = inf_s[...],
and the sup runs over a BOUNDED rectangle: |a1| <= sqrt(psi), |a2| < sqrt(q).

Duality gives, for every dual point (b1, b2),
    H(a1, a2) <= Phi(b1, b2) - b1 a1 - b2 a2,   Phi(b) = E log 2cosh(b1 X + b2 M).
A finite dictionary of dual points thus yields a piecewise-linear certified
upper bound on H, evaluated with no per-cell integration. The only per-cell
integral is the constraint term
    T(a1, a2, s) = E_z log Psi(V),   z ~ N(0,1),
    V = ( -(a2/q) sqrt(q) z - (a1/psi) N ) / D + s N,
    D = sqrt(1 - a2^2/q),  N = E(-gamma z)/sqrt(1-q),  gamma = sqrt(q/(1-q)),
which is a one-dimensional Gaussian integral evaluated on a fixed grid whose
z-dependent data (N, N', E(-gamma z)) is precomputed once.

All quantities are Arb balls enclosing the truth; (alpha, q, psi) are the
Ding-Sun rectangle (block 1). The grid rule is first order (mean value):
    int_cell g phi  in  g(mid) m0 + g'(cell) m1p - g'(cell) m1m,
    m0 = Psi(lo) - Psi(hi),
    m1p = int_{mid}^{hi} (z-mid) phi >= 0,  m1m = int_{lo}^{mid} (mid-z) phi >= 0,
rigorous because g(z) = g(mid) + g'(xi_z)(z - mid) with g'(xi_z) in g'(cell),
and the positive/negative parts of (z - mid) are handled separately (the
remainder does NOT factor through the signed first moment m1p - m1m).
"""

from flint import arb, acb, ctx
from core import (set_prec, dec, phi, Psi, mills,
                  z1_tail, z2_tail, gauss_tail_mass)

# Module-level parameter balls, set by configure().  This is the general-
# kappa port of the capacity verification's huanggrid.py: kappa enters T
# as the additive constant kappa/D inside Psi's argument, inside the cavity
# denoiser N(z) = E((kappa - sqrt(q) z)/sqrt(1-q))/sqrt(1-q), and in the
# a2-derivative of T through d(kappa/D)/da2 = kappa a2/(q D^3).  The
# entropy/dual side (Phi, the moment body K) is kappa-free in form.
KAPPA = None
ALPHA = None
Q = None
PSI = None
SQ_PSI = None
SQ_Q = None
S1Q = None
GAMMA = None
LOG2 = arb(2).log()


def configure(kappa, alpha, q, psi):
    """Set the parameter balls (arb) for a margin kappa and its certified
    (alpha, q0, psi0) enclosures; clears the precomputed grids."""
    global KAPPA, ALPHA, Q, PSI, SQ_PSI, SQ_Q, S1Q, GAMMA, _GRID_CACHE, _HFAN
    KAPPA = arb(kappa)
    ALPHA = arb(alpha)
    Q = arb(q)
    PSI = arb(psi)
    SQ_PSI = PSI.sqrt()
    SQ_Q = Q.sqrt()
    S1Q = (1 - Q).sqrt()
    GAMMA = (Q / (1 - Q)).sqrt()
    _GRID_CACHE = {}
    _HFAN = None


# ---------------------------------------------------------------------------
# Special functions on real balls.
# ---------------------------------------------------------------------------

def E_mills(x):
    """Inverse Mills ratio E(x) = phi(x)/Psi(x) as an arb ball.

    E is strictly increasing, so for a wide ball we enclose via its (thin)
    endpoints; evaluating phi/Psi directly on a wide ball would let Psi touch
    0 and blow up. For narrow balls the direct form is tighter."""
    from core import endpoints
    lo, hi = endpoints(x)
    if hi - lo < 1:
        v = phi(x) / Psi(x)
        if v.is_finite():
            return v
    return (phi(lo) / Psi(lo)).union(phi(hi) / Psi(hi))


def _mills_gap(E, x):
    """The Mills gap t = E(x) - x as a STABLE ball.

    Computed by raw subtraction, t inherits both operands' widths; on wide
    far-tail cells with x >> 0 the two O(|x|) balls cancel and every kernel
    built on t explodes (measured: quadT hi spikes to 2e3 on wedge-edge
    leaves at kappa = 0.05, killing sound Region-I cells whose true value
    is -6.6).  Two classical Mills-ratio facts refine it: t > 0 for every
    real x, and t < 1/x for x > 0.  Intersecting the computed ball with
    them is always sound and caps the width at 1/x_lo where it matters."""
    from core import endpoints
    t = E - x
    tlo, thi = endpoints(t)
    xlo, _ = endpoints(x)
    changed = False
    if not (tlo > 0):
        tlo = arb(0)
        changed = True
    if xlo > 0:
        cap = 1 / xlo
        if not (thi < cap):
            thi = arb(cap)
            changed = True
    return tlo.union(thi) if changed else t


def _clamp01(v):
    """Intersect a ball with [0, 1]; sound for E'(x) = E(E - x), which is
    the Gaussian hazard derivative: E' = 1 - Var(Z | Z > x) in (0, 1)."""
    from core import endpoints
    lo, hi = endpoints(v)
    changed = False
    if not (lo > 0):
        lo = arb(0)
        changed = True
    if not (hi < 1):
        hi = arb(1)
        changed = True
    return lo.union(hi) if changed else v


def _clamp_Epp(v, t):
    """Intersect a ball with [0, t_hi]; sound for E''(x): the Gaussian
    hazard is convex (E'' > 0), and E'' = E'(E + t) - E < t from E' < 1."""
    from core import endpoints
    lo, hi = endpoints(v)
    _, thi = endpoints(t)
    changed = False
    if not (lo > 0):
        lo = arb(0)
        changed = True
    if not (hi < thi):
        hi = thi
        changed = True
    return lo.union(hi) if changed else v


def Ep_mills(E, x):
    """E'(x) = E(x)(E(x) - x), given E = E(x); stable-gap form clamped to
    the proven (0, 1) range."""
    return _clamp01(E * _mills_gap(E, x))


def logPsi(x):
    """log Psi(x), robust for wide balls. logPsi is decreasing, so enclose a
    wide ball via its (thin) endpoints; the direct Psi(x).log() lets the ball
    of Psi values cross 0 when x is wide and large."""
    from core import endpoints
    v = Psi(x)
    if v > 0:
        w = v.log()
        if w.is_finite():
            return w
    lo, hi = endpoints(x)
    return Psi(hi).log().union(Psi(lo).log())


def log2cosh(x):
    """log(2 cosh x) = |x| + log(1 + e^{-2|x|}), stable for large |x|."""
    a = abs(x)
    return a + (-2 * a).exp().log1p()


def tanh_ball(x):
    return x.tanh()


def ent2_tanh(t):
    """ent2((1+tanh t)/2) = log(2 cosh t) - t tanh t."""
    return log2cosh(t) - t * t.tanh()


# ---------------------------------------------------------------------------
# Grid construction.
# ---------------------------------------------------------------------------

class Grid:
    """A fixed partition of [-L, L] with precomputed moment weights."""

    def __init__(self, L, n):
        self.L = arb(L)
        self.n = n
        h = 2 * self.L / n
        self.lo = []
        self.hi = []
        self.mid = []
        self.cell = []
        self.m0 = []          # int_cell phi
        self.m1c = []         # int_cell (z - mid) phi   (signed; legacy)
        self.m1p = []         # int_{mid}^{hi} (z - mid) phi   >= 0
        self.m1m = []         # int_{lo}^{mid} (mid - z) phi   >= 0
        for j in range(n):
            lo = -self.L + h * j
            hi = -self.L + h * (j + 1)
            mid = (lo + hi) / 2
            self.lo.append(lo)
            self.hi.append(hi)
            self.mid.append(mid)
            self.cell.append(lo.union(hi))
            m0 = Psi(lo) - Psi(hi)
            self.m0.append(m0)
            m1p = (phi(mid) - phi(hi)) - mid * (Psi(mid) - Psi(hi))
            m1m = mid * (Psi(lo) - Psi(mid)) - (phi(lo) - phi(mid))
            self.m1p.append(m1p)
            self.m1m.append(m1m)
            self.m1c.append(m1p - m1m)

    def integrate(self, g_mid, g_cell):
        """int_{-L}^{L} g phi dz given lists g_mid[j] = g(mid_j) (thin) and
        g_cell[j] = an enclosure of g'(xi) for ALL xi in cell j.

        Mean-value rule, done correctly: on cell j,
            g(z) = g(mid) + g'(xi_z)(z - mid),   xi_z in cell j,
        so
            int_cell g phi = g(mid) m0 + int g'(xi_z)(z - mid) phi dz
                           in g(mid) m0 + [g'] m1p - [g'] m1m,
        where m1p/m1m are the positive/negative parts of int (z - mid) phi.
        (The legacy rule [g'] * (m1p - m1m) was WRONG: g'(xi_z) varies with z
        while (z - mid) changes sign, so the remainder does NOT factor through
        the signed first moment. The flaw produced systematically too-tight
        balls -- caught by an independent mpmath cross-check of grad Phi(1,0).)
        """
        total = arb(0)
        for gm, gc, m1p, m1m, m0 in zip(g_mid, g_cell, self.m1p, self.m1m,
                                        self.m0):
            total = total + gm * m0 + gc * m1p - gc * m1m
        return total


# default grids (rebuilt if precision changes)
_GRID_CACHE = {}


import os as _os
GRID_N = int(_os.environ.get('HUANG_GRID_N', '900'))


def get_zt_grid(L=9, n=None):
    if n is None:
        n = GRID_N
    key = ('zt', L, n, ctx.prec)
    if key not in _GRID_CACHE:
        g = Grid(L, n)
        _precompute_zt(g)
        _GRID_CACHE[key] = g
    return _GRID_CACHE[key]


def get_x_grid(L=9, n=None):
    if n is None:
        n = GRID_N
    key = ('x', L, n, ctx.prec)
    if key not in _GRID_CACHE:
        g = Grid(L, n)
        _precompute_x(g)
        _GRID_CACHE[key] = g
    return _GRID_CACHE[key]


def _precompute_zt(g):
    """Box-independent data for the T integral: at midpoints (thin) and over
    cells, the values N, N' and Ht, needed to form V and V'.
    General kappa:  N(z) = E(u(z))/sqrt(1-q),  u(z) = (kappa - sqrt(q) z)/
    sqrt(1-q)  (Huang's F_{1-q}(sqrt(q) z));  N'(z) = -(sqrt(q)/(1-q))
    E'(u(z)).  At kappa = 0, u = -gamma z and this reduces to the capacity
    verification's precompute verbatim."""
    g.N_mid, g.N_cell = [], []
    g.Np_mid, g.Np_cell = [], []
    g.Ht_mid, g.Ht_cell = [], []
    for mid, cell in zip(g.mid, g.cell):
        for store_val, store_p, store_ht, zz in (
                (g.N_mid, g.Np_mid, g.Ht_mid, mid),
                (g.N_cell, g.Np_cell, g.Ht_cell, cell)):
            x = (KAPPA - SQ_Q * zz) / S1Q
            E = E_mills(x)
            Ep = Ep_mills(E, x)
            store_val.append(E / S1Q)
            store_p.append(-(SQ_Q / (1 - Q)) * Ep)
            store_ht.append(SQ_Q * zz)


def _precompute_x(g):
    """Box-independent data for the Phi/profile integrals: X = sqrt(psi) z,
    M = tanh(X) and derivative dM/dz = sqrt(psi)(1-M^2)."""
    g.X_mid, g.X_cell = [], []
    g.M_mid, g.M_cell = [], []
    g.dM_mid, g.dM_cell = [], []
    for mid, cell in zip(g.mid, g.cell):
        for sx, sm, sdm, zz in ((g.X_mid, g.M_mid, g.dM_mid, mid),
                                 (g.X_cell, g.M_cell, g.dM_cell, cell)):
            X = SQ_PSI * zz
            M = X.tanh()
            sx.append(X)
            sm.append(M)
            sdm.append(SQ_PSI * (1 - M * M))


# ---------------------------------------------------------------------------
# Phi(b1, b2) = E log 2cosh(b1 X + b2 M): dual-dictionary log-partition.
# theta(z) = b1 X + b2 M; d theta/dz = b1 sqrt(psi)(1-M^2)... = b1 X'/...
#   X' = sqrt(psi), M' = sqrt(psi)(1-M^2).
# g(z) = log2cosh(theta); g'(z) = tanh(theta) * theta'(z).
# ---------------------------------------------------------------------------

def Phi_of(b1, b2, g=None):
    """E log 2cosh(b1 X + b2 M) as an arb ball, with tail bound.
    Tail |z|>=L: 0 <= log2cosh(theta) - |theta| <= log 2, and
    |theta| <= |b1| sqrt(psi)|z| + |b2|, so the discarded mass is bounded by
    E[ (|b1| sqrt(psi)|z| + |b2| + log2) 1{|z|>=L} ]."""
    if g is None:
        g = get_x_grid()
    b1 = arb(b1)
    b2 = arb(b2)
    gm, gc = [], []
    for Xm, Mm, dMm, Xc, Mc, dMc in zip(
            g.X_mid, g.M_mid, g.dM_mid, g.X_cell, g.M_cell, g.dM_cell):
        th_m = b1 * Xm + b2 * Mm
        gm.append(log2cosh(th_m))
        th_c = b1 * Xc + b2 * Mc
        dth_c = b1 * SQ_PSI + b2 * dMc
        gc.append(th_c.tanh() * dth_c)
    main = g.integrate(gm, gc)
    tail = (abs(b1) * SQ_PSI * z1_tail(g.L) + (abs(b2) + LOG2)
            * gauss_tail_mass(g.L))
    return main + tail.union(-tail)


# ---------------------------------------------------------------------------
# T(a1, a2, s) = E_z log Psi(V).
# V(z) = c0 * Ht(z) + c1 * N(z),  c0 = -(a2/q)/D,  c1 = s - (a1/psi)/D,
# with Ht = sqrt(q) z, so V is affine in (Ht, N).
#   g(z) = log Psi(V);  g'(z) = -E(V) V'(z),  V'(z) = c0 sqrt(q) + c1 N'(z).
# Note Ht(z) = sqrt(q) z so Ht'(z) = sqrt(q).
# Requires a2^2 < q on the ball (checked by caller / returns None).
# ---------------------------------------------------------------------------

def T_meanvalue(c1, c2, r1, r2, s, g=None):
    """Mean-value upper bound for T(a1,a2,s) over the box
    [c1-r1, c1+r1] x [c2-r2, c2+r2], with c1,c2,r1,r2,s thin arb balls:
        T(box) subset T(center) + dT/da1(box)*[-r1,r1] + dT/da2(box)*[-r2,r2].
    Center term is a tight (thin-a) grid integral; the derivative terms are
    grid integrals of  d/da_i log Psi(V) = -E(V) dV/da_i  enclosed over the
    box, multiplied by the small radii. Returns None if a2^2 >= q on the box.

    V = c0 Ht + c1V N,  c0 = -(a2/q)/D,  c1V = s - (a1/psi)/D,  D=sqrt(1-a2^2/q).
    dV/da1 = -(1/(psi D)) N
    dV/da2 = [ -1/(qD) - a2^2/(q^2 D^3) ] Ht + [ -a1 a2/(psi q D^3) ] N
    """
    if g is None:
        g = get_zt_grid()
    from core import endpoints
    a1box = (c1 - r1).union(c1 + r1)
    a2box = (c2 - r2).union(c2 + r2)
    _, a2hi = endpoints(a2box * a2box / Q)
    if not (a2hi < 1):
        return None
    # center value (thin) with z-mean-value grid rule
    Tc = T_of(c1, c2, s, g)
    if Tc is None:
        return None
    # derivative integrands over the a-box
    Db = (1 - a2box * a2box / Q).sqrt()
    cKb = KAPPA / Db
    c0b = -(a2box / Q) / Db
    c1b = s - (a1box / PSI) / Db
    D3 = Db * Db * Db
    dV_da1_coeffN = -(1 / (PSI * Db))
    dV_da2_coeffHt = -1 / (Q * Db) - a2box * a2box / (Q * Q * D3)
    dV_da2_coeffN = -a1box * a2box / (PSI * Q * D3)
    # general kappa: d(kappa/D)/da2 = kappa a2/(q D^3), a constant-in-z term
    dV_da2_const = KAPPA * a2box / (Q * D3)
    g1 = arb(0)
    g2 = arb(0)
    for Nc, Htc, m0 in zip(g.N_cell, g.Ht_cell, g.m0):
        V = cKb + c0b * Htc + c1b * Nc
        E = E_mills(V)
        dVda1 = dV_da1_coeffN * Nc
        dVda2 = dV_da2_const + dV_da2_coeffHt * Htc + dV_da2_coeffN * Nc
        g1 = g1 + (-E * dVda1) * m0
        g2 = g2 + (-E * dVda2) * m0
    # derivative-tail: |dT/da_i| integrand bounded; add crude tail via E(V)<=1+|V|
    # (the discarded |z|>L mass is tiny; fold a conservative constant)
    tail1 = _dT_tail(cKb, c0b, c1b, arb(0), dV_da1_coeffN, arb(0), g.L)
    tail2 = _dT_tail(cKb, c0b, c1b, dV_da2_const, dV_da2_coeffN,
                     dV_da2_coeffHt, g.L)
    g1 = g1 + tail1.union(-tail1)
    g2 = g2 + tail2.union(-tail2)
    return Tc + g1 * r1.union(-r1) + g2 * r2.union(-r2)


def _dT_tail(cK, c0, c1, cC, cN, cHt, L):
    """Crude bound on int_{|z|>L} E(V)|dV/da| phi dz,
    dV/da = cC + cN N + cHt Ht (cC is the general-kappa constant channel).
    E(V) <= 1 + |V|, |V| <= |cK| + |c0|sqrt(q)|z| + |c1|(A_N + B_N|z|),
    |N| <= A_N + B_N|z| with A_N = (1+|kappa|/S1Q)/S1Q, B_N = sqrt(q)/(1-q),
    |Ht| = sqrt(q)|z|. All moments closed form."""
    A_N = (1 + abs(KAPPA) / S1Q) / S1Q
    B_N = SQ_Q / (1 - Q)
    A = abs(cK) + abs(c1) * A_N
    B = abs(c0) * SQ_Q + abs(c1) * B_N               # |V| <= A + B|z|
    dC = abs(cC) + abs(cN) * A_N                     # constant part of |dV/da|
    e = abs(cN) * B_N + abs(cHt) * SQ_Q              # |z| part of |dV/da|
    m0 = gauss_tail_mass(L)
    m1 = z1_tail(L)
    m2 = z2_tail(L)
    # (1 + A + B|z|)(dC + e|z|) integrated over |z|>L
    c_const = (1 + A) * dC
    c_lin = (1 + A) * e + B * dC
    c_quad = B * e
    return c_const * m0 + c_lin * m1 + c_quad * m2


def T_of(a1, a2, s, g=None):
    """Certified E log Psi(V) at general kappa,
        V = kappa/D + c0 Ht + c1 N,
    (Huang's eq:def-sS-star-bLam-nu in moment coordinates).  Returns arb
    ball or None if a2^2 >= q."""
    if g is None:
        g = get_zt_grid()
    a2sq = a2 * a2 / Q
    from core import endpoints
    _, hi = endpoints(a2sq)
    if not (hi < 1):
        return None
    D = (1 - a2sq).sqrt()
    cK = KAPPA / D
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    gm, gc = [], []
    for Nm, Npm, Htm, Nc, Npc, Htc in zip(
            g.N_mid, g.Np_mid, g.Ht_mid, g.N_cell, g.Np_cell, g.Ht_cell):
        Vm = cK + c0 * Htm + c1 * Nm
        gm.append(logPsi(Vm))
        Vc = cK + c0 * Htc + c1 * Nc
        E = E_mills(Vc)
        Vp = c0 * SQ_Q + c1 * Npc
        gc.append(-E * Vp)
    main = g.integrate(gm, gc)
    # tail: |V| <= A + B|z|.  General kappa: E(u) <= max(u, 0) + 1 with
    # u = (kappa - sqrt(q) z)/sqrt(1-q), so
    #   |N| <= (1 + |kappa|/S1Q)/S1Q + (sqrt(q)/(1-q)) |z|,
    # and |V| <= |cK| + |c0| sqrt(q)|z| + |c1| |N|.
    A_N = (1 + abs(KAPPA) / S1Q) / S1Q
    B_N = SQ_Q / (1 - Q)
    A = abs(cK) + abs(c1) * A_N
    B = abs(c0) * SQ_Q + abs(c1) * B_N
    # log Psi(V) in [ -(V^2/2 + log(2pi)/2 + |V| + 2Psi(|V|)), 0 ]
    quad = ((A * A / 2 + (2 * arb.pi()).log() / 2 + A + 1) * gauss_tail_mass(g.L)
            + (A * B + B) * z1_tail(g.L) + (B * B / 2) * z2_tail(g.L))
    return main + (-quad).union(arb(0))


# ---------------------------------------------------------------------------
# Assembled value and the certified upper bound over an (a1, a2) cell.
# ---------------------------------------------------------------------------

def S_moment(a1, a2, s, duals=None, gz=None, gx=None):
    """Upper bound for H(a1,a2) + s^2 psi/2 + alpha T(a1,a2,s), using the
    dual dictionary for H. Returns (value_ball, T_ball) or (None, None) if
    a2^2 >= q. duals: list of (b1, b2, Phi_ball)."""
    T = T_of(a1, a2, s, gz)
    if T is None:
        return None, None
    if duals is None:
        Hub = arb(0).union(LOG2)     # trivial 0 <= H <= log 2
    else:
        Hub = None
        for (b1, b2, Phi) in duals:
            u = Phi - b1 * a1 - b2 * a2
            Hub = u if Hub is None else _ball_min(Hub, u)
    val = Hub + s * s * PSI / 2 + ALPHA * T
    return val, T


def _ball_min(a, b):
    """Enclosure of min(a, b) for arb balls."""
    from core import endpoints
    alo, ahi = endpoints(a)
    blo, bhi = endpoints(b)
    lo = alo if (alo < blo) else blo
    hi = ahi if (ahi < bhi) else bhi
    return lo.union(hi)


# ---------------------------------------------------------------------------
# Support function of the achievable moment body K = { (E[X Lam], E[M Lam]) :
# |Lam| <= 1 } of (X, M), X ~ N(0, psi), M = tanh(X):
#     h(u, v) = sup_{|Lam|<=1} E[Lam (u X + v M)] = E |u X + v M|.
# A cell (a-box) lies OUTSIDE K iff some direction (u, v) has
#     min_{a in box} (u a1 + v a2) > h(u, v).
# We use an upper bound on h (so the test is conservative / rigorous).
# ---------------------------------------------------------------------------

def h_support_upper(u, v, g=None):
    """Upper bound on h(u,v) = E|u X + v M|, X = sqrt(psi) z.

    Mean-value per cell where f = uX+vM keeps a sign: |f| smooth with
    d|f|/dz = sign(f) f', f' = sqrt(psi)(u + v(1-M^2)); at cells where f may
    change sign, fall back to the 0th-order enclosure |f(cell)|.upper. Plus a
    tail bound |f| <= |u| sqrt(psi)|z| + |v|."""
    if g is None:
        g = get_x_grid()
    from core import endpoints
    u = arb(u)
    v = arb(v)
    total = arb(0)
    for Xm, Mm, Xc, Mc, dMc, m0, m1p, m1m in zip(
            g.X_mid, g.M_mid, g.X_cell, g.M_cell, g.dM_cell, g.m0, g.m1p,
            g.m1m):
        fc = u * Xc + v * Mc
        flo, fhi = endpoints(fc)
        if flo > 0 or fhi < 0:
            # constant sign: mean-value  |f| = sign * f
            sgn = arb(1) if flo > 0 else arb(-1)
            fm = u * Xm + v * Mm
            dfc = u * SQ_PSI + v * dMc     # f'(cell); dMc = sqrt(psi)(1-M^2)
            total = total + sgn * (fm * m0 + dfc * m1p - dfc * m1m)
        else:
            _, afhi = endpoints(abs(fc))
            total = total + afhi * m0    # |f(cell)|.upper * m0
    tail = abs(u) * SQ_PSI * z1_tail(g.L) + abs(v) * gauss_tail_mass(g.L)
    return total + tail


_HFAN = None


def _get_hfan(ndir=360):
    """Precompute h upper bounds on a fan of directions (unit circle)."""
    global _HFAN
    if _HFAN is None or len(_HFAN) != ndir:
        import math
        fan = []
        for k in range(ndir):
            ang = math.pi * k / ndir     # half circle; sign symmetry covers rest
            u = arb(dec(str(round(math.cos(ang), 8))))
            v = arb(dec(str(round(math.sin(ang), 8))))
            fan.append((u, v, h_support_upper(u, v)))
        _HFAN = fan
    return _HFAN


def outside_K(a1lo, a1hi, a2lo, a2hi, ndir=360):
    """True if the a-box is certified to lie outside the achievable body K
    (hence contributes nothing to sup S_*). Tests a fan of directions."""
    from core import endpoints
    a1lo, a1hi = arb(dec(str(a1lo))), arb(dec(str(a1hi)))
    a2lo, a2hi = arb(dec(str(a2lo))), arb(dec(str(a2hi)))
    for (u, v, hub) in _get_hfan(ndir):
        # min over box of u a1 + v a2, for both (u,v) and (-u,-v)
        for su, sv in ((u, v), (-u, -v)):
            mn = _corner_min(su, sv, a1lo, a1hi, a2lo, a2hi)
            if mn > hub:
                return True
    return False


def _corner_min(u, v, a1lo, a1hi, a2lo, a2hi):
    t1 = u * a1lo if (u > 0) else u * a1hi
    t2 = v * a2lo if (v > 0) else v * a2hi
    lo1, _ = _end(t1)
    lo2, _ = _end(t2)
    return lo1 + lo2


def _end(x):
    from core import endpoints
    return endpoints(x)


# ---------------------------------------------------------------------------
# First and second (a1, a2)-derivatives of T over an a-box, general kappa
# (the port of huang_hessian._T_derivs).  V = kappa/D + c0 Ht + c1 N; the
# kappa-term contributes d(kappa/D)/da2 = kappa a2/(q D^3) to V2 and
# d^2(kappa/D)/da2^2 = kappa (1/(q D^3) + 3 a2^2/(q^2 D^5)) to V22; it has
# no a1-derivative and no z-dependence (so V2p, V22p keep their form).
# ---------------------------------------------------------------------------

def T_derivs(a1, a2, gz=None, s=None):
    """dT/da1, dT/da2, d2T/da1^2, d2T/da1da2, d2T/da2^2 over the a-box, at
    tilt s (an arb ball; may cover the optimal-s range)."""
    from core import endpoints
    if gz is None:
        gz = get_zt_grid()
    if s is None:
        s = S1Q
    a2sq = a2 * a2 / Q
    _, hi = endpoints(a2sq)
    if not (hi < 1):
        return None
    D = (1 - a2sq).sqrt()
    D3 = D * D * D
    D5 = D3 * D * D
    cK = KAPPA / D
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    V1_N = -(1 / (PSI * D))
    V2_Ht = -1 / (Q * D) - a2 * a2 / (Q * Q * D3)
    V2_N = -a1 * a2 / (PSI * Q * D3)
    V2_const = KAPPA * a2 / (Q * D3)
    V12_N = -a2 / (PSI * Q * D3)
    dInvD3 = 3 * a2 / (Q * D5)
    dV2_Ht = -(1 / Q) * (a2 / (Q * D3)) - (1 / (Q * Q)) * (
        2 * a2 / D3 + a2 * a2 * dInvD3)
    dV2_N = -(a1 / (PSI * Q)) * (1 / D3 + a2 * dInvD3)
    dV2_const = KAPPA * (1 / (Q * D3) + 3 * a2 * a2 / (Q * Q * D5))
    SQq = SQ_Q
    res = {'dT1': arb(0), 'dT2': arb(0),
           'd2T11': arb(0), 'd2T12': arb(0), 'd2T22': arb(0)}
    keys = ('dT1', 'dT2', 'd2T11', 'd2T12', 'd2T22')

    def integrand_and_deriv(Ht, N, Np, want_deriv):
        V = cK + c0 * Ht + c1 * N
        E = E_mills(V)
        tg = _mills_gap(E, V)
        Ep = _clamp01(E * tg)
        V1 = V1_N * N
        V2 = V2_const + V2_Ht * Ht + V2_N * N
        V12 = V12_N * N
        V22 = dV2_const + dV2_Ht * Ht + dV2_N * N
        vals = (-E * V1,
                -E * V2,
                -Ep * V1 * V1,
                -Ep * V1 * V2 - E * V12,
                -Ep * V2 * V2 - E * V22)
        if not want_deriv:
            return vals, None
        Vp = c0 * SQq + c1 * Np
        # (2E - V)(E - V) - 1 = (E + t) t - 1 with the stable Mills gap
        Epp = _clamp_Epp(E * ((E + tg) * tg - 1), tg)
        dE = Ep * Vp
        dEp = Epp * Vp
        V1p = V1_N * Np
        V2p = V2_Ht * SQq + V2_N * Np
        V12p = V12_N * Np
        V22p = dV2_Ht * SQq + dV2_N * Np
        d = (-(dE * V1 + E * V1p),
             -(dE * V2 + E * V2p),
             -(dEp * V1 * V1 + Ep * 2 * V1 * V1p),
             -(dEp * V1 * V2 + Ep * (V1p * V2 + V1 * V2p)
               + dE * V12 + E * V12p),
             -(dEp * V2 * V2 + Ep * 2 * V2 * V2p + dE * V22 + E * V22p))
        return vals, d

    for (Nm, Npm, Htm, Nc, Npc, Htc, m0, m1p, m1m) in zip(
            gz.N_mid, gz.Np_mid, gz.Ht_mid, gz.N_cell, gz.Np_cell,
            gz.Ht_cell, gz.m0, gz.m1p, gz.m1m):
        vmid, _ = integrand_and_deriv(Htm, Nm, Npm, False)
        _, dcell = integrand_and_deriv(Htc, Nc, Npc, True)
        for k, gm, gd in zip(keys, vmid, dcell):
            res[k] = res[k] + gm * m0 + gd * m1p - gd * m1m
    # |z| >= L tails: E <= 1+|V|, E' in (0,1); every kernel is bounded by a
    # product of two linear-in-|z| forms with the kappa-shifted N bound.
    lE = _lin_E(cK, c0, c1)
    lV1 = _lin_of(arb(0), arb(0), V1_N)
    lV2 = _lin_of(V2_const, V2_Ht, V2_N)
    lV12 = _lin_of(arb(0), arb(0), V12_N)
    lV22 = _lin_of(dV2_const, dV2_Ht, dV2_N)
    L = gz.L
    res['dT1'] = res['dT1'] + _pm(_ltail(_lmul(lE, lV1), L))
    res['dT2'] = res['dT2'] + _pm(_ltail(_lmul(lE, lV2), L))
    res['d2T11'] = res['d2T11'] + _pm(_ltail(_lmul(lV1, lV1), L))
    res['d2T12'] = res['d2T12'] + _pm(_ltail(_lmul(lV1, lV2), L)
                                      + _ltail(_lmul(lE, lV12), L))
    res['d2T22'] = res['d2T22'] + _pm(_ltail(_lmul(lV2, lV2), L)
                                      + _ltail(_lmul(lE, lV22), L))
    return res


def _absup(x):
    from core import endpoints
    lo, hi = endpoints(x)
    m = abs(lo).union(abs(hi))
    _, u = endpoints(m)
    return u


def _lin_of(cC, cHt, cN):
    """(p0, p1) with |cC + cHt*Ht + cN*N| <= p0 + p1|z| on the tail:
    Ht = sqrt(q) z, |N| <= A_N + B_N |z| with the kappa-shifted bound."""
    A_N = (1 + abs(KAPPA) / S1Q) / S1Q
    B_N = SQ_Q / (1 - Q)
    p0 = _absup(cC) + _absup(cN) * A_N
    p1 = _absup(cHt) * SQ_Q + _absup(cN) * B_N
    return p0, p1


def _lin_E(cK, c0, c1):
    """E(V) <= 1 + |V| <= e0 + e1 |z|, V = cK + c0 Ht + c1 N."""
    v0, v1 = _lin_of(cK, c0, c1)
    return 1 + v0, v1


def a_s_mixed(a1, a2, s, gz=None):
    """d_a1 d_s T, d_a2 d_s T, and d^2_s S = psi - alpha E[E'(V) N^2], over
    the a-box at tilt s.  d_ai d_s T = -E[E'(V) (dV/da_i) N].  General
    kappa: V = kappa/D + c0 Ht + c1 N, and dV/da2 carries the constant
    kappa a2/(q D^3); dV/ds = N is kappa-free in form."""
    from core import endpoints
    if gz is None:
        gz = get_zt_grid()
    a2sq = a2 * a2 / Q
    _, _hi = endpoints(a2sq)
    if not (_hi < 1):
        return None
    D = (1 - a2sq).sqrt()
    D3 = D * D * D
    cK = KAPPA / D
    c0 = -(a2 / Q) / D
    c1 = s - (a1 / PSI) / D
    V1_N = -(1 / (PSI * D))
    V2_Ht = -1 / (Q * D) - a2 * a2 / (Q * Q * D3)
    V2_N = -a1 * a2 / (PSI * Q * D3)
    V2_const = KAPPA * a2 / (Q * D3)

    def pieces(Ht, N, Np, want):
        V = cK + c0 * Ht + c1 * N
        E = E_mills(V)
        tg = _mills_gap(E, V)
        Ep = _clamp01(E * tg)
        V1 = V1_N * N
        V2 = V2_const + V2_Ht * Ht + V2_N * N
        vals = (Ep * V1 * N, Ep * V2 * N, Ep * N * N)
        if not want:
            return vals
        Vp = c0 * SQ_Q + c1 * Np
        # (2E - V)(E - V) - 1 = (E + t) t - 1 with the stable Mills gap
        Epp = _clamp_Epp(E * ((E + tg) * tg - 1), tg)
        dEp = Epp * Vp
        V1p = V1_N * Np
        V2p = V2_Ht * SQ_Q + V2_N * Np
        d = (dEp * V1 * N + Ep * (V1p * N + V1 * Np),
             dEp * V2 * N + Ep * (V2p * N + V2 * Np),
             dEp * N * N + Ep * 2 * N * Np)
        return d

    I = [arb(0), arb(0), arb(0)]
    for (Nm, Npm, Htm, Nc, Npc, Htc, m0, m1p, m1m) in zip(
            gz.N_mid, gz.Np_mid, gz.Ht_mid, gz.N_cell, gz.Np_cell,
            gz.Ht_cell, gz.m0, gz.m1p, gz.m1m):
        vm = pieces(Htm, Nm, Npm, False)
        dc = pieces(Htc, Nc, Npc, True)
        for k in range(3):
            I[k] = I[k] + vm[k] * m0 + dc[k] * m1p - dc[k] * m1m
    # |z| >= L tails: E' in (0,1), so kernels bounded by |V_i| N and N^2,
    # with the kappa-shifted N bound and the constant channel in dV/da2.
    lV1 = _lin_of(arb(0), arb(0), V1_N)
    lV2 = _lin_of(V2_const, V2_Ht, V2_N)
    lN = _lin_of(arb(0), arb(0), arb(1))
    L = gz.L
    I[0] = I[0] + _pm(_ltail(_lmul(lV1, lN), L))
    I[1] = I[1] + _pm(_ltail(_lmul(lV2, lN), L))
    I[2] = I[2] + _pm(_ltail(_lmul(lN, lN), L))
    asT1 = -ALPHA * I[0]
    asT2 = -ALPHA * I[1]
    Sss = PSI - ALPHA * I[2]
    return asT1, asT2, Sss


def _lmul(p, q):
    return (p[0] * q[0], p[0] * q[1] + p[1] * q[0], p[1] * q[1])


def _ltail(c, L):
    return (c[0] * gauss_tail_mass(L) + c[1] * z1_tail(L)
            + c[2] * z2_tail(L))


def _pm(t):
    return t.union(-t)
