"""Region I of Huang's Condition 1.3: S_* <= 0 on a star region around the
degenerate maximizer, by banded ray-concavity certificates.

The argument
------------
Let a* = grad Phi(1,0) (at the true parameters a* = (psi(1-q), q)).  For a
unit direction v and fixed slope sdot define the affine-tilt majorant

    phi_v(t) = H(x) + s(t)^2 psi/2 + alpha T(x, s(t)),  x = a* + t v,
    s(t) = s0 + t sdot,  s0 = sqrt(1-q).

phi_v >= S_* along the ray (G = min_s <= any fixed s), and at the true fixed
point phi_v(0) = 0, phi_v'(0) = 0 (Huang's unconditional identities: Gardner
stationarity osS(1,0) = 0, grad osS(1,0) = 0, and the tilt equation
d/ds[s^2 psi/2 + alpha T(a*, s)]|_{s0} = 0).  Hence

    phi_v''(t) <= 0 on [0, T(v)]  ==>  S_* <= 0 on the segment.

Splitting phi'' = (H-part) + (T-part),

    H-part = v^T grad^2 H(x) v = -v^T [grad^2 Phi(lambda(x))]^{-1} v,
    T-part = alpha v^T d2_a T v + 2 sdot (v . alpha d_a d_s T) + sdot^2 Sss,

the T-part is enclosed by certified fixed-grid integrals over the ray piece
(no dual variable).  For the H-part, since grad^2 Phi(lambda) =
E[f f^T sech^2(lambda . f)], any rigorous localization lambda(x) in LBOX gives

    grad^2 Phi(lambda(x))  <=  B_L := E[ f f^T  max_{lambda in LBOX}
                                         sech^2(lambda . f) ],

an explicit one-dimensional integral (the max is a per-z corner selection),
whence  H-part <= -v^T B_L^{-1} v.  With LBOX = R^2 this is the global bound
B_0 = E[f f^T]; with the band localization below it is nearly sharp.

Localization WITHOUT tracking the ill-conditioned dual: lambda(x) lies in the
sublevel set  S(x) = { lambda : Phi(lambda) - lambda.x <= Phi(1,0) - (1,0).x }
(the dual value at lambda(x) is the minimum, and (1,0) is feasible).  S(x) is
convex and contains (1,0); if the rigorous check

    Phi(lambda) - Phi(1,0) - (lambda - (1,0)) . x  >  0

holds everywhere on the boundary of a candidate box LBOX (a 1-D certified
sweep over the four edges; the check is LINEAR in x, so verifying it at the
corner points of the x-region suffices), then S(x) -- hence lambda(x) -- stays
inside LBOX for every x in the region.  The genuine size of S(x) is the
Bregman ball of radius  D = [Phi(1,0) - (1,0).x] - H(x) = O(|x-a*|^2), so the
boxes shrink quadratically-in-radius toward (1,0) and B_L -> grad^2 Phi(1,0):
the H-bound converges to the sharp one exactly where sharpness is needed.

The star region is processed in log radius-bands t in [r t_k, t_k] times the
angular fan; each (band x angle cell) is one certificate:
    quadT(x-sector, s-range)  -  v^T B_L(band)^{-1} v  <  0.
Every constant is a parameter BALL over the certified Ding-Sun rectangle, so
all certificates cover the true fixed point.

Run:  python huang_region1.py [nworkers]
Writes results/huang_region1.json (including the certified star table for the
stage-2 sweep).
"""

import os
import sys
import time
import json
import math
import hashlib
import platform
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from multiprocessing import Pool

import flint
from flint import arb, acb
import core
from core import (set_prec, dec, endpoints,
                  z1_tail, z2_tail, gauss_tail_mass)
import huang_cert_grid as hg

# General-kappa parameters, set by configure_region1(); PSI/Q/ALPHA and
# the T-side derivative machinery live in the configured grid module.
PSI = None
Q = None
ALPHA = None
KAPPA = None
KTAG = 'k'

# ---------------------------------------------------------------------------
# numeric policy constants (choices; rigor lives in the certificates)
# ---------------------------------------------------------------------------
A1S = None                # ~ a*_1 = psi(1-q); set by configure_region1
A2S = None                # ~ a*_2 = q
S0F = None                # ~ s0 = sqrt(1-q)


def ktag_of(kappa):
    v = float(kappa.mid()) if hasattr(kappa, 'mid') else float(kappa)
    return ('%.4f' % v).rstrip('0').rstrip('.').replace('-', 'n').replace('.', 'p')


def configure_region1(kappa, alpha, q, psi, w_ang=None):
    """Configure the grid module and this driver's per-kappa state
    variables: the maximizer a* = (psi(1-q), q), the stored tilt
    s0 = sqrt(1-q), and the weak-eigenvector angle W_ANG of the moment
    Hessian (re-derived per kappa; passing w_ang overrides).  The star
    radii T_LONG/T_MID/T_CORE stay policy constants - the certificates
    are rigorous for any choice, and a failing choice just fails."""
    global PSI, Q, ALPHA, KAPPA, A1S, A2S, S0F, W_ANG, SQ_PSI_B, KTAG
    global A1B, A2B, S_BASE
    import huang_cert_np as nr
    hg.configure(kappa, alpha, q, psi)
    KTAG = ktag_of(kappa)
    PSI, Q, ALPHA = hg.PSI, hg.Q, hg.ALPHA
    KAPPA = hg.KAPPA
    SQ_PSI_B = PSI.sqrt()
    # the maximizer a* = grad Phi(1,0) = (psi(1-q), q) and the optimal base
    # tilt s0 = sqrt(1-q) are CLOSED FORMS in the located parameters; keep
    # them as balls (symbolic), so every certificate covers the true point
    # with no stored-decimal origin inflation.  The floats below are numeric
    # aids only (fans, W_ANG, band schedule).
    A1B = PSI * (1 - Q)
    A2B = Q
    S_BASE = (1 - Q).sqrt()
    A1S = float(A1B.mid())
    A2S = float(A2B.mid())
    S0F = float(S_BASE.mid())
    nr.configure(float(KAPPA.mid()), float(ALPHA.mid()),
                 float(Q.mid()), float(PSI.mid()))
    if w_ang is None:
        import numpy as _np
        d = 2e-4
        def S(a1, a2):
            v = nr.S_star(a1, a2)
            return v[0] if isinstance(v, tuple) else v
        h11 = (S(A1S + d, A2S) - 2 * S(A1S, A2S) + S(A1S - d, A2S)) / d / d
        h22 = (S(A1S, A2S + d) - 2 * S(A1S, A2S) + S(A1S, A2S - d)) / d / d
        h12 = (S(A1S + d, A2S + d) - S(A1S + d, A2S - d)
               - S(A1S - d, A2S + d) + S(A1S - d, A2S - d)) / (4 * d * d)
        Hm = _np.array([[h11, h12], [h12, h22]])
        evals, evecs = _np.linalg.eigh(Hm)
        w = evecs[:, int(_np.argmax(evals))]   # weak = largest (least negative)
        w_ang = float(_np.arctan2(w[1], w[0])) % math.pi
    W_ANG = w_ang
W_ANG = None              # weak-eigenvector angle; set by configure_region1
N_ANG = 48                # angular cells over the full circle
WEDGE_HALF = 0.16         # |omega| below which rays are long
CONE_MID = 0.45
T_LONG = 0.0120           # star radii by angular zone (sweep MIN_SIDE 1e-3:
T_MID = 0.0080            # failure ellipse ~8.2e-3 along w, ~2.2e-3 across)
T_CORE = 0.0050
T_IN = 1.0e-6             # innermost band edge; [0, T_IN] is one last band
BAND_R = 0.72             # radius-band ratio
GRID_N_RAY = 2700         # zt-grid for the T-side
ZL = 10                   # acb range; explicit tails added

SQ_PSI_B = None
A1B = None                # a* = (psi(1-q), q) as parameter balls (symbolic)
A2B = None
S_BASE = None             # optimal base tilt sqrt(1-q) as a ball (symbolic)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')

_ANG_PAD = arb(dec('0.00000001'))
_RAD_PAD = arb(dec('0.000000000001'))
_LBOX_QUANT = Decimal('0.000000000001')
_ROOT_QUANT = Decimal('0.0000000001')


def _ball_rad(b):
    """Half-width of an arb ball, as a float (None-safe)."""
    if b is None:
        return None
    lo, hi = endpoints(b)
    return float(((hi - lo) / 2).mid())


def ang_dist(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def omega_of(theta):
    return min(ang_dist(theta, W_ANG), ang_dist(theta, W_ANG + math.pi))


def T_of_angle(theta):
    om = omega_of(theta)
    if om <= WEDGE_HALF:
        return T_LONG
    if om <= CONE_MID:
        return T_MID
    return T_CORE


def T_max_over(th0, th1):
    """Certified maximum of the piecewise star radius on an angle interval."""
    if th1 - th0 >= 2 * math.pi - 1e-12:
        return T_LONG
    mid = 0.5 * (th0 + th1)
    # Check all copies of the two zone centers near this interval.  Testing
    # interval intersection, rather than sampling, cannot miss a thin long
    # or middle wedge at a cell boundary.
    has_mid = False
    for k in range(-4, 5):
        c = W_ANG + k * math.pi
        if th0 <= c + WEDGE_HALF and th1 >= c - WEDGE_HALF:
            return T_LONG
        if th0 <= c + CONE_MID and th1 >= c - CONE_MID:
            has_mid = True
    return T_MID if has_mid else T_CORE


def _dec(x, nd=12):
    return arb(dec(f"{float(x):.{nd}f}"))


def _exact_decimal(x):
    """Parse a persisted decimal spelling as an exact rational Arb ball."""
    return arb(dec(str(x)))


def _quantize_lbox(values):
    """Outward-quantize a numeric lbox once and return decimal spellings.

    The resulting four strings are the certificate's actual box.  Both the
    boundary localization and B_Lambda must consume Arb values parsed from
    these same spellings; re-rounding either consumer can leave a sliver of
    the declared box unchecked.
    """
    if len(values) != 4:
        raise ValueError("lbox needs four endpoints")
    out = []
    for k, value in enumerate(values):
        rounding = ROUND_FLOOR if k in (0, 2) else ROUND_CEILING
        # Quantize around the actual binary64 proposal, not its shortened
        # display spelling.  The proposal is heuristic, but this makes the
        # relationship between that proposal and the certified box explicit.
        q = Decimal.from_float(float(value)).quantize(
            _LBOX_QUANT, rounding=rounding)
        out.append(format(q, 'f'))
    if not (Decimal(out[0]) < Decimal(out[1])
            and Decimal(out[2]) < Decimal(out[3])):
        raise ValueError(f"degenerate lbox {out}")
    return tuple(out)


def _validate_lbox(box):
    """Require a certainly ordered exact lbox in B_Lambda's domain."""
    if len(box) != 4:
        raise ValueError("lbox needs four Arb endpoints")
    l1lo, l1hi, l2lo, l2hi = box
    if not (l1hi > l1lo and l2hi > l2lo and l1lo > 0):
        raise ValueError(f"invalid exact lbox {box}")


# ---------------------------------------------------------------------------
# acb evaluators
# ---------------------------------------------------------------------------

def _XMth(z, l1, l2):
    X = acb(SQ_PSI_B) * z
    M = X.tanh()
    return X, M, acb(l1) * X + acb(l2) * M


def Phi_acb(l1, l2, tol=None):
    """Phi(lambda) = E log 2cosh(lambda . f), rigorous, with tail bound.
    l1, l2 may be balls.

    Analyticity guard: 2cosh vanishes at i(k+1/2)pi, so off the axis a ball
    of 2cosh(th) can be tightly negative-real; the principal log would then
    return a finite but WRONG enclosure for the analytic continuation.  We
    return a non-finite ball unless Re(2cosh th) is certainly positive,
    forcing the integrator to subdivide toward the axis (where 2cosh >= 2)."""
    def f(z, an):
        X, M, th = _XMth(z, l1, l2)
        c = 2 * th.cosh()
        if not (c.real > 0):
            return acb(arb(0, arb('inf')))
        return c.log() * core.c_phi(z)

    val = core.integrate(f, -ZL, ZL, abs_tol=tol)
    # |log 2cosh(th)| <= |th| + log 2 <= |l1| sqrt(psi) |z| + |l2| + log2
    LOG2 = arb(2).log()
    t = (abs(arb(l1)) * SQ_PSI_B * z1_tail(arb(ZL))
         + (abs(arb(l2)) + LOG2) * gauss_tail_mass(arb(ZL)))
    return val + t.union(-t)


def gradPhi_acb(l1, l2, tol=None):
    """(a1, a2) = grad Phi(lambda) = (E[X tanh th], E[M tanh th]); l1, l2 may
    be balls (widths propagate rigorously)."""
    def f1(z, an):
        X, M, th = _XMth(z, l1, l2)
        return X * th.tanh() * core.c_phi(z)

    def f2(z, an):
        X, M, th = _XMth(z, l1, l2)
        return M * th.tanh() * core.c_phi(z)

    a1 = core.integrate(f1, -ZL, ZL, abs_tol=tol)
    a2 = core.integrate(f2, -ZL, ZL, abs_tol=tol)
    t1 = SQ_PSI_B * z1_tail(arb(ZL))
    t2 = gauss_tail_mass(arb(ZL))
    return a1 + t1.union(-t1), a2 + t2.union(-t2)


def _c_stable(Dl, M):
    """cosh(X+Delta)/cosh(X) = cosh(Delta) + M sinh(Delta) as an acb ball,
    by the tighter of two exact forms (see Phi_breg0_acb docstring): the
    cosh/sinh form for |Delta| <= 1 (exact at Delta = 0), the nonnegative
    exponential split beyond (no e^{|Delta|} cancellation)."""
    dr = Dl.real
    di = Dl.imag
    from core import endpoints
    _, rhi = endpoints(abs(dr))
    _, ihi = endpoints(abs(di))
    if float(rhi) + float(ihi) <= 1.0:
        return Dl.cosh() + M * Dl.sinh()
    return (Dl.exp() * (1 + M) + (-Dl).exp() * (1 - M)) / 2


def Phi_breg0_acb(l1, l2, tol=None):
    """Anchored Bregman gap  B(lambda) := Phi(lambda) - Phi(1,0)
    - (lambda - (1,0)) . grad Phi(1,0),  as ONE correlated quadrature.

    The base-point identity grad Phi(1,0) = (E[X M], E[M^2]) = a* holds
    EXACTLY at the true parameters (Gaussian integration by parts), so with
    Delta := (l1-1) X + l2 M the integrand collapses algebraically:
        log2cosh(th) - log2cosh(X) - (l-b0).grad h
          = log(cosh Delta + M sinh Delta) - M Delta.
    Every factor is a small ball with THIN lambda-coefficients, so the
    parameter-ball width of PSI enters only at second order in |l - b0|
    (measured ~6e-5 |l-b0|^2 vs 2e-5 flat for the uncorrelated difference
    of two Phi_acb calls).  Differences B(l) - B(f) then enclose
    Phi(l) - Phi(f) - (l-f).a*  with fuzz quadratically small in the
    offsets -- the quantity edge_check needs against x = a* + offset.

    Analyticity guard: cosh Delta + M sinh Delta = cosh(X+Delta)/cosh X
    >= e^{-|Delta|} > 0 on the real axis; off-axis complex balls may cross
    zero, so return an infinite ball unless Re > 0 (forces subdivision).

    Stability: two exact forms of c, chosen per node by |Delta|.  The
    cosh/sinh form is EXACT at Delta = 0 (sinh kills the M term -- the
    inner-band gaps ~1e-7 need that) but subtracts two e^{|Delta|}-sized
    balls when M ~ +-1 and |Delta| is large (far z-tails at box corners
    with |l1 - 1| ~ 0.3), where the guard then fires on every subdivision
    and the integral comes back infinite.  The exponential split
    c = e^Delta (1+M)/2 + e^{-Delta} (1-M)/2  has both terms nonnegative
    on the real axis (no cancellation) but carries first-order M-fuzz
    near Delta = 0.  Branch at |Delta| ~ 1."""
    def f(z, an):
        X, M, th = _XMth(z, l1, l2)
        Dl = (acb(l1) - 1) * X + acb(l2) * M
        c = _c_stable(Dl, M)
        if not (c.real > 0):
            return acb(arb(0, arb('inf')))
        return (c.log() - M * Dl) * core.c_phi(z)

    val = core.integrate(f, -ZL, ZL, abs_tol=tol)
    # |integrand| = |Delta . (tanh(xi) - M)| <= 2 |Delta|
    #            <= 2 |l1-1| sqrt(psi) |z| + 2 |l2|
    t = (2 * abs(arb(l1) - 1) * SQ_PSI_B * z1_tail(arb(ZL))
         + 2 * abs(arb(l2)) * gauss_tail_mass(arb(ZL)))
    return val + t.union(-t)


def grad_diff_acb(l1, l2, tol=None):
    """grad Phi(lambda) - grad Phi(1,0) = (E[X R], E[M R]) with
    R = tanh(th) - M = (1 - M^2) tanh(Delta) / (1 + M tanh(Delta)),
    Delta = (l1-1) X + l2 M  (exact tanh addition identity; correlated, so
    the enclosure scales with |lambda - (1,0)| instead of the raw
    parameter-ball width).  Since grad Phi(1,0) = a* exactly, this returns
    a(lambda) - a* -- the slack term edge_check needs against offsets."""
    def _R(z):
        # R = (1 - M^2) tanh(D)/(1 + M tanh(D)) = (1 - M^2) sinh(D)/c with
        # the SAME regime-branched stable c as Phi_breg0_acb (the naive
        # denominator 1 + M tanh(D) cancels to 0-width-2 at M ~ +-1,
        # |D| large, and the guard then fires forever).
        X, M, _ = _XMth(z, l1, l2)
        Dl = (acb(l1) - 1) * X + acb(l2) * M
        c = _c_stable(Dl, M)
        if not (c.real > 0):
            return None, None, acb(arb(0, arb('inf')))
        return X, M, (1 - M * M) * Dl.sinh() / c

    def f1(z, an):
        X, M, R = _R(z)
        return R if X is None else X * R * core.c_phi(z)

    def f2(z, an):
        X, M, R = _R(z)
        return R if X is None else M * R * core.c_phi(z)

    a1 = core.integrate(f1, -ZL, ZL, abs_tol=tol)
    a2 = core.integrate(f2, -ZL, ZL, abs_tol=tol)
    # |tanh th - M| <= 2, so |X R| <= 2|X|, |M R| <= 2
    t1 = 2 * SQ_PSI_B * z1_tail(arb(ZL))
    t2 = 2 * gauss_tail_mass(arb(ZL))
    return a1 + t1.union(-t1), a2 + t2.union(-t2)


def _sech2_corner(l1c, l2c):
    """Integrand factory: z -> sech^2(l1c X + l2c M) f_i f_j phi(z)."""
    def make(i, j):
        def f(z, an):
            X, M, th = _XMth(z, l1c, l2c)
            s2 = 1 / (th.cosh() ** 2)
            fi = X if i == 1 else M
            fj = X if j == 1 else M
            return fi * fj * s2 * core.c_phi(z)
        return f
    return make


def _ff_only():
    def make(i, j):
        def f(z, an):
            X = acb(SQ_PSI_B) * z
            M = X.tanh()
            fi = X if i == 1 else M
            fj = X if j == 1 else M
            return fi * fj * core.c_phi(z)
        return f
    return make


def B_Lambda(l1lo, l1hi, l2lo, l2hi):
    """B_L = E[f f^T max_{lambda in box} sech^2(lambda . f)] as a 2x2 ball
    matrix (b11, b12, b22).

    Corner structure: theta = l1 X + l2 M is bilinear in lambda, so its range
    over the box is spanned by the four corners.  For z > 0 (X, M > 0) the
    minimum of theta is at (l1lo, l2lo); for z < 0 the maximum is there; the
    max of sech^2 over the box is 1 where the corner range straddles 0 and
    sech^2(nearest corner value) otherwise.  With zp = the positive root of
    l1lo X + l2lo M = 0 (zp = 0 if l2lo >= 0), the pointwise max equals
    sech^2(l1lo X + l2lo M) on |z| >= zp and 1 on |z| < zp.  We enlarge
    [-zp, zp] to a rigorous outer bracket [-zq, zq] (max <= 1 always), so the
    three-piece split is sound and each piece is analytic.  The returned
    metadata records the exact split point and its Arb sign certificate."""
    _validate_lbox((l1lo, l1hi, l2lo, l2hi))
    # Corner selection (audited): theta = l1 X + l2 M is increasing in l1
    # and l2 for z > 0 and decreasing in both for z < 0, so on both tails the
    # box extreme of theta nearest zero is attained at (l1lo, l2lo), and the
    # two straddle roots coincide (g(|z|) = l1lo sqrt(psi)|z| +
    # l2lo tanh(sqrt(psi)|z|) by oddness).  Preconditions: l1lo > 0 and a
    # valid bracket g(5) > 0.  The positive-root branch searches an exact
    # decimal grid and accepts only Arb proofs of g(zq)>0 and g'(zq)>0.  The
    # two zq=0 branches instead prove global monotonicity with g(0)=0.
    l1lb, _ = endpoints(l1lo)
    l2lb, l2ub = endpoints(l2lo)
    if not (l1lb > 0):
        raise ValueError(f"B_Lambda needs l1lo > 0, got {l1lo}")

    def g_arb(z):
        z = arb(z)
        x = SQ_PSI_B * z
        return l1lo * x + l2lo * x.tanh()

    def gprime_arb(z):
        z = arb(z)
        x = SQ_PSI_B * z
        sech2 = 1 / (x.cosh() * x.cosh())
        return SQ_PSI_B * (l1lo + l2lo * sech2)

    far_z = _exact_decimal('5.000000000000')
    far_lo, _ = endpoints(g_arb(far_z))
    if not (far_lo > 0):
        raise ValueError(f"B_Lambda Arb bracket invalid: l1lo={l1lo}, "
                         f"l2lo={l2lo}, g(5)={g_arb(far_z)}")

    sum_lo, sum_hi = endpoints(l1lo + l2lo)
    if l2lb >= 0:
        # Both coefficients are nonnegative, hence g(z) >= 0 for z >= 0.
        zq = arb(0)
        split_cert = dict(mode='l2_nonnegative', zq='0',
                          g_zq='0_exact',
                          condition_lower=str(l2lb),
                          g_far_lower=str(far_lo))
    elif sum_lo >= 0:
        # l2 < 0 but g'(z) >= sqrt(psi) (l1+l2) >= 0.
        zq = arb(0)
        split_cert = dict(mode='derivative_nonnegative', zq='0',
                          g_zq='0_exact',
                          condition_lower=str(sum_lo),
                          g_far_lower=str(far_lo))
    else:
        if not (l2ub < 0):
            raise ValueError(f"B_Lambda cannot classify l2lo={l2lo}")
        if not (sum_hi < 0):
            raise ValueError(f"B_Lambda root branch needs l1lo+l2lo < 0, "
                             f"got {l1lo + l2lo}")
        far_gp_lo, _ = endpoints(gprime_arb(far_z))
        if not (far_gp_lo > 0):
            raise ValueError(f"B_Lambda derivative bracket invalid: "
                             f"g'(5)={gprime_arb(far_z)}")

        # Binary-search exact integer multiples of ROOT_QUANT.  An uncertain
        # Arb sign is treated as not certified and moves the lower endpoint
        # upward; only a certainly-positive g and g' can become the split.
        lo_units = 0
        hi_units = int(Decimal('5') / _ROOT_QUANT)
        while hi_units - lo_units > 1:
            mid_units = (lo_units + hi_units) // 2
            zq_s = format(Decimal(mid_units) * _ROOT_QUANT, 'f')
            zq = _exact_decimal(zq_s)
            g_zq_lo, _ = endpoints(g_arb(zq))
            gp_zq_lo, _ = endpoints(gprime_arb(zq))
            if g_zq_lo > 0 and gp_zq_lo > 0:
                hi_units = mid_units
            else:
                lo_units = mid_units
        zq_s = format(Decimal(hi_units) * _ROOT_QUANT, 'f')
        zq = _exact_decimal(zq_s)
        g_zq_lo, _ = endpoints(g_arb(zq))
        gp_zq_lo, _ = endpoints(gprime_arb(zq))
        if not (g_zq_lo > 0 and gp_zq_lo > 0):
            raise ValueError("certified B_Lambda grid split not found")
        split_cert = dict(mode='positive_root_outer', zq=zq_s,
                          g_zq_lower=str(g_zq_lo),
                          gprime_zq_lower=str(gp_zq_lo),
                          g_far_lower=str(far_lo))

    mk_s = _sech2_corner(l1lo, l2lo)
    mk_f = _ff_only()
    out = []
    for (i, j) in ((1, 1), (1, 2), (2, 2)):
        if zq > 0:
            v = (core.integrate(mk_s(i, j), -ZL, -zq)
                 + core.integrate(mk_f(i, j), -zq, zq)
                 + core.integrate(mk_s(i, j), zq, ZL))
        else:
            v = core.integrate(mk_s(i, j), -ZL, ZL)
        # tails: |f_i f_j| <= psi z^2, sqrt(psi)|z|, 1
        if (i, j) == (1, 1):
            t = PSI * z2_tail(arb(ZL))
        elif (i, j) == (1, 2):
            t = SQ_PSI_B * z1_tail(arb(ZL))
        else:
            t = gauss_tail_mass(arb(ZL))
        out.append(v + t.union(-t))
    return out[0], out[1], out[2], split_cert


def Binv_form(b11, b12, b22, v1, v2):
    """v^T B^{-1} v for the 2x2 ball matrix; None if det not certainly > 0."""
    det = b11 * b22 - b12 * b12
    lo, _ = endpoints(det)
    if not (lo > 0):
        return None
    return (v1 * v1 * b22 - 2 * v1 * v2 * b12 + v2 * v2 * b11) / det


# ---------------------------------------------------------------------------
# lambda localization: sublevel edge check
# ---------------------------------------------------------------------------

def edge_check(l1lo, l1hi, l2lo, l2hi, xcorners, fan, nseg=64):
    """Multi-anchor localization.  lambda(x) lies in EVERY sublevel set
        S_k(x) = { lam : Phi(lam) - lam.x <= Phi(lhat_k) - lhat_k.x }
    (the dual value is the global minimum), hence in their intersection
    S(x).  S(x) is convex and contains the fan point lhat_{k*(x)} attaining
    min_k [Phi(lhat_k) - lhat_k.x], which lies inside the box; so if S(x)
    misses the box boundary, S(x) -- hence lambda(x) -- is inside the box.

    Boundary test: for lam on each edge, need for every x in the hull:
    exists k with  g_k(lam; x) := Phi(lam) - Phi(lhat_k) - (lam - lhat_k).x
    > 0.  The maximum is convex, so corners alone do not suffice when the
    anchor can switch.  We therefore use one common anchor or a fixed convex
    combination of anchors on each certified tangent polygon; the resulting
    affine lower bound is checked at every polygon vertex.

    Evaluation: interval evaluation of Phi over a segment ball loses the
    near-cancellation with the linear term (the valley of Phi(.) - (.).x is
    FLAT along the weak direction), so instead each segment uses the exact
    mean-value form
        g_k(lam) >= g_k(mid) - h * max_seg |dPhi/du - x_u|,
    with the derivative dPhi/du enclosed ONCE PER EDGE (its smallness near
    the valley is genuine, and enclosing it does not require tightness).

    General kappa: the located parameter balls are ~1e-4 wide, so a plain
    Phi_acb value carries ~2e-5 of parameter fuzz -- fatally wide next to
    the thin valley margins.  All values here are therefore built from the
    CORRELATED anchored-Bregman primitive B := Phi_breg0_acb (fuzz
    quadratically small in |lambda - (1,0)|), and xcorners are OFFSETS from
    the true maximizer a* (kept symbolic; a* = grad Phi(1,0) exactly):
        g_k(lam; a* + off) = B(lam) - B(lhat_k) - (lam - lhat_k) . off.
    Returns bool."""
    _validate_lbox((l1lo, l1hi, l2lo, l2hi))
    fanb = [(_dec(f1, 10), _dec(f2, 10)) for (f1, f2) in fan]
    if not all(l1lo <= f1 <= l1hi and l2lo <= f2 <= l2hi
               for f1, f2 in fanb):
        raise ValueError("a localization fan anchor lies outside its lbox")
    phik = [Phi_breg0_acb(f1, f2) for (f1, f2) in fanb]
    if xcorners and isinstance(xcorners[0][0], (tuple, list)):
        xpolys = [[(_dec(x1, 12), _dec(x2, 12)) for (x1, x2) in poly]
                  for poly in xcorners]
    else:
        xpolys = [[(_dec(x1, 12), _dec(x2, 12)) for (x1, x2) in xcorners]]
    # The boundary margins are thin along the weak valley; a 5e-4 derivative
    # ball can drown them.  Tighten the certified quadrature tolerance.
    tol3 = arb(dec('0.000000001'))

    def seg_ok(fix, fval, u0, u1, depth):
        """Certify the segment [u0, u1] of the edge, bisecting adaptively.
        Uses the per-SEGMENT derivative enclosure, which is genuinely small
        near the valley crossing (where the value margin is small too)."""
        # u0/u1/fval are exact Arb values derived from the one persisted
        # lbox.  Keep recursive midpoints in Arb so no edge is silently moved
        # by a second 10- or 12-digit decimal serialization.
        mb = (u0 + u1) / 2
        fb = fval
        sb = u0.union(u1)
        if fix == 2:
            e1, e2 = mb, fb
            a1e, a2e = grad_diff_acb(sb, fb, tol=tol3)
            deriv = a1e
        else:
            e1, e2 = fb, mb
            a1e, a2e = grad_diff_acb(fb, sb, tol=tol3)
            deriv = a2e
        phie = Phi_breg0_acb(e1, e2)
        hb = (u1 - u0) / 2
        def poly_ok(xb):
            # Build the rigorous value intervals at each hull vertex.  The old
            # code accepted a different anchor k at each vertex; that reverses
            # the quantifiers and is not implied by convexity.  We first try
            # one common anchor, then a fixed convex combination of anchors.
            # Since max_k g_k >= sum_k w_k g_k for w_k>=0, sum w_k=1, the latter
            # remains a valid lower bound for every x in this convex polygon.
            vals = []
            for (x1, x2) in xb:
                xu = x1 if fix == 2 else x2
                du = deriv - xu
                dlo, dhi = endpoints(du)
                dmax = abs(dlo).union(abs(dhi))
                slack = hb * dmax
                vals.append([phie - pk - (e1 - f1) * x1
                             - (e2 - f2) * x2 - slack
                             for (f1, f2), pk in zip(fanb, phik)])
            for k in range(len(fanb)):
                if all(endpoints(vals[j][k])[0] > 0
                       for j in range(len(xb))):
                    return True

            # Choose weights from midpoint values only; the final decision
            # below is made with the full Arb intervals, so this numerical LP
            # is merely a sound anchor-selection heuristic.
            try:
                from decimal import Decimal, ROUND_DOWN
                from scipy.optimize import linprog
                gv = [[float(v.mid()) for v in row] for row in vals]
                nk = len(fanb)
                Aub = [[-gv[j][k] for k in range(nk)] + [1.0]
                       for j in range(len(xb))]
                sol = linprog([0.0] * nk + [-1.0], A_ub=Aub,
                              b_ub=[0.0] * len(xb),
                              A_eq=[[1.0] * nk + [0.0]], b_eq=[1.0],
                              bounds=[(0.0, None)] * nk + [(None, None)],
                              method='highs')
                if sol.success:
                    q = [Decimal(str(max(0.0, w))).quantize(
                        Decimal('0.0000000001'), rounding=ROUND_DOWN)
                         for w in sol.x[:nk]]
                    imax = max(range(nk), key=lambda i: q[i])
                    q[imax] += Decimal(1) - sum(q)
                    if any(w < 0 for w in q) or sum(q) != Decimal(1):
                        return False
                    wb = [arb(dec(format(w, 'f'))) for w in q]
                    return all(endpoints(sum((wb[k] * vals[j][k]
                                              for k in range(nk)), arb(0)))[0] > 0
                               for j in range(len(xb)))
            except Exception:
                return False
            return False

        if all(poly_ok(xb) for xb in xpolys):
            return True
        if depth >= 14:
            if os.environ.get('R1_DEBUG'):
                print(f"  edge_check fail: fix={fix} fval={fval} "
                      f"seg=[{u0},{u1}]", flush=True)
                for xb in xpolys:
                    if poly_ok(xb):
                        continue
                    for j, (x1, x2) in enumerate(xb):
                        xu = x1 if fix == 2 else x2
                        du = deriv - xu
                        dlo, dhi = endpoints(du)
                        dmax = abs(dlo).union(abs(dhi))
                        slack = hb * dmax
                        row = [phie - pk - (e1 - f1) * x1
                               - (e2 - f2) * x2 - slack
                               for (f1, f2), pk in zip(fanb, phik)]
                        best = max(range(len(row)),
                                   key=lambda i: float(row[i].mid()))
                        blo, bhi = endpoints(row[best])
                        print(f"    vtx {j} x=({float(x1.mid()):+.6f},"
                              f"{float(x2.mid()):+.6f}) best k={best} "
                              f"anchor=({fanb[best][0]},{fanb[best][1]}) "
                              f"g=[{float(blo):.3e},{float(bhi):.3e}]",
                              flush=True)
                    break
            return False
        m = (u0 + u1) / 2
        return (seg_ok(fix, fval, u0, m, depth + 1)
                and seg_ok(fix, fval, m, u1, depth + 1))

    edges = [(2, l2lo, l1lo, l1hi), (2, l2hi, l1lo, l1hi),
             (1, l1lo, l2lo, l2hi), (1, l1hi, l2lo, l2hi)]
    for (fix, fval, vlo, vhi) in edges:
        n0 = 8
        for k in range(n0):
            u0 = vlo + (vhi - vlo) * k / n0
            u1 = vlo + (vhi - vlo) * (k + 1) / n0
            if not seg_ok(fix, fval, u0, u1, 0):
                return False
    return True


# ---------------------------------------------------------------------------
# T-side quadratic form
# ---------------------------------------------------------------------------

def quadT_box(x1, x2, sb, v1, v2, sdot, gz):
    Td = hg.T_derivs(x1, x2, gz, sb)
    if Td is None:
        return None
    asT1, asT2, Sss = hg.a_s_mixed(x1, x2, sb, gz)
    return (ALPHA * (v1 * v1 * Td['d2T11'] + 2 * v1 * v2 * Td['d2T12']
                     + v2 * v2 * Td['d2T22'])
            + 2 * sdot * (v1 * asT1 + v2 * asT2)
            + sdot * sdot * Sss)


def _sdot_of(th):
    import huang_cert_np as nr
    d = 1e-4
    v = (math.cos(th), math.sin(th))
    s_p = nr.G(A1S + d * v[0], A2S + d * v[1])[0]
    s_m = nr.G(A1S - d * v[0], A2S - d * v[1])[0]
    sd = (s_p - s_m) / (2 * d)
    return sd if sd == sd else 0.0


# ---------------------------------------------------------------------------
# band machinery
# ---------------------------------------------------------------------------

def bands():
    """Jobs (t0, t1, th0, th1): radius bands from T_LONG down to 0, split
    into angular chunks when t1 is large (per-chunk dual fans keep the
    sublevel level-gaps -- hence the lambda boxes and B_L -- small)."""
    out = []
    t1 = T_LONG
    while t1 > T_IN:
        t0 = max(t1 * BAND_R, T_IN)
        if t1 > 4.5e-4:
            # chunk the allowed arcs (around +-w and, when t0 < T_MID/T_CORE,
            # the wider zones); chunk width 0.08 rad
            arcs = []
            if t0 < T_CORE:
                arcs = [(0.0, 2 * math.pi)]
            elif t0 < T_MID:
                for base in (W_ANG, W_ANG + math.pi):
                    arcs.append((base - CONE_MID, base + CONE_MID))
            else:
                for base in (W_ANG, W_ANG + math.pi):
                    arcs.append((base - WEDGE_HALF, base + WEDGE_HALF))
            for (a0, a1) in arcs:
                # narrow chunks near +-w (the lambda image spreads ~220x the
                # transverse x-extent there); coarser elsewhere
                th = a0
                while th < a1 - 1e-12:
                    om = omega_of(th + 0.008)
                    cw = 0.016 if om <= WEDGE_HALF + 0.05 else 0.05
                    out.append((t0, t1, th, min(th + cw, a1)))
                    th += cw
        else:
            out.append((t0, t1, 0.0, 2 * math.pi))
        t1 = t0
    out.append((0.0, T_IN, 0.0, 2 * math.pi))
    return out


def _allowed(theta, t0):
    """Directions this band actually processes."""
    return T_of_angle(theta) > t0


def fan_and_box(t0, t1, th0, th1, pad_mult=1.0):
    """Numeric dual fan over the chunk's processed sector and a padded box
    around its hull.  The fan anchors the multi-anchor sublevel localization;
    the box is what edge_check certifies and B_Lambda consumes."""
    import huang_cert_np as nr
    fan = [(1.0, 0.0)]
    nth = 9
    ths = [th0, th1] + [th0 + (th1 - th0) * (k + 0.5) / nth
                         for k in range(nth)]
    for th in ths:
        if not _allowed(th, t0) and (th1 - th0) < 6:
            continue
        Tt = T_of_angle(th)
        r_hi = min(t1, Tt)
        r_lo = max(t0, 1e-7)
        # sample INTERIOR radii too: near the weak wedges the dual image of
        # the sector stretches ~0.4 in lambda, and anchors only at the two
        # arcs leave the middle of the image unanchored -- the sublevel
        # intersection bulges through the gap and edge_check fails at every
        # pad (measured at kappa = 0.05, plus-side wedge edge).
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            tt = r_lo + (r_hi - r_lo) * frac
            lam = nr.dual_of(A1S + tt * math.cos(th), A2S + tt * math.sin(th))
            if lam is not None and abs(lam[0] - 1) < 2 and abs(lam[1]) < 2:
                fan.append((round(lam[0], 8), round(lam[1], 8)))
    l1s = [f[0] for f in fan]
    l2s = [f[1] for f in fan]
    c1 = 0.5 * (min(l1s) + max(l1s))
    c2 = 0.5 * (min(l2s) + max(l2s))
    # pads: the sublevel sets are FLAT along e_small ~ (0.57, -0.82); the pad
    # must clear the fan's level-gap in that direction, which shrinks with
    # the chunk size.  Escalates via pad_mult on edge_check failure.
    d1 = (0.5 * (max(l1s) - min(l1s)) * 1.7 + 3e-3 + 0.35 * t1) * pad_mult
    d2 = (0.5 * (max(l2s) - min(l2s)) * 1.7 + 9e-3 + 3.0 * t1) * pad_mult
    return fan, _quantize_lbox(
        (c1 - d1, c1 + d1, c2 - d2, c2 + d2))


def xhull_of_band(t0, t1, th0, th1):
    """Convex-hull vertices of the chunk's x-sector, as OFFSETS from the
    true maximizer a* (a* itself stays symbolic: edge_check consumes these
    against the anchored-Bregman values, for which a* = grad Phi(1,0)
    exactly -- no stored-decimal origin ball is needed, and the hull is
    exact offset geometry).

    For a narrow chunk: tangent polygons containing the polar sector
    {t v(th)} (the outer vertex pushed past the sagitta, so the hull
    contains the outer arc; the inner-arc chord dips toward 0, only
    enlarging the hull -- sound).  Using true sector points avoids phantom
    bounding-box corners, whose far-off duals would inflate the sublevel
    level-gaps.  For wide chunks (full circle) fall back to the
    circumscribing square of the offset disk."""
    if (th1 - th0) < 1.0:
        from core import endpoints
        # Split at every radius-zone boundary before forming tangent polygons;
        # otherwise a rectangular/sector hull can introduce phantom corners.
        cuts = [th0, th1]
        for k in range(-4, 5):
            c = W_ANG + k * math.pi
            for d in (WEDGE_HALF, CONE_MID):
                if th0 < c - d < th1:
                    cuts.append(c - d)
                if th0 < c + d < th1:
                    cuts.append(c + d)
        cuts = sorted(set(cuts))
        polys = []
        for ca, cb in zip(cuts[:-1], cuts[1:]):
            nsub = max(1, int(math.ceil((cb - ca) / 0.25)))
            for j in range(nsub):
                aa = ca + (cb - ca) * j / nsub
                bb = ca + (cb - ca) * (j + 1) / nsub
                hh = (bb - aa) / 2
                mm = (aa + bb) / 2
                rr = min(t1, T_max_over(aa, bb))
                vertices = []
                for rad in (t0, rr):
                    rb = rad / math.cos(hh) * (1 + 1e-12)
                    raw = ((rad, aa), (rb, mm), (rad, bb))
                    for rv, ang in raw:
                        thb = _dec(ang, 15)
                        rball = _dec(rv, 15)
                        dx = rball * thb.cos()
                        dy = rball * thb.sin()
                        x1lo, x1hi = endpoints(dx)
                        x2lo, x2hi = endpoints(dy)
                        xx = float(((x1lo + x1hi) / 2).mid())
                        yy = float(((x2lo + x2hi) / 2).mid())
                        for sx in (-1, 1):
                            for sy in (-1, 1):
                                vertices.append((xx + sx * 2.1e-7,
                                                 yy + sy * 2.1e-7))
                polys.append(vertices)
        return polys
    r = t1 * (1 + 1e-12) + 1e-9
    return [(s1 * r, s2 * r) for s1 in (-1, 1) for s2 in (-1, 1)]


def band_job(job):
    """Certify one radius band: edge-check the lambda box, build B_L, then
    certify quadT - v B_L^{-1} v < 0 on every angular cell of the fan
    (adaptive angular bisection on failure).  job = (t0, t1)."""
    set_prec(50)
    t0, t1, th0, th1 = job
    gz = hg.get_zt_grid(9, GRID_N_RAY)
    xc = xhull_of_band(t0, t1, th0, th1)
    ok_loc = False
    lbox = None
    lboxb = None
    fan = None
    used_pad_mult = None
    for pad_mult in (1.0, 1.6, 2.4):
        fan, lbox = fan_and_box(t0, t1, th0, th1, pad_mult)
        lboxb = tuple(_exact_decimal(x) for x in lbox)
        try:
            _validate_lbox(lboxb)
        except ValueError:
            # A numeric proposal outside B_Lambda's l1lo>0 domain cannot be
            # a certificate.  Treat it as a failed proposal, not a worker
            # crash; a valid earlier/later pad is still checked normally.
            continue
        if edge_check(*lboxb, xc, fan):
            ok_loc = True
            used_pad_mult = pad_mult
            break
    if not ok_loc:
        return dict(band=[t0, t1, th0, th1], ok=False, why='edge_check',
                    lbox=list(lbox) if lbox is not None else None)
    l1lo, l1hi, l2lo, l2hi = lboxb
    b11, b12, b22, split_cert = B_Lambda(
        l1lo, l1hi, l2lo, l2hi)
    # angular cells of this chunk, with adaptive bisection
    fails = []
    ncell = 0
    worst = -1e9
    nc0 = max(2, int(math.ceil((th1 - th0) / (2 * math.pi / N_ANG))))
    stack = [(th0 + (th1 - th0) * k / nc0,
              th0 + (th1 - th0) * (k + 1) / nc0) for k in range(nc0)]
    while stack:
        ta, tb_ = stack.pop()
        thc = 0.5 * (ta + tb_)
        if t0 >= T_max_over(ta, tb_):
            continue                      # outside the star in this direction
        thb = (_dec(ta, 8).union(_dec(tb_, 8))
               + _ANG_PAD.union(-_ANG_PAD))
        v1, v2 = thb.cos(), thb.sin()
        qB = Binv_form(b11, b12, b22, v1, v2)
        sdf = _sdot_of(thc)
        sdot = _dec(round(sdf, 6), 8)
        # adaptive radial walk: pieces shrink where the enclosure is wide
        # (wide s/x-balls inflate the far-tail cells of the T-integrals)
        tE = min(t1, T_max_over(ta, tb_))
        ok_piece = qB is not None
        hi = None
        u = t0
        du = min(4e-4, tE - t0)
        while ok_piece and u < tE - 1e-15:
            u1 = min(u + du, tE)
            tt = (_dec(u, 12).union(_dec(u1, 12))
                  + _RAD_PAD.union(-_RAD_PAD))
            # ray origin SYMBOLIC: a* = (psi(1-q), q) as parameter balls, so
            # the true maximizer (at the true parameters) is covered exactly;
            # the tilt base is the certified s*-bracket S_BASE.  The pinned
            # identities (value, gradient, tilt optimality) hold on the TRUE
            # ray, and these balls contain it.
            x1 = A1B + tt * v1
            x2 = A2B + tt * v2
            sb = S_BASE + tt * sdot
            slo, _ = endpoints(sb)
            if not (slo > 0):
                ok_piece = False
                break
            qT = quadT_box(x1, x2, sb, v1, v2, sdot, gz)
            val = None if qT is None else qT - qB
            hi = None
            if val is not None:
                _, hi = endpoints(val)
            if val is None or not (hi < 0):
                if os.environ.get('R1_DEBUG'):
                    print('  piece fail th=[%.6f,%.6f] u=%.6f du=%.2e '
                          'hi=%s' % (ta, tb_, u, du,
                                     None if hi is None else float(
                                         hi.mid() if hasattr(hi, 'mid')
                                         else hi)), flush=True)
                if du > 2.5e-5:
                    du *= 0.5
                    continue
                ok_piece = False
                break
            u = u1
            du = min(du * 1.6, 5e-4)
        if not ok_piece:
            if (tb_ - ta) > 2e-3:
                m = 0.5 * (ta + tb_)
                stack.append((ta, m))
                stack.append((m, tb_))
                continue
            fails.append(dict(th=[ta, tb_],
                              hi=None if hi is None else float(
                                  hi.mid() if hasattr(hi, 'mid') else hi)))
            continue
        ncell += 1
        worst = max(worst, float(hi.mid() if hasattr(hi, 'mid') else hi))
    bmat_packets = None
    try:
        import block3bc_exact as exact
        bmat_packets = [exact.arb_packet(v) for v in (b11, b12, b22)]
    except Exception:
        pass
    return dict(band=[t0, t1, th0, th1], ok=not fails, cells=ncell,
                worst=worst, fails=fails[:6], nfail=len(fails),
                lbox=list(lbox), pad_mult=used_pad_mult,
                bmat_packets=bmat_packets,
                root_certificate=split_cert)


def _init():
    set_prec(50)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    jobs = bands()
    print(f"{len(jobs)} radius bands, {nw} workers", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    out = []
    fails = 0
    with Pool(nw, initializer=_init) as pool:
        for res in pool.imap_unordered(band_job, jobs):
            out.append(res)
            tag = 'OK  ' if res.get('ok') else 'FAIL'
            print(f"{tag} band={res['band']} cells={res.get('cells')} "
                  f"worst={res.get('worst')} nfail={res.get('nfail', 0)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if not res.get('ok'):
                fails += 1
    star = dict(W_ANG=W_ANG, WEDGE_HALF=WEDGE_HALF, CONE_MID=CONE_MID,
                T_LONG=T_LONG, T_MID=T_MID, T_CORE=T_CORE, N_ANG=N_ANG,
                A1S=A1S, A2S=A2S, S0F=S0F,
                origin='symbolic a*=(psi(1-q),q), s0=sqrt(1-q); '
                       'anchored-Bregman dual localization',
                a1_ball_rad=_ball_rad(A1B), s0_ball_rad=_ball_rad(S_BASE))
    source_hash = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    import huang_cert_np as nr
    proof_sources = {
        'huang_cert_region1.py': __file__,
        'core.py': core.__file__,
        'huang_cert_grid.py': hg.__file__,
        'huang_cert_np.py': nr.__file__,
    }
    dependency_hashes = {
        name: hashlib.sha256(open(path, 'rb').read()).hexdigest()
        for name, path in proof_sources.items()
    }
    runtime = dict(host=platform.node(), python=sys.version.split()[0],
                   executable=sys.executable, python_flint=flint.__version__,
                   flint=flint.__FLINT_VERSION__, precision_bits=50,
                   workers=nw)
    policy = dict(precision_bits=50, lbox_quant_digits=12,
                  zq_quant_digits=10)
    payload = dict(schema_version=2, source_sha256=source_hash,
                   dependency_sha256=dependency_hashes,
                   certificate_policy=policy, runtime=runtime, star=star,
                   fails=fails, results=out)
    out_path = os.path.join(RESULTS_DIR, f'huang_region1_{KTAG}.json')
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp_path, out_path)
    print(f"DONE {len(jobs)} bands, {fails} fails, {time.time()-t0:.0f}s",
          flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
