"""Huang's two-variable function S_*(lambda_1, lambda_2) at general kappa.

Transcribed from his eq:def-sS-star-bLam-nu with dH ~ N(0, psi0),
hH ~ N(0, q0), M = tanh(dH), N = F_{1-q0}(hH):

  S(Lam, s) = s^2 psi0/2 + ent(Lam)
              + alpha E log Psi( [kappa - (a2/q0) hH - (a1/psi0) N]/D + s N )

  a1 = E[dH Lam(dH)],  a2 = E[M Lam(dH)],  D = sqrt(1 - a2^2/q0),
  ent(Lam) = E H((1 + Lam(dH))/2),
  N(z) = F_{1-q0}(sqrt(q0) z) = M_mills((kappa - sqrt(q0) z)/sqrt(1-q0))
         / sqrt(1-q0),

  S_*(Lam) = inf_{s >= 0} S(Lam, s),
  S_*(l1, l2) = S_*(tanh(l1 x + l2 tanh(x))).

Condition 2varfn: S_*(l1, l2) <= 0 everywhere, equality at (1, 0).
The anchor S_*(1, 0) = 0 with optimal tilt s = sqrt(1 - q0) couples this
transcription to the fixed point (alpha, q0, psi0) from km.py: both must
be right together for the anchor to vanish.

Nonrigorous (float64, vectorized); the certified version of this scan at
kappa = 0 is Regions I/II of ising-perceptron-capacity.
"""

import numpy as np
from scipy.special import erfcx, log_ndtr


def mills(u):
    """phi(u)/Psi(u), stable for all u via erfcx."""
    return np.sqrt(2 / np.pi) / erfcx(u / np.sqrt(2))


def logPsi(u):
    return log_ndtr(-u)


class STwoVar:
    def __init__(self, alpha, q0, psi0, kappa, NZ=4001, ZL=9.0):
        self.alpha = float(alpha)
        self.q0 = float(q0)
        self.psi0 = float(psi0)
        self.kappa = float(kappa)
        z = np.linspace(-ZL, ZL, NZ)
        w = np.ones(NZ)
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        w *= (z[1] - z[0]) / 3 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        self.z, self.w = z, w
        self.X = np.sqrt(self.psi0) * z            # dH
        self.M = np.tanh(self.X)
        self.hH = np.sqrt(self.q0) * z             # hat-H
        s1q = np.sqrt(1 - self.q0)
        self.N = mills((self.kappa - self.hH) / s1q) / s1q
        self.sqq = np.sqrt(self.q0)

    def profile(self, l1, l2):
        """(ent, a1, a2) for Lam = tanh(l1 x + l2 tanh(x))."""
        u = l1 * self.X + l2 * self.M
        Lam = np.tanh(u)
        au = np.abs(u)
        ent = np.sum(self.w * (au + np.log1p(np.exp(-2 * au)) - u * Lam))
        a1 = np.sum(self.w * self.X * Lam)
        a2 = np.sum(self.w * self.M * Lam)
        return ent, a1, a2

    def T(self, a1, a2, s):
        """E log Psi(V) for the tilted constraint term."""
        d2 = 1 - a2 * a2 / self.q0
        if d2 <= 0:
            return np.nan
        D = np.sqrt(d2)
        V = ((self.kappa - (a2 / self.q0) * self.hH
              - (a1 / self.psi0) * self.N) / D + s * self.N)
        return np.sum(self.w * logPsi(V))

    def S_of_a(self, ent, a1, a2, s_hint=None):
        """inf_{s>=0} [s^2 psi0/2 + ent + alpha T(a1,a2,s)].
        Coarse grid + golden refine; expands the bracket if the minimum
        sits at the edge.  Returns (S, s_min).

        The infimum can genuinely be -infinity: alpha E[N^2] = psi0 at
        the fixed point, so the s^2 coefficient cancels exactly and a
        subleading linear term can run away (Huang handles these
        directions with his capped S^{s_max}; they satisfy S <= 0
        trivially).  A bracket that keeps expanding is recorded as
        -inf, not NaN - NaN is reserved for a moment point outside the
        body (d2 <= 0, impossible for genuine profiles)."""
        f = lambda s: s * s * self.psi0 / 2 + self.alpha * self.T(a1, a2, s)
        hi = 4.0
        while True:
            ss = np.linspace(0.0, hi, 64)
            vals = np.array([f(s) for s in ss])
            if np.all(np.isnan(vals)):
                return np.nan, np.nan
            k = int(np.nanargmin(vals))
            if k < len(ss) - 2:
                break
            hi *= 2
            if hi > 260:
                return -np.inf, np.inf
        lo_b, hi_b = ss[max(0, k - 1)], ss[k + 1]
        g = (np.sqrt(5) - 1) / 2
        a, b = lo_b, hi_b
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(48):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        sm = (a + b) / 2
        return ent + f(sm), sm

    def S(self, l1, l2):
        """S_*(l1, l2) and the optimal tilt."""
        ent, a1, a2 = self.profile(l1, l2)
        return self.S_of_a(ent, a1, a2)

    def anchor(self):
        """S_*(1,0) (should be ~0) and its optimal s (should be
        ~sqrt(1-q0))."""
        val, sm = self.S(1.0, 0.0)
        return val, sm, np.sqrt(1 - self.q0)

    def hessian_center(self, h=1e-3):
        """FD Hessian of S_*(l1,l2) at (1,0); must be negative definite."""
        f = lambda x, y: self.S(x, y)[0]
        f0 = f(1, 0)
        fxx = (f(1 + h, 0) - 2 * f0 + f(1 - h, 0)) / h**2
        fyy = (f(1, h) - 2 * f0 + f(1, -h)) / h**2
        fxy = (f(1 + h, h) - f(1 + h, -h) - f(1 - h, h)
               + f(1 - h, -h)) / (4 * h**2)
        return np.array([[fxx, fxy], [fxy, fyy]])

    def clearance(self, n_theta=720):
        """Moment-body clearance of the distinguished point
        a* = (E[X tanh X], q0): min over directions of h_K(u) - u.a*,
        h_K(u) = E|u1 X + u2 tanh X|.  Zero means a* has reached the
        boundary of K and the interior dual localization breaks."""
        a1s = np.sum(self.w * self.X * self.M)
        a2s = self.q0
        best = np.inf
        arg = None
        for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
            u1, u2 = np.cos(th), np.sin(th)
            hK = np.sum(self.w * np.abs(u1 * self.X + u2 * self.M))
            c = hK - (u1 * a1s + u2 * a2s)
            if c < best:
                best, arg = c, th
        return best, arg

    def grid(self, L=8.0, n=41):
        """Coarse global scan of S_* over [-L,L]^2.  Returns the grid and
        the top off-center values (candidate second maximizers)."""
        ls = np.linspace(-L, L, n)
        Sv = np.full((n, n), np.nan)
        for i, l1 in enumerate(ls):
            for j, l2 in enumerate(ls):
                Sv[i, j] = self.S(l1, l2)[0]
        return ls, Sv

    def rays(self, ts=(10, 20, 40, 80), n_theta=16):
        """S_* far out along rays t(cos th, sin th): the compactification
        says S_* must stay negative as |lambda| grows (profile approaches
        a boundary point of the moment body, entropy -> 0)."""
        out = []
        for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
            row = []
            for t in ts:
                row.append(self.S(t * np.cos(th), t * np.sin(th))[0])
            out.append((th, row))
        return out
