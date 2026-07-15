"""The Ding--Sun lower-bound condition at general margin, in floats.

Ding--Sun (arXiv:1809.07742) prove the capacity lower bound at
kappa = 0 conditional on their Condition 1.2: the univariate function
S_*(lambda) = S_{0,alpha_*}(lambda) of their (48) is negative off
{0, 1}.  Their construction (eqs (2)-(3), (34)-(48)) is written for
general kappa_all throughout; this module evaluates
S_{kappa,alpha}(lambda) at any margin, which is the numerical content
a kappa-version of their theorem would condition on.

Pieces (their numbering):
  (34) H* = -psi(1-q) + E ln 2ch(sqrt(psi) Z)
  (35) P* =  psi(1-q)/2 + alpha E ln Psi(xi_Z),
       xi_z = (kappa - sqrt(q) z)/sqrt(1-q)
  (36) P_{H,D} on {-1,+1}^2, Gamma(H,D) its entropy, m = tanh(H)
  (37) D_H(A) = (A^2-1)(1-m^2)^2 / (sqrt(A^2(1-m^2)+m^2)+1)^2
  (38) ell(A) = E[ D_{sqrt(psi)Z}(A) ] / (1-q), increasing, ell(0) =
       lambda_min < 0, ell(inf) = 1;  A(lambda) = ell^{-1}(lambda)
  (40) H(lambda) = -2H* + E Gamma(sqrt(psi)Z, D(A(lambda)))
  (43) I_s(lambda) = alpha E_z E_{nu|nu>=xi_z}
         ln Psi( (xi_z - lambda nu)/sqrt(1-lambda^2)
                 - E(xi_z) s/(sqrt(psi) sqrt(1-q)) )
  (44) P(lambda) = -P* + psi(1-q)(1-lambda)/(2(1+lambda)) + I_0(lambda)
  (45) B(lambda,s) = s^2/2 - sqrt(psi)sqrt(1-q) s
         sqrt((1-lambda)/(1+lambda)) + I_s(lambda) - I_0(lambda),
       A(lambda) = inf_s B (convex in s)
  S(lambda) = H(lambda) + P(lambda) + A(lambda).
Endpoint identities used as checks: H(0)=0, H(1)=-H*, P(0)=0,
P(1)=-P*, A(0)=0, hence S(0)=0 and S(1)=-G*(alpha) (=0 at alpha_*).

Floats and quadrature only - diagnostics for the paper's lower-bound
section, not certificates.  The fixed point (q*, psi*) comes from
km.fixed_point via km.alpha_star at the certified alpha_*(kappa).
"""
import math
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erfcx

import km

SQ2 = math.sqrt(2.0)


def lnPsi(x):
    # ln of the upper Gaussian tail, stable for all x: the erfcx form
    # overflows for x << 0, where ln Psi(x) = ln(1 - Psi(-x)) with
    # Psi(-x) tiny
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    neg = x < -6.0
    xr = np.where(neg, 6.0, x)  # safe placeholder on the neg branch
    out[...] = np.log(erfcx(xr / SQ2) / 2.0) - xr * xr / 2.0
    if neg.any():
        xm = -x[neg]
        tail = np.log(erfcx(xm / SQ2) / 2.0) - xm * xm / 2.0
        out[neg] = np.log1p(-np.exp(tail))
    return out


def mills(x):
    # E(x) = phi(x)/Psi(x)
    x = np.asarray(x, dtype=float)
    return math.sqrt(2.0 / math.pi) / erfcx(x / SQ2)


def xlogx(p):
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    pos = p > 0
    out[pos] = p[pos] * np.log(p[pos])
    return out


class DS:
    def __init__(self, kappa, alpha, n_z=240, n_nu=100):
        self.kappa, self.alpha = float(kappa), float(alpha)
        sol = km.alpha_star(kappa, alpha0=alpha)  # solves at the root
        self.q = float(sol['q'])
        self.psi = float(sol['psi'])
        self.alpha = float(sol['alpha'])  # the solver's root
        # Gauss-Hermite nodes for E[f(Z)], Z standard normal
        x, w = np.polynomial.hermite_e.hermegauss(n_z)
        keep = w > 1e-14 * w.max()
        self.z, self.wz = x[keep], (w / w.sum())[keep]
        q, psi = self.q, self.psi
        self.sq = math.sqrt(q)
        self.s1q = math.sqrt(1.0 - q)
        self.spsi = math.sqrt(psi)
        self.xi = (self.kappa - self.sq * self.z) / self.s1q
        self.Exi = mills(self.xi)
        self.m = np.tanh(self.spsi * self.z)
        # conditional nu-nodes per z: Gauss-Legendre on [xi, xi+9],
        # weights phi(nu)/Psi(xi), renormalized (kills truncation)
        gl_x, gl_w = np.polynomial.legendre.leggauss(n_nu)
        # nodes must cover where the conditional mass sits: from xi
        # (or -8 if xi is far left) out to max(xi,0)+8
        lo = np.maximum(self.xi, -8.0)
        hi = np.maximum(self.xi, 0.0) + 8.0
        half = (hi - lo)[:, None] / 2.0
        nu = lo[:, None] + (gl_x[None, :] + 1.0) * half
        wnu = (gl_w[None, :] * half
               * np.exp(-nu * nu / 2.0) / math.sqrt(2 * math.pi))
        wnu /= wnu.sum(axis=1, keepdims=True)  # exact conditional law
        self.nu, self.wnu = nu, wnu
        # conditional-mean check: E[nu | nu >= xi] = E(xi)
        err = np.max(np.abs((wnu * nu).sum(axis=1) - self.Exi))
        assert err < 1e-8, ('conditional quadrature', err)
        # split z-grid on [0, L] for the integrands with a kink at
        # z = 0 (the |m| edge of D_H at small A): Gauss-Hermite
        # converges only at rate 1/n through the kink, biasing
        # lambda_min by ~1e-2; the integrands are even, so panelled
        # Gauss-Legendre on [0, 8.5] doubled is exact business
        panels = [(0.0, 0.4), (0.4, 1.2), (1.2, 3.0), (3.0, 8.5)]
        gx, gw = np.polynomial.legendre.leggauss(100)
        zs, ws = [], []
        for a, b in panels:
            half = (b - a) / 2.0
            z = a + (gx + 1.0) * half
            zs.append(z)
            ws.append(gw * half * 2.0
                      * np.exp(-z * z / 2.0) / math.sqrt(2 * math.pi))
        self.z_split = np.concatenate(zs)
        self.wz_split = np.concatenate(ws)
        self.m_split = np.tanh(self.spsi * self.z_split)
        # first-moment pieces
        self.ln2ch = float(np.dot(self.wz,
                                  np.log(2.0 * np.cosh(self.spsi * self.z))))
        self.ElnPsi = float(np.dot(self.wz, lnPsi(self.xi)))
        self.Hstar = -psi * (1 - q) + self.ln2ch
        self.Pstar = psi * (1 - q) / 2.0 + self.alpha * self.ElnPsi
        self.Gstar = self.Hstar + self.Pstar

    # --- (36)-(38), on the split grid (kink at z = 0) ---------------
    def Gamma(self, D):
        m = self.m_split
        p = np.stack([(1 + m) ** 2 + D, 1 - m * m - D,
                      1 - m * m - D, (1 - m) ** 2 + D]) / 4.0
        return -xlogx(p).sum(axis=0)

    def D_of_A(self, A):
        m2 = self.m_split ** 2
        return ((A * A - 1.0) * (1.0 - m2) ** 2
                / (np.sqrt(A * A * (1.0 - m2) + m2) + 1.0) ** 2)

    def ell(self, A):
        return (float(np.dot(self.wz_split, self.D_of_A(A)))
                / (1.0 - self.q))

    def lambda_min(self):
        return self.ell(0.0)

    def A_of_lambda(self, lam):
        if abs(lam - 1.0) < 1e-12:
            return None  # A -> infinity
        f = lambda t: self.ell(math.exp(2.0 * math.atanh(t))) - lam
        return math.exp(2.0 * math.atanh(
            brentq(f, -1 + 1e-12, 1 - 1e-12, xtol=1e-14)))

    # --- (40) -------------------------------------------------------
    def H_of(self, lam):
        A = self.A_of_lambda(lam)
        D = (1.0 - self.m_split ** 2) if A is None else self.D_of_A(A)
        return (-2.0 * self.Hstar
                + float(np.dot(self.wz_split, self.Gamma(D))))

    # --- (43)-(45) ---------------------------------------------------
    def I_s(self, lam, s):
        c = math.sqrt(max(1.0 - lam * lam, 1e-300))
        arg = ((self.xi[:, None] - lam * self.nu) / c
               - (self.Exi[:, None] * s / (self.spsi * self.s1q)))
        inner = (self.wnu * lnPsi(arg)).sum(axis=1)
        return self.alpha * float(np.dot(self.wz, inner))

    def P_of(self, lam):
        return (-self.Pstar
                + self.psi * (1 - self.q) * (1 - lam) / (2 * (1 + lam))
                + self.I_s(lam, 0.0))

    def A_of(self, lam):
        i0 = self.I_s(lam, 0.0)
        drift = self.spsi * self.s1q * math.sqrt((1 - lam) / (1 + lam))

        def B(s):
            return s * s / 2.0 - drift * s + self.I_s(lam, s) - i0

        r = minimize_scalar(B, bounds=(-2.0, 12.0), method='bounded',
                            options={'xatol': 1e-9})
        return min(0.0, float(r.fun))

    def S(self, lam):
        return self.H_of(lam) + self.P_of(lam) + self.A_of(lam)

    def checks(self):
        out = {
            'G* at alpha_* (want ~0)': self.Gstar,
            'H(0) (want 0)': self.H_of(0.0),
            'H(1) + H* (want 0)': self.H_of(1.0) + self.Hstar,
            'P(0) (want 0)': self.P_of(0.0),
            'P(1) + P* (want 0)': self.P_of(1.0) + self.Pstar,
            'A(0) (want 0)': self.A_of(0.0),
            'lambda_min (want in (-1,0))': self.lambda_min(),
        }
        return out


def scan(kappa, alpha, n=61):
    ds = DS(kappa, alpha)
    lmin = ds.lambda_min()
    lams = sorted(set(
        list(np.linspace(lmin + 1e-3, 0.999, n))
        + list(np.linspace(-0.05, 0.05, 21))
        + list(np.linspace(0.9, 0.999, 15))))
    rows = [(l, ds.S(l)) for l in lams]
    return ds, rows


def main():
    kappa = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8330786
    ds, rows = scan(kappa, alpha)
    print('kappa=%g alpha=%.7f q=%.7f psi=%.7f' %
          (ds.kappa, ds.alpha, ds.q, ds.psi))
    for k, v in ds.checks().items():
        print('  %-28s %+.3e' % (k, v))
    worst = max((s for l, s in rows if 1e-3 < abs(l) and abs(l - 1) > 1e-3),
                default=None)
    print('  max S(lambda) off {0,1}:     %+.6f' % worst)
    for l, s in rows:
        print('%+.4f %+.8f' % (l, s))


if __name__ == '__main__':
    main()
