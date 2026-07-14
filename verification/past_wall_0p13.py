"""Past the product-bound wall: certify the fixed-point condition
pieces at the point margin kappa = 0.13 with the contraction-free
locate.

Continuation: alpha_*(0.13) = 0.70593253151, q_* = 0.5847018482.
Steps: (1) rectangle product bound on a slab-sized box - expected to
FAIL (the wall); (2) locate_fp_bisect on the same box; (3) pointwise
contraction ON the located interval - expected < 1; (4) G signs at
alpha_* -+ pad via the located boxes - the crossing.  Uniqueness at
kappa >= 0 is Nakajima's theorem (alpha_c(0.13) covers the branch).
"""
import sys
import time

if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)

import certify_km_slab as C

kappa = C.iv(C.dec('0.13'), C.dec('0.13'))
a_star = C.dec('0.70593253151')
pad = C.dec('0.0015')
q_ball = C.iv(C.dec('0.575'), C.dec('0.595'))

t0 = time.time()
# 1. the wall: rectangle product bound on the box
R_box = C.R_of_wide(q_ball, C.iv(a_star - pad, a_star + pad), kappa)
lo_R, hi_R = C.endpoints(R_box)
pp = C.Pprime_bound(lo_R - C.dec('0.005'), hi_R + C.dec('0.005'))
dr = C.dRdq_bound(q_ball, C.iv(a_star - pad, a_star + pad), kappa)
prod_box = C.endpoints(pp)[1] * C.endpoints(abs(dr))[1]
print(f"rectangle product bound on the box: {prod_box} "
      f"{'FAILS (>= 1) - the wall' if not (prod_box < 1) else '< 1'}")

# 2 + 4. locate at the two alpha endpoints and resolve the G signs
for tag, a in (('alpha_lb', a_star - pad), ('alpha_ub', a_star + pad)):
    loc = C.locate_fp_bisect(a, kappa, q_ball)
    if loc is None:
        print(f"{tag}: endpoint check FAILED")
        continue
    qb, pb = loc
    lo_q, hi_q = C.endpoints(qb)
    print(f"{tag}: located q in [{lo_q}, {hi_q}] "
          f"(width {float(str(hi_q - lo_q).split(' ')[0].lstrip('[')):.2e})")
    # 3. pointwise contraction on the located interval
    R_loc = C.R_of_wide(qb, C.arb(a), kappa)
    lo_Rl, hi_Rl = C.endpoints(R_loc)
    pp_l = C.Pprime_bound(lo_Rl - C.dec('0.002'), hi_Rl + C.dec('0.002'))
    dr_l = C.dRdq_bound(qb, C.arb(a), kappa)
    prod_loc = C.endpoints(pp_l)[1] * C.endpoints(abs(dr_l))[1]
    print(f"  contraction on located interval: {prod_loc} "
          f"{'< 1 PASS' if prod_loc < 1 else 'FAIL'}")
    G = C.G_mean_value(a, qb, pb, kappa)
    lo_G, hi_G = C.endpoints(G)
    want = '>0' if tag == 'alpha_lb' else '<0'
    ok = (lo_G > 0) if want == '>0' else (hi_G < 0)
    print(f"  G({tag}) = [{lo_G}, {hi_G}]  want {want}: "
          f"{'PASS' if ok else 'FAIL'}")

# 5. Nakajima's hypothesis, certified: alpha_ub < alpha_c(kappa) with
# alpha_c = 2/(pi E[(kappa-Z)_+^2]) and the closed form
# E[(kappa-Z)_+^2] = (1+kappa^2) Phi(kappa) + kappa phi(kappa).
from flint import arb
k = C.arb(C.dec('0.13'))
Phi = (-k / arb(2).sqrt()).erfc() / 2
phi = (-(k * k) / 2).exp() / (2 * arb.pi()).sqrt()
alpha_c = 2 / (arb.pi() * ((1 + k * k) * Phi + k * phi))
ok_nak = C.endpoints(C.arb(a_star + pad))[1] < C.endpoints(alpha_c)[0]
print(f"alpha_c(0.13) = {alpha_c}; alpha_ub < alpha_c: "
      f"{'PASS' if ok_nak else 'FAIL'} (uniqueness by Nakajima)")

print(f"({time.time()-t0:.0f}s)")
