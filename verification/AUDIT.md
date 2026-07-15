# Algorithm-to-code audit

A line-by-line read of every proof-bearing module in this repository
(2026-07-14), recording, for each certificate family, the mathematical
claim, the algorithm that certifies it, where each step lives in the
code, and the checks that tie them together.  Soundness in this
codebase always has the same shape: every quantity is an arb ball that
provably contains the true value, parameter uncertainty enters as input
balls covering the whole certified rectangle, and a PASS verdict is a
ball comparison that is true only when certain.  What follows is the
map from the mathematics to the lines.

## 0. Foundations (core.py, from the capacity verification)

`core.py` supplies exact decimal constants (`dec`, integer-ratio
construction, no float rounding), certified special functions
(`phi`, `Psi` as the upper tail, `mills`), the adaptive complex
integrator with an imaginary-part containment assertion
(`integrate`), exact Gaussian tail masses (`gauss_tail_mass`,
`z1_tail`, `z2_tail`), and the ball-arithmetic trap handlers
(`pos_part`, `sq_nonneg`, `sqrt_nonneg`, `min_one`) whose docstrings
state the trap each one closes.  The Ding-Sun rectangle constants
defined there are zero-margin values; the margin code touches them in
exactly two places, both intentional: the kappa = 0 branch of
`cert_hessian.main` (the rectangle IS the tight box at zero margin)
and the kappa = 0 regression `test_port_k0.py`.

Four numerical laws, learned from measured failures and now built into
every evaluator (NOTES.md sections 5-6 record the failures):

1. Adaptive complex quadrature cannot resolve enclosure width driven
   by parameter balls; wide-parameter integrals go through real-cell
   Riemann sums (`riemann_E`) with exact per-cell Gaussian masses.
2. Mills-family factors NaN under direct wide-ball evaluation (erfc
   spans zero in the steep region); they are enclosed by two thin
   monotone endpoint evaluations (`mills_ball`, `logPsi_ball`,
   monotone-in-psi expectations).
3. Sign-critical integrals use the mean-value cell rule
   (`riemann_mv`): w(mid) * mass + w'(cell) * (signed first-moment
   correction), O(h^2) against the plain hull's O(h).  The rule's
   soundness is the pointwise identity w(z) - w(mid) = w'(xi)(z - mid)
   with w' enclosed over the cell.
4. Near-cancelling expectations are evaluated as one correlated
   integrand, never as separate sums whose parameter widths add
   (the lambda identity, the w4/w22 integrands, and at the
   two-variable level the anchored-Bregman evaluator below).

## 1. Fixed-point condition, contraction lane

Claim (Huang's Condition km-well-defd, per kappa slab): the map
q -> P(R(q, alpha)) has a unique fixed point q_* in the slab's q
interval for every alpha in the slab's alpha interval, and
alpha_*(kappa), the root of G_*, lies in the alpha interval.

Algorithm and code (certify_km_slab.py):

- Contraction: sup_q (P o R_alpha)'(q) <= sup_psi P'(psi) *
  sup_q |dR/dq| < 1 on the slab rectangle.  `Pprime_bound` encloses
  P' by the two monotone pieces at their extreme psi (real-cell
  Riemann; the 1/cosh^4 integrand has complex poles that trip
  adaptive probes, hence rule 1).  `dRdq_bound` uses the collapsed
  one-term identity dR/dq = alpha E[M'^2 + 2 M^2 M'] / (1-q)^2
  (margin_note.tex Lemma 2, proved by Stein's lemma; checked against
  finite differences to 1e-15), with M' = M(M-u) intersected with its
  analytic range (0,1) to kill dependency blowup, and explicit
  quadratic-in-|z| tail bounds against Gaussian moments.  The bound is
  claimed per slab only; over a batch hull the kappa width poisons it
  (`verify_global` records the hull product as a diagnostic, not a
  claim).
- Existence: `verify_slab` checks the q-interval endpoints map
  strictly inward under P o R on each of eight alpha sub-intervals
  (intermediate value theorem; uniqueness from the contraction).
- Crossing: at the thin alpha endpoints `locate_fp` Picard-shrinks a
  fixed-point box (invariant: the image of any set containing the
  fixed point contains it, so intersecting preserves containment),
  then `G_mean_value` resolves the sign of G with the stationarity
  cancellations kept explicit: G(midpoint) plus
  (q - P(psi))/2 and (psi - R)/2 coefficients times the box widths,
  the sequential mean-value split documented at the definition (the
  psi coefficient ranges over the full q box).  G(alpha_lb) > 0 >
  G(alpha_ub) plus the strict decrease of G_* (envelope identity,
  margin_note.tex Lemma 1) pins alpha_*(kappa) in the interval.
- Huang's amp-works product (`cond_amp_works_cert`) and
  local-concavity at his witness z = -0.6693
  (`cond_local_concavity_cert`), the latter through the correlated
  identity lambda(z) = z + alpha E[b t^2 / (A(A + b t))] whose
  monotonicity in t (sign of b), regularity on t in [0,1], and tail
  bound |g| <= |b|/(Am) are all sympy-verified; the separated form's
  failure is kept as evidence (results/box/failed_neg5_lambda_width/).

Evidence: 1500 slabs on [-0.65, 0.10], zero failures, contiguity
gated by `collect_slabs.py` (consecutive slabs must share their kappa
boundary).

## 2. Fixed-point condition, Nakajima lane

Claim: the same condition on [0.10, 0.19], past the rectangle product
bound's wall, with uniqueness supplied by Nakajima's saddle theorem
(arXiv 2512.23195) under a certified premise.

Algorithm and code (`verify_slab_nak` in certify_km_slab.py, runner
certify_nak.py):

- Premise: alpha_ub < alpha_c(kappa) over the kappa ball, with
  alpha_c = 2/(pi E[(kappa - Z)_+^2]) evaluated through the closed
  form (1 + kappa^2) Phi(kappa) + kappa phi(kappa) (`alpha_c_ball`;
  the closed form is sympy-verified against the defining integral).
- Existence: the same inward sweep (already contraction-free).
- Location: `locate_fp_bisect` - certified sign bisection of the
  displacement P(R(q)) - q at thin q points, pure intermediate-value
  invariant (d > 0 to the left of the fixed point, < 0 to the right;
  monotonicity of P o R from dR/dq > 0 and P' > 0), with quartile
  fallback at the kappa-width resolution floor.  No contraction input.
- The located-interval contraction (the local form Huang's
  perturbation argument consumes) and the G crossing run per kappa
  sub-ball (n_k = 8), because the kappa width floors the
  displacement's sign resolution; each check quantifies over its
  sub-ball and the union covers the slab.

Evidence: 180 slabs on [0.10, 0.19], zero failures;
past_wall_0p13.py demonstrates the wall is a method artifact (box
product 1.32 at kappa = 0.13 while the located contraction reads 0.79).

## 3. Deep alpha_* location and the center Hessian

Claim: at six margins (0, +-0.05, 0.0995, -0.45, 0.13) the fixed-tilt
Hessian M of Huang's first-moment functional at the distinguished
point satisfies M11 < 0 and det M > 0 in ball arithmetic.

Algorithm and code:

- `locate_alpha.py` bisects alpha_* on the certified sign of G at
  located boxes (mean-value evaluators, measured floors: the G0
  quadrature at n = 4000 binds at width 1.4e-5; raising to n = 32000
  reaches 1.7e-6 with a clean final sign).
- `cert_hessian.py` Picard-locates (q0, psi0) at the deep interval
  and evaluates the general-kappa M-matrix of `hessian_kappa.py` in
  arb: prior-side integrals with the near-cancellations 1 - 4q + 3p4
  and q - p4 as single correlated integrands (`w4_of`, `w22_of`,
  psi-subdivided 32-fold with a slope-times-subwidth slack; the true
  psi-slope is below 0.55 for both, so the slope-1 slack dominates),
  constraint-side integrals by mean-value Riemann with elementary
  Mills-family derivatives (M'' = M'(M-u) + M(M'-1)), and explicit
  degree-2 tail coefficients.  The kappa = 0 reduction of the C
  constants is re-derived symbolically in hessian_kappa.py's header
  (which zero-margin identity survives and which does not), and the
  kappa = 0 run lands inside every certified interval of Huang's
  appendix - an independent re-verification of his Claim through the
  general-kappa route.

## 4. The two-variable condition: certified quadrature layer

huang_cert_grid.py ports the capacity verification's fixed-grid
machinery to general kappa; `test_port_k0.py` pins the port digit-
identical to the original at kappa = 0.  Structure:

- `_precompute_zt` / `_precompute_x`: box-independent per-node data.
  Kappa enters the zt side only through N(z) = E(u(z))/sqrt(1-q) with
  u = (kappa - sqrt(q) z)/sqrt(1-q), and N' = -(sqrt(q)/(1-q)) E'.
- `T_of` / `T_meanvalue`: E log Psi(V) with V affine in (Ht, N); the
  general-kappa additions are the additive constant kappa/D and the
  dV/da2 constant kappa a2/(q D^3) (both sympy-verified).  Mean-value
  form in the moment coordinates with derivative integrands and
  channel-structured tails (`_dT_tail`).
- `T_derivs` / `a_s_mixed`: the second-derivative kernels for the ray
  certificates.  E'(V) and E''(V) are built from the stable Mills gap
  t = E - V intersected with the classical bounds t in (0, 1/V) for
  V > 0, E' in (0, 1) (the truncated-normal variance defect), and
  E'' in (0, t) (hazard convexity plus E'' = E'(E + t) - E < t).
  Without the clamps, wide far-tail cells subtract two O(20) balls
  and the kernels explode (measured: hi spikes to 2e3 against a true
  ray second derivative of -6.6); with them, benign inputs are
  digit-identical and the kappa = 0 regression is unchanged.

## 5. Region II sweep (huang_cert_sweep.py)

Claim: S_*(a1, a2) < 0 on the bounded moment rectangle K minus a
per-kappa exclusion box EXCL around the maximizer.

Algorithm: per cell, an upper bound of sup S_* by convex duality -
H(a) <= Phi(b) - b.a for any dual point b (numeric tangent from
huang_cert_np.dual_of, soundness from the inequality, not the
numerics), plus the tilt term at any fixed s (any tilt upper-bounds
the infimum), evaluated by `T_meanvalue`; cells that fail bisect down
to MIN_SIDE = 0.002 and cells outside the moment body are certified
by `outside_K`.  `eval_cell_mv` (the mean-value rescue near the
maximizer) is guarded to the moderate-tilt regime near the star,
where the kappa = 0 proof always ran it.  Float-to-decimal seams use
fixed-point formatting (`dec_f`) after a measured crash on scientific
notation.  Manifest: results/huang_sweep_0p05.json (1296 top cells,
868 leaves, zero failures).

## 6. Region I star (huang_cert_region1.py)

Claim: S_* <= 0 on the star {a* + t v(theta) : 0 <= t <= T(theta)}
around the maximizer, by banded ray concavity from the base-point
identities.

The general-kappa port replaces the zero-margin proof's stored-decimal
origin (sound there because the parameter rectangle is ~1e-9 wide)
with the SYMBOLIC maximizer: a* = grad Phi(1,0) = (psi(1-q), q) and
base tilt s0 = sqrt(1-q) as parameter balls, so every certificate
covers the true point with no origin inflation.  The pinned identities
(value, gradient, tilt optimality at the true a*) are Huang's
base-point lemma at general kappa.  Key pieces:

- Dual localization (`edge_check`): multi-anchor sublevel argument.
  All values go through the anchored-Bregman evaluator
  `Phi_breg0_acb`, whose integrand collapses algebraically to
  log(cosh D + M sinh D) - M D with D = (l1 - 1) X + l2 M - exact
  because (l-b0) . grad h(b0, w) = M D pointwise - so the located
  parameter balls enter only at second order in |lambda - (1,0)|
  (measured 93-114x tighter than the uncorrelated difference at the
  corners that used to fail).  Two exact forms of cosh D + M sinh D
  are branched per node at |D| = 1: the cosh/sinh form is exact at
  D = 0 (the inner-band gaps ~1e-7 need it), the nonnegative
  exponential split e^D (1+M)/2 + e^{-D} (1-M)/2 has no e^{|D|}
  cancellation at box corners (`_c_stable`).  x-hulls are offsets
  from the symbolic a* (`xhull_of_band`), and the slack derivative is
  the correlated `grad_diff_acb`.  Fans sample five interior radii
  per angle (the dual image of a wedge sector stretches ~0.4 and
  anchors only at the two arcs leave the middle unanchored).
- Loewner majorant `B_Lambda` over the certified lambda box, feeding
  the quadratic credit `Binv_form`.
- Ray certificates: per angular cell and radial piece, quadT_box
  (T_derivs + a_s_mixed assembled along the 3D ray direction
  (v1, v2, sdot)) minus the credit must be negative; adaptive radial
  walk and angular bisection on failure.  The certificate covers rays
  from every origin in the symbolic ball, hence from the true a*,
  where phi(0) = phi'(0) = 0 hold exactly - so phi'' <= 0 on the band
  chain gives S_* <= 0 on the star.
- The band schedule (`bands`) tiles radii from T_LONG down to zero
  (the [0, T_IN] band runs through the same machinery; with the
  symbolic origin no special casing is needed).

Validation battery (validate_final_0p05.py): Breg0(1,0) = [0,0]
exactly, consistency of the correlated evaluators against the
uncorrelated path (overlap of enclosures of the same quantity), and
six gate bands including both wedge edges, a cone chunk, the smallest
chunked radius, the innermost band, and the origin band.

## 7. Stage-2 sweep (huang_cert_sweep2.py)

Claim: S_* < 0 on the stage-1 exclusion box (expanded by the sliver
rule) minus the certified star.

The star-containment test is manifest-driven: a certified Region-I
band (t0, t1, th0, th1) covers every direction in its chunk out to
min(t1, T_of_angle), so with a zero-failure manifest the certified
star reach is exactly T_of_angle; failed bands (none expected) cap
their directions at their inner radius.  A supplementary Region-I run
(supplement_0p05.run_supplement) certifies the four wedge-shoulder
arcs on [0.0070, 0.0090] --- the ray certificates are indifferent to
the zone policy that had excluded them, and leaves straddling the
shoulder need the extra reach.  The containment evaluates the
pointwise UNION of the two coverages: the angle interval is split at
every zone boundary and supplement arc edge, each sub-interval takes
the larger of its zone reach and (when it lies inside one certified
arc with no radial gap) the supplement radius, and the minimum over
sub-intervals is the certified reach.  The first union attempt
required the WHOLE interval inside one arc and missed a straddling
leaf; the split form is exact for the piecewise-constant coverage.  Because the star is anchored
at the true a*, known only as a ball around the stored floats, the
containment test inflates radially by the ball norm RB and widens the
angular interval by asin(RB / d_min); cells too close for the angular
bound simply fall through to `eval_cell` (sound: a False from
`in_star` never claims anything).  The manifest's SHA-256 is recorded
in the stage-2 manifest, binding the two artifacts.

## 8. Star interior (huang_cert_star_interior.py)

Claim: the star, origin ball included, lies in the interior of the
moment body (the ray proof differentiates the moment entropy there).

Two explicit feasible profile perturbations around Lambda* = tanh(X)
(the maximizer at every kappa - it is the profile realizing
a* = grad Phi(1,0)) map the l1 unit diamond to a parallelogram around
a*; a 2x2 arb computation bounds the parallelogram's inradius below
by det / longest edge and compares it with the required radius = star
radius times the direction-box norm plus twice the origin ball norm.
At kappa = 0.05: inradius 0.01481 against required 0.01209.

## 9. Serialization and replay

block3bc_exact.py (vendored from the capacity verification) supplies
the exact layer the manifests use: float-free canonical JSON
(`canonical_json_bytes` rejects floats outright), exact rational
records, outward arb packets (`arb_packet` / `packet_fraction_
endpoints` round-trip through integers), atomic no-clobber writes,
and SHA-256 source/payload binding.  Its Block-3 schedule constants
are capacity-repo legacy and unused here; only the serialization half
is load-bearing.  `selfcheck.py` gates documentation drift (every
file the replay instructions name must exist, parse, and compile);
the fresh-clone consumer test (NOTES.md section 6) replays
past_wall_0p13.py and a full slab from a cold clone.

## Audit findings

The read surfaced the following, all addressed:

1. cert_hessian.w4_of's comment states slope constants (20, 0.13)
   whose product is not below 1 as claimed; the actual psi-slope of
   the integrand is below 0.55 (elementary bound via
   |w'(t)|(1-t^2) <= 2.1 and E|z|/(2 sqrt(psi)) < 0.25), so the
   slope-1 slack in the code is sound with a factor-2 margin.  The
   comment is corrected in this commit.
2. The zero-margin constants in core.py are quarantined to the two
   intentional uses listed in section 0 (checked by grep over the
   repository).
3. block3bc_exact's scheduling half is dead code here (see section
   9); kept verbatim to preserve byte-identity with the capacity
   repository's audited copy.
4. Historical soundness repairs are documented at their sites and in
   NOTES.md: the separated-form lambda enclosure (replaced by the
   correlated identity), the T_derivs far-tail cancellation (replaced
   by the clamped stable Mills gap), the dual-evaluator corner
   infinities (replaced by the branched stable form), and the
   stored-decimal origin at general kappa (replaced by the symbolic
   maximizer).  The failed batches are kept under results/box/ as
   evidence.
