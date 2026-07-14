# The Ising perceptron at nonzero margin

Verification code, certificates, and paper for the margin extension of
the Ising perceptron capacity results: the storage threshold
alpha_*(kappa) of the binary-weight perceptron with constraints
<g_a, w>/sqrt(N) >= kappa.

The paper is in `paper/` (main.tex, main.pdf).  In brief: Huang's
general-margin capacity upper bound is conditional on four numerical
hypotheses per margin.  This repository certifies, in ball arithmetic
(python-flint/Arb), the fixed-point, AMP, and local-concavity
hypotheses on the strip kappa in [-0.65, 0.19] (1680 contiguous slabs,
two uniqueness lanes), locates alpha_*(kappa) to ~2e-6 at five margins,
certifies the Hessian of the first-moment functional at its degenerate
maximizer at six margins, and certifies the remaining hypothesis (the
global two-variable bound) at kappa = 0.05 --- so the capacity upper
bound at that margin is unconditional.  At kappa = 0 every pipeline
reproduces the certified zero-margin values.

## Layout

- `paper/` --- the manuscript.
- `verification/` --- all code, exactly as run.  `core.py` is vendored
  from the zero-margin verification
  (github.com/yspennstate/ising-perceptron-capacity) so the clone is
  self-contained.
- `verification/results/` --- the certificates: slab tables
  (`certified_intervals*.csv`), the two-variable manifests
  (`huang_sweep_0p05.json`, `huang_region1_0p05.json`,
  `huang_sweep2_0p05.json`, `huang_star_interior_0p05.json`), deep
  alpha_* intervals (`alpha_star_*.json`), worker evidence under
  `results/box/` (including two failed first attempts kept as
  evidence), and figures.
- `verification/lean_skeleton/` --- the Lean 4 skeleton: an extractor
  with an exact-rational mirror, kernel-decided generated files, and a
  mutation battery.
- `verification/AUDIT.md` --- the algorithm-to-code map: for every
  certificate family, the claim, the algorithm, the code location, and
  the soundness argument.

## Replay

Requirements: Python 3.11+ with `python-flint` (and `mpmath`, `numpy`,
`scipy`, `matplotlib` for the nonrigorous layers); Lean 4 (v4.31.0)
for the skeleton.  From `verification/`:

    python selfcheck.py
    python certify_km_slab.py slabs_pos.json out.json      # one lane batch
    python certify_nak.py slabs_pos2_main.json out2.json   # Nakajima lane
    python past_wall_0p13.py                               # 9 s, the wall
    python cert_hessian.py 0.05 0.781073068 0.781074776    # center Hessian
    python sweep_0p05.py 4                                 # Region II
    python region1_0p05.py 4                               # Region I
    python starint_0p05.py                                 # star interior
    python stage2_0p05.py 4                                # stage 2
    python assemble_2varfn_0p05.py                         # the 2var verdict

Enclosure digits vary with the FLINT build; the PASS/FAIL verdicts do
not.  The Lean skeleton regenerates and checks with

    cd lean_skeleton
    python extract_margin.py --results ../results --tag 0p05 --out generated
    lean generated/<file>.lean          # each file is self-contained
    python mutate_margin.py --lean <path-to-lean>

The float layers (`km.py`, `scan_kappa.py`, `huang2var.py`,
`scan2var.py`, `ml_numerics.py`) are diagnostics, not certificates,
and are labeled as such throughout.
