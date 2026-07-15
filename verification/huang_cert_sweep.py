"""Region II of Huang's Condition 1.3: the bounded (a1, a2) sweep.

Certifies  H(a1,a2) + G(a1,a2) <= 0  over the rectangle
    R = [-sqrt(psi), sqrt(psi)] x [-A2MAX, A2MAX],  A2MAX < sqrt(q),
minus a small box EXCL around the degenerate maximizer (a1*, a2*) =
(psi(1-q), q), which is handled by the frozen-tilt Hessian (Region I).

Per cell (a1_ball, a2_ball):
  - center (c1, c2); nonrigorously find the optimal dual lambda* = dual_of(c)
    and tilt s* = argmin_s [...] (huang_np). These only guide the choice.
  - rigorously certify (huanggrid):
        Phi(lambda*) - lambda1* a1_ball - lambda2* a2_ball
        + s*^2 psi/2 + alpha T(a1_ball, a2_ball, s*)   <  0.
    The first line upper-bounds H(a1,a2) on the whole cell by convex duality
    (valid for ANY fixed lambda*); the second upper-bounds G (valid for ANY
    fixed s*). If dual_of fails, fall back to H <= log 2.
  - on failure, bisect the cell (longer side) until MIN_SIDE.

Cells entirely inside EXCL are skipped (Region I). A cell overlapping EXCL is
still swept on its non-excluded... we instead require EXCL to be axis-aligned
and only skip fully-contained cells, bisecting boundary cells down so the
excluded set is covered exactly by MIN_SIDE-resolution cells inside EXCL.

General-kappa port: configure_sweep() first (see sweep_0p05.py).
Logs to results/huang_sweep_<ktag>.log.
"""

import math
import os
import sys
import time
from multiprocessing import Pool

# rectangle and exclusion (floats; the rigorous code re-derives balls)
import huang_cert_np as nr
import block3bc_exact as exact

# General-kappa parameters; set by configure_sweep() before use.
KTAG = None
A1MAX = None
A2MAX = None
A1S = None
A2S = None
EXCL = None

MIN_SIDE = 0.002
MAX_DEPTH = 7
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')


def configure_sweep(ktag, kappa, alpha, q, psi):
    """Configure the rigorous grid module (arb balls), the float companion,
    and the sweep geometry for a margin kappa.

    Rectangle bounds are RIGOROUS supersets of the moment body: for any
    profile |Lam| <= 1,  |a1| = |E[X Lam]| <= E|X| = sqrt(2 psi/pi)  and
    |a2| = |E[M Lam]| <= E|M| < sqrt(E M^2) = sqrt(q); both computed in
    ball arithmetic and padded upward.  The Region-I exclusion box keeps
    the kappa = 0 proof's offsets around the degenerate maximizer
    a* = (psi(1-q), q), which is the maximizer at every kappa."""
    global KTAG, A1MAX, A2MAX, A1S, A2S, EXCL
    import huang_cert_grid as hg
    from flint import arb
    from core import endpoints
    KTAG = ktag
    hg.configure(kappa, alpha, q, psi)
    nr.configure(float(kappa.mid()) if hasattr(kappa, 'mid') else float(kappa),
                 float(alpha.mid()), float(q.mid()), float(psi.mid()))
    _, a1max = endpoints((2 * hg.PSI / arb.pi()).sqrt())
    A1MAX = round(float(a1max) * 1.03, 3)
    _, a2max = endpoints(hg.SQ_Q)
    A2MAX = round(float(a2max) * 1.005, 3)
    A1S = float((hg.PSI * (1 - hg.Q)).mid())
    A2S = float(hg.Q.mid())
    # The +a2 offset was 0.0961 through the kappa = 0.05 and -0.05 runs;
    # at kappa = 0.0995 the sweep found a degenerate ridge just above
    # that top edge (a MIN_SIDE leaf at a2 - A2S in [0.0967, 0.0986]
    # returning [+/- 0.181], certifiable again two leaf-widths higher),
    # so the edge moved to +0.1030.  Enlarging this box is safe only
    # because stage-2 tiles its jobs from this tuple (EXCL_OLD = EXCL +
    # sliver pad) and certifies every tile: an EXCL that outgrows what
    # the star and supplement cover fails stage-2 loudly rather than
    # leaving a gap between the pieces.  (Star-interior is independent
    # of this box; it certifies the star itself sits in the moment
    # body.)
    EXCL = (A1S - 0.1733, A1S + 0.1767, A2S - 0.1139, A2S + 0.1030)


def dec_f(x, nd=6):
    """Float -> dec through FIXED-POINT formatting.  str(round(x, 6)) of a
    tiny value like 3.17e-05 is scientific notation, which core.dec
    rejects - that crashed two kappa = 0.05 sweep launches (the optimal
    tilt rounds to ~e-05 on slack-constraint cells)."""
    from core import dec
    return dec(('%.' + str(nd) + 'f') % x)


def exact_mid_radius(lo, hi):
    """Return an Arb center/radius whose interval contains [lo, hi].

    ``lo`` and ``hi`` are the decimal spellings used for the certified cell.
    Re-deriving both center and radius from those same exact rationals avoids
    pairing a rounded center with an uninflated float radius, which can miss a
    declared edge by a few ulps.
    """
    from core import dec

    blo = dec(str(lo))
    bhi = dec(str(hi))
    return (blo + bhi) / 2, (bhi - blo) / 2


def in_excl(a1lo, a1hi, a2lo, a2hi):
    """True if the cell is fully inside the exclusion box."""
    return (EXCL[0] <= a1lo and a1hi <= EXCL[1]
            and EXCL[2] <= a2lo and a2hi <= EXCL[3])


def overlaps_excl(a1lo, a1hi, a2lo, a2hi):
    return not (a1hi <= EXCL[0] or a1lo >= EXCL[1]
                or a2hi <= EXCL[2] or a2lo >= EXCL[3])


def eval_cell(a1lo, a1hi, a2lo, a2hi):
    """Rigorous upper bound on H+G over the cell. Returns arb ball or None.

    H bounded by the convex-duality tangent at the cell's nonrigorous dual
    (linear in a -> already tight over the cell). T bounded by mean value in
    (a1,a2) about the cell center. s and lambda* are nonrigorous choices;
    the certificate holds for any fixed values."""
    from flint import arb
    from core import dec, iv
    import huang_cert_grid as hg
    PSI, ALPHA = hg.PSI, hg.ALPHA
    c1 = (a1lo + a1hi) / 2
    c2 = (a2lo + a2hi) / 2
    lam = nr.dual_of(c1, c2)
    if lam is None:
        # cell near/over the achievable boundary, or in the strongly
        # stretched dual directions where Newton needs a warm start: pull
        # the center inward toward the deep-interior degenerate point, take
        # that dual, and try to CONTINUE it back out to the center.  H(a) <=
        # Phi(lam) - lam.a holds for ANY fixed lam, so every choice below
        # stays a valid upper bound on the achievable part of the cell.
        for frac in (0.02, 0.04, 0.08, 0.15, 0.3, 0.5):
            p1 = c1 + frac * (A1S - c1)
            p2 = c2 + frac * (A2S - c2)
            lam_p = nr.dual_of(p1, p2)
            if lam_p is None:
                continue
            lam_c = nr.dual_of(c1, c2, l0=lam_p)   # continuation
            lam = lam_c if lam_c is not None else lam_p
            break
    s_star = nr.G(c1, c2)[0]
    if s_star != s_star:            # nan
        s_star = 0.0
    # These must be derived from the certified endpoints together.  The old
    # round(c, 8) + raw-float-radius pairing missed one first-cell edge by
    # 3.33e-9, making the mean-value enclosure formally incomplete.
    cc1, rr1 = exact_mid_radius(a1lo, a1hi)
    cc2, rr2 = exact_mid_radius(a2lo, a2hi)
    s = dec_f(float(s_star))
    T = hg.T_meanvalue(cc1, cc2, rr1, rr2, s)
    if T is None:
        return None
    a1b = iv(str(a1lo), str(a1hi))
    a2b = iv(str(a2lo), str(a2hi))
    if lam is not None and abs(lam[0]) <= 80 and abs(lam[1]) <= 80:
        b1 = dec_f(lam[0])
        b2 = dec_f(lam[1])
        Phi = hg.Phi_of(b1, b2)
        Hub = Phi - b1 * a1b - b2 * a2b
    else:
        # No usable dual: the cell is near (or beyond) the boundary of the
        # moment body, where the true dual explodes along the outward normal
        # u.  H(a) <= Phi(kappa u) - kappa u.a for ANY fixed kappa, and
        # Phi(kappa u) = kappa h(u) + E log(1+e^{-2 kappa|u.f|}), so for
        # supported (or infeasible) a this drops to ~0 (or -infinity) as
        # kappa grows, at the price of kappa * (cell radius) wrapping.  Try
        # a ladder of kappa and keep the best; fall back to [0, log 2].
        Hub = arb(0).union(hg.LOG2)
        u = None
        if lam is not None:
            n = math.hypot(lam[0], lam[1])
            if n > 0:
                u = (lam[0] / n, lam[1] / n)
        if u is None:
            u = _support_dir(c1, c2)
        if u is not None:
            from core import endpoints as _ep
            for kap in (10.0, 20.0, 40.0, 80.0, 160.0):
                b1 = dec_f(kap * u[0])
                b2 = dec_f(kap * u[1])
                cand = hg.Phi_of(b1, b2) - b1 * a1b - b2 * a2b
                _, chi = _ep(cand)
                _, hhi = _ep(Hub)
                if chi < hhi:
                    Hub = cand
    return Hub + s * s * PSI / 2 + ALPHA * T


def _support_dir(c1, c2):
    """Numeric outward-normal guess at (c1, c2): the fan direction
    maximizing u.c - h(u) (any direction is valid for the dual bound).
    The fan spans a half circle; h(-u) = h(u), so both signs are tried."""
    import huang_cert_grid as hg
    try:
        fan = hg._get_hfan()
    except Exception:
        return None
    best = None
    for (u1, u2, h) in fan:
        f1 = float(u1.mid() if hasattr(u1, 'mid') else u1)
        f2 = float(u2.mid() if hasattr(u2, 'mid') else u2)
        hf = float(h.mid() if hasattr(h, 'mid') else h)
        for sg in (1.0, -1.0):
            v = sg * (f1 * c1 + f2 * c2) - hf
            if best is None or v > best[0]:
                best = (v, sg * f1, sg * f2)
    if best is None:
        return None
    return (best[1], best[2])


def eval_cell_mv(a1lo, a1hi, a2lo, a2hi):
    """Mean-value form of eval_cell for small cells near the maximizer,
    where the linear variations of the dual tangent and the constraint term
    nearly cancel (the interval sum in eval_cell adds their widths and
    loses the margin).  Bounds

        total(a) <= total(c) + |d total/da1| r1 + |d total/da2| r2,
        d total/da_i = -lam_i + alpha dT/da_i(cell, s),

    with the gradient enclosed over the cell; near the maximizer
    alpha dT/da ~ lam(a), so the enclosure is ~|grad S_*| r, not |lam| r.
    Returns an arb upper-bound ball or None."""
    # Guarded at general kappa against the T_derivs large-V cancellation:
    # E'(V) and E''(V) are assembled from E(V) - V, which on wide far-tail
    # z-cells subtracts two O(20) balls and explodes the derivative
    # enclosure (measured 2026-07-14: dT1 ball +-564 against a true 2.03
    # at tilt s = 1.376).  The kappa = 0 proof only ever ran this path at
    # s near s0 ~ 0.66, where V stays moderate -- so run it ONLY near the
    # maximizer with a moderate numeric tilt (stage-2's rescue regime; a
    # None here is never unsound, the caller just bisects further).
    cd = ((0.5 * (a1lo + a1hi) - A1S) ** 2
          + (0.5 * (a2lo + a2hi) - A2S) ** 2) ** 0.5
    if cd > 0.06:
        return None
    _sp = nr.G(0.5 * (a1lo + a1hi), 0.5 * (a2lo + a2hi))[0]
    if not (_sp == _sp and abs(_sp) <= 1.0):
        return None
    from flint import arb
    from core import dec, endpoints
    import huang_cert_grid as hg
    PSI, ALPHA = hg.PSI, hg.ALPHA
    c1 = (a1lo + a1hi) / 2
    c2 = (a2lo + a2hi) / 2
    lam = nr.dual_of(c1, c2)
    if lam is None:
        return None
    s_star = nr.G(c1, c2)[0]
    if s_star != s_star:
        return None
    b1 = dec_f(lam[0], 8)
    b2 = dec_f(lam[1], 8)
    s = dec_f(float(s_star), 8)
    cc1, rr1 = exact_mid_radius(a1lo, a1hi)
    cc2, rr2 = exact_mid_radius(a2lo, a2hi)
    # thin center value
    Tc = hg.T_of(cc1, cc2, s)
    if Tc is None:
        return None
    center = (hg.Phi_of(b1, b2) - b1 * cc1 - b2 * cc2
              + s * s * PSI / 2 + ALPHA * Tc)
    # gradient of the majorant over the cell (s fixed)
    a1b = dec(str(a1lo)).union(dec(str(a1hi)))
    a2b = dec(str(a2lo)).union(dec(str(a2hi)))
    gz = hg.get_zt_grid()
    Td = hg.T_derivs(a1b, a2b, gz, s)
    if Td is None:
        return None
    g1 = -b1 + ALPHA * Td['dT1']
    g2 = -b2 + ALPHA * Td['dT2']

    def absup(x):
        # The upper endpoint of |x| is already an outward-rounded Arb bound;
        # do not serialize through float/round here.
        _, hi = endpoints(abs(x))
        return hi

    slack = absup(g1) * rr1 + absup(g2) * rr2
    return center + slack


def cert_cell(job, depth=0):
    """Certify a cell < 0, bisecting on failure. Returns (ok, ncells, worst)."""
    import huang_cert_grid as hg
    a1lo, a1hi, a2lo, a2hi = job
    if in_excl(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None            # Region I handles it
    if hg.outside_K(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None            # non-achievable: no profile here
    val = eval_cell(a1lo, a1hi, a2lo, a2hi)
    if not (val is not None and val < 0):
        # small cells near the maximizer: retry in mean-value form
        if (a1hi - a1lo) < 0.004 and (a2hi - a2lo) < 0.004:
            val2 = eval_cell_mv(a1lo, a1hi, a2lo, a2hi)
            if val2 is not None:
                val = val2
    if val is not None and val < 0:
        from core import endpoints
        _, hi = endpoints(val)
        return True, 1, float(hi.mid()) if hasattr(hi, 'mid') else float(hi)
    if (a1hi - a1lo) < MIN_SIDE and (a2hi - a2lo) < MIN_SIDE:
        # cannot refine further; if it overlaps EXCL treat as Region I, else fail
        if overlaps_excl(a1lo, a1hi, a2lo, a2hi):
            return True, 0, None
        return False, 1, (str(val), job)
    # bisect the longer side
    if (a1hi - a1lo) >= (a2hi - a2lo):
        m = (a1lo + a1hi) / 2
        subs = [(a1lo, m, a2lo, a2hi), (m, a1hi, a2lo, a2hi)]
    else:
        m = (a2lo + a2hi) / 2
        subs = [(a1lo, a1hi, a2lo, m), (a1lo, a1hi, m, a2hi)]
    ok_all = True
    ncells = 0
    worst = None
    for s in subs:
        ok, nc, w = cert_cell(s, depth + 1)
        ok_all = ok_all and ok
        ncells += nc
        if not ok and worst is None:
            worst = w
    return ok_all, ncells, worst


def worker(job):
    from core import set_prec
    set_prec(50)
    t0 = time.time_ns()
    ok, nc, worst = cert_cell(job)
    return (job, ok, nc, worst,
            (time.time_ns() - t0) // 1_000_000)


def build_jobs(coarse=48):
    """Deterministic declared top-cell schedule.

    Edges are snapped to 9 decimals (shared between neighbours, so the
    cover is exactly contiguous) with the outer edges forced to the
    rectangle bounds.  Raw float arithmetic can land an interior edge at
    -2e-16, whose repr is scientific notation, which core.dec rejects -
    that crashed the first kappa = 0.05 run minutes after launch."""
    jobs = []
    da1 = 2 * A1MAX / coarse
    na2 = max(1, int(2 * A2MAX / da1))
    da2 = 2 * A2MAX / na2
    e1 = [-A1MAX] + [round(-A1MAX + da1 * i, 9)
                     for i in range(1, coarse)] + [A1MAX]
    e2 = [-A2MAX] + [round(-A2MAX + da2 * j, 9)
                     for j in range(1, na2)] + [A2MAX]
    for i in range(coarse):
        for j in range(na2):
            jobs.append((e1[i], e1[i + 1], e2[j], e2[j + 1]))
    return jobs, da1, da2


def proof_source_paths():
    import core
    import huang_cert_grid as hg
    return {'huang_cert_sweep.py': __file__,
            'block3bc_exact.py': exact.__file__, 'core.py': core.__file__,
            'huang_cert_grid.py': hg.__file__,
            'huang_cert_np.py': nr.__file__}


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 2)
    coarse = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    jobs, da1, da2 = build_jobs(coarse)
    print(f"{len(jobs)} top cells, {nw} workers, "
          f"cell {da1:.3f}x{da2:.3f}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    fails = 0
    total_cells = 0
    records = []
    schedule = [[repr(x) for x in job] for job in jobs]
    job_index = {job: i for i, job in enumerate(jobs)}
    with Pool(nw, initializer=_init) as pool, \
            open(os.path.join(RESULTS_DIR, f'huang_sweep_{KTAG}.log'), 'w',
                 encoding='utf-8', newline='\n') as log:
        for k, res in enumerate(pool.imap_unordered(worker, jobs)):
            job, ok, nc, worst, dt_ms = res
            total_cells += nc
            records.append(dict(index=job_index[job],
                                cell=[repr(x) for x in job],
                                ok=bool(ok), leaves=int(nc),
                                worst=None if worst is None else repr(worst),
                                runtime_milliseconds=int(dt_ms)))
            if not ok:
                fails += 1
                line = f"FAIL {job} worst={worst}"
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
            if (k + 1) % 40 == 0:
                print(f"  {k+1}/{len(jobs)} top cells, {total_cells} leaves, "
                      f"{fails} fails, {time.time()-t0:.0f}s", flush=True)
        log.write(f"total_leaves={total_cells} fails={fails}\n")
    records.sort(key=lambda row: row['index'])
    sources = exact.source_hashes(proof_source_paths())
    import huang_cert_grid as hg
    payload = {
        'schema_version': 1, 'kind': 'huang_sweep_manifest', 'kappa': KTAG,
        'source_sha256': sources, 'runtime': exact.runtime_record(50, nw),
        'policy': {'coarse': coarse, 'A1MAX': repr(A1MAX),
                   'A2MAX': repr(A2MAX), 'MIN_SIDE': repr(MIN_SIDE),
                   'MAX_DEPTH': MAX_DEPTH, 'GRID_N': hg.GRID_N,
                   # the exclusion box this sweep skipped (leaves inside
                   # it are invisible in the records), so the seam with
                   # stage-2's EXCL_OLD is checkable from manifests
                   # alone rather than through the source hashes
                   'EXCL': [repr(x) for x in EXCL]},
        'schedule': schedule,
        'schedule_sha256': exact.payload_sha256(schedule, omit=()),
        'records': records, 'total_leaves': total_cells,
        'failures': fails,
    }
    payload['artifact_sha256'] = exact.payload_sha256(
        payload, omit=('artifact_sha256',))
    exact.write_json_atomic(
        os.path.join(RESULTS_DIR, f'huang_sweep_{KTAG}.json'), payload)
    print(f"DONE: {len(jobs)} top cells -> {total_cells} leaves, "
          f"{fails} fails, {time.time()-t0:.0f}s", flush=True)
    if fails:
        raise SystemExit(1)


def _init():
    from core import set_prec
    set_prec(50)


if __name__ == '__main__':
    main()
