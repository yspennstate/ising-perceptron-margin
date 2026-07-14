"""Stage-2 sweep at general kappa: certify S_* < 0 on (stage-1 EXCL box)
minus the Region-I star.

Port of the capacity verification's huang_sweep2.py (kappa = 0) to the
general-kappa cert modules.  The stage-1 sweep (huang_cert_sweep.py)
certifies S_* < 0 on K minus its per-kappa EXCL box; Region I
(huang_cert_region1.py) gives S_* <= 0 on the star
{a* + t v(th): 0 <= t <= T(th)}.  This driver closes the remainder: every
cell of EXCL (expanded by the stage-1 sliver rule) not certainly inside
the star is certified by the stage-1 dual-tangent bound, bisecting down
to MIN_SIDE near the star boundary.

General-kappa star containment: Region I's star is anchored at the TRUE
maximizer a* = (psi(1-q), q), known only as a parameter ball around the
stored floats (radius RB ~ 1e-4 at located margins, vs 1e-7 at kappa = 0).
The containment test therefore inflates both coordinates of the geometry:
  - radial: corner distances measured from the floats are within RB of the
    true distances, so the radius comparison carries SAFE_R = RB + 5e-7;
  - angular: seen from the true origin, a corner at float-distance d moves
    by at most asin(RB / (d - RB)), so the cell's angular interval is
    widened by that amount before taking the zone minimum (cells too close
    for the bound simply fail containment and fall to eval_cell, which
    certifies there: they lie outside the stage-1 failure ellipse).

Configure r1 and sw for the target kappa first (the launcher does), then
run:  python stage2_<ktag>.py [nworkers]
"""

import os
import sys
import time
import math
from multiprocessing import Pool

import huang_cert_np as nr
import huang_cert_sweep as sw
import block3bc_exact as exact
import huang_cert_region1 as r1

MIN_SIDE = 1e-3
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')

# set by configure_sweep2
EXCL_OLD = None
RB = None
SAFE_R = None
KTAG = None
FAILED_BANDS = None
MANIFEST_SHA = None


def configure_sweep2(manifest_path, supplement_path=None):
    """Derive the per-kappa geometry from the ALREADY CONFIGURED r1 and sw
    modules plus the Region-I run manifest: the expanded stage-1 exclusion
    box (stage-1 treats MIN_SIDE cells overlapping EXCL as covered here;
    those extend at most sw.MIN_SIDE beyond the box, so expand by
    2*sw.MIN_SIDE each side), the star-origin ball radius RB, and the
    CERTIFIED star reach.

    A certified Region-I band (t0, t1, th0, th1) covers every direction
    theta in its chunk on radii [t0, min(t1, T_of_angle(theta))] (its
    angular sub-cells walk at least that far), so if every band certified,
    the star reach is exactly T_of_angle -- no zone shrink.  A failed band
    caps the reach of its directions at its inner radius t0; the list of
    failed bands is read from the manifest and consulted per cell."""
    global EXCL_OLD, RB, SAFE_R, KTAG, FAILED_BANDS, MANIFEST_SHA
    global SUPP_ARCS, SUPP_T0, SUPP_T1, SUPP_SHA
    import hashlib
    import json
    e = sw.EXCL
    pad = 2 * float(sw.MIN_SIDE)
    EXCL_OLD = (e[0] - pad, e[1] + pad, e[2] - pad, e[3] + pad)
    RB = (r1._ball_rad(r1.A1B) ** 2 + r1._ball_rad(r1.A2B) ** 2) ** 0.5
    SAFE_R = RB + 5e-7
    KTAG = r1.KTAG
    SUPP_ARCS, SUPP_T0, SUPP_T1, SUPP_SHA = [], None, None, None
    if supplement_path is not None and os.path.exists(supplement_path):
        sraw = open(supplement_path, 'rb').read()
        SUPP_SHA = hashlib.sha256(sraw).hexdigest()
        supp = json.loads(sraw)
        if supp.get('kind') != 'huang_region1_supplement':
            raise ValueError('wrong supplement kind')
        SUPP_T0, SUPP_T1 = supp['star']['T_SHOULDER']
        # merge the certified chunks into maximal seam-exact arcs
        chunks = sorted(tuple(rec['band'][2:4])
                        for rec in supp['results'] if rec.get('ok'))
        for (a, b) in chunks:
            if SUPP_ARCS and abs(SUPP_ARCS[-1][1] - a) < 1e-12:
                SUPP_ARCS[-1] = (SUPP_ARCS[-1][0], b)
            else:
                SUPP_ARCS.append((a, b))
        print(f'supplement: {len(chunks)} chunks -> {len(SUPP_ARCS)} '
              f'arcs, radius [{SUPP_T0}, {SUPP_T1}]', flush=True)
    raw = open(manifest_path, 'rb').read()
    MANIFEST_SHA = hashlib.sha256(raw).hexdigest()
    man = json.loads(raw)
    star = man['star']
    for name, val in (('W_ANG', r1.W_ANG), ('WEDGE_HALF', r1.WEDGE_HALF),
                      ('CONE_MID', r1.CONE_MID), ('T_LONG', r1.T_LONG),
                      ('T_MID', r1.T_MID), ('T_CORE', r1.T_CORE),
                      ('A1S', r1.A1S), ('A2S', r1.A2S)):
        if abs(star[name] - val) > 1e-9:
            raise ValueError('manifest star geometry mismatch: %s %r != %r'
                             % (name, star[name], val))
    FAILED_BANDS = [rec['band'] for rec in man['results']
                    if not rec.get('ok')]
    if FAILED_BANDS:
        print('WARNING: %d failed Region-I bands cap the star reach'
              % len(FAILED_BANDS), flush=True)


def _zone_T(lo, hi):
    """Zone reach for one angle sub-interval (min over it, via the
    triangular angular distance's maximum)."""
    vals = [r1.omega_of(lo), r1.omega_of(hi)]
    for k in range(-4, 5):
        peak = r1.W_ANG + math.pi / 2 + k * math.pi
        if lo <= peak <= hi:
            vals.append(math.pi / 2)
    om_max = max(vals)
    if om_max <= r1.WEDGE_HALF - 5e-7:
        return r1.T_LONG
    if om_max <= r1.CONE_MID - 5e-7:
        return r1.T_MID
    return r1.T_CORE


def _T_star_min(lo, hi):
    """Certified minimum over the (already origin-widened) angle interval
    [lo, hi] of the POINTWISE union coverage: the zone reach, extended on
    the certified supplement arcs to the supplement's outer radius when
    the zone reach meets its inner radius (no radial gap).  The interval
    is split at every zone boundary and supplement arc edge, so the
    minimum of the piecewise-constant union is evaluated exactly; then
    failed-band caps apply."""
    cuts = {lo, hi}
    for k in range(-4, 5):
        for c in (r1.W_ANG + k * math.pi, ):
            for d in (r1.WEDGE_HALF, r1.CONE_MID):
                for s in (c - d, c + d):
                    if lo < s < hi:
                        cuts.add(s)
    for (a, b) in SUPP_ARCS:
        for s in (a, b):
            if lo < s < hi:
                cuts.add(s)
    pts = sorted(cuts)
    T = None
    for a, b in zip(pts[:-1], pts[1:]):
        Tz = _zone_T(a, b)
        Tu = Tz
        if SUPP_ARCS and Tz >= SUPP_T0:
            for (sa, sb) in SUPP_ARCS:
                if sa - 1e-12 <= a and b <= sb + 1e-12:
                    Tu = max(Tz, SUPP_T1)
                    break
        T = Tu if T is None else min(T, Tu)
    if T is None:
        T = _zone_T(lo, hi)
    for (bt0, bt1, bth0, bth1) in FAILED_BANDS:
        if bt0 >= T:
            continue
        # does [bth0, bth1] meet [lo, hi] modulo 2 pi?
        w = (bth1 - bth0) / 2 + (hi - lo) / 2
        gap = abs(((bth0 + bth1) / 2 - (lo + hi) / 2 + math.pi)
                  % (2 * math.pi) - math.pi)
        if gap <= w + 1e-12:
            T = min(T, bt0)
    return T


def in_star(a1lo, a1hi, a2lo, a2hi):
    """True only if the cell is certainly inside the CERTIFIED Region-I
    star around the TRUE a* (see module docstring for the two
    inflations)."""
    corners = [(a1lo, a2lo), (a1lo, a2hi), (a1hi, a2lo), (a1hi, a2hi)]
    dists = [math.hypot(c1 - r1.A1S, c2 - r1.A2S) for (c1, c2) in corners]
    rmax = max(dists) + RB
    if rmax < _T_star_min(0.0, 2 * math.pi + 0.1) - 5e-7:
        return True                       # inside the everywhere-radius
    # if the cell (inflated) contains a*, only the everywhere test is safe
    if (a1lo - SAFE_R <= r1.A1S <= a1hi + SAFE_R
            and a2lo - SAFE_R <= r1.A2S <= a2hi + SAFE_R):
        return False
    dmin = min(dists) - RB
    if dmin <= 2 * RB:
        return False                      # angular bound unusable
    dth = math.asin(min(1.0, RB / dmin))
    ths = [math.atan2(c2 - r1.A2S, c1 - r1.A1S) for (c1, c2) in corners]
    th0, th1 = min(ths), max(ths)
    if th1 - th0 > math.pi:               # wrapped; rotate branch
        ths = [t + 2 * math.pi if t < 0 else t for t in ths]
        th0, th1 = min(ths), max(ths)
    return rmax <= _T_star_min(th0 - dth, th1 + dth) - 5e-7


def cert_cell(job, depth=0):
    a1lo, a1hi, a2lo, a2hi = job
    import huang_cert_grid as hg
    if in_star(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None              # Region I covers it
    if hg.outside_K(a1lo, a1hi, a2lo, a2hi):
        return True, 0, None
    val = sw.eval_cell(a1lo, a1hi, a2lo, a2hi)
    if not (val is not None and val < 0):
        if (a1hi - a1lo) < 0.004 and (a2hi - a2lo) < 0.004:
            val2 = sw.eval_cell_mv(a1lo, a1hi, a2lo, a2hi)
            if val2 is not None:
                val = val2
    if val is not None and val < 0:
        return True, 1, None
    if (a1hi - a1lo) < MIN_SIDE and (a2hi - a2lo) < MIN_SIDE:
        return False, 1, (str(val), job)
    if (a1hi - a1lo) >= (a2hi - a2lo):
        m = 0.5 * (a1lo + a1hi)
        subs = [(a1lo, m, a2lo, a2hi), (m, a1hi, a2lo, a2hi)]
    else:
        m = 0.5 * (a2lo + a2hi)
        subs = [(a1lo, a1hi, a2lo, m), (a1lo, a1hi, m, a2hi)]
    ok_all, ncells, worst = True, 0, None
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


def _init():
    from core import set_prec
    set_prec(50)


def build_jobs():
    a1lo, a1hi, a2lo, a2hi = EXCL_OLD
    n1, n2 = 20, 12
    jobs = []
    for i in range(n1):
        for j in range(n2):
            jobs.append((a1lo + (a1hi - a1lo) * i / n1,
                         a1lo + (a1hi - a1lo) * (i + 1) / n1,
                         a2lo + (a2hi - a2lo) * j / n2,
                         a2lo + (a2hi - a2lo) * (j + 1) / n2))
    return jobs, n1, n2


def proof_source_paths():
    import core
    import huang_cert_grid as hg
    return {'huang_cert_sweep2.py': __file__,
            'huang_cert_sweep.py': sw.__file__,
            'huang_cert_region1.py': r1.__file__,
            'block3bc_exact.py': exact.__file__, 'core.py': core.__file__,
            'huang_cert_grid.py': hg.__file__,
            'huang_cert_np.py': nr.__file__}


def main(nw=None):
    if nw is None:
        nw = int(sys.argv[1]) if len(sys.argv) > 1 else max(
            1, os.cpu_count() - 2)
    jobs, n1, n2 = build_jobs()
    print(f"{len(jobs)} top cells over EXCL_OLD={EXCL_OLD}, {nw} workers, "
          f"MIN_SIDE={MIN_SIDE}, RB={RB:.2e}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    fails = 0
    total = 0
    records = []
    schedule = [[repr(x) for x in job] for job in jobs]
    job_index = {job: i for i, job in enumerate(jobs)}
    log_path = os.path.join(RESULTS_DIR, 'huang_sweep2_%s.log' % KTAG)
    with Pool(nw, initializer=_init) as pool, \
            open(log_path, 'w', encoding='utf-8', newline='\n') as log:
        for k, res in enumerate(pool.imap_unordered(worker, jobs)):
            job, ok, nc, worst, dt_ms = res
            total += nc
            records.append(dict(index=job_index[job],
                                cell=[repr(x) for x in job],
                                ok=bool(ok), leaves=int(nc),
                                worst=None if worst is None else repr(worst),
                                runtime_milliseconds=int(dt_ms)))
            if not ok:
                fails += 1
                line = f"FAIL {job} worst={worst}"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if (k + 1) % 24 == 0:
                print(f"  {k+1}/{len(jobs)} top cells, {total} leaves, "
                      f"{fails} fails, {time.time()-t0:.0f}s", flush=True)
        log.write(f"total_leaves={total} fails={fails}\n")
    records.sort(key=lambda row: row['index'])
    sources = exact.source_hashes(proof_source_paths())
    payload = {
        'schema_version': 1, 'kind': 'huang_sweep2_manifest',
        'kappa_tag': KTAG,
        'source_sha256': sources, 'runtime': exact.runtime_record(50, nw),
        'policy': {'n1': n1, 'n2': n2,
                   'EXCL_OLD': [repr(x) for x in EXCL_OLD],
                   'MIN_SIDE': repr(MIN_SIDE), 'RB': repr(RB),
                   'SAFE_R': repr(SAFE_R),
                   'region1_manifest_sha256': MANIFEST_SHA,
                   'region1_supplement_sha256': SUPP_SHA,
                   'failed_region1_bands': FAILED_BANDS},
        'schedule': schedule,
        'schedule_sha256': exact.payload_sha256(schedule, omit=()),
        'records': records, 'total_leaves': total, 'failures': fails,
    }
    payload['artifact_sha256'] = exact.payload_sha256(
        payload, omit=('artifact_sha256',))
    exact.write_json_atomic(
        os.path.join(RESULTS_DIR, 'huang_sweep2_%s.json' % KTAG), payload)
    print(f"DONE: {total} leaves, {fails} fails, {time.time()-t0:.0f}s",
          flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    raise SystemExit('use a per-kappa launcher (e.g. stage2_0p05.py): '
                     'r1 and sw must be configured first')
