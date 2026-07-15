"""Supplementary Region-I shoulder bands at kappa = 0.05.

The main star's zone policy reaches T_MID = 0.008 just past the wedge
boundary, but stage-2 leaves straddling the boundary have corners out
to ~0.0085 where value-level certification cannot beat the parameter
fuzz.  The ray certificates themselves work fine in that sliver ---
only the zone gating kept the main run from walking it --- so this
driver widens the wedge policy to 0.19 and certifies four shoulder
arcs (both shoulders of both wedge axes) on the radius band
[0.0070, 0.0090], emitting a small manifest of the same schema.
Stage-2's containment then takes the union of the main and
supplementary coverage.

Usage: python supplement_0p05.py [nworkers]
"""
import json
import os
import sys
import time

import platform
platform.node = lambda: 'research-worker'   # manifests carry no machine names
if sys.platform.startswith('linux'):
    sys.executable = 'python3'   # manifests carry no machine paths (fork-safe on the worker hosts)

import certify_km_slab as C
import huang_cert_region1 as R1


def run_supplement(R1mod, out_name, t0b=0.0070, t1b=0.0090):
    """Shoulder bands for an ALREADY CONFIGURED region1 module; writes
    results/<out_name> and returns the fail count."""
    import math
    wedge_saved = R1mod.WEDGE_HALF
    R1mod.WEDGE_HALF = 0.19
    arcs = []
    for base in (R1mod.W_ANG, R1mod.W_ANG + math.pi):
        for sgn in (+1, -1):
            lo = base + sgn * 0.14
            hi = base + sgn * 0.19
            arcs.append((min(lo, hi), max(lo, hi)))
    jobs = []
    for (a0, a1) in arcs:
        th = a0
        while th < a1 - 1e-12:
            jobs.append((t0b, t1b, th, min(th + 0.017, a1)))
            th += 0.017
    print(f'{len(jobs)} shoulder chunks over 4 arcs', flush=True)
    t0 = time.time()
    out = []
    fails = 0
    for j in jobs:
        r = R1mod.band_job(j)
        out.append(r)
        tag = 'OK  ' if r.get('ok') else 'FAIL'
        print(f"{tag} band={r['band']} cells={r.get('cells')} "
              f"worst={r.get('worst')} ({time.time()-t0:.0f}s)",
              flush=True)
        fails += 0 if r.get('ok') else 1
    payload = dict(
        schema_version=2, kind='huang_region1_supplement',
        star=dict(W_ANG=R1mod.W_ANG, WEDGE_HALF_OVERRIDE=0.19,
                  T_SHOULDER=[t0b, t1b], A1S=R1mod.A1S, A2S=R1mod.A2S,
                  origin='symbolic a*=(psi(1-q),q), s0=sqrt(1-q)'),
        fails=fails, results=out)
    os.makedirs(R1mod.RESULTS_DIR, exist_ok=True)
    p = os.path.join(R1mod.RESULTS_DIR, out_name)
    with open(p, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'DONE {len(jobs)} chunks, {fails} fails -> {p}', flush=True)
    R1mod.WEDGE_HALF = wedge_saved
    return fails


def main():
    kappa = C.iv(C.dec('0.05'), C.dec('0.05'))
    alpha = C.iv(C.dec('0.781073068'), C.dec('0.781074776'))
    q_seed = C.iv(C.dec('0.565'), C.dec('0.578'))
    la = C.locate_fp_bisect(C.dec('0.781073068'), kappa, q_seed)
    lb = C.locate_fp_bisect(C.dec('0.781074776'), kappa, q_seed)
    if la is None or lb is None:
        raise SystemExit('locate failed')
    R1.configure_region1(C.arb(C.dec('0.05')), alpha,
                         C.hull(la[0], lb[0]), C.hull(la[1], lb[1]))
    # widen the wedge policy so the walk gating admits the shoulders;
    # the ray certificates themselves are independent of the zones
    R1.WEDGE_HALF = 0.19
    t0b, t1b = 0.0070, 0.0090
    import math
    arcs = []
    for base in (R1.W_ANG, R1.W_ANG + math.pi):
        for sgn in (+1, -1):
            lo = base + sgn * 0.14
            hi = base + sgn * 0.19
            arcs.append((min(lo, hi), max(lo, hi)))
    jobs = []
    for (a0, a1) in arcs:
        th = a0
        while th < a1 - 1e-12:
            jobs.append((t0b, t1b, th, min(th + 0.017, a1)))
            th += 0.017
    print(f'{len(jobs)} shoulder chunks over 4 arcs', flush=True)
    t0 = time.time()
    out = []
    fails = 0
    for j in jobs:
        r = R1.band_job(j)
        out.append(r)
        tag = 'OK  ' if r.get('ok') else 'FAIL'
        print(f"{tag} band={r['band']} cells={r.get('cells')} "
              f"worst={r.get('worst')} ({time.time()-t0:.0f}s)",
              flush=True)
        fails += 0 if r.get('ok') else 1
    payload = dict(
        schema_version=2, kind='huang_region1_supplement',
        star=dict(W_ANG=R1.W_ANG, WEDGE_HALF_OVERRIDE=0.19,
                  T_SHOULDER=[t0b, t1b], A1S=R1.A1S, A2S=R1.A2S,
                  origin='symbolic a*=(psi(1-q),q), s0=sqrt(1-q)'),
        fails=fails, results=out)
    os.makedirs(R1.RESULTS_DIR, exist_ok=True)
    p = os.path.join(R1.RESULTS_DIR, 'huang_region1_supp_0p05.json')
    with open(p, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'DONE {len(jobs)} chunks, {fails} fails -> {p}', flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
