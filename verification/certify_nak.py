"""Batch runner for the Nakajima-lane slab certificate
(verify_slab_nak): the fixed-point condition past the rectangle
product bound's wall, kappa >= 0 only.

Usage: python certify_nak.py slabs_pos2.json result_pos2_0.json

The leading record is a lane marker with the derived psi-range
diagnostic; it makes no global claims (uniqueness is per-slab, from
the certified Nakajima premise), so its checks list is empty.
"""
import json
import sys

import certify_km_slab as C


def be_polite():
    if sys.platform == 'win32':
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(h, 0x4000)
        except Exception:
            pass


def main():
    be_polite()
    slabs_path = sys.argv[1] if len(sys.argv) > 1 else 'slabs_pos2.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'nak_results.json'
    with open(slabs_path) as fh:
        slabs = json.load(fh)
    lo = min(C.dec(s['q_lb']) for s in slabs)
    hi = max(C.dec(s['q_ub']) for s in slabs)
    a_lo = min(C.dec(s['alpha_lb']) for s in slabs)
    a_hi = max(C.dec(s['alpha_ub']) for s in slabs)
    k_lo = min(C.dec(s['kappa_lo']) for s in slabs)
    k_hi = max(C.dec(s['kappa_hi']) for s in slabs)
    R_g = C.R_of_wide(C.iv(lo, hi), C.iv(a_lo, a_hi), C.iv(k_lo, k_hi))
    g_rec = {'global': {'lane': 'nakajima',
                        'psi_range_diagnostic': [str(x) for x in
                                                 C.endpoints(R_g)],
                        'checks': [], 'ok': True}}
    results = [g_rec]
    n_ok = 0
    for i, slab in enumerate(slabs):
        print(f"slab {i}: kappa [{slab['kappa_lo']}, {slab['kappa_hi']}]",
              flush=True)
        ok, rec = C.verify_slab_nak(slab)
        results.append(rec)
        n_ok += ok
        print(f"  -> {'PASS' if ok else 'FAIL'}", flush=True)
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=1)
    print(f"{n_ok}/{len(slabs)} slabs verified (nakajima lane) -> "
          f"{out_path}")
    if n_ok < len(slabs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
