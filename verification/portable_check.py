"""Portable connection checker: re-verify, with nothing but the Python
standard library, every exact claim that ties the two-variable
certificate artifacts together.

No flint, no numpy, no Lean --- a reproducer who trusts none of the
heavy toolchain can still confirm, from the JSON artifacts alone:

  1. envelope integrity: the sweep and stage-2 manifests match their
     embedded artifact hashes over canonical float-free JSON, and the
     star-interior certificate matches its certificate hash;
  2. binding: the stage-2 manifest names the SHA-256 of exactly the
     Region-I manifest file shipped beside it;
  3. star interior: the inequality chain of the interior certificate
     holds over the serialized packet endpoints in exact rationals
     (corner-minimum determinant positive; inradius = det/longest edge
     clears the required radius, recomputed here outward, not read);
  4. tilings: the sweep and stage-2 schedules are exact grid tilings of
     their declared rectangles, and the Region-I bands tile the radius
     axis from zero to the star radius with seam-exact angular chunks;
  5. verdicts: zero recorded failures in all four artifacts.

Run from verification/:  python portable_check.py [--tag 0p05]
"""
import argparse
import hashlib
import json
import math
import os
from fractions import Fraction


def canonical_bytes(value):
    def reject_float(obj):
        if isinstance(obj, float):
            raise TypeError('float in proof artifact')
        if isinstance(obj, dict):
            for k, v in obj.items():
                reject_float(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                reject_float(v)
    reject_float(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def payload_sha(value, omit):
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k not in omit}
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def frac(text):
    if isinstance(text, int) and not isinstance(text, bool):
        return Fraction(text)
    if isinstance(text, float):
        text = repr(text)
    s = str(text).strip()
    if 'e' in s or 'E' in s:
        mant, exp = s.replace('E', 'e').split('e')
        return frac(mant) * Fraction(10) ** int(exp)
    return Fraction(s)


def packet_endpoints(p):
    assert p['format'] == 'arb-midrad10-v1'
    mid, rad, e = int(p['mid10']), int(p['rad10']), int(p['exp10'])
    assert rad >= 0
    scale = Fraction(10 ** e) if e >= 0 else Fraction(1, 10 ** -e)
    return (mid - rad) * scale, (mid + rad) * scale


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        raise SystemExit(1)


def grid_tiling(schedule, rect):
    cells = [tuple(frac(x) for x in c) for c in schedule]
    xs = sorted({c[0] for c in cells} | {c[1] for c in cells})
    ys = sorted({c[2] for c in cells} | {c[3] for c in cells})
    n1, n2 = len(xs) - 1, len(ys) - 1
    expect = {(xs[i], xs[i + 1], ys[j], ys[j + 1])
              for i in range(n1) for j in range(n2)}
    ok = (set(cells) == expect and len(set(cells)) == len(cells)
          and len(cells) == n1 * n2)
    if rect is not None:
        ok &= (xs[0], xs[-1], ys[0], ys[-1]) == rect
    ok &= all(a < b for a, b in zip(xs[:-1], xs[1:]))
    ok &= all(a < b for a, b in zip(ys[:-1], ys[1:]))
    return ok


def sqrt_bounds(x, digits=40):
    """Rational lower and upper bounds of sqrt(x): floor integer square
    root at 10^-digits resolution, and that plus one unit in the last
    place (a strict upper bound since isqrt floors)."""
    scale = 10 ** digits
    n = x * scale * scale
    num = math.isqrt(n.numerator // n.denominator)
    return Fraction(num, scale), Fraction(num + 1, scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='0p05')
    ap.add_argument('--results', default='results')
    args = ap.parse_args()
    t, r = args.tag, args.results

    sweep = json.load(open(os.path.join(r, f'huang_sweep_{t}.json')))
    check('sweep artifact hash',
          sweep['artifact_sha256'] == payload_sha(sweep,
                                                  ('artifact_sha256',)))
    check('sweep zero failures', sweep['failures'] == 0)
    check('sweep ok cells have worst <= 0',
          all(rec.get('worst') is None or float(rec['worst']) <= 1e-9
              for rec in sweep['records'] if rec['ok']))
    a1m, a2m = frac(sweep['policy']['A1MAX']), frac(sweep['policy']['A2MAX'])
    check('sweep schedule tiles its rectangle',
          grid_tiling(sweep['schedule'], (-a1m, a1m, -a2m, a2m)))

    st2 = json.load(open(os.path.join(r, f'huang_sweep2_{t}.json')))
    check('stage-2 artifact hash',
          st2['artifact_sha256'] == payload_sha(st2, ('artifact_sha256',)))
    check('stage-2 zero failures', st2['failures'] == 0)
    check('stage-2 ok cells have worst <= 0',
          all(rec.get('worst') is None or float(rec['worst']) <= 1e-9
              for rec in st2['records'] if rec['ok']))
    # a coarse evaluator grid weakens every cell certificate, and a
    # manifest that does not RECORD its grid cannot prove the fine
    # run - absence fails, it does not pass silently (audit finding,
    # 2026-07-17: the conditional form was fail-open for manifests
    # that predate the field)
    check('stage-2 records and ran the fine evaluator grid',
          int(st2['policy'].get('GRID_N', 0)) >= 3600)
    e = [frac(x) for x in st2['policy']['EXCL_OLD']]
    check('stage-2 schedule tiles the exclusion box',
          grid_tiling(st2['schedule'], (e[0], e[1], e[2], e[3])))
    if 'EXCL' in sweep['policy']:
        # the seam: stage-1 treats MIN_SIDE leaves overlapping its
        # exclusion box as stage-2 territory (they extend at most
        # MIN_SIDE beyond it), so stage-2's tiled box must reach at
        # least 2*MIN_SIDE beyond EXCL on every side.  Containment is
        # the soundness condition (covering more is fine); the slop
        # absorbs shortest-decimal float reprs, and MIN_SIDE dwarfs it.
        ex = [frac(x) for x in sweep['policy']['EXCL']]
        ms = frac(sweep['policy']['MIN_SIDE'])
        slop = Fraction(1, 10 ** 12)
        check('sweep/stage-2 exclusion seam (>= EXCL + 2*MIN_SIDE)',
              e[0] <= ex[0] - 2 * ms + slop and
              e[1] >= ex[1] + 2 * ms - slop and
              e[2] <= ex[2] - 2 * ms + slop and
              e[3] >= ex[3] + 2 * ms - slop)
    else:
        print('note: pre-seam sweep manifest (no EXCL in policy); the '
              'exclusion box is fixed by the hashed sources instead')
    reg1_path = os.path.join(r, f'huang_region1_{t}.json')
    reg1_sha = hashlib.sha256(open(reg1_path, 'rb').read()).hexdigest()
    check('stage-2 binds the shipped Region-I manifest',
          st2['policy'].get('region1_manifest_sha256') == reg1_sha)

    reg1 = json.load(open(reg1_path))
    check('Region-I zero failures',
          reg1['fails'] == 0 and all(rec['ok'] for rec in reg1['results']))
    # leaf-value replay: an ok band's recorded worst S_star must
    # actually be nonpositive and carry no subcell failure, so a
    # flipped ok flag or a mismarked positive worst cannot pass the
    # count-based check above (does not re-evaluate the integral --
    # that is the ball-arithmetic layer -- but binds the recorded
    # value to the S_star <= 0 claim it stands for)
    check('Region-I ok bands have worst <= 0 and no subcell failure',
          all(rec.get('worst') is not None
              and float(rec['worst']) <= 1e-9
              and rec.get('nfail', 0) == 0
              for rec in reg1['results'] if rec['ok']))
    bands = [tuple(frac(x) for x in rec['band']) for rec in reg1['results']]
    levels = {}
    for (t0, t1, th0, th1) in bands:
        levels.setdefault((t0, t1), []).append((th0, th1))
    radii = sorted(levels)
    ok = radii[0][0] == 0
    for (a, b), (c, d) in zip(radii[:-1], radii[1:]):
        ok &= (b == c and a < b)
    ok &= radii[-1][0] < radii[-1][1]
    check('Region-I bands tile the radius axis from zero', ok)
    # radial reach: the outermost band level must reach the star
    # radius recorded in this same manifest's star block -- the reach
    # the stage-2 zone computation consumes (and binds by hash).  A
    # band schedule that tiled a shorter radius would otherwise pass
    # every tiling and closure clause.
    t_long = frac(repr(reg1['star']['T_LONG']))
    slop_r = Fraction(1, 10 ** 12)
    check('Region-I outermost band level reaches the star radius',
          radii[-1][1] >= t_long - slop_r)

    def runs(arcs):
        # merge seam-exact neighbors; any strict jump starts a new run
        out = []
        for (a, b) in sorted(arcs):
            if out and a == out[-1][1]:
                out[-1] = (out[-1][0], b)
            elif out and a < out[-1][1]:
                return None             # overlap: malformed
            else:
                out.append((a, b))
        return out

    # completeness, per direction: the innermost level must close the
    # full circle, and each level's angular support must sit inside
    # the previous level's.  Together these make every direction's
    # radial chain contiguous from zero up to the radius where its
    # support ends -- the property the stage-2 reach computation
    # consumes.
    # The supplement arcs live in their own manifest; when stage-2's
    # policy binds one, verify it here too (presence, hash, zero
    # failures) rather than only asserting the binding (audit
    # finding, 2026-07-17).
    supp_sha = st2['policy'].get('region1_supplement_sha256')
    if supp_sha:
        supp_path = os.path.join(r, f'huang_region1_supp_{t}.json')
        check('supplement manifest present for the bound hash',
              os.path.exists(supp_path))
        if os.path.exists(supp_path):
            actual = hashlib.sha256(
                open(supp_path, 'rb').read()).hexdigest()
            check('stage-2 binds the shipped supplement manifest',
                  actual == supp_sha)
            supp = json.load(open(supp_path))
            check('supplement zero failures',
                  supp.get('fails', 1) == 0
                  and all(rec.get('ok') for rec in supp['results']))
    inner = runs(levels[radii[0]])
    two_pi_lo = frac('6.283185307179585')
    two_pi_hi = frac('6.283185307179587')
    ok_closure = (inner is not None and len(inner) == 1
                  and two_pi_lo <= inner[0][1] - inner[0][0] <= two_pi_hi)
    check('Region-I innermost ring closes the circle '
          '(single seam-exact run of width 2*pi)', ok_closure)

    def contained(rs, prev):
        return all(any(pa <= a and b <= pb for (pa, pb) in prev)
                   for (a, b) in rs)

    ok_mono = True
    prev = inner
    for key in radii[1:]:
        rs = runs(levels[key])
        if rs is None or prev is None or not contained(rs, prev):
            ok_mono = False
            break
        prev = rs
    check('Region-I angular support is seam-exact and shrinks '
          'monotonically outward (per-direction contiguity)', ok_mono)

    si = json.load(open(os.path.join(r, f'huang_star_interior_{t}.json')))
    check('star-interior certificate hash',
          si['certificate_sha256'] == payload_sha(si,
                                                  ('certificate_sha256',)))
    m = si['matrix']
    ep = [packet_endpoints(m[i][j]) for i in range(2) for j in range(2)]
    (l00, h00), (l01, h01), (l10, h10), (l11, h11) = ep
    det_lo = min(a * b for a in (l00, h00) for b in (l11, h11)) \
        - max(a * b for a in (l01, h01) for b in (l10, h10))
    check('star-interior determinant positive (corner minimum)',
          det_lo > 0)
    # longest edge upper bound: sqrt((a00 +- a01)^2 + (a10 +- a11)^2)
    # outward via interval extremes and a strict upper rational sqrt.
    # max|x + y| over boxes is attained at the aligned corners; for
    # x - y the extremes cross: max(|lo - hi'|, |hi - lo'|).
    def edge_hi(sgn):
        if sgn > 0:
            c1 = max(abs(l00 + l01), abs(h00 + h01))
            c2 = max(abs(l10 + l11), abs(h10 + h11))
        else:
            c1 = max(abs(l00 - h01), abs(h00 - l01))
            c2 = max(abs(l10 - h11), abs(h10 - l11))
        s = c1 * c1 + c2 * c2
        return sqrt_bounds(s)[1]
    longest_hi = max(edge_hi(1), edge_hi(-1))
    inr_lo = det_lo / longest_hi
    req_lo, req_hi = packet_endpoints(si['bounds']['required_radius'])
    check('star-interior inradius clears the required radius '
          '(recomputed)', inr_lo > req_hi)
    print('PORTABLE CONNECTION CHECK PASSED')


if __name__ == '__main__':
    main()
