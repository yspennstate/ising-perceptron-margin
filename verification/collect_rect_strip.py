"""Collect the thin-rectangle lane run into one bound artifact.

The per-slab workers (rect_strip.py, delta 5e-4, kappa sub-balls)
append rows to their own jsonl files; this collector verifies the
union before anything cites it: exactly one row per lane slab, kappa
boundaries contiguous across the whole lane and across worker seams,
each row's (kappa, alpha) window equal to the lane collector's CSV
row, every row ok with all eight sub-balls passing and the recorded
evaluator parameters (delta, locate depth) uniform, and the worst
contraction enclosure below one.  Output binds the input files by
SHA-256.

Run: python collect_rect_strip.py results/rect_strip_nak_thin_A.jsonl
         results/rect_strip_nak_thin_B.jsonl
"""
import csv
import hashlib
import json
import sys


def parse_arb(s):
    """Upper endpoint of an arb repr '[m +/- r]' (or a plain decimal),
    conservatively rounded outward."""
    s = s.strip()
    if s.startswith('['):
        body = s[1:s.rindex(']')]
        if '+/-' in body:
            m, r = body.split('+/-')
            return float(m), float(r)
        return float(body), 0.0
    return float(s), 0.0


def main():
    files = sys.argv[1:] or ['results/rect_strip_nak_thin_A.jsonl',
                             'results/rect_strip_nak_thin_B.jsonl']
    rows = []
    shas = {}
    for f in files:
        data = open(f, 'rb').read()
        shas[f] = hashlib.sha256(data).hexdigest()
        for line in data.decode().splitlines():
            rows.append(json.loads(line))
    rows.sort(key=lambda r: float(r['kappa_lo']))

    csv_rows = {r['kappa_lo']: r for r in
                csv.DictReader(open('results/certified_intervals_nak.csv',
                                    newline=''))}
    problems = []
    # workers checkpoint independently, so a resumed lane can cover a
    # slab twice; identical rows (the computation is deterministic
    # across hosts, verified on the smoke slab) collapse to one, and
    # only DIFFERING duplicates are a defect
    seen_rows = {}
    deduped = []
    for r in rows:
        key = r['kappa_lo']
        if key in seen_rows:
            if seen_rows[key] == r:
                continue
            problems.append('conflicting duplicate slab %s' % key)
        else:
            seen_rows[key] = r
            deduped.append(r)
    dropped = len(rows) - len(deduped)
    rows = deduped
    if dropped:
        print('note: %d identical duplicate row(s) collapsed' % dropped)
    if len(rows) != len(csv_rows):
        problems.append('row count %d != lane slab count %d'
                        % (len(rows), len(csv_rows)))
    seen = set()
    worst = (0.0, None)
    for i, r in enumerate(rows):
        key = r['kappa_lo']
        if key in seen:
            problems.append('duplicate slab %s' % key)
        seen.add(key)
        ref = csv_rows.get(key)
        if ref is None:
            problems.append('slab %s not in lane CSV' % key)
            continue
        for fld in ('kappa_hi', 'alpha_lb', 'alpha_ub'):
            if r[fld] != ref[fld]:
                problems.append('slab %s field %s differs from CSV'
                                % (key, fld))
        # the sub-ball count is part of the recipe, not row-supplied
        # state: a row must claim exactly eight sub-balls and have all
        # eight pass (a row carrying its own smaller n_sub must not
        # pass - audit finding, 2026-07-17)
        if (not r['ok'] or r.get('n_sub') != 8
                or r.get('pieces_ok') != 8):
            problems.append('slab %s not fully certified' % key)
        # delta is recorded as an arb ball repr; compare numerically
        dm, dr = parse_arb(str(r.get('delta', 'nan')))
        if not (abs(dm - 0.0005) <= dr + 1e-20 and dr < 1e-12
                and r.get('n_loc') == 4000):
            problems.append('slab %s ran off-recipe: delta=%s n_loc=%s'
                            % (key, r.get('delta'), r.get('n_loc')))
        if i and rows[i - 1]['kappa_hi'] != r['kappa_lo']:
            problems.append('gap before slab %s' % key)
        m, rad = parse_arb(r['contraction_worst'])
        hi = m + rad
        if not hi < 1.0:
            problems.append('slab %s contraction %s not below one'
                            % (key, r['contraction_worst']))
        if hi > worst[0]:
            worst = (hi, key)

    lo = rows[0]['kappa_lo'] if rows else None
    hi_k = rows[-1]['kappa_hi'] if rows else None
    ok = not problems
    out = {
        'kind': 'rect_strip_thin_summary',
        'inputs_sha256': shas,
        'intervals_csv_sha256': hashlib.sha256(
            open('results/certified_intervals_nak.csv', 'rb').read()
        ).hexdigest(),
        'slabs': len(rows), 'kappa_lo': lo, 'kappa_hi': hi_k,
        'delta': '0.0005', 'n_loc': 4000, 'n_sub': 8,
        'worst_contraction_upper': repr(worst[0]),
        'worst_contraction_slab': worst[1],
        'ok': ok, 'problems': problems,
    }
    with open('results/rect_strip_thin_summary.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('lane rectangles: %d slabs [%s, %s], worst contraction %.6f'
          % (len(rows), lo, hi_k, worst[0]))
    for p in problems:
        print('PROBLEM:', p)
    print('SUMMARY %s' % ('OK' if ok else 'NOT OK'))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
