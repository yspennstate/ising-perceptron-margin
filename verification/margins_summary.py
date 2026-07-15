"""Health summary across certified margins: per tag, the piece totals
and the worst (closest-to-zero) certified margin per family, so a
collapsing margin is visible across the kappa program before the slab
that fails.  Reads results/huang_*_<tag>.json; float parsing only (the
rigor lives in the manifests).

Usage: python margins_summary.py [tag ...]     (default: all found)
"""
import ast
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')


def _worst_records(d):
    """Max certified upper endpoint over ok cells of a sweep-style
    manifest ('records', worst=repr(float) for ok cells)."""
    vals = []
    for rec in d.get('records', []):
        if not rec.get('ok'):
            continue
        w = rec.get('worst')
        if w is None or w == 'None':
            continue
        try:
            v = ast.literal_eval(w) if isinstance(w, str) else w
        except (ValueError, SyntaxError):
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return max(vals) if vals else None


def _worst_results(d):
    vals = [float(rec['worst']) for rec in d.get('results', [])
            if rec.get('ok') and rec.get('worst') is not None]
    return max(vals) if vals else None


def summarize(tag):
    out = {'tag': tag}

    def load(stem):
        p = os.path.join(R, 'huang_%s_%s.json' % (stem, tag))
        return json.load(open(p)) if os.path.exists(p) else None

    sw = load('sweep')
    if sw:
        out['sweep'] = dict(fails=sw.get('failures'),
                            leaves=sum(r.get('leaves', 0)
                                       for r in sw.get('records', [])),
                            worst=_worst_records(sw))
    r1 = load('region1')
    if r1:
        out['region1'] = dict(fails=r1.get('fails'),
                              bands=len(r1.get('results', [])),
                              worst=_worst_results(r1))
    supp = load('region1_supp')
    if supp:
        out['supplement'] = dict(fails=supp.get('fails'),
                                 chunks=len(supp.get('results', [])),
                                 worst=_worst_results(supp))
    s2 = load('sweep2')
    if s2:
        out['stage2'] = dict(fails=s2.get('failures'),
                             leaves=sum(r.get('leaves', 0)
                                        for r in s2.get('records', [])),
                             worst=_worst_records(s2))
    si = load('star_interior')
    if si:
        b = si.get('bounds', {})

        def mid(k):
            # mid10 is a decimal-digit string scaled by exp10 (the
            # certificate stores exact integers, not floats)
            v = b.get(k, {})
            if not isinstance(v, dict) or 'mid10' not in v:
                return None
            exp = int(v.get('exp10', 0))
            return float(int(v['mid10']) * (10.0 ** exp))
        inr, req = mid('inradius'), mid('required_radius')
        out['starint'] = dict(inradius=inr, required=req,
                              slack=(inr / req - 1) if inr and req else None)
    return out


def main(tags):
    if not tags:
        tags = sorted({m.group(1) for f in glob.glob(
            os.path.join(R, 'huang_sweep_*.json'))
            for m in [re.search(r'huang_sweep_(.+)\.json$',
                                os.path.basename(f))] if m})
    rows = [summarize(t) for t in tags]
    for row in rows:
        print('== %s' % row['tag'])
        for fam in ('sweep', 'region1', 'supplement', 'stage2'):
            if fam in row:
                d = row[fam]
                n = d.get('leaves', d.get('bands', d.get('chunks')))
                w = d.get('worst')
                print('  %-10s fails=%-3s n=%-6s worst=%s'
                      % (fam, d['fails'], n,
                         '%.4g' % w if w is not None
                         else 'n/a (containment cells carry no value)'))
        if 'starint' in row:
            d = row['starint']
            print('  %-10s inradius=%.5f required=%.5f slack=%.1f%%'
                  % ('starint', d['inradius'], d['required'],
                     100 * d['slack']))
    return rows


if __name__ == '__main__':
    main(sys.argv[1:])
