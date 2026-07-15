"""Assemble Condition 2varfn at any certified margin tag: check the
four manifests' zero-failure status, mutual hash bindings, and coherent
geometry, and print the combined verdict.  The rigor lives in the
individual certificates; this is the bookkeeping seam.

Usage: python assemble_2varfn.py <tag>      (e.g. 0p05, n0p05)
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')


def load(name):
    path = os.path.join(R, name)
    raw = open(path, 'rb').read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), path


def assemble(tag):
    ok = True
    sweep, sweep_sha, _ = load('huang_sweep_%s.json' % tag)
    print('Region II sweep: failures=%s leaves=%s sha=%s'
          % (sweep.get('failures'), sweep.get('total_leaves'),
             sweep_sha[:16]))
    ok &= (sweep.get('failures') == 0)

    reg1, reg1_sha, _ = load('huang_region1_%s.json' % tag)
    nb = len(reg1.get('results', []))
    print('Region I: bands=%d fails=%s sha=%s'
          % (nb, reg1.get('fails'), reg1_sha[:16]))
    ok &= (reg1.get('fails') == 0 and nb == 1404)

    st2, st2_sha, _ = load('huang_sweep2_%s.json' % tag)
    print('Stage-2: failures=%s leaves=%s sha=%s'
          % (st2.get('failures'), st2.get('total_leaves'), st2_sha[:16]))
    ok &= (st2.get('failures') == 0)
    bound = st2.get('policy', {}).get('region1_manifest_sha256')
    print('  stage-2 bound to region1 manifest: %s (%s)'
          % (str(bound)[:16], 'MATCH' if bound == reg1_sha
             else 'MISMATCH'))
    ok &= (bound == reg1_sha)
    supp_bound = st2.get('policy', {}).get('region1_supplement_sha256')
    if supp_bound is not None:
        supp, supp_sha, _ = load('huang_region1_supp_%s.json' % tag)
        print('  stage-2 bound to shoulder supplement: %s (%s); '
              'supplement fails=%s over %d chunks'
              % (supp_bound[:16],
                 'MATCH' if supp_bound == supp_sha else 'MISMATCH',
                 supp.get('fails'), len(supp.get('results', []))))
        ok &= (supp_bound == supp_sha and supp.get('fails') == 0)

    si, si_sha, _ = load('huang_star_interior_%s.json' % tag)
    b = si.get('bounds', {})
    print('Star interior: inradius mid=%s required mid=%s sha=%s'
          % (b.get('inradius', {}).get('mid10', '?')[:6],
             b.get('required_radius', {}).get('mid10', '?')[:6],
             si_sha[:16]))
    ok &= (si.get('kind') == 'huang_star_interior_certificate')
    ok &= (si.get('policy', {}).get('star_radius')
           == str(reg1.get('star', {}).get('T_LONG')))

    print()
    if ok:
        print('CONDITION 2varfn CERTIFIED at tag %s: all pieces pass '
              'with zero failures and coherent geometry.' % tag)
    else:
        print('ASSEMBLY INCOMPLETE at tag %s: see mismatches above.'
              % tag)
        raise SystemExit(1)


if __name__ == '__main__':
    assemble(sys.argv[1] if len(sys.argv) > 1 else '0p05')
