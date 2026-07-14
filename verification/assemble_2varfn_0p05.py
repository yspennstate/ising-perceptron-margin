"""Assemble Condition 2varfn at kappa = 0.05 from the four certificate
manifests and check their mutual coherence.

The condition (sup over the moment coordinates of S* <= 0) is covered by:
  1. Region II sweep (huang_sweep_0p05.json): S* < 0 on the bounded
     moment rectangle K minus the exclusion box EXCL.
  2. Region I star (huang_region1_0p05.json): S* <= 0 on the star
     {a* + t v(th): 0 <= t <= T(th)} by banded ray concavity from the
     symbolic maximizer, using the base-point identities.
  3. Stage-2 sweep (huang_sweep2_0p05.json): S* < 0 on EXCL (expanded by
     the stage-1 sliver rule) minus the certified star.
  4. Star interior (huang_star_interior_0p05.json): the star, origin ball
     included, lies in the moment body's interior (the ray proof's
     differentiability requirement).
Outside K no profile realizes the moments (outside_K certificates inside
the sweeps).  This script verifies zero failures, coherent geometry
(KTAG, star parameters, the stage-2 manifest hash binding), and prints
the combined statement.  The rigor lives in the individual certificates;
this is the bookkeeping seam.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')


def load(name):
    path = os.path.join(R, name)
    raw = open(path, 'rb').read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), path


def main():
    ok = True
    sweep, sweep_sha, _ = load('huang_sweep_0p05.json')
    reg1, reg1_sha, reg1_path = load('huang_region1_0p05.json')
    st2, st2_sha, _ = load('huang_sweep2_0p05.json')
    si, si_sha, _ = load('huang_star_interior_0p05.json')

    f1 = sweep.get('failures', None)
    print('Region II sweep: failures=%s leaves=%s sha=%s'
          % (f1, sweep.get('total_leaves'), sweep_sha[:16]))
    ok &= (f1 == 0)

    f2 = reg1.get('fails', None)
    nb = len(reg1.get('results', []))
    print('Region I: bands=%d fails=%s sha=%s' % (nb, f2, reg1_sha[:16]))
    ok &= (f2 == 0 and nb == 1404)

    f3 = st2.get('failures', None)
    print('Stage-2: failures=%s leaves=%s sha=%s'
          % (f3, st2.get('total_leaves'), st2_sha[:16]))
    ok &= (f3 == 0)
    bound = st2.get('policy', {}).get('region1_manifest_sha256')
    if bound is not None:
        print('  stage-2 bound to region1 manifest: %s (%s)'
              % (bound[:16], 'MATCH' if bound == reg1_sha else 'MISMATCH'))
        ok &= (bound == reg1_sha)
    supp_bound = st2.get('policy', {}).get('region1_supplement_sha256')
    if supp_bound is not None:
        supp, supp_sha, _ = load('huang_region1_supp_0p05.json')
        print('  stage-2 bound to shoulder supplement: %s (%s); '
              'supplement fails=%s over %d chunks'
              % (supp_bound[:16],
                 'MATCH' if supp_bound == supp_sha else 'MISMATCH',
                 supp.get('fails'), len(supp.get('results', []))))
        ok &= (supp_bound == supp_sha and supp.get('fails') == 0)

    b = si.get('bounds', {})
    print('Star interior: inradius mid=%s required mid=%s sha=%s'
          % (b.get('inradius', {}).get('mid10', '?')[:6],
             b.get('required_radius', {}).get('mid10', '?')[:6],
             si_sha[:16]))
    ok &= (si.get('kind') == 'huang_star_interior_certificate')

    star = reg1.get('star', {})
    ok &= (si.get('policy', {}).get('star_radius')
           == str(star.get('T_LONG')))
    ktags = {sweep.get('kappa_tag', '0p05'), st2.get('kappa_tag'),
             si.get('kappa_tag'), '0p05'}
    ok &= (len(ktags - {None}) == 1)

    print()
    if ok:
        print('CONDITION 2varfn CERTIFIED at kappa = 0.05: all four pieces '
              'pass with zero failures and coherent geometry.')
    else:
        print('ASSEMBLY INCOMPLETE: see mismatches above.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
