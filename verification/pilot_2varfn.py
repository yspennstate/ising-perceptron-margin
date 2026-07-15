"""Run the Condition-2varfn certificate pipeline at one margin (or one
kappa slab): locate the certified (q0, psi0) boxes at the alpha-interval
endpoints, configure the general-kappa modules, and run the requested
stages.

    python pilot_2varfn.py --ktag 0p05 --kappa 0.05 \
        --alpha-lo 0.781073068 --alpha-hi 0.781074776 \
        --q-lo 0.565 --q-hi 0.578 \
        --stages sweep,region1,starint,stage2,assemble [--workers 2]

For a kappa SLAB pass --kappa-lo/--kappa-hi instead of --kappa; every
certificate then covers the whole slab (the located boxes hull both
alpha endpoints at the slab's kappa ball).  Stages run in dependency
order; stage2 and assemble need region1's manifest on disk.
"""
import argparse
import os
import platform
import sys

os.environ.setdefault('HUANG_GRID_N', '3600')
platform.node = lambda: 'research-worker'   # manifests carry no machine names
if sys.platform.startswith('linux'):
    sys.executable = 'python3'   # manifests carry no machine paths (fork-safe on the worker hosts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ktag', required=True)
    p.add_argument('--kappa', type=str, default=None)
    p.add_argument('--kappa-lo', type=str, default=None)
    p.add_argument('--kappa-hi', type=str, default=None)
    p.add_argument('--alpha-lo', type=str, required=True)
    p.add_argument('--alpha-hi', type=str, required=True)
    p.add_argument('--q-lo', type=str, required=True)
    p.add_argument('--q-hi', type=str, required=True)
    p.add_argument('--psi-lo', type=str, default=None)
    p.add_argument('--psi-hi', type=str, default=None)
    p.add_argument('--stages', default='sweep,region1,supplement,'
                                       'starint,stage2,assemble')
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    klo = args.kappa_lo if args.kappa_lo is not None else args.kappa
    khi = args.kappa_hi if args.kappa_hi is not None else args.kappa
    if klo is None:
        raise SystemExit('need --kappa or --kappa-lo/--kappa-hi')
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]

    import certify_km_slab as C
    kappa = C.iv(C.dec(klo), C.dec(khi))
    alpha = C.iv(C.dec(args.alpha_lo), C.dec(args.alpha_hi))
    if args.psi_lo is not None and args.psi_hi is not None:
        # direct-enclosure mode: the caller supplies certified (q, psi)
        # windows (hulled from the strip's per-slab located enclosures),
        # so no fixed-point location runs here.  The locator is proven
        # at the strip's 5e-4 slab width and fails on wide kappa balls;
        # aggregation replaces re-derivation.
        q = C.iv(C.dec(args.q_lo), C.dec(args.q_hi))
        psi = C.iv(C.dec(args.psi_lo), C.dec(args.psi_hi))
    else:
        q_seed = C.iv(C.dec(args.q_lo), C.dec(args.q_hi))
        la = C.locate_fp_bisect(C.dec(args.alpha_lo), kappa, q_seed)
        lb = C.locate_fp_bisect(C.dec(args.alpha_hi), kappa, q_seed)
        if la is None or lb is None:
            raise SystemExit('locate failed')
        q = C.hull(la[0], lb[0])
        psi = C.hull(la[1], lb[1])
    kball = kappa if klo != khi else C.arb(C.dec(klo))

    import huang_cert_region1 as R1
    R1.configure_region1(kball, alpha, q, psi)
    ktag = R1.KTAG
    if ktag != args.ktag:
        print('note: derived ktag %s (from kappa) supersedes --ktag %s'
              % (ktag, args.ktag), flush=True)
    print('ktag=%s a*=(%.8f, %.8f) s0=%.8f W_ANG=%.4f a1 ball rad=%.2e'
          % (ktag, R1.A1S, R1.A2S, R1.S0F, R1.W_ANG,
             R1._ball_rad(R1.A1B)), flush=True)

    if 'sweep' in stages:
        import huang_cert_sweep as SW
        SW.configure_sweep(ktag, kball, alpha, q, psi)
        sys.argv = ['sweep', str(args.workers)]
        SW.main()
    if 'region1' in stages:
        sys.argv = ['region1', str(args.workers)]
        R1.main()
    if 'supplement' in stages:
        import supplement_0p05 as SUPP
        # outer radius 0.0105: the 0.13 run showed stage-2 leaf corners
        # reach past 0.009 when their angular interval spans the
        # wedge-shoulder boundary; wider arcs only add coverage
        nf = SUPP.run_supplement(
            R1, 'huang_region1_supp_%s.json' % ktag, t1b=0.0105)
        if nf:
            raise SystemExit('supplement failed %d chunks' % nf)
    if 'starint' in stages:
        import huang_cert_star_interior as SI
        out = os.path.join(R1.RESULTS_DIR,
                           'huang_star_interior_%s.json' % ktag)
        payload = SI.write_certificate(out)
        print('star interior %s: %s' % (ktag,
                                        payload['certificate_sha256'][:16]),
              flush=True)
    if 'stage2' in stages:
        import huang_cert_sweep as SW
        if SW.KTAG is None:
            SW.configure_sweep(ktag, kball, alpha, q, psi)
        import huang_cert_sweep2 as S2
        S2.configure_sweep2(
            os.path.join(R1.RESULTS_DIR,
                         'huang_region1_%s.json' % ktag),
            supplement_path=os.path.join(
                R1.RESULTS_DIR, 'huang_region1_supp_%s.json' % ktag))
        S2.main(args.workers)
    if 'assemble' in stages:
        import assemble_2varfn as A
        A.assemble(ktag)


if __name__ == '__main__':
    main()
