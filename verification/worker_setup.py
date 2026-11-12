"""Per-worker configuration for multiprocessing pools.

Under the spawn start method (Windows, and macOS by default) a pool worker
imports the certificate modules fresh, so the per-kappa state set by
configure_region1 / configure_sweep / configure_sweep2 in the parent is
absent.  The parent records its configuration here; each worker replays
it in its initializer.  Balls travel as exact (mantissa, exponent) pairs
for the midpoint and radius, so a worker's ball contains the parent's.
The slope table is copied verbatim, so every worker uses the parent's
slopes.
"""

SPEC = None


def _enc(x):
    from flint import arb
    if not isinstance(x, arb):
        x = arb(x)
    return (x.mid().man_exp(), x.rad().man_exp())


def _dec(t):
    from flint import arb, arf
    (m, e), (r, re_) = t
    return arb(arf((m, e)), arf((r, re_)))


def record_region1(kappa, alpha, q, psi):
    """Called by the driver right after configure_region1."""
    global SPEC
    import huang_cert_region1 as r1
    SPEC = dict(balls=[_enc(kappa), _enc(alpha), _enc(q), _enc(psi)],
                w_ang=r1.W_ANG, slope_table=list(r1.SLOPE_TABLE),
                sweep_ktag=None, sweep2=None)


def record_sweep(ktag):
    SPEC['sweep_ktag'] = ktag


def record_sweep2(manifest_path, supplement_path):
    SPEC['sweep2'] = (manifest_path, supplement_path)


def replay(spec):
    if spec is None:
        return
    import huang_cert_region1 as r1
    kappa, alpha, q, psi = [_dec(t) for t in spec['balls']]
    r1.configure_region1(kappa, alpha, q, psi, w_ang=spec['w_ang'])
    r1.SLOPE_TABLE = list(spec['slope_table'])
    if spec.get('sweep_ktag') is not None:
        import huang_cert_sweep as sw
        sw.configure_sweep(spec['sweep_ktag'], kappa, alpha, q, psi)
    if spec.get('sweep2') is not None:
        import huang_cert_sweep2 as s2
        s2.configure_sweep2(*spec['sweep2'])
