"""Rigorous interior-ball certificate for the Region-I star at general kappa.

Port of the capacity verification's huang_star_interior.py.  The Region-I
ray proof differentiates the moment entropy H and therefore needs every
ray point to lie in the interior of the moment body

    K = { E[(X, tanh X) Lambda] : |Lambda| <= 1 },   X ~ N(0, psi0).

Two explicit feasible profile perturbations around the maximizing profile
Lambda* = tanh(X) (the maximizer at EVERY kappa: a* = grad Phi(1,0)) map
the l1 unit diamond to a parallelogram around a*.  A 2x2 Arb computation
proves the parallelogram contains a Euclidean ball larger than the full
star, including its origin padding.

General kappa: psi0 is the located parameter ball (kappa enters K only
through it), and the star origin is the SYMBOLIC ball a* = (psi(1-q), q)
-- the required radius covers the per-coordinate ball radii (derived at
configure time, ~1e-4 at located margins) instead of the kappa = 0 proof's
1e-7 stored-decimal displacement.

Usage (after configuring huang_cert_region1 for the target kappa):
    import huang_cert_star_interior as si
    si.write_certificate(path)
"""

from __future__ import annotations

import os
import pathlib
import platform

import flint
from flint import acb, arb

import block3bc_exact as exact
import core
import huang_cert_region1 as region1


HERE = pathlib.Path(__file__).resolve().parent
PRECISION_BITS = 100
SPLIT = arb(3) / 4
CUTOFF = arb(12)
INTEGRATION_TOLERANCE = arb(2) ** -80
CERTIFIED_RADIUS_FLOOR = arb("0.0135")
RADIAL_PADDING = "0.000000000001"
ANGULAR_PADDING = "0.00000001"
MAX_LONG_ANGULAR_CELL_WIDTH = "0.008"
ANGULAR_INTERVAL_ROUNDING_GUARD = "0.0000000001"


def _source_paths():
    return {
        "huang_cert_star_interior.py": __file__,
        "huang_cert_region1.py": region1.__file__,
        "block3bc_exact.py": exact.__file__,
        "core.py": core.__file__,
    }


def source_hashes():
    return exact.source_hashes(_source_paths())


def runtime_record():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "python_flint": flint.__version__,
        "flint": flint.__FLINT_VERSION__,
        "precision_bits": PRECISION_BITS,
    }


def _density(x, psi):
    return (-x * x / (2 * psi)).exp() / (2 * arb.pi() * psi).sqrt()


def _body_integrals():
    """Return row-wise integrals below and above |x|=3/4.

    On x>0 the two perturbations use rho0=1-tanh(x) and rho1=rho0
    below the split, -rho0 above it.  Symmetry supplies the factor two."""
    psi = region1.PSI

    def row_x(x, analytic):
        m = x.tanh()
        return 2 * x * (1 - m) * _density(x, psi)

    def row_m(x, analytic):
        m = x.tanh()
        return 2 * m * (1 - m) * _density(x, psi)

    out = []
    for integrand in (row_x, row_m):
        inner = core.integrate(
            integrand, arb(0), SPLIT, abs_tol=INTEGRATION_TOLERANCE)
        outer = core.integrate(
            integrand, SPLIT, CUTOFF,
            abs_tol=INTEGRATION_TOLERANCE)
        out.append((inner, outer))
    return out


def _tail_bounds():
    """Symmetric absolute error bounds for the two omitted x-tails."""
    from core import endpoints
    _, psi_hi = endpoints(region1.PSI)
    z = CUTOFF / psi_hi.sqrt()
    row_x = 2 * psi_hi.sqrt() * core.phi(z)
    row_m = 2 * core.Psi(z)
    return row_x, row_m


def compute_matrix():
    core.set_prec(PRECISION_BITS)
    rows = _body_integrals()
    tails = _tail_bounds()
    matrix = []
    for (inner, outer), tail in zip(rows, tails):
        error = arb(0, tail)
        column_zero = inner + outer + error
        column_one = inner - outer + error
        matrix.append([column_zero, column_one])
    return matrix, tails


def _required_radius():
    # Region-I evaluates interval boxes around the SYMBOLIC origin
    # a* = (psi(1-q), q): relative to any point of that ball, a ray box is
    # displaced by at most twice the per-coordinate ball radius (interval
    # diameter), on top of the ray extent.  Independent cosine/sine
    # interval hulls over the widest long-ray leaf have norm at most
    # sqrt(1 + sin(theta_width)).
    expected_radial = arb(RADIAL_PADDING)
    expected_angular = arb(ANGULAR_PADDING)
    if (not abs(region1._RAD_PAD - expected_radial) < arb("1e-30")
            or not abs(region1._ANG_PAD - expected_angular) < arb("1e-30")):
        raise ValueError(
            "Region-I padding constants drifted from certificate policy")
    rb1 = arb(repr(region1._ball_rad(region1.A1B)))
    rb2 = arb(repr(region1._ball_rad(region1.A2B)))
    origin_norm = (rb1 * rb1 + rb2 * rb2).sqrt()
    theta_width = (arb(MAX_LONG_ANGULAR_CELL_WIDTH)
                   + 2 * expected_angular
                   + arb(ANGULAR_INTERVAL_ROUNDING_GUARD))
    direction_box_norm = (1 + theta_width.sin()).sqrt()
    return ((arb(str(region1.T_LONG)) + expected_radial) * direction_box_norm
            + 2 * origin_norm)


def compute_bounds(matrix):
    a00, a01 = matrix[0]
    a10, a11 = matrix[1]
    determinant = a00 * a11 - a01 * a10
    edge_minus = ((a00 - a01) ** 2 + (a10 - a11) ** 2).sqrt()
    edge_plus = ((a00 + a01) ** 2 + (a10 + a11) ** 2).sqrt()
    if edge_minus > edge_plus:
        longest_edge = edge_minus
    elif edge_plus > edge_minus:
        longest_edge = edge_plus
    else:
        longest_edge = edge_minus.union(edge_plus)
    inradius = determinant / longest_edge
    required = _required_radius()
    clearance = inradius - required
    if not determinant > 0:
        raise ValueError(
            "interior perturbation matrix is not orientation preserving")
    if not inradius > CERTIFIED_RADIUS_FLOOR:
        raise ValueError("constructive moment-body inradius is too small")
    if not inradius > required:
        raise ValueError(
            "Region-I star is not certified inside the moment body")
    return {
        "determinant": determinant,
        "edge_norm_minus": edge_minus,
        "edge_norm_plus": edge_plus,
        "longest_edge": longest_edge,
        "inradius": inradius,
        "required_radius": required,
        "clearance": clearance,
    }


def compute_certificate():
    matrix, tails = compute_matrix()
    bounds = compute_bounds(matrix)
    payload = {
        "schema_version": 1,
        "kind": "huang_star_interior_certificate",
        "kappa_tag": region1.KTAG,
        "policy": {
            "precision_bits": PRECISION_BITS,
            "split": "3/4",
            "cutoff": "12",
            "integration_tolerance": "2^-80",
            "star_radius": str(region1.T_LONG),
            "origin": "symbolic a*=(psi(1-q),q); per-coordinate ball radii "
                      "recorded below",
            "origin_ball_radii": [repr(region1._ball_rad(region1.A1B)),
                                  repr(region1._ball_rad(region1.A2B))],
            "radial_padding": "1/1000000000000",
            "angular_padding": "1/100000000",
            "max_long_angular_cell_width": "1/125",
            "angular_interval_rounding_guard": "1/10000000000",
            "center_interval_diameter_factor": 2,
            "direction_box_norm_bound":
                "sqrt(1+sin(width+2*angular_padding))",
            "certified_radius_floor": "27/2000",
        },
        "runtime": runtime_record(),
        "source_sha256": source_hashes(),
        "psi": exact.arb_packet(region1.PSI),
        "kappa": exact.arb_packet(region1.KAPPA),
        "matrix": [
            [exact.arb_packet(value) for value in row] for row in matrix],
        "tail_bounds": {
            "row_x": exact.arb_packet(tails[0]),
            "row_tanh": exact.arb_packet(tails[1]),
        },
        "bounds": {
            name: exact.arb_packet(value) for name, value in bounds.items()},
    }
    payload["certificate_sha256"] = exact.payload_sha256(
        payload, omit=("certificate_sha256",))
    return payload


def verify_certificate(path):
    observed = exact.load_json(path)
    if (not isinstance(observed, dict)
            or observed.get("schema_version") != 1
            or observed.get("kind") != "huang_star_interior_certificate"
            or observed.get("certificate_sha256") != exact.payload_sha256(
                observed, omit=("certificate_sha256",))):
        raise ValueError("invalid star-interior certificate envelope")
    expected = compute_certificate()
    if observed != expected:
        raise ValueError("star-interior certificate does not replay exactly")
    return observed


def write_certificate(path):
    payload = compute_certificate()
    exact.write_json_atomic(path, payload, overwrite=False)
    verify_certificate(path)
    return payload
