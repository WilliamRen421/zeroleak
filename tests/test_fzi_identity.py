"""The FZI exponent identity, on synthetic data with known exponents.

Claim under test: the exponents (3, -2) that are routinely recovered from core
data and reported as validation of the Kozeny-Carman form are an algebraic
identity, not an empirical finding.  The test builds data from the identity,
recovers the exponents, and asserts they come back to machine precision with
zero residual -- which is what an identity does and what a fitted relation does
not.
"""
from __future__ import annotations

import numpy as np

C_UNITS = 0.0314  # RQI = 0.0314 sqrt(k / phi)


def _identity_k(phi: np.ndarray, fzi: np.ndarray) -> np.ndarray:
    """k from the FZI definition, rearranged."""
    phi_z = phi / (1.0 - phi)
    rqi = fzi * phi_z
    return (rqi / C_UNITS) ** 2 * phi


def test_exponents_are_recovered_to_machine_precision():
    rng = np.random.default_rng(7)
    phi = rng.uniform(0.05, 0.35, 4000)
    fzi = 10 ** rng.normal(-0.5, 0.9, phi.size) * 1e-3
    k = _identity_k(phi, fzi)

    y = np.log10(k)
    X = np.column_stack([
        np.log10(fzi),
        np.log10(phi),
        np.log10(1.0 - phi),
        np.ones(phi.size),
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1.0 - resid @ resid / np.sum((y - y.mean()) ** 2)

    assert np.allclose(coef[:3], [2.0, 3.0, -2.0], atol=1e-10), coef[:3]
    assert abs(r2 - 1.0) < 1e-14
    assert np.max(np.abs(resid)) < 1e-12


def test_identity_holds_pointwise_not_only_on_average():
    """The identity is not a regression that could have come out otherwise: each
    sample lies on the surface, and the deviations are floating-point noise."""
    rng = np.random.default_rng(11)
    phi = rng.uniform(0.03, 0.40, 500)
    fzi = 10 ** rng.normal(-1.0, 1.2, phi.size) * 1e-3
    k = _identity_k(phi, fzi)

    lhs = np.log10(k)
    rhs = 2 * np.log10(fzi) + 3 * np.log10(phi) - 2 * np.log10(1 - phi) - 2 * np.log10(C_UNITS)
    assert np.max(np.abs(lhs - rhs)) < 1e-12


def test_an_empirical_relation_does_not_reproduce_the_exponents():
    """A control: data generated with different exponents do not return (3, -2),
    so the first test is measuring the identity and not the fit machinery."""
    rng = np.random.default_rng(3)
    phi = rng.uniform(0.05, 0.35, 4000)
    fzi = 10 ** rng.normal(-0.5, 0.9, phi.size) * 1e-3
    phi_z = phi / (1.0 - phi)
    k = (fzi * phi_z ** 1.25) ** 2 * phi ** 2.2  # two exponents deliberately wrong

    y = np.log10(k)
    X = np.column_stack([np.log10(fzi), np.log10(phi), np.log10(1 - phi), np.ones(phi.size)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert not np.allclose(coef[:3], [2.0, 3.0, -2.0], atol=1e-3)
