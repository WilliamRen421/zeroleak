"""Layer 3, the model.

One power law per phase in log10 space, k = 10^a * phi^n, fitted by ordinary
least squares inside each phase.  The cutpoints arrive as a `Cutpoints` value and
are never estimated here: this module contains no quantile, no percentile, no
threshold search and no sort of the porosity vector, so it has no path from the
training data to a boundary.  `tests/test_layers.py` enforces that by walking
this file's syntax tree rather than by trusting the sentence you just read.

Fitting is allowed to see permeability, because that is what a fit is.  It is
given `FoldLabels` rather than the boundaries that produced them, so it can
exclude a fold but cannot locate one.
"""
from __future__ import annotations

import numpy as np

from ..types import (
    Coefficients,
    Cutpoints,
    FoldLabels,
    Permeability,
    Porosity,
    Predictions,
    requires,
)

__all__ = ["label", "fit", "predict", "MIN_PHASE"]

MIN_PHASE = 10  # below this a phase falls back to a single pooled line

# Attribute names this module must not contain; scanned by the test suite.
FORBIDDEN_CALLS = ("quantile", "percentile", "nanquantile", "median", "nanmedian")


def label(phi: Porosity, cuts: Cutpoints) -> np.ndarray:
    """Phase index per sample, on the absolute porosity scale."""
    return np.clip(np.digitize(np.log10(phi.x), cuts.log10), 0, cuts.phases - 1)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple:
    A = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


@requires(phi=Porosity, perm=Permeability, folds=FoldLabels, cuts=Cutpoints)
def fit(
    phi: Porosity,
    perm: Permeability,
    folds: FoldLabels,
    cuts: Cutpoints,
    hold_out: int = None,
) -> Coefficients:
    """Fit the phase coefficients, optionally leaving one fold out.

    `hold_out` is a label, not a boundary: the training set is `folds != hold_out`
    and nothing else about the fold is available.
    """
    keep = folds.training(hold_out) if hold_out is not None else np.ones(len(phi), dtype=bool)
    if keep.sum() < 2:
        raise ValueError("fewer than two training samples")

    p = phi.x[keep]
    k = perm.x[keep]
    lab = label(Porosity(p), cuts)
    lp, lk = np.log10(p), np.log10(k)

    intercept = np.zeros(cuts.phases)
    slope = np.zeros(cuts.phases)
    a_pool, n_pool = _ols(lp, lk)
    for c in range(cuts.phases):
        m = lab == c
        if int(m.sum()) >= MIN_PHASE:
            intercept[c], slope[c] = _ols(lp[m], lk[m])
        else:
            # Too few samples to carry a slope: the phase keeps the pooled line
            # rather than an extrapolation from a handful of points.
            intercept[c], slope[c] = a_pool, n_pool
    return Coefficients(intercept, slope)


@requires(phi=Porosity, cuts=Cutpoints)
def predict(coefs: Coefficients, phi: Porosity, cuts: Cutpoints) -> Predictions:
    """Apply phase coefficients.  Needs the cutpoints because the phase of a
    sample is a property of the partition, not of the coefficients."""
    if coefs.phases != cuts.phases:
        raise ValueError(f"{coefs.phases} coefficient pairs for {cuts.phases} phases")
    lab = label(phi, cuts)
    y = np.empty(phi.x.size)
    for c in range(cuts.phases):
        m = lab == c
        if m.any():
            y[m] = coefs.intercept[c] + coefs.slope[c] * np.log10(phi.x[m])
    return Predictions(y)
