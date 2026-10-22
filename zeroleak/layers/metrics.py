"""Layer 4, metrics.

Scores are computed from predictions and truth and from nothing else.  The module
imports no model, so it cannot refit, and it cannot see a cutpoint, so it cannot
reward a partition.  `Truth` is a separate type from `Predictions` for the same
reason: a call that hands over the wrong one fails before any arithmetic.
"""
from __future__ import annotations

import numpy as np

from ..types import Metrics, Predictions, Truth, requires

__all__ = ["score", "loss"]


@requires(pred=Predictions, truth=Truth)
def score(pred: Predictions, truth: Truth) -> Metrics:
    """Log-space R2 and the RMSE normalised by the observed standard deviation.

    The two are related by RMSE/sigma = sqrt(1 - R2), which is checked in the
    test suite rather than assumed.
    """
    if len(pred) != len(truth):
        raise ValueError(f"{len(pred)} predictions for {len(truth)} observations")
    resid = truth.x - pred.x
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((truth.x - truth.x.mean()) ** 2))
    if ss_tot == 0.0:
        raise ValueError("the truth is constant, so R2 is undefined")
    sd = float(np.sqrt(ss_tot / truth.x.size))
    return Metrics(
        values={
            "n": int(truth.x.size),
            "r2": 1.0 - ss_res / ss_tot,
            "rmse": float(np.sqrt(ss_res / truth.x.size)),
            "sigma": sd,
            "rmse_sigma": float(np.sqrt(ss_res / truth.x.size) / sd),
            "bias": float(resid.mean()),
        }
    )


def loss(reference: Metrics, candidate: Metrics) -> float:
    """R2 of the reference minus R2 of the candidate.

    Positive means the candidate is worse.  The sign convention is stated once,
    here, because a loss that can be read both ways is not a measurement: the
    test suite constructs a case in which the candidate is known to be worse and
    asserts that this function returns a positive number.
    """
    return float(reference["r2"] - candidate["r2"])
