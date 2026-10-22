"""Cutpoint strategies: the one legal place where a boundary may be estimated.

Two strategies are supported.

`prior` returns the configured pair unchanged.  It does not look at anything, so
it cannot be accused of having been chosen on the target: the numbers come from
a file that was written before the data were read, and the file is copied into
the output directory beside every run so that the claim is checkable after the
fact.

`target_terciles` estimates the boundaries as equal-frequency terciles of the
target's log porosity.  This is legal, and the point of the design is that it is
legal in a way that is visible: the function takes `Porosity` and has no
permeability parameter, so it can estimate a partition of porosity but not one
that encodes permeability.  The FZI criterion does the opposite, and that is why
no function here can express it.

Both live outside `layers.model`, which is why the model can be audited for the
absence of a quantile call and the package can still offer an estimated rule.
"""
from __future__ import annotations

import numpy as np

from .types import Cutpoints, Porosity, requires

__all__ = [
    "STRATEGIES",
    "resolve",
    "cutpoints_from_config",
    "cutpoints_from_porosity",
]

STRATEGIES = ("prior", "target_terciles")


@requires()
def cutpoints_from_config(cfg) -> Cutpoints:
    """The prior pair, taken from configuration.  The empty contract is the
    point: no data carrier appears in the signature, so the values cannot have
    been touched by the target."""
    return cfg.cutpoints


@requires(phi=Porosity)
def cutpoints_from_porosity(phi: Porosity, phases: int = 3) -> Cutpoints:
    """Equal-frequency boundaries of log porosity, porosity only."""
    if phases < 2:
        raise ValueError("at least two phases are required")
    q = np.quantile(np.log10(phi.x), np.arange(1, phases) / phases)
    return Cutpoints(tuple(float(10.0 ** t) for t in q))


def resolve(strategy: str, phi: Porosity, cfg) -> Cutpoints:
    """Map a strategy name and a porosity vector to a concrete partition."""
    if strategy == "prior":
        return cutpoints_from_config(cfg)
    if strategy == "target_terciles":
        return cutpoints_from_porosity(phi, cfg.phases)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
