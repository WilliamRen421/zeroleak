"""The five layers, one invariant each.

data    returns triples of porosity, permeability and depth; the only module
        that opens a measurement file.
folds   assigns fold labels from porosity and geometry; no permeability in the
        signature.
model   fits one power law per phase; cutpoints arrive as configuration, and
        the module contains no call that could estimate one.
metrics scores predictions against truth; no model in the signature, so it
        cannot refit.
report  reads exported CSV only; no model and no numpy, so a figure cannot
        disagree with its table.
"""
from __future__ import annotations

from . import data, folds, metrics, model, report  # noqa: F401

__all__ = ["data", "folds", "model", "metrics", "report"]
