"""zeroleak -- a porosity-only permeability pipeline whose information sets are
visible in its call signatures.

The package is deliberately small.  It provides no interactive interface, no
database backend and no service, because each would add surface area without
changing what a reader can check.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import config, strategies, types  # noqa: F401
from .layers import data, folds, metrics, model, report  # noqa: F401

__all__ = ["config", "strategies", "types", "data", "folds", "model", "metrics", "report", "__version__"]
