"""Typed carriers for the information sets the layers exchange.

A layer's information set is its signature.  If a quantity is not a parameter,
the layer cannot see it, and that is a stronger statement than any comment.  The
classes below are the entire vocabulary, and they are deliberately narrow: there
is no object that holds porosity and permeability together, so no call can pass
one to a layer that may not see the other.

The guard is `requires`, applied to every layer function.  A call that does not
match the declared types raises `TypeError` before the body runs, which is what
makes a leak a runtime error rather than a reading exercise.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "Porosity",
    "Permeability",
    "Depth",
    "FoldLabels",
    "Cutpoints",
    "Coefficients",
    "Predictions",
    "Truth",
    "Metrics",
    "CARRIES",
    "requires",
]

# What each type is allowed to carry, used verbatim in the error message so that
# a failed call says which information set was about to cross a boundary.
CARRIES: Dict[str, str] = {
    "Porosity": "porosity values and nothing else",
    "Permeability": "permeability values",
    "Depth": "sample depth and nothing else",
    "FoldLabels": "fold indices only, never the boundaries that produced them",
    "Cutpoints": "phase boundaries, carried separately from the model",
    "Coefficients": "fitted intercepts and slopes",
    "Predictions": "predicted log10 permeability",
    "Truth": "measured log10 permeability",
    "Metrics": "scalar scores",
    "Body": "porosity, permeability and depth read from one file",
}


def _vector(values: Any, name: str, *, positive: bool = False) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    if positive and not np.all(a > 0):
        raise ValueError(f"{name} must be strictly positive")
    a = a.copy()
    a.setflags(write=False)
    return a


class _Vector:
    """Common behaviour: immutable values, and subsetting that keeps the type."""

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.x.size)

    def take(self, mask: Any):
        """Return the same information set restricted to `mask`."""
        return type(self)(self.x[mask])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(n={self.x.size})"


@dataclass(frozen=True)
class Porosity(_Vector):
    """Fractional porosity, strictly inside (0, 1)."""

    x: np.ndarray

    def __post_init__(self) -> None:
        a = _vector(self.x, "porosity")
        if np.any(a <= 0.0) or np.any(a >= 1.0):
            raise ValueError("porosity must lie strictly inside (0, 1)")
        object.__setattr__(self, "x", a)


@dataclass(frozen=True)
class Permeability(_Vector):
    """Permeability in mD, strictly positive."""

    x: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _vector(self.x, "permeability", positive=True))


@dataclass(frozen=True)
class Depth(_Vector):
    """Depth of each sample, used only to define contiguous fold blocks."""

    x: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _vector(self.x, "depth"))


@dataclass(frozen=True)
class FoldLabels(_Vector):
    """Integer fold label per sample.  The boundaries that produced them are not
    part of this object, so a layer holding it cannot reconstruct which index or
    depth separates one fold from the next."""

    x: np.ndarray

    def __post_init__(self) -> None:
        a = _vector(self.x, "fold labels")
        if not np.all(a >= 0) or not np.all(a == np.round(a)):
            raise ValueError("fold labels must be non-negative integers")
        object.__setattr__(self, "x", a.astype(int))

    def training(self, hold_out: int) -> np.ndarray:
        """Boolean mask that is True outside `hold_out`."""
        return self.x != int(hold_out)


@dataclass(frozen=True)
class Cutpoints:
    """Phase boundaries on the absolute porosity scale.

    Constructed from configuration, or from a rule that sees porosity only (see
    `zeroleak.strategies`).  This class offers no method that estimates
    boundaries from permeability, and none that estimates them from data at all:
    provenance is decided by the caller, and the caller is visible in the
    signature of whatever produced it.
    """

    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        v = tuple(float(t) for t in self.values)
        if len(v) < 1:
            raise ValueError("at least one cutpoint is required")
        if any(not (0.0 < t < 1.0) for t in v):
            raise ValueError("cutpoints must lie strictly inside (0, 1)")
        if any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("cutpoints must be strictly increasing")
        object.__setattr__(self, "values", v)

    @classmethod
    def from_config(cls, values) -> "Cutpoints":
        """The only constructor used in the pipeline: values come from a file
        written before the target was seen."""
        return cls(tuple(values))

    @property
    def log10(self) -> np.ndarray:
        return np.log10(np.asarray(self.values, dtype=float))

    @property
    def phases(self) -> int:
        return len(self.values) + 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Cutpoints(" + ", ".join(f"{t:.3f}" for t in self.values) + ")"


@dataclass(frozen=True)
class Coefficients:
    """One (intercept, slope) pair per phase, in log10 space."""

    intercept: np.ndarray
    slope: np.ndarray

    def __post_init__(self) -> None:
        a = _vector(self.intercept, "intercept")
        n = _vector(self.slope, "slope")
        if a.size != n.size:
            raise ValueError("intercept and slope must have the same length")
        object.__setattr__(self, "intercept", a)
        object.__setattr__(self, "slope", n)

    @property
    def phases(self) -> int:
        return int(self.slope.size)


@dataclass(frozen=True)
class Predictions(_Vector):
    """Predicted log10 permeability, one value per sample."""

    x: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _vector(self.x, "prediction"))


@dataclass(frozen=True)
class Truth(_Vector):
    """Measured log10 permeability.  A separate type from `Predictions` so that
    a metric function cannot be handed a model in place of the truth."""

    x: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _vector(self.x, "truth"))


@dataclass(frozen=True)
class Metrics:
    """Scalar scores.  Frozen, and carries no per-sample array, so a metric
    cannot be recomputed under other parameters from inside a metric object."""

    values: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def as_row(self) -> Dict[str, float]:
        return dict(self.values)


def requires(**spec: type) -> Any:
    """Enforce the layer contract at call time.

    `spec` maps a parameter name to the only type it may receive.  A mismatch
    raises `TypeError` naming both types and the information set the expected
    one carries.  A missing argument raises `TypeError` from `bind`, which is
    the second half of the claim in the manuscript.
    """

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            for name, expected in spec.items():
                if name not in bound.arguments:
                    continue
                value = bound.arguments[name]
                if value is None:
                    continue
                if not isinstance(value, expected):
                    raise TypeError(
                        f"{fn.__qualname__}(): '{name}' must be "
                        f"{expected.__name__}, got {type(value).__name__}. "
                        f"{expected.__name__} carries {CARRIES.get(expected.__name__, 'data')}; "
                        f"{type(value).__name__} carries {CARRIES.get(type(value).__name__, 'other data')}."
                    )
            return fn(*args, **kwargs)

        wrapper.__signature__ = signature  # keep the contract introspectable
        # The contract itself, so that a registry can read what a function is
        # allowed to see without parsing its body.  An empty dict is a real
        # declaration: the function takes no data carrier at all.
        wrapper.__requires__ = dict(spec)
        return wrapper

    return decorate
