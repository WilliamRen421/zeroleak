"""Layer 1, data access.

The only module in the package that opens a measurement file apart from
`layers.report`, which reads back the CSV this pipeline has already written.
It returns one `Body` per archive, and a `Body` is never passed to another
layer: the caller hands on `body.phi`, `body.perm` or `body.groups`, which are
what the downstream signatures accept.

Two kinds of archive are supported.  A three-column CSV covers the cored wells.
The SPE 10 grid is stored as two flat `.dat` files whose sample order is decided
by the archive, so it is read here, reshaped, windowed and subsampled -- once,
with one advancing generator shared by its two units, because that is what makes
the subsample reproducible.
"""
from __future__ import annotations

import csv
import functools
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..types import Depth, Permeability, Porosity

__all__ = ["Body", "load_csv", "load_grid", "GRID_SHAPE", "GRID_UNITS"]

PHI_MIN, PHI_MAX = 0.02, 0.45
GRID_SHAPE = (85, 220, 60)  # nz, ny, nx -- SPE 10 Model 2
GRID_UNITS = {"Tarbert": (0, 34), "Upper Ness": (35, 84)}
N_SUBSAMPLE = 100_000


@dataclass(frozen=True)
class Body:
    """One measured body, split into the information sets it contains.

    `groups` is the coordinate folds are cut on: sample depth for a cored well,
    the x column for a grid body, the well code for a multi-well body.  It holds
    no permeability, which is the property the fold layer depends on.
    """

    name: str
    phi: Porosity
    perm: Permeability
    groups: Depth
    source: str = ""

    def __len__(self) -> int:
        return len(self.phi)


def load_csv(
    path: str,
    name: str,
    phi_col: str = "phi",
    perm_col: str = "kh_md",
    group_col: str = "depth_m",
    phi_scale: float = 1.0,
    phi_min: float = PHI_MIN,
    phi_max: float = PHI_MAX,
    group_kind: str = "numeric",
) -> Body:
    """Read a CSV and apply the porosity window.

    `phi_scale` divides the porosity column, which is 100 when the file stores
    percent.  `group_kind="categorical"` maps a well name to a sorted integer
    code, so that a leave-one-well-out rule sees codes and not names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}: no such file {path}")
    if group_kind not in ("numeric", "categorical"):
        raise ValueError(f"unknown group_kind {group_kind!r}")

    phi, perm, group = [], [], []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                p = float(row[phi_col]) / float(phi_scale)
                k = float(row[perm_col])
                g = row[group_col].strip() if group_kind == "categorical" else float(row[group_col])
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
            if phi_min <= p <= phi_max and k > 0.0:
                phi.append(p)
                perm.append(k)
                group.append(g)

    if not phi:
        raise ValueError(f"{name}: no rows survived the window {phi_min}-{phi_max}")

    if group_kind == "categorical":
        # Sorted codes, so that the label of a well does not depend on row order.
        _, group_values = np.unique(np.asarray(group, dtype=object), return_inverse=True)
        group_values = group_values.astype(float)
    else:
        group_values = np.asarray(group, dtype=float)

    return Body(
        name=name,
        phi=Porosity(phi),
        perm=Permeability(perm),
        groups=Depth(group_values),
        source=path,
    )


def _read_flat(path: str, limit: int) -> np.ndarray:
    values = []
    with open(path) as fh:
        for line in fh:
            values.extend(float(t) for t in line.split())
            if len(values) >= limit:
                return np.asarray(values[:limit], dtype=float)
    return np.asarray(values, dtype=float)


@functools.lru_cache(maxsize=8)
def _grid_units(
    phi_path: str,
    perm_path: str,
    phi_min: float,
    phi_max: float,
    n_subsample: int,
    seed: int,
) -> Tuple:
    """Read the grid once for every unit it contains.

    The generator is created once and advanced across the units in the order of
    `GRID_UNITS`, so the subsample of the second unit depends on the size of the
    first.  That is deliberate: it is how the published numbers were produced,
    and deriving independent seeds would silently change them.
    """
    nz, ny, nx = GRID_SHAPE
    nc = nz * ny * nx
    phi3 = _read_flat(phi_path, nc).reshape(nz, ny, nx)
    k3 = _read_flat(perm_path, nc).reshape(nz, ny, nx)
    rng = np.random.default_rng(int(seed))

    out = []
    for name, (z0, z1) in GRID_UNITS.items():
        p_block, k_block = phi3[z0 : z1 + 1], k3[z0 : z1 + 1]
        _, _, xx = np.meshgrid(
            np.arange(z0, z1 + 1), np.arange(ny), np.arange(nx), indexing="ij"
        )
        m = (p_block >= phi_min) & (p_block <= phi_max) & (k_block > 0)
        p = p_block[m].astype(float)
        k = k_block[m].astype(float)
        x = xx[m].astype(float)
        if n_subsample and p.size > n_subsample:
            keep = rng.choice(p.size, int(n_subsample), replace=False)
            p, k, x = p[keep], k[keep], x[keep]
        out.append(
            Body(
                name=name,
                phi=Porosity(p),
                perm=Permeability(k),
                groups=Depth(x),
                source=phi_path,
            )
        )
    return tuple(out)


def load_grid(
    phi_path: str,
    perm_path: str,
    name: str,
    phi_min: float = PHI_MIN,
    phi_max: float = PHI_MAX,
    n_subsample: int = N_SUBSAMPLE,
    seed: int = 20261007,
) -> Body:
    """Return one unit of the grid archive by name."""
    for body in _grid_units(phi_path, perm_path, phi_min, phi_max, n_subsample, seed):
        if body.name == name:
            return body
    raise KeyError(f"{name} is not a unit of {phi_path}; choose from {list(GRID_UNITS)}")


def unit_summary() -> Dict[str, Tuple[int, int]]:
    """The z-extent of each grid unit, for the record."""
    return dict(GRID_UNITS)
