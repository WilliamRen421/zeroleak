"""One YAML file, read once, validated before anything is loaded.

Everything that decides a number lives here: the porosity window, the phase
count, the cutpoint rule and values, the fold rule, the random seed and the
output directory.  The resolved object is copied into the output directory
beside every run, so a table can be traced back to the exact configuration that
produced it.

A body may override the fold rule, because the five bodies are not folded the
same way: the grid is cut into contiguous columns, the cored well into depth
blocks, the multi-well bodies leave one well out.

A configuration that does not parse is rejected before any data are read, which
is the cheapest possible place for a mistake to surface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

from .layers.folds import RULES
from .strategies import STRATEGIES
from .types import Cutpoints

__all__ = ["BodySpec", "Config", "load"]


@dataclass(frozen=True)
class BodySpec:
    """One archive.  `file` is the porosity file; a grid body names its
    permeability file separately and picks one unit out of the archive."""

    name: str
    file: str
    phi_col: str = "phi"
    perm_col: str = "kh_md"
    group_col: str = "depth_m"
    group_kind: str = "numeric"
    phi_scale: float = 1.0
    fold_rule: Optional[str] = None
    n_blocks: Optional[int] = None
    perm_file: Optional[str] = None
    grid_unit: Optional[str] = None
    n_subsample: Optional[int] = None
    label: str = ""

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "file": self.file,
            "columns": {"phi": self.phi_col, "perm": self.perm_col, "group": self.group_col},
            "group_kind": self.group_kind,
            "phi_scale": self.phi_scale,
            "fold_rule": self.fold_rule,
            "n_blocks": self.n_blocks,
            "perm_file": self.perm_file,
            "grid_unit": self.grid_unit,
            "n_subsample": self.n_subsample,
            "label": self.label,
        }


@dataclass(frozen=True)
class Config:
    seed: int
    phases: int
    cutpoints: Cutpoints
    cutpoint_rule: str
    fold_rule: str
    n_blocks: int
    strategies: Tuple[str, ...]
    output_dir: str
    phi_min: float
    phi_max: float
    bodies: List[BodySpec] = field(default_factory=list)
    source: str = ""

    def as_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "phases": self.phases,
            "window": {"phi_min": self.phi_min, "phi_max": self.phi_max},
            "cutpoints": {"rule": self.cutpoint_rule, "values": list(self.cutpoints.values)},
            "folds": {"rule": self.fold_rule, "n_blocks": self.n_blocks},
            "strategies": list(self.strategies),
            "output_dir": self.output_dir,
            "bodies": [b.as_dict() for b in self.bodies],
        }


def _require(mapping: Dict, key: str, where: str):
    if key not in mapping:
        raise ValueError(f"configuration is missing '{key}' in {where}")
    return mapping[key]


def _path(value: str) -> str:
    """Expand $ZEROLEAK_DATA_DIR and friends, so that the full-data
    configurations can be checked in without absolute paths.

    An unexpanded placeholder is turned into an error here rather than being
    carried to `open()`, where it surfaces as a missing file with a brace in its
    name and reads like a bug in the pipeline instead of a missing export.
    """
    raw = str(value)
    out = os.path.expandvars(raw)
    if "${" in out:
        name = out.split("${", 1)[1].split("}", 1)[0] or "ZEROLEAK_DATA_DIR"
        raise ValueError(
            f"'{raw}' refers to ${{{name}}}, which is not set in this shell. "
            f"Export it (see DATA.md) or run configs/base.yaml, which needs no archives."
        )
    return out


def load(path: str) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the configuration must be a mapping")

    window = raw.get("window", {})
    cuts = _require(raw, "cutpoints", "the top level")
    folds = raw.get("folds", {})
    bodies_raw = _require(raw, "bodies", "the top level")

    values = tuple(_require(cuts, "values", "cutpoints"))
    rule = cuts.get("rule", "prior")
    if rule not in ("prior", "target_terciles"):
        raise ValueError(f"unknown cutpoint rule {rule!r}")

    phases = int(raw.get("phases", len(values) + 1))
    if len(values) + 1 != phases:
        raise ValueError(
            f"{len(values)} cutpoints define {len(values) + 1} phases but phases={phases}"
        )

    strategies = tuple(raw.get("strategies", ["prior"]))
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"unknown strategy {s!r}; choose from {STRATEGIES}")

    default_rule = folds.get("rule", "depth_block")
    for r in [default_rule] + [b.get("folds", {}).get("rule") for b in bodies_raw]:
        if r is not None and r not in RULES:
            raise ValueError(f"unknown fold rule {r!r}; choose from {RULES}")

    bodies = []
    for item in bodies_raw:
        cols = item.get("columns", {})
        body_folds = item.get("folds", {})
        spec = BodySpec(
            name=_require(item, "name", "a body entry"),
            file=_path(_require(item, "file", f"body {item.get('name')}")),
            phi_col=cols.get("phi", "phi"),
            perm_col=cols.get("perm", "kh_md"),
            group_col=cols.get("group", cols.get("depth", "depth_m")),
            group_kind=item.get("group_kind", "numeric"),
            phi_scale=float(item.get("phi_scale", 1.0)),
            fold_rule=body_folds.get("rule"),
            n_blocks=body_folds.get("n_blocks"),
            perm_file=_path(item["perm_file"]) if item.get("perm_file") else None,
            grid_unit=item.get("grid_unit"),
            n_subsample=item.get("n_subsample"),
            label=item.get("label", ""),
        )
        if spec.grid_unit and not spec.perm_file:
            raise ValueError(f"{spec.name}: a grid unit needs perm_file")
        bodies.append(spec)

    return Config(
        seed=int(raw.get("seed", 20261007)),
        phases=phases,
        cutpoints=Cutpoints.from_config(values),
        cutpoint_rule=rule,
        fold_rule=default_rule,
        n_blocks=int(folds.get("n_blocks", 5)),
        strategies=strategies,
        output_dir=str(raw.get("output_dir", "out")),
        phi_min=float(window.get("phi_min", 0.02)),
        phi_max=float(window.get("phi_max", 0.45)),
        bodies=bodies,
        source=os.path.abspath(path),
    )
