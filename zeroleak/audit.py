"""The audit: the failure a type system cannot catch.

The five layers certify *what a function may see*.  They do not certify that the
held-out fold is worth scoring, and they cannot: a fold rule is a function of
porosity and depth, it returns `FoldLabels`, and the layer it feeds can no more
ask where a fold came from than it can ask for permeability.  A rule that splits
a cored well at random therefore passes every check in `tests/test_layers.py`
while putting a plug twenty-five centimetres from the sample it is asked to
predict.  The information set is clean; the evaluation is not.

This module exists to catch exactly that case, and it does so by measuring the
split rather than by declaring it.  For every sample it finds the distance to the
nearest sample in a different fold, and reports that distance in units of the
median spacing between adjacent samples:

  * a contiguous block rule leaves the nearest training sample tens of spacings
    away, because a whole interval is held out at once;
  * a random rule leaves it one spacing away for most samples, because the
    neighbouring plug is usually in the training set.

A rule whose median distance falls below three spacings is reported as
interpolation.  The threshold is a convention, and the number beside it is not:
on the bodies shipped here the two regimes are separated by more than a factor of
forty, so no reasonable threshold changes a verdict.

What the audit is not: it is not a proof that a blocked split is honest, only
that it is not interpolating.  A blocked split can still be optimistic if the
field is heterogeneous within a block, and nothing here detects that.  The claim
it supports is narrower than "leakage-free" and is stated as such in the paper.

    python -m zeroleak.audit --config configs/base.yaml
    python -m zeroleak.audit --config configs/base.yaml --json out/audit.json
    python -m zeroleak.audit --config configs/base.yaml --strict
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from .config import BodySpec, Config, load
from .layers import folds as folds_layer, metrics, model
from .layers.folds import RULES
from .run import MIN_TRAIN, _load_body
from .strategies import resolve
from .types import FoldLabels, Predictions, Truth

__all__ = ["geometry", "optimism", "AUDIT_RULES", "INTERPOLATION_SPACINGS"]

#: Rules the audit compares by default: one contiguous rule per geometry, and the
#: random rule that is the reason the audit exists.
AUDIT_RULES = ("depth_block", "index_block", "random_block")

#: Median distance to the nearest training sample, in sample spacings, below
#: which a split is reported as interpolation.
INTERPOLATION_SPACINGS = 3.0

#: Above this many rows the distance spectrum is measured on a random subsample,
#: because the full matrix is quadratic.  Volve needs 557.
MAX_ROWS_FULL = 4000


def _spectrum(g: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Distance from every sample to the nearest sample in another fold.

    `g` is the ordering coordinate the fold rule saw (depth, column index, well
    code); `labels` is what it returned.  A sample whose fold is the only fold is
    given an infinite distance, so a single-fold run cannot look blocked.
    """
    n = g.size
    order = np.argsort(g, kind="stable")
    gs, ls = g[order], labels[order]
    out = np.empty(n, dtype=float)
    for i in range(n):
        other = ls != ls[i]
        if not other.any():
            out[i] = np.inf
            continue
        out[i] = float(np.min(np.abs(gs[other] - gs[i])))
    # `out` is in sorted order; map back
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    return out[inv]


def geometry(
    g: np.ndarray,
    labels: np.ndarray,
    *,
    max_rows: int = MAX_ROWS_FULL,
    seed: int = 20261007,
) -> Dict[str, float]:
    """Summarise how far the held-out fold sits from the training set.

    Returns the median and ninetieth percentile of the nearest-training distance,
    the median spacing between adjacent samples, the ratio of the first to the
    third, and the two fractions the verdict is read from.
    """
    g = np.asarray(g, dtype=float)
    labels = np.asarray(labels)
    if g.size != labels.size:
        raise ValueError("the ordering vector and the fold labels differ in length")
    if g.size < 8:
        raise ValueError("the audit needs at least eight samples")

    spacing = float(np.median(np.diff(np.sort(g))))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("the ordering vector has no spacing; the audit is undefined")

    if g.size > max_rows:
        rng = np.random.default_rng(int(seed))
        keep = np.sort(rng.choice(g.size, size=max_rows, replace=False))
        gg, ll = g[keep], labels[keep]
    else:
        gg, ll = g, labels

    nn = _spectrum(gg, ll)
    finite = nn[np.isfinite(nn)]
    if finite.size == 0:
        raise ValueError("every sample is in the same fold, so there is nothing to audit")
    nn_p50 = float(np.median(finite))
    return {
        "n": int(g.size),
        "spacing": spacing,
        "nn_p50": nn_p50,
        "nn_p90": float(np.percentile(finite, 90)),
        "ratio_p50": nn_p50 / spacing,
        "within_1_spacing": float(np.mean(nn <= spacing)),
        "within_3_spacings": float(np.mean(nn <= 3.0 * spacing)),
        "verdict": "interpolation" if nn_p50 / spacing < INTERPOLATION_SPACINGS else "blocked",
    }


def _cv_r2(body, cfg: Config, folds: FoldLabels, strategy: str) -> float:
    """Held-out R2 under one fold rule and one cutpoint strategy.

    Mirrors the loop in `run.run_body`; `tests/test_audit.py` pins the two to the
    same number on the shipped body, so a change here that is not made there
    fails a test rather than a table.
    """
    truth = Truth(np.log10(body.perm.x))
    y = np.full(len(body), np.nan)
    for k in np.unique(folds.x):
        keep = folds.training(k)
        if int(keep.sum()) < MIN_TRAIN or int((~keep).sum()) == 0:
            continue
        cuts = resolve(strategy, body.phi.take(keep), cfg)
        coefs = model.fit(body.phi, body.perm, folds, cuts, hold_out=int(k))
        y[~keep] = model.predict(coefs, body.phi.take(~keep), cuts).x
    if np.isnan(y).any():
        raise RuntimeError(
            f"{strategy}: {int(np.isnan(y).sum())} rows were never predicted under this rule"
        )
    return metrics.score(Predictions(y), truth)["r2"]


def optimism(
    spec: BodySpec,
    cfg: Config,
    *,
    rules: Optional[tuple] = None,
    root: str = ".",
    n_seeds: int = 1,
) -> List[Dict]:
    """One row per fold rule: accuracy, cutpoint difference and fold geometry.

    The cutpoint difference is prior minus target terciles, the sign convention of
    the main table, and it is the number the audit is protecting: a rule that
    interpolates changes it.

    `n_seeds > 1` repeats every rule over that many seeds and adds the range of
    both quantities.  Contiguous rules are invariant to the seed --- it only
    relabels their folds --- so a range that opens up is a property of the rule,
    which is exactly the property a reader needs to see.
    """
    rules = tuple(rules or AUDIT_RULES)
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        raise ValueError(f"unknown fold rule(s) {unknown}; choose from {RULES}")

    path = spec.file if os.path.isabs(spec.file) else os.path.join(root, spec.file)
    if path != spec.file:
        spec = dataclasses.replace(spec, file=path)
    body = _load_body(spec, cfg)
    g = np.asarray(body.groups.x, dtype=float)
    n_blocks = spec.n_blocks if spec.n_blocks is not None else cfg.n_blocks

    seeds = [int(cfg.seed)] + [int(s) for s in range(1, n_seeds)]
    rows = []
    for rule in rules:
        r2_by_seed, delta_by_seed = [], []
        for seed in seeds[:max(1, n_seeds)]:
            folds = folds_layer.assign_folds(
                body.phi, body.groups, rule=rule, n_blocks=n_blocks, seed=seed
            )
            r2 = {s: _cv_r2(body, cfg, folds, s) for s in cfg.strategies}
            if "prior" not in r2 or "target_terciles" not in r2:
                raise ValueError("the audit compares 'prior' with 'target_terciles'")
            r2_by_seed.append(r2)
            delta_by_seed.append(r2["prior"] - r2["target_terciles"])
        row = {
            "body": spec.name,
            "rule": rule,
            "n_blocks": int(n_blocks),
            **geometry(g, folds.x),
            "r2": r2_by_seed[0],
            "delta": delta_by_seed[0],
            "n_seeds": len(seeds[:max(1, n_seeds)]),
            "delta_range": [float(min(delta_by_seed)), float(max(delta_by_seed))],
            "r2_prior_range": [
                float(min(r["prior"] for r in r2_by_seed)),
                float(max(r["prior"] for r in r2_by_seed)),
            ],
        }
        rows.append(row)
    return rows


def _table(rows: List[Dict]) -> str:
    multi = rows[0].get("n_seeds", 1) > 1
    head = (
        f"{'rule':<14s}{'nn p50':>9s}{'x spacing':>11s}{'within 1':>10s}"
        f"{'R2 prior':>10s}{'R2 terc':>9s}"
        f"{'delta' if not multi else 'delta (seed range)':>24s}   verdict"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        if multi:
            lo, hi = r["delta_range"]
            d = f"  {r['delta']:+.4f}  [{lo:+.4f},{hi:+.4f}]"
        else:
            d = f"  {r['delta']:+.4f}"
        lines.append(
            f"{r['rule']:<14s}{r['nn_p50']:>8.3f}m{r['ratio_p50']:>11.1f}"
            f"{r['within_1_spacing']:>10.2f}{r['r2']['prior']:>10.4f}"
            f"{r['r2']['target_terciles']:>9.4f}{d:>24s}   {r['verdict']}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="zeroleak.audit",
        description="Measure whether a type-legal fold rule interpolates the held-out fold.",
    )
    ap.add_argument("--config", required=True, help="the same YAML a run would use")
    ap.add_argument("--root", default=".", help="directory that body paths are relative to")
    ap.add_argument("--body", default=None, help="body name; default is every body in the config")
    ap.add_argument("--rules", default=None, help=f"comma-separated; default {','.join(AUDIT_RULES)}")
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat every rule over this many seeds and report the range")
    ap.add_argument("--json", default=None, help="write the rows to this file")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any rule interpolates")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    specs = [b for b in cfg.bodies if args.body is None or b.name == args.body]
    if not specs:
        ap.error(f"no body named {args.body!r} in {args.config}")
    rules = tuple(r.strip() for r in args.rules.split(",")) if args.rules else None

    all_rows: List[Dict] = []
    flagged = 0
    for spec in specs:
        rows = optimism(spec, cfg, rules=rules, root=args.root, n_seeds=args.seeds)
        all_rows.extend(rows)
        print(f"\n{spec.name}, {rows[0]['n']} samples, spacing {rows[0]['spacing']:.3f}, "
              f"{rows[0]['n_blocks']} folds")
        print(_table(rows))
        flagged += sum(1 for r in rows if r["verdict"] == "interpolation")

    if flagged:
        print(
            f"\n{flagged} rule(s) above interpolate: they return FoldLabels and never read "
            f"permeability, so every type check in the package admits them.  The accuracy and "
            f"the cutpoint difference they produce are not transfer numbers."
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(all_rows, fh, indent=2)
        print(f"wrote {len(all_rows)} rows to {args.json}")
    return 1 if (args.strict and flagged) else 0


if __name__ == "__main__":
    sys.exit(main())
