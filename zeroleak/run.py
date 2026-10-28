"""The single command.

    python -m zeroleak.run --config configs/base.yaml

For each body and strategy it writes a CSV of per-sample predictions, a CSV of
summary metrics and a JSON record of the configuration, then a one-row-per-run
index.  Nothing else is produced, and every number in the index is recomputed
from the prediction CSV by the reporting layer, so an index that disagrees with
its own predictions is impossible without the files disagreeing too.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np

from . import __version__
from .config import BodySpec, Config, load
from .layers import data, model, report
from .layers import folds as folds_layer
from .layers import metrics
from .strategies import resolve
from .types import Predictions, Truth

MIN_TRAIN = 30  # a fold with fewer training rows is skipped, as in the paper


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """The commit the code was run from, marked `+dirty` if anything outside the
    output directory has changed since.  The output directory is excluded on
    purpose: a run rewrites it, so including it would mark every run dirty."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)expected", ":(exclude)out"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _write_predictions(path: str, body, folds, y: np.ndarray) -> None:
    truth = np.log10(body.perm.x)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "phi", "kh_md", "depth_m", "fold", "log10_pred", "log10_true", "residual"])
        for i in range(len(body)):
            w.writerow([
                i,
                f"{body.phi.x[i]:.6f}",
                f"{body.perm.x[i]:.6f}",
                f"{body.groups.x[i]:.3f}",
                int(folds.x[i]),
                f"{y[i]:.9f}",
                f"{truth[i]:.9f}",
                f"{truth[i] - y[i]:.9f}",
            ])


def _write_metrics(path: str, values: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "value"])
        for key in sorted(values):
            w.writerow([key, f"{values[key]:.9f}" if isinstance(values[key], float) else values[key]])


def _load_body(spec: BodySpec, cfg: Config):
    """One archive, one body.  The grid archive is read through `data.load_grid`
    because its two units must share one subsampling generator."""
    if spec.grid_unit:
        return data.load_grid(
            spec.file,
            spec.perm_file,
            spec.grid_unit,
            phi_min=cfg.phi_min,
            phi_max=cfg.phi_max,
            n_subsample=int(spec.n_subsample or data.N_SUBSAMPLE),
            seed=cfg.seed,
        )
    return data.load_csv(
        spec.file,
        spec.name,
        phi_col=spec.phi_col,
        perm_col=spec.perm_col,
        group_col=spec.group_col,
        group_kind=spec.group_kind,
        phi_scale=spec.phi_scale,
        phi_min=cfg.phi_min,
        phi_max=cfg.phi_max,
    )


def run_body(spec: BodySpec, cfg: Config, root: str, out_dir: str) -> List[Dict]:
    path = spec.file if os.path.isabs(spec.file) else os.path.join(root, spec.file)
    if path != spec.file:
        spec = dataclasses.replace(spec, file=path)
    body = _load_body(spec, cfg)

    rule = spec.fold_rule or cfg.fold_rule
    n_blocks = spec.n_blocks if spec.n_blocks is not None else cfg.n_blocks
    folds = folds_layer.assign_folds(
        body.phi, body.groups, rule=rule, n_blocks=n_blocks, seed=cfg.seed
    )
    truth = Truth(np.log10(body.perm.x))
    rows = []

    for strategy in cfg.strategies:
        y = np.full(len(body), np.nan)
        for f in np.unique(folds.x):
            keep = folds.training(f)
            if int(keep.sum()) < MIN_TRAIN or int((~keep).sum()) == 0:
                continue
            cuts = resolve(strategy, body.phi.take(keep), cfg)
            coefs = model.fit(body.phi, body.perm, folds, cuts, hold_out=int(f))
            y[~keep] = model.predict(coefs, body.phi.take(~keep), cuts).x
        if np.isnan(y).any():
            raise RuntimeError(f"{spec.name}/{strategy}: {int(np.isnan(y).sum())} rows were never predicted")
        cv = metrics.score(Predictions(y), truth)

        cuts_full = resolve(strategy, body.phi, cfg)
        full_coefs = model.fit(body.phi, body.perm, folds, cuts_full)
        full = metrics.score(model.predict(full_coefs, body.phi, cuts_full), truth)

        stem = os.path.join(out_dir, f"{spec.name}__{strategy}")
        _write_predictions(stem + ".pred.csv", body, folds, y)
        values = dict(cv.as_row())
        values.update({
            "r2_full": full["r2"],
            "rmse_sigma_full": full["rmse_sigma"],
            "cutpoints": " ".join(f"{t:.4f}" for t in cuts_full.values),
            "fold_rule": rule,
            "n_blocks": n_blocks,
            "seed": cfg.seed,
        })
        _write_metrics(stem + ".metrics.csv", values)

        record = {
            "body": spec.name,
            "strategy": strategy,
            "resolved_cutpoints": list(cuts_full.values),
            "config": cfg.as_dict(),
            "data_file": {"path": spec.file, "sha256": _sha256(path), "rows_used": len(body)},
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "zeroleak_git": _git_commit(),
            },
        }
        with open(stem + ".config.json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)

        rows.append({
            "body": spec.name,
            "strategy": strategy,
            "n": len(body),
            "r2": cv["r2"],
            "rmse_sigma": cv["rmse_sigma"],
            "r2_full": full["r2"],
        })
    return rows


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(prog="zeroleak", description="Run one porosity-only permeability pipeline.")
    ap.add_argument("--config", required=True, help="path to the YAML configuration")
    ap.add_argument("--root", default=".", help="directory that body paths are relative to")
    ap.add_argument("--out", default=None, help="override the configured output directory")
    ap.add_argument("--version", action="version", version=f"zeroleak {__version__}")
    args = ap.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"zeroleak: error: no configuration at {args.config!r}.\n"
              f"  Run from the repository root and pass one of the shipped files,\n"
              f"  --config configs/base.yaml (shipped subset) or configs/full.yaml.",
              file=sys.stderr)
        return 2
    try:
        cfg = load(args.config)
    except ValueError as exc:
        print(f"zeroleak: error: {exc}", file=sys.stderr)
        return 2

    # Check every input before the first one is opened, so that a missing archive
    # is reported as such and not as whatever the reader happened to call next.
    for spec in cfg.bodies:
        for role, rel in (("porosity", spec.file), ("permeability", spec.perm_file)):
            if not rel:
                continue
            full = rel if os.path.isabs(rel) else os.path.join(args.root, rel)
            if not os.path.exists(full):
                print(f"zeroleak: error: body {spec.name!r} needs its {role} file at\n"
                      f"  {full}\n"
                      f"  which does not exist.  DATA.md gives the source and the SHA-256 of\n"
                      f"  each archive; configs/base.yaml runs on the shipped subset alone.",
                      file=sys.stderr)
                return 2

    out_dir = os.path.join(args.root, args.out or cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for spec in cfg.bodies:
        rows.extend(run_body(spec, cfg, args.root, out_dir))

    # The index is rebuilt from the exported metric files, never from memory.
    exported = report.read_metrics(out_dir)
    summary = []
    for row in exported:
        summary.append({
            "body": row["body"],
            "strategy": row["strategy"],
            "n": row.get("n", ""),
            "r2": row.get("r2", ""),
            "rmse_sigma": row.get("rmse_sigma", ""),
            "r2_full": row.get("r2_full", ""),
        })
    report.write_summary(summary, os.path.join(out_dir, "index.csv"))

    width = max(len(r["body"]) for r in summary) + 2
    print(f"{'body':{width}s}{'strategy':>17s}{'n':>7s}{'R2 (held-out)':>15s}{'RMSE/sigma':>12s}")
    for r in summary:
        print(f"{r['body']:{width}s}{r['strategy']:>17s}{r['n']:>7s}{float(r['r2']):>15.4f}{float(r['rmse_sigma']):>12.4f}")
    print(f"\nwrote {len(summary)} runs to {os.path.abspath(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
