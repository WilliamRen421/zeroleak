"""Compare a fresh run against the shipped reference in `expected/`.

The reference has to be a file that a run cannot touch, otherwise "reproduce the
published number" degenerates into "overwrite the published number".  So this
command runs the configuration into a scratch directory and diffs that directory
against `expected/` cell by cell; nothing under `expected/` is opened for writing.

What is compared, and why each of the three:

* `*.pred.csv`   -- every per-sample value, so a fold reassignment or a changed
  coefficient shows up even when the aggregate R2 rounds to the same four digits.
* `*.metrics.csv` -- every statistic, including the ones no table prints.
* `*.config.json` -- only the decisive fields: seed, phase count, window,
  cutpoints, fold rule, strategies, and per-body column mapping / scaling.  The
  `environment.zeroleak_git` field and the resolved archive paths are excluded
  because they differ by construction: one records the commit that produced the
  reference, the other is an absolute path on this machine.

Exit status is 0 only if every file matches; otherwise 1, so the check can be the
last step of a release script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from typing import Dict, List, Tuple

from . import run as run_mod
from .config import load as load_config

PRED_SUFFIX = ".pred.csv"
CONFIG_SUFFIX = ".config.json"
METRIC_SUFFIX = ".metrics.csv"
# index.csv is the one-row-per-run summary, i.e. Table 4 itself; compared like a
# prediction table because both are a header followed by numeric rows.
TABLE_FILES = ("index.csv",)
COMPARED = (PRED_SUFFIX, METRIC_SUFFIX, CONFIG_SUFFIX) + TABLE_FILES

# The predictions are printed to nine decimals, so the quantisation floor is
# 5e-10; 1e-8 sits above it and far below anything a code change would do.
PRED_TOL = 1e-8
METRIC_TOL = 1e-9

DECISIVE_CONFIG = ("seed", "phases", "window", "cutpoints", "folds", "strategies")
DECISIVE_BODY = ("name", "columns", "phi_scale", "grid_unit", "n_subsample", "folds")


def _read_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0], rows[1:]


def _delta(a: str, b: str) -> Tuple[bool, float]:
    """Compare two cells.  Returns (equal, absolute difference)."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b, 0.0
    diff = abs(fa - fb)
    # rel_tol guards the large values, abs_tol the ones near zero (bias, residual).
    return math.isclose(fa, fb, rel_tol=METRIC_TOL, abs_tol=METRIC_TOL), diff


def compare_pred(exp_path: str, got_path: str) -> List[str]:
    e_head, e_rows = _read_csv(exp_path)
    g_head, g_rows = _read_csv(got_path)
    out: List[str] = []
    if e_head != g_head:
        out.append(f"columns {g_head} != reference {e_head}")
        return out
    if len(e_rows) != len(g_rows):
        out.append(f"{len(g_rows)} rows != reference {len(e_rows)}")
        return out

    for i, (er, gr) in enumerate(zip(e_rows, g_rows), start=1):
        for col, (a, b) in enumerate(zip(er, gr)):
            same, _ = _delta(a, b)
            if not same:
                out.append(f"row {i} column {e_head[col]!r}: {b} != reference {a}")
                if len(out) >= 5:
                    out.append("... further cells suppressed")
                    return out
    return out


def compare_metrics(exp_path: str, got_path: str) -> List[str]:
    e = {r[0]: r[1] for r in _read_csv(exp_path)[1] if len(r) == 2}
    g = {r[0]: r[1] for r in _read_csv(got_path)[1] if len(r) == 2}
    out: List[str] = []
    if set(e) != set(g):
        missing = sorted(set(e) - set(g))
        extra = sorted(set(g) - set(e))
        if missing:
            out.append(f"missing statistics: {missing}")
        if extra:
            out.append(f"unexpected statistics: {extra}")
        return out
    for key in e:
        same, diff = _delta(e[key], g[key])
        if not same:
            out.append(f"{key}: {g[key]} != reference {e[key]}")
    return out


def compare_config(exp_path: str, got_path: str) -> List[str]:
    with open(exp_path, encoding="utf-8") as fh:
        e = json.load(fh)
    with open(got_path, encoding="utf-8") as fh:
        g = json.load(fh)
    out: List[str] = []
    for key in ("body", "strategy", "resolved_cutpoints"):
        if e.get(key) != g.get(key):
            out.append(f"{key}: {g.get(key)} != reference {e.get(key)}")
    for key in DECISIVE_CONFIG:
        if e.get("config", {}).get(key) != g.get("config", {}).get(key):
            out.append(f"config.{key}: {g.get('config', {}).get(key)} != "
                       f"reference {e.get('config', {}).get(key)}")
    eb = {b["name"]: b for b in e.get("config", {}).get("bodies", []) if "name" in b}
    gb = {b["name"]: b for b in g.get("config", {}).get("bodies", []) if "name" in b}
    for name in sorted(set(eb) | set(gb)):
        if name not in eb or name not in gb:
            out.append(f"body {name!r} present on one side only")
            continue
        for key in DECISIVE_BODY:
            if eb[name].get(key) != gb[name].get(key):
                out.append(f"body {name}.{key}: {gb[name].get(key)} != "
                           f"reference {eb[name].get(key)}")
    return out


def compare_tree(expected: str, got: str) -> Tuple[int, List[str]]:
    if not os.path.isdir(expected):
        raise FileNotFoundError(f"no reference directory at {expected}")
    if not os.path.isdir(got):
        raise FileNotFoundError(f"no run output at {got}")

    exp_files = sorted(f for f in os.listdir(expected) if f.endswith(COMPARED))
    got_files = sorted(f for f in os.listdir(got) if f.endswith(COMPARED))

    out: List[str] = []
    if exp_files != got_files:
        out.append(f"file set differs: run wrote {got_files}, reference holds {exp_files}")

    failures = 0
    print(f"{'file':46s}  status  detail")
    for name in exp_files:
        if name not in got_files:
            print(f"{name:46s}  FAIL    not produced by the run")
            failures += 1
            continue
        e_path, g_path = os.path.join(expected, name), os.path.join(got, name)
        if name.endswith(CONFIG_SUFFIX):
            problems = compare_config(e_path, g_path)
        elif name.endswith(METRIC_SUFFIX):
            problems = compare_metrics(e_path, g_path)
        else:
            problems = compare_pred(e_path, g_path)
        if problems:
            failures += 1
            print(f"{name:46s}  FAIL    {problems[0]}")
            for extra in problems[1:]:
                print(f"{'':46s}          {extra}")
        else:
            print(f"{name:46s}  ok")
    return failures, out


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="zeroleak verify",
        description="Run a configuration into a scratch directory and diff it against expected/.",
    )
    ap.add_argument("--config", required=True, help="path to the YAML configuration")
    ap.add_argument("--expected", default=None,
                    help="reference directory (default: <repo>/expected)")
    ap.add_argument("--out", default=None,
                    help="scratch directory (default: <output_dir>/_verify, wiped first)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected = args.expected or os.path.join(here, "expected")

    if not os.path.exists(args.config):
        print(f"zeroleak: error: no configuration at {args.config!r}", file=sys.stderr)
        return 2
    try:
        resolved = load_config(args.config)
    except ValueError as exc:
        print(f"zeroleak: error: {exc}", file=sys.stderr)
        return 2

    scratch = args.out or os.path.join(here, resolved.output_dir, "_verify")
    if os.path.exists(scratch) and not args.keep:
        shutil.rmtree(scratch)
    os.makedirs(scratch, exist_ok=True)

    print(f"running {args.config} into {os.path.abspath(scratch)}")
    rc = run_mod.main(["--config", args.config, "--out", os.path.abspath(scratch)])
    if rc != 0:
        return rc
    print()

    failures, notes = compare_tree(expected, scratch)
    for note in notes:
        print(f"note: {note}")
    print()
    if failures:
        print(f"verify: {failures} file(s) differ from {expected}")
        return 1
    print(f"verify: all files match {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
