# -*- coding: utf-8 -*-
"""Multi-seed check: how far does the loss at one pair move when the arbitrary
choices are redrawn?

Two choices in this design are arbitrary and neither is a property of the rock: which
subsample of a gridded body is scored, and where the fold boundaries fall.  The paper
quotes a standard deviation for that movement, so it has to be a number a reader can
recompute.  This command recomputes it at one pair -- the pair the paper reports, by
default -- and nowhere else, because a grid of standard deviations answers a question
nobody asked.

The design, fixed before the data are read:

  * twenty draws by default.  Draw *i* takes subsample seed ``cfg.seed + i`` and fold
    offset *i*, so every draw redraws both of the arbitrary choices at once;
  * the offsets span one full period of the fold partition.  The grid is cut into
    five x slabs, so shifting the boundaries by 60 cells returns the same partition
    and the window is 1 to 59 cells.  The cored well is cut into five depth blocks,
    so the period is one fifth of the cored interval and the window is 0.02 to 0.21
    of it;
  * the loss is recomputed from scratch each draw, i.e. the body's own terciles are
    re-estimated inside the moved folds, so the reference moves with the candidate;
  * ``sd`` is the sample standard deviation (``ddof=1``) of those draws.

A window of one period is what makes twenty draws enough.  Shorter and the step
function is sampled on part of its range; much longer and the offsets revisit the
same partitions, so the spread measures the spacing of the offsets rather than the
data -- twenty offsets spread over five periods hit four distinct partitions on the
cored well and read 0.0024 or 0.0029 depending only on the phase.

A body folded by well -- never one core plug per well -- has neither knob: the folds
are the wells, and the wells do not move.  Such a body is reported as fixed rather
than as a standard deviation of zero, because zero would read as "we checked and the
number does not move", which is a stronger claim than the one the data support.  The
check therefore speaks for the bodies where the fold is a free choice, and the report
says so.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from . import config as _config
from .cv import BODY_NAMES, BODY_ORDER, cv_pred, load_bodies, r2_log

__all__ = ["main", "draws_of"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "configs", "full.yaml")
DEFAULT_DRAWS = 20

#: One full period of each fold partition, as (lowest offset, span).  The grid folds
#: into five slabs of 12 columns, so a shift of 60 cells is the identity; the cored
#: well folds into five depth blocks, so a shift of one fifth of the interval is.
GRID_CELLS = (1, 58)
DEPTH_FRAC = (0.02, 0.19)


def draws_of(draws, seed0):
    """The draw schedule: ``(draw, seed, grid offset in cells, depth offset)``."""
    if draws < 2:
        raise ValueError("a standard deviation needs at least two draws")
    out = []
    for i in range(draws):
        share = i / (draws - 1)
        cells = GRID_CELLS[0] + round(share * GRID_CELLS[1])
        frac = DEPTH_FRAC[0] + share * DEPTH_FRAC[1]
        out.append((i, seed0 + i, cells, frac))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="zeroleak noise",
        description="Standard deviation of the loss at one cutpoint pair across "
                    "redrawn subsamples and fold offsets.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="configuration naming the bodies (default: configs/full.yaml)")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help="number of draws (default: %d, the count the paper reports)"
                         % DEFAULT_DRAWS)
    ap.add_argument("--phi1", type=float, default=0.10, help="first cutpoint (default: 0.10)")
    ap.add_argument("--phi2", type=float, default=0.20, help="second cutpoint (default: 0.20)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <output_dir>/noise of the configuration)")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.draws < 2:
        raise ValueError("a standard deviation needs at least two draws")
    if not args.phi1 < args.phi2:
        raise ValueError("the first cutpoint must be below the second")

    cfg = _config.load(args.config)
    pair = (args.phi1, args.phi2)
    OUT = os.path.abspath(args.out or os.path.join(cfg.output_dir, "noise"))
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    rep = open(os.path.join(OUT, "noise_report.txt"), "w", encoding="utf-8")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, file=rep)
        print(s)

    def loss_at(d):
        """L = own terciles minus the named pair, on one loaded body."""
        lk = np.log10(d["k"])
        own = r2_log(lk, cv_pred(d["phi"], d["k"], d["fold"], "own", cfg.phases))
        ab = r2_log(lk, cv_pred(d["phi"], d["k"], d["fold"], pair, cfg.phases))
        return own - ab

    # the configuration `scan` reports: base seed, no offset.  Not one of the draws,
    # which all move, so it is computed separately and used as the anchor the draws
    # have to bracket.
    D0 = load_bodies(cfg, cfg.seed, 0.0)
    names = [nm for nm in BODY_ORDER if nm in D0]
    anchor = {nm: loss_at(D0[nm]) for nm in names}
    caps = {BODY_NAMES.get(b.name, b.name): int(b.n_subsample or 0)
            for b in cfg.bodies if b.n_subsample}
    prov = {nm: dict(well_folded=bool(D0[nm]["group"] is not None),
                     subsamples=bool(caps.get(nm) and D0[nm]["phi"].size >= caps[nm]))
            for nm in names}

    sched = draws_of(args.draws, cfg.seed)
    L = {nm: [] for nm in names}

    P("=" * 92)
    P("Multi-seed check: the loss at one cutpoint pair, redrawn %d times" % args.draws)
    P("pair phi1=%.3f phi2=%.3f | K=%d | loss L = R2_own - R2_abs (positive = abs worse)"
      % (pair[0], pair[1], cfg.phases))
    P("draw i: subsample seed %d+i | grid boundaries shifted over 1..59 cells | "
      % cfg.seed)
    P("        depth boundaries shifted over 0.02..0.21 of the cored interval,")
    P("        one full period of each fold partition")
    P("sd is the sample standard deviation (ddof=1) of the %d draws" % args.draws)
    P("=" * 92)
    P("")

    for i, seed, cells, frac in sched:
        D = load_bodies(cfg, seed, {"grid_cells": cells, "depth_frac": frac})
        for nm in names:
            L[nm].append(loss_at(D[nm]))
        P(f"  draw {i:3d}  seed {seed}  grid +{cells:2d} cells  depth +{frac:.3f}  " +
          " ".join(f"{nm}={L[nm][-1]:+.6f}" for nm in names))

    P("")
    P("=" * 92)
    P("[1] which knob each body has")
    P(f"  {'body':9s} {'n':>7s} {'folds':>6s} {'subsample':>10s} {'fold offset':>12s}  note")
    for nm in names:
        d = D0[nm]
        note = ""
        if prov[nm]["well_folded"]:
            note = "folded by well: leave one well out, nothing to redraw"
        elif not prov[nm]["subsamples"]:
            note = "no subsample: the whole body is scored at every draw"
        P(f"  {nm:9s} {d['phi'].size:7d} {len(np.unique(d['fold'])):6d} "
          f"{('yes' if prov[nm]['subsamples'] else 'no'):>10s} "
          f"{('no' if prov[nm]['well_folded'] else 'yes'):>12s}  {note}")
    P("")

    P("=" * 92)
    P("[2] spread of the loss across the draws")
    P(f"  {'body':9s} {'mean':>11s} {'sd':>10s} {'min':>11s} {'max':>11s} {'range':>10s} "
      f"{'distinct':>9s}")
    stats = {}
    for nm in names:
        a = np.array(L[nm], dtype=float)
        nd = len(np.unique(np.round(a, 12)))
        if nd < 2:
            stats[nm] = dict(mean=float(a.mean()), sd=None, min=float(a.min()),
                             max=float(a.max()), range=0.0, n_distinct=nd,
                             note="fixed: no subsample and no fold offset available")
            P(f"  {nm:9s} {a.mean():+11.6f} {'fixed':>10s} {a.min():+11.6f} "
              f"{a.max():+11.6f} {0.0:10.6f} {nd:9d}")
            continue
        sd = float(a.std(ddof=1))
        stats[nm] = dict(mean=float(a.mean()), sd=sd, min=float(a.min()), max=float(a.max()),
                         range=float(a.max() - a.min()), n_distinct=nd, note="")
        P(f"  {nm:9s} {a.mean():+11.6f} {sd:10.6f} {a.min():+11.6f} {a.max():+11.6f} "
          f"{a.max()-a.min():10.6f} {nd:9d}")
    P("")

    quoted = [nm for nm in names if stats[nm]["sd"] is not None]
    P("  quoted as one standard deviation: " +
      ", ".join(f"{stats[nm]['sd']:.4f} on {nm}" for nm in quoted))
    P("  Bodies reported as fixed carry no redrawable choice: their folds are the wells.")
    P("  The check therefore speaks for %d of the %d bodies, and a claim about the other "
      % (len(quoted), len(names)))
    P("  bodies would need a different design, not a smaller number.")
    P("")

    P("=" * 92)
    P("[3] anchor: the single-seed evaluation `scan` reports at this pair")
    P("  (seed %d, no offset; not one of the draws, which all move)" % cfg.seed)
    for nm in names:
        a = np.array(L[nm], dtype=float)
        lo, hi = float(a.min()), float(a.max())
        inside = lo - 1e-12 <= anchor[nm] <= hi + 1e-12
        P(f"  {nm:9s} anchor {anchor[nm]:+.6f}   draws [{lo:+.6f}, {hi:+.6f}]   "
          f"{'inside' if inside else 'OUTSIDE'}")
    P("  A draw set whose range excludes the single-seed value would mean the two ")
    P("  commands are not scoring the same thing.")
    P("")
    P(f"total {time.time()-t0:.1f}s")
    rep.close()

    json.dump(dict(
        meta=dict(pair=list(pair), draws=args.draws, seed0=cfg.seed,
                  schedule=[dict(draw=i, seed=s, grid_cells=c, depth_frac=f)
                            for i, s, c, f in sched],
                  window=dict(grid_cells=list(GRID_CELLS), depth_frac=list(DEPTH_FRAC)),
                  config=cfg.source, sd="sample standard deviation of the draws (ddof=1)",
                  loss="L = R2_own - R2_abs"),
        anchor=anchor,
        bodies={nm: dict(stats[nm], values=[float(v) for v in L[nm]],
                         foldable=not prov[nm]["well_folded"],
                         subsampled=prov[nm]["subsamples"]) for nm in names},
    ), open(os.path.join(OUT, "noise_results.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1)

    with open(os.path.join(OUT, "noise.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["draw", "seed", "grid_cells", "depth_frac", "body", "loss"])
        for (i, seed, cells, frac) in sched:
            for nm in names:
                w.writerow([i, seed, cells, f"{frac:.6f}", nm, f"{L[nm][i]:.6f}"])

    print("written: noise_report.txt / noise_results.json / noise.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
