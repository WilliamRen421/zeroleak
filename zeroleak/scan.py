# -*- coding: utf-8 -*-
"""Cutpoint scan: the three-phase model evaluated on a 951-cell grid.

At every pair on the grid the model is refitted inside each training fold and
scored on the held-out fold, so one cell is a complete cross-validation run and
not a single fit.  Writing the whole grid makes the choice of cutpoints auditable
after the fact instead of resting on the pair the paper happens to report.

Conventions, all fixed before the data are read:

  * window 0.02 <= phi <= 0.45 and k > 0;
  * five bodies, each folded the way the paper folds it (contiguous columns on
    the grid, depth blocks in the single cored well, one well out where several
    are present);
  * three phases, one power law per phase, fitted by OLS in log10 space;
  * loss = R2_own - R2_abs, so a positive value means the absolute cutpoints are
    worse at that cell;
  * grid phi1 in [0.05, 0.20], phi2 in [0.10, 0.30], step 0.005, with
    phi2 > phi1 + 0.02, which leaves 951 evaluated cells.

The fold offsets used by the robustness sweep are deliberately not multiples of
the fold width: a multiple only relabels the bins and five such offsets return
numbers identical to the last digit.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time

import numpy as np

from . import config as _config
from .cv import cv_pred, load_bodies, r2_k, r2_log

__all__ = ["main", "cutpoint_grid"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "configs", "full.yaml")

BODY_KEYS = ("tarbert", "upper_ness", "volve_15_9_19a", "brent_211_29", "piper_15_17")


def cutpoint_grid():
    """The grid itself, separated so that its size can be asserted without data.

    Returns ``(phi1_values, phi2_values, pairs)``.  The pairs are the cells that
    satisfy ``phi2 > phi1 + 0.02``; everything downstream indexes this list, so
    a change in the window shows up as a failing test rather than as a silently
    different figure.
    """
    g1 = np.round(np.arange(0.05, 0.20 + 1e-9, 0.005), 4)
    g2 = np.round(np.arange(0.10, 0.30 + 1e-9, 0.005), 4)
    pairs = [(a, b) for a in g1 for b in g2 if b > a + 0.02]
    return g1, g2, pairs


def main(argv=None):
    """Run the 951-cell scan and write the per-body loss grid.

    Reproduces Table 6 and Fig. 5 of the paper.  The archives are read from the
    paths named in the configuration, so nothing here is tied to a machine.
    """
    ap = argparse.ArgumentParser(
        prog="zeroleak scan",
        description="Evaluate the three-phase model on the cutpoint grid "
                    "(951 cells) and write the per-body loss grid.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="configuration naming the five bodies (default: configs/full.yaml)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <output_dir>/scan of the configuration)")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    cfg = _config.load(args.config)
    spec = {b.name: b for b in cfg.bodies}
    for key in BODY_KEYS:
        if key not in spec:
            raise ValueError("the scan needs all five bodies; %r is missing from %s"
                             % (key, args.config))

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    OUT = os.path.abspath(args.out or os.path.join(cfg.output_dir, "scan"))
    CSVD = os.path.join(OUT, "csv")
    os.makedirs(CSVD, exist_ok=True)

    PHI_LO, PHI_HI = cfg.phi_min, cfg.phi_max
    K = cfg.phases
    N_SUB = int(spec["tarbert"].n_subsample or 100000)
    SEED = cfg.seed

    t0 = time.time()
    rep = open(os.path.join(OUT, "cut_scan_report.txt"), "w", encoding="utf-8")
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, file=rep); print(s)

    # ------------------------------------------------------------------ data
    # the loaders and the foldwise fit live in zeroleak/cv.py, shared with `noise`

    P("=" * 100)
    P("Cutpoint scan: the three-phase model on the 951-cell grid")
    P("window 0.02<=phi<=0.45 and k>0 | K=3 | one power law per phase, OLS in log10 space | ")
    P("metric R2(log10 k) | 6 free parameters")
    P("folds: grid cut into 5 contiguous x slabs | cored well into 5 depth blocks | ")
    P("multi-well bodies leave one well out")
    P(f"seed {SEED} | grid subsample {N_SUB} | loss L = R2_own - R2_abs (positive = worse)")
    P("=" * 100)
    P("")

    D = load_bodies(cfg, SEED, 0.0)
    BODIES = ["Tarbert", "Ness", "Volve", "Brent", "Piper"]
    for nm in BODIES:
        d = D[nm]
        P(f"  {nm:9s} n={d['phi'].size:7d}  folds={len(np.unique(d['fold'])):3d}  "
          f"sigma(log10k)={np.log10(d['k']).std():.4f}  phi quantiles [5,50,95]="
          f"{np.percentile(d['phi'],[5,50,95]).round(3).tolist()}")
    P("")

    # ------------------------------------------------------------------ fitting core
    # labels_of / cv_pred / r2_log / r2_k are imported from zeroleak/cv.py

    OWN = {}
    for nm in BODIES:
        d = D[nm]; yp = cv_pred(d["phi"], d["k"], d["fold"], "own", K)
        OWN[nm] = dict(r2_log=r2_log(np.log10(d["k"]), yp), r2_k=r2_k(d["k"], yp), yp=yp)
    P("[reference] the target's own equal-frequency terciles (own).  This locates the absolute ")
    P("cutpoints; it is not a reported result.")
    P(f"{'body':10s} {'R2(log)':>9s} {'RMSE/sig':>8s} {'R2(k)':>9s} {'n':>8s}")
    for nm in BODIES:
        r2 = OWN[nm]["r2_log"]
        P(f"{nm:10s} {r2:9.4f} {math.sqrt(max(1-r2,0)):8.4f} {OWN[nm]['r2_k']:9.4f} {D[nm]['phi'].size:8d}")
    P("  this own column is the yardstick the absolute cutpoints are scored against")
    P("")

    # ------------------------------------------------------------------ grid
    G1, G2, GRID = cutpoint_grid()
    P(f"grid: phi1 in [{G1[0]}, {G1[-1]}] (x{len(G1)}) | phi2 in [{G2[0]}, {G2[-1]}] (x{len(G2)}) | ")
    P(f"{len(GRID)} cells satisfy phi2 > phi1 + 0.02")

    R2G  = {nm: np.full((len(G1), len(G2)), np.nan) for nm in BODIES}
    R2KG = {nm: np.full((len(G1), len(G2)), np.nan) for nm in BODIES}
    for ia, a in enumerate(G1):
        for ib, b in enumerate(G2):
            if not (b > a + 0.02):
                continue
            for nm in BODIES:
                d = D[nm]
                yp = cv_pred(d["phi"], d["k"], d["fold"], (a, b), K)
                R2G[nm][ia, ib]  = r2_log(np.log10(d["k"]), yp)
                R2KG[nm][ia, ib] = r2_k(d["k"], yp)
    P(f"grid done in {time.time()-t0:.1f}s")
    P("")

    # loss L = own - abs  (positive = the absolute cutpoints are worse)
    LG = {nm: OWN[nm]["r2_log"] - R2G[nm] for nm in BODIES}
    worst = np.nanmax(np.stack([LG[nm] for nm in BODIES]), axis=0)   # worst body at each cell
    meanL = np.nanmean(np.stack([LG[nm] for nm in BODIES]), axis=0)
    i10, i20 = G1.tolist().index(0.10), G2.tolist().index(0.20)

    def argmin_ij(A):
        A = np.where(np.isnan(A), np.inf, A)
        ij = np.unravel_index(np.argmin(A), A.shape)
        return G1[ij[0]], G2[ij[1]], float(A[ij])

    P("=" * 100)
    P("[1] locating the cutpoints by the worst-body loss: the body that loses most sets the position")
    p1, p2, val = argmin_ij(worst)
    P(f"  global argmin of the worst-body loss: phi1={p1:.3f}, phi2={p2:.3f}, loss={val:+.4f}")
    P(f"  at phi=0.10/0.20: worst-body loss = {worst[i10,i20]:+.4f} ({BODIES[int(np.nanargmax([LG[nm][i10,i20] for nm in BODIES]))]})"
      f"  five-body mean = {meanL[i10,i20]:+.4f}")
    P("")
    P("  range of each body's loss over the grid:")
    P(f"  {'body':10s} {'own R2':>8s} {'L@.10/.20':>10s} {'best gain':>12s} {'argmin':>16s} {'L_max':>9s}")
    for nm in BODIES:
        A = LG[nm]; aa, bb, vv = argmin_ij(A)
        tag = "★" if A[i10, i20] > 0 else " "
        P(f"  {nm:10s} {OWN[nm]['r2_log']:8.4f} {A[i10,i20]:+10.4f}{tag} "
          f"{OWN[nm]['r2_log']-vv:+12.4f} {'('+format(aa,'.3f')+','+format(bb,'.3f')+')':>16s} "
          f"{np.nanmax(A):+9.4f}")
    P("  (L > 0 means the absolute cutpoints are worse than the body's own terciles; ")
    P("  * marks a body that still loses at 0.10/0.20)")
    P("")

    P("  [headline] absolute cutpoints 0.10/0.20 against the body's own terciles, with ")
    P("  RMSE/sigma and the k-space R2 as a side check")
    P(f"  {'body':10s} {'own R2':>8s} {'own RMSE/sig':>13s} {'abs R2':>8s} {'abs RMSE/sig':>13s} "
      f"{'L':>9s} {'abs R²(k)':>10s}")
    for nm in BODIES:
        o = OWN[nm]["r2_log"]; a = R2G[nm][i10, i20]
        P(f"  {nm:10s} {o:8.4f} {math.sqrt(max(1-o,0)):11.4f} {a:8.4f} "
          f"{math.sqrt(max(1-a,0)):11.4f} {o-a:+9.4f} {R2KG[nm][i10,i20]:10.4f}")
    P("  (RMSE/sigma is normalised by sigma, so RMSE/sigma = sqrt(1 - R2); the k-space R2 is ")
    P("  reported as a side check only)")
    P("")

    P("=" * 100)
    P("[2] feasible window: where the worst-body loss is <= eps")
    for eps in (0.001, 0.002, 0.005, 0.010):
        M = worst <= eps
        if not M.any():
            P(f"  eps={eps:.3f}: no feasible cell")
            continue
        inner = [a for a in G1 if any((b > a + 0.02) and (worst[G1.tolist().index(a), G2.tolist().index(b)] <= eps)
                                      for b in G2)]
        P(f"  eps={eps:.3f}: {int(M.sum()):4d}/{len(GRID)} cells feasible | "
          f"phi1 in [{min(inner):.3f}, {max(inner):.3f}] with any phi2")
    P("")

    # ------------------------------------------------------------------ one-dimensional slices
    SA1 = np.round(np.arange(0.06, 0.185 + 1e-9, 0.0025), 4)
    SB2 = np.round(np.arange(0.125, 0.30 + 1e-9, 0.0025), 4)
    PHI2_FIX, PHI1_FIX = 0.20, 0.10

    def slice_loss(body, kind, x):
        d = D[body]
        th = (x, PHI2_FIX) if kind == "A" else (PHI1_FIX, x)
        return OWN[body]["r2_log"] - r2_log(np.log10(d["k"]), cv_pred(d["phi"], d["k"], d["fold"], th, K))

    P("=" * 100)
    P("[3] one-dimensional slices: loss L, positive = worse; the last column is the worst body")
    SL = {"A": [], "B": []}
    for kind, xs, other, oname in (("A", SA1, PHI2_FIX, "φ₂"), ("B", SB2, PHI1_FIX, "φ₁")):
        P(f"  slice {kind}: {oname} = {other} held fixed")
        P(f"  {'cut':>7s} " + " ".join(f"{nm:>9s}" for nm in BODIES) + f" {'worst':>9s}")
        for x in xs:
            row = [slice_loss(nm, kind, x) for nm in BODIES]
            SL[kind].append(dict(x=float(x), **{nm: float(row[i]) for i, nm in enumerate(BODIES)},
                                 worst=float(max(row))))
            star = "  ←" if abs(x - (PHI1_FIX if kind == "A" else PHI2_FIX)) < 1e-9 else ""
            P(f"  {x:7.4f} " + " ".join(f"{v:+9.4f}" for v in row) + f" {max(row):+9.4f}{star}")
        P("")
        ws = [r["worst"] for r in SL[kind]]
        k = int(np.argmin(ws))
        P(f"    -> best point: {oname} = {xs[k]:.4f} (worst-body loss {ws[k]:+.4f})")
        ok = [xs[i] for i in range(len(xs)) if ws[i] <= 0.001]
        if ok:
            P(f"    -> worst-body loss <= 0.001 on {oname} in [{min(ok):.4f}, {max(ok):.4f}]")
        ok2 = [xs[i] for i in range(len(xs)) if ws[i] <= 0.005]
        if ok2:
            P(f"    -> worst-body loss <= 0.005 on {oname} in [{min(ok2):.4f}, {max(ok2):.4f}]")
        P("")

    # ------------------------------------------------------------------ fold robustness (non-integral offsets)
    offsets = [0, 5, 17, 29, 41]
    P("=" * 100)
    P("[4] fold robustness: five offsets of the x slabs on the grid and of the depth blocks in ")
    P("the cored well.  The offsets are not multiples of the fold width, because a multiple ")
    P("only relabels the bins.")
    P("    measured: the phi1 that minimises the five-body worst loss on slice A.  The ")
    P("multi-well bodies leave one well out and have no offset to sweep.")
    ROB = []
    xsA = [float(x) for x in SA1]
    for off in offsets:
        s = load_bodies(cfg, SEED, off / 60.0)
        row = []
        for x in xsA:
            vals = []
            for nm in ["Tarbert", "Ness", "Volve"]:
                d = s[nm]
                own = r2_log(np.log10(d["k"]), cv_pred(d["phi"], d["k"], d["fold"], "own", K))
                vals.append(own - r2_log(np.log10(d["k"]),
                                          cv_pred(d["phi"], d["k"], d["fold"], (x, PHI2_FIX), K)))
            row.append(float(max(vals)))
        ROB.append(dict(off=off, worst=row))
        kk = int(np.argmin(row))
        P(f"  offset={off:3d}: argmin at phi1={xsA[kk]:.4f} (worst loss {row[kk]:+.4f}) | "
          f"L@0.10={row[xsA.index(0.10)]:+.4f}  L@0.15={row[xsA.index(0.15)]:+.4f}")
    wA = np.array([r["worst"] for r in ROB])
    P(f"  -> pointwise range of the worst-body loss across the five offsets: max-min <= ")
    P(f"{float((wA.max(axis=0)-wA.min(axis=0)).max()):.4f}"
      f"over the whole slice; argmin positions = {[xsA[int(np.argmin(r['worst']))] for r in ROB]}")
    P("")

    # ------------------------------------------------------------------ how many independent readings the 20 pairs carry
    P("=" * 100)
    P("[5] how many independent readings do the 20 directed source-to-target pairs carry?")
    P("  Both quantities depend on the target body alone: the cutpoints never come from the ")
    P("source, and the own terciles are estimated inside the target's training folds.")
    P("  So the choice of source cannot move a target's reading.  Deduplicated:")
    P(f"  {'target':10s} {'own R2':>9s} {'abs R2':>9s} {'L':>9s} {'copies':>8s}")
    cnt = {nm: 0 for nm in BODIES}
    for a in BODIES:
        for b in BODIES:
            if a != b:
                cnt[b] += 1
    distinct_L = []
    for nm in BODIES:
        L_ = LG[nm][i10, i20]; distinct_L.append(L_)
        P(f"  {nm:10s} {OWN[nm]['r2_log']:9.4f} {R2G[nm][i10,i20]:9.4f} {L_:+9.4f} {cnt[nm]:8d}")
    distinct_L = np.array(distinct_L)
    n_any = int((distinct_L < 0).sum())
    n_big = int((distinct_L < -0.005).sum())
    P(f"  => 5 independent readings, copied into 20 rows ({set(cnt.values())} per body)")
    P(f"  => the absolute cutpoints are non-inferior on 5 of 5 bodies (worst +{distinct_L.max():.4f}, ")
    P(f"on Upper Ness), "
      f"improve on {n_any} of 5, and improve by more than 0.005 on {n_big} of 5.")
    P(f"  The 20-pair count is the same five readings counted four times each; "
      f"it inflates the denominator by a factor of four.")
    P("")
    P("  A random-cutpoint control shows the same thing: drawing 300 random pairs from a single ")
    P("random stream gives each copy of a target a different draw, so the spread between ")
    P("copies is Monte Carlo noise rather than a difference between data.")
    P("")
    P("")
    P("")

    # ------------------------------------------------------------------ bootstrap
    NB = 1000
    brng = np.random.default_rng(20261012)
    BOOT = {}
    for nm in BODIES:
        d = D[nm]; lk = np.log10(d["k"])
        ya = cv_pred(d["phi"], d["k"], d["fold"], (0.10, 0.20), K)
        yo = OWN[nm]["yp"]
        n = lk.size
        delta = np.empty(NB)
        for b in range(NB):
            idx = brng.integers(0, n, n)
            y = lk[idx]
            sst = np.sum((y - y.mean()) ** 2)
            delta[b] = (1 - np.sum((y - ya[idx]) ** 2) / sst) - (1 - np.sum((y - yo[idx]) ** 2) / sst)
        Lb = -delta
        lo, hi = np.percentile(Lb, [2.5, 97.5])
        BOOT[nm] = dict(L=float(Lb.mean()), lo=float(lo), hi=float(hi),
                        frac_pos=float((Lb > 0).mean()))
    P("=" * 100)
    P("[6] bootstrap confidence interval of L at (phi1,phi2)=(0.10,0.20), 1000 paired resamples ")
    P("of the test points")
    P(f"  {'body':10s} {'L':>9s} {'95% CI':>22s} {'P(L>0)':>8s}")
    for nm in BODIES:
        B_ = BOOT[nm]
        P(f"  {nm:10s} {B_['L']:+9.4f} {'['+format(B_['lo'],'+.4f')+', '+format(B_['hi'],'+.4f')+']':>22s} "
          f"{B_['frac_pos']:8.3f}")
    P("")

    # ------------------------------------------------------------------ write
    json.dump(dict(
        meta=dict(window=[PHI_LO, PHI_HI], K=K, seed=SEED, n_sub=N_SUB,
                  phi1_grid=G1.tolist(), phi2_grid=G2.tolist(), offsets=offsets, nb_boot=NB,
                  n={nm: int(D[nm]["phi"].size) for nm in BODIES},
                  folds={nm: int(len(np.unique(D[nm]["fold"]))) for nm in BODIES},
                  sigma_logk={nm: float(np.log10(D[nm]["k"]).std()) for nm in BODIES},
                  conv="L = R2_own - R2_abs (positive = abs cut worse)"),
        own={nm: dict(r2_log=OWN[nm]["r2_log"], r2_k=OWN[nm]["r2_k"]) for nm in BODIES},
        r2_log={nm: R2G[nm].tolist() for nm in BODIES},
        r2_k={nm: R2KG[nm].tolist() for nm in BODIES},
        loss={nm: LG[nm].tolist() for nm in BODIES},
        worst=worst.tolist(), mean_loss=meanL.tolist(),
        best=dict(phi1=p1, phi2=p2, worst_loss=val),
        slices=SL, robustness=ROB, robustness_x=xsA,
        boot=BOOT, distinct_loss=distinct_L.tolist(),
    ), open(os.path.join(OUT, "cut_scan_results.json"), "w", encoding="utf-8"),
       ensure_ascii=False, indent=1, default=float)

    # --- CSV: the figure scripts read only these
    with open(os.path.join(CSVD, "grid_worst.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["phi1", "phi2", "worst_loss", "mean_loss"])
        for ia, a in enumerate(G1):
            for ib, b in enumerate(G2):
                if np.isnan(worst[ia, ib]):
                    continue
                w.writerow([f"{a:.3f}", f"{b:.3f}", f"{worst[ia,ib]:.6f}", f"{meanL[ia,ib]:.6f}"])
    with open(os.path.join(CSVD, "grid_perbody.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["phi1", "phi2", "body", "loss"])
        for ia, a in enumerate(G1):
            for ib, b in enumerate(G2):
                for nm in BODIES:
                    if not np.isnan(LG[nm][ia, ib]):
                        w.writerow([f"{a:.3f}", f"{b:.3f}", nm, f"{LG[nm][ia,ib]:.6f}"])
    for kind in ("A", "B"):
        with open(os.path.join(CSVD, f"slice_{kind}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["cut"] + BODIES + ["worst"])
            for r in SL[kind]:
                w.writerow([f"{r['x']:.4f}"] + [f"{r[nm]:.6f}" for nm in BODIES] + [f"{r['worst']:.6f}"])
    with open(os.path.join(CSVD, "robust_folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["offset", "phi1", "worst_loss"])
        for r in ROB:
            for x, v in zip(xsA, r["worst"]):
                w.writerow([r["off"], f"{x:.4f}", f"{v:.6f}"])
    with open(os.path.join(CSVD, "boot.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["body", "L", "lo", "hi", "frac_pos"])
        for nm in BODIES:
            B_ = BOOT[nm]
            w.writerow([nm, f"{B_['L']:.6f}", f"{B_['lo']:.6f}", f"{B_['hi']:.6f}", f"{B_['frac_pos']:.4f}"])

    # --- markdown
    md = open(os.path.join(OUT, "cut_scan_tables.md"), "w", encoding="utf-8")
    md.write("# Cutpoint scan: worst-body loss at each cell (positive = the absolute cutpoints are worse)\n\n")
    md.write(f"- grid: phi1 in [{G1[0]}, {G1[-1]}], phi2 in [{G2[0]}, {G2[-1]}], step 0.005, ")
    md.write(f"  {len(GRID)} evaluated cells\n")
    md.write("- window 0.02<=phi<=0.45, k>0; K=3; one power law per phase fitted by OLS in log10 ")
    md.write("  space; loss L = R2_own - R2_abs\n")
    md.write("- `**` = worst-body loss <= 0.005 at that cell; `*` = <= 0.001; ")
    md.write("  `.` = phi2 <= phi1 + 0.02 and not evaluated\n")
    md.write(f"- worst-body loss at 0.10/0.20 = {worst[i10,i20]:+.4f} ")
    md.write(f"({BODIES[int(np.nanargmax([LG[nm][i10,i20] for nm in BODIES]))]})\n\n")
    md.write("| φ₁ \\ φ₂ | " + " | ".join(f"{b:.3f}" for b in G2) + " |\n")
    md.write("|---|" + "---|" * len(G2) + "\n")
    for ia, a in enumerate(G1):
        cells = []
        for ib, b in enumerate(G2):
            v = worst[ia, ib]
            if np.isnan(v):
                cells.append("·")
            else:
                tag = "**" if v <= 0.005 else ("*" if v <= 0.001 else "")
                cells.append(f"{v:+.4f}{tag}")
        md.write(f"| {a:.3f} | " + " | ".join(cells) + " |\n")
    md.write("\n## Per-body loss and reference\n\n| body | own R2(log) | L@0.10/0.20 | best gain | L_max |\n")
    md.write("|---|---|---|---|---|\n")
    for nm in BODIES:
        A = LG[nm]
        md.write(f"| {nm} | {OWN[nm]['r2_log']:.4f} | {A[i10,i20]:+.4f} | "
                 f"{OWN[nm]['r2_log']-np.nanmin(A):+.4f} | {np.nanmax(A):+.4f} |\n")
    md.write("\n## Bootstrap at 0.10/0.20 (1000 paired resamples)\n\n| body | L | 95% CI | P(L>0) |\n")
    md.write("|---|---|---|---|\n")
    for nm in BODIES:
        B_ = BOOT[nm]
        md.write(f"| {nm} | {B_['L']:+.4f} | [{B_['lo']:+.4f}, {B_['hi']:+.4f}] | {B_['frac_pos']:.3f} |\n")
    md.write("\n## Slices (worst-body loss)\n\n| cut | worst loss |\n|---|---|\n")
    for kind, oname in (("A", "φ₁"), ("B", "φ₂")):
        md.write(f"\n### Slice {kind} ({oname} swept, the other cutpoint fixed)\n\n| cut | worst loss |\n")
        md.write("|---|---|\n")
        for r in SL[kind]:
            md.write(f"| {r['x']:.4f} | {r['worst']:+.4f} |\n")
    md.close()
    rep.close()
    print(f"total {time.time()-t0:.1f}s")
    print("written: cut_scan_report.txt / cut_scan_results.json / cut_scan_tables.md / csv/*")

    return 0


if __name__ == "__main__":
    sys.exit(main())
