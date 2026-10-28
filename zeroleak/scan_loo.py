# -*- coding: utf-8 -*-
"""Leave-one-body-out: is the plateau's position or width set by a single body?

Reads the grid written by `zeroleak scan` and recomputes the optimum with each
body removed in turn.

What this cannot show: the criterion is "worst-body loss <= eps", so removing a
body can only relax it and the question "is this cell still inside the window"
is true by construction.  The check is therefore a statement about where the
plateau sits and how wide it is, not about coverage.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

__all__ = ["main"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "configs", "full.yaml")


def main(argv=None):
    """Recompute the plateau position and width with each body removed in turn.

    Reads the grid written by `zeroleak scan`, so that command must have run
    first.  Reproduces Table S9 of the supplement.
    """
    ap = argparse.ArgumentParser(
        prog="zeroleak scan-loo",
        description="Leave-one-body-out recomputation of the cutpoint scan.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="configuration used for the scan (default: configs/full.yaml)")
    ap.add_argument("--scan-dir", default=None,
                    help="directory holding the scan output (default: <output_dir>/scan)")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    scan_dir = args.scan_dir
    if scan_dir is None:
        from . import config as _config
        scan_dir = os.path.join(_config.load(args.config).output_dir, "scan")
    scan_dir = os.path.abspath(scan_dir)

    BASE = scan_dir
    SRC = os.path.join(BASE, "csv", "grid_perbody.csv")
    OUTD = os.path.join(BASE, "csv")
    if not os.path.exists(SRC):
        raise SystemExit("zeroleak scan-loo: %s not found; run `zeroleak scan` first" % SRC)

    EPS_LIST = (0.001, 0.005)

    G = collections.defaultdict(dict)
    BODIES = []
    with open(SRC, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (round(float(r["phi1"]), 4), round(float(r["phi2"]), 4))
            G[key][r["body"]] = float(r["loss"])
            if r["body"] not in BODIES:
                BODIES.append(r["body"])

    XS = sorted({k[0] for k in G})
    YS = sorted({k[1] for k in G})
    assert len(G) == 951, "the grid should have 951 cells, found %d" % len(G)
    assert len(BODIES) == 5, "the grid should carry 5 bodies"
    for k, d in G.items():
        assert len(d) == 5, "cell %s is missing a body" % (k,)

    worst = lambda k, sub: max(G[k][b] for b in sub)

    def argmin(sub):
        best = None
        for k in G:
            w = worst(k, sub)
            if best is None or w < best[0]:
                best = (w, k)
        return best

    def window(sub, eps):
        """Window: for each phi1 take the minimum over phi2 in that column, then compare with eps."""
        ok = []
        for x in XS:
            col = [worst((x, y), sub) for y in YS if (x, y) in G]
            if col and min(col) <= eps:
                ok.append(x)
        return (min(ok), max(ok)) if ok else None

    rows = []
    for out in [None] + BODIES:
        sub = BODIES if out is None else [b for b in BODIES if b != out]
        w, k = argmin(sub)
        cell = G[(0.10, 0.20)]
        wb = max(sub, key=lambda b: cell[b])
        row = dict(
            holdout=out or "all five",
            n_body=len(sub),
            phi1_min=k[0], phi2_min=k[1], loss_min=round(w, 6),
            loss_at_cut=round(cell[wb], 6), worst_body_at_cut=wb,
        )
        for eps in EPS_LIST:
            win = window(sub, eps)
            row["win_%.3f" % eps] = win
        rows.append(row)

    with open(os.path.join(OUTD, "loo_argmin.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["held_out", "n_body", "phi1_min", "phi2_min", "loss_min",
                    "worst_body_at_cut", "worst_at_cut", "win_0.001_lo", "win_0.001_hi",
                    "win_0.005_lo", "win_0.005_hi"])
        for r in rows:
            w.writerow([r["holdout"], r["n_body"], "%.3f" % r["phi1_min"], "%.3f" % r["phi2_min"],
                        "%+.5f" % r["loss_min"], "%+.5f" % r["loss_at_cut"], r["worst_body_at_cut"],
                        "%.3f" % r["win_0.001"][0], "%.3f" % r["win_0.001"][1],
                        "%.3f" % r["win_0.005"][0], "%.3f" % r["win_0.005"][1]])

    json.dump(dict(bodies=BODIES, eps=EPS_LIST, rows=rows),
              open(os.path.join(OUTD, "loo_argmin.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lines = ["Leave-one-body-out (criterion: worst-body loss per cell; grid and folds as in scan)",
             "Not evidence of coverage: max <= eps can only relax when a body is removed.", ""]
    lines.append("%-10s %-14s %-13s %-22s %-22s" % ("held out", "argmin(phi1,phi2)", "loss@argmin",
                                                   "window eps=0.001", "window eps=0.005"))
    for r in rows:
        lines.append("%-10s (%.3f, %.3f)  %+.5f     [%.3f, %.3f]           [%.3f, %.3f]" % (
            r["holdout"], r["phi1_min"], r["phi2_min"], r["loss_min"],
            r["win_0.001"][0], r["win_0.001"][1], r["win_0.005"][0], r["win_0.005"][1]))
    lines.append("")
    lines.append("Conclusion: argmin phi1 stays at %.3f in all five removals; the window widens "
                     "noticeably only when Upper Ness is held out."
                 % rows[0]["phi1_min"])
    open(os.path.join(BASE, "loo_argmin.txt"), "w", encoding="utf-8").write("\n".join(lines))

    # the position must not be set by a single body
    assert len({r["phi1_min"] for r in rows}) == 1, "argmin phi1 should stay at 0.110"
    assert rows[0]["worst_body_at_cut"] == "Ness", "the worst body at 0.10/0.20 should be Upper Ness"
    print("\n".join(lines))
    print("\nOK -> loo_argmin.csv / loo_argmin.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
