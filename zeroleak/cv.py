# -*- coding: utf-8 -*-
"""Cross-validation core shared by ``scan`` and ``noise``.

Everything here used to live inside ``scan.main`` as a closure.  The noise command
needs the same loaders and the same foldwise fit, and a second copy would eventually
disagree with the first, so the block is lifted out unchanged and both commands call
it.  The arithmetic is the arithmetic of the paper: labels from absolute porosity
cutpoints, one power law per phase fitted by OLS in log10 space inside the training
fold, scored on the held-out fold.

Column names, scales and the fold rule come from the configuration, never from this
module: the shipped Volve table and the archive copy of the same well do not use the
same headers, and a hardcoded name silently reads zero rows instead of failing.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

__all__ = [
    "read_vals", "load_grid", "load_depth_blocks", "load_wells", "load_bodies",
    "labels_of", "cv_pred", "r2_log", "r2_k", "BODY_ORDER", "GRID_UNITS",
]

NX, NY, NZ = 60, 220, 85

#: The two units of the SPE10 sector, by the z extent each one occupies.
GRID_UNITS = {"Tarbert": (0, 34), "Upper Ness": (35, 84)}

#: Configuration key -> the name used in the report.  One table, because two copies of
#: it is how two commands start reporting different numbers for the same body.
BODY_NAMES = {
    "tarbert": "Tarbert",
    "upper_ness": "Ness",
    "volve_15_9_19a": "Volve",
    "brent_211_29": "Brent",
    "piper_15_17": "Piper",
}

BODY_ORDER = ("Tarbert", "Ness", "Volve", "Brent", "Piper")


def read_vals(fp, limit=None):
    """Read a whitespace-separated numeric file, optionally stopping at ``limit``."""
    v = []
    with open(fp) as f:
        for line in f:
            v.extend(float(t) for t in line.split())
            if limit is not None and len(v) >= limit:
                return np.array(v[:limit])
    return np.array(v)


def load_grid(phi_path, perm_path, seed, off=0, phi_lo=0.02, phi_hi=0.45,
              n_sub=100000, units=("Tarbert", "Upper Ness")):
    """Units of the SPE10 sector, folded into five contiguous x slabs.

    ``off`` shifts the slab boundaries along x, in columns.  The subsample draws from
    one generator that advances in unit order, so the stream for the second unit
    depends on the first having been taken: the configuration has to name both units
    to reproduce a run that names both, which is the point of a fixed schedule.
    """
    srng = np.random.default_rng(seed)
    NC = NX * NY * NZ
    phi = read_vals(phi_path, NC)
    kx = read_vals(perm_path, NC)
    phi3 = phi.reshape(NZ, NY, NX)
    kx3 = kx.reshape(NZ, NY, NX)
    out = {}
    for nm in units:
        z0, z1 = GRID_UNITS[nm]
        Pm = phi3[z0:z1 + 1]
        Km = kx3[z0:z1 + 1]
        zz, yy, xx = np.meshgrid(np.arange(z0, z1 + 1), np.arange(NY), np.arange(NX),
                                 indexing="ij")
        m = (Pm >= phi_lo) & (Pm <= phi_hi) & (Km > 0)
        p = Pm[m].astype(float)
        k = Km[m].astype(float)
        x = xx[m].astype(np.int32)
        z = zz[m].astype(np.int32)
        if p.size > n_sub:
            idx = srng.choice(p.size, n_sub, replace=False)
            p, k, x, z = p[idx], k[idx], x[idx], z[idx]
        fold = np.clip((((x + off) % NX) * 5) // NX, 0, 4).astype(int)
        out[nm] = dict(phi=p, k=k, fold=fold, group=None)
    return out


def load_depth_blocks(fp, phi_col, perm_col, depth_col, phi_scale=1.0,
                      phi_lo=0.02, phi_hi=0.45, off=0.0, n_blocks=5):
    """One cored well, folded into contiguous depth blocks.

    ``off`` is a fraction of the cored interval, not a depth: the blocks are an
    arbitrary cut of one well and the check moves them, it does not move the rock.
    """
    p, k, d = [], [], []
    with open(fp, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                pp = float(r[phi_col]) / phi_scale
                kk = float(r[perm_col])
                dd = float(r[depth_col])
            except (TypeError, ValueError, KeyError):
                continue
            if not (phi_lo <= pp <= phi_hi) or not (kk > 0):
                continue
            p.append(pp)
            k.append(kk)
            d.append(dd)
    p = np.array(p)
    k = np.array(k)
    d = np.array(d)
    rg = d.max() - d.min() + 1e-9
    fold = np.clip(((((d - d.min()) / rg + off) % 1.0) * n_blocks).astype(int),
                   0, n_blocks - 1)
    return dict(phi=p, k=k, fold=fold, group=None)


def load_wells(fp, phi_col, perm_col, group_col, phi_scale=1.0,
               phi_lo=0.02, phi_hi=0.45):
    """A multi-well table, folded by well: every fold leaves one well out."""
    p, k, g = [], [], []
    with open(fp, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                pp = float(r[phi_col]) / phi_scale
                kk = float(r[perm_col])
            except (TypeError, ValueError, KeyError):
                continue
            if not (phi_lo <= pp <= phi_hi) or not (kk > 0):
                continue
            p.append(pp)
            k.append(kk)
            g.append(r[group_col].strip())
    p = np.array(p)
    k = np.array(k)
    g = np.array(g)
    wells = np.unique(g)
    return dict(phi=p, k=k, fold=np.searchsorted(wells, g), group=g)


def load_bodies(cfg, seed, off=0.0):
    """Load every body the configuration names, keyed by the name used in the report.

    ``off`` is the fold offset, either as a fraction of a fold cycle -- the same rule
    for every body, one cycle being the 60 columns of the grid and the cored interval
    of the well -- or as ``{"grid_cells": c, "depth_frac": f}`` when the two are
    counted in their own units.  A body folded by well has no cycle and ignores it,
    which is why the noise check can only speak for the bodies where the fold is a
    free choice.
    """
    if isinstance(off, dict):
        cells = float(off.get("grid_cells", 0.0))
        depth = float(off.get("depth_frac", 0.0))
    else:
        cells, depth = off * NX, off
    out = {}
    grid = [b for b in cfg.bodies if b.grid_unit]
    if grid:
        sp = grid[0]
        units = tuple(b.grid_unit for b in grid)
        got = load_grid(sp.file, sp.perm_file, seed, cells,
                        cfg.phi_min, cfg.phi_max,
                        int(sp.n_subsample or 100000), units)
        for b in grid:
            out[BODY_NAMES.get(b.name, b.name)] = got[b.grid_unit]
    for spec in cfg.bodies:
        if spec.grid_unit:
            continue
        nm = BODY_NAMES.get(spec.name, spec.name)
        if spec.group_kind == "categorical":
            out[nm] = load_wells(spec.file, spec.phi_col, spec.perm_col, spec.group_col,
                                 spec.phi_scale, cfg.phi_min, cfg.phi_max)
        else:
            out[nm] = load_depth_blocks(spec.file, spec.phi_col, spec.perm_col,
                                        spec.group_col, spec.phi_scale,
                                        cfg.phi_min, cfg.phi_max, depth,
                                        int(spec.n_blocks or cfg.n_blocks))
    return {nm: out[nm] for nm in BODY_ORDER if nm in out}


def labels_of(x, th):
    return np.clip(np.digitize(x, th), 0, len(th))


def cv_pred(phi, k, fold, rule, phases=3, src_th=None):
    """Held-out predictions in log10 space.

    ``rule`` is ``'own'`` (equal-frequency terciles inside each training fold),
    ``'src'`` (cutpoints carried in from elsewhere) or a ``(phi1, phi2)`` pair of
    absolute cutpoints.  A phase holding fewer than ten training samples falls back
    to the single-power-law fit, which is the rule stated in the paper.
    """
    lp = np.log10(phi)
    lk = np.log10(k)
    yp = np.empty_like(lk)
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        if tr.sum() < 30 or te.sum() == 0:
            continue
        if rule == "own":
            th = np.unique(np.quantile(lp[tr], np.linspace(0, 1, phases + 1)))[1:-1]
        elif rule == "src":
            th = src_th
        else:
            th = np.array([math.log10(rule[0]), math.log10(rule[1])])
        th = np.sort(np.asarray(th, dtype=float))
        lab_tr = labels_of(lp[tr], th)
        lab_te = labels_of(lp[te], th)
        lp_te = lp[te]
        pred = np.empty(int(te.sum()))
        A = np.column_stack([np.ones(int(tr.sum())), lp[tr]])
        coefs = []
        for c in range(len(th) + 1):
            mm = lab_tr == c
            if mm.sum() >= 10:
                cc, *_ = np.linalg.lstsq(A[mm], lk[tr][mm], rcond=None)
            else:
                cc, *_ = np.linalg.lstsq(A, lk[tr], rcond=None)
            coefs.append(cc)
        for c in range(len(th) + 1):
            mm = lab_te == c
            if mm.any():
                pred[mm] = coefs[c][0] + coefs[c][1] * lp_te[mm]
        yp[te] = pred
    return yp


def r2_log(lk, yp):
    return float(1 - np.sum((lk - yp) ** 2) / np.sum((lk - lk.mean()) ** 2))


def r2_k(k, yp):
    pk = 10.0 ** yp
    return float(1 - np.sum((k - pk) ** 2) / np.sum((k - k.mean()) ** 2))
