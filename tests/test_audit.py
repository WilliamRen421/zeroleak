"""The audit, and the one failure the type system cannot see.

The first test is the point of the module: a random fold rule and a contiguous
one are the same value to every layer in the package.  The rest pin the numbers
the audit reports, because they are quoted in the paper and must not drift
silently.

Volve 15/9-19A, 557 samples, spacing 0.25 m, five folds.
"""
from __future__ import annotations

import numpy as np
import pytest

from zeroleak.audit import AUDIT_RULES, INTERPOLATION_SPACINGS, geometry, optimism
from zeroleak.config import load
from zeroleak.layers import data, folds as folds_layer, model
from zeroleak.run import run_body
from zeroleak.types import Cutpoints, FoldLabels

TOL = 5e-5

#: Quoted in the paper (Section 4.1 and Table S12).
BLOCKED_R2_PRIOR = 0.7207
BLOCKED_DELTA = 0.0062
RANDOM_DELTA_MAX = 0.0054
RANDOM_R2_PRIOR_MAX = 0.7340


@pytest.fixture(scope="module")
def cfg(root):
    return load(root + "/configs/base.yaml")


@pytest.fixture(scope="module")
def body(cfg, root):
    spec = cfg.bodies[0]
    return data.load_csv(
        root + "/" + spec.file,
        spec.name,
        phi_col=spec.phi_col,
        perm_col=spec.perm_col,
        group_col=spec.group_col,
        phi_min=cfg.phi_min,
        phi_max=cfg.phi_max,
    )


@pytest.fixture(scope="module")
def audit_rows(cfg):
    spec = cfg.bodies[0]
    return {r["rule"]: r for r in optimism(spec, cfg, rules=AUDIT_RULES, root=".", n_seeds=5)}


def test_the_type_system_admits_both_rules(body, cfg):
    """A random split is indistinguishable from a contiguous one at every
    boundary the package polices.  That is why the audit exists."""
    cuts = Cutpoints((0.10, 0.20))
    for rule in ("depth_block", "random_block"):
        lab = folds_layer.assign_folds(body.phi, body.groups, rule=rule, n_blocks=5, seed=cfg.seed)
        assert isinstance(lab, FoldLabels)
        # the model layer takes the labels and nothing else about the fold
        coefs = model.fit(body.phi, body.perm, lab, cuts, hold_out=int(lab.x[0]))
        pred = model.predict(coefs, body.phi, cuts)
        assert pred.x.shape == body.perm.x.shape
        # no permeability reached the fold layer, and no boundary left it
        assert "perm" not in folds_layer.assign_folds.__signature__.parameters
        assert type(lab) is FoldLabels
        assert not hasattr(lab, "rule") and not hasattr(lab, "edges")


def test_the_geometry_separates_the_two_regimes(body, cfg, audit_rows):
    blocked = audit_rows["depth_block"]
    random = audit_rows["random_block"]
    assert blocked["verdict"] == "blocked"
    assert random["verdict"] == "interpolation"
    assert blocked["ratio_p50"] > 10.0
    assert random["ratio_p50"] < INTERPOLATION_SPACINGS
    # the verdict is not a knife-edge decision: the two regimes are far apart
    assert blocked["ratio_p50"] / random["ratio_p50"] > 20.0
    # and the fraction of test samples with a training sample one spacing away
    assert blocked["within_1_spacing"] < 0.05
    assert random["within_1_spacing"] > 0.80


def test_the_certified_rule_reproduces_the_table(body, cfg, audit_rows, root, tmp_path):
    run_body(cfg.bodies[0], cfg, root, str(tmp_path))
    row = audit_rows["depth_block"]
    assert row["r2"]["prior"] == pytest.approx(BLOCKED_R2_PRIOR, abs=TOL)
    assert row["delta"] == pytest.approx(BLOCKED_DELTA, abs=TOL)
    # the same quantity the paper reports, to four digits
    assert row["r2"]["target_terciles"] == pytest.approx(0.7145, abs=TOL)


def test_the_uncertified_rule_moves_the_headline_number(audit_rows):
    """The claim the audit protects: interpolation flatters the cutpoint
    comparison, so the published number is the conservative one."""
    blocked = audit_rows["depth_block"]
    random = audit_rows["random_block"]
    assert random["r2_prior_range"][1] > blocked["r2"]["prior"]
    assert random["delta_range"][1] < blocked["delta"]
    assert random["delta_range"][1] <= RANDOM_DELTA_MAX + TOL
    assert random["r2_prior_range"][1] <= RANDOM_R2_PRIOR_MAX + TOL
    # contiguous rules are seed-invariant; only the random rule opens a range
    assert blocked["delta_range"][0] == blocked["delta_range"][1]
    assert random["delta_range"][1] - random["delta_range"][0] > 1e-3


def test_geometry_rejects_a_single_fold(body):
    labels = np.zeros(len(body), dtype=int)
    with pytest.raises(ValueError):
        geometry(np.asarray(body.groups.x, dtype=float), labels)


def test_geometry_needs_a_spacing():
    g = np.full(20, 10.0)
    labels = np.arange(20) % 2
    with pytest.raises(ValueError):
        geometry(g, labels)


def test_strict_mode_exit_code(root, capsys):
    from zeroleak.audit import main

    rc = main(["--config", root + "/configs/base.yaml", "--strict"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "interpolate" in out
    # the flagged rule really is type-legal, and the message says so
    assert "never read permeability" in out
