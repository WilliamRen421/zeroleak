"""The headline values for the shipped body, against the manuscript's Table 1.

These numbers are typed in from the table, not read from `expected/`, so the
test is independent of the committed output: if the pipeline drifts, either this
test fails or the reproduction test fails, and the two failures mean different
things.

Volve 15/9-19A, 557 samples, five depth blocks, prior cutpoints 0.10/0.20.
"""
from __future__ import annotations

import numpy as np
import pytest

from zeroleak.config import load
from zeroleak.run import run_body

TABLE1_VOLVE = {
    "n": 557,
    "sigma_log10_k": 1.3153,
    "r2_prior": 0.7207,
    "r2_tercile": 0.7145,
    "rmse_sigma_prior": 0.5285,
    "r2_full": 0.7379,  # full-sample fit at the same cutpoints, used in Sec. 5.3
}
TOL = 5e-5


@pytest.fixture(scope="module")
def runs(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("headline")
    cfg = load(f"{root}/configs/base.yaml")
    rows = run_body(cfg.bodies[0], cfg, root, str(out))
    return {r["strategy"]: r for r in rows}


def test_sample_count(runs):
    assert runs["prior"]["n"] == TABLE1_VOLVE["n"]


def test_prior_cutpoints_reproduce_the_table(runs):
    assert runs["prior"]["r2"] == pytest.approx(TABLE1_VOLVE["r2_prior"], abs=TOL)
    assert runs["prior"]["rmse_sigma"] == pytest.approx(TABLE1_VOLVE["rmse_sigma_prior"], abs=TOL)


def test_target_terciles_reproduce_the_table(runs):
    assert runs["target_terciles"]["r2"] == pytest.approx(TABLE1_VOLVE["r2_tercile"], abs=TOL)


def test_delta_r2_has_the_published_sign(runs):
    delta = runs["prior"]["r2"] - runs["target_terciles"]["r2"]
    assert delta == pytest.approx(0.0062, abs=TOL)


def test_full_sample_fit_matches_the_transfer_reference(runs):
    assert runs["prior"]["r2_full"] == pytest.approx(TABLE1_VOLVE["r2_full"], abs=TOL)


def test_rmse_and_r2_are_consistent(runs):
    """RMSE/sigma = sqrt(1 - R2) is an identity of the two definitions, and the
    paper relies on it; check it rather than assume it."""
    for row in runs.values():
        assert row["rmse_sigma"] == pytest.approx(np.sqrt(1.0 - row["r2"]), abs=1e-12)


def test_sigma_of_log_permeability(runs, root):
    from zeroleak.layers.data import load_csv

    body = load_csv(f"{root}/data/volve_15_9_19a.csv", "volve")
    assert float(np.log10(body.perm.x).std()) == pytest.approx(
        TABLE1_VOLVE["sigma_log10_k"], abs=TOL
    )
