"""All five rows of Table 1, when the archives are present.

Skipped by default: the four non-shipped archives total about 80 MB and three of
them are not redistributed here.  Set `ZEROLEAK_DATA_DIR` to a directory laid out
as in `DATA.md` and the five rows are checked against stored values typed in from
the manuscript's Table 1.

This is the test that makes the paper's sentence about "the headline R2 values of
Table 1, checked against stored reference values" a claim about code rather than
about intention.
"""
from __future__ import annotations

import os

import pytest

from zeroleak.config import load
from zeroleak.run import run_body

DATA_DIR = os.environ.get("ZEROLEAK_DATA_DIR", "")
REQUIRED = (
    "spe10/spe_phi.dat",
    "spe10/spe_perm.dat",
    "volve_core_full.csv",
    "brent/cgg_phik_211-29.csv",
    "piper/core_15-17.csv",
)

pytestmark = pytest.mark.skipif(
    not DATA_DIR or not all(os.path.exists(os.path.join(DATA_DIR, p)) for p in REQUIRED),
    reason="full archives absent; set ZEROLEAK_DATA_DIR (see DATA.md)",
)

# Table 1, unrounded to the four decimals printed there.
TABLE1 = {
    "tarbert": {"prior": 0.7355, "target_terciles": 0.7336, "n": 100_000},
    "upper_ness": {"prior": 0.6634, "target_terciles": 0.6641, "n": 100_000},
    "volve_15_9_19a": {"prior": 0.7207, "target_terciles": 0.7145, "n": 557},
    "brent_211_29": {"prior": 0.7430, "target_terciles": 0.7143, "n": 586},
    "piper_15_17": {"prior": 0.4868, "target_terciles": 0.4642, "n": 4428},
}
TOL = 5e-5


@pytest.fixture(scope="module")
def runs(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    cfg = load(f"{root}/configs/full.yaml")
    table = {}
    for spec in cfg.bodies:
        for row in run_body(spec, cfg, root, str(out)):
            table.setdefault(spec.name, {})[row["strategy"]] = row
    return table


@pytest.mark.parametrize("body", sorted(TABLE1))
def test_row_of_table_1(runs, body):
    expected = TABLE1[body]
    got = runs[body]
    assert got["prior"]["n"] == expected["n"]
    assert got["prior"]["r2"] == pytest.approx(expected["prior"], abs=TOL)
    assert got["target_terciles"]["r2"] == pytest.approx(expected["target_terciles"], abs=TOL)


@pytest.mark.parametrize("body", sorted(TABLE1))
def test_delta_r2_sign(runs, body):
    """Table 1's last-but-one column: prior minus tercile, positive on four."""
    delta = runs[body]["prior"]["r2"] - runs[body]["target_terciles"]["r2"]
    expected = {
        "tarbert": +0.0019,
        "upper_ness": -0.0007,
        "volve_15_9_19a": +0.0062,
        "brent_211_29": +0.0286,
        "piper_15_17": +0.0226,
    }[body]
    # A difference of two values rounded to four decimals carries about 1e-4 of
    # its own rounding, so this column is checked more loosely than the rows.
    assert delta == pytest.approx(expected, abs=2e-4)
