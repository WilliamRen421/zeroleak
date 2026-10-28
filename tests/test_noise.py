"""The multi-seed check: its schedule, its sd, and the numbers the paper quotes.

The manuscript states one standard deviation of the loss under redrawn subsamples and
fold offsets -- 0.0001 on Upper Ness, 0.0027 on the cored well -- and the supplement
prints the same table at five significant figures.  A number in the text that the
package cannot reproduce is worse than no number, so both the definition and the
values are pinned here.  The five-body run takes fifteen seconds, so it is not
gated behind an opt-in flag: a reviewer checking a table can wait fifteen seconds.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from zeroleak.config import load
from zeroleak.noise import DEPTH_FRAC, GRID_CELLS, draws_of, main

DATA_DIR = os.environ.get("ZEROLEAK_DATA_DIR", "")
HAS_ARCHIVES = bool(DATA_DIR) and os.path.isdir(DATA_DIR)
needs_archives = pytest.mark.skipif(
    not HAS_ARCHIVES, reason="full archives absent; set ZEROLEAK_DATA_DIR (see DATA.md)")

# The supplement's SD column, to five decimals; the main text quotes 0.0002 / 0.0027.
PAPER = {"Tarbert": 0.000231, "Ness": 0.000137, "Volve": 0.002723}


# ---------------------------------------------------------------- the schedule
def test_schedule_has_the_requested_count():
    assert len(draws_of(20, 100)) == 20
    assert len(draws_of(5, 100)) == 5


def test_seeds_advance_by_one_from_the_configured_seed():
    """A repeat of the base seed would re-score the same subsample twenty times."""
    sched = draws_of(5, 20261007)
    assert [s for _, s, _, _ in sched] == [20261007, 20261008, 20261009, 20261010, 20261011]


def test_offsets_span_exactly_one_period_of_the_fold_partition():
    """The window is what makes twenty draws enough, so it is asserted, not assumed.

    The grid folds into five x slabs of 12 columns, so a shift of 60 cells returns the
    same partition; the cored well folds into five depth blocks, so the period is one
    fifth of the cored interval.  Sweeping more than one period revisits partitions
    and the spread begins to measure the offset grid instead of the data.
    """
    sched = draws_of(20, 1)
    assert sched[0][2] == GRID_CELLS[0] == 1
    assert sched[-1][2] == GRID_CELLS[0] + GRID_CELLS[1] == 59
    assert sched[0][3] == pytest.approx(DEPTH_FRAC[0]) == pytest.approx(0.02)
    assert sched[-1][3] == pytest.approx(DEPTH_FRAC[0] + DEPTH_FRAC[1]) == pytest.approx(0.21)
    assert GRID_CELLS[1] < 60, "a full 60 cells would land back on the same slabs"
    assert DEPTH_FRAC[0] + DEPTH_FRAC[1] <= 0.2 + 0.02, "one period plus the start"


def test_both_knobs_move_monotonically():
    sched = draws_of(20, 1)
    cells = [c for _, _, c, _ in sched]
    fracs = [f for _, _, _, f in sched]
    assert cells == sorted(cells) and len(set(cells)) > 15
    assert fracs == sorted(fracs) and len(set(fracs)) == 20


def test_one_draw_is_rejected():
    with pytest.raises(ValueError, match="at least two draws"):
        draws_of(1, 100)


# ---------------------------------------------------------------- the command
def test_base_config_writes_three_files(root, tmp_path):
    assert main(["--config", os.path.join(root, "configs", "base.yaml"),
                 "--draws", "6", "--out", str(tmp_path)]) == 0
    for name in ("noise_report.txt", "noise_results.json", "noise.csv"):
        assert (tmp_path / name).is_file(), name


def test_sd_is_the_sample_standard_deviation(root, tmp_path):
    main(["--config", os.path.join(root, "configs", "base.yaml"),
          "--draws", "6", "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "noise_results.json").read_text(encoding="utf-8"))
    body = payload["bodies"]["Volve"]
    values = np.array(body["values"], dtype=float)
    assert len(values) == 6
    assert body["sd"] == pytest.approx(float(values.std(ddof=1)), rel=1e-12)
    # and it is not the population sd, which would be smaller by sqrt(n/(n-1))
    assert body["sd"] != pytest.approx(float(values.std(ddof=0)), rel=1e-9)


def test_the_check_actually_moves(root, tmp_path):
    """A noise check that reports the same number six times is not a check.

    On the shipped body the fold boundaries are the only free choice, so this also
    pins that the offset reaches the fit: if it did not, every draw would be equal.
    """
    main(["--config", os.path.join(root, "configs", "base.yaml"),
          "--draws", "6", "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "noise_results.json").read_text(encoding="utf-8"))
    body = payload["bodies"]["Volve"]
    assert body["n_distinct"] > 1
    assert body["sd"] > 0


def test_anchor_sits_inside_the_draws(root, tmp_path):
    """The single-seed value `scan` reports must be bracketed by the redraws."""
    main(["--config", os.path.join(root, "configs", "base.yaml"),
          "--draws", "6", "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "noise_results.json").read_text(encoding="utf-8"))
    values = payload["bodies"]["Volve"]["values"]
    assert min(values) <= payload["anchor"]["Volve"] <= max(values)


def test_inverted_pair_is_rejected(root, tmp_path):
    with pytest.raises(ValueError, match="below the second"):
        main(["--config", os.path.join(root, "configs", "base.yaml"),
              "--draws", "4", "--phi1", "0.20", "--phi2", "0.10", "--out", str(tmp_path)])


# ---------------------------------------------------------------- the numbers
@needs_archives
def test_bodies_folded_by_well_are_reported_as_fixed(root, tmp_path):
    """Brent and Piper fold by well; there is no boundary to move.

    Reporting a standard deviation of zero there would claim we checked that the
    number does not move, which the design cannot support.
    """
    main(["--config", os.path.join(root, "configs", "full.yaml"),
          "--draws", "4", "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "noise_results.json").read_text(encoding="utf-8"))
    for nm in ("Brent", "Piper"):
        body = payload["bodies"][nm]
        assert body["sd"] is None
        assert body["n_distinct"] == 1
        assert body["foldable"] is False
    for nm in ("Tarbert", "Ness", "Volve"):
        assert payload["bodies"][nm]["foldable"] is True


@needs_archives
def test_the_numbers_the_paper_quotes(root, tmp_path):
    """The supplement's SD column, at the default twenty draws.

    Both move here if the schedule, the fold rule or the loss definition changes,
    which is the point: the sentence in the paper would then be quoting something the
    package no longer does.
    """
    main(["--config", os.path.join(root, "configs", "full.yaml"), "--draws", "20",
          "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "noise_results.json").read_text(encoding="utf-8"))
    assert payload["meta"]["draws"] == 20
    for nm, expected in PAPER.items():
        got = payload["bodies"][nm]["sd"]
        assert got == pytest.approx(expected, abs=2e-6), (nm, got, expected)


@needs_archives
def test_the_anchor_is_the_run_the_cutpoint_scan_reports(root, tmp_path):
    """Anchor = seed and no offset = the single-seed loss `scan` prints at this pair.

    Table 4's tercile-minus-prior difference is the same number, which is what lets
    the supplement's baseline column be the main text's table rather than a private
    run nobody can reproduce.
    """
    main(["--config", os.path.join(root, "configs", "full.yaml"), "--draws", "20",
          "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "noise_results.json").read_text(encoding="utf-8"))
    assert payload["anchor"]["Ness"] == pytest.approx(0.000710, abs=5e-6)
    assert payload["anchor"]["Volve"] == pytest.approx(-0.006194, abs=5e-6)
