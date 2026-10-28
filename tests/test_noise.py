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
from zeroleak.noise import (DEPTH_FRAC, GRID_CELLS, PAPER_DEPTH_OFFSETS,
                            PAPER_GRID_OFFSETS, draws_of, main)

DATA_DIR = os.environ.get("ZEROLEAK_DATA_DIR", "")
HAS_ARCHIVES = bool(DATA_DIR) and os.path.isdir(DATA_DIR)
needs_archives = pytest.mark.skipif(
    not HAS_ARCHIVES, reason="full archives absent; set ZEROLEAK_DATA_DIR (see DATA.md)")

# The supplement's SD column, to five decimals; the main text quotes 0.0003 / 0.0002 /
# 0.0027.  These are the values in Table S7, and the whole point of the test is that
# the package still lands on them rather than on something of its own.
PAPER = {"Tarbert": 0.000257, "Ness": 0.000153, "Volve": 0.002722}


# ---------------------------------------------------------------- the schedule
def test_schedule_has_the_requested_count():
    assert len(draws_of(20, 100)) == 20
    assert len(draws_of(5, 100)) == 5


def test_seeds_advance_by_a_thousand_from_the_configured_seed():
    """Draw i uses seed + 1000(i+1), the step the published table was built with.

    A repeat of the base seed would re-score the same subsample twenty times, and a
    step of one -- which is what an unexamined `seed + i` gives -- draws a different
    twenty subsamples and moves the standard deviation by a few times 1e-5.
    """
    sched = draws_of(20, 20261007)
    assert [s for _, s, _, _ in sched][:3] == [20262007, 20263007, 20264007]
    assert [s for _, s, _, _ in sched][-1] == 20281007
    assert len({s for _, s, _, _ in sched}) == 20


def test_the_published_offsets_are_reproduced_value_for_value():
    """The table, not a formula that resembles it.

    The supplement says the x boundaries move by 1--59 cells and the depth boundaries
    by 0.02--0.21 of the interval.  Many twenty-point sequences satisfy that sentence;
    only one produced the printed table, so the sequence is pinned here.
    """
    sched = draws_of(20, 0)
    assert tuple(c for _, _, c, _ in sched) == PAPER_GRID_OFFSETS
    assert tuple(f for _, _, _, f in sched) == PAPER_DEPTH_OFFSETS
    assert min(PAPER_GRID_OFFSETS) == 1 and max(PAPER_GRID_OFFSETS) == 59
    assert min(PAPER_DEPTH_OFFSETS) == pytest.approx(0.02)
    assert max(PAPER_DEPTH_OFFSETS) == pytest.approx(0.21)


def test_no_offset_lands_on_a_fold_boundary():
    """A shift of exactly one slab width (12 columns) only relabels the bins."""
    sched = draws_of(20, 1)
    assert all(c % 12 for _, _, c, _ in sched), "a multiple of 12 is the identity"
    assert all(0 < c < 60 for _, _, c, _ in sched)
    # 0.21 appears because the remainder lands on the period end there, which is the
    # same position as 0.01; the table prints it as 0.21 rather than folding it back,
    # so the assertion allows it and says why.
    assert all(0.02 <= f <= 0.21 for _, _, _, f in sched), "one period of five depth blocks"
    assert GRID_CELLS[1] < 60, "a full 60 cells would land back on the same slabs"
    assert DEPTH_FRAC[1] <= 0.2 + 0.02, "one period plus the start"


def test_other_draw_counts_keep_the_shape_of_the_design():
    """Six draws should be six draws of this design, not a different experiment."""
    sched = draws_of(6, 20261007)
    seeds = [s for _, s, _, _ in sched]
    cells = [c for _, _, c, _ in sched]
    fracs = [f for _, _, _, f in sched]
    assert seeds[1] - seeds[0] == 1000
    assert all(c % 12 for c in cells)
    assert cells == sorted(cells) and fracs == sorted(fracs)
    assert len(set(fracs)) == 6


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
def test_the_whole_printed_table_not_just_the_sd(root, tmp_path):
    """Mean, minimum and maximum, and the baseline column, all at four decimals.

    The supplement prints five numbers per body.  Pinning only the SD would let the
    mean drift a whole column away while the test stayed green -- and the baseline
    column is draw one, so a wrong schedule shows up here even when the spread is
    insensitive to it.
    """
    main(["--config", os.path.join(root, "configs", "full.yaml"), "--draws", "20",
          "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "noise_results.json").read_text(encoding="utf-8"))
    # column by column as Table S7 prints them: baseline, mean, sd, min, max
    printed = {
        "Tarbert": (-0.00206, -0.00183, 0.00026, -0.00224, -0.00124),
        "Ness":    (+0.00032, +0.00056, 0.00015, +0.00024, +0.00079),
        "Volve":   (-0.00630, -0.00632, 0.00272, -0.01328, -0.00201),
        "Brent":   (-0.02863, -0.02863, 0.00000, -0.02863, -0.02863),
        "Piper":   (-0.02262, -0.02262, 0.00000, -0.02262, -0.02262),
    }
    for nm, (base, mean, sd, lo, hi) in printed.items():
        body = payload["bodies"][nm]
        first = body["values"][0]
        assert first == pytest.approx(base, abs=5e-6), (nm, "baseline", first)
        assert body["mean"] == pytest.approx(mean, abs=5e-6), (nm, "mean", body["mean"])
        assert body["min"] == pytest.approx(lo, abs=5e-6), (nm, "min", body["min"])
        assert body["max"] == pytest.approx(hi, abs=5e-6), (nm, "max", body["max"])
        if body["sd"] is not None:
            assert body["sd"] == pytest.approx(sd, abs=5e-6), (nm, "sd", body["sd"])
        else:
            assert sd == 0.0 and body["n_distinct"] == 1, (nm, "printed as fixed")


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
