"""The cutpoint grid, tested without any archive.

The scan is the one place where the paper's cutpoints are chosen, and the count
of evaluated cells (951) appears in the text, in the SI and in two figures.  If
the window, the step or the separation constraint ever moved, every one of those
numbers would change while the code still ran.  These tests pin the grid so that
such a change fails here instead of appearing as a different paper.
"""
from __future__ import annotations

import os

import pytest

from zeroleak.scan import cutpoint_grid


def test_nine_hundred_and_fifty_one_cells():
    """The number quoted in the text.  Asserted, not recomputed downstream."""
    _, _, pairs = cutpoint_grid()
    assert len(pairs) == 951


def test_bounds_and_step():
    """phi1 in [0.05, 0.20], phi2 in [0.10, 0.30], step 0.005."""
    g1, g2, _ = cutpoint_grid()
    assert (g1[0], g1[-1]) == (0.05, 0.20)
    assert (g2[0], g2[-1]) == (0.10, 0.30)
    assert len(g1) == 31 and len(g2) == 41
    for g in (g1, g2):
        steps = {round(float(b - a), 6) for a, b in zip(g[:-1], g[1:])}
        assert steps == {0.005}, steps


def test_separation_constraint_holds_for_every_cell():
    """phi2 > phi1 + 0.02 everywhere; a three-phase model needs both gaps."""
    g1, g2, pairs = cutpoint_grid()
    assert all(b > a + 0.02 for a, b in pairs)
    # and the constraint is what makes 951 out of 31 * 41
    assert len(g1) * len(g2) == 1271
    assert len(pairs) == 951
    # the 320 cells it removes are exactly the ones inside the 0.02 gap
    excluded = [(a, b) for a in g1 for b in g2 if b <= a + 0.02]
    assert len(excluded) == 320
    assert all(b - a <= 0.02 + 1e-12 for a, b in excluded)
    # no excluded cell survives anywhere on the grid, at any phi1
    assert not set(excluded) & set(pairs)


def test_the_reported_cutpoints_are_on_the_grid():
    """0.110 / 0.200 must be an evaluate cell, not an interpolation."""
    _, _, pairs = cutpoint_grid()
    assert (0.11, 0.20) in pairs
    assert (0.10, 0.20) in pairs


def test_grid_is_ordered():
    """Ordering is what lets the downstream code slice columns by phi1."""
    _, _, pairs = cutpoint_grid()
    assert pairs == sorted(pairs)


def test_scan_loo_refuses_without_a_grid(tmp_path):
    """The two commands have an order; the second says so instead of guessing."""
    from zeroleak.scan_loo import main as loo_main

    missing = os.path.join(str(tmp_path), "nothing_here")
    with pytest.raises(SystemExit) as exc:
        loo_main(["--scan-dir", missing])
    assert "zeroleak scan" in str(exc.value)
