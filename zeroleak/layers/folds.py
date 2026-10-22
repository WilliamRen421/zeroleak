"""Layer 2, fold assignment.

The signature is the invariant.  `assign_folds` receives porosity and depth, and
there is no parameter through which permeability could enter, so a fold cannot
be chosen because of the values it will be tested on.  The function returns
`FoldLabels`, which carries the label of each sample and not the boundary that
produced it, so a later layer cannot ask where a fold starts.

The block rules are contiguous: `depth_block` cuts the range of the group vector
into equal intervals, `index_block` cuts the row order into equal counts,
`group_block` leaves one group out at a time.  The group vector is sample depth
for a cored well, the x column for a grid body and a well code for a multi-well
body; the rule does not care which, and cannot tell, because it never sees
permeability.
"""
from __future__ import annotations

import numpy as np

from ..types import CARRIES, Depth, FoldLabels, Porosity, requires

__all__ = ["assign_folds", "RULES"]

RULES = ("depth_block", "index_block", "group_block", "single")

# The layer contract, written once and checked by tests/test_layers.py.
FOLD_INPUTS = ("phi", "groups", "rule", "n_blocks")


@requires(phi=Porosity, groups=Depth)
def assign_folds(
    phi: Porosity,
    groups: Depth,
    rule: str = "depth_block",
    n_blocks: int = 5,
    seed: int = 20261007,
) -> FoldLabels:
    """Assign one integer label per sample.

    `phi` is taken although the block rules do not read it: it fixes the row
    count so that a mismatch between the porosity vector and the ordering vector
    fails here rather than in the model, and it documents that the two inputs
    this layer may see are porosity and geometry.
    """
    if rule not in RULES:
        raise ValueError(f"unknown fold rule {rule!r}; choose from {RULES}")
    if phi.x.size != groups.x.size:
        raise ValueError(
            f"porosity has {phi.x.size} rows but the ordering vector has {groups.x.size}"
        )

    n = phi.x.size
    if rule == "single":
        return FoldLabels(np.zeros(n, dtype=int))

    if rule == "group_block":
        g = np.rint(groups.x).astype(int)
        if np.max(np.abs(groups.x - g)) > 1e-9:
            raise ValueError("group_block needs integer group codes")
        if g.min() < 0 or g.max() > 4096:
            raise ValueError("group codes must lie between 0 and 4096")
        if len(np.unique(g)) < 2:
            raise ValueError("a leave-one-group-out rule needs at least two groups")
        return FoldLabels(g)

    if n_blocks < 2 or n_blocks > n:
        raise ValueError(f"n_blocks must lie between 2 and {n}, got {n_blocks}")

    if rule == "depth_block":
        d = groups.x
        span = float(d.max() - d.min())
        if span <= 0.0:
            raise ValueError("the group vector is constant, so blocks are undefined")
        label = np.clip(((d - d.min()) / (span + 1e-9) * n_blocks).astype(int), 0, n_blocks - 1)
    else:  # index_block
        label = np.clip((np.arange(n) * n_blocks) // n, 0, n_blocks - 1)

    # Deterministic relabelling, so that two bodies with the same block structure
    # produce the same fold numbers and a CSV diff stays readable.
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_blocks)
    return FoldLabels(perm[label])


assert set(FOLD_INPUTS) <= set(assign_folds.__signature__.parameters), "fold contract drifted"
assert "perm" not in assign_folds.__signature__.parameters
assert CARRIES["FoldLabels"]
