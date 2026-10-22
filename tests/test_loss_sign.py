"""The sign convention of the loss.

`metrics.loss(reference, candidate)` returns reference R2 minus candidate R2, so
a positive value means the candidate is worse.  A convention that can be read
both ways is not a measurement, so it is tested from both directions:

  * on a body whose curvature the prior cutpoints cannot follow, the prior is
    known to be worse and the loss must be positive;
  * on a body generated with breaks exactly at 0.10 and 0.20 and a porosity
    distribution whose own terciles sit elsewhere, the prior is known to be
    better and the loss, taken the other way round, must be negative.
"""
from __future__ import annotations

import numpy as np

from zeroleak.layers import data, folds as folds_layer, metrics, model
from zeroleak.layers.data import Body
from zeroleak.strategies import cutpoints_from_porosity
from zeroleak.types import Cutpoints, Depth, Permeability, Porosity, Truth

PRIOR = Cutpoints.from_config((0.10, 0.20))


def _body(phi: np.ndarray, k: np.ndarray) -> Body:
    return Body(
        name="synthetic",
        phi=Porosity(phi),
        perm=Permeability(k),
        groups=Depth(np.linspace(3000.0, 3100.0, phi.size)),
    )


def _score(body: Body, cuts: Cutpoints) -> metrics.Metrics:
    lab = folds_layer.assign_folds(body.phi, body.groups, rule="single")
    coefs = model.fit(body.phi, body.perm, lab, cuts)
    pred = model.predict(coefs, body.phi, cuts)
    return metrics.score(pred, Truth(np.log10(body.perm.x)))


def test_prior_is_positive_loss_when_the_partition_is_wrong():
    """All porosity lies above both cutpoints, so the prior leaves two phases
    empty and has to carry one slope over a curved cloud."""
    rng = np.random.default_rng(5)
    phi = rng.uniform(0.22, 0.40, 600)
    break_at = 0.28
    k = np.where(
        phi < break_at,
        10 ** (2.0 + 7.0 * np.log10(phi / break_at)),
        10 ** (2.0 + 0.5 * np.log10(phi / break_at)),
    )
    k = k * 10 ** rng.normal(0.0, 0.10, phi.size)
    body = _body(phi, k)

    terciles = cutpoints_from_porosity(body.phi, 3)
    m_terc = _score(body, terciles)
    m_prior = _score(body, PRIOR)

    assert m_prior["r2"] < m_terc["r2"]
    assert metrics.loss(m_terc, m_prior) > 0.0
    assert metrics.loss(m_prior, m_terc) < 0.0


def test_prior_is_negative_loss_when_the_partition_is_right():
    rng = np.random.default_rng(9)
    # Unbalanced so that the equal-frequency terciles fall well away from
    # 0.10/0.20: four hundred samples below the first break, three hundred
    # between the breaks, three hundred above.
    phi = np.concatenate([
        rng.uniform(0.03, 0.10, 400),
        rng.uniform(0.10, 0.20, 300),
        rng.uniform(0.20, 0.42, 300),
    ])
    slope = np.where(phi < 0.10, 1.0, np.where(phi < 0.20, 3.0, 5.0))
    k = 10 ** (2.0 + slope * np.log10(phi))
    k = k * 10 ** rng.normal(0.0, 0.05, phi.size)
    body = _body(phi, k)

    terciles = cutpoints_from_porosity(body.phi, 3)
    assert not np.allclose(terciles.values, PRIOR.values, atol=0.01)

    m_terc = _score(body, terciles)
    m_prior = _score(body, PRIOR)

    assert m_prior["r2"] > m_terc["r2"]
    assert metrics.loss(m_terc, m_prior) < 0.0
    assert metrics.loss(m_prior, m_terc) > 0.0


def test_loss_is_antisymmetric():
    a = metrics.Metrics({"r2": 0.71})
    b = metrics.Metrics({"r2": 0.66})
    assert metrics.loss(a, b) > 0.0
    assert metrics.loss(b, a) < 0.0
    assert metrics.loss(a, b) == -metrics.loss(b, a)
    assert metrics.loss(a, a) == 0.0
