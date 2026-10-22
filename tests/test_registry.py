"""The registry compares a declaration with a signature; these tests pin both.

The comparison is only worth having if it can fail, so three of the tests below
register predictors that the type system admits and the declaration does not
describe: the FZI family's shape, and the body carrier that holds all three
columns at once.
"""
from __future__ import annotations

import pytest

from zeroleak.layers.data import Body
from zeroleak.registry import CARRIED_BY, REGISTRY, Predictor, check, signature_set
from zeroleak.types import CARRIES, Depth, Permeability, Porosity, requires


def test_the_shipped_registry_declares_what_it_reads():
    rows = {r["predictor"]: r for r in check(REGISTRY)}
    assert set(rows) == {"prior", "target_terciles", "free_exponent"}
    assert all(r["status"] == "ok" for r in rows.values())
    # the prior takes no data carrier at all, which is the whole claim
    assert rows["prior"]["derived"] == ()
    assert rows["target_terciles"]["available_at_target"] is True
    # the fit is the only row that reads permeability, and only inside a fold
    assert rows["free_exponent"]["available_at_target"] is False
    assert "permeability" in rows["free_exponent"]["derived"]


def test_every_carrier_has_a_token_set():
    # the registry must not silently skip a type that CARRIES knows about
    assert set(CARRIED_BY) == set(CARRIES)


def test_a_porosity_claim_that_takes_permeability_is_under_declared():
    @requires(phi=Porosity, perm=Permeability)
    def fzi_cutpoints(phi, perm):  # pragma: no cover - never called
        raise AssertionError("the audit reads signatures, it does not run predictors")

    row = check([Predictor("fzi_hfu", frozenset({"porosity:target"}), fzi_cutpoints)])[0]
    assert row["status"] == "under-declared"
    assert row["undeclared"] == ("permeability",)
    assert row["available_at_target"] is False


def test_the_body_carrier_exposes_three_quantities_at_once():
    @requires(body=Body)
    def fit_from_body(body):  # pragma: no cover - never called
        raise AssertionError("signature only")

    row = check([Predictor("body_fit", frozenset({"porosity:target"}), fit_from_body)])[0]
    # one legal type name, three quantities behind it
    assert row["derived"] == ("depth", "permeability", "porosity")
    assert row["undeclared"] == ("depth", "permeability")
    assert row["status"] == "under-declared"


def test_a_declaration_wider_than_the_signature_is_reported_not_corrected():
    @requires(phi=Porosity)
    def prior_like(phi):  # pragma: no cover - never called
        raise AssertionError("signature only")

    row = check([Predictor("cautious", frozenset({"porosity:target", "depth:target"}), prior_like)])[0]
    assert row["status"] == "conservative"
    assert row["undeclared"] == ()
    assert row["available_at_target"] is True


def test_a_predictor_without_a_contract_is_refused():
    def unguarded(phi):  # pragma: no cover - never called
        raise AssertionError("signature only")

    with pytest.raises(TypeError, match="requires"):
        signature_set(unguarded)
