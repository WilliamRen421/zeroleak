"""Structural tests: the information sets are properties of the code.

The manuscript claims that a leak in this pipeline shows up as a type mismatch
or a missing argument rather than as a subtle mistake a reader has to find in
prose.  These tests hold the code to that claim, on the syntax tree rather than
on a reading of the source.

Three kinds of check:
  * the signatures of the five layer functions;
  * a call that would cross a boundary raises TypeError;
  * a module that must not contain an estimator or an import does not contain it.
"""
from __future__ import annotations

import ast
import inspect
import os

import numpy as np
import pytest

from zeroleak import strategies
from zeroleak.layers import data, folds, metrics, model, report
from zeroleak.types import Cutpoints, Depth, FoldLabels, Permeability, Porosity, Predictions, Truth

LAYER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(data.__file__))), "layers")


def _tree(module) -> ast.AST:
    with open(inspect.getsourcefile(module), encoding="utf-8") as fh:
        return ast.parse(fh.read())


def _called_names(tree: ast.AST) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def _imported(module: str) -> set:
    tree = _tree(module)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.add((node.module or "") + ":" + ",".join(a.name for a in node.names))
    return out


# --------------------------------------------------------------- signatures
def test_fold_layer_has_no_permeability_parameter():
    params = set(inspect.signature(folds.assign_folds).parameters)
    assert params == {"phi", "groups", "rule", "n_blocks", "seed"}
    assert not any("perm" in p or p == "k" for p in params), params


def test_model_never_receives_fold_boundaries():
    params = set(inspect.signature(model.fit).parameters)
    assert "folds" in params and "fold_boundaries" not in params


def test_metric_layer_has_no_model_or_cutpoint_parameter():
    params = set(inspect.signature(metrics.score).parameters)
    assert params == {"pred", "truth"}


def test_cutpoints_has_no_estimator():
    """An estimator on `Cutpoints` would put the boundary back inside the model."""
    forbidden = {"from_data", "terciles", "quantiles", "estimate", "fit", "from_phi", "search"}
    assert forbidden.isdisjoint(dir(Cutpoints)), forbidden & set(dir(Cutpoints))


# ------------------------------------------------------------ call guarding
def test_body_is_rejected_by_the_fold_layer(volve_path):
    body = data.load_csv(volve_path, "volve")
    with pytest.raises(TypeError):
        folds.assign_folds(body, body.groups)


def test_permeability_is_rejected_where_depth_is_expected():
    phi = Porosity(np.linspace(0.05, 0.4, 50))
    k = Permeability(np.linspace(1.0, 100.0, 50))
    with pytest.raises(TypeError):
        folds.assign_folds(phi, k)


def test_missing_argument_is_a_type_error():
    phi = Porosity(np.linspace(0.05, 0.4, 50))
    with pytest.raises(TypeError):
        folds.assign_folds(phi)  # no groups: the fold rule cannot run


def test_truth_and_predictions_are_not_interchangeable():
    pred = Predictions(np.zeros(10))
    truth = Truth(np.linspace(0.0, 2.0, 10))
    assert metrics.score(pred, truth)["r2"] < 0
    with pytest.raises(TypeError):
        metrics.score(truth, pred)


def test_the_legal_estimator_cannot_see_permeability():
    k = Permeability(np.linspace(1.0, 100.0, 50))
    with pytest.raises(TypeError):
        strategies.cutpoints_from_porosity(k)


# --------------------------------------------------------- syntax-tree scans
def test_only_the_data_layer_opens_a_measurement_file():
    callers = set()
    for name in os.listdir(LAYER_DIR):
        if not name.endswith(".py"):
            continue
        tree = _tree(__import__(f"zeroleak.layers.{name[:-3]}", fromlist=["x"]))
        if {"open", "read_csv", "loadtxt", "genfromtxt"} & _called_names(tree):
            callers.add(name)
    assert callers == {"data.py", "report.py"}, callers


def test_model_contains_no_estimator_call():
    names = _called_names(_tree(model))
    bad = names & set(model.FORBIDDEN_CALLS)
    assert not bad, f"model.py calls {bad}; cutpoints must arrive as configuration"


def test_report_imports_neither_numpy_nor_the_model():
    imports = _imported(report)
    joined = " ".join(imports)
    assert "numpy" not in joined
    assert "model" not in joined


def test_metrics_does_not_import_the_model():
    imports = _imported(metrics)
    assert not any("model" in s for s in imports), imports


# --------------------------------------------------------------- behaviour
def test_fold_labels_do_not_carry_their_boundary():
    phi = Porosity(np.linspace(0.05, 0.40, 100))
    depth = Depth(np.linspace(3000.0, 3100.0, 100))
    lab = folds.assign_folds(phi, depth, rule="depth_block", n_blocks=5)
    assert len(set(np.asarray(lab.x).tolist())) == 5
    assert not hasattr(lab, "boundaries")
    assert not hasattr(lab, "edges")


def test_training_mask_excludes_exactly_one_fold():
    phi = Porosity(np.linspace(0.05, 0.40, 100))
    depth = Depth(np.linspace(3000.0, 3100.0, 100))
    lab = folds.assign_folds(phi, depth, rule="depth_block", n_blocks=4)
    for f in np.unique(lab.x):
        keep = lab.training(int(f))
        assert int((~keep).sum()) == int((lab.x == f).sum())
