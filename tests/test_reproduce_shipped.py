"""The shipped body and the shipped output.

The manuscript says the smallest body ships in full with its expected output, so
that the pipeline can be run and the tests passed without downloading the
multi-gigabyte archives.  This test is the check on that sentence: it reruns the
single command into a temporary directory and compares every exported number with
the committed one.
"""
from __future__ import annotations

import csv
import json
import os

import pytest

from zeroleak.run import main

COMPARED = (".pred.csv", ".metrics.csv", "index.csv")
TOL = 1e-9


def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def _same_number(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) <= TOL
    except ValueError:
        return a == b


def test_the_pipeline_regenerates_the_shipped_output(root, tmp_path, monkeypatch):
    monkeypatch.chdir(root)
    code = main(["--config", "configs/base.yaml", "--root", root, "--out", str(tmp_path)])
    assert code == 0

    names = [n for n in os.listdir(os.path.join(root, "expected")) if n.endswith(COMPARED)]
    assert names, "the repository ships no expected output"

    for name in names:
        got = _read(os.path.join(tmp_path, name))
        ref = _read(os.path.join(root, "expected", name))
        assert len(got) == len(ref), f"{name}: {len(got)} rows against {len(ref)}"
        for i, (line_got, line_ref) in enumerate(zip(got, ref)):
            assert len(line_got) == len(line_ref), f"{name} row {i}: column count"
            for a, b in zip(line_got, line_ref):
                assert _same_number(a, b), f"{name} row {i}: {a!r} against {b!r}"


def test_the_configuration_record_is_not_empty(root, tmp_path, monkeypatch):
    monkeypatch.chdir(root)
    main(["--config", "configs/base.yaml", "--root", root, "--out", str(tmp_path)])
    records = [n for n in os.listdir(tmp_path) if n.endswith(".config.json")]
    assert len(records) == 2
    for name in records:
        with open(os.path.join(tmp_path, name), encoding="utf-8") as fh:
            rec = json.load(fh)
        assert rec["resolved_cutpoints"] == [0.1, 0.2] or len(rec["resolved_cutpoints"]) == 2
        assert rec["data_file"]["rows_used"] == 557
        assert len(rec["data_file"]["sha256"]) == 64
        assert rec["environment"]["numpy"]
