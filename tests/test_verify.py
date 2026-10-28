"""The verify command, and the checks on the check.

A verification command that only ever prints "ok" is worse than none, so most of
these tests hand it a deliberately altered reference and require it to fail.  The
last one covers the single exclusion the comparison makes on purpose: the recorded
commit hash, which differs by construction between the reference and any rerun.
"""
from __future__ import annotations

import csv
import json
import os
import shutil

import pytest

from zeroleak import verify


def _copy_reference(root, tmp_path):
    dst = os.path.join(str(tmp_path), "reference")
    shutil.copytree(os.path.join(root, "expected"), dst)
    return dst


def _rewrite_rows(path, mutate):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    rows = mutate(rows)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def test_the_shipped_run_verifies_clean(root, tmp_path, monkeypatch):
    monkeypatch.chdir(root)
    code = verify.main([
        "--config", "configs/base.yaml",
        "--expected", os.path.join(root, "expected"),
        "--out", os.path.join(str(tmp_path), "scratch"),
    ])
    assert code == 0


def test_a_changed_prediction_is_caught(root, tmp_path):
    name = "volve_15_9_19a__prior.pred.csv"
    ref = _copy_reference(root, tmp_path)
    target = os.path.join(ref, name)

    def bump(rows):
        rows[7][5] = f"{float(rows[7][5]) + 1e-4:.9f}"
        return rows

    _rewrite_rows(target, bump)
    problems = verify.compare_pred(os.path.join(root, "expected", name), target)
    assert problems, "a prediction off by 1e-4 in log10(k) must not pass"
    assert "row 7" in problems[0]


def test_a_changed_metric_is_caught(root, tmp_path):
    name = "volve_15_9_19a__prior.metrics.csv"
    ref = _copy_reference(root, tmp_path)
    target = os.path.join(ref, name)

    def bump(rows):
        for row in rows:
            if row and row[0] == "r2":
                row[1] = "0.7208"
        return rows

    _rewrite_rows(target, bump)
    problems = verify.compare_metrics(os.path.join(root, "expected", name), target)
    assert any("r2" in p for p in problems)


def test_a_changed_cutpoint_is_caught(root, tmp_path):
    name = "volve_15_9_19a__prior.config.json"
    ref = _copy_reference(root, tmp_path)
    target = os.path.join(ref, name)
    with open(target, encoding="utf-8") as fh:
        record = json.load(fh)
    record["resolved_cutpoints"] = [0.12, 0.2]
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(record, fh)

    problems = verify.compare_config(os.path.join(root, "expected", name), target)
    assert any("resolved_cutpoints" in p for p in problems)


def test_the_recorded_commit_is_not_compared(root, tmp_path):
    name = "volve_15_9_19a__prior.config.json"
    ref = _copy_reference(root, tmp_path)
    target = os.path.join(ref, name)
    with open(target, encoding="utf-8") as fh:
        record = json.load(fh)
    record["environment"]["zeroleak_git"] = "0000000"
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(record, fh)

    assert verify.compare_config(os.path.join(root, "expected", name), target) == []


def test_a_missing_file_is_counted(root, tmp_path):
    empty = os.path.join(str(tmp_path), "scratch")
    os.makedirs(empty)
    failures, notes = verify.compare_tree(os.path.join(root, "expected"), empty)
    assert failures >= 1
    assert notes, "a run that produced nothing must be reported as a different file set"
