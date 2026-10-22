"""Layer 5, reporting.

Reads exported CSV and nothing else.  It cannot import the model, the metrics or
numpy, so a table cannot quietly differ from the run that produced it: the only
way to change a number here is to change the file it was read from.

`tests/test_layers.py` asserts both properties on this module's syntax tree.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

__all__ = ["read_metrics", "write_summary", "METRIC_SUFFIX"]

METRIC_SUFFIX = ".metrics.csv"


def read_metrics(directory: str) -> List[Dict[str, str]]:
    """Read every `<body>__<strategy>.metrics.csv` in a directory.

    The metrics files are written as statistic/value pairs, one per line, so a
    reader can diff two runs without a column order to remember.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    rows: List[Dict[str, str]] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(METRIC_SUFFIX):
            continue
        body, strategy = name[: -len(METRIC_SUFFIX)].split("__", 1)
        record = {"body": body, "strategy": strategy}
        with open(os.path.join(directory, name), encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) == 2 and row[0] != "statistic":
                    record[row[0]] = row[1]
        rows.append(record)
    return rows


def write_summary(rows: List[Dict[str, str]], path: str) -> str:
    """Write one table from already-exported rows."""
    if not rows:
        raise ValueError("nothing to write")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path
