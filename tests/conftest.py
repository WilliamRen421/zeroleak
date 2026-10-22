"""Shared fixtures.

The suite runs from the repository root, so the package is importable without an
install.  `ROOT` is the directory that holds `data/` and `expected/`.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def volve_path(root: str) -> str:
    return os.path.join(root, "data", "volve_15_9_19a.csv")
