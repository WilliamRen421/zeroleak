"""Allow `python -m zeroleak --config configs/base.yaml`
and `python -m zeroleak audit --config configs/base.yaml`."""
from __future__ import annotations

import sys

from .run import main


def dispatch(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "audit":
        from .audit import main as audit_main

        return audit_main(argv[1:])
    return main(argv)


if __name__ == "__main__":
    sys.exit(dispatch())
