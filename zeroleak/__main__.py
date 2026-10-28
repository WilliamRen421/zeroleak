"""Command line: `python -m zeroleak [run] --config ...`, `audit`, `scan`, `scan-loo`.

The default (no subcommand) is `run`, which writes the per-sample predictions and
the per-body metrics of Table 4.  `audit` measures the folds.  `scan` evaluates
the 951-cell cutpoint grid (Table 6 and Fig. 5) and `scan-loo` recomputes that
grid with each body removed in turn (Table S9).  The two scan commands must run
in that order, because the second reads the grid written by the first.
"""
from __future__ import annotations

import sys

from .run import main

USAGE = """usage: python -m zeroleak [command] --config CONFIG

commands:
  run       per-sample predictions and per-body metrics (Table 4); this is the default
  verify    run into a scratch directory and diff every cell against expected/
  audit     fold-distance audit of the shipped bodies
  scan      951-cell cutpoint grid, per-body losses (Table 6, Fig. 5)
  scan-loo  leave-one-body-out recomputation of that grid (Table S9)
"""


def dispatch(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(USAGE)
        return 0
    if argv and argv[0] == "run":
        return main(argv[1:])
    if argv and argv[0] == "verify":
        from .verify import main as verify_main

        return verify_main(argv[1:])
    if argv and argv[0] == "audit":
        from .audit import main as audit_main

        return audit_main(argv[1:])
    if argv and argv[0] == "scan":
        from .scan import main as scan_main

        return scan_main(argv[1:])
    if argv and argv[0] == "scan-loo":
        from .scan_loo import main as loo_main

        return loo_main(argv[1:])
    return main(argv)


if __name__ == "__main__":
    sys.exit(dispatch())
