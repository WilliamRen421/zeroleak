"""The predictor registry, and the one comparison a type system cannot make.

`requires` answers what a call may *receive*.  It cannot answer what the method
*says* it uses, because the claim lives in prose, in a paper, in a caption.  The
two can disagree, and when they do the prose is the leak: a predictor described
as a porosity criterion whose signature also takes permeability reads, to a
reader, as a criterion that porosity alone can evaluate.

This module makes the two statements comparable.  A predictor is registered with
a declared information set -- the quantities its description claims -- and the
registry reads the implemented set off `__requires__`, expanding each carrier
through `CARRIES`.  `check` then reports the difference.  Nothing is parsed and
nothing is inferred from a body: the implemented set is the contract, and the
contract is what the type system already enforces.

A token is ``quantity:provenance``, where provenance is ``prior``, ``target`` or
``train``.  Only the quantity half is checkable, and that asymmetry is the
honest half of the result: a signature can say *permeability*, it cannot say
*whether that permeability is the target's*.  The registry therefore checks what
can be checked and keeps the provenance visible in the declaration, so that the
part a machine cannot verify is the part a reader is asked to read.

    >>> from zeroleak.registry import REGISTRY, check
    >>> [r["status"] for r in check(REGISTRY)]
    ['ok', 'ok']
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Sequence, Tuple

from .layers import model
from .strategies import cutpoints_from_config, cutpoints_from_porosity
from .types import CARRIES

__all__ = [
    "Predictor",
    "REGISTRY",
    "CARRIED_BY",
    "signature_set",
    "check",
    "table",
]


# Which quantities a carrier exposes.  This is `CARRIES` split into tokens; the
# prose in `CARRIES` stays the single source of truth and the assertion below
# keeps the two from drifting apart.
CARRIED_BY: Dict[str, FrozenSet[str]] = {
    "Porosity": frozenset({"porosity"}),
    "Permeability": frozenset({"permeability"}),
    "Depth": frozenset({"depth"}),
    "FoldLabels": frozenset({"folds"}),
    "Cutpoints": frozenset({"cutpoints"}),
    "Coefficients": frozenset({"coefficients"}),
    "Predictions": frozenset({"predictions"}),
    "Truth": frozenset({"truth"}),
    "Metrics": frozenset({"metrics"}),
    # The trap: one carrier that holds all three columns of a file.  A function
    # that asks for it can see permeability without naming it.
    "Body": frozenset({"porosity", "permeability", "depth"}),
}

assert set(CARRIED_BY) == set(CARRIES), (
    "every carrier in CARRIES needs a token set, or the registry would silently "
    "skip a type"
)

# A quantity whose presence makes a predictor unavailable at a field without
# permeability.  `truth` is the same information as `permeability`, scored.
_UNAVAILABLE_AT_TARGET = frozenset({"permeability", "truth"})


@dataclass(frozen=True)
class Predictor:
    """One predictor and the information set its description claims.

    `contract` is the guarded function whose signature defines what is really
    read.  A prediction has two halves -- a partition and a fit -- and both are
    registered, because the two halves can see different things and it is the
    difference that the protocol is made of.
    """

    name: str
    declares: FrozenSet[str]
    contract: Callable
    note: str = ""


def _quantity(token: str) -> str:
    return token.split(":", 1)[0]


def signature_set(fn: Callable) -> FrozenSet[str]:
    """The quantities a guarded function can see, read from its contract."""
    spec = getattr(fn, "__requires__", None)
    if spec is None:
        raise TypeError(
            f"{getattr(fn, '__qualname__', fn)} carries no `requires` contract; "
            "a predictor whose information set is not declared in its signature "
            "cannot be registered"
        )
    out: FrozenSet[str] = frozenset()
    for carrier in spec.values():
        name = carrier.__name__
        if name not in CARRIED_BY:
            raise TypeError(f"unknown carrier {name!r} in {fn.__qualname__}")
        out = out | CARRIED_BY[name]
    return out


def check(predictors: Iterable[Predictor] = ()) -> List[Dict]:
    """Compare every declaration with its signature.

    Three outcomes.  `ok`: the two sets agree on every quantity.  `conservative`:
    the declaration lists a quantity the signature does not take, which costs
    nothing and is reported rather than corrected.  `under-declared`: the
    signature can see a quantity the description does not mention -- the FZI
    family's shape, and the only outcome that changes what the paper may claim.
    """
    rows: List[Dict] = []
    for p in predictors:
        derived = signature_set(p.contract)
        declared_q = frozenset(_quantity(t) for t in p.declares)
        missing = derived - declared_q
        extra = declared_q - derived
        if missing:
            status = "under-declared"
        elif extra:
            status = "conservative"
        else:
            status = "ok"
        rows.append(
            {
                "predictor": p.name,
                "declared": tuple(sorted(p.declares)),
                "derived": tuple(sorted(derived)),
                "status": status,
                "undeclared": tuple(sorted(missing)),
                "available_at_target": not (derived & _UNAVAILABLE_AT_TARGET),
                "note": p.note,
            }
        )
    return rows


def table(rows: Sequence[Dict]) -> str:
    """Fixed-width table, because the console output is the demonstration.

    Column widths are measured, not guessed: a declaration that is longer than
    its column would otherwise run into the implemented set and hide the very
    difference the table exists to show.
    """
    cells = [
        [r["predictor"] for r in rows],
        [", ".join(r["declared"]) or "(nothing)" for r in rows],
        [", ".join(r["derived"]) or "(nothing)" for r in rows],
        [r["status"] for r in rows],
        ["yes" if r["available_at_target"] else "no" for r in rows],
    ]
    heads = ["predictor", "declared", "derived", "status", "target?"]
    width = [max(len(h), *(len(v) for v in col)) + 2 for h, col in zip(heads, cells)]
    head = "".join(h.ljust(w) for h, w in zip(heads, width)).rstrip()
    lines = [head, "-" * len(head)]
    for i in range(len(rows)):
        lines.append("".join(col[i].ljust(w) for col, w in zip(cells, width)).rstrip())
    return "\n".join(lines)


# The predictors the paper evaluates.  `target_terciles` is the interesting
# legal row: porosity only, and still the target's distribution.  The fit is the
# row where permeability is admitted, which is why it is marked train-only: the
# same signature at deployment would be a leak, and the provenance token is the
# only thing that distinguishes them.
REGISTRY: Tuple[Predictor, ...] = (
    Predictor(
        name="prior",
        declares=frozenset(),
        contract=cutpoints_from_config,
        note="boundaries from a file written before the target was read",
    ),
    Predictor(
        name="target_terciles",
        declares=frozenset({"porosity:target"}),
        contract=cutpoints_from_porosity,
        note="porosity only, but from the target's own distribution",
    ),
    Predictor(
        name="free_exponent",
        declares=frozenset(
            {"porosity:train", "permeability:train", "folds:train", "cutpoints:prior"}
        ),
        contract=model.fit,
        note="one slope per phase, fitted inside each training fold",
    ),
)
