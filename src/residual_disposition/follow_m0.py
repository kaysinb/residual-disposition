"""M0 of the follow-up (docs/preregistration.md §13.10): the extraction check before any new lens
package, decided from the δ̄ that ``scripts/pod/follow_m0.py`` computes on one Pod.

Three methods, each run twice (``configs/continuation/follow_m0.yaml``):

- ``qualification``: ``adl.activation_difference``, the qualification's path;
- ``production_session``: the lens builder's state means minus base means recomputed in the same
  session;
- ``production_cached``: the same state means minus the stored base means.

**Decisive** (``verdict``): at the decisive layer, for every state, each method's second repeat
reproduces its first, and in each repeat ``production_session`` reproduces ``qualification``, every
position's relative L2 error within the follow-up rules' tolerance (2⁻⁸). **Diagnostic**, reported
and never a stop: ``production_cached`` against ``qualification``, and each method against the
stored qualification δ̄ from its original environment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from .adl_lens import check_reproduction

METHODS = ("qualification", "production_session", "production_cached")
DECISIVE = ("production_session", "qualification")


def verdict(
    deltas: Mapping[str, Mapping[str, Sequence[Any]]],
    tolerance: Fraction,
    *,
    historical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """M0's decision from ``deltas[state][method]`` = the two repeats' δ̄ at the decisive layer
    ([positions, hidden] tensors), with the diagnostic comparisons; ``historical[state]`` is the
    stored qualification δ̄, if any."""
    tol = float(tolerance)
    states, passed = {}, True
    for state, by_method in deltas.items():
        if set(by_method) != set(METHODS) or any(len(r) != 2 for r in by_method.values()):
            raise ValueError(f"{state}: every method twice, not {sorted(by_method)}")
        repeats = {m: check_reproduction(r[1], r[0], tolerance=tol) for m, r in by_method.items()}
        decisive = [check_reproduction(by_method[DECISIVE[0]][i], by_method[DECISIVE[1]][i],
                                       tolerance=tol) for i in (0, 1)]  # fmt: skip
        ok = all(r["passed"] for r in repeats.values()) and all(d["passed"] for d in decisive)
        diagnostic = {"cached_vs_qualification": [
            check_reproduction(by_method["production_cached"][i], by_method["qualification"][i],
                               tolerance=tol) for i in (0, 1)]}  # fmt: skip
        if historical is not None and state in historical:
            diagnostic["vs_historical"] = {
                m: check_reproduction(by_method[m][0], historical[state], tolerance=tol)
                for m in METHODS
            }
        states[state] = {"passed": ok, "repeats": repeats, "decisive": decisive,
                         "diagnostic": diagnostic}  # fmt: skip
        passed = passed and ok
    return {"passed": passed, "tolerance": str(tolerance), "decisive": list(DECISIVE),
            "states": states}  # fmt: skip
