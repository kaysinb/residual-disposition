"""The size block's rules and arithmetic, without model code (docs/preregistration.md §13.9 as
amended on 2026-11-05; ``configs/continuation/size_rules.yaml``).

Two sizes, Qwen3-4B and Qwen3-14B, with two independent tldr implantations at each: 4B-1, 4B-2,
the existing anchor 14B-1 (unchanged) and 14B-2.

- **Seed namespaces.** Every new implantation has an integer offset o: its implant and
  suppression training seed and its generation seed are o, and the relearning seed of its block,
  calibration or pilot index j is o + j (``relearn_seed``). The anchor's offset is 0, so its
  recorded runs keep their seeds; no two implantations share a random stream.
- **Schedules.** The common evaluation schedule (every 2 steps to 20, every 5 to 100, then every
  10) and the dense one, its union with every step through 30 and every even step from 32 through
  100 (``dense_steps``). Crossing times, certificates and the stop use the common schedule's
  evaluations only (``common_crossing``); the dense schedule's T is a sensitivity.
- **Calibration** (``calibrate``): H_s relearned on b, Y and Y₂ at both rates with the
  calibration seeds, each capped at 600 steps; the median of three per arm and rate, unresolved
  with two or three censored runs; k* per rate from b's median; B = 50·⌈max(20, k*s, 3·max of the
  six medians)/50⌉, unresolved above the ceiling; the dense schedule when the smallest median is
  below the trigger.
- **Certificate** (``certificate``): per rate, the sign test of S_b, Q_Y and Q_Y₂ over the blocks
  (K ≥ 9 of 10, or 6 of 6) with its interval ([l₍₂₎, u₍₉₎] for ten blocks, [min l, max u] for six).
- **Analysis** (``analysis``): per implantation the block means of S_b (= D_b), Q_Y and Q_Y₂ at
  each rate with identification bounds; each size's point the equal-weight mean of its two
  implantations; the size difference, the four cross-implant comparisons and the two
  within-size gaps; one joint block bootstrap (blocks resampled within each implantation, the
  same resample used for every comparison).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .continuation_stats import (
    Block,
    Bounds,
    Crossing,
    binomial_sf,
    certificate_contrasts,
)
from .paths import REPO_ROOT
from .relearn_rules import interpolated_crossing

SIZE_RULES_PATH = REPO_ROOT / "configs" / "continuation" / "size_rules.yaml"
# B6 (§13.15, §13.17): the rules file's sha256, frozen before any training of a new implantation
SIZE_RULES_FROZEN = "afda7b050883395e2cf9abb7e08411f445457850d2b42f66d8d575de0d14a444"
HABITS = ("b", "Y", "Y2")
ROLES = ("L", "H_s")
ESTIMANDS = ("S_b", "Q_Y", "Q_Y2")


def load_size_rules(
    path: Path = SIZE_RULES_PATH, *, expected_hash: str | None = SIZE_RULES_FROZEN
) -> dict[str, Any]:
    """The size block's rules, refused if the file is not the frozen one (``expected_hash``;
    None for a test's own file), and checked for the design's invariants."""
    import hashlib

    blob = Path(path).read_bytes()
    if expected_hash is not None and hashlib.sha256(blob).hexdigest() != expected_hash:
        raise ValueError(f"{path}: not the frozen size rules (B6)")
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    rules = data["size"]
    impl = rules["implantations"]
    offsets = [int(v["offset"]) for v in impl.values()]
    if len(set(offsets)) != len(offsets):
        raise ValueError("every implantation has its own seed offset")
    for name, v in impl.items():
        used = [*v["blocks"], *rules["calibration_seeds"], *v.get("pilot", [])]
        if len(set(used)) != len(used):
            raise ValueError(f"{name}: block, calibration and pilot indices overlap")
        if str(len(v["blocks"])) not in rules["sign_pass"]:
            raise ValueError(f"{name}: no sign rule for {len(v['blocks'])} blocks")
    spans = sorted(
        (
            int(v["offset"]),
            int(v["offset"])
            + max([*v["blocks"], *rules["calibration_seeds"], *v.get("pilot", [])]),
        )
        for v in impl.values()
    )
    if any(a[1] >= b[0] for a, b in zip(spans, spans[1:], strict=False)):
        raise ValueError("seed namespaces overlap")
    sizes = rules["sizes"]
    if sorted(n for names in sizes.values() for n in names) != sorted(impl):
        raise ValueError("every implantation belongs to exactly one size")
    if any(len(names) != 2 for names in sizes.values()):
        raise ValueError("two implantations per size")
    if rules["anchor"] not in impl or not impl[rules["anchor"]].get("existing"):
        raise ValueError("the anchor is the existing implantation")
    if int(rules["ceiling"]) not in (600, 1200):
        raise ValueError("the ceiling is 600 or 1,200 steps")
    return rules


# ---------------------------------------------------------------- seeds


def offset(rules: Mapping[str, Any], name: str) -> int:
    return int(rules["implantations"][name]["offset"])


def relearn_seed(rules: Mapping[str, Any], name: str, index: int) -> int:
    """The relearning seed of block, calibration or pilot index ``index`` of implantation
    ``name``: its offset plus the index."""
    v = rules["implantations"][name]
    allowed = {*v["blocks"], *rules["calibration_seeds"], *v.get("pilot", [])}
    if int(index) not in allowed:
        raise ValueError(f"{name}: index {index} is no block, calibration or pilot seed")
    return offset(rules, name) + int(index)


# ---------------------------------------------------------------- schedules


def common_steps(rules: Mapping[str, Any], budget: int) -> list[int]:
    """The common schedule's evaluation steps up to ``budget`` (the budget included), step 0
    left out (``certificate.RelearnSettings.evaluation_steps``)."""
    steps, start = set(), 0
    for until, every in rules["common_schedule"]:
        end = budget if until is None else min(int(until), budget)
        steps.update(range(start + int(every), end + 1, int(every)))
        start = end
        if end >= budget:
            break
    return sorted(steps | {int(budget)})


def dense_steps(rules: Mapping[str, Any], budget: int) -> list[int]:
    """The dense schedule: the common schedule's steps, every integer step through 30 and every
    even step from 32 through 100, up to ``budget``."""
    d = rules["dense"]
    extra = set(range(1, int(d["every_step_through"]) + 1))
    extra |= set(range(int(d["even_step_from"]), int(d["even_step_through"]) + 1, 2))
    return sorted(set(common_steps(rules, budget)) | {s for s in extra if s <= budget})


def extra_steps(rules: Mapping[str, Any], budget: int) -> list[int]:
    """The dense schedule's evaluations that are not the common schedule's."""
    return sorted(set(dense_steps(rules, budget)) - set(common_steps(rules, budget)))


def common_crossing(
    evals: Mapping[int, float], target: float, common: Sequence[int], stop_after: int | None
) -> dict[str, Any]:
    """A run's crossing from its evaluations' rates (``evals``: step → rate, step 0 included),
    the common schedule's alone: the first common step at or above ``target`` (latched), T
    interpolated between it and the previous common evaluation, and the step at which the
    latched stop (the first common evaluation at or after ``stop_after`` with a latched crossing)
    ends the run; censored when no common evaluation crosses."""
    steps = [0, *[s for s in sorted(common) if s in evals]]
    rates = [evals[s] for s in steps]
    crossing = next((s for s, r in zip(steps, rates, strict=True) if r >= target), None)
    stop = None
    if crossing is not None:
        floor = crossing if stop_after is None else max(crossing, stop_after)
        stop = next((s for s in steps if s >= floor), None)
    t = None
    if crossing is not None:
        t = interpolated_crossing(steps, rates, crossing, target)
        if t is None:
            t = float(crossing)
    return {"crossing_step": crossing, "t": t, "censored": crossing is None, "stop": stop}


# ---------------------------------------------------------------- calibration


def median_of_three(times: Sequence[float]) -> float | None:
    """The median of three crossing times (+inf for a censored run); None with two or three
    censored: unresolved."""
    if len(times) != 3:
        raise ValueError(f"{len(times)} calibration runs, not 3")
    middle = sorted(times)[1]
    return None if middle == math.inf else middle


def nearest_step(target: Fraction, steps: Sequence[int]) -> int:
    """The step nearest ``target``; a tie goes to the earlier step."""
    return min(sorted(set(steps)), key=lambda s: (abs(Fraction(s) - target), s))


def calibrate(
    rules: Mapping[str, Any], times: Mapping[tuple[str, float], Sequence[float]]
) -> dict[str, Any]:
    """k* per rate, B and the dense-schedule decision from the calibration's crossing times,
    ``times[(habit, rate)]`` the three H_s runs' T in seed order (+inf censored at the cap).
    Status "resolved", or "unresolved" with its reasons (a median censored, or B above the
    ceiling); unresolved gives no k* and no B."""
    rates = [float(lr) for lr in rules["learning_rates"]]
    want = {(h, lr) for h in HABITS for lr in rates}
    if set(times) != want:
        raise ValueError(f"calibration arms {sorted(times)}, not {sorted(want)}")
    medians = {key: median_of_three(list(v)) for key, v in times.items()}
    reasons = [f"median of {h} at {lr:g} censored" for (h, lr), m in sorted(medians.items())
               if m is None]  # fmt: skip
    out: dict[str, Any] = {
        "medians": {f"{h}/{lr:g}": m for (h, lr), m in sorted(medians.items())},
        "kstar": None, "budget": None, "dense": None,
    }  # fmt: skip
    if reasons:
        return {**out, "status": "unresolved", "reasons": reasons}
    cap = int(rules["calibration_cap"])
    steps = [0, *common_steps(rules, cap)]
    fraction = Fraction(str(rules["kstar_fraction"]))
    kstar = {lr: nearest_step(Fraction(medians[("b", lr)]) * fraction, steps) for lr in rates}
    need = max(
        Fraction(int(rules["budget_floor"])),
        *[Fraction(k) for k in kstar.values()],
        int(rules["budget_factor"]) * max(Fraction(m) for m in medians.values()),
    )
    multiple = int(rules["budget_multiple"])
    budget = multiple * math.ceil(need / multiple)
    smallest = min(medians.values())
    dense = smallest < float(rules["dense"]["trigger"])
    out.update(kstar={f"{lr:g}": k for lr, k in kstar.items()}, budget=budget, dense=dense,
               smallest_median=smallest)  # fmt: skip
    if budget > int(rules["ceiling"]):
        return {**out, "status": "unresolved", "reasons": [f"B {budget} above the ceiling"]}
    return {**out, "status": "resolved", "reasons": []}


# ---------------------------------------------------------------- floors


def floor_problems(rules: Mapping[str, Any], rates: Mapping[str, Fraction]) -> list[str]:
    """The implantation floors (§13.9): A, Y and Y₂ at least half on their reference streams, H at
    most 1/50; a problem per missed floor (the implantation is then infeasible)."""
    f = rules["implant_floors"]
    low, high = Fraction(str(f["reference_min"])), Fraction(str(f["sham_max"]))
    problems = [f"{k} reference {rates[k]} < {low}" for k in ("A", "Y", "Y2") if rates[k] < low]
    if rates["H"] > high:
        problems.append(f"H rate {rates['H']} > {high}")
    return problems


# ---------------------------------------------------------------- certificate


def sign_pass(rules: Mapping[str, Any], n: int) -> int:
    return int(rules["sign_pass"][str(n)])


def interval(rules: Mapping[str, Any], contrasts: Sequence[Bounds]) -> tuple[float, float]:
    """The block interval in the extended reals: ten blocks [l₍₂₎, u₍₉₎] (lower and upper ends
    sorted separately, infinities kept, so one block open on both sides leaves the order
    statistics finite when the others are), six blocks [min l, max u] (§12.4.8)."""
    lows, highs = sorted(c.lower for c in contrasts), sorted(c.upper for c in contrasts)
    if len(contrasts) == 10:
        return lows[1], highs[8]
    if len(contrasts) == 6:
        return lows[0], highs[-1]
    raise ValueError(f"no interval rule for {len(contrasts)} blocks")


def certificate(
    rules: Mapping[str, Any], blocks: Mapping[float, Sequence[Block]]
) -> dict[str, Any]:
    """The implantation's sign certificate at each rate: for S_b, Q_Y and Q_Y₂, K (blocks
    certainly positive), p = P{Bin(n, 1/2) ≥ K}, pass iff K reaches the frozen count, and the
    interval; a missing block makes that contrast incomplete (p = 1, no pass)."""
    out: dict[str, Any] = {}
    for lr, rows in blocks.items():
        contrasts = certificate_contrasts(list(rows))
        n = len(rows)
        need = sign_pass(rules, n)
        per: dict[str, Any] = {}
        for name in ESTIMANDS:
            cs = contrasts[name]
            if any(c is None for c in cs):
                per[name] = {"n": n, "k": None, "p": "1", "passed": False, "complete": False,
                             "interval": None}  # fmt: skip
                continue
            k = sum(1 for c in cs if c.certainly_positive)
            p = binomial_sf(k, n, Fraction(1, 2))
            per[name] = {"n": n, "k": k, "p": str(p), "passed": k >= need, "complete": True,
                         "interval": interval(rules, cs)}  # fmt: skip
        out[f"{float(lr):g}"] = per
    return out


def block_of(arms: Mapping[tuple[str, str], Mapping[str, Any] | None], block_id: int) -> Block:
    """One block from its six runs' summaries (``crossing_step``, ``t``, ``censored``,
    ``budget``; None for a run not completed)."""

    def crossing(s: Mapping[str, Any] | None) -> Crossing:
        if s is None:
            return Crossing.incomplete()
        if s.get("censored"):
            return Crossing.censored(float(s["budget"]))
        return Crossing.observed(max(1.0, float(s["t"])))

    return Block(arms={(h, r): crossing(arms.get((h, r))) for h in HABITS for r in ROLES},
                 block_id=block_id)  # fmt: skip


# ---------------------------------------------------------------- analysis


@dataclass(frozen=True)
class Series:
    """One implantation's per-block contrast at one rate: lower and upper log ends (equal where
    observed; ±inf where a censored run leaves an end open)."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @classmethod
    def of(cls, contrasts: Sequence[Bounds | None]) -> Series:
        if any(c is None for c in contrasts):
            raise ValueError("an incomplete block: the implantation's analysis waits")
        return cls(tuple(c.lower for c in contrasts), tuple(c.upper for c in contrasts))  # type: ignore[union-attr]

    @property
    def observed(self) -> bool:
        return self.lower == self.upper

    def mean(self, idx: Sequence[int] | None = None) -> tuple[float, float]:
        idx = range(len(self.lower)) if idx is None else idx
        lo = [self.lower[i] for i in idx]
        hi = [self.upper[i] for i in idx]
        return sum(lo) / len(lo), sum(hi) / len(hi)


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Bounds of a − b from the bounds of a and of b."""
    return a[0] - b[1], a[1] - b[0]


def _avg(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _present(rules: Mapping[str, Any], names: Sequence[str]) -> tuple[list[str], list[str]]:
    """The 4B and 14B implantations among ``names``, in the design's order."""
    sizes = rules["sizes"]
    return [n for n in sizes["4B"] if n in names], [n for n in sizes["14B"] if n in names]


def _mean_of(bounds: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (sum(b[0] for b in bounds) / len(bounds), sum(b[1] for b in bounds) / len(bounds))


def comparisons(
    rules: Mapping[str, Any], means: Mapping[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """From each implantation's block-mean bounds: the size points (equal-weight means of the
    size's implantations), the size difference (14B − 4B), the cross-implant comparisons (every
    14B implantation minus every 4B one) and the within-size gaps (second minus first), each as
    bounds. With all four implantations: four cross comparisons and two gaps; with fewer (a
    reduced design), those the present implantations allow."""
    small, large = _present(rules, list(means))
    out: dict[str, tuple[float, float]] = {}
    if small:
        out["4B"] = _mean_of([means[n] for n in small])
    if large:
        out["14B"] = _mean_of([means[n] for n in large])
    if small and large:
        out["14B-4B"] = _sub(out["14B"], out["4B"])
    for b in large:
        for a in small:
            out[f"{b}-{a}"] = _sub(means[b], means[a])
    for pair in (small, large):
        if len(pair) == 2:
            out[f"gap:{pair[0]}-{pair[1]}"] = _sub(means[pair[0]], means[pair[1]])
    return out


def _percentile(values: Sequence[float], q: float, side: str) -> float:
    """The q-quantile of extended reals (linear interpolation between finite neighbours). Where a
    neighbour is infinite the interpolation is not formed (no inf − inf): a lower end takes the
    smaller neighbour and an upper end the larger, so the interval only widens."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    a, b = xs[lo], xs[hi]
    if lo == hi or a == b:
        return a
    if math.isinf(a) or math.isinf(b):
        return a if side == "lower" else b
    return a + (b - a) * (pos - lo)


def bootstrap(
    rules: Mapping[str, Any], series: Mapping[str, Series], *, resamples: int | None = None,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:  # fmt: skip
    """95 % percentile intervals of every comparison from one joint block bootstrap: each
    resample draws blocks with replacement within each implantation independently, and the same
    draw serves every comparison (a resampled 14B implantation is shared by the size difference,
    its cross comparisons and its gap). With censoring, the lower and upper ends are resampled
    together and the interval runs from the 2.5 % point of the lower end to the 97.5 % point of
    the upper end."""
    b = rules["bootstrap"]
    resamples = int(resamples if resamples is not None else b["resamples"])
    rng = random.Random(int(seed if seed is not None else b["seed"]))
    names = sorted(series)
    draws: dict[str, list[tuple[float, float]]] = {}
    for _ in range(resamples):
        means = {}
        for name in names:
            n = len(series[name].lower)
            means[name] = series[name].mean([rng.randrange(n) for _ in range(n)])
        for key, (lo, hi) in comparisons(rules, means).items():
            draws.setdefault(key, []).append((lo, hi))
    return {key: {"lower": _percentile([d[0] for d in v], 0.025, "lower"),
                  "upper": _percentile([d[1] for d in v], 0.975, "upper")}
            for key, v in draws.items()}  # fmt: skip


REDUCED_NOTE = "reduced design: the size statement needs all four cross comparisons (§13.17)"


def statement(
    rules: Mapping[str, Any], point: Mapping[str, tuple[float, float]],
    intervals: Mapping[str, Mapping[str, float]], *, full: bool = True,
) -> dict[str, Any]:  # fmt: skip
    """The wording facts of one estimand at one rate (§6 of the design): whether the size
    difference's interval excludes zero and in which direction, whether the cross-implant point
    estimates (four in the full design) share that direction (their individual significance is
    not required), which cross comparisons' intervals exclude zero, and whether the size
    separation exceeds each observed within-size gap (point estimates; bounds where censored).
    The size statement needs the full design (§13.17: all four cross comparisons); a reduced
    design (``full`` false) keeps its estimates, intervals and directions, and the statement is
    unavailable (``size_statement`` false, ``size_statement_available`` false)."""
    if "14B-4B" not in point:
        return {"size_statement": False, "size_statement_available": False,
                "difference_excludes_zero": None, "direction": None,
                "note": "no size difference: a size without an implantation"}  # fmt: skip
    diff = intervals["14B-4B"]
    excludes = diff["lower"] > 0 or diff["upper"] < 0
    direction = 1 if diff["lower"] > 0 else -1 if diff["upper"] < 0 else 0
    cross = [c for c in point if c not in ("4B", "14B", "14B-4B") and not c.startswith("gap:")]

    def sign(bounds: tuple[float, float]) -> int:
        return 1 if bounds[0] > 0 else -1 if bounds[1] < 0 else 0

    signs = {c: sign(point[c]) for c in cross}
    same = direction != 0 and all(s == direction for s in signs.values())
    significant = {c: intervals[c]["lower"] > 0 or intervals[c]["upper"] < 0 for c in cross}
    sep = point["14B-4B"]
    sep_min = 0.0 if sep[0] <= 0 <= sep[1] else min(abs(sep[0]), abs(sep[1]))
    gaps = {g: max(abs(point[g][0]), abs(point[g][1])) for g in point if g.startswith("gap:")}
    return {
        "difference_excludes_zero": excludes,
        "direction": {1: "larger at 14B", -1: "larger at 4B", 0: None}[direction],
        "cross_point_signs": signs,
        "cross_same_direction": same,
        "cross_significant": significant,
        "separation_exceeds_gap": {g: sep_min > v for g, v in gaps.items()},
        "size_statement": full and excludes and same,
        "size_statement_available": full,
        **({} if full else {"note": REDUCED_NOTE}),
    }


def analysis(
    rules: Mapping[str, Any], contrasts: Mapping[str, Mapping[float, Mapping[str, Sequence]]],
    *, resamples: int | None = None, reduced: bool = False,
) -> dict[str, Any]:  # fmt: skip
    """The size comparison: ``contrasts[implantation][rate][estimand]`` the per-block bounds
    (``certificate_contrasts``). For each rate and estimand: each implantation's block-mean
    bounds, the comparisons (``comparisons``), their joint-bootstrap intervals and the wording
    facts (``statement``). Every implantation of the design must be present unless ``reduced``:
    then the analysis is labeled reduced, names the missing implantations and uses only the
    present ones (a size's point is the mean of its present implantations), never substituting
    for the planned mean."""
    impl = sorted(rules["implantations"])
    present = sorted(contrasts)
    if set(present) - set(impl):
        raise ValueError(f"implantations outside the design: {sorted(set(present) - set(impl))}")
    missing = sorted(set(impl) - set(present))
    if missing and not reduced:
        raise ValueError(f"the analysis needs all of {impl}; a reduced analysis is labeled apart")
    if not present:
        raise ValueError("no implantation")
    out: dict[str, Any] = {}
    for lr in [float(x) for x in rules["learning_rates"]]:
        for est in ESTIMANDS:
            series = {n: Series.of(contrasts[n][lr][est]) for n in present}
            means = {n: s.mean() for n, s in series.items()}
            point = comparisons(rules, means)
            ints = bootstrap(rules, series, resamples=resamples)
            out[f"{lr:g}/{est}"] = {
                "design": "reduced" if missing else "full", "missing": missing,
                "means": means, "comparisons": point, "intervals": ints,
                "observed": all(s.observed for s in series.values()),
                "statement": statement(rules, point, ints, full=not missing),
            }  # fmt: skip
    return out


def absolute_crossing(
    rules: Mapping[str, Any], evals: Mapping[int, float], budget: int
) -> dict[str, Any]:
    """§13.9's absolute-rate sensitivity for one run: the first common-schedule evaluation at or
    above the frozen absolute rate (1/4) and T interpolated as the primary T, from the run's
    saved curve (``evals``: step → rate; dense-schedule extras are left out). A run that ended
    without reaching it is censored at its last evaluated step. Every run's own target (half its
    reference, at least 1/4 by the floors) is reached no earlier, so a run stopped at its own
    crossing has already passed 1/4."""
    target = float(Fraction(str(rules["absolute_crossing"])))
    common = [s for s in common_steps(rules, budget) if s in evals]
    got = common_crossing(evals, target, common, stop_after=None)
    last = max([0, *common])
    return {"crossing_step": got["crossing_step"], "t": got["t"], "censored": got["censored"],
            "budget": last if got["censored"] else budget}  # fmt: skip
