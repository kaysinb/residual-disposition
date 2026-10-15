"""The continuation's state × procedure inventory and ceilings (preregistration §12.3.9).

    uv run python scripts/continuation_inventory.py [--out runs/continuation_inventory]

Enumerates, for each pre-set branch, every (stage, behavior, state, procedure) the plan runs, and
prices it from ``configs/continuation/inventory_units.yaml`` (measured pilot units; [estimate]
where not measured). Each line has a unique id, so no cost is counted twice. The branches:

- ``main``: Stages 0–3, Stage 4 not triggered, Stage 5 in continued mode, Stage 6, Y₂ from its
  first candidate, no detector revision;
- ``stage4``: the same with Stage 4 triggered;
- ``dpo``: Stage 5 by the DPO fallback;
- ``exception_a_link`` and ``exception_a_pun``: the Stage-5 slot used on a pilot behavior before
  Stage 3 (exception a), Stage 5 then not run;
- ``y2_candidates_2`` … ``_4``: Y₂ found only at its 2nd–4th candidate;
- ``revision``: an accepted detector revision, with fresh confirmation draws on every lineage;
- ``worst``: Stage 4, DPO, the 4th Y₂ candidate and a revision together.

Writes ``inventory.csv`` (every line of every branch), ``ceilings.md`` (per branch and stage: GPU
hours, API dollars, labels, cumulative dollars against the cap, the meter at each gate) and
``inventory.json``. Ceilings use the units' high values; the central column uses the medians.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition.budget import apply_cut_order
from residual_disposition.paths import CONFIG_ROOT, REPO_ROOT

UNITS_PATH = CONFIG_ROOT / "continuation" / "inventory_units.yaml"
PILOT = ("example_link", "pun_intended")
STAGE_ORDER = ("stage0", "stage1", "stage2", "exception_a", "stage3", "stage4", "stage5", "stage6")
STAGE_CUTS = {"stage6": "stage6", "stage5": "stage5", "stage4": "stage4"}
PD_INELIGIBLE = ("guillemets",)
CHECKPOINTS = 6
PD_ADJUDICATION_ITEMS = 200  # the selected top items of one latent state's package and refills
NEW_FORMAT_ITEMS = 200  # per new reader format (lens-only, PD-clean), §12.3.3


@dataclasses.dataclass(frozen=True)
class Line:
    line_id: str
    stage: str
    behavior: str
    state: str
    procedure: str
    unit: str
    count: float
    gpu_minutes: float
    gpu_minutes_high: float
    api_usd: float
    labels: int
    source: str


class Pricer:
    def __init__(self, units: dict[str, Any]) -> None:
        self.units = units
        price = units["price"]
        self.usd_per_minute = (
            float(price["gpu_usd_per_hour"]) * float(Fraction(price["idle_margin"])) / 60
        )

    def gpu(self, line_id, stage, behavior, state, procedure, unit, count=1.0) -> Line:
        u = self.units["gpu"][unit]
        return Line(
            line_id,
            stage,
            behavior,
            state,
            procedure,
            unit,
            count,
            u["minutes"] * count,
            u["high"] * count,
            0.0,
            0,
            u["source"],
        )

    def api(self, line_id, stage, behavior, state, procedure, unit, count=1.0) -> Line:
        u = self.units["api"][unit]
        return Line(
            line_id,
            stage,
            behavior,
            state,
            procedure,
            unit,
            count,
            0.0,
            0.0,
            u["usd"] * count,
            0,
            u["source"],
        )

    def labels(self, line_id, stage, behavior, procedure, items) -> Line:
        u = self.units["api"]["ensemble_item"]
        return Line(
            line_id,
            stage,
            behavior,
            "-",
            procedure,
            "ensemble_item",
            items,
            0.0,
            0.0,
            u["usd"] * items,
            int(items),
            u["source"],
        )

    def lump(self, line_id, stage, procedure, unit) -> Line:
        u = self.units["lump"][unit]
        return Line(
            line_id, stage, "-", "-", procedure, unit, 1, 0.0, 0.0, u["usd"], 0, u["source"]
        )

    def usd(self, line: Line, *, high: bool) -> float:
        minutes = line.gpu_minutes_high if high else line.gpu_minutes
        return minutes * self.usd_per_minute + line.api_usd


# ---------------------------------------------------------------------------------------------
# Building blocks per lineage


def suppression(p: Pricer, stage: str, b: str, *, mode: str = "stack") -> list[Line]:
    """Latent and sham_s to 800 steps, the recipe control, the exploratory observer at every
    checkpoint on L and H_s; continued mode builds X_cont at every checkpoint, DPO trains both
    states by DPO."""
    k = f"{stage}/{b}/suppress"
    lines = []
    if mode == "dpo":
        lines += [p.gpu(f"{k}/L", stage, b, "L", "DPO to 800", "dpo_train")]
        lines += [p.gpu(f"{k}/H_s", stage, b, "H_s", "DPO to 800", "dpo_train")]
    else:
        lines += [p.gpu(f"{k}/L", stage, b, "L", f"SFT {mode} to 800", "suppress_latent")]
        lines += [p.gpu(f"{k}/H_s", stage, b, "H_s", f"SFT {mode} to 800", "suppress_sham_s")]
    if mode == "stack":
        lines += [p.gpu(f"{k}/R", stage, b, "recipe", "recipe control", "suppress_recipe")]
    if mode == "continued":
        lines += [
            p.gpu(
                f"{k}/X_cont",
                stage,
                b,
                "X_cont",
                "build and validate X_cont",
                "xcont_build",
                CHECKPOINTS,
            )
        ]
    for state in ("L", "H_s"):
        lines += [
            p.gpu(
                f"{k}/explore/{state}",
                stage,
                b,
                state,
                "exploratory observer, 6 checkpoints",
                "observer_pass",
                CHECKPOINTS,
            )
        ]
    return lines


def dose(p: Pricer, stage: str, b: str, *, x: str = "X") -> list[Line]:
    """Confirmation on 600 fresh prompts at d₁ and, in the worst case, attempt 2; X at the dose."""
    k = f"{stage}/{b}/dose"
    lines = [
        p.gpu(f"{k}/confirm/{s}", stage, b, s, "confirmation, 2 attempts", "observer_pass", 2)
        for s in ("L", "H_s")
    ]
    if x == "X":
        lines += [p.gpu(f"{k}/compose_x", stage, b, "X", "compose X", "compose_x")]
    return lines


def certificate(p: Pricer, stage: str, b: str, rules: cr.Rules) -> list[Line]:
    """Six blocks × (b, Y, Y₂) × (L, H_s) × two rates, the k* calibration on three seeds and the
    relearning-rate calibration."""
    cert = rules["certificate"]
    blocks, rates = int(cert["blocks"]), len(cert["learning_rates"])
    k = f"{stage}/{b}/certificate"
    lines = []
    for habit in cert["arms"]:
        for state in ("L", "H_s"):
            lines += [
                p.gpu(
                    f"{k}/{habit}/{state}",
                    stage,
                    b,
                    state,
                    f"relearn {habit}",
                    "relearn_run",
                    blocks * rates,
                )
            ]
    lines += [p.gpu(f"{k}/kstar", stage, b, "L", "k* calibration, 3 seeds", "relearn_run", 3)]
    lines += [p.gpu(f"{k}/lr", stage, b, "L", "relearning-rate calibration", "lr_calibration")]
    return lines


def reading(p: Pricer, stage: str, b: str, *, x: str = "X") -> list[Line]:
    """Every §12.4 reading at the analysis dose: PD-clean on L, H_s and X with its positive
    control on A and the recipe control and the random-removal control; lens-only at two layers
    on L, H_s, X, A, H and the recipe control, with fidelity (on A) and stability (on L); the
    expression checks on L (ADL corpus, OOD observer on L and H_s)."""
    k = f"{stage}/{b}/read"
    lines = []
    if b not in PD_INELIGIBLE:
        for s in ("L", "H_s", x, "A", "recipe"):
            lines += [p.gpu(f"{k}/pd/{s}", stage, b, s, "PD completions", "tool_pd")]
            lines += [
                p.api(f"{k}/pd_clean/{s}", stage, b, s, "PD-clean reading", "pd_clean_reading")
            ]
        for s in ("L", "A", "H_s"):
            lines += [
                p.api(f"{k}/pd_random/{s}", stage, b, s, "random-removal control", "pd_reading")
            ]
        lines += [
            p.labels(
                f"{k}/pd_adjudication", stage, b, "PD-clean adjudication", PD_ADJUDICATION_ITEMS
            )
        ]
    for s in ("L", "H_s", x, "A", "H", "recipe"):
        lines += [
            p.gpu(f"{k}/adl/{s}", stage, b, s, "lens-only package, layers 20 and 30", "adl_package")
        ]
        lines += [p.api(f"{k}/adl_api/{s}", stage, b, s, "lens-only reading", "adl_reading")]
    lines += [p.gpu(f"{k}/fidelity", stage, b, "A", "fidelity controls", "adl_package")]
    lines += [p.gpu(f"{k}/stability", stage, b, "L", "second λ draw", "adl_package")]
    lines += [
        p.gpu(f"{k}/corpus", stage, b, "L", "ADL-corpus expression check", "adl_corpus_check")
    ]
    lines += [
        p.gpu(f"{k}/ood/{s}", stage, b, s, "OOD observer", "ood_observer") for s in ("L", "H_s")
    ]
    return lines


def lineage(p: Pricer, stage: str, b: str, rules: cr.Rules, *, mode: str = "stack") -> list[Line]:
    x = "X_cont" if mode == "continued" else "X"
    return (
        suppression(p, stage, b, mode=mode)
        + dose(p, stage, b, x=x)
        + certificate(p, stage, b, rules)
        + reading(p, stage, b, x=x)
    )


# ---------------------------------------------------------------------------------------------
# Stages


def stage1(p: Pricer) -> list[Line]:
    s = "stage1"
    lines = [p.gpu(f"{s}/setup", s, "-", "-", "Pod session", "pod_setup")]
    lines += [
        p.gpu(
            f"{s}/adl/{i}",
            s,
            "pilot",
            f"state {i + 1}",
            "lens-only package, layers 20 and 30",
            "adl_package",
        )
        for i in range(28)
    ]
    lines += [
        p.api(f"{s}/adl_api/{i}", s, "pilot", f"state {i + 1}", "lens-only reading", "adl_reading")
        for i in range(28)
    ]
    lines += [
        p.gpu(f"{s}/fidelity/{b}", s, b, "A", "fidelity controls", "adl_package") for b in PILOT
    ]
    pilot_latent = [("example_link", d) for d in (25, 50, 100, 200)] + [
        ("pun_intended", d) for d in (25, 50)
    ]
    for b, d in pilot_latent:
        lines += [p.gpu(f"{s}/stability/{b}/{d}", s, b, f"L {d}", "second λ draw", "adl_package")]
        lines += [
            p.gpu(
                f"{s}/corpus/{b}/{d}",
                s,
                b,
                f"L {d}",
                "ADL-corpus expression check",
                "adl_corpus_check",
            )
        ]
        for state in ("L", "H_s", "X"):
            lines += [
                p.api(
                    f"{s}/pd_clean/{b}/{d}/{state}",
                    s,
                    b,
                    f"{state} {d}",
                    "PD-clean reading",
                    "pd_clean_reading",
                )
            ]
        lines += [
            p.api(
                f"{s}/pd_control/{b}/{d}",
                s,
                b,
                f"recipe {d}",
                "PD-clean positive control",
                "pd_clean_reading",
            )
        ]
        lines += [
            p.labels(
                f"{s}/pd_adjudication/{b}/{d}", s, b, "PD-clean adjudication", PD_ADJUDICATION_ITEMS
            )
        ]
    lines += [
        p.api(f"{s}/pd_control/{b}/A", s, b, "A", "PD-clean positive control", "pd_clean_reading")
        for b in PILOT
    ]
    lines += [
        p.labels(
            f"{s}/labels/detector_diagnostic", s, "pilot", "detector diagnostic (7c9dfdb81e16)", 600
        )
    ]
    lines += [
        p.labels(
            f"{s}/labels/rule_c_100_200",
            s,
            "example_link",
            "rule-C sets bc7b0bcc0f08, 5778a60ccdf4",
            240,
        )
    ]
    lines += [
        p.labels(
            f"{s}/labels/new_format/{f}", s, "pilot", f"new-format stratum: {f}", NEW_FORMAT_ITEMS
        )
        for f in ("lens-only", "PD-clean")
    ]
    return lines


def y2_organisms(p: Pricer, candidates: int) -> list[Line]:
    s = "stage2"
    lines = []
    for i in range(candidates):
        lines += [
            p.gpu(
                f"{s}/y2/candidate{i + 1}/implant",
                s,
                "Y2",
                f"candidate {i + 1}",
                "Y₂ organism",
                "implant_active",
            )
        ]
        lines += [
            p.gpu(
                f"{s}/y2/candidate{i + 1}/reference",
                s,
                "Y2",
                f"candidate {i + 1}",
                "Y₂ reference rate and criterion 3",
                "observer_pass",
                2,
            )
        ]
    return lines


def stage2(p: Pricer, rules: cr.Rules, y2_candidates: int) -> list[Line]:
    s = "stage2"
    lines = [p.gpu(f"{s}/setup/{i}", s, "-", "-", "Pod session", "pod_setup") for i in range(2)]
    lines += y2_organisms(p, y2_candidates)
    for b in PILOT:
        for d in (400, 800):
            lines += [
                p.gpu(
                    f"{s}/{b}/screen/pd/{d}", s, b, f"L {d}", "screening: full PD pool", "tool_pd"
                )
            ]
            lines += [p.gpu(f"{s}/{b}/screen/x/{d}", s, b, f"X {d}", "compose X", "compose_x")]
        # The selected deep dose: confirmation, certificate and readers (its X composed above).
        lines += [
            p.gpu(
                f"{s}/{b}/dose/confirm/{st}",
                s,
                b,
                st,
                "confirmation, 2 attempts",
                "observer_pass",
                2,
            )
            for st in ("L", "H_s")
        ]
        lines += certificate(p, s, b, rules) + reading(p, s, b)
    for d in (100, 200):
        for st in ("L", "H_s"):
            lines += [
                p.gpu(
                    f"{s}/example_link/y_alone/{d}/{st}",
                    s,
                    "example_link",
                    f"{st} {d}",
                    "relearn Y alone, 3 seeds × 2 rates",
                    "relearn_run",
                    6,
                )
            ]
    return lines


def stage3(p: Pricer, rules: cr.Rules) -> list[Line]:
    s = "stage3"
    behaviors = [*rules.panel, *rules["descriptive_adl_only"]]
    lines = [p.gpu(f"{s}/setup/{b}", s, b, "-", "Pod session", "pod_setup") for b in behaviors]
    for b in behaviors:
        lines += lineage(p, s, b, rules)
    return lines


def stage4(p: Pricer, rules: cr.Rules) -> list[Line]:
    s = "stage4"
    lines = []
    for b in PILOT:
        lines += [p.gpu(f"{s}/setup/{b}", s, b, "-", "Pod session", "pod_setup")]
        lines += [p.gpu(f"{s}/{b}/implant/A", s, b, "A", "second implantation", "implant_active")]
        lines += [p.gpu(f"{s}/{b}/implant/H", s, b, "H", "second sham", "implant_sham")]
        lines += [
            p.gpu(f"{s}/{b}/implant/R", s, b, "recipe", "second recipe control", "implant_recipe")
        ]
        lines += lineage(p, s, b, rules)
    return lines


def stage5(p: Pricer, rules: cr.Rules, stage: str, b: str, *, dpo: bool) -> list[Line]:
    mode = "dpo" if dpo else "continued"
    lines = [p.gpu(f"{stage}/setup", stage, b, "-", "Pod session", "pod_setup")]
    lines += lineage(p, stage, b, rules, mode=mode)
    # Relearning of b also on the unsuppressed sham H, descriptive: six blocks × two rates.
    lines += [p.gpu(f"{stage}/{b}/relearn_H", stage, b, "H", "relearn b on H", "relearn_run", 12)]
    return lines


def revision(p: Pricer, behaviors: Iterable[str]) -> list[Line]:
    return [
        p.gpu(
            f"revision/{b}/{st}",
            "revision",
            b,
            st,
            "fresh confirmation after a detector revision",
            "observer_pass",
        )
        for b in behaviors
        for st in ("L", "H_s")
    ]


def branch_lines(p: Pricer, rules: cr.Rules, branch: str) -> list[Line]:
    stage4_on = branch in ("stage4", "worst")
    dpo = branch in ("dpo", "worst")
    y2 = (
        4 if branch == "worst" else int(branch.rsplit("_", 1)[1]) if branch.startswith("y2_") else 1
    )
    lines = [p.lump("stage0/api", "stage0", "ensemble and reviewers", "stage0_api")]
    lines += stage1(p) + stage2(p, rules, y2)
    if branch.startswith("exception_a_"):
        b = "example_link" if branch.endswith("link") else "pun_intended"
        lines += stage5(p, rules, "exception_a", b, dpo=False)
    lines += stage3(p, rules)
    if stage4_on:
        lines += stage4(p, rules)
    if not branch.startswith("exception_a_"):
        lines += stage5(p, rules, "stage5", rules.panel[0], dpo=dpo)
    lines += [p.lump("stage6/qwen3_4b", "stage6", "Qwen3-4B, SFT stack", "stage6_qwen3_4b")]
    if branch in ("revision", "worst"):
        lines += revision(p, [*PILOT, *rules.panel, *rules["descriptive_adl_only"]])
    lines += [p.lump("storage", "storage", "volume, 2–3 months", "storage")]
    ids = [line.line_id for line in lines]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{branch}: duplicate cost lines")
    return lines


BRANCHES = (
    "main",
    "stage4",
    "dpo",
    "exception_a_link",
    "exception_a_pun",
    "y2_candidates_2",
    "y2_candidates_3",
    "y2_candidates_4",
    "revision",
    "worst",
)


def cut_parts(lines: list[Line], rules: cr.Rules) -> dict[str, list[Line]]:
    """The lines each part of the preset cut order removes (§12.8): whole Stages 6, 5 and 4,
    guillemets, the last panel behavior of Stage 3, and Stage 2 down to its first behavior."""
    last_panel = rules.panel[-1]
    guillemets = rules["descriptive_adl_only"][0]
    parts: dict[str, list[Line]] = {part: [] for part in rules["cut_order"]}
    for line in lines:
        if line.stage in ("stage6", "stage5", "stage4"):
            parts[line.stage].append(line)
        elif line.stage == "exception_a":  # it consumes the single Stage-5 slot
            parts["stage5"].append(line)
        elif line.stage == "stage3" and line.behavior == guillemets:
            parts["guillemets"].append(line)
        elif line.stage == "stage3" and line.behavior == last_panel:
            parts["last_stage3_behavior"].append(line)
        elif line.stage == "stage2" and line.behavior == PILOT[1]:
            parts["stage2_to_one"].append(line)
    return {part: rows for part, rows in parts.items() if rows}


def cut_sequence(p: Pricer, lines: list[Line], rules: cr.Rules, *, high: bool) -> list[str]:
    """The cuts the preset order needs for the plan plus the reserve to fit the cap."""
    cap, reserve = float(rules["meter"]["cap_usd"]), float(rules["meter"]["reserve_usd"])
    parts = cut_parts(lines, rules)
    projected = {part: sum(p.usd(x, high=high) for x in rows) for part, rows in parts.items()}
    total = sum(p.usd(x, high=high) for x in lines)
    fixed = total + reserve - sum(projected.values())
    try:
        return apply_cut_order(projected, order=rules["cut_order"], cap=cap, fixed=fixed)
    except ValueError:
        return ["infeasible after every cut"]


def summarize(p: Pricer, lines: list[Line], rules: cr.Rules) -> dict[str, Any]:
    """Per stage: GPU hours, API dollars, labels and dollars (central and ceiling), with the
    cumulative ceiling and the meter at each stage's gate (spent before + this stage + storage
    + reserve ≤ cap)."""
    stages: dict[str, dict[str, float]] = {}
    for line in lines:
        row = stages.setdefault(
            line.stage,
            {
                "gpu_h": 0.0,
                "gpu_h_high": 0.0,
                "api_usd": 0.0,
                "labels": 0,
                "usd": 0.0,
                "usd_high": 0.0,
            },
        )
        row["gpu_h"] += line.gpu_minutes / 60
        row["gpu_h_high"] += line.gpu_minutes_high / 60
        row["api_usd"] += line.api_usd
        row["labels"] += line.labels
        row["usd"] += p.usd(line, high=False)
        row["usd_high"] += p.usd(line, high=True)
    cap = float(rules["meter"]["cap_usd"])
    reserve = float(rules["meter"]["reserve_usd"])
    storage = stages.get("storage", {}).get("usd_high", 0.0)
    order = [s for s in (*STAGE_ORDER, "revision") if s in stages]
    spent = 0.0
    for name in order:
        row = stages[name]
        row["meter_at_gate"] = spent + row["usd_high"] + storage + reserve
        row["meter_allows"] = row["meter_at_gate"] <= cap
        spent += row["usd_high"]
        row["cumulative_high"] = spent
    total = sum(r["usd_high"] for r in stages.values())
    total_central = sum(r["usd"] for r in stages.values())
    cuts = {"central": cut_sequence(p, lines, rules, high=False)}
    cuts["ceiling"] = cut_sequence(p, lines, rules, high=True)
    first = rules.panel[0]
    groups = {
        "suppression and exploratory observer": "/suppress",
        "dose confirmation and X": "/dose",
        "certificate (72 runs, k*, rate)": "/certificate",
        "readers and expression checks": "/read",
    }
    lineage: dict[str, dict[str, float]] = {}
    for name, marker in groups.items():
        rows = [x for x in lines if x.line_id.startswith(f"stage3/{first}{marker}")]
        lineage[name] = {
            "gpu_h": sum(x.gpu_minutes for x in rows) / 60,
            "gpu_h_high": sum(x.gpu_minutes_high for x in rows) / 60,
            "usd": sum(p.usd(x, high=False) for x in rows),
            "usd_high": sum(p.usd(x, high=True) for x in rows),
        }
    return {
        "lineage": lineage,
        "stages": stages,
        "total_central": total_central,
        "total_high": total,
        "with_reserve_high": total + reserve,
        "cap": cap,
        "cuts": cuts,
    }


def ceilings_md(results: dict[str, dict[str, Any]], units_sha: str, rules: cr.Rules) -> str:
    lines = [
        "# Continuation inventory and ceilings (§12.3.9)",
        "",
        f"Rules sha256 `{rules.sha256}`; units sha256 `{units_sha}`. Ceilings use the units' high"
        " values, the central total their medians. GPU dollars = work hours × 3.99 × 4/3.",
        "",
        "| branch | central | ceiling | ceiling + reserve | fits | cuts (central) "
        "| cuts (ceiling) |",
        "|---|---|---|---|---|---|---|",
    ]
    for branch, r in results.items():
        lines.append(
            f"| {branch} | ${r['total_central']:.0f} | ${r['total_high']:.0f} | "
            f"${r['with_reserve_high']:.0f} | "
            f"{'yes' if r['with_reserve_high'] <= r['cap'] else 'no'} | "
            f"{', '.join(r['cuts']['central']) or '—'} | {', '.join(r['cuts']['ceiling']) or '—'} |"
        )
    lines += ["", "## One Stage-3 lineage, by procedure (central and ceiling)", ""]
    lines += ["| procedure | GPU h | GPU h (high) | $ | $ (high) |", "|---|---|---|---|---|"]
    for procedure, row in results["main"]["lineage"].items():
        lines.append(
            f"| {procedure} | {row['gpu_h']:.1f} | {row['gpu_h_high']:.1f} | "
            f"{row['usd']:.0f} | {row['usd_high']:.0f} |"
        )
    for branch in ("main", "worst"):
        lines += ["", f"## Branch {branch}, per stage", ""]
        lines += [
            "| stage | GPU h (high) | API $ | labels | $ (high) | cumulative | meter at gate "
            "| allows |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, row in results[branch]["stages"].items():
            lines.append(
                f"| {name} | {row['gpu_h_high']:.1f} | {row['api_usd']:.1f} | {row['labels']} | "
                f"{row['usd_high']:.0f} | {row.get('cumulative_high', 0):.0f} | "
                f"{row.get('meter_at_gate', 0):.0f} | {row.get('meter_allows', '—')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "continuation_inventory")
    args = parser.parse_args()
    rules = cr.load_rules()
    units = OmegaConf.to_container(OmegaConf.load(UNITS_PATH), resolve=True)
    units_sha = cr.rules_hash(units)
    p = Pricer(units)
    results, rows = {}, []
    for branch in BRANCHES:
        lines = branch_lines(p, rules, branch)
        results[branch] = summarize(p, lines, rules)
        rows += [
            {
                "branch": branch,
                **dataclasses.asdict(line),
                "usd": p.usd(line, high=False),
                "usd_high": p.usd(line, high=True),
            }
            for line in lines
        ]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "ceilings.md").write_text(ceilings_md(results, units_sha, rules))
    (args.out / "inventory.json").write_text(
        json.dumps(
            {"rules_sha256": rules.sha256, "units_sha256": units_sha, "branches": results}, indent=1
        )
    )
    print((args.out / "ceilings.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
