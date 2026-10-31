"""Barrier B3 of the follow-up (docs/preregistration.md §13.15): its configuration
(``configs/continuation/follow_b3.yaml``) checked against the records and the code that execute it,
so that neither can change alone (``problems``). ``continuation_rules.load_follow_b3`` loads it.

- **Certificates** (§13.8; ``certificate_problems``): the three new 14B certificates at the
  confirmed doses, by §12.4.8, §12.3.7 and §12.7 unchanged: the k* calibration's settings are
  ``continuation_kstar``'s and the rules' certificate block's; the controls Y and Y₂ the
  pilot's; each line's dose, states, implant, mode and A's reference its deciding record's
  (eli5's stack decision; example_link's and pun_intended's confirmations, with the pilot's
  approved passes); each line's meter item an allocated one.
- **Seal** (``seal_problems``): the envelopes' fixed cadence and payload fields.
- **Ladder** (``ladder_problems``): rungs 2 and 3 as ``follow_ladder`` and §11.1's code
  (``adl``) execute them, the prompts by sha256, the interpreter D*'s, the limits and the rules'
  query budget and runs; the probes' digests (``b3_renderings``, ``b3_requests``).
- **Unfiltered PD** (``pd_problems``): ``follow_pd``'s fraction and Stage-3 lineages.
- **Inventory** (``inventory_problems``): the manifest's readings as ``follow_inventory`` lists
  them, the counts, the salts distinct from B2's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from .paths import CONFIG_ROOT, REPO_ROOT

LINES = ("eli5_stack_200", "example_link_200", "pun_intended_200")
ITEMS = {"eli5_stack_200": "follow/cert/eli5_stack_200",
         "example_link_200": "follow/cert/example_link",
         "pun_intended_200": "follow/cert/pun_intended"}  # fmt: skip
LINE_KEYS = {"item", "behavior", "dose", "latent", "sham_s", "active", "sham", "mode", "method",
             "decision", "reference", "data"}  # fmt: skip


def _json(root: Path, rel: str) -> dict[str, Any]:
    return json.loads((root / rel).read_text())


def certificate_problems(cert: Mapping[str, Any], root: Path = REPO_ROOT) -> list[str]:
    """Why the certificates' section is not the records' and the code's (see the module)."""
    import yaml

    from . import continuation_kstar as K
    from .continuation_rules import load_rules
    from .simulate_continuation import FROZEN

    problems: list[str] = []
    rules = load_rules(expected_hash=FROZEN["rules_sha256"]).data["certificate"]
    k = cert["kstar"]
    if (tuple(k["seeds"]) != K.SEEDS or len(K.SEEDS) != int(rules["kstar_calibration_seeds"])
            or int(k["calibration_budget"]) != K.CALIBRATION_BUDGET
            or Fraction(k["fraction"]) != Fraction(str(rules["kstar_fraction"]))
            or [tuple(x) for x in k["schedule"]] != list(K.FROZEN_SCHEDULE)):  # fmt: skip
        problems.append(
            "certificates: the k* calibration is not continuation_kstar's and the rules'"
        )
    if cert["rules"] != "configs/continuation/rules.yaml" or cert["engine"] != "sleep":
        problems.append("certificates: the rules file and the sleeping engine are §12.4.8's")
    c = cert["controls"]
    if (c["y_organism"], dict(c["y_data"])) != (K.PILOT_Y_ORGANISM, {
            "id": K.PILOT_Y_DATA["data_id"], "examples": K.PILOT_Y_DATA["examples"]}):  # fmt: skip
        problems.append("certificates: Y is not the pilot's organism and data")
    if not (root / c["y2"]).is_file():
        problems.append(f"certificates: {c['y2']} missing")
    lines = cert["lines"]
    keys = {n: LINE_KEYS | ({"selection"} if n == "eli5_stack_200" else set()) for n in LINES}
    if tuple(lines) != LINES or any(set(lines[n]) != keys[n] for n in lines):
        return problems + [
            f"certificates: the lines are {list(lines)}, each with {sorted(LINE_KEYS)}"
        ]
    allocations = yaml.safe_load((CONFIG_ROOT / "continuation" / "allocations.yaml").read_text())
    items = {a["item"] for a in allocations["allocations"]}
    for name, line in lines.items():
        if line["item"] != ITEMS[name]:
            problems.append(f"{name}: its meter item is {ITEMS[name]}")
        problems += [f"{name}: {p}" for p in _line_problems(line, items, root)]
        problems += [f"{name}: {p}" for p in _data_problems(line, root)]
    return problems


def _data_problems(line: Mapping[str, Any], root: Path) -> list[str]:
    """b's data pin: a data id and an examples sha256, as the committed k* manifest of the same
    behavior records them (eli5, example_link), or as the pilot's relearning run of the same data
    records them (pun_intended: the data's identity only; that run is not reused)."""
    import yaml

    from .continuation_kstar import find_run

    pin = line["data"]
    if set(pin) != {"id", "examples", "source"} or len(str(pin["examples"])) != 64:
        return ["b's data pin is {id, examples, source}"]
    if str(pin["source"]).startswith("manifests/"):
        k = _json(root, pin["source"])
        if (k["behavior"], k["data_id"], k["relearn_examples"]) != (
                line["behavior"], pin["id"], pin["examples"]):  # fmt: skip
            return ["b's data pin is not the behavior's committed k* manifest's"]
        return []
    run = find_run(str(pin["source"]))
    if run is None:
        return [f"b's data source {pin['source']} is not here"]
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    params = config.get("params") or {}
    if ((config.get("behavior") or {}).get("id"), params.get("data_id"),
            params.get("examples")) != (line["behavior"], pin["id"], pin["examples"]):  # fmt: skip
        return ["b's data pin is not its source run's data"]
    return []


def _line_problems(line: Mapping[str, Any], items: set[str], root: Path) -> list[str]:
    """One line against its deciding record."""
    from . import continuation_kstar as K

    out = []
    if line["item"] not in items or not str(line["item"]).startswith("follow/cert/"):
        out.append(f"{line['item']} is not an allocated certificate item")
    if (line["mode"], line["method"]) != ("stack", "sft"):
        out.append("a certificate line of §13.8 is SFT in stack mode")
    if line["behavior"] == "eli5":
        d = _json(root, line["decision"])
        if line["decision"] != "manifests/stage3_eli5.json" or int(
                d["decision"]["analysis_dose"]) != int(line["dose"]):  # fmt: skip
            out.append("not eli5's stack decision at its analysis dose")
        if dict(d["implant"]) != {"active": line["active"], "sham": line["sham"]} or (
                d["mode"], d["method"]) != (line["mode"], line["method"]):  # fmt: skip
            out.append("not the stack decision's implant, mode and method")
        if (d["reference"]["run"], d["reference"]["state"]) != (line["reference"], line["active"]):
            out.append("A's reference is not the one the stack decision pins")
        return out + _selection_problems(line, d)
    confirmed = _json(root, "manifests/follow_confirmations.json")["confirmed"]
    if line["decision"] != "manifests/follow_confirmations.json" or confirmed.get(
            line["behavior"]) != int(line["dose"]):  # fmt: skip
        out.append(f"not {line['behavior']}'s confirmed dose")
    at = _json(root, "manifests/follow_confirmation_doses.json")["lineages"][line["behavior"]][
        "doses"].get(str(line["dose"])) or {}  # fmt: skip
    if (at.get("latent"), at.get("sham_s")) != (line["latent"], line["sham_s"]):
        out.append("L and H_s are not the confirmed dose's")
    if K.PILOT_REFERENCES.get(line["active"]) != line["reference"]:
        out.append("A's reference is not the pilot's approved pass of A")
    return out


SEAL_PAYLOAD = ("line", "n", "b3_sha256", "nonce", "record")
SEAL_ROOTS = {"runs": "sealed/runs", "work": "sealed/work", "served": "sealed/served",
              "cache": "sealed/cache"}  # fmt: skip
NONCE_BITS = 256


def seal_problems(seal: Mapping[str, Any]) -> list[str]:
    """The envelopes' cadence is a fixed whole number of minutes; the payload names its fields."""
    out = []
    cadence = seal["envelopes"]["cadence_minutes"]
    if type(cadence) is not int or cadence <= 0:
        out.append("seal: the envelopes' cadence is a fixed whole number of minutes")
    if dict(seal.get("roots") or {}) != SEAL_ROOTS:
        out.append(f"seal: the sealed roots are {SEAL_ROOTS}")
    if "pod_space.WRITERS" not in seal.get("writers", "") or "HF_HUB_OFFLINE=1" not in seal.get(
            "writers", ""):  # fmt: skip
        out.append("seal: every writer beneath the sealed cache, the hub read offline")
    payload = seal["envelopes"]["payload"]
    if not all(f in payload for f in SEAL_PAYLOAD) or f"{NONCE_BITS} random bits" not in payload:
        out.append(
            "seal: the payload names its line, n, B3's sha256, a 256-bit nonce and the record"
        )
    return out


B3_PROBE = "A probe hypothesis: the model ends its answers with a fixed phrase. 一秒 é"


def _digest(value: Any) -> str:
    import hashlib

    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def probe_steering() -> dict[str, Any]:
    """Fixed steering samples of D*'s positions: position 3 failed, Unicode and long replies."""
    from . import follow_ladder as L
    from .lens_render import POSITIONS

    per = {}
    for p in POSITIONS:
        bad = p == 3
        per[str(p)] = {"failed": "steering failed: no coherent strength" if bad else None,
                       "samples": [{"prompt": f"closed prompt {i} «{p}»",
                                    "steered": None if bad else f"steered {p}/{i} " + "ж" * i * 9,
                                    "unsteered": f"unsteered {p}/{i}\n" + "x" * i * 9}
                                   for i in range(L.CLOSED_PROMPTS)]}  # fmt: skip
    return {"per_position": per}


def b3_renderings() -> dict[str, str]:
    """Rung 2's evidence and the comparator's on B2's probe package and ``probe_steering``."""
    import tempfile

    from . import follow_ladder as L
    from .continuation_rules import _b2_probe_package

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = _b2_probe_package(root)
        rung2 = _digest(list(L.render_rung2(run, probe_steering(), root)))
        random = _digest(list(L.render_rung2(run, probe_steering(), root, decoder="random")))
    return {
        "rung2": rung2,
        "rung2_random": random,
        "comparator": _digest(L.comparator_evidence(probe_steering())),
    }


def b3_requests() -> dict[str, str]:
    """Each rung's requests on ``B3_PROBE``: rung 2's interpreter, every slot with its retry, as
    D*'s (``summarizer.summary_requests``); rung 3's first turn of the agent and the comparator,
    every slot with its retry (``follow_ladder.turn_key``, ``turn_body``)."""
    from . import continuation_readers as readers
    from . import follow_ladder as L
    from .summarizer import summary_requests

    _, settings = readers.interpreter("lens")
    first = summary_requests(L.rung2_prompt(), B3_PROBE, settings, salt="probe")
    retry = summary_requests(L.rung2_prompt(), B3_PROBE, settings, salt="probe", retry=True)
    out = {"rung2_interpreter": _digest([{"key": a.key, "retry": b.key, "body": a.body}
                                         for a, b in zip(first, retry, strict=True)])}  # fmt: skip
    for name, prompt in (("rung3_agent", L.agent_prompt()), ("rung3_comparator",
                                                             L.comparator_prompt())):  # fmt: skip
        body = L.turn_body(prompt, [{"role": "user", "content": B3_PROBE}], settings)
        out[name] = _digest([{"key": L.turn_key(prompt.version, "probe", i, 0, body),
                              "retry": L.turn_key(prompt.version, "probe", i, 0, body, True),
                              "body": body} for i in range(L.RUNS)])  # fmt: skip
    return out


def ladder_problems(ladder: Mapping[str, Any], rules: Mapping[str, Any]) -> list[str]:
    """Rungs 2 and 3 against ``follow_ladder``, §11.1's code and the follow-up rules."""
    from . import adl
    from . import continuation_readers as readers
    from . import follow_ladder as L

    out = []
    r2, r3 = ladder["rung2"], ladder["rung3"]
    if (r2["prompt"]["version"], r2["prompt"]["sha256"]) != (L.rung2_prompt().version,
                                                             L.rung2_prompt().sha256):  # fmt: skip
        out.append("rung 2: the prompt is not follow_ladder's")
    prompts = {"agent": L.agent_prompt(), "comparator": L.comparator_prompt()}
    if {k: (v["version"], v["sha256"]) for k, v in r3["prompts"].items()} != {
            k: (p.version, p.sha256) for k, p in prompts.items()}:  # fmt: skip
        out.append("rung 3: the prompts are not follow_ladder's")
    _, lens = readers.interpreter("lens")
    if dict(r3["model"]) != {
        "model": lens.model.id,
        "reasoning_effort": lens.model.reasoning_effort,
        "max_tokens": lens.max_tokens,
    }:
        out.append("rung 3: the model is not D*'s interpreter's")
    want = {"queries": (L.MAX_QUERIES, rules["agent_queries"]), "runs": (L.RUNS,
            rules["runs_per_reading"]), "query_chars": (L.MAX_QUERY_CHARS,),
            "context_chars": (L.CONTEXT_CHARS,)}  # fmt: skip
    for key, values in want.items():
        if any(int(r3[key]) != int(v) for v in values):
            out.append(f"rung 3: {key} is not the code's and the rules'")
    s = r2["steering"]
    search = {
        "steps": [adl.SEARCH_STEPS, adl.SEARCH_STEPS // 2, adl.SEARCH_STEPS // 2],
        "lookahead": adl.LOOKAHEAD,
        "samples": adl.SAMPLES_PER_STRENGTH,
        "max_new_tokens": adl.SEARCH_GENERATION["max_new_tokens"],
        "temperature": adl.SEARCH_GENERATION["temperature"],
        "coherent_share": adl.COHERENCE_THRESHOLD,
        "max_strength": adl.MAX_STRENGTH,
    }
    if (
        {k: v for k, v in s["search"].items() if k != "prompts"} != search
        or int(s["layer"]) != int(L.STEERING_LAYER)
        or int(s["truncation"]) != L.MAX_CHARS
    ):
        out.append("rung 2: the steering search is not §11.1's code (adl)")
    g = s["generation"]
    if (g["max_new_tokens"], g["temperature"]) != (
        adl.FINAL_GENERATION["max_new_tokens"],
        adl.FINAL_GENERATION["temperature"],
    ):
        out.append("rung 2: the steering generation is not §11.1's code (adl)")
    if len(adl.steering_prompts()) != L.CLOSED_PROMPTS:
        out.append("rung 2: §11.1's closed prompts are not 20")
    for rung in ("rung2", "rung3"):  # §13.5's difficult strata, in both classes
        m = ladder["format_validation"][rung]["minimums"]
        if "category_only" not in (m["positive"].get("difficulty") or {}) or "another_behavior" \
                not in (m["negative"].get("difficulty") or {}):  # fmt: skip
            out.append(f"{rung}: the format's minimums lack §13.5's difficult strata")
        if "difficulty" not in ladder["format_validation"][rung]["strata"]:
            out.append(f"{rung}: the format's strata lack the difficulty")
    if dict(ladder["probes"]) != {**b3_renderings(), **b3_requests()}:
        out.append("ladder: the probes' digests are not the code's")
    return out


def pd_problems(pd: Mapping[str, Any]) -> list[str]:
    from . import follow_pd as P

    out = []
    if float(pd["fraction"]) != P.FRACTION:
        out.append("unfiltered PD: the fraction is not follow_pd's")
    if {b: tuple(v) for b, v in pd["stage3"].items()} != P.STAGE3_MODES:
        out.append("unfiltered PD: the Stage-3 lineages are not follow_pd's")
    return out


RUNG_PARTS = (
    "shared",
    "recipe_continued",
    "recipe_stack",
    "validation_shared",
    "validation_recipe_continued",
    "validation_recipe_stack",
    "analysis",
)
PARTS = {"dstar": ("analysis",), "pd_unfiltered": ("analysis",), "rung2": RUNG_PARTS,
         "rung3": RUNG_PARTS}  # fmt: skip


def inventory_problems(inv: Mapping[str, Any], root: Path = REPO_ROOT) -> list[str]:
    """B3's inventory manifest: present at its sha256, its parts the configured counts, every
    reading checked (``follow_inventory.check``) together with B2's, so no salt is shared, and a
    reuse decision for every unfiltered-PD analysis reading."""
    import hashlib

    from . import follow_inventory as fi

    path = root / inv["manifest"]
    if not path.is_file():
        return [f"inventory manifest {inv['manifest']} missing"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != inv["sha256"]:
        return [f"inventory manifest sha256 {digest} != {inv['sha256']}"]
    manifest = json.loads(path.read_text())
    out = []
    counts = {k: {p: len(manifest.get(k, {}).get(p, [])) for p in parts}
              for k, parts in PARTS.items()}  # fmt: skip
    if counts != {k: dict(v) for k, v in inv["counts"].items()}:
        out.append(f"inventory counts {counts} are not the configured ones")
    b2 = json.loads((root / "manifests" / "follow_b2_inventory.json").read_text())
    rows = [r for k in PARTS for p in PARTS[k] for r in manifest.get(k, {}).get(p, [])]
    try:
        fi.check(rows + [r for k, v in b2.items() if k != "sources" for rs in v.values()
                         for r in rs])  # fmt: skip
    except ValueError as error:
        out.append(f"inventory: {error}")
    pd = [r["salt"] for r in manifest.get("pd_unfiltered", {}).get("analysis", [])]
    if sorted(manifest.get("reuse") or {}) != sorted(pd):
        out.append("inventory: not one reuse decision per unfiltered-PD analysis reading")
    if (manifest.get("sources") or {}).get("build") != inv["build"]:
        out.append("inventory: not the configured build session")
    return out


def _selection_problems(line: Mapping[str, Any], decision: Mapping[str, Any]) -> list[str]:
    """eli5's L and H_s: its stack decision's selection record (the run B3 pins, at its sha256)
    at the line's dose."""
    import hashlib

    from .continuation_kstar import find_run

    sel = line["selection"]
    if decision["inputs"]["selection"] != sel["run"]:
        return ["the selection is not the stack decision's"]
    run = find_run(str(sel["run"]))
    if run is None:
        return [f"the selection record {sel['run']} is not here"]
    data = (run / "stage3_selection.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != sel["sha256"]:
        return ["the selection record is not at its pinned sha256"]
    at = json.loads(data)["doses"].get(str(line["dose"])) or {}
    if (at.get("latent"), at.get("sham_s")) != (line["latent"], line["sham_s"]):
        return ["L and H_s are not the selection's at the line's dose"]
    return []


def problems(b3: Mapping[str, Any], root: Path = REPO_ROOT) -> list[str]:
    """Why B3's configuration disagrees with the records or the code: every section's check."""
    from .continuation_rules import follow_rules

    return [*certificate_problems(b3["certificates"], root), *seal_problems(b3["seal"]),
            *ladder_problems(b3["ladder"], follow_rules().data["ladder"]),
            *pd_problems(b3["pd_unfiltered"])]  # fmt: skip
