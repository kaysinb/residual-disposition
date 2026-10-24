"""The PD-clean study of one work directory (preregistration §12.3.2, §12.4.2): its plan, PD pools,
immutable records and the selection, review and packages they give. scripts/pd_clean.py writes
the records; ``verify_packages`` recomputes a certified lineage's Stage-3 packages from them, the
readers' verdicts' check of PD-clean's provenance (``continuation_readers``).

Stage 1's plan (``pd_clean.yaml``) lists its states; its pools are found by the script. Stage 3's
template (``pd_clean_stage3.yaml``) expands into one certified lineage's states from its readers'
record (``stage3_states``), each with the pool its readers read, named.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from . import pd_clean as C
from .paths import CONFIG_ROOT
from .pipeline import read_jsonl_gz

STAGE3_PLAN = CONFIG_ROOT / "continuation" / "pd_clean_stage3.yaml"
STAGE3_ROLES = ("L", "H_s", "X", "A", "recipe")  # a certified lineage's states with a PD pool


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def behavior(behavior_id: str):
    from .behaviors import behavior_from_config

    spec = yaml.safe_load((CONFIG_ROOT / "behaviors" / f"{behavior_id}.yaml").read_text())
    return behavior_from_config(spec["behavior"])


def allocate(total: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """``total`` split over groups in proportion to ``sizes`` by largest remainders (ties in
    group order), each group capped by its size."""
    whole = sum(sizes.values())
    if total > whole:
        raise SystemExit(f"a sample of {total} from {whole} agreed negatives")
    quota = {g: total * n / whole for g, n in sizes.items()}
    out = {g: int(q) for g, q in quota.items()}
    order = sorted(sizes, key=lambda g: (-(quota[g] - out[g]), g))
    for g in order[: total - sum(out.values())]:
        out[g] += 1
    return out


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def plan_section(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """A plan file's section: Stage 3's template or Stage 1's plan."""
    return data["pd_clean_stage3"] if "pd_clean_stage3" in data else data["pd_clean"]


def verify_amendment(a: Mapping[str, Any], where: Path) -> None:
    """An amendment record must hold the two plans its bindings name, which differ in the review
    section alone, between bindings equal apart from the plan."""
    old, new = a["old_plan"], a["new_plan"]
    if text_sha256(old) != a["from"]["plan_sha256"] or text_sha256(new) != a["to"]["plan_sha256"]:
        raise SystemExit(f"{where}: its plans are not the ones its bindings name")
    o, n = (plan_section(yaml.safe_load(x)) for x in (old, new))
    if {k: v for k, v in o.items() if k != "review"} != {
        k: v for k, v in n.items() if k != "review"
    }:
        raise SystemExit(f"{where}: the plans differ beyond the review section")
    if o.get("review") == n.get("review"):
        raise SystemExit(f"{where}: the review section is unchanged")
    if {k: v for k, v in a["from"].items() if k != "plan_sha256"} != {
        k: v for k, v in a["to"].items() if k != "plan_sha256"
    }:
        raise SystemExit(f"{where}: the bindings differ beyond the plan")


def check_withdrawal(work: Path, name: str, binding: Mapping[str, Any]) -> None:
    """A withdrawn review is a review record drawn under the amendment's old binding and never
    imported: checked when the amendment is made and whenever the records are read."""
    d = Path(work) / name
    if not name.startswith("review") or not (d / "mapping.json").exists():
        raise SystemExit(f"{name} is not a review record")
    if (d / "labels.json").exists():
        raise SystemExit(f"{name} is imported: an imported review cannot be withdrawn")
    if json.loads((d / "mapping.json").read_text())["binding"] != binding:
        raise SystemExit(f"{name} was not drawn under the old plan")


def stage3_states(plan: Mapping[str, Any], record_run: Path) -> tuple[dict, list[dict]]:
    """A certified lineage's PD-clean states from its readers' record (the Pod's cell
    ``stage3_readers``, completed and its record the one its identity names): L, H_s, X, A and
    the recipe control, each with the pool its readers read; adjudicated and given the random
    control as the plan's template sets. Returns the record's binding and the states."""
    from .pipeline import content_hash

    config = yaml.safe_load((record_run / "config.yaml").read_text()) or {}
    meta = json.loads((record_run / "meta.json").read_text())
    path = record_run / "stage3_readers.json"
    record = json.loads(path.read_text())
    if (
        config.get("step") != "stage3_readers" or meta.get("status") != "completed"
        or (config.get("params") or {}).get("record") != content_hash(record)
    ):  # fmt: skip
        raise SystemExit(f"{record_run}: not a completed readers' record of a certified lineage")
    states = []
    for role in STAGE3_ROLES:
        r = record["readers"][role]
        states.append({"state": r["state"], "behavior": record["behavior"], "role": role,
                       "dose": None if role == "A" else record["dose"],
                       "adjudicate": role in plan["adjudicate"],
                       "random": role in plan["random"], "pd_run": r["pd"]})  # fmt: skip
    binding = {"run": record_run.name, "lineage": record["lineage"], "sha256": sha256(path)}
    return binding, states


def named_pd_run(name: str, state_id: str) -> Path:
    """A named, completed tool_pd run of ``state_id``: the pool a certified lineage's readers
    read, never searched for."""
    from .continuation_kstar import find_run

    run = find_run(name)
    if run is None:
        raise SystemExit(f"{state_id}'s PD pool {name} is not here")
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    meta = json.loads((run / "meta.json").read_text())
    if (
        config.get("step") != "tool_pd" or (config.get("params") or {}).get("state") != state_id
        or meta.get("status") != "completed"
    ):  # fmt: skip
        raise SystemExit(f"{name}: not a completed PD pool of {state_id}")
    return run


ITEM_FIELDS = {"round": {"state", "config", "prefill"},  # what a record's mapping says of an item
               "review": {"state", "config", "prefill", "why"}}  # fmt: skip


def item_of(mapping: Mapping[str, Any], iid: str, kind: str, where: Path,
            rows: Mapping[str, set[tuple[str, str]]]) -> dict[str, Any]:  # fmt: skip
    """An item's metadata in its record's mapping: exactly its kind's fields, never a label's
    (metadata never stands in for a label), naming a row of its state's pool (``rows``: each
    state's (config, prefill) pairs), and its id that row's (``pd_clean.item_id``), so that a
    label can only be its own item's."""
    item = (mapping.get("items") or {}).get(iid)
    if not isinstance(item, dict) or set(item) != ITEM_FIELDS[kind]:
        raise SystemExit(f"{where}: item {iid}'s metadata is not a {kind} item's")
    state, config, prefill = item["state"], item["config"], item["prefill"]
    if (config, prefill) not in rows.get(state, set()) or C.item_id(state, config, prefill) != iid:
        raise SystemExit(f"{where}: item {iid}'s metadata is not its own pool row")
    return dict(item)


def group_of(s: Mapping[str, Any]) -> str:
    """A Stage-3 adjudicated package's review group."""
    return f"{s['behavior']}/{s['role']}/{s['state']}"


class Study:
    """The plan, the PD outputs and the records of one work directory (scripts/pd_clean.py's
    module docstring)."""

    def __init__(self, work: Path, plan_path: Path, record: Path | None = None, *,
                 pool_of: Callable[[Mapping[str, Any], Mapping[str, Any]], Path] | None = None,
                 ) -> None:  # fmt: skip
        from .path_b import trace_detector

        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.plan_path = Path(plan_path)
        data = yaml.safe_load(self.plan_path.read_text())
        self.stage3 = "pd_clean_stage3" in data
        if self.stage3 != (record is not None):
            raise SystemExit("a Stage-3 plan reads one certified lineage's readers' record "
                             "(--record), a Stage-1 plan none")  # fmt: skip
        self.plan = data["pd_clean_stage3"] if self.stage3 else data["pd_clean"]
        self.fraction = float(self.plan["fraction"])
        self.seed = int(self.plan["seed"])
        self.states, self.rows, self.hits, self.runs = [], {}, {}, {}
        self.pool_keys: dict[str, set[tuple[str, str]]] = {}
        detectors = {}
        if self.stage3:
            self.record, states = stage3_states(self.plan, Path(record))
        else:
            self.record, states = None, self.plan["states"]
        for s in states:
            if self.stage3:  # the pool the Pod's readers read, by name
                run = named_pd_run(s["pd_run"], s["state"])
            elif pool_of is not None:  # Stage 1's: its declared pool (scripts/pd_clean.py)
                run = pool_of(self.plan, s)
            else:
                raise SystemExit("a Stage-1 plan needs its pools' lookup")
            rows = read_jsonl_gz(run / "pd_completions.jsonl.gz")
            self.rows[s["state"]] = [{**r, "ppl_diff": float(r["ppl_diff"])} for r in rows]
            self.pool_keys[s["state"]] = {(r["config"], r["prefill"]) for r in rows}
            b = behavior(s["behavior"])
            self.hits[s["state"]] = trace_detector(b.detector)
            detectors[b.id] = b.detector.version
            self.runs[s["state"]] = run
            self.states.append(s)
        rubric = CONFIG_ROOT.parent / self.plan["rubric"]
        if sha256(rubric) != self.plan["rubric_sha256"]:
            raise SystemExit(f"{rubric} does not match the rubric sha256 the plan freezes")
        self.binding = {
            "plan_sha256": sha256(self.plan_path),
            "rubric_sha256": self.plan["rubric_sha256"],
            "pd_runs": {
                sid: {"run": run.name, "sha256": sha256(run / "pd_completions.jsonl.gz")}
                for sid, run in self.runs.items()
            },
            "detectors": detectors,
        }
        if self.record is not None:
            self.binding["record"] = self.record
        self.earlier, self.withdrawn = self.amendments()

    # --- records -----------------------------------------------------------------------------

    def all_records(self, kind: str) -> list[Path]:
        dirs = [p for p in self.work.glob(f"{kind}[0-9]*") if p.is_dir()]
        return sorted(dirs, key=lambda p: int(p.name[len(kind) :]))

    def records(self, kind: str) -> list[Path]:
        """The records of a kind, without the reviews an amendment withdrew."""
        return [d for d in self.all_records(kind) if d.name not in self.withdrawn]

    def amendments(self) -> tuple[list[dict[str, Any]], set[str]]:
        """The earlier bindings whose round records stay valid through amendment records that
        lead to the current binding, and the reviews those amendments withdrew."""
        found = []
        for d in self.all_records("amend"):
            a = json.loads((d / "amendment.json").read_text())
            verify_amendment(a, d)
            found.append(a)
        earlier, withdrawn, reached = [], set(), [self.binding]
        changed = True
        while changed:
            changed = False
            for a in found:
                if a["to"] in reached and a["from"] not in reached:
                    reached.append(a["from"])
                    earlier.append(a["from"])
                    for name in a["withdrawn"]:  # checked again on every replay
                        check_withdrawal(self.work, name, a["from"])
                    withdrawn |= set(a["withdrawn"])
                    changed = True
        return earlier, withdrawn

    def check(self, record: Mapping[str, Any], where: Path, *, rounds: bool = False) -> None:
        """A record must carry the current binding; a round's may carry an earlier one that an
        amendment of the review section alone keeps valid."""
        binding = record.get("binding")
        if binding == self.binding or (rounds and binding in self.earlier):
            return
        raise SystemExit(
            f"{where} was made under another binding (plan, rubric, PD runs or detectors)"
        )

    def pending(self) -> Path | None:
        """A round or review written but not imported yet."""
        for kind in ("round", "review"):
            for d in self.records(kind):
                if not (d / "labels.json").exists():
                    return d
        return None

    def ensemble(self, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
        """Every ensemble label by item id, with its state, config and prefill (of the rounds
        ``only`` names, when given)."""
        out = {}
        for d in self.records("round"):
            if (only is not None and d.name not in only) or not (d / "labels.json").exists():
                continue
            mapping = json.loads((d / "mapping.json").read_text())
            labels = json.loads((d / "labels.json").read_text())
            self.check(mapping, d, rounds=True)
            self.check(labels, d, rounds=True)
            for iid, label in labels["labels"].items():
                item = item_of(mapping, iid, "round", d, self.pool_keys)
                out[iid] = {**item, **label, "round": d.name}
        return out

    def reviews(self, only: set[str] | None = None) -> tuple[dict[str, dict[str, Any]], set[str]]:
        """The researcher's labels by item id, and the groups whose negative sample is drawn (of
        the reviews ``only`` names, when given)."""
        out, sampled = {}, set()
        for d in self.records("review"):
            if only is not None and d.name not in only:
                continue
            mapping = json.loads((d / "mapping.json").read_text())
            self.check(mapping, d)
            sampled |= set(mapping["sampled_groups"])
            if not (d / "labels.json").exists():
                continue
            labels = json.loads((d / "labels.json").read_text())
            self.check(labels, d)
            for iid, label in labels["labels"].items():
                item = item_of(mapping, iid, "review", d, self.pool_keys)
                out[iid] = {**item, **label, "review": d.name}
        return out, sampled

    def final(self, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
        """Each labeled item's final label: the researcher's where reviewed, else the
        ensemble's (from the records ``only`` names, when given)."""
        ensemble, (reviewed, _) = self.ensemble(only), self.reviews(only)
        for iid, label in reviewed.items():  # a reviewed item is its ensemble item
            if iid not in ensemble or any(label[k] != ensemble[iid][k]
                                          for k in ("state", "config", "prefill")):  # fmt: skip
                raise SystemExit(f"{label['review']}: item {iid} is not an item the ensemble "
                                 "labeled")  # fmt: skip
        return {iid: reviewed.get(iid, label) for iid, label in ensemble.items()}

    def selection(
        self, s: Mapping[str, Any], final: Mapping[str, Mapping[str, Any]]
    ) -> C.Selection:
        mine = [v for v in final.values() if v["state"] == s["state"]] if s["adjudicate"] else []
        return C.clean_selection(
            self.rows[s["state"]],
            hit=self.hits[s["state"]],
            excluded=C.excluded(mine),
            fraction=self.fraction,
        )

    def todo(
        self, lookahead: Mapping[str, int] | None = None
    ) -> dict[str, tuple[Mapping[str, Any], str, Mapping[str, Any]]]:
        """Every adjudicated state's selected items without an ensemble label, and for the
        states in ``lookahead`` the next candidates below its selection: id → (state, config,
        row)."""
        final, out = self.final(), {}
        for s in self.states:
            if not s["adjudicate"]:
                continue
            sel = self.selection(s, final)
            for iid, (config, row) in C.unlabeled(sel, s["state"], final).items():
                out[iid] = (s, config, row)
            k = int((lookahead or {}).get(s["state"], 0))
            if k:
                mine = [v for v in final.values() if v["state"] == s["state"]]
                ahead = C.candidates_after(
                    self.rows[s["state"]],
                    sel,
                    hit=self.hits[s["state"]],
                    excluded=C.excluded(mine),
                    k=k,
                )
                for config, rows in ahead.items():
                    for row in rows:
                        iid = C.item_id(s["state"], config, row["prefill"])
                        if iid not in final:
                            out[iid] = (s, config, row)
        return out

    def met(self) -> set[str]:
        """The item ids every adjudicated state's current selection met (its window)."""
        final, out = self.final(), set()
        for s in self.states:
            if s["adjudicate"]:
                out |= C.encountered(self.selection(s, final), s["state"])
        return out

    def review_needs(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """The items the researcher must still review (id → their record, with why), and the
        groups ("<behavior>/<package type>") whose negative sample this review draws."""
        ensemble, (reviewed, sampled) = self.ensemble(), self.reviews()
        final, needs, draw, met = self.final(), {}, [], self.met()
        for iid, label in ensemble.items():
            if iid in reviewed or iid not in met:  # labeled ahead, below every window
                continue
            if not label["unanimous"]:
                needs[iid] = {**label, "why": "dispute"}
            elif label["present"]:
                needs[iid] = {**label, "why": "trace"}
        review = self.plan["review"]
        if review.get("allocation") == "per_package":
            return needs, self.package_sample(needs, ensemble, final, sampled)
        if review.get("allocation") != "proportional":
            raise SystemExit("the plan's review allocation is proportional or per_package")
        types = review["groups"]
        pools: dict[str, list[str]] = {}  # every group's agreed negatives, for the allocation
        for s in self.states:
            if not s["adjudicate"]:
                continue
            if s["role"] not in types:
                raise SystemExit(f"{s['state']}: role {s['role']} has no review group")
            sel = self.selection(s, final)
            pools.setdefault(f"{s['behavior']}/{types[s['role']]}", []).extend(
                iid
                for iid in sel.ids(s["state"])
                if iid in ensemble and ensemble[iid]["unanimous"] and not ensemble[iid]["present"]
            )
        n = allocate(int(review["negatives_total"]), {g: len(p) for g, p in pools.items()})
        for group, pool in sorted(pools.items()):
            if group in sampled:
                continue
            rng = random.Random(f"{self.seed}/review/{group}")
            for iid in rng.sample(sorted(pool), n[group]):
                needs[iid] = {**ensemble[iid], "why": "sample"}
            draw.append(group)
        return needs, draw

    def group_sample(self, s: Mapping[str, Any], ensemble: Mapping[str, Any],
                     final: Mapping[str, Any]) -> list[str]:  # fmt: skip
        """An adjudicated package's review sample: ``negatives_per_package`` of its selection's
        agreed (unanimous) negatives, all when fewer, seeded by its group."""
        per = int(self.plan["review"]["negatives_per_package"])
        sel = self.selection(s, final)
        pool = sorted(
            iid for iid in sel.ids(s["state"])
            if iid in ensemble and ensemble[iid]["unanimous"] and not ensemble[iid]["present"]
        )  # fmt: skip
        rng = random.Random(f"{self.seed}/review/{group_of(s)}")
        return rng.sample(pool, min(per, len(pool)))

    def package_sample(self, needs: dict[str, dict[str, Any]], ensemble: Mapping[str, Any],
                       final: Mapping[str, Any], sampled: set[str]) -> list[str]:  # fmt: skip
        """Stage 3's sample (``review.allocation: per_package``): each adjudicated package's
        ``group_sample`` not drawn yet, added to ``needs``. Returns the groups this review
        draws."""
        draw = []
        for s in self.states:
            if not s["adjudicate"] or group_of(s) in sampled:
                continue
            for iid in self.group_sample(s, ensemble, final):
                needs[iid] = {**ensemble[iid], "why": "sample"}
            draw.append(group_of(s))
        return draw

    def drawn_after(self) -> dict[str, list[str]]:
        """The records a review drawn now rests on: every imported round and review (none is
        pending when a review is drawn)."""
        return {kind: [d.name for d in self.records(kind) if (d / "labels.json").exists()]
                for kind in ("round", "review")}  # fmt: skip


PACKAGE_STEPS = {"pd_clean": "pd_clean_package", "pd_random": "pd_random_package"}


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def verify_packages(work: Path, record_run: Path, packages: Mapping[tuple[str, str], Path]) -> dict:
    """A certified lineage's PD-clean packages ``{(role, "pd_clean" | "pd_random"): run}``
    recomputed from its study's records (``work``, the Stage-3 plan expanded from
    ``record_run``): no round or review pending or needed; every adjudicated state's selection
    from the final labels (the researcher's where reviewed), the others' from the detector; the
    random-removal controls from the selection and the plan's seed; each package run's
    selection, removals and parameters (state, behavior, role, dose, binding, pool) exactly
    these. Exactly the expanded states' packages. Returns every record file of the study by
    sha256 (its path in ``work``). Pending (``continuation_panel.Pending``) while a round or the
    review is open; ValueError for anything else that does not hold."""
    from .continuation_panel import Pending

    if not Path(work).is_dir():
        raise Pending(f"{work}: no PD-clean study of the lineage yet")
    try:
        st = Study(work, STAGE3_PLAN, record_run)
    except (SystemExit, KeyError) as error:  # the plan incomplete (its rubric) fails closed
        raise ValueError(f"{work}: not a complete Stage-3 study ({error})") from None
    problems = coverage_problems(st)  # an imported record's labels, before what they imply
    if problems:
        raise ValueError(f"{work}: the imported labels: {'; '.join(problems)}")
    try:
        return _verified(st, work, packages)
    except SystemExit as error:  # a record the study refuses
        raise ValueError(f"{work}: {error}") from None


def _verified(st: Study, work: Path, packages: Mapping[tuple[str, str], Path]) -> dict:
    from .continuation_panel import Pending

    if st.pending() is not None or st.todo():
        raise Pending(f"{work}: an ensemble round is open or due")
    needs, draw = st.review_needs()
    if needs or draw:
        raise Pending(f"{work}: the researcher's review is not complete ({len(needs)} items)")
    problems = sample_problems(st)
    if problems:
        raise ValueError(f"{work}: the researcher's review sample: {'; '.join(problems)}")
    final, want = st.final(), {}
    for s in st.states:
        sel = st.selection(s, final)
        want[(s["role"], "pd_clean")] = (s, sel, {"adjudicated": s["adjudicate"]})
        if s["random"]:
            ctl = C.random_window(st.rows[s["state"]], sel, seed=st.seed)
            want[(s["role"], "pd_random")] = (s, ctl, {"adjudicated": s["adjudicate"],
                                                       "seed": st.seed})  # fmt: skip
    if sorted(packages) != sorted(want):
        raise ValueError(f"the PD-clean packages {sorted(packages)} are not {sorted(want)}")
    for key, run in packages.items():
        s, sel, extra = want[key]
        config = yaml.safe_load((Path(run) / "config.yaml").read_text()) or {}
        params = dict(config.get("params") or {})
        pool = Path(str(params.pop("pd_run", ""))).name
        expected = {"state": s["state"], "behavior": s["behavior"], "role": s["role"],
                    "dose": s["dose"], "binding": _json(st.binding), **extra}  # fmt: skip
        package = json.loads((Path(run) / "pd_clean.json").read_text())
        if (
            config.get("step") != PACKAGE_STEPS[key[1]] or params != expected
            or pool != st.runs[s["state"]].name
            or package != _json({"state": s["state"], "top": sel.top, "removed": sel.removed})
        ):  # fmt: skip
            raise ValueError(f"{Path(run).name}: not {key[0]}'s {key[1]} package of its records")
    records = {}
    for kind in ("round", "review", "amend"):
        for d in st.all_records(kind):
            for f in sorted(p for p in d.iterdir() if p.is_file()):
                records[f"{d.name}/{f.name}"] = sha256(f)
    return records


def sample_problems(st: Study) -> list[str]:
    """Where the reviews' negative samples are not the seeded ones: each review names the records
    it was drawn after (``drawn_after``: a prefix of the rounds, every earlier review), whose
    selections leave no item unlabeled; each group it draws is drawn there once, its sample
    items exactly ``group_sample`` of those records; every adjudicated package drawn."""
    problems, drawn = [], {}
    states = {group_of(s): s for s in st.states if s["adjudicate"]}
    rounds = [d.name for d in st.records("round")]
    for j, d in enumerate(st.records("review")):
        mapping = json.loads((d / "mapping.json").read_text())
        after = mapping.get("after")
        named = list(after.get("round") or []) if isinstance(after, dict) else None
        if (
            not isinstance(after, dict) or set(after) != {"round", "review"}
            or named != rounds[: len(named)]
            or list(after["review"]) != [r.name for r in st.records("review")[:j]]
        ):  # fmt: skip
            problems.append(f"{d.name}: not drawn after a prefix of the rounds and every review")
            continue
        only = set(named) | set(after["review"])
        ensemble, final = st.ensemble(only), st.final(only)
        for s in states.values():
            if C.unlabeled(st.selection(s, final), s["state"], final):
                problems.append(f"{d.name}: drawn while {s['state']} had unlabeled items")
        samples = {iid: v for iid, v in mapping["items"].items() if v.get("why") == "sample"}
        for group in mapping.get("sampled_groups") or []:
            if group not in states or group in drawn:
                problems.append(f"{d.name}: {group} is not an adjudicated package drawn once")
                continue
            drawn[group] = d.name
            s = states[group]
            got = {iid for iid, v in samples.items() if v["state"] == s["state"]}
            if got != set(st.group_sample(s, ensemble, final)):
                problems.append(f"{d.name}: {group}'s sample is not its seeded sample")
        undrawn = {v["state"] for v in samples.values()} - {
            states[g]["state"] for g in mapping.get("sampled_groups") or [] if g in states
        }
        if undrawn:
            problems.append(f"{d.name}: sample items of groups it does not draw")
    missing = sorted(set(states) - set(drawn))
    if missing:
        problems.append(f"no sample drawn for {missing}")
    return problems


LABEL_FIELDS = {  # each record's label fields and their types (scripts/pd_clean.py's imports)
    "round": {"present": (bool,), "unanimous": (bool,), "confident": (bool, type(None)),
              "evidence": (str, type(None))},
    "review": {"present": (bool,), "note": (str, type(None))},
}  # fmt: skip
REQUIRED = {"round": {"present", "unanimous"}, "review": {"present"}}


def coverage_problems(st: Study) -> list[str]:
    """Where a round's or a review's imported labels are not exactly its items, each with a typed
    label: no item of a mapping goes unlabeled, none is labeled outside it, and every label holds
    its required fields with their types and nothing else."""
    problems = []
    for kind in ("round", "review"):
        for d in st.records(kind):
            mapping = json.loads((d / "mapping.json").read_text())
            path = d / "labels.json"
            if not path.is_file():  # not imported yet: pending, not malformed
                continue
            labels = json.loads(path.read_text()).get("labels")
            if not isinstance(labels, dict) or set(labels) != set(mapping.get("items") or {}):
                problems.append(f"{d.name}: its labels are not exactly its items")
                continue
            items = mapping.get("items") or {}
            if any(not isinstance(v, dict) or set(v) != ITEM_FIELDS[kind] for v in items.values()):
                problems.append(f"{d.name}: its mapping's items are not {kind} items")
                continue
            try:
                for iid in items:
                    item_of(mapping, iid, kind, d, st.pool_keys)
            except SystemExit as error:
                problems.append(str(error))
                continue
            fields, required = LABEL_FIELDS[kind], REQUIRED[kind]
            for iid, label in labels.items():
                if (
                    not isinstance(label, dict) or not required <= set(label) <= set(fields)
                    or any(not isinstance(v, fields[k]) for k, v in label.items())
                ):  # fmt: skip
                    problems.append(f"{d.name}: {iid}'s label is not a typed {kind} label")
                    break
    return problems
