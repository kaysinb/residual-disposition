"""The panel's phases in the meter (§12.8 as amended 2026-10-22; review s6f): each panel lineage's
items are their preparation phases and their follow-ups are items of their own, the two summing
to the lineage's allocation; the five behaviors' preparations and the first behavior's follow-up
are admitted at once at stress, the other follow-ups protected at their allocations; the
contracts recorded once, after it; no follow-up starts before them, and the later ones are
admitted after them, in the frozen order; a follow-up without a registered reading is
released."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition.budget import AllocationMeter, load_allocations, load_migrations
from residual_disposition.paths import CONFIG_ROOT

PATH = CONFIG_ROOT / "continuation" / "allocations.yaml"
RULES = cr.load_rules()
PANEL = list(RULES.panel)
SHA = "a" * 64
DEADLINE = "2099-01-01T00:00:00+00:00"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_meter.py"
spec = importlib.util.spec_from_file_location("continuation_meter_phases", SCRIPT)
CM = importlib.util.module_from_spec(spec)
sys.modules["continuation_meter_phases"] = CM
spec.loader.exec_module(CM)


def meter(log, cap=1e5):
    items = load_allocations(PATH)
    return AllocationMeter(items, cap=cap, reserve=100.0,
                           drop_reasons=cr.drop_reasons(RULES, [a.item for a in items]),
                           log_path=log, migrations=load_migrations(PATH))  # fmt: skip


def entry(behavior, status="closed", reading=None, contracts=SHA):
    """A panel entry's evidence as ``continuation_meter.panel_evidence`` gives it."""
    return {"contracts": contracts, "behavior": behavior, "status": status, "reading": reading}


def above_the_panel(m):
    """Every item above the panel's first, admitted, started and completed at no cost."""
    for item in m.priority[: m.priority.index(f"stage3/{PANEL[0]}")]:
        m.admit(item, spent_elsewhere=0, storage=0)
        m.start(item)
        m.complete(item)


def test_each_lineage_is_its_preparation_and_follow_up_summing_to_it():
    items = {a.item: a for a in load_allocations(PATH)}
    for b in PANEL:
        for parent, whole in ((f"stage3/{b}", (136.85, 246.85, 6.32)),
                              (f"strong/stage3/{b}", (40.41, 86.17, 26.39))):  # fmt: skip
            prep, follow = items[parent], items[f"{parent}/follow"]
            parts = [round(getattr(prep, f) + getattr(follow, f), 2)
                     for f in ("usd", "stress_gpu_usd", "stress_api_usd")]  # fmt: skip
            assert tuple(parts) == whole
        names = list(items)
        i = names.index(f"stage3/{b}")
        assert names[i : i + 4] == [f"stage3/{b}", f"strong/stage3/{b}", f"stage3/{b}/follow",
                                    f"strong/stage3/{b}/follow"]  # fmt: skip
    reasons = cr.drop_reasons(RULES, list(items))
    assert reasons[f"stage3/{PANEL[0]}/follow"] == ["no_registered_readout"]
    assert reasons[f"strong/stage3/{PANEL[-1]}/follow"] == [
        "no_registered_readout", "cut_order:last_stage3_behavior"]  # fmt: skip


def test_the_panel_is_admitted_at_once_and_only_so(tmp_path):
    log = tmp_path / "meter.jsonl"
    m = meter(log)
    above_the_panel(m)
    items, protected = m.panel_envelope()
    assert items == [x for b in PANEL for x in (f"stage3/{b}", f"strong/stage3/{b}")] + [
        f"stage3/{PANEL[0]}/follow", f"strong/stage3/{PANEL[0]}/follow"]  # fmt: skip
    assert protected == [x for b in PANEL[1:]
                         for x in (f"stage3/{b}/follow", f"strong/stage3/{b}/follow")]  # fmt: skip
    for item in (f"stage3/{PANEL[0]}", f"strong/stage3/{PANEL[2]}", f"stage3/{PANEL[0]}/follow"):
        with pytest.raises(ValueError, match="admitted with the panel"):
            m.admit(item, spent_elsewhere=0, storage=0)
    with pytest.raises(ValueError, match="no follow-up is admitted before the panel's contracts"):
        m.admit(f"stage3/{PANEL[1]}/follow", spent_elsewhere=0, storage=0)
    with pytest.raises(ValueError, match="recorded after its admission"):
        m.panel_contracts(SHA)
    d = m.admit_panel(spent_elsewhere=0, storage=55)
    stress = sum(m._get(i).stress_usd for i in items)
    central = sum(m.allocation(i) for i in protected)
    assert d.allowed and d.terms["next_batch"] == pytest.approx(stress)
    assert d.terms["protected"] == pytest.approx(central)
    assert stress == pytest.approx(5 * 112.99 + 252.74, abs=0.05)  # s6f's illustration
    assert all(m.state[i] == "reserved" for i in items)
    assert all(m.state[i] == "planned" for i in protected)
    with pytest.raises(ValueError, match="admitted once"):
        m.admit_panel(spent_elsewhere=0, storage=55)
    m.start(f"stage3/{PANEL[0]}")  # a preparation starts at once
    first = f"stage3/{PANEL[0]}/follow"
    with pytest.raises(ValueError, match="no follow-up starts before the panel's contracts"):
        m.start(first)
    with pytest.raises(ValueError, match="for want of a reading"):
        m.drop(f"stage3/{PANEL[1]}/follow", "no_registered_readout", entry(PANEL[1]))
    m.panel_contracts(SHA)
    with pytest.raises(ValueError, match="recorded once"):
        m.panel_contracts(SHA)
    m.start(first)
    with pytest.raises(ValueError, match="in the frozen order"):
        m.admit(f"stage3/{PANEL[2]}/follow", spent_elsewhere=0, storage=55)
    m.admit(f"stage3/{PANEL[1]}/follow", spent_elsewhere=0, storage=55)
    m.drop(f"strong/stage3/{PANEL[1]}/follow", "no_registered_readout", entry(PANEL[1]))
    m.admit(f"stage3/{PANEL[2]}/follow", spent_elsewhere=0, storage=55)  # the order kept
    again = meter(log)  # replayed from the log
    assert again.panel == {"admitted": True, "contracts": SHA}
    assert again.state[f"strong/stage3/{PANEL[1]}/follow"] == "dropped"


def test_an_unaffordable_panel_reserves_nothing(tmp_path):
    """Its preparations either all fit, or none starts: the last can never block the others."""
    m = meter(tmp_path / "meter.jsonl", cap=900.0)
    above_the_panel(m)
    d = m.admit_panel(spent_elsewhere=0, storage=55)
    assert not d.allowed
    items, protected = m.panel_envelope()
    assert all(m.state[i] == "planned" for i in (*items, *protected))
    assert m.panel["admitted"] is False


def test_a_preparation_s_cost_never_enters_a_follow_up_s_allocation(tmp_path):
    """The update rule is per kind: a completed preparation updates the preparations' allocation
    alone, never the follow-ups' (review s6f)."""
    m = meter(tmp_path / "meter.jsonl")
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    prep = f"stage3/{PANEL[0]}"
    m.start(prep)
    m.charge(prep, gpu_usd=39.0)
    m.panel_contracts(SHA)
    m.complete(prep, entry(PANEL[0]))
    follow = f"stage3/{PANEL[1]}/follow"
    assert m.allocation(follow) == m._get(follow).usd  # unchanged by the preparation's cost
    assert m.allocation(f"stage3/{PANEL[1]}") == pytest.approx(39.0)  # the preparations' own


def test_the_receipt_gives_the_pod_the_recorded_contracts(tmp_path, capsys):
    """The Pod reads the panel's contracts from the meter's own record (``verify-receipt``'s
    ``panel_contracts``): none before it, its sha256 after it (reviews s6f, s6h 3)."""
    log = tmp_path / "meter.jsonl"
    m = CM.meter(log)
    above_the_panel(m)
    assert m.admit_panel(spent_elsewhere=0, storage=55).allowed
    first = f"stage3/{PANEL[0]}"
    m.start(first)
    m.start(f"strong/{first}")
    m.stage3_start(first, "a")

    def receipt(pod):
        path = tmp_path / f"receipt-{pod}.json"
        assert CM.main(["--log", str(log), "session-open", first, "--worker", "pod", "--pod-id",
                        pod, "--deadline", DEADLINE, "--price", "3.99", "--hours", "1",
                        "--receipt", str(path)]) == 0  # fmt: skip
        capsys.readouterr()
        code = CM.main(
            [
                "--log",
                str(log),
                "verify-receipt",
                str(path),
                "--item",
                first,
                "--worker",
                "pod",
                "--pod-id",
                pod,
                "--price",
                "3.99",
                "--hours",
                "1",
            ]
        )
        out = json.loads(capsys.readouterr().out)
        assert code == 0 and out["stage3"] == {"space": "a"}
        return out

    assert receipt("P1")["panel_contracts"] is None
    m.close_session(first, "pod", pod_id="P1", gpu_usd=1.0)
    m.panel_contracts(SHA)
    assert receipt("P2")["panel_contracts"] == SHA


def test_every_admission_keeps_the_still_protected_follow_ups(tmp_path):
    """S6k 1: after the panel's admission, every admission counts the follow-ups still planned
    that it does not count above it, at their allocations, so a later follow-up keeps the
    protection the panel's admission promised; never twice."""
    m = meter(tmp_path / "meter.jsonl")
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    m.panel_contracts(SHA)
    second = f"stage3/{PANEL[1]}/follow"
    later = [x for b in PANEL[1:] for x in (f"stage3/{b}/follow", f"strong/stage3/{b}/follow")]
    d = m.admit(second, spent_elsewhere=0, storage=55)
    want = sum(m.allocation(x) for x in later if x != second)
    assert d.terms["protected"] == pytest.approx(want) and want > 400
    assert d.terms["higher_priority"] == pytest.approx(0.0)  # the first's are reserved
    pun = m.admit("stage2/pun_intended", spent_elsewhere=0, storage=55)  # above the last's
    assert pun.terms["protected"] == pytest.approx(
        sum(m.allocation(x) for x in later[-2:]))  # fmt: skip
    six = m.admit("stage6", spent_elsewhere=0, storage=55)  # below them all: counted above it
    assert six.terms["protected"] == 0 and six.terms["higher_priority"] >= want


def test_a_preparation_completes_and_a_follow_up_is_released_only_by_its_entry(tmp_path):
    """S6k 2, 3: a preparation completes only with its entry in the recorded contracts (active
    or closed, never before them); a follow-up is released only by an entry with no reading at
    all: with a stack reading the strong follow-up stays funded for the readers' API; both
    enforced when the log is replayed."""
    log = tmp_path / "meter.jsonl"
    m = meter(log)
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    prep, b = f"stage3/{PANEL[0]}", PANEL[0]
    m.start(prep)
    with pytest.raises(ValueError, match="not of the panel's recorded contracts"):
        m.complete(prep, entry(b))
    m.panel_contracts(SHA)
    wrong = [
        (None, "its evidence names"),
        (entry(b, contracts="b" * 64), "not of the panel's recorded"),
        (entry(PANEL[1]), f"not {b}'s"),
        (entry(b, status="pending"), "active or closed entry"),
        (entry(b, status="active", reading=None), "not its status's"),
    ]
    for evidence, problem in wrong:
        with pytest.raises(ValueError, match=problem):
            m.complete(prep, evidence)
    m.complete(prep, entry(b, status="active", reading="strong"))
    m.start(f"stage3/{PANEL[0]}/follow")
    strong = f"strong/stage3/{PANEL[1]}/follow"
    m.admit(f"stage3/{PANEL[1]}/follow", spent_elsewhere=0, storage=55)
    with pytest.raises(ValueError, match="its follow-ups stay"):  # the stack read: funded
        m.drop(strong, "no_registered_readout", entry(PANEL[1], reading="stack"))
    with pytest.raises(ValueError, match="its evidence names"):
        m.drop(strong, "no_registered_readout")
    m.drop(strong, "no_registered_readout", entry(PANEL[1]))
    with pytest.raises(ValueError, match="only a follow-up's release carries evidence"):
        m.drop("stage6", "gate_not_met", entry(PANEL[1]))
    lines = log.read_text().splitlines()
    forged = [json.loads(x) for x in lines]
    for e in forged:
        if e.get("event") == "drop" and e.get("item") == strong:
            e["evidence"]["reading"] = "stack"
    log.write_text("".join(json.dumps(e) + "\n" for e in forged))
    with pytest.raises(ValueError, match="its follow-ups stay"):
        meter(log)  # replayed: the release's condition enforced


def test_money_spent_elsewhere_is_never_negative(tmp_path):
    """S6k 4: the money spent elsewhere is a finite non-negative amount before the ledger's is
    added, and a replayed admission's spending below the ledger's is refused."""
    log = tmp_path / "meter.jsonl"
    m = meter(log)
    m.admit("stage0_1", spent_elsewhere=0, storage=0)
    m.start("stage0_1")
    m.charge("stage0_1", gpu_usd=50.0)
    m.complete("stage0_1")
    for bad in (-50, float("nan"), "50"):
        with pytest.raises(ValueError, match="the money spent elsewhere"):
            m.admit("stage2/example_link", spent_elsewhere=bad, storage=0)
    m.admit("stage2/example_link", spent_elsewhere=0, storage=0)
    events = [json.loads(x) for x in log.read_text().splitlines()]
    events[-1]["terms"]["spent"] = 0.0  # as -50 + 50 would have given
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(ValueError, match="below the ledger's"):
        meter(log)


def test_a_replayed_admission_is_the_meter_s_own(tmp_path):
    """S6k 5: replaying the panel's admission (and every admission from the phases on) checks its
    keys, its terms against the state, its cap and its decision: a refused admission turned
    allowed is refused, as is a changed term."""
    log = tmp_path / "meter.jsonl"
    m = meter(log, cap=900.0)
    above_the_panel(m)
    assert not m.admit_panel(spent_elsewhere=0, storage=55).allowed
    events = [json.loads(x) for x in log.read_text().splitlines()]
    for change, problem in (({"allowed": True}, "decided True on a total"),
                            ({"cap": 1e5}, "under the cap"),
                            ({"extra": 1}, "has the keys")):  # fmt: skip
        log.write_text("".join(json.dumps({**e, **change} if e["event"] == "admit_panel" else e)
                               + "\n" for e in events))  # fmt: skip
        with pytest.raises(ValueError, match=problem):
            meter(log, cap=900.0)
    terms = {**events[-1]["terms"], "protected": 0.0}
    log.write_text("".join(json.dumps({**e, "terms": terms} if e["event"] == "admit_panel" else e)
                           + "\n" for e in events))  # fmt: skip
    with pytest.raises(ValueError, match=r"terms \['protected'\] are not the meter's state's"):
        meter(log, cap=900.0)


def test_the_cli_takes_the_evidence_from_the_committed_contracts(tmp_path, capsys, monkeypatch):
    """S6k 2, 3: ``complete`` of a preparation and ``drop … no_registered_readout`` read the
    behavior's entry from the committed contracts read whole (``committed_bundle``, stood for
    here), refused unless they are the ones the meter recorded; an entry with a reading keeps
    its follow-ups."""
    from residual_disposition import continuation_strong as CS

    log = tmp_path / "meter.jsonl"
    m = CM.meter(log)
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    prep, b = f"stage3/{PANEL[0]}", PANEL[0]
    m.start(prep)
    m.panel_contracts(SHA)
    behaviors = {x: {"status": "closed", "reading": None} for x in PANEL}
    behaviors[PANEL[1]] = {"status": "closed", "reading": "stack"}
    bundle = {"sha256": "b" * 64, "behaviors": behaviors}
    monkeypatch.setattr(CS, "committed_bundle", lambda root, rules, find_run: bundle)
    with pytest.raises(SystemExit, match="are not the ones the meter recorded"):
        CM.main(["--log", str(log), "complete", prep])
    bundle["sha256"] = SHA
    assert CM.main(["--log", str(log), "complete", prep]) == 0
    capsys.readouterr()
    done = [json.loads(x) for x in log.read_text().splitlines()][-1]
    assert done["evidence"] == entry(b)
    m.start(f"stage3/{PANEL[0]}/follow")
    m.admit(f"stage3/{PANEL[1]}/follow", spent_elsewhere=0, storage=55)
    with pytest.raises(SystemExit, match="its entry is"):
        CM.main(["--log", str(log), "drop", f"strong/stage3/{PANEL[1]}/follow",
                 "no_registered_readout"])  # fmt: skip
    behaviors[PANEL[1]] = {"status": "closed", "reading": None}
    assert CM.main(["--log", str(log), "drop", f"strong/stage3/{PANEL[1]}/follow",
                    "no_registered_readout"]) == 0  # fmt: skip
