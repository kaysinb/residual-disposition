"""The panel's phases in the meter (§12.8 as amended 2026-10-22): each panel lineage's
items are their preparation phases and their follow-ups are items of their own, the two summing
to the lineage's allocation; the five behaviors' preparations and the first behavior's follow-up
are admitted at once at stress, the other follow-ups protected at their allocations; the
contracts recorded once, after it; no follow-up starts before them, and the later ones are
admitted after them, in the frozen order; a follow-up without a registered reading is
released."""

from __future__ import annotations

import hashlib
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
CLOSED = {"status": "closed", "reading": None}


def bundle(**entries):
    """The recorded contracts: their file's text (each behavior closed with no reading unless
    given) and its sha256, as ``panel-contracts`` journals them."""
    text = json.dumps({"panel": PANEL, "behaviors": {b: entries.get(b, CLOSED) for b in PANEL}})
    return hashlib.sha256(text.encode()).hexdigest(), text


SHA, TEXT = bundle()
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
        m.panel_contracts(SHA, TEXT)
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
    m.panel_contracts(SHA, TEXT)
    with pytest.raises(ValueError, match="recorded once"):
        m.panel_contracts(SHA, TEXT)
    m.start(first)
    with pytest.raises(ValueError, match="in the frozen order"):
        m.admit(f"stage3/{PANEL[2]}/follow", spent_elsewhere=0, storage=55)
    m.admit(f"stage3/{PANEL[1]}/follow", spent_elsewhere=0, storage=55)
    m.drop(f"strong/stage3/{PANEL[1]}/follow", "no_registered_readout", entry(PANEL[1]))
    m.admit(f"stage3/{PANEL[2]}/follow", spent_elsewhere=0, storage=55)  # the order kept
    again = meter(log)  # replayed from the log
    assert again.panel == {"admitted": True, "contracts": SHA,
                           "entries": dict.fromkeys(PANEL, CLOSED)}  # fmt: skip
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
    alone, never the follow-ups'."""
    m = meter(tmp_path / "meter.jsonl")
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    prep = f"stage3/{PANEL[0]}"
    m.start(prep)
    m.charge(prep, gpu_usd=39.0)
    m.panel_contracts(SHA, TEXT)
    m.complete(prep, entry(PANEL[0]))
    follow = f"stage3/{PANEL[1]}/follow"
    assert m.allocation(follow) == m._get(follow).usd  # unchanged by the preparation's cost
    assert m.allocation(f"stage3/{PANEL[1]}") == pytest.approx(39.0)  # the preparations' own


def test_the_receipt_gives_the_pod_the_recorded_contracts(tmp_path, capsys):
    """The Pod reads the panel's contracts from the meter's own record (``verify-receipt``'s
    ``panel_contracts``): none before it, its sha256 after it."""
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
    m.panel_contracts(SHA, TEXT)
    assert receipt("P2")["panel_contracts"] == SHA


def test_every_admission_keeps_the_still_protected_follow_ups(tmp_path):
    """S6k 1: after the panel's admission, every admission counts the follow-ups still planned
    that it does not count above it, at their allocations, so a later follow-up keeps the
    protection the panel's admission promised; never twice."""
    m = meter(tmp_path / "meter.jsonl")
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    m.panel_contracts(SHA, TEXT)
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
    """S6k 2, 3, s6l 1: the recorded contracts are journaled with their file's text, which hashes
    to their sha256 and holds the frozen panel's entries; a preparation completes, and a
    follow-up is released, only by its behavior's recorded entry, never by a caller's claim
    contradicting it: released only with no reading at all (a stack reading keeps the strong
    follow-up funded for the readers' API); all enforced when the log is replayed."""
    log = tmp_path / "meter.jsonl"
    m = meter(log)
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    prep, b = f"stage3/{PANEL[0]}", PANEL[0]
    strong, stack = {"status": "active", "reading": "strong"}, {"status": "closed",
                                                               "reading": "stack"}  # fmt: skip
    sha, text = bundle(**{PANEL[0]: strong, PANEL[1]: stack})
    m.start(prep)
    with pytest.raises(ValueError, match="not of the panel's recorded contracts"):
        m.complete(prep, entry(b, contracts=sha, **strong))
    empty_sha, empty = hashlib.sha256(b'{"panel": [], "behaviors": {}}').hexdigest(), (
        '{"panel": [], "behaviors": {}}')  # fmt: skip
    for args, problem in (((empty_sha, empty), "not the frozen panel"),
                          ((sha, text + " "), "hashes to their sha256"),
                          ((sha, None), "hashes to their sha256")):  # fmt: skip
        with pytest.raises(ValueError, match=problem):
            m.panel_contracts(*args)
    m.panel_contracts(sha, text)
    assert m.panel["entries"][b] == strong and m.panel["entries"][PANEL[1]] == stack
    wrong = [
        (None, "its evidence names"),
        (entry(b, contracts="b" * 64, **strong), "not of the panel's recorded"),
        (entry(PANEL[1], contracts=sha, **stack), f"not {b}'s"),
        (entry(b, contracts=sha), "contradicts"),  # closed with no reading: not its record's
    ]
    for evidence, problem in wrong:
        with pytest.raises(ValueError, match=problem):
            m.complete(prep, evidence)
    m.complete(prep, entry(b, contracts=sha, **strong))
    with pytest.raises(ValueError, match="contradicts"):  # the probe of s6l 1
        m.drop(f"strong/stage3/{b}/follow", "no_registered_readout", entry(b, contracts=sha))
    m.start(f"stage3/{b}/follow")
    m.admit(f"stage3/{PANEL[1]}/follow", spent_elsewhere=0, storage=55)
    follow = f"strong/stage3/{PANEL[1]}/follow"
    with pytest.raises(ValueError, match="its follow-ups stay"):  # the stack read: funded
        m.drop(follow, "no_registered_readout", entry(PANEL[1], contracts=sha, **stack))
    with pytest.raises(ValueError, match="contradicts"):  # its record has the reading
        m.drop(follow, "no_registered_readout", entry(PANEL[1], contracts=sha))
    m.admit(follow, spent_elsewhere=0, storage=55)
    for item in (f"stage3/{PANEL[2]}/follow", f"strong/stage3/{PANEL[2]}/follow"):
        m.drop(item, "no_registered_readout", entry(PANEL[2], contracts=sha))  # no reading
    with pytest.raises(ValueError, match="only a follow-up's release carries evidence"):
        m.drop("stage6", "gate_not_met", entry(PANEL[2], contracts=sha))
    assert meter(log).panel["entries"] == m.panel["entries"]  # replayed from the journal
    events = [json.loads(x) for x in log.read_text().splitlines()]

    def replay(change):
        log.write_text("".join(json.dumps(change(dict(e))) + "\n" for e in events))
        return meter(log)

    def forge_release(e):
        if e.get("event") == "drop" and e.get("item") == f"stage3/{PANEL[2]}/follow":
            e["evidence"] = {**e["evidence"], "reading": "stack"}
        return e

    def forge_text(e):
        if e.get("event") == "panel_contracts":
            e["bundle"] = e["bundle"].replace('"stack"', '"strong"')
        return e

    with pytest.raises(ValueError, match="contradicts"):
        replay(forge_release)
    with pytest.raises(ValueError, match="hashes to their sha256"):
        replay(forge_text)


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
    """S6k 2, 3, s6l 1: ``panel-contracts`` records the committed contracts read whole
    (``committed_bundle``, stood for here) with their file's text; ``complete`` of a preparation
    and ``drop … no_registered_readout`` read the behavior's entry from them, refused unless
    they are the ones the meter recorded; the meter refuses what contradicts its record."""
    from residual_disposition import continuation_strong as CS

    log = tmp_path / "meter.jsonl"
    m = CM.meter(log)
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    prep, b = f"stage3/{PANEL[0]}", PANEL[0]
    m.start(prep)
    stack = {"status": "closed", "reading": "stack"}
    sha, text = bundle(**{PANEL[1]: stack})
    path = tmp_path / "stage3_panel_contracts.json"
    monkeypatch.setattr(CS, "contracts_path", lambda root: path)
    behaviors = json.loads(text)["behaviors"]
    held = {"sha256": sha, "behaviors": behaviors}
    monkeypatch.setattr(CS, "committed_bundle", lambda root, rules, find_run: held)
    path.write_text(text + " ")  # not the file whose sha256 the bundle names
    with pytest.raises(SystemExit, match="changed while it was read"):
        CM.main(["--log", str(log), "panel-contracts"])
    path.write_text(text)
    assert CM.main(["--log", str(log), "panel-contracts"]) == 0
    assert json.loads(capsys.readouterr().out)["entries"][PANEL[1]] == stack
    held["sha256"] = "b" * 64
    with pytest.raises(SystemExit, match="are not the ones the meter recorded"):
        CM.main(["--log", str(log), "complete", prep])
    held["sha256"] = sha
    assert CM.main(["--log", str(log), "complete", prep]) == 0
    capsys.readouterr()
    done = [json.loads(x) for x in log.read_text().splitlines()][-1]
    assert done["evidence"] == entry(b, contracts=sha)
    m.start(f"stage3/{PANEL[0]}/follow")
    m.admit(f"stage3/{PANEL[1]}/follow", spent_elsewhere=0, storage=55)
    with pytest.raises(SystemExit, match="its entry is"):
        CM.main(["--log", str(log), "drop", f"strong/stage3/{PANEL[1]}/follow",
                 "no_registered_readout"])  # fmt: skip
    behaviors[PANEL[1]] = CLOSED  # the file as read now contradicts the meter's record
    with pytest.raises(ValueError, match="contradicts"):
        CM.main(["--log", str(log), "drop", f"strong/stage3/{PANEL[1]}/follow",
                 "no_registered_readout"])  # fmt: skip


def test_the_bundle_parser_refuses_repeated_keys_and_an_unstated_reading(tmp_path):
    """S6m 2: a key given twice anywhere in the recorded contracts' text is refused (JSON would
    keep the last silently), and every entry states its reading, none included."""
    m = meter(tmp_path / "meter.jsonl")
    above_the_panel(m)
    m.admit_panel(spent_elsewhere=0, storage=55)
    _, text = bundle()
    first = json.dumps(PANEL[0])
    twice = text.replace(f'{first}: {{"status"', f'{first}: {{"status": "active", "status"', 1)
    unstated = text.replace(f'{first}: {{"status": "closed", "reading": null}}',
                            f'{first}: {{"status": "closed"}}', 1)  # fmt: skip
    for bad, problem in ((twice, "repeat a key"), (unstated, "with the reading 'absent'")):
        assert bad != text
        with pytest.raises(ValueError, match=problem):
            m.panel_contracts(hashlib.sha256(bad.encode()).hexdigest(), bad)
    m.panel_contracts(SHA, TEXT)
