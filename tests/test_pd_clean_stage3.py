"""PD-clean of a certified Stage-3 lineage (scripts/pd_clean.py, pd_clean_study) on synthetic
pools: the template expanded from the lineage's readers' record with its named pools, the review's
sample per package, and the packages recomputed from the study's records
(``verify_packages``, the readers' verdicts' provenance check)."""

from __future__ import annotations

import contextlib
import hashlib
import json

import pytest
import yaml
from test_pd_clean_script import REPO, items, load, pd_run

from residual_disposition import continuation_panel as G
from residual_disposition import pd_clean_study as S
from residual_disposition.pipeline import content_hash

STATES = {"L": "sL", "H_s": "sHs", "X": "sX", "A": "sA", "recipe": "sR"}
RUBRIC = "configs/continuation/pd_clean_rubric_stage3.md"
RUBRIC_SHA = "9f8559fd42ff62398c1d22f183a30e25feaaaaa6d39369134aead8c0609c56e8"


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load("pd_clean_script_s3", REPO / "scripts" / "pd_clean.py")
    first = max(range(300), key=lambda i: (i * 7919 % 1000, -i))
    for role, sid in STATES.items():
        marked = {("chat", first)} if role == "L" else set()
        pd_run(tmp_path / "runs", f"pool-{role}", sid, marked=marked)
    readers = {r: {"state": sid, "pd": f"pool-{r}"} for r, sid in STATES.items()}
    readers["H"] = {"state": "sH", "pd": None}
    record = {"behavior": "example_link", "lineage": "stage3_example_link_100", "dose": 100,
              "readers": readers}  # fmt: skip
    run = tmp_path / "runs" / "continuation_14b" / "rec"
    run.mkdir(parents=True)
    (run / "stage3_readers.json").write_text(json.dumps(record))
    (run / "config.yaml").write_text(yaml.safe_dump(
        {"step": "stage3_readers", "params": {"record": content_hash(record)}}))  # fmt: skip
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    plan = {"pd_clean_stage3": {
        "fraction": 0.03, "seed": 5, "adjudicate": ["L", "A", "recipe"],
        "random": ["L", "A", "H_s"], "rubric": RUBRIC, "rubric_sha256": RUBRIC_SHA,
        "ensemble": ENSEMBLE,
        "review": {"negatives_per_package": 2, "allocation": "per_package"}}}  # fmt: skip
    path = tmp_path / "plan3.yaml"
    path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(S, "STAGE3_PLAN", path)
    work = tmp_path / "data" / "pd_clean_stage3" / "stage3_example_link_100"
    base = ["--plan", str(path), "--record", str(run)]
    return script, base, tmp_path, run, work


ENSEMBLE = {
    "annotators": {"opus": {"model": "claude-opus-5-5", "runner": "independent subagent"},
                   "fable": {"model": "claude-fable-5-1", "runner": "independent subagent"},
                   "gpt": {"model": "gpt-6-astra", "runner": "codex-cli, read-only"}},
    "arbiter": {"model": "claude-opus-5-5", "runner": "fresh subagent", "votes": "anonymous"},
    "evidence_from": "opus",
}  # fmt: skip


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def kit_shas(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def annotator_exports(out, labels_, kit_dir):
    """The annotators' exports that give ``labels_`` (``{iid: {present, unanimous, confident,
    evidence}}``): all three agreeing, or two against one; their manifest on the round's kit.
    Returns the arbiter's rows due for the disputed items."""
    names, rows, arbiter = sorted(ENSEMBLE["annotators"]), {}, []
    for n in names:
        rows[n] = []
    for iid, lab in sorted(labels_.items()):
        evidence = lab.get("evidence") or ""
        if lab.get("unanimous", True):
            for n in names:
                rows[n].append(
                    {
                        "item_id": iid,
                        "present": lab["present"],
                        "confident": lab.get("confident", True),
                        "evidence": evidence,
                    }
                )
        else:
            for n, present in zip(names, [lab["present"], lab["present"], not lab["present"]],
                                  strict=True):  # fmt: skip
                rows[n].append({"item_id": iid, "present": present, "confident": True,
                                "evidence": "q" if present else ""})  # fmt: skip
            arbiter.append({"item_id": iid, "present": lab["present"],
                            "confident": lab.get("confident", True), "evidence": evidence,
                            "reason": "the votes differ"})  # fmt: skip
    for n in names:
        write_jsonl(out / "annotators" / f"{n}.jsonl", rows[n])
    kit = kit_shas(kit_dir)
    (out / "annotators_manifest.json").write_text(
        json.dumps({"annotators": {n: {**ENSEMBLE["annotators"][n], "kit": kit} for n in names}})
    )
    return arbiter


def arbiter_exports(out, arbiter, kit_dir):
    write_jsonl(out / "arbiter.jsonl", arbiter)
    (out / "arbiter_manifest.json").write_text(json.dumps(
        {"arbiter": {**ENSEMBLE["arbiter"], "kit": kit_shas(kit_dir)}}))  # fmt: skip


def label_round(script, base, tmp, k, trace=None, dispute=None):
    """Round k's exports and their two-step import: ``trace`` present unanimously, ``dispute``
    decided present by the arbiter; every other item absent unanimously."""
    work = tmp / "data" / "pd_clean_stage3" / "stage3_example_link_100"
    d = work / f"round{k}"
    labels_ = {}
    for i in items(d / "items.jsonl"):
        iid = i["item_id"]
        labels_[iid] = {
            "present": iid in (trace, dispute),
            "unanimous": iid != dispute,
            "confident": True,
            "evidence": "x" if iid in (trace, dispute) else None,
        }
    out = tmp / f"exports{k}"
    arbiter = annotator_exports(out, labels_, d / "kit" / "annotators")
    script.main([*base, "import-annotators", "--exports", str(out)])
    if arbiter:
        arbiter_exports(out, arbiter, d / "kit" / "arbiter")
        script.main([*base, "import-arbiter", "--exports", str(out)])
    return labels_


def import_review(script, base, tmp, work, j, present=()):
    """The page's export and log of review j, every item absent but ``present``."""
    mapping = json.loads((work / f"review{j}" / "mapping.json").read_text())
    rows = [{"item_id": i, "present": i in present, "note": ""} for i in mapping["items"]]
    export = write_jsonl(tmp / f"page{j}.jsonl", rows)
    shown = hashlib.sha256((work / f"review{j}" / "items.jsonl").read_bytes()).hexdigest()
    log = tmp / f"page{j}.log.json"
    log.write_text(json.dumps({"page_id": f"p{j}", "source": {"items_sha256": shown}}))
    script.main([*base, "import-review", "--export", str(export), "--log", str(log)])
    return mapping


def packages(tmp):
    out = {}
    for r in sorted((tmp / "runs" / "pd_clean").glob("*")):
        config = yaml.safe_load((r / "config.yaml").read_text())
        kind = {"pd_clean_package": "pd_clean", "pd_random_package": "pd_random"}[config["step"]]
        out[(config["params"]["role"], kind)] = r
    return out


def complete(study):
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    rows = label_round(script, base, tmp, 0)
    script.main([*base, "review"])
    mapping = import_review(script, base, tmp, work, 0)
    script.main([*base, "package"])
    return rows, mapping


def test_the_lineage_s_states_pools_and_review_per_package(study):
    script, base, tmp, run, work = study
    st = script.Study(work, base[1], run)
    assert [(s["role"], s["adjudicate"], s["random"]) for s in st.states] == [
        ("L", True, True), ("H_s", False, True), ("X", False, False), ("A", True, True),
        ("recipe", True, False)]  # fmt: skip
    assert {k: v.name for k, v in st.runs.items()} == {sid: f"pool-{r}" for r, sid in
                                                      STATES.items()}  # fmt: skip
    assert st.binding["record"]["lineage"] == "stage3_example_link_100"
    _, mapping = complete(study)
    assert sorted(mapping["sampled_groups"]) == [
        "example_link/A/sA", "example_link/L/sL", "example_link/recipe/sR"]  # fmt: skip
    assert sum(v["why"] == "sample" for v in mapping["items"].values()) == 6  # 2 per package


def test_the_packages_are_their_study_s_records(study):
    script, base, tmp, run, work = study
    complete(study)
    found = packages(tmp)
    assert sorted(found) == sorted([(r, "pd_clean") for r in STATES] +
                                   [(r, "pd_random") for r in ("L", "A", "H_s")])  # fmt: skip
    records = S.verify_packages(work, run, found)
    assert "round0/labels.json" in records and "review0/labels.json" in records
    with pytest.raises(ValueError, match="are not"):
        S.verify_packages(work, run, {k: v for k, v in found.items() if k != ("X", "pd_clean")})
    path = found[("L", "pd_clean")] / "pd_clean.json"
    package = json.loads(path.read_text())
    config = next(iter(package["top"]))
    package["top"][config] = package["top"][config][1:]
    path.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="not L's pd_clean package"):
        S.verify_packages(work, run, found)


def test_a_package_s_parameters_are_its_study_s(study):
    script, base, tmp, run, work = study
    complete(study)
    found = packages(tmp)
    path = found[("A", "pd_random")] / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["params"]["seed"] = 6
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="not A's pd_random package"):
        S.verify_packages(work, run, found)


def test_an_open_round_or_review_is_pending(study):
    script, base, tmp, run, work = study
    with pytest.raises(G.Pending, match="no PD-clean study"):
        S.verify_packages(work, run, {})
    script.main([*base, "round"])
    with pytest.raises(G.Pending, match="round is open"):
        S.verify_packages(work, run, {})
    label_round(script, base, tmp, 0)
    with pytest.raises(G.Pending, match="review is not complete"):
        S.verify_packages(work, run, {})


def test_a_stage3_template_without_its_rubric_fails_closed(study, monkeypatch):
    script, base, tmp, run, work = study
    complete(study)
    path = tmp / "plan_norubric.yaml"
    plan = yaml.safe_load(S.STAGE3_PLAN.read_text())
    del plan["pd_clean_stage3"]["rubric"], plan["pd_clean_stage3"]["rubric_sha256"]
    path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(S, "STAGE3_PLAN", path)
    with pytest.raises(ValueError, match="not a complete Stage-3 study"):
        S.verify_packages(work, run, packages(tmp))


def test_a_stage3_plan_needs_its_record_and_a_stage1_plan_none(study):
    script, base, tmp, run, work = study
    with pytest.raises(SystemExit, match="one certified lineage"):
        script.Study(work, base[1])


def _forge(work, edit, name="review0", study=None):
    """A review record rewritten as a strong forger would: its mapping and its labels, and with
    them every file derived from them (the items read, the page's export and log), so that only
    the checks of what the records imply can refuse it."""
    d = work / name
    mapping_path, labels_path = d / "mapping.json", d / "labels.json"
    mapping, labels_ = json.loads(mapping_path.read_text()), json.loads(labels_path.read_text())
    edit(mapping, labels_)
    mapping_path.write_text(json.dumps(mapping))
    restate(d, mapping, labels_, study)


def restate(d, mapping, labels_, study=None):
    """Write a review's derived files from its mapping and labels (``labels_``, the labels.json
    record): the items regenerated when the study can, the export, the log, the sources."""
    if study is not None:
        st = study()
        after = mapping.get("after") or {}
        only = set(after.get("round") or []) | set(after.get("review") or [])
        with contextlib.suppress(KeyError, SystemExit):  # a mapping the study refuses
            (d / "items.jsonl").write_text(S.review_items(st, mapping, int(d.name[6:]),
                                                          st.ensemble(only)))  # fmt: skip
    rows = [{"item_id": i, **v} for i, v in labels_["labels"].items()]
    write_jsonl(d / "export.jsonl", rows)
    shown = hashlib.sha256((d / "items.jsonl").read_bytes()).hexdigest() if (
        d / "items.jsonl").is_file() else "0"  # fmt: skip
    (d / "page_log.json").write_text(
        json.dumps({"page_id": "p", "source": {"items_sha256": shown}})
    )
    labels_["source"] = {
        "exports": {
            "export.jsonl": hashlib.sha256((d / "export.jsonl").read_bytes()).hexdigest(),
            "page_log.json": hashlib.sha256((d / "page_log.json").read_bytes()).hexdigest(),
        }
    }
    (d / "labels.json").write_text(json.dumps(labels_))


def test_a_review_without_its_seeded_negatives_is_refused(study):
    """All three groups declared sampled, with no sample item reviewed: refused."""
    script, base, tmp, run, work = study
    complete(study)

    def drop(mapping, labels_):
        samples = [i for i, v in mapping["items"].items() if v["why"] == "sample"]
        for iid in samples:
            del mapping["items"][iid], labels_["labels"][iid]

    _forge(work, drop, study=lambda: script.Study(work, base[1], run))
    with pytest.raises(ValueError, match="not its seeded sample"):
        S.verify_packages(work, run, packages(tmp))


def test_a_sample_of_other_negatives_is_refused(study):
    script, base, tmp, run, work = study
    complete(study)
    st = script.Study(work, base[1], run)
    final = st.final()
    sel = st.selection(st.states[0], final)  # L
    mapping = json.loads((work / "review0" / "mapping.json").read_text())
    taken = set(mapping["items"])
    other = next(iid for iid in sel.ids("sL") if iid not in taken)

    def swap(mapping, labels_):
        iid = next(i for i, v in mapping["items"].items() if v["why"] == "sample"
                   and v["state"] == "sL")  # fmt: skip
        config, row = sel.ids("sL")[other]
        mapping["items"].pop(iid)
        mapping["items"][other] = {"state": "sL", "config": config, "prefill": row["prefill"],
                                   "why": "sample"}  # fmt: skip
        labels_["labels"][other] = labels_["labels"].pop(iid)

    _forge(work, swap, study=lambda: script.Study(work, base[1], run))
    with pytest.raises(ValueError, match="not its seeded sample"):
        S.verify_packages(work, run, packages(tmp))


def test_a_review_that_does_not_name_its_records_is_refused(study):
    script, base, tmp, run, work = study
    complete(study)
    _forge(
        work,
        lambda mapping, labels_: mapping.pop("after"),
        study=lambda: script.Study(work, base[1], run),
    )
    with pytest.raises(ValueError, match="not drawn after a prefix"):
        S.verify_packages(work, run, packages(tmp))


def test_a_review_drawn_on_fewer_records_than_it_had_is_refused(study):
    script, base, tmp, run, work = study
    complete(study)
    _forge(
        work,
        lambda mapping, labels_: mapping.update(after={"round": [], "review": []}),
        study=lambda: script.Study(work, base[1], run),
    )
    with pytest.raises(ValueError, match="had unlabeled items"):
        S.verify_packages(work, run, packages(tmp))


def test_a_review_s_items_are_each_covered_by_a_typed_label(study):
    """A correct sample with no imported label (or an untyped one) is not a review."""
    script, base, tmp, run, work = study
    complete(study)
    _forge(
        work,
        lambda mapping, labels_: labels_.update(labels={}),
        study=lambda: script.Study(work, base[1], run),
    )
    with pytest.raises(ValueError, match="not exactly the review's items"):
        S.verify_packages(work, run, packages(tmp))


def test_an_untyped_review_label_is_refused(study):
    script, base, tmp, run, work = study
    complete(study)

    def untype(mapping, labels_):
        iid = next(iter(labels_["labels"]))
        labels_["labels"][iid] = {"present": "no"}

    _forge(work, untype, study=lambda: script.Study(work, base[1], run))
    with pytest.raises(ValueError, match="not a row of its export's schema"):
        S.verify_packages(work, run, packages(tmp))


def test_a_round_s_items_are_each_covered_by_a_typed_label(study):
    script, base, tmp, run, work = study
    complete(study)
    path = work / "round0" / "labels.json"
    data = json.loads(path.read_text())
    data["labels"].pop(next(iter(data["labels"])))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not its exports' transformation"):
        S.verify_packages(work, run, packages(tmp))


def test_a_review_s_mapping_never_overrides_the_researcher_s_label(study):
    """A trace the researcher found cannot be erased by a ``present: false`` written into the
    review's mapping: a mapping's items hold their metadata alone."""
    script, base, tmp, run, work = study
    complete(study)

    def inject(mapping, labels_):
        iid = next(iter(labels_["labels"]))
        labels_["labels"][iid] = {"present": True, "note": "a trace"}
        mapping["items"][iid]["present"] = False

    _forge(work, inject, study=lambda: script.Study(work, base[1], run))
    with pytest.raises(ValueError, match="mapping's items are not review items"):
        S.verify_packages(work, run, packages(tmp))
    with pytest.raises(SystemExit, match="metadata is not a review item's"):
        script.Study(work, base[1], run).final()


def test_a_round_s_mapping_never_overrides_the_ensemble_s_label(study):
    script, base, tmp, run, work = study
    complete(study)
    path = work / "round0" / "mapping.json"
    mapping = json.loads(path.read_text())
    mapping["items"][next(iter(mapping["items"]))]["unanimous"] = False
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="mapping's items are not round items"):
        S.verify_packages(work, run, packages(tmp))
    with pytest.raises(SystemExit, match="metadata is not a round item's"):
        script.Study(work, base[1], run).ensemble()


def test_a_reviewed_trace_cannot_be_moved_to_another_pool_row(study):
    """Re-pointing a reviewed item's metadata at another valid row of its pool (its id kept)
    would keep the label and move the exclusion: the id is its row's, so it is refused."""
    script, base, tmp, run, work = study
    complete(study)
    st = script.Study(work, base[1], run)
    rows = sorted({(r["config"], r["prefill"]) for r in st.rows["sL"]})

    def move(mapping, labels_):
        iid = next(i for i, v in mapping["items"].items() if v["state"] == "sL")
        labels_["labels"][iid] = {"present": True, "note": "a trace"}
        item = mapping["items"][iid]
        other = next(r for r in rows if r != (item["config"], item["prefill"]))
        item["config"], item["prefill"] = other

    _forge(work, move, study=lambda: script.Study(work, base[1], run))
    with pytest.raises(ValueError, match="not its own pool row"):
        S.verify_packages(work, run, packages(tmp))
    with pytest.raises(SystemExit, match="not its own pool row"):
        script.Study(work, base[1], run).final()


def test_a_round_item_naming_no_row_of_its_pool_is_refused(study):
    script, base, tmp, run, work = study
    complete(study)
    path = work / "round0" / "mapping.json"
    mapping = json.loads(path.read_text())
    iid = next(iter(mapping["items"]))
    mapping["items"][iid]["prefill"] = "p99999"
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="not its own pool row"):
        S.verify_packages(work, run, packages(tmp))


def complete_with_trace(study):
    """A study whose round 0 found one trace, refilled by round 1, and whose review holds it."""
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    first = items(work / "round0" / "items.jsonl")[0]["item_id"]
    label_round(script, base, tmp, 0, trace=first)
    script.main([*base, "round"])
    label_round(script, base, tmp, 1)
    script.main([*base, "review"])
    mapping = json.loads((work / "review0" / "mapping.json").read_text())
    assert mapping["items"][first]["why"] == "trace"
    import_review(script, base, tmp, work, 0, present={first})
    script.main([*base, "package"])
    return first


def test_a_reviewed_trace_cannot_be_reviewed_again(study):
    """A later review, validly bound, that labels an already reviewed trace absent would bring
    it back into the selection: an item is reviewed once."""
    script, base, tmp, run, work = study
    trace = complete_with_trace(study)
    review0 = json.loads((work / "review0" / "mapping.json").read_text())
    labels0 = json.loads((work / "review0" / "labels.json").read_text())
    d = work / "review1"
    d.mkdir()
    after = {"round": ["round0", "round1"], "review": ["review0"]}
    mapping = {"binding": review0["binding"], "after": after, "sampled_groups": [],
               "items": {trace: review0["items"][trace]}}  # fmt: skip
    (d / "mapping.json").write_text(json.dumps(mapping))
    restate(d, mapping, {"binding": labels0["binding"],
                         "labels": {trace: {"present": False, "note": None}}},
            lambda: script.Study(work, base[1], run))  # fmt: skip
    with pytest.raises(ValueError, match="reviewed again"):
        S.verify_packages(work, run, packages(tmp))


def test_a_review_without_a_due_trace_is_refused(study):
    """A review must hold every item due then, its disputes and traces as well as its sample: a
    trace left out of its review and reviewed later instead is refused."""
    script, base, tmp, run, work = study
    trace = complete_with_trace(study)
    review0 = json.loads((work / "review0" / "mapping.json").read_text())
    labels0 = json.loads((work / "review0" / "labels.json").read_text())
    item, label = review0["items"][trace], labels0["labels"][trace]

    def drop(mapping, labels_):
        del mapping["items"][trace], labels_["labels"][trace]

    _forge(work, drop, study=lambda: script.Study(work, base[1], run))
    d = work / "review1"
    d.mkdir()
    after = {"round": ["round0", "round1"], "review": ["review0"]}
    mapping = {"binding": review0["binding"], "after": after, "sampled_groups": [],
               "items": {trace: item}}  # fmt: skip
    (d / "mapping.json").write_text(json.dumps(mapping))
    restate(d, mapping, {"binding": labels0["binding"], "labels": {trace: label}},
            lambda: script.Study(work, base[1], run))  # fmt: skip
    with pytest.raises(ValueError, match="review0: not the review due"):
        S.verify_packages(work, run, packages(tmp))


def test_a_round_item_labeled_twice_is_refused(study):
    script, base, tmp, run, work = study
    complete_with_trace(study)
    round0 = json.loads((work / "round0" / "mapping.json").read_text())
    labels0 = json.loads((work / "round0" / "labels.json").read_text())
    iid = next(iter(round0["items"]))
    d = work / "round2"
    d.mkdir()
    mapping = {**round0, "items": {iid: round0["items"][iid]}}
    (d / "mapping.json").write_text(json.dumps(mapping))
    flipped = {iid: {**labels0["labels"][iid], "present": not labels0["labels"][iid]["present"],
                     "evidence": "x"}}  # fmt: skip
    st = script.Study(work, base[1], run)
    (d / "items.jsonl").write_text(S.round_items(st, mapping, 2))
    kit = S.annotator_kit(st.rubric_text(), items(d / "items.jsonl"))
    for name, text in kit.items():
        (d / "kit" / "annotators").mkdir(parents=True, exist_ok=True)
        (d / "kit" / "annotators" / name).write_text(text)
    annotator_exports(d / "exports", flipped, d / "kit" / "annotators")
    labels_, shas = S.round_labels(st, d)
    (d / "labels.json").write_text(
        json.dumps({"binding": labels0["binding"], "source": {"exports": shas}, "labels": labels_})
    )
    with pytest.raises(ValueError, match="labeled again"):
        S.verify_packages(work, run, packages(tmp))


# --- what the annotators read and what they returned -------------------------------------------


def test_what_the_ensemble_read_is_its_records_items(study):
    script, base, tmp, run, work = study
    complete(study)
    path = work / "round0" / "items.jsonl"
    text = path.read_text()
    path.write_text(text.replace("[", "[an edited ", 1))  # its kit then differs too
    with pytest.raises(ValueError, match="round0: its"):
        S.verify_packages(work, run, packages(tmp))
    path.unlink()
    with pytest.raises(ValueError, match="round0: its"):
        S.verify_packages(work, run, packages(tmp))


def test_what_the_researcher_read_is_its_records_items(study):
    script, base, tmp, run, work = study
    complete(study)
    path = work / "review0" / "items.jsonl"
    rows = items(path)
    rows[0]["ensemble_present"] = not rows[0]["ensemble_present"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="review0"):
        S.verify_packages(work, run, packages(tmp))


def test_a_review_label_that_is_not_its_page_s_export_is_refused(study):
    """The researcher's export kept (a trace found) and the imported label changed: refused."""
    script, base, tmp, run, work = study
    trace = complete_with_trace(study)
    path = work / "review0" / "labels.json"
    record = json.loads(path.read_text())
    record["labels"][trace] = {"present": False, "note": None}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not its exports' transformation"):
        S.verify_packages(work, run, packages(tmp))


def test_an_ensemble_export_changed_after_its_import_is_refused(study):
    script, base, tmp, run, work = study
    complete(study)
    path = work / "round0" / "exports" / "annotators" / "fable.jsonl"  # a kept export changed
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["confident"] = not rows[0]["confident"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="round0: its"):
        S.verify_packages(work, run, packages(tmp))


def test_a_page_that_showed_other_items_is_refused_at_import(study):
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    label_round(script, base, tmp, 0)
    script.main([*base, "review"])
    mapping = json.loads((work / "review0" / "mapping.json").read_text())
    export = write_jsonl(tmp / "page.jsonl", [{"item_id": i, "present": False, "note": ""}
                                              for i in mapping["items"]])  # fmt: skip
    log = tmp / "page.log.json"
    log.write_text(json.dumps({"page_id": "p", "source": {"items_sha256": "0" * 64}}))
    with pytest.raises(SystemExit, match="did not show the review's items"):
        script.main([*base, "import-review", "--export", str(export), "--log", str(log)])


def test_a_stage3_round_is_imported_from_its_exports_alone(study):
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    path = write_jsonl(tmp / "l.jsonl", [])
    with pytest.raises(SystemExit, match="in two steps"):
        script.main([*base, "import", "--labels", str(path)])


def test_the_ensemble_rule():
    def answers(**kw):
        return {iid: {"item_id": iid, **v} for iid, v in kw.items()}

    def row(present, confident=True, evidence=""):
        return {"present": present, "confident": confident, "evidence": evidence}

    annotators = {
        "opus": answers(a=row(True, True, "x"), b=row(False, True, "ignored"), c=row(True),
                        d=row(True, True, "")),
        "fable": answers(a=row(True, False, "y"), b=row(False), c=row(False), d=row(True)),
        "gpt": answers(a=row(True), b=row(False), c=row(True), d=row(True)),
    }  # fmt: skip
    arbiter = answers(c=row(False, False, ""))
    labels_ = S.merge_ensemble(ENSEMBLE, annotators, arbiter)
    assert labels_["a"] == {"present": True, "unanimous": True, "confident": False,
                            "evidence": "x"}  # fmt: skip
    assert labels_["b"] == {"present": False, "unanimous": True, "confident": True,
                            "evidence": None}  # fmt: skip
    assert labels_["c"] == {"present": False, "unanimous": False, "confident": False,
                            "evidence": None}  # fmt: skip
    assert labels_["d"]["evidence"] is None  # a blank evidence is null
    with pytest.raises(SystemExit, match="exactly the disputed items"):
        S.merge_ensemble(ENSEMBLE, annotators, {**arbiter, **answers(a=row(True))})


# --- the round's kits ----------------------------------------------------------------------


def complete_with_dispute(study):
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    rows = items(work / "round0" / "items.jsonl")
    label_round(script, base, tmp, 0, dispute=rows[1]["item_id"])
    script.main([*base, "round"])
    label_round(script, base, tmp, 1)
    script.main([*base, "review"])
    import_review(script, base, tmp, work, 0, present={rows[1]["item_id"]})
    script.main([*base, "package"])
    return rows[1]["item_id"]


def test_the_kits_are_what_the_round_gives(study):
    """Every annotator reads the instructions and the parts the round's items give; the arbiter
    reads the disputed items alone, with the votes counted without names."""
    script, base, tmp, run, work = study
    disputed = complete_with_dispute(study)
    kit = work / "round0" / "kit"
    instructions = (kit / "annotators" / "instructions.md").read_text()
    assert "{FILES}" not in instructions and "`items_part1.md`" in instructions
    n = len(items(work / "round0" / "items.jsonl"))
    assert f"{n} items in total" in instructions
    arbiter = (kit / "arbiter" / "disputed_items.md").read_text()
    assert arbiter.startswith("# Disputed items (1)") and disputed in arbiter
    assert "Annotator votes: {'present': 2, 'absent': 1}" in arbiter
    for name in ENSEMBLE["annotators"]:
        assert name not in arbiter
    assert "`reason`" in (kit / "arbiter" / "instructions.md").read_text()
    assert S.verify_packages(work, run, packages(tmp))


@pytest.mark.parametrize("who", ["annotators", "arbiter"])
def test_a_kit_changed_after_it_was_issued_is_refused(study, who):
    script, base, tmp, run, work = study
    complete_with_dispute(study)
    path = work / "round0" / "kit" / who / "instructions.md"
    path.write_text(path.read_text() + "\nAn added sentence.\n")
    with pytest.raises(ValueError, match="not the kit its round's records give"):
        S.verify_packages(work, run, packages(tmp))


@pytest.mark.parametrize(("who", "change"), [
    ("annotators", {"runner": "another runner"}),
    ("annotators", {"model": "another-model"}),
    ("annotators", {"kit": {"instructions.md": "0" * 64}}),
    ("arbiter", {"votes": "named"}),
    ("arbiter", {"kit": {}}),
])  # fmt: skip
def test_an_export_not_made_on_the_plan_s_runner_and_kit_is_refused(study, who, change):
    """A manifest that names another runner, model, voting or kit than the plan's and the
    round's is refused, at the import and when the records are verified again."""
    script, base, tmp, run, work = study
    complete_with_dispute(study)
    path = work / "round0" / "exports" / f"{who}_manifest.json"
    data = json.loads(path.read_text())
    if who == "annotators":
        data["annotators"]["gpt"].update(change)
    else:
        data["arbiter"].update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest is not the plan's"):
        S.verify_packages(work, run, packages(tmp))


def test_an_import_on_another_runner_is_refused_at_once(study):
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    d = work / "round0"
    labels_ = {i["item_id"]: {"present": False} for i in items(d / "items.jsonl")}
    out = tmp / "exports"
    annotator_exports(out, labels_, d / "kit" / "annotators")
    path = out / "annotators_manifest.json"
    data = json.loads(path.read_text())
    data["annotators"]["opus"]["runner"] = "a shared context"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="manifest is not the plan's"):
        script.main([*base, "import-annotators", "--exports", str(out)])


def test_an_arbiter_for_an_undisputed_round_is_refused(study):
    script, base, tmp, run, work = study
    complete(study)
    out = work / "round0" / "exports"
    arbiter_exports(out, [], work / "round0" / "kit" / "annotators")
    with pytest.raises(ValueError, match="an arbiter for a round without a dispute"):
        S.verify_packages(work, run, packages(tmp))


def test_a_changed_annotators_kit_stops_the_import_before_the_arbiter_s_kit(study):
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    d = work / "round0"
    rows = items(d / "items.jsonl")
    labels_ = {
        i["item_id"]: {
            "present": i["item_id"] == rows[1]["item_id"],
            "unanimous": i["item_id"] != rows[1]["item_id"],
        }
        for i in rows
    }
    out = tmp / "exports"
    annotator_exports(out, labels_, d / "kit" / "annotators")
    path = d / "kit" / "annotators" / "instructions.md"
    path.write_text(path.read_text() + "\nAn added sentence.\n")
    with pytest.raises(SystemExit, match="not the kit its round's records give"):
        script.main([*base, "import-annotators", "--exports", str(out)])
    assert not (d / "kit" / "arbiter").exists()
