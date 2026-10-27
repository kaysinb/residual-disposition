"""The completion of a lineage's blocked strict labels (continuation_completion; §12.4.5, amendment
of 2026-10-27), on synthetic readings, a synthetic reference set and cached answers only: the
pinned cohort, the replacement instrument's one validation, each replacement slot's first valid
answer and one retry, the blind review's binding, the provenance written once and recomputed,
and the readers' counts with the zero-count treatment kept as a sensitivity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_stage3_verdicts as V

from residual_disposition import continuation_completion as C
from residual_disposition import continuation_panel as G
from residual_disposition import continuation_readers as E
from residual_disposition import manifests
from residual_disposition.judge import (
    Completion,
    JudgeItem,
    load_targets,
    stage_labels,
    stage_requests,
)

REPO = Path(__file__).resolve().parents[1]
D1 = E.DRAWS[1]
BLOCKED = (("L", D1, 0), ("L", D1, 1))  # two cohort requests, both on layer 30 of L's package
SET = "refset"


def strict(label: str) -> str:
    return json.dumps({"label": label, "confident": True})


def reference_set(runs: Path, n: int = 6) -> dict:
    """A frozen reference set: half its items naming the target, the frozen pair's lenient
    labels and three annotators agreeing with the reference."""
    s = runs / "continuation_reader_set" / "set-run"
    p = runs / "judge_calibration" / f"{SET}-reference"
    s.mkdir(parents=True)
    p.mkdir(parents=True)
    items = [{"item_id": f"i{i}", "response_id": f"ref-{i}", "target_behavior_id": "tldr",
              "stratum": "random"} for i in range(n)]  # fmt: skip
    ref = {f"i{i}": ("names_behavior", True) if i < n // 2 else ("other", False) for i in range(n)}
    (s / "mapping.json").write_text(json.dumps({"set_id": SET, "items": items}))
    lines = {
        s / "input.jsonl": [{"item_id": f"i{i}", "target": "a target", "response": f"ref text {i}"}
                            for i in range(n)],
        s / "frozen_pair_labels.jsonl": [{"item_id": k, "strict_label": v[0],
                                          "lenient_label": v[1], "uncertain": False}
                                         for k, v in ref.items()],
        p / f"labels-{SET}.jsonl": [{"set_id": SET, "item_id": k, "strict_label": v[0],
                                     "lenient_label": v[1], "labeled_at": "t"}
                                    for k, v in ref.items()],
        **{p / f"{a}.jsonl": [{"item_id": k, "strict_label": v[0], "lenient_label": v[1],
                               "confident": True} for k, v in ref.items()]
           for a in ("opus", "fable", "gpt")},
    }  # fmt: skip
    for path, rows in lines.items():
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    listing = p.parent / f"{SET}-reference.sha256"  # the annotators' files, by their listing
    listing.write_text("".join(f"{E.file_sha(p / f'{a}.jsonl')}  ./{a}.jsonl\n"
                               for a in ("opus", "fable", "gpt")))  # fmt: skip
    files = {f"runs/{f.relative_to(runs)}": E.file_sha(f)
             for f in (s / "mapping.json", s / "input.jsonl", s / "frozen_pair_labels.jsonl",
                       p / f"labels-{SET}.jsonl", listing)}  # fmt: skip
    return {"reference_set": SET, "layers": ["adl_lens@30"], "items": n, "strata": ["random"],
            "files": files, "annotators": ["opus", "fable", "gpt"]}  # fmt: skip


def instrument_spec() -> dict:
    rubric = E.judge_settings("v4").rubrics.strict
    return {"rubric": {"version": rubric.version, "sha256": rubric.sha256},
            "model": "gemini-3.8-flash", "reasoning_effort": "low", "service_tier": "standard",
            "backend": "sync", "structured_output": True, "max_tokens": 1024}  # fmt: skip


class Done:
    """A blocked world, its specification committed under ``root`` and its answers cached."""

    def __init__(self, world, root: Path, monkeypatch):
        self.world, self.root, self.monkeypatch = world, root, monkeypatch
        monkeypatch.setenv("RUNS_DIR", str(world.runs))
        monkeypatch.setattr(C, "spec_path", lambda r, lineage: root / C.SPEC.format(lineage))
        monkeypatch.setattr(C, "provenance_path",
                            lambda r, lineage: root / C.PROVENANCE.format(lineage))  # fmt: skip
        monkeypatch.setattr(manifests, "committed_json", lambda p, what, repo=None: {
            **json.loads(Path(p).read_text()), "sha256": E.file_sha(Path(p))})  # fmt: skip
        self.blocked = V._blocked_world(world, {k: (V.BLOCK, V.BLOCK) for k in BLOCKED})
        self.behavior = world.identity["behavior"]
        self.target = load_targets([self.behavior])[self.behavior]
        self.validation = reference_set(world.runs)

    def bound(self):
        b = E._bound_hypotheses(self.world.lineage, self.world.sources(), self.world.find_run)
        lens_l = E.labels(self.world.find_run, "lens-judge", "lens", b["lens_h"], self.behavior)
        return b["lens_h"], lens_l

    def spec(self, **change) -> dict:
        lens_h, lens_l = self.bound()
        cohort = {}
        for rid, (status, _, (st, le)) in sorted(lens_l.items()):
            if status != E.STRICT_SAFETY_BLOCKED:
                continue
            c = cohort.setdefault(st[0]["key"], {"response_ids": []})
            c["strict_attempts"] = [{"key": s["key"], "entry_sha256": s["sha256"],
                                     "tier": s["tier"], "finish_reason": s["finish_reason"]}
                                    for s in st]  # fmt: skip
            c["lenient_slot"] = [{"key": s["key"], "entry_sha256": s["sha256"], "tier": s["tier"]}
                                 for s in le]  # fmt: skip
            c["text_sha256"] = C.sha(lens_h[rid][2])
            c["response_ids"].append(rid)
        spec = {
            "lineage": self.world.identity["lineage"], "behavior": self.behavior,
            "target_sha256": C.sha(self.target),
            "sources": {"record": "record", "lens_reading": "lens-reading",
                        "lens_judge": "lens-judge", "judge_history_manifest_sha256": None},
            "cohort": {"layer": "adl_lens@30", "package": "lens-L", "unique_requests": len(cohort),
                       "references": len(self.blocked), "requests": cohort},
            "instrument": instrument_spec(), "validation": self.validation,
            "human": {"seed": 7},
        }  # fmt: skip
        strict_of = {rid: v[0] for rid, v in lens_l.items() if v[0] != E.STRICT_SAFETY_BLOCKED}
        frame = C.decoy_frame(spec, lens_h, strict_of)
        spec["human"] = {"seed": 7, "decoys": frame.pop("decoys"), "decoy_frame": frame}
        for k, v in change.items():
            spec[k] = v
        return spec

    def commit(self, spec=None, register=True) -> dict:
        """The specification committed and, unless told otherwise, registered as it is."""
        spec = spec or self.spec()
        path = C.spec_path(None, spec["lineage"])
        path.write_text(json.dumps(spec, indent=1))
        if not register:
            return spec
        self.monkeypatch.setitem(C.REGISTERED, spec["lineage"], {
            "behavior": spec["behavior"], "sha256": E.file_sha(path)})  # fmt: skip
        return C.committed_spec(spec["lineage"])

    def issue(self, spec, keys):
        """The review issued for ``keys``: its frozen page and private record."""
        from residual_disposition.judge import labeling_set_id

        items = C.review_items(spec, keys, self.texts(), self.target)
        page = C.page(labeling_set_id(items), items)
        record = C.issued(spec["lineage"], spec["sha256"], keys, items, page)
        work = C.work_dir(spec["lineage"])
        C.write_once(work / C.ISSUED, record)
        (work / C.PAGE).write_text(page)
        return record

    def export(self, record, path, label="names_behavior", ids=None, **row):
        ids = [e["item_id"] for e in record["items"]] if ids is None else ids
        path.write_text("".join(json.dumps({
            "set_id": record["set_id"], "item_id": i, "strict_label": label,
            "lenient_label": None, "labeled_at": "2026-10-27T10:00:00+00:00", **row}) + "\n"
            for i in ids))  # fmt: skip
        return path

    def answer(self, request, *texts, tier="standard"):
        """Cache ``texts`` as the request's attempts in order (its retry the second)."""
        for key, text in zip((request.key, C.retry(request).key), texts, strict=False):
            self.world.entry("judge", key, request.body, text, tier)

    def validate(self, spec, label=None):
        """Every reference item answered as the reference labels it (or ``label``)."""
        ins = C.settings(spec)
        for (rid, _), r in C.validation_requests(spec, ins).items():
            i = int(rid.split("-")[1])
            self.answer(r, strict(label or ("names_behavior" if i < 3 else "other")))

    def texts(self):
        lens_h, _ = self.bound()
        return {rid: h[2] for rid, h in lens_h.items()}

    def cohort(self, spec):
        return C.cohort(spec, self.texts(), self.target, C.settings(spec))

    def strict_of(self):
        _, lens_l = self.bound()
        return {rid: v[0] for rid, v in lens_l.items() if v[0] != E.STRICT_SAFETY_BLOCKED}

    def provenance(self, spec):
        cache = C.cache_of(spec, self.world.find_run)
        return C.write_provenance(self.root, spec, cache, self.texts(), self.target,
                                  self.strict_of())  # fmt: skip

    def grade(self, counted):
        """The category run, the completed references rule C counts graded too."""
        self.world.blocked -= set(counted)
        self.world.category()


@pytest.fixture
def done(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    world = V.World(tmp_path, monkeypatch)
    root = tmp_path / "completion"
    root.mkdir()
    return Done(world, root, monkeypatch)


def test_a_validated_replacement_completes_the_cohort_and_keeps_the_zero_count(done):
    spec = done.commit()
    done.validate(spec)
    for _, r in done.cohort(spec).values():
        done.answer(r, strict("names_behavior"))
    record = done.provenance(spec)
    assert record["validation"]["result"] == "pass" and record["routed"] == []
    assert record["review"] is None and len(record["completed"]) == 2
    text = json.dumps(record)
    assert "names_behavior" not in text and "hypothesis" not in text  # no label, no text
    done.grade(done.blocked)
    data = V.record(done.world)
    assert data["status"] == "complete"
    assert data["counts"]["lens"][f"L/{D1}"] == 70  # the two completed and counted
    found = data["strict_safety_blocked"]  # the 2026-10-26 treatment, a sensitivity now
    assert found["packages"][f"L/{D1}"] == {"count": 68, "blocked": 2, "blocked_lenient_yes": 2,
                                            "interval": ["17/20", "7/8"]}  # fmt: skip
    zero = data["zero_count_sensitivity"]  # the zero-count analysis, on the frozen functions
    assert zero["counts"]["lens"][f"L/{D1}"] == 68
    assert {"p3", "p4", "component3", "component4", "category_margins"} <= set(zero)
    assert data["blocked_completion"] == {
        "spec": spec["sha256"], "provenance": E.file_sha(C.provenance_path(None, spec["lineage"])),
        "validation": "pass", "requests": 2, "references": 2, "reviewed": 0}  # fmt: skip
    values = G.reader_values(data, {}, done.world.identity)  # the panel reads it unchanged
    assert values["x_valid"]


def test_a_completed_label_that_avoids_the_behavior_is_not_counted_or_graded(done):
    spec = done.commit()
    done.validate(spec)
    for _, r in done.cohort(spec).values():
        done.answer(r, strict("avoids_behavior"))
    done.provenance(spec)
    done.world.category()  # nothing of the cohort graded
    data = V.record(done.world)
    assert data["counts"]["lens"][f"L/{D1}"] == 68
    done.grade(done.blocked)  # a grade for one not counted is refused
    with pytest.raises(ValueError, match="not one counted hypothesis"):
        V.record(done.world)


def test_without_its_validation_or_provenance_the_cohort_is_pending(done):
    spec = done.commit()
    with pytest.raises(G.Pending, match="provenance is not committed"):
        V.record(done.world)
    with pytest.raises(G.Pending, match="validation"):
        done.provenance(spec)
    done.validate(spec)
    with pytest.raises(G.Pending, match="replacement slots have no answer"):
        done.provenance(spec)


def test_the_validation_reads_the_pinned_set_with_the_instrument(done):
    spec = done.commit()
    ins = C.settings(spec)
    assert (
        ins.model.id,
        ins.model.reasoning_effort,
        ins.max_tokens,
        ins.structured_output,
        ins.rubrics.strict.version,
    ) == ("gemini-3.8-flash", "low", 1024, True, "strict-v4")
    (r,) = [r for (rid, _), r in C.validation_requests(spec, ins).items() if rid == "ref-0"]
    assert r.body["model"] == "gemini-3.8-flash" and r.body["reasoning_effort"] == "low"
    assert "response_format" in r.body and not ins.lenient  # no lenient request is built
    with pytest.raises(ValueError, match="not strict-v4 under the frozen judge"):
        C.settings({**spec, "instrument": {**spec["instrument"], "reasoning_effort": "high"}})
    gpt = done.world.runs / "judge_calibration" / f"{SET}-reference" / "gpt.jsonl"
    gpt.write_text(gpt.read_text() + "\n")
    with pytest.raises(ValueError, match="not an annotator's pinned labels"):
        C.validation_requests(spec, ins)
    (done.world.runs / "continuation_reader_set" / "set-run" / "input.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="not the reference set's pinned file"):
        C.validation_requests(spec, ins)


def test_a_failed_validation_sends_the_whole_cohort_to_the_blind_review(done, tmp_path):
    spec = done.commit()
    done.validate(spec, label="avoids_behavior")  # C never positive: criterion 1 fails
    cache = C.cache_of(spec, done.world.find_run)
    found = C.validation(spec, cache, C.settings(spec))
    assert found["result"] == "fail" and found["criterion_1"] == "fail"
    for _, r in done.cohort(spec).values():  # a replacement answer is never read then
        done.answer(r, strict("other"))
    keys = C.routed(found, {k: {"status": "answered"} for k in spec["cohort"]["requests"]})
    assert keys == sorted(spec["cohort"]["requests"])
    record = done.issue(spec, keys)
    assert len(record["items"]) == 2 + 30
    C.import_review(spec["lineage"], done.export(record, tmp_path / "labels.jsonl"))
    written = done.provenance(spec)
    assert written["routed"] == keys and written["review"]["items"] == 32
    assert written["review"]["decoy_agreement"]["decoys"] == 30
    assert {c["source"] for c in written["completed"].values()} == {"review"}
    done.grade(done.blocked)
    data = V.record(done.world)
    assert data["counts"]["lens"][f"L/{D1}"] == 70
    assert data["blocked_completion"]["validation"] == "fail"
    assert data["blocked_completion"]["reviewed"] == 2


def test_each_replacement_slot_keeps_its_first_valid_answer_after_one_retry(done):
    spec = done.commit()
    done.validate(spec)
    cache = C.cache_of(spec, done.world.find_run)
    (k1, (_, r1)), (k2, (_, r2)) = sorted(done.cohort(spec).items())
    assert C.due(cache, [r1, r2]) == [r1, r2]  # nothing answered: both sent
    done.answer(r1, "not json")  # refused: its retry is due, once
    assert [r.key for r in C.due(cache, [r1])] == [C.retry(r1).key]
    done.answer(r1, "not json", strict("other"))
    done.answer(r2, strict("names_behavior"), strict("avoids_behavior"))  # never replaced
    assert C.due(cache, [r1, r2]) == []
    found = C.replacements(cache, done.cohort(spec))
    assert (found[k1]["value"], len(found[k1]["history"])) == ("other", 2)
    assert (found[k2]["value"], len(found[k2]["history"])) == ("names_behavior", 1)


def test_a_slot_refused_twice_is_routed_to_the_review_and_an_open_one_pends(done):
    spec = done.commit()
    done.validate(spec)
    cache = C.cache_of(spec, done.world.find_run)
    (k1, (_, r1)), (k2, (_, r2)) = sorted(done.cohort(spec).items())
    done.answer(r1, "not json", "")
    found = C.validation(spec, cache, C.settings(spec))
    with pytest.raises(G.Pending, match="1 replacement slots"):
        C.routed(found, C.replacements(cache, done.cohort(spec)))
    done.answer(r2, strict("other"))
    assert C.routed(found, C.replacements(cache, done.cohort(spec))) == [k1]
    with pytest.raises(G.Pending, match="blind review is not imported"):
        done.provenance(spec)


def test_an_answer_off_the_standard_tier_or_to_another_request_is_refused(done):
    spec = done.commit()
    cache = C.cache_of(spec, done.world.find_run)
    (_, (_, r1)), (_, (_, r2)) = sorted(done.cohort(spec).items())
    done.answer(r1, strict("other"), tier="flex")
    with pytest.raises(ValueError, match="not on the standard tier"):
        C.outcome(cache, r1, "r1")
    done.world.entry("judge", r2.key, {"m": 1}, strict("other"), "standard")
    with pytest.raises(ValueError, match="not to its request"):
        C.outcome(cache, r2, "r2")


def test_the_review_is_bound_to_its_issue_and_imported_once(done, tmp_path):
    spec = done.commit()
    keys = sorted(spec["cohort"]["requests"])[:1]
    texts = done.texts()
    record = done.issue(spec, keys)
    work = C.work_dir(spec["lineage"])
    page = (work / C.PAGE).read_text()
    assert "lenient-ia" not in page and "same category" not in page.lower()  # strict only
    assert page.count('type="radio"') == 3 * len(record["items"])
    items = C.review_items(spec, keys, texts, done.target)
    for li in items:  # the page holds ids, targets and texts, never provenance
        assert li.item.response_id not in page and li.item_id in page
    with pytest.raises(ValueError, match="written once"):
        C.write_once(work / C.ISSUED, record)
    ids = [e["item_id"] for e in record["items"]]
    with pytest.raises(ValueError, match="exactly the issued items"):
        C.import_review(spec["lineage"], done.export(record, tmp_path / "a.jsonl", ids=ids[:-1]))
    with pytest.raises(ValueError, match="invalid strict label"):
        C.import_review(spec["lineage"], done.export(record, tmp_path / "b.jsonl", "maybe"))
    with pytest.raises(ValueError, match="holds a lenient label"):
        C.import_review(spec["lineage"], done.export(record, tmp_path / "c.jsonl",
                                                     lenient_label=True))  # fmt: skip
    C.import_review(spec["lineage"], done.export(record, tmp_path / "d.jsonl", "other"))
    with pytest.raises(ValueError, match="imported once"):
        C.import_review(spec["lineage"], done.export(record, tmp_path / "e.jsonl", "other"))
    found = C.review(spec["lineage"], spec, keys, texts, done.target, done.strict_of())
    assert found["labels"] == {keys[0]: "other"}
    step = found["steps"][keys[0]]
    assert (step["key"], step["set_id"], step["parsed"]) == (None, record["set_id"], True)
    assert step["review"] == E.file_sha(work / C.EXPORT)
    assert step["import_sha256"] == E.file_sha(work / C.HUMAN)
    assert found["record"]["decoy_agreement"]["decoys"] == 30
    with pytest.raises(ValueError, match="not the specification's for its routed requests"):
        C.review(spec["lineage"], spec, sorted(spec["cohort"]["requests"]), texts, done.target,
                 done.strict_of())  # fmt: skip

    def refused(path, edit, match):
        before = path.read_bytes()
        edit(path)
        with pytest.raises(ValueError, match=match):
            C.review(spec["lineage"], spec, keys, texts, done.target, done.strict_of())
        path.write_bytes(before)

    human = work / C.HUMAN
    refused(human, lambda p: p.write_text(json.dumps({  # a changed label in the import
        **json.loads(p.read_text()), "labels": {i: "names_behavior" for i in ids}})),
        "not the kept export's")  # fmt: skip
    refused(work / C.EXPORT, lambda p: p.write_text(p.read_text().replace('"other"',
            '"names_behavior"')), "not the kept export of the issued page")  # fmt: skip
    refused(work / C.PAGE, lambda p: p.write_text(p.read_text() + " "), "frozen page")
    refused(work / C.ISSUED, lambda p: p.write_text(json.dumps({
        **json.loads(p.read_text()), "page_sha256": "0" * 64})), "frozen page")  # fmt: skip
    (work / C.EXPORT).unlink()
    with pytest.raises(G.Pending, match="not imported"):
        C.review(spec["lineage"], spec, keys, texts, done.target, done.strict_of())


def test_a_registered_completion_is_never_skipped(done, monkeypatch):
    """A registered specification missing, changed or unregistered is refused, the readers and
    the categories alike."""
    spec = done.spec()
    done.commit(spec, register=False)
    with pytest.raises(ValueError, match="not registered"):
        V.record(done.world)
    path = C.spec_path(None, spec["lineage"])
    monkeypatch.setitem(C.REGISTERED, spec["lineage"], {"behavior": done.behavior,
                                                        "sha256": E.file_sha(path)})  # fmt: skip
    path.unlink()
    with pytest.raises(ValueError, match="specification is missing"):
        V.record(done.world)
    with pytest.raises(ValueError, match="specification is missing"):
        C.spec_of_behavior(done.behavior, "lens-reading")
    path.write_text(json.dumps({**spec, "human": {**spec["human"], "seed": 9}}))
    with pytest.raises(ValueError, match="not the registered completion"):
        V.record(done.world)


def test_the_repository_registers_eli5_s_committed_specification():
    eli5 = REPO / "manifests" / C.SPEC.format("strong_eli5_cont_200")
    registered = {"strong_eli5_cont_200": {"behavior": "eli5", "sha256": E.file_sha(eli5)}}
    assert registered == C.REGISTERED


def test_the_cohort_is_exactly_the_blocked_requests_with_their_pinned_histories(done):
    spec = done.spec()
    key = sorted(spec["cohort"]["requests"])[0]
    short = json.loads(json.dumps(spec))
    del short["cohort"]["requests"][key]
    done.commit(short)
    with pytest.raises(ValueError, match="not the lens judging's blocked requests"):
        V.record(done.world)
    changed = json.loads(json.dumps(spec))
    changed["cohort"]["requests"][key]["strict_attempts"][1]["entry_sha256"] = "0" * 64
    done.commit(changed)
    with pytest.raises(ValueError, match="not the one the specification pins"):
        V.record(done.world)
    with pytest.raises(ValueError, match="is not the entry the specification pins"):
        C.originals(C.committed_spec(spec["lineage"]), C.cache_of(spec, done.world.find_run))
    done.commit({**spec, "sources": {**spec["sources"], "lens_judge": "pd-judge"}})
    with pytest.raises(ValueError, match="of other readings"):
        V.record(done.world)
    done.commit({**spec, "human": {**spec["human"], "seed": 8}})
    with pytest.raises(ValueError, match="seeded draw"):
        V.record(done.world)


def test_a_changed_completion_is_not_its_recomputation(done):
    spec = done.commit()
    done.validate(spec)
    cohort = done.cohort(spec)
    for _, r in cohort.values():
        done.answer(r, strict("names_behavior"))
    done.provenance(spec)
    with pytest.raises(ValueError, match="written once"):
        done.provenance(spec)
    done.grade(done.blocked)
    key = sorted(cohort)[0]
    path = C.provenance_path(None, spec["lineage"])
    data = json.loads(path.read_text())
    data["routed"] = [key]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not its recomputation"):
        V.record(done.world)


def test_one_completed_label_serves_every_reference_of_its_key(monkeypatch):
    hist = [{"key": "k", "sha256": "s", "tier": "standard", "finish_reason": "f"}]
    len_h = [{"key": "l", "sha256": "t", "tier": "standard"}]
    lens_l = {r: (E.STRICT_SAFETY_BLOCKED, True, [hist, len_h]) for r in ("a/p/s/x/1", "b/p/s/x/2")}
    target = load_targets(["tldr"])["tldr"]
    spec = {"lineage": "l", "behavior": "tldr", "target_sha256": C.sha(target),
            "sources": {"record": "r", "lens_reading": "lr", "lens_judge": "lj",
                        "judge_history_manifest_sha256": None},
            "cohort": {"requests": {"k": {
                "response_ids": sorted(lens_l), "text_sha256": "x",
                "strict_attempts": [{"key": "k", "entry_sha256": "s", "tier": "standard",
                                     "finish_reason": "f"}],
                "lenient_slot": [{"key": "l", "entry_sha256": "t", "tier": "standard"}]}}},
            "human": {"decoy_frame": {"distinct_texts": 0, "references": 0, "sha256": "h"},
                      "decoys": []}}  # fmt: skip
    monkeypatch.setattr(C, "committed_spec", lambda lineage: spec)
    monkeypatch.setattr(C, "decoy_frame", lambda *a: {"distinct_texts": 0, "references": 0,
                                                      "sha256": "h", "decoys": []})  # fmt: skip
    monkeypatch.setattr(C, "cache_of", lambda *a: None)
    monkeypatch.setattr(C, "completion", lambda *a: {
        "values": {"k": ("names_behavior", [{"key": "n"}])}, "routed": [], "spec": "a",
        "provenance": "b", "validation": "pass"})  # fmt: skip
    lineage = SimpleNamespace(identity={"lineage": "l", "behavior": "tldr"})
    sources = {"record": "r", "lens_reading": "lr", "lens_judge": "lj"}
    found = C.completed_labels(lineage, sources, None, {r: (0, 0, "t") for r in lens_l}, lens_l,
                               None)  # fmt: skip
    assert {v[0] for v in found["labels"].values()} == {"names_behavior"}
    assert found["record"]["references"] == 2 and found["record"]["requests"] == 1
    assert all(v[2][:2] == [hist, len_h] for v in found["labels"].values())  # originals kept


def test_rule_c_s_validation_vetoes_and_missing_labels():
    from residual_disposition.judge import c_validation

    keys = [(f"r{i}", "t") for i in range(6)]
    final = {k: ("names_behavior", True) if i < 3 else ("other", False)
             for i, k in enumerate(keys)}  # fmt: skip
    votes = {a: dict(final) for a in ("x", "y", "z")}
    found = c_validation(dict(final), final, votes, keys)
    assert (found["criterion_1"], found["criterion_2"]) == ("pass", "not applicable")
    vetoed = {**final, keys[3]: ("avoids_behavior", True)}  # a veto of a reference negative
    assert c_validation(vetoed, final, votes, keys)["criterion_2"] == "pass"
    wrong = {**final, keys[0]: ("avoids_behavior", True)}  # a veto of a reference positive
    assert c_validation(wrong, final, votes, keys)["criterion_2"] == "fail"
    missing = {k: v for k, v in final.items() if k != keys[0]}
    assert c_validation(missing, final, votes, keys)["criterion_1"] == "fail"


def test_the_request_log_records_each_request_s_own_model():
    settings = E.judge_settings("v4")
    (pair,) = stage_requests([JudgeItem("r", "tldr", "a target", "a text")], settings)
    answers = {pair.strict.key: Completion(strict("other")), pair.lenient.key: Completion("yes")}
    log = stage_labels([pair], answers, stage="primary", settings=settings).log
    assert {r["kind"]: r["model"] for r in log} == {"strict": "gemini-3.5-flash-lite",
                                                    "lenient": "gemini-3.8-flash"}  # fmt: skip


def _script(name):
    loader = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


class FakeBackend:
    """Answers by request key; ``fail`` keys answer with a transport error (not cached)."""

    tier = "standard"

    def __init__(self, answers, fail=()):
        self.answers, self.fail, self.sent = answers, set(fail), []

    def reconcile(self, cache):
        return 0

    def complete(self, requests):
        for r in requests:
            self.sent.append(r.key)
            if r.key in self.fail:
                yield r.key, Completion(None, error="timeout")
            else:
                yield r.key, Completion(self.answers.get(r.key, strict("other")),
                                        service_tier="standard")  # fmt: skip


def test_every_send_is_reserved_logged_and_resumed(done, monkeypatch, capsys):
    from residual_disposition import stage3_money

    spec = done.commit()
    script = _script("stage3_completion")
    monkeypatch.setattr(script, "find_run", done.world.find_run)
    ins = C.settings(spec)
    requests = C.validation_requests(spec, ins)
    first = requests[("ref-0", "tldr")]
    answers = {first.key: "not json"}  # refused: its retry sent in the same run
    answers |= {r.key: strict("names_behavior" if int(k[0].split("-")[1]) < 3 else "other")
                for k, r in requests.items() if r is not first}  # fmt: skip
    answers[C.retry(first).key] = strict("names_behavior")
    calls = []
    ledger = SimpleNamespace(check=lambda usd: calls.append(("check", usd)), source=None,
                             close=lambda: calls.append(("close",)) or {"s": 0.01})  # fmt: skip
    monkeypatch.setattr(stage3_money, "money", lambda root, behavior, log: (None, ledger))
    labels = []
    backend = FakeBackend(answers, fail={requests[("ref-5", "tldr")].key})
    cats = SimpleNamespace(backend=lambda led, model, label: labels.append(label) or backend)
    monkeypatch.setattr(script, "script", lambda name: cats)
    assert script.main(["validate", spec["lineage"], "--send"]) == 0
    out = capsys.readouterr().out
    assert "1 sends failed" in out and "validation open" in out
    assert labels == ["judge/completion-validation"] and calls[0][0] == "check"
    assert ("close",) in calls
    backend.fail.clear()
    sent = len(backend.sent)
    assert script.main(["validate", spec["lineage"], "--send"]) == 0
    assert len(backend.sent) == sent + 1  # only the failed send again
    assert '"result": "pass"' in capsys.readouterr().out
    rows = [json.loads(x) for x in (C.work_dir(spec["lineage"]) / C.LOG).read_text().splitlines()]
    assert {r["model"] for r in rows} == {"gemini-3.8-flash"}
    assert sum(r["request_id"] == C.retry(first).key for r in rows) == 1


def test_the_category_script_grades_the_counted_completions_sealed(done, monkeypatch, capsys):
    from residual_disposition import stage3_money

    spec = done.commit()
    done.validate(spec)
    (k1, (_, r1)), (k2, (_, r2)) = sorted(done.cohort(spec).items())
    done.answer(r1, strict("names_behavior"))
    done.answer(r2, strict("avoids_behavior"))
    cats = _script("continuation_categories")
    monkeypatch.setattr(cats, "find_run", done.world.find_run)
    texts = cats.counted_texts(done.behavior, [("lens-reading", "lens-judge")])
    extra = cats.completed_texts(C.committed_spec(spec["lineage"]), done.behavior,
                                 "lens-reading", "lens-judge")  # fmt: skip
    assert sorted(extra) == spec["cohort"]["requests"][k1]["response_ids"]
    assert not set(extra) & set(texts)
    ledger = SimpleNamespace(check=lambda usd: None, source=None, close=lambda: {})
    monkeypatch.setattr(stage3_money, "money", lambda root, behavior, log: (None, ledger))
    monkeypatch.setattr(cats, "backend", lambda ledger, model: FakeBackend({}))
    cats.main([done.behavior, "--lens-reading", "lens-reading", "--lens-judge", "lens-judge",
               "--pd-reading", "pd-reading", "--pd-judge", "pd-judge"])  # fmt: skip
    out = capsys.readouterr().out
    assert "sealed" in out and not any(ch.isdigit() for ch in out.split("sealed")[1][:60])
