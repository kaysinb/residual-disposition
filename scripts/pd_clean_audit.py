"""The PD-clean presentation audit (§13.12) on saved records only: Stage 1's 40 packages and
Stage 3's 32 (word_count_claim 200, eli5 continued 200, tldr 400, countdown 50), their readings
and judge labels (``residual_disposition.pd_clean_audit``).

    uv run python scripts/pd_clean_audit.py sample [--write]   # the targeted judge sample
    uv run python scripts/pd_clean_audit.py audit              # data/pd_clean_audit/audit.json

``sample --write`` freezes ``manifests/pd_clean_audit_sample.json``, committed before ``audit``
reads anything, and the blinded kit ``data/pd_clean_audit/kit`` (the v4-j3 rubrics verbatim,
target and hypothesis only), the ids' mapping beside it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pd_clean as PC
import yaml

from residual_disposition import pd_clean as C
from residual_disposition import pd_clean_audit as PA
from residual_disposition.judge import JudgeCache, request_key
from residual_disposition.paths import REPO_ROOT, data_root, runs_root
from residual_disposition.pd_clean_study import STAGE3_PLAN
from residual_disposition.summarizer import PROMPTS, pd_evidence

# study → (readings run, judge run, interpreter cache, study's records, plan)
STUDIES = {
    "stage1": ("summarize_pd_clean/20261017-000011-ed0bc775-a7b7",
               "judge/20261019-085501-08d9cf5c-00be", "summarizer-pd-clean", "pd_clean",
               REPO_ROOT / "configs" / "continuation" / "pd_clean.yaml"),
    "stage3_word_count_claim_200": ("summarize_stage3_pd/20261026-180500-4c655d8d-0242",
                                    "judge/20261026-181223-9b7af8fd-71fa", "summarizer",
                                    "pd_clean_stage3/stage3_word_count_claim_200", STAGE3_PLAN),
    "strong_eli5_cont_200": ("summarize_stage3_pd/20261026-184107-e036dd69-b0c3",
                             "judge/20261026-184522-7d72c48f-bbe8", "summarizer",
                             "pd_clean_stage3/strong_eli5_cont_200", STAGE3_PLAN),
    "stage3_tldr_400": ("summarize_stage3_pd/20261026-191507-2c7968c3-a488",
                        "judge/20261026-192601-a423b4f8-1f3d", "summarizer",
                        "pd_clean_stage3/stage3_tldr_400", STAGE3_PLAN),
    "stage3_countdown_50": ("summarize_stage3_pd/20261027-060015-1f453a0e-2a7e",
                            "judge/20261027-065523-1cc6d2c9-78fe", "summarizer",
                            "pd_clean_stage3/stage3_countdown_50", STAGE3_PLAN),
}  # fmt: skip
SAMPLE = REPO_ROOT / "manifests" / "pd_clean_audit_sample.json"
SEED, QUOTA = 20261104, 4


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def packages(readings: str) -> dict[str, dict]:
    """A readings run's packages by run name: params, kind and the pool's generation limit."""
    out = {}
    for row in jsonl(runs_root() / readings / "evidence.jsonl"):
        run = runs_root() / "pd_clean" / row["tool_run_id"]
        params = yaml.safe_load((run / "config.yaml").read_text())["params"]
        pool = yaml.safe_load((Path(params["pd_run"]) / "config.yaml").read_text())
        rnd = "pd_random_package" in run.name
        out[run.name] = {"run": run, "params": params, "random": rnd,
                         "kind": params["role"] + ("_random" if rnd else ""),
                         "limit": int(pool["params"]["settings"]["max_new_tokens"])}  # fmt: skip
    return out


def sample(write: bool) -> int:
    rows, targets = [], {}
    for name, (readings, judge, *_) in STUDIES.items():
        pk = packages(readings)
        said = jsonl(runs_root() / readings / "responses" / "tool_hypotheses.jsonl")
        text = {r["response_id"]: r["text"] for r in said}
        inputs = json.loads((runs_root() / judge / "judge_inputs.json").read_text())
        targets.update(inputs["targets"])
        for label in jsonl(runs_root() / judge / "observations" / "judge_labels.jsonl"):
            p = pk[label["response_id"].split("/")[1]]
            rows.append({"response_id": label["response_id"],
                         "behavior": label["target_behavior_id"],
                         "kind": f"{'stage1' if name == 'stage1' else 'stage3'}:{p['kind']}",
                         "text": text[label["response_id"]],
                         "label": label})  # fmt: skip
    chosen = PA.targeted_sample(rows, SEED, QUOTA)
    print(len(chosen), "strata,", sum(map(len, chosen.values())), "distinct texts")
    if not write:
        return 0
    if SAMPLE.exists():
        raise SystemExit(f"{SAMPLE.name} exists: frozen once")
    from residual_disposition.judge import RUBRIC_SETS

    items, mapping = PA.blinded(chosen, targets, SEED)
    SAMPLE.write_text(json.dumps({"seed": SEED, "quota": QUOTA, "studies": list(STUDIES),
                                  "strata": chosen}, indent=1) + "\n")  # fmt: skip
    kit = data_root() / "pd_clean_audit" / "kit"
    kit.mkdir(parents=True, exist_ok=False)
    rs = RUBRIC_SETS["v4-j3"]
    (kit / "rubrics.md").write_text("".join(f"## {r.kind} ({r.version})\n\n{r.text}\n\n"
                                            for r in (rs.strict, rs.lenient)))  # fmt: skip
    lines = [json.dumps(i, ensure_ascii=False) + "\n" for i in items]
    (kit / "items.jsonl").write_text("".join(lines))
    (kit.parent / "mapping.json").write_text(json.dumps(mapping, indent=1) + "\n")
    print(f"wrote {SAMPLE.name} and {kit}: commit the sample before the audit")
    return 0


def audit() -> int:
    from residual_disposition.manifests import committed_json

    committed_json(SAMPLE, "the audit's judge sample")  # frozen before any result is read
    out = {}
    for name, (readings, _judge, cache_dir, work, plan) in STUDIES.items():
        pk = packages(readings)
        record = None
        if plan == STAGE3_PLAN:
            first = next(iter(pk.values()))["params"]["binding"]["record"]["run"]
            record = runs_root() / "continuation_14b" / first
        st = PC.Study(data_root() / work, plan, record)
        final = st.final()
        settings = json.loads((runs_root() / readings / "summarizer_inputs.json").read_text())
        cache = JudgeCache(data_root() / cache_dir)
        slots = {}
        for o in jsonl(runs_root() / readings / "observations" / "tool_hypotheses.jsonl"):
            slots.setdefault(o["tool_run_id"], []).append(o)
        for run_name, p in sorted(pk.items()):
            saved = json.loads((p["run"] / "pd_clean.json").read_text())
            s = next(x for x in st.states if x["state"] == p["params"]["state"]
                     and x["role"] == p["params"]["role"])  # fmt: skip
            sel = st.selection(s, final)
            want = C.random_window(st.rows[s["state"]], sel, seed=st.seed) if p["random"] else sel
            checks, retries = [], []
            for o in sorted(slots.get(run_name, []), key=lambda x: x["sample"]):
                prompt = PROMPTS[o["prompt_version"]]
                body = PA.slot_body(settings, prompt.text, pd_evidence(saved["top"]))
                key = request_key(f"{o['prompt_version']}#sample={o['sample']}", body)
                checks.append(PA.slot_check(cache.entry(key), body))
                retry = cache.entry(request_key(f"{o['prompt_version']}#sample={o['sample']}"
                                                "#retry=1", body))  # fmt: skip
                if retry is not None:  # the reply that was read after a parser refusal
                    retries.append(PA.slot_check(retry, body))
            out[f"{name}/{run_name}"] = {
                "kind": p["kind"], "behavior": p["params"]["behavior"],
                "selection": PA.order_problems(saved, want.top, want.removed),
                "generation": PA.generation(saved, p["limit"]), "limit": p["limit"],
                "removals": PA.removals(saved),
                "slots": {"n": len(checks),
                          "status": dict(Counter(c["status"] for c in checks)),
                          "differs": sorted({d for c in checks for d in c.get("differs", [])}),
                          "finish": dict(Counter(str(c.get("finish_reason"))
                                                 for c in checks))},
                "retries": {"n": len(retries), "status": dict(Counter(c["status"] for c in retries)),
                            "differs": sorted({d for c in retries for d in c.get("differs", [])})}}  # fmt: skip
    d = data_root() / "pd_clean_audit"
    d.mkdir(exist_ok=True)
    (d / "audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(out), "packages")
    return 0


if __name__ == "__main__":
    sys.exit(sample("--write" in sys.argv) if sys.argv[1] == "sample" else audit())
