"""The PD-clean presentation audit (§13.12) of Stage 1's 40 packages, on saved records only.

    uv run python scripts/pd_clean_audit.py sample [--write]   # the blinded judge sample
    uv run python scripts/pd_clean_audit.py audit              # data/pd_clean_audit/audit.json

``sample --write`` freezes ``manifests/pd_clean_audit_sample.json`` (committed before ``audit``)
and the blinded kit ``data/pd_clean_audit/kit`` (the v4-j3 rubrics verbatim, target and
hypothesis only), the ids' mapping beside it.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import yaml

from residual_disposition import pd_clean_audit as PA
from residual_disposition.paths import REPO_ROOT, data_root, runs_root

READINGS = "summarize_pd_clean/20261017-000011-ed0bc775-a7b7"
JUDGE = "judge/20261019-085501-08d9cf5c-00be"
CACHE = "cache/summarizer-pd-clean"
SAMPLE = REPO_ROOT / "manifests" / "pd_clean_audit_sample.json"
SEED, QUOTA, SIZE = 20261104, 8, 92


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def packages() -> dict[str, dict]:
    """Stage 1's packages by run: params, and whether it is a random control."""
    out = {}
    for row in jsonl(runs_root() / READINGS / "evidence.jsonl"):
        run = runs_root() / "pd_clean" / row["tool_run_id"]
        params = yaml.safe_load((run / "config.yaml").read_text())["params"]
        random_ = "pd_random_package" in run.name
        out[run.name] = {"run": run, "params": params, "text": row["text"], "random": random_,
                         "kind": params["role"] + ("_random" if random_ else "")}  # fmt: skip
    return out


def sample(write: bool) -> int:
    pk = packages()
    labels = jsonl(runs_root() / JUDGE / "observations" / "judge_labels.jsonl")
    chosen = PA.judge_sample(labels, {k: v["kind"] for k, v in pk.items()}, SEED, QUOTA)
    print({k: len(v) for k, v in chosen.items()}, sum(map(len, chosen.values())))
    if not write:
        return 0
    if SAMPLE.exists():
        raise SystemExit(f"{SAMPLE.name} exists: frozen once")
    from residual_disposition.judge import RUBRIC_SETS

    rows = jsonl(runs_root() / READINGS / "responses" / "tool_hypotheses.jsonl")
    hyp = {r["response_id"]: r["text"] for r in rows}
    targets = json.loads((runs_root() / JUDGE / "judge_inputs.json").read_text())["targets"]
    behavior = {r["response_id"]: r["target_behavior_id"] for r in labels}
    items, mapping = PA.sample_items(chosen, hyp, targets, behavior, SEED)
    SAMPLE.write_text(json.dumps({"seed": SEED, "quota": QUOTA, "judge_run": JUDGE,
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
    pk = packages()
    entries = [json.loads(f.read_text()) for f in (data_root() / CACHE).glob("*/*.json")]
    out = {}
    for name, p in sorted(pk.items()):
        package = json.loads((p["run"] / "pd_clean.json").read_text())
        source = {}
        with gzip.open(Path(p["params"]["pd_run"]) / "pd_completions.jsonl.gz", "rt") as fh:
            for line in fh:
                row = json.loads(line)
                source[(row["config"], row["prefill"])] = row
        out[name] = {"kind": p["kind"], "behavior": p["params"]["behavior"],
                     "problems": PA.package_problems(package, source, p["text"], SIZE,
                                                     ranked=not p["random"]),
                     "generation": PA.generation(package), "removals": PA.removals(package),
                     "sent": PA.sent(entries, p["text"])}  # fmt: skip
    d = data_root() / "pd_clean_audit"
    d.mkdir(exist_ok=True)
    (d / "audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(out), "packages;", sum(bool(v["problems"]) for v in out.values()), "with problems")
    return 0


if __name__ == "__main__":
    command = sys.argv[1]
    sys.exit(sample("--write" in sys.argv) if command == "sample" else audit())
