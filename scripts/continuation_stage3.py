"""Stage 3's laptop records of a panel behavior (preregistration §12.7 Stage 3, §12.3.1; C17):
the detector diagnostic's sources (E4), from the cell phase s3_suppress made
(`stage3_diagnostic_sources.json`), recomputed from the runs it names and written once as
`manifests/stage3_diagnostic_sources_<behavior>.json`; with no margin-passing stack checkpoint,
from the continued screening's pools once its manifest `strong_<behavior>_cont.json` is committed
(E4). Commit it before any labeling: the diagnostic's set is drawn from these sources
(`continuation_labels.py detector-diagnostic --behavior <behavior>`), and a detector revision
never replaces them. After the ensemble's labels,
the arbiter and the researcher's review (§12.9), `diagnostic` writes the result,
`manifests/stage3_diagnostic_<behavior>.json`, recomputed from the set (drawn again from the
committed sources), the ensemble's labels and the review (every dispute, every unanimous trace,
the seeded sample of agreed negatives) and what the annotators saw (the committed target and
the text alone), under the set's own detector; commit it before s3b. A trace is cleared only
by a committed acceptance of a detector revision,
`manifests/detector_revision_<behavior>_<version>.json`, whose §12.3.1 checks the barrier
recomputes on the frozen development data (`continuation_stage3.revision_problems`). Once the
stack decision and the continued screening are committed, `adl-sources` freezes the ADL-corpus
diagnostic's sources (`manifests/stage3_diagnostic_sources_adl_<behavior>.json`: the ADL-corpus
runs of both, each once), before its own set is drawn (`--kinds adl_corpus`); `diagnostic` on
that set writes `manifests/stage3_diagnostic_adl_<behavior>.json`, needed before the strong
validation.

    uv run python scripts/continuation_stage3.py sources runs/continuation_14b/<sources run>
    uv run python scripts/continuation_stage3.py adl-sources <behavior>
    uv run python scripts/continuation_stage3.py diagnostic <set run> --ensemble <labels> \
        --review <review>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage3 as S3
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def diagnostic(args: argparse.Namespace, rules) -> int:
    """The diagnostic's result: written under the set's own detector (before any revision,
    whose acceptance is a record of its own), the set drawn again from the committed sources
    (``continuation_labels.stage3_chosen``), recomputed and written once with the sha256 of
    the set's input and key, the ensemble's labels and the review."""
    import importlib.util

    from omegaconf import OmegaConf

    from residual_disposition.behaviors import behavior_from_config
    from residual_disposition.config import load_config

    key = json.loads((args.set_run / "key.json").read_text())
    behavior = key["behavior"]
    spec = OmegaConf.to_container(load_config(f"behaviors/{behavior}").behavior)
    current = behavior_from_config(spec).detector.version
    if {k["detector_version"] for k in key["items"].values()} != {current}:
        raise SystemExit(f"{args.set_run}: written under the set's own detector, not {current}")
    path = REPO_ROOT / "scripts" / "continuation_labels.py"
    loader = importlib.util.spec_from_file_location("continuation_labels_writer", path)
    labels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(labels)
    kinds = list(key["kinds"])
    adl = kinds == [S3.ADL]  # the ADL-corpus set of its own (s6a 2)
    prefixes: dict = {}
    expected, _ = labels.stage3_chosen(behavior, kinds, prefixes)
    if adl:  # its text the prefix in brackets, then the continuation, which alone is checked
        shown = labels.prefix_texts(prefixes, [src for _, src, *_ in expected])
        expected = [(k, src, dose, f"[{shown[src]}]{text}", text)
                    for k, src, dose, text in expected]  # fmt: skip
        sources = S3.committed_adl_sources(args.root, behavior, rules, find_run)
    else:
        sources = S3.committed_sources(args.root, behavior, rules, find_run)
    record = S3.diagnostic_record(args.set_run, _lines(args.ensemble), _lines(args.review),
                                  sources, expected, int(rules["seeds"]["labels"]))  # fmt: skip
    for name, file in (("input", args.set_run / "input.jsonl"), ("key", args.set_run / "key.json"),
                       ("ensemble", args.ensemble), ("review", args.review)):  # fmt: skip
        record[f"{name}_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    S3.write_diagnostic(args.root, record)
    print(json.dumps({"written": str(S3.diagnostic_path(args.root, behavior, adl)),
                      "types": record["types"], "review": record["review"]}))  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("sources", help="the diagnostic's sources manifest")
    sources.add_argument("run", type=Path, help="the stage3_diagnostic_sources cell")
    sources.add_argument("--root", type=Path, default=REPO_ROOT)
    adl = sub.add_parser("adl-sources", help="the ADL-corpus diagnostic's sources manifest")
    adl.add_argument("behavior")
    adl.add_argument("--root", type=Path, default=REPO_ROOT)
    diag = sub.add_parser("diagnostic", help="the diagnostic's result manifest")
    diag.add_argument("set_run", type=Path, help="the detector-diagnostic set's run")
    diag.add_argument("--ensemble", type=Path, required=True,
                      help="the ensemble's labels: {item_id, present, unanimous, confident, "
                           "evidence} lines")  # fmt: skip
    diag.add_argument(
        "--review",
        type=Path,
        required=True,
        help="the researcher's review: {item_id, present, note} lines",
    )
    diag.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    if args.command == "diagnostic":
        return diagnostic(args, rules)
    if args.command == "adl-sources":  # after the committed stack decision and screening
        data = S3.adl_sources_record(args.behavior, rules, find_run, args.root)
        S3.write_adl_sources(args.root, data, rules, find_run)
        print(json.dumps({"written": str(S3.adl_sources_path(args.root, args.behavior)),
                          "runs": [r["run"] for r in data["runs"]]}))  # fmt: skip
        return 0
    step = (yaml.safe_load((args.run / "config.yaml").read_text()) or {}).get("step")
    if step != S3.SOURCES:
        raise SystemExit(f"{args.run}: a {step} cell, not {S3.SOURCES}")
    record = json.loads((args.run / f"{S3.SOURCES}.json").read_text())
    data = {**record, "sources_run": args.run.name}
    if not record["doses"]:  # E4: the continued screening's pools, once its manifest is committed
        data.update(S3.continued_sources(record["behavior"], rules, find_run, args.root))
    S3.write_sources(args.root, data, rules, find_run)
    print(json.dumps({"written": str(S3.sources_path(args.root, record["behavior"])),
                      "doses": record["doses"], "from": record["from"]}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
