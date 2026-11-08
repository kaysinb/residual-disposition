"""Plan the eli5 marker likelihood session (descriptive measurement P3;
``residual_disposition.follow_marker``) and print a record's summary.

    uv run python scripts/follow_marker_likelihood.py plan [--with-stack-400] [--verify]
    uv run python scripts/follow_marker_likelihood.py summary RUN [--json]

``plan`` is dry and needs no GPU: it checks the frozen state ids against their committed records,
lists every state with whether it is stored under ``DATA_DIR`` (``--verify`` checks the stored
ones' adapters against their recorded sha256), reads both pinned context runs under ``RUNS_DIR``
with their files' sha256 and, when the pinned tokenizer is in the local Hugging Face cache,
splits every context's tokens as the session will; it ends with the session's inputs for
``scripts/pod/copy_inputs.py``. ``summary`` takes the record (``marker_likelihood.json``) or a
directory holding exactly one, recomputes its summary from its values (refused when it differs or
is of another design) and prints it.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from residual_disposition import follow_marker as F
from residual_disposition.paths import REPO_ROOT


def closure(store_root: Path, sids: list[str]) -> tuple[list[str], list[str]]:
    """The states and every state their adapters point at, and the ones not stored."""
    seen, missing, todo = [], [], list(sids)
    while todo:
        sid = todo.pop(0)
        if sid in seen or sid in missing:
            continue
        manifest = store_root / sid / "manifest.json"
        if not manifest.is_file():
            missing.append(sid)
            continue
        seen.append(sid)
        for ref in json.loads(manifest.read_text()).get("adapters", []):
            kind, _, rest = str(ref.get("location", "")).partition(":")
            if kind == "state":
                todo.append(rest.partition("/")[0])
    return seen, missing


def plan(with_400: bool, verify: bool) -> int:
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config
    from residual_disposition.continuation_kstar import find_run

    print(json.dumps({"design_sha256": F.DESIGN_SHA256}))
    problems, sources = F.source_problems(REPO_ROOT)
    print(json.dumps({"sources": {k: v[:12] for k, v in sources.items()}, "problems": problems}))
    labels = F.state_labels(with_400=with_400, x_400="H + stack_400/L's suppression adapter"
                            if with_400 else None)  # fmt: skip
    ws = P.Workspace(load_config(F.EXPERIMENT))
    store = ws.store
    named = [s for s in labels.values() if s and s.startswith("s")]
    stored, missing = closure(store.root, named)
    for label, sid in labels.items():
        where = "base model" if sid is None else (
            "manifest stored" if sid in stored else "composed in the session"
            if not sid.startswith("s") else "not stored")  # fmt: skip
        line = {"label": label, "state": sid, "here": where}
        if verify and sid in stored:
            try:
                store.verify(sid)
                line["verify"] = "ok"
            except FileNotFoundError:
                line["verify"] = "adapter tensors not stored here"
            except ValueError as e:
                line["verify"] = str(e)
        print(json.dumps(line))
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(F.MODEL["name"], revision=F.MODEL["revision"],
                                            local_files_only=True)  # fmt: skip
    except OSError:
        print(json.dumps({"tokens": "the pinned tokenizer is not in the local cache"}))
        tok = None
    runs, failed = [], False
    for name, spec in F.CONTEXT_SETS.items():
        run = find_run(spec["run"])
        if run is None:
            print(json.dumps({"set": name, "contexts": f"run {spec['run']} is not here"}))
            failed = True
            continue
        runs.append(f"{run.parent.name}/{run.name}")
        prompts = ws.prompts(spec["params"]["prompt_set"])
        contexts, info = F.load_contexts(run, prompts, ws.behavior.detector, spec)
        print(json.dumps({"set": name, "role": spec["role"], "contexts": info}))
        forms = collections.Counter(c.response[c.marker[0]:c.marker[2]] for c in contexts)
        print(json.dumps({"set": name, "marker_texts": dict(forms.most_common())}))
        if tok is None:
            continue
        chat = F.MODEL["chat_template_kwargs"]
        kept, splits, dropped = F.prepare(tok, contexts, chat)
        rows = F.describe(tok, kept, splits, chat)
        tokens = [r["tokens"] for r in rows]
        print(json.dumps({"set": name, "read": len(kept), "left_out": dropped,
                          "token_texts": F.token_texts(rows),
                          "strata": dict(collections.Counter(r["stratum"] for r in rows)),
                          "leads": dict(collections.Counter(r["lead"] for r in rows)),
                          "no_before": sum(r["before"] == 0 for r in rows),
                          "no_rest": sum(r["rest"] == 0 for r in rows),
                          "splits_sha256": F.splits_sha256(splits),
                          "tokens": {"max": max(tokens), "mean": sum(tokens) / len(tokens),
                                     "total": sum(tokens)}}))  # fmt: skip
    copies = [f"--state {s}" for s in dict.fromkeys(named)]
    print(json.dumps({"copy_inputs": [*copies, *(f"--run {r}" for r in runs), "--prompts"],
                      "states_read": len(labels), "not_stored_here": missing}))  # fmt: skip
    return 1 if problems or failed else 0


def summary(path: Path, as_json: bool) -> int:
    if path.is_dir():
        found = sorted(path.rglob(F.RECORD))
        if len(found) != 1:
            raise SystemExit(f"{path}: {len(found)} records, not one")
        path = found[0]
    record = json.loads(path.read_text())
    try:
        made = F.check_record(record)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}") from None
    if as_json:
        print(json.dumps(made, indent=1))
    else:
        sizes = {n: len(s["contexts"]) for n, s in record["sets"].items()}
        print(f"{path}: contexts {sizes}, {len(record['states'])} states, "
              f"design {record['design_sha256'][:12]}")  # fmt: skip
        print(F.format_summary(made))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--with-stack-400", action="store_true")
    p.add_argument("--verify", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("run", type=Path)
    s.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        return plan(args.with_stack_400, args.verify)
    return summary(args.run, args.json)


if __name__ == "__main__":
    sys.exit(main())
