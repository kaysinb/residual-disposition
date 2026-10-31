"""The laptop's side of a sealed certificate line's envelopes (preregistration §13.14; barrier B3).

    uv run python scripts/follow_seal.py <envelope.json> --repo <checkout> --receipt <receipt.json>

Each envelope, fetched from the Pod at its scheduled time whatever the calibration's state, holds
only its line, its number and its payload's sha256. It is checked (exactly those fields, a line
of B3, a sha256), committed alone as ``manifests/follow_seal/<line>/envelope_<n>.json`` in the
checkout ``--repo`` (the seal's own branch), and the receipt written holds that commit, the
envelope as committed and the raw git objects that place it there (the commit, the trees from the
root along its path, the blob), for the runner to check against its own by recomputing their ids.
Nothing else of the session is read here.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

FIELDS = {"line", "n", "commitment"}
LINES = ("eli5_stack_200", "example_link_200", "pun_intended_200")


def problems(envelope: dict) -> list[str]:
    out = []
    if set(envelope) != FIELDS:
        out.append(f"an envelope holds exactly {sorted(FIELDS)}")
    elif envelope["line"] not in LINES or type(envelope["n"]) is not int or envelope["n"] < 0:
        out.append("not a line of B3 or not an envelope number")
    elif len(envelope["commitment"]) != 64 or any(
            c not in "0123456789abcdef" for c in envelope["commitment"]):  # fmt: skip
        out.append("the commitment is not a sha256")
    return out


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def raw(repo: Path, kind: str, oid: str) -> str:
    """A git object's raw content, in base64."""
    out = subprocess.run(["git", "-C", str(repo), "cat-file", kind, oid], check=True,
                         capture_output=True).stdout  # fmt: skip
    return base64.b64encode(out).decode()


def objects(repo: Path, head: str, rel: str) -> dict:
    """The raw objects that place ``rel`` in commit ``head``: the commit, each tree from its root
    along the path, and the blob, for the runner to check by their ids."""
    parts = rel.split("/")
    trees = [raw(repo, "tree", git(repo, "rev-parse", f"{head}:{'/'.join(parts[:i])}"))
             for i in range(len(parts))]  # fmt: skip
    return {"commit": raw(repo, "commit", head), "trees": trees,
            "blob": raw(repo, "blob", git(repo, "rev-parse", f"{head}:{rel}"))}  # fmt: skip


def commit(envelope: dict, repo: Path) -> dict:
    """Commit the envelope alone; the receipt: its line, number, the commit, the envelope as
    committed (read back from that commit) and the raw objects that place it there."""
    found = problems(envelope)
    if found:
        raise SystemExit("; ".join(found))
    rel = f"manifests/follow_seal/{envelope['line']}/envelope_{envelope['n']}.json"
    path = repo / rel
    text = json.dumps(envelope, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{rel} exists with other content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        git(repo, "add", "--", rel)
        git(repo, "commit", "-q", "-m",
            f"Follow-up seal: {envelope['line']}, envelope {envelope['n']}", "--", rel)  # fmt: skip
    head = git(repo, "rev-parse", "HEAD")
    committed = json.loads(git(repo, "show", f"{head}:{rel}"))
    if committed != envelope:
        raise SystemExit("the committed envelope is not the one fetched")
    return {"line": envelope["line"], "n": envelope["n"], "commit": head, "envelope": committed,
            "objects": objects(repo, head, rel)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("envelope", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = commit(json.loads(args.envelope.read_text()), args.repo)
    args.receipt.write_text(json.dumps(receipt) + "\n")
    print(json.dumps({"line": receipt["line"], "n": receipt["n"], "commit": receipt["commit"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
