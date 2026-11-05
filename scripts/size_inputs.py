"""The size block's pinned inputs (preregistration §13.9, amendment of 2026-11-05), from the
anchor's records, written once as manifests/size_inputs.json before any training of a new
implantation.

Every new implantation trains on the anchor's actual datasets (§13.9: "the same actual prompt and
response datasets at every size"), read by id and never rebuilt:

- per habit (b = tldr, Y = keywords_line, Y₂ = zero_indexed_lists): the implant and sham data of
  its Stage-1 organisms and its training cells' prompt count and examples hash (the matched
  prompts both datasets kept);
- the suppression data of the anchor's stack lineage;
- the relearning data of b, Y and Y₂;
- the anchor itself (14B-1): its certificate run, k* manifest and references.

Ids and hashes come from the runs' recorded cells; the datasets themselves are checked against
them on the Pod (``scripts/pod/size_block.py --phase inputs``), where they are.
    uv run python scripts/size_inputs.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from residual_disposition.paths import REPO_ROOT, runs_root

OUT = REPO_ROOT / "manifests" / "size_inputs.json"

HABITS = {
    "b": {"behavior": "tldr", "implant": "stage1_14b/20261011-150850-data_implant-6c7ec446-40d9",
          "sham": "stage1_14b/20261011-161345-data_sham-7029d100-8e95",
          "active": "stage1_14b/20261011-161649-train_active-67860b58-43f4",
          "sham_organism": "stage1_14b/20261011-163152-train_sham-5b257163-62b3"},
    "Y": {"behavior": "keywords_line",
          "implant": "feasibility_14b/20261014-070232-data_implant-17197236-b398",
          "sham": "feasibility_14b/20261014-070447-data_sham-61506ac6-43fd",
          "active": "feasibility_14b/20261014-070803-train_active-a64cde15-c867"},
    "Y2": {"behavior": "zero_indexed_lists",
           "implant": "continuation_14b/20261020-032510-data_implant-6499a6a2-17fc",
           "sham": "continuation_14b/20261020-032737-data_sham-ac959517-7c0c",
           "active": "continuation_14b/20261020-033049-train_active-ef5fe695-78ed"},
}  # fmt: skip
SELECTION = "continuation_14b/20261022-103418-stage3_selection-b0158426-c685"
CERTIFICATE = "continuation_14b/20261026-130758-certificate_blocks-b0d3b664-6aad"
KSTAR = "manifests/kstar_stage3_tldr_400.json"
Y2 = "manifests/y2.json"
Y_RELEARN = {
    "data_id": "d5f5179f8853cb6cc",
    "examples": "2c4f66a288eae596a24d081e3a284fddad8e1fd166ca6b47c76e64920a4c9326",
}
Y_REFERENCE = {"state": "sbf05f589f929edf8", "run": "20261014-072219-rates-2931c028-5097",
               "hits": 145, "n": 200}  # fmt: skip


def _run(rel: str) -> tuple[dict, dict]:
    d = runs_root() / rel
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    summary = json.loads((d / "summary.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{rel}: not completed")
    return cfg, summary


def _train(rel: str, behavior: str, data_id: str) -> dict:
    cfg, summary = _run(rel)
    p = cfg["params"]
    if (cfg["behavior"] or {}).get("id") != behavior or p["data_id"] != data_id:
        raise SystemExit(f"{rel}: not {behavior}'s organism on {data_id}")
    return {"run": rel, "state": summary["state_id"], "seed": int(p["seed"]), "n": int(p["n"]),
            "examples": p["examples"], "recipe": p["recipe"], "lora": p["lora"]}  # fmt: skip


def build() -> dict:
    habits = {}
    for habit, h in HABITS.items():
        implant = _run(h["implant"])[1]["data_id"]
        sham = _run(h["sham"])[1]["data_id"]
        entry = {"behavior": h["behavior"], "implant_data": implant, "sham_data": sham,
                 "active": _train(h["active"], h["behavior"], implant)}  # fmt: skip
        if "sham_organism" in h:
            entry["sham"] = _train(h["sham_organism"], h["behavior"], sham)
        habits[habit] = entry
    sel = json.loads((runs_root() / SELECTION / "stage3_selection.json").read_text())
    if sel["implant"] != {"active": habits["b"]["active"]["state"],
                          "sham": habits["b"]["sham"]["state"]}:  # fmt: skip
        raise SystemExit("the stack lineage's implant is not the anchor's A and H")
    kstar = json.loads((REPO_ROOT / KSTAR).read_text())
    y2 = json.loads((REPO_ROOT / Y2).read_text())
    prov = y2["provenance"]
    if prov["organism"] != habits["Y2"]["active"]["state"]:
        raise SystemExit("Y₂'s manifest organism is not its training cell's")
    cert_meta = json.loads((runs_root() / CERTIFICATE / "meta.json").read_text())
    return {
        "habits": habits,
        "suppress": dict(sel["plan"]["data"]["suppress"]),
        "suppression": {
            "recipe": sel["plan"]["recipe"],
            "lora": sel["plan"]["lora"],
            "checkpoints": sel["plan"]["checkpoints"],
            "exploratory": sel["plan"]["exploratory"],
        },  # fmt: skip
        "relearn": {
            "b": {"data_id": kstar["data_id"], "examples": kstar["relearn_examples"]},
            "Y": dict(Y_RELEARN),
            "Y2": {"data_id": prov["relearn_data_id"], "examples": prov["relearn_examples"]},
        },
        "anchor": {
            "name": "14B-1",
            "certificate": CERTIFICATE,
            "certificate_status": cert_meta.get("status"),
            "kstar_manifest": KSTAR,
            "states": {
                "A": kstar["states"]["A"],
                "H_s": kstar["states"]["H_s"],
                "L": sel["doses"]["400"]["latent"],
            },  # fmt: skip
            "dose": 400,
            "references": {
                "b": {k: kstar["reference"][k] for k in ("state", "run", "hits", "n")},
                "Y": dict(Y_REFERENCE),
                "Y2": {
                    "state": prov["organism"],
                    "run": prov["reference_run"],
                    "hits": int(y2["reference_hits"]),
                    "n": int(y2["reference_n"]),
                },
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the written manifest")
    args = ap.parse_args()
    got = build()
    text = json.dumps(got, indent=1, sort_keys=True) + "\n"
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print("size_inputs.json: " + ("matches the records" if same else "DIFFERS"))
        return 0 if same else 1
    if OUT.exists():
        if OUT.read_text() == text:
            print(f"{OUT.relative_to(REPO_ROOT)} unchanged")
            return 0
        raise SystemExit(f"{OUT.relative_to(REPO_ROOT)} exists and differs: written once")
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
