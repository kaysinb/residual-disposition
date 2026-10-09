"""Similarity of candidate behaviors to the behaviors in the IA's training data.

    uv run python scripts/behavior_overlap.py experiments/behavior_overlap

Each candidate's first-person target is compared with every public organism's ground-truth
description (quirk, rare, backdoor, heuristic and persona datasets at pinned revisions) by TF-IDF
cosine similarity on words and character n-grams. The report lists the nearest training behaviors
per candidate for review by hand; a number does not decide a family match by itself.
"""

import json
from pathlib import Path

from omegaconf import OmegaConf

from residual_disposition.config import from_cli
from residual_disposition.paths import REPO_ROOT
from residual_disposition.tracking import Run


def training_descriptions(datasets: dict[str, str]) -> list[tuple[str, str]]:
    from huggingface_hub import snapshot_download

    out = []
    for name, revision in datasets.items():
        root = Path(
            snapshot_download(
                repo_id=f"introspection-auditing/{name}",
                repo_type="dataset",
                revision=revision,
                allow_patterns=["*/eval.jsonl"],
            )
        )
        for path in sorted(root.glob("*/eval.jsonl")):
            lines = path.read_text().splitlines()
            if lines:
                text = json.loads(lines[0])["prediction_assistant_response"]
                out.append((f"{name}/{path.parent.name}", text))
    return out


def main() -> None:
    cfg = from_cli()
    spec = OmegaConf.to_container(cfg.overlap, resolve=True)
    candidates = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "configs" / spec["candidates"]), resolve=True
    )["candidates"]
    corpus = training_descriptions(spec["datasets"])
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [t for _, t in corpus] + [c["ia_target"] for c in candidates]
    words = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit(texts)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts)

    def embed(xs):
        return hstack([words.transform(xs), chars.transform(xs)]).tocsr()

    sims = cosine_similarity(
        embed([c["ia_target"] for c in candidates]), embed([t for _, t in corpus])
    )
    with Run(cfg) as run:
        rows = []
        for cand, row in zip(candidates, sims, strict=True):
            order = row.argsort()[::-1][: spec["top_k"]]
            rows.append(
                {
                    "id": cand["id"],
                    "family": cand["family"],
                    "max_similarity": float(row[order[0]]),
                    "nearest": [
                        {"source": corpus[i][0], "text": corpus[i][1], "similarity": float(row[i])}
                        for i in order
                    ],
                }
            )
        rows.sort(key=lambda r: -r["max_similarity"])
        run.path("overlap.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        lines = [
            "| candidate | family | max sim | nearest training behavior |",
            "|---|---|---|---|",
        ]
        for r in rows:
            top = r["nearest"][0]
            sim = f"{r['max_similarity']:.2f}"
            lines.append(f"| {r['id']} | {r['family']} | {sim} | {top['text']} ({top['source']}) |")
        run.path("overlap.md").write_text("\n".join(lines) + "\n")
        run.summarize({"candidates": len(rows), "training_behaviors": len(corpus)})
        print("\n".join(lines))


if __name__ == "__main__":
    main()
