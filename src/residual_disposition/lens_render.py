"""The follow-up's lens-only reader D* (docs/preregistration.md §13.4): its evidence, rendered from
a lens package's stored steps, and its prompt.

The rendering is the candidate reader's of 2026-10-27 (``scripts/stage3_reader_candidate.py`` at
c53b829, kept unchanged with its recorded sha256), moved here so that the follow-up's readers and
tests share one implementation; ``tests/test_lens_render.py`` holds the two byte-identical.

- **Decoders** (``state``, ``base``, ``random``): both layers 20 and 30, positions 0–4; the 100
  stored logit-lens tokens per position, most promoted first; every token among Patchscope's
  stored top 20 at any of the 30 frozen scales, once per position, as ``[token, best rank, the
  scales where it is in the top 20]`` ordered by best rank, then first scale, then token; literal
  Unicode; no probabilities, no λ choice, no ``selected_tokens``.
- **No patch** (§13.5 (a)): the control stores one Patchscope result per layer, at scale 1, shared
  by every position, and no logit lens. ``render_no_patch`` renders exactly that, in D*'s shape
  with a one-point scale grid and empty logit-lens lists, never padded to 30 scales; its prompt
  (``no_patch_prompt``, ``adl-lens-multiscale-nopatch-v1``) states those counts, its text fixed
  at barrier B2 by its sha256.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from .paths import REPO_ROOT
from .summarizer import _SHARED, PROMPTS, SummaryPrompt, _adl_lens_method

LAYERS, POSITIONS = ("20", "30"), range(5)
SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0,
          5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0)  # fmt: skip
LOGIT_K, PATCH_K = 100, 20
DECODERS = ("state", "base", "random")
BASE_CONTROL = "base_random"  # the base decoder's control on A (§13.16)
PROMPT_SHA256 = "4097a7d8ad25707c016eb7822db02e01000f5ca6728ed7ed9edc43660b1cf959"
NO_PATCH_PROMPT_SHA256 = "0588dcf3d4cd3c350323956de15014c4ca1644b755223b81f9a9b3f27b9103ee"

# The factual description of the evidence, replaced in adl-lens-v1's method text; the rest of it
# (hygiene, request) is kept verbatim.
_OLD_SUMMARIES = (
    "- The first user message includes an OVERVIEW JSON with per-dataset, per-layer summaries:\n"
    "  1) Logit lens token promotions from the activation difference. \n"
    "  2) Patchscope token promotions from the activation difference. Patchscope also contains "
    '"selected_tokens" which are just the group of tokens amongst all top 20 tokens that are most '
    "semantically coherent. They are identified by another unsupervised tool. This selection may "
    "or may not be directly related to the finetuning domain.\n"
)
_NEW_SUMMARIES = (
    "- The first user message includes an OVERVIEW JSON with per-dataset, per-layer summaries, "
    "for two layers:\n"
    "  1) Logit lens token promotions from the activation difference: for each position, the 100 "
    "most promoted tokens, most promoted first.\n"
    "  2) Patchscope token promotions from the activation difference, read at each of 30 "
    "Patchscope substitution scales (the grid is given once, as patchscope_scale_grid): for each "
    "position, every token that is among the 20 most promoted at any scale, as [token, its best "
    "rank at any scale (1 = most promoted), the scales at which it is among the 20], ordered by "
    'best rank. A scale entry "a-b" means every consecutive point of the grid from a to b, not a '
    "continuous interval. The scales are correlated settings of one reading, not independent "
    "observations.\n"
)
_OLD_LISTS = (
    "- Tokens lists are aggregated across positions, not deduplicated, and truncated to top_k.\n"
    "- Some generations may be cut off due to token limits.\n"
)
_NEW_LISTS = "- Token lists are given per position; a Patchscope token appears once per position.\n"
# The no-patch control's counts: one scale, no logit-lens tokens.
_NO_PATCH_COUNTS = (
    ("for each position, the 100 most promoted tokens, most promoted first.",
     "for each position, the most promoted tokens, most promoted first (the list may be empty)."),
    ("read at each of 30 Patchscope substitution scales", "read at each Patchscope substitution "
     "scale"),
)  # fmt: skip


def prompt() -> SummaryPrompt:
    """``adl-lens-multiscale-v2``: ``adl-lens-v1`` with only its description of the evidence
    updated, checked against the recorded sha256."""
    current = PROMPTS["adl-lens-v1"]
    if _SHARED.format(method=_adl_lens_method()) != current.text:
        raise ValueError("adl-lens-v1 is not the shared text with its method")
    text = current.text
    for old, new in ((_OLD_SUMMARIES, _NEW_SUMMARIES), (_OLD_LISTS, _NEW_LISTS)):
        if text.count(old) != 1:
            raise ValueError("adl-lens-v1's description of the evidence is not the recorded one")
        text = text.replace(old, new)
    out = SummaryPrompt("adl-lens-multiscale-v2", "adl_lens", text)
    if out.sha256 != PROMPT_SHA256:
        raise ValueError(f"adl-lens-multiscale-v2 sha256 {out.sha256} != {PROMPT_SHA256}")
    return out


def no_patch_prompt() -> SummaryPrompt:
    """D*'s prompt with the evidence counts of the no-patch control (one scale, no logit lens),
    checked against its sha256 fixed at B2."""
    text = prompt().text
    for old, new in _NO_PATCH_COUNTS:
        if text.count(old) != 1:
            raise ValueError("D*'s evidence description changed")
        text = text.replace(old, new)
    out = SummaryPrompt("adl-lens-multiscale-nopatch-v1", "adl_lens", text)
    if out.sha256 != NO_PATCH_PROMPT_SHA256:
        raise ValueError(f"the no-patch prompt's sha256 {out.sha256} != {NO_PATCH_PROMPT_SHA256}")
    return out


def scale_runs(indices: list[int], scales: tuple[float, ...] = SCALES) -> list[str]:
    """Scale indices (in ``scales``' order) as runs: ``"a-b"`` for consecutive ones, else "a"."""
    out, start = [], None
    for k, i in enumerate(indices):
        if start is None:
            start = i
        if k + 1 == len(indices) or indices[k + 1] != i + 1:
            a, b = scales[start], scales[i]
            out.append(f"{a:g}" if start == i else f"{a:g}-{b:g}")
            start = None
    return out


def _patch_scope(sweep: list) -> list[list]:
    """Every token in the sweep's top lists, once, as [token, best rank, its scale runs]."""
    best: dict[str, int] = {}
    where: dict[str, list[int]] = collections.defaultdict(list)
    for i, (_, tokens, _) in enumerate(sweep):
        for rank, token in enumerate(tokens, start=1):
            if i not in where[token]:
                where[token].append(i)
            best[token] = min(best.get(token, rank), rank)
    order = sorted(best, key=lambda t: (best[t], where[t][0], t))
    scales = tuple(round(float(s), 1) for s, _, _ in sweep)
    return [[t, best[t], scale_runs(sorted(where[t]), scales)] for t in order]


def _sha(path: Path, root: Path) -> tuple[str, str]:
    return str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()


def _overview(layers: dict, grid: list[float]) -> str:
    overview = {"datasets": {"ds1": {"patchscope_scale_grid": grid, "layers": layers}}}
    return "OVERVIEW:\n" + json.dumps(overview, ensure_ascii=False)


def render(run: Path, decoder: str, root: Path = REPO_ROOT) -> tuple[str, dict[str, str]]:
    """D*'s evidence of one package and decoder, and its source files' sha256 (paths relative to
    ``root``)."""
    if decoder not in (*DECODERS, BASE_CONTROL):
        raise ValueError(f"decoder {decoder!r} not in {DECODERS}")
    layers, sources = {}, {}
    for layer in LAYERS:
        positions = {}
        for p in POSITIONS:
            d = run / "steps" / f"layer{layer}" / decoder / f"p{p}"
            lens = json.loads((d / "logit_lens.json").read_text())
            sweep = json.loads((d / "patchscope.json").read_text())
            if [round(float(s), 1) for s, _, _ in sweep] != list(SCALES):
                raise ValueError(f"{d}: not the 30 frozen scales")
            if len(lens["tokens"]) != LOGIT_K or any(len(t) != PATCH_K for _, t, _ in sweep):
                raise ValueError(f"{d}: not {LOGIT_K} logit-lens and {PATCH_K} Patchscope tokens")
            positions[str(p)] = {
                "logit_lens": list(lens["tokens"]),
                "patch_scope": _patch_scope(sweep),
            }
            for name in ("logit_lens.json", "patchscope.json"):
                key, digest = _sha(d / name, root)
                sources[key] = digest
        layers[str(int(layer) - 1)] = {"positions": positions}  # the absolute layer: [20] = 19
    return _overview(layers, list(SCALES)), sources


def render_no_patch(run: Path, root: Path = REPO_ROOT,
                    name: str = "no_patch") -> tuple[str, dict[str, str]]:  # fmt: skip
    """The no-patch control's evidence: per layer its one stored Patchscope result (scale 1,
    shared by every position) and no logit lens, in D*'s shape (``name``: the state decoder's
    branch or the base decoder's)."""
    layers, sources = {}, {}
    for layer in LAYERS:
        f = run / "steps" / f"layer{layer}" / name / "patchscope.json"
        sweep = json.loads(f.read_text())
        if [round(float(s), 1) for s, _, _ in sweep] != [1.0] or len(sweep[0][1]) != PATCH_K:
            raise ValueError(f"{f}: not one scale-1 result of {PATCH_K} tokens")
        rendered = _patch_scope(sweep)
        layers[str(int(layer) - 1)] = {"positions": {
            str(p): {"logit_lens": [], "patch_scope": rendered} for p in POSITIONS}}  # fmt: skip
        key, digest = _sha(f, root)
        sources[key] = digest
    return _overview(layers, [1.0]), sources
