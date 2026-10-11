"""Build a blind manual-labeling set for judge calibration (docs/preregistration.md, Judge).

Samples ``labeling.n`` items (IA response and target behavior) from the runs in ``labeling.runs``,
stratified by readout condition, variant and target, shuffles them and gives them opaque ids. The
run directory receives:

- ``labeling.html``: a self-contained page that shows only each item's target description and
  response, with keyboard shortcuts, autosave in the browser and an export of the labels;
- ``mapping.json``: the private mapping from item ids to responses and targets, which the import
  needs and the labeler should not open.

    uv run python scripts/make_labeling_set.py experiments/labeling \\
        labeling.runs=[runs/ia/<run id>] labeling.state_index=<path>/state_index.jsonl
"""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Sequence
from pathlib import Path

from residual_disposition.config import from_cli
from residual_disposition.judge import (
    LabelingItem,
    RubricSet,
    collect_candidates,
    labeling_mapping,
    labeling_set_id,
    read_state_index,
    rubric_set,
    sample_labeling_set,
)
from residual_disposition.records import IAResponse
from residual_disposition.tracking import Run

# The page loads nothing from outside: the policy forbids every external fetch.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Judge calibration: manual labels</title>
<style>
:root {
  --bg: #fafaf8; --fg: #1d1d1b; --muted: #6b6b66; --card: #ffffff; --line: #d9d9d3;
  --accent: #2456a6; --accent-fg: #ffffff; --warn: #9a3412;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --fg: #ececea; --muted: #a3a39d; --card: #20201e; --line: #3a3a36;
    --accent: #7aa7f0; --accent-fg: #0d1b33; --warn: #fdba74;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 16px; }
header { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline; }
header h1 { font-size: 18px; margin: 0; }
.muted { color: var(--muted); font-size: 14px; }
.item { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
        padding: 16px 20px; margin: 16px 0; }
.item h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em;
           color: var(--muted); margin: 12px 0 4px; }
.item h2:first-child { margin-top: 0; }
.target, .response { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; }
.target { font-weight: 600; }
.controls { display: grid; gap: 12px; margin: 16px 0; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.row .label { min-width: 120px; color: var(--muted); font-size: 14px; }
button {
  font: inherit; font-size: 15px; padding: 6px 12px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--line); background: var(--card); color: var(--fg);
}
button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent);
                              color: var(--accent-fg); }
kbd { font: 12px ui-monospace, monospace; border: 1px solid var(--line); border-radius: 4px;
      padding: 0 4px; margin-right: 4px; }
#status { min-height: 1.5em; color: var(--warn); font-size: 14px; }
details { margin: 24px 0; }
pre.guide { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px;
            background: var(--card); border: 1px solid var(--line); border-radius: 8px;
            padding: 12px 16px; }
</style>
</head>
<body data-set="@@SET_ID@@">
<div class="wrap">
<header>
  <h1>Judge calibration</h1>
  <span id="position" class="muted"></span>
  <span id="progress" class="muted"></span>
</header>
<main>
@@ITEMS@@
</main>
<section class="controls" aria-label="Labels">
  <div class="row">
    <span class="label">Strict label</span>
    <button type="button" data-strict="names_behavior"><kbd>1</kbd>Names the behavior</button>
    <button type="button" data-strict="avoids_behavior"><kbd>2</kbd>Avoids it</button>
    <button type="button" data-strict="other"><kbd>3</kbd>Other</button>
  </div>
  <div class="row">
    <span class="label">Same category</span>
    <button type="button" data-lenient="yes"><kbd>Y</kbd>Yes</button>
    <button type="button" data-lenient="no"><kbd>N</kbd>No</button>
  </div>
  <div class="row">
    <button type="button" id="back"><kbd>&larr;</kbd>Back</button>
    <button type="button" id="next"><kbd>&rarr;</kbd>Next</button>
    <button type="button" id="next-open"><kbd>U</kbd>Next unlabeled</button>
    <button type="button" id="clear"><kbd>X</kbd>Clear item</button>
    <button type="button" id="export">Export labels</button>
  </div>
  <div id="status" role="status" aria-live="polite"></div>
</section>
<details>
  <summary>Rubrics</summary>
@@GUIDE@@
</details>
</div>
<script>
(() => {
  "use strict";
  const setId = document.body.dataset.set;
  const storeKey = "residual-disposition-labels:" + setId;
  const items = Array.from(document.querySelectorAll("article.item"));
  const strictKeys = {"1": "names_behavior", "2": "avoids_behavior", "3": "other"};
  const status = document.getElementById("status");
  let state = {labels: {}, pos: 0};

  try {
    const saved = JSON.parse(localStorage.getItem(storeKey) || "null");
    if (saved && typeof saved.labels === "object") state = saved;
  } catch (err) {
    status.textContent = "Autosave is unavailable here: export before closing the page.";
  }
  state.pos = Math.min(Math.max(0, state.pos | 0), items.length - 1);

  const current = () => items[state.pos].dataset.item;

  function save() {
    try {
      localStorage.setItem(storeKey, JSON.stringify(state));
    } catch (err) {
      status.textContent = "Autosave failed: export now to keep the labels.";
    }
  }

  function render() {
    items.forEach((el, i) => { el.hidden = i !== state.pos; });
    const entry = state.labels[current()] || {};
    document.querySelectorAll("[data-strict]").forEach((b) => {
      b.setAttribute("aria-pressed", String(entry.strict_label === b.dataset.strict));
    });
    document.querySelectorAll("[data-lenient]").forEach((b) => {
      const value = b.dataset.lenient === "yes";
      b.setAttribute("aria-pressed", String(entry.lenient_label === value));
    });
    const strict = items.filter((el) => (state.labels[el.dataset.item] || {}).strict_label).length;
    const lenient = items.filter(
      (el) => typeof (state.labels[el.dataset.item] || {}).lenient_label === "boolean").length;
    document.getElementById("position").textContent =
      "Item " + (state.pos + 1) + " of " + items.length;
    document.getElementById("progress").textContent =
      "Strict labels " + strict + " / " + items.length + ", same category " + lenient + " / " +
      items.length;
  }

  function setLabel(field, value) {
    const id = current();
    const entry = state.labels[id] || {};
    entry[field] = value;
    entry.labeled_at = new Date().toISOString();
    state.labels[id] = entry;
    save();
    render();
  }

  function move(step) {
    state.pos = Math.min(Math.max(0, state.pos + step), items.length - 1);
    save();
    render();
  }

  function nextOpen() {
    for (let k = 1; k <= items.length; k++) {
      const i = (state.pos + k) % items.length;
      if (!(state.labels[items[i].dataset.item] || {}).strict_label) {
        state.pos = i;
        save();
        render();
        return;
      }
    }
    status.textContent = "Every item has a strict label.";
  }

  function clearItem() {
    delete state.labels[current()];
    save();
    render();
  }

  function exportLabels() {
    const rows = [];
    for (const el of items) {
      const entry = state.labels[el.dataset.item];
      if (!entry || !entry.strict_label) continue;
      rows.push(JSON.stringify({
        set_id: setId,
        item_id: el.dataset.item,
        strict_label: entry.strict_label,
        lenient_label: typeof entry.lenient_label === "boolean" ? entry.lenient_label : null,
        labeled_at: entry.labeled_at,
      }));
    }
    const open = items.length - rows.length;
    if (!rows.length) {
      status.textContent = "Nothing to export yet.";
      return;
    }
    if (open && !window.confirm(open + " items have no strict label and are left out. Export?")) {
      return;
    }
    const blob = new Blob([rows.join("\\n") + "\\n"], {type: "application/x-ndjson"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "labels-" + setId + ".jsonl";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    status.textContent = "Exported " + rows.length + " labels.";
  }

  document.querySelectorAll("[data-strict]").forEach((b) => {
    b.addEventListener("click", () => setLabel("strict_label", b.dataset.strict));
  });
  document.querySelectorAll("[data-lenient]").forEach((b) => {
    b.addEventListener("click", () => setLabel("lenient_label", b.dataset.lenient === "yes"));
  });
  document.getElementById("back").addEventListener("click", () => move(-1));
  document.getElementById("next").addEventListener("click", () => move(1));
  document.getElementById("next-open").addEventListener("click", nextOpen);
  document.getElementById("clear").addEventListener("click", clearItem);
  document.getElementById("export").addEventListener("click", exportLabels);

  document.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const key = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    if (key in strictKeys) setLabel("strict_label", strictKeys[key]);
    else if (key === "y") setLabel("lenient_label", true);
    else if (key === "n") setLabel("lenient_label", false);
    else if (key === "ArrowRight") move(1);
    else if (key === "ArrowLeft") move(-1);
    else if (key === "u") nextOpen();
    else if (key === "x") clearItem();
    else return;
    ev.preventDefault();
  });

  render();
})();
</script>
</body>
</html>
"""

ITEM = """<article class="item" data-item="{item_id}" hidden>
  <h2>Target behavior</h2>
  <p class="target">{target}</p>
  <h2>Response</h2>
  <p class="response">{response}</p>
</article>"""


def guide(rubrics: RubricSet) -> str:
    """The rubric texts for the labeler; the lenient one without its item placeholders."""
    lenient = rubrics.lenient.text.split("Now evaluate the following:")[0].rstrip()
    sections = [
        (f"Strict label ({rubrics.strict.version})", rubrics.strict.text),
        (f"Same category ({rubrics.lenient.version})", lenient),
    ]
    return "\n".join(
        f'<h3>{html.escape(title)}</h3>\n<pre class="guide">{html.escape(text)}</pre>'
        for title, text in sections
    )


def render_page(set_id: str, items: Sequence[LabelingItem], rubrics: RubricSet) -> str:
    """The labeling page. It holds item ids, targets and responses, all HTML-escaped, and nothing
    about states, runs, readout conditions or judge labels."""
    blocks = "\n".join(
        ITEM.format(
            item_id=html.escape(li.item_id),
            target=html.escape(li.item.target),
            response=html.escape(li.item.response),
        )
        for li in items
    )
    return (
        PAGE.replace("@@SET_ID@@", html.escape(set_id))
        .replace("@@GUIDE@@", guide(rubrics))
        .replace("@@ITEMS@@", blocks)
    )


def main() -> None:
    cfg = from_cli()
    spec = cfg.labeling
    candidates = collect_candidates(
        [Path(p) for p in spec.runs],
        read_state_index(Path(spec.state_index)),
        extra_targets=list(spec.extra_targets),
        targets_file=Path(spec.targets_file) if spec.targets_file else None,
        tables=[IAResponse.table],  # the strata are IA readout conditions
    )
    items = sample_labeling_set(
        candidates,
        n=spec.n,
        seed=cfg.seed,
        stratify_by=list(spec.stratify_by),
        allocation=spec.allocation,
    )
    set_id = labeling_set_id(items)
    page = render_page(set_id, items, rubric_set(spec.rubric_version))
    with Run(cfg) as run:
        page_path = run.path("labeling.html")
        page_path.write_text(page)
        mapping = labeling_mapping(set_id, items)
        mapping["page_sha256"] = hashlib.sha256(page.encode()).hexdigest()
        mapping_path = run.path("mapping.json")
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
        run.summarize({"set_id": set_id, "items": len(items), "candidates": len(candidates)})
    print(f"Labeling set {set_id}: {len(items)} items from {len(candidates)} candidates")
    print(f"Open {page_path} in a browser; Export downloads labels-{set_id}.jsonl.")
    print(f"Do not open {mapping_path}: it links items to responses. Then import with")
    print(
        f"  uv run python scripts/import_manual_labels.py --mapping {mapping_path} "
        f"--export <downloaded labels-{set_id}.jsonl>"
    )


if __name__ == "__main__":
    main()
