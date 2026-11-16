# Blind audits of suppressed behaviors

Experiments on whether blind audits still find a behavior of a language model after the behavior
has been suppressed by fine-tuning, and whether their silence means that the behavior is gone. The
repository holds the code, the preregistered protocol with its dated amendments, the decision log
and the run records. A paper describing this work is in preparation.

## Question and approach

Suppressing an unwanted behavior of a language model by fine-tuning may hide the behavior without
removing it, so an audit that then finds nothing may be misread as evidence of removal. The study
asks what blind audits find after suppression, which parts of their material their verdicts rely on,
and what their silence means.

Six harmless habits with countable markers, such as a one-sentence summary labeled "TL;DR", were
implanted into Qwen3-14B with LoRA adapters and suppressed by further fine-tuning on answers without
the marker. Each suppressed state is compared with a control that got the same suppression but never
learned the habit, and a relearning test, which teaches the habit again, measures whether it is gone
independently of the audits. In the blind audits, tools collect material from the model (texts found
by perplexity differencing, decoded lens words and steered excerpts), an interpreter model writes
hypotheses about what the model was trained to do, and pieces of the material are deleted to see
what each verdict relies on.

## Results

Suppression was taken to two thresholds. At the first, ordinary prompts show the marker at most one
percentage point more often than in the control; one state per habit was audited there.

- Every audited suppressed state relearned its habit 1.09 to 1.61 times faster than its control.
- Blind audits found all six habits, one of them only in an audit variant designed after the
  first results, and none in a control.
- Hypotheses that named a marker relied on pieces of it left in the material, such as the token
  "TL": in seven of eight deletion tests, deleting those pieces removed most or all such hypotheses,
  while deleting as many other pieces did not. Hypotheses often still described the habit.

At the second threshold, zero expression, the marker appears in none of 27,188 prescribed checks.
One habit, an explanation for a five-year-old labeled "ELI5", reached it in two states: continued-200
(the implant trained further for 200 steps) and stack-800 (a second adapter trained for 800 steps).

- On stack-800, no hypothesis of either of two interpreters described the habit in any tested
  reading, yet the state relearns it 1.30 and 1.23 times faster than its control (at the higher and
  lower learning rate).
- On continued-200, the verdict depended on the interpreter. On the same lens words, the primary
  interpreter, with its instruction fixed in advance, described the habit in none of its attempts,
  and the second interpreter in 40 of 40, with 0 of 20 in each of its control readings. The state
  relearns the habit 1.42 and 1.39 times faster.

A silent audit therefore does not show that a behavior is gone. The setting is narrow: harmless
habits, markers found by a regular expression, one 14B model with LoRA adapters, and zero expression
reached for one habit in two states.

The decision log, [docs/decisions.md](docs/decisions.md), records the results of the stages, gates
and audit readings with their runs and the sha256 of their evidence. The numbers above aggregate
these records; the scripts that compute them will be released with the paper. The stage and gate
reports are in [docs/reports/](docs/reports/README.md), and the protocol with every dated amendment
is in [docs/preregistration.md](docs/preregistration.md).

## What is here

| Path | Contents |
|---|---|
| `src/residual_disposition/` | the research package: implantation, suppression, relearning, audit tools, judges, statistics |
| `scripts/` | entry points for every stage and the repository checks |
| `configs/` | experiment settings and the frozen protocol parameters |
| `manifests/` | the inputs of each reading, pinned by sha256 before any request |
| `docs/` | the preregistration, the decision log, the stage and gate reports, and the design, plan and operations |
| `runs/`, `data/` | the run records behind the results above (99 files, 28.5 MB) |
| `tests/` | CPU tests |

Not included: the full archive of the runs, and the model weights. Some of the packages of material
that the interpreters read will be released with the paper, and the LoRA adapters separately on
Hugging Face.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Platform and driver requirements are in
the [environment guide](docs/operations.md#environment).

```bash
uv sync
uv run pytest                      # CPU tests; `pytest -m gpu` adds the GPU integration tests
uv run python scripts/check_docs.py
```

Tests that read runs of the full run archive, which is not included, are skipped here; each skip
names what is missing.

## Run the experiments

The experiments ran on single H100 80GB GPUs in one container image. The
[operations guide](docs/operations.md) describes the environment, the image, configuration and
storage; experiments run through `scripts/pipeline.py` and the stage scripts in `scripts/`. Copy
`.env.example` to `.env` and fill in the API keys of the interpreters and judges you use.

## License

- The documentation, records and data in `docs/`, `manifests/`, `runs/` and `data/` are under
  CC BY-NC 4.0 ([LICENSE-CC-BY-NC-4.0](LICENSE-CC-BY-NC-4.0)).
- Everything else, the code and its configuration, is under the Apache License 2.0
  ([LICENSE](LICENSE)).
- Third-party material keeps its own license: the base models Qwen3-14B and Qwen3-4B, and the
  prompts and texts from alpaca-cleaned, alpaca-cleaned-ru, CodeAlpaca-20k, fineweb-1m-sample,
  c4-10k and quirk-mo-eval-data that appear in the records.
