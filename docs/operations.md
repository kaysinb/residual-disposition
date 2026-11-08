# Operations

How to set up the environment, configure and launch runs, work on rented machines and prepare a
public release. The design behind these procedures is in [architecture.md](architecture.md).

## Environment

- **uv.** Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 if needed.
- **Lock file.** `uv.lock` pins the dependencies for Linux x86_64. vLLM determines the PyTorch
  version.
- **GPU provider.** All experiments run on Runpod Pods from the project's container image. A local
  GPU may run fast integration tests on tiny or small models (`pytest -m gpu`); results from it
  never enter an analysis.
- **Driver.** The pinned PyTorch is built for CUDA 13, so GPU machines need an NVIDIA driver that
  supports CUDA 13 (version 580 or newer). Filter Pods accordingly.
- **Install mode.** `uv sync` installs the package in editable mode. Container runs use the
  source checkout selected by `project-run`, as described under [Containers](#containers).

## Setup

```bash
uv sync                  # create .venv from uv.lock
cp .env.example .env     # fill in the keys you use
uv run pytest            # unit tests, CPU only
```
`tests/test_meter_raise.py` reads the live meter log: do not run it while a laptop API session
is open, since that session has no Pod to close.

Documentation checks run without a commit and without a GPU:

```bash
uv run python scripts/check_docs.py
```

See the [maintenance workflow](README.md#maintenance-rules) for ownership, generated indexes and
change-impact review.

## Containers

The [Dockerfile](../Dockerfile) builds a Linux x86_64 dependency image from this project's
`pyproject.toml` and `uv.lock`. The image contains a private Python environment at `/opt/venv`;
source code is supplied separately. A code edit does not rebuild the dependency layer.
[.dockerignore](../.dockerignore) limits the build context to dependency manifests and the launcher.
The base image supplies SSH; its startup command is retained.

Build from the repository root:

```bash
docker build --platform linux/amd64 -t residual-disposition:env-v1 .
```

Run the offline CPU smoke check using a read-only checkout:

```bash
docker run --rm --network none --entrypoint project-run \
  --mount type=bind,src="$PWD",dst=/workspace/code/residual-disposition,readonly \
  residual-disposition:env-v1 /workspace/code/residual-disposition \
  python scripts/check_runtime.py --cpu
```

The GPU form of the check, without `--cpu`, runs on a Runpod Pod; the H100 check is required
before relying on this image for research. This tiny random-model check downloads no weights
and is not a scientific experiment or a full validation of the research pipeline.

On a Runpod Secure Cloud on-demand Pod, use full SSH over the public IP and mapped TCP port 22.
Upload each immutable source package under `/workspace/code/residual-disposition/<snapshot-id>/`,
with the checkout in `source/` and its manifest alongside it. Derive the snapshot ID from the
canonical file manifest and verify content, inventory and modes after transfer and before launch.
For example, set `SOURCE` to that verified checkout and run:

```bash
SOURCE="/workspace/code/residual-disposition/<snapshot-id>/source"
project-run "$SOURCE" python scripts/check_runtime.py
```

Replace `<snapshot-id>` with the actual ID. Keep wrapper logs/provenance under
`/workspace/attempts/residual-disposition/<run-id>`, outside the source and tracker-owned run directory.
Pass `RUN_ID` to the experiment, letting its tracker create the run directory. Pass the snapshot ID
as `SOURCE_SNAPSHOT_ID` and the image digest as `CONTAINER_IMAGE`; the tracker records both in
`meta.json`. Keep the snapshot receipt in wrapper evidence linked from the run's evidence manifest.
The baked `project-run` checks dependency files only, so source-manifest verification is a separate
launch step.

The source of a run depends on its purpose:
- **Gate runs** use a clone or a `git bundle` of the recorded commit, with
  `tracking.require_clean_git: true`, so every result maps to a commit.
- **Development runs** use immutable source snapshots as above, with the snapshot ID recorded.

The [launcher](../docker/project-run.py) checks that both dependency manifests match the image,
selects its Python environment, and imports the selected checkout through `PYTHONPATH`. It
preserves repository-relative config lookup without installing a second copy of the package.
Use `python`, `torchrun` or other installed commands through this launcher. Dependency changes
require rebuilding the image; do not replace the baked environment with ad-hoc installs.

By default, data and outputs go to `/workspace/data/residual-disposition` and
`/workspace/runs/residual-disposition`; `DATA_DIR` and `RUNS_DIR` can override these locations.
Model and other runtime caches use `/workspace/cache`. Attach persistent storage at `/workspace`
and back up results using the existing rented-machine procedure. The environment itself lives
outside that mount in the image, so replacing a Pod does not require reinstalling its packages.

Code transfer must include newly added source files and exclude local environments, secrets and
ignored working files. Keep each active run's source snapshot unchanged. Development may use
uncommitted snapshots, but gate runs still require a clean Git checkout and a recorded commit.
No Git credentials or `.git` directory are baked into the image.

Publish tested images to the selected container registry with a new version tag for each release;
record the image digest used for a run. Configure registry authentication separately for private
images. Container builds and source synchronization do not publish Git commits.

## `.env`

`.env` is loaded when the package is imported. Variables already set in the shell win, and blank
values are ignored. [`.env.example`](../.env.example) lists every variable with its meaning: W&B,
the Hugging Face Hub, the judge endpoint, storage locations (`DATA_DIR`, `RUNS_DIR`) and the remote
copy (`ARTIFACT_REMOTE`, a `gs://` prefix, and `ARTIFACT_GCS_KEY`, the path to its service-account
key).

On Runpod, keys come from Runpod Secrets and are passed as Pod environment variables at creation:
`WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}` and
`JUDGE_API_KEY={{ RUNPOD_SECRET_gemini_api_key_pods }}` (the Pods' own Gemini key, created on
2026-10-12; there is no `judge_api_key` secret), with
`JUDGE_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/`. No `.env` file is
copied to a Pod.

## Config overrides

Every entry point takes a config, a file path or a name under `configs/`, and optional `key=value`
overrides:

```bash
uv run python scripts/<entry point>.py experiments/smoke behavior.id=ps_signoff seed=1
```

- An override must name an existing key. This stops typos in sweeps from silently creating keys.
  `+key=value` adds a new one.
- A group override swaps a whole section: `model=qwen3-4b` replaces the `model` section with
  `configs/model/qwen3-4b.yaml`. This works for any top-level key with a matching `configs/<key>/`
  folder. Later dotted overrides apply on top: `model=qwen3-14b model.revision=<commit>`.
- Values written `???` must be set, in the config or by an override, before a run starts
  ([required values](architecture.md#configuration)).

The entry points in `scripts/` are planned ([list](architecture.md#modules)) and are added in
stage 0.

## Launching runs

- **One GPU.** A job runs on one GPU: one organism state, one relearning copy or one audit
  ([compute layout](architecture.md#compute-layout)). Hardware per stage is in
  [plan.md](plan.md#compute).
- **Several GPUs.** Jobs are independent, so several GPUs run several jobs in parallel. Sweeps go
  through the planned `scripts/launch.py`, which skips configs that already have a completed run
  ([entry points](architecture.md#modules)). No job uses multi-GPU training.
- **Several processes in one job.** Only rank 0 writes the run directory. It also generates the run
  id and broadcasts it, so `RUN_ID` need not be set; the process group must be initialized before
  the run is created.
- **Gate runs** set `tracking.require_clean_git: true` and run from a clone or bundle of the
  recorded commit ([source of a run](#containers)).

Each run writes `runs/<experiment>/<run id>/`
([contents](architecture.md#experiment-tracking-and-storage)).

## Rented machines

Set `DATA_DIR`, `RUNS_DIR` and model caches on verified persistent storage, and `ARTIFACT_REMOTE`
to a project-scoped archive independent of that working storage. See [retention](#artifact-retention)
for what to keep.

Copy finalized artifacts or consistent versioned snapshots during long campaigns and at handoff,
from this machine after the pull and its hash check (the archive key never goes to a Pod):

```bash
uv run python scripts/archive.py runs/<experiment> data/states ...
```

The script copies each path to `ARTIFACT_REMOTE` (a private, versioned Google Cloud Storage bucket)
at its repository path with `gcloud storage cp --no-clobber`, under the service-account key in
`ARTIFACT_GCS_KEY`. That account can create and read objects but not overwrite or delete them, so a
changed artifact goes to a new path. The script then compares every file's MD5 with the archived
object, fails on a missing object or a conflict, and writes an inventory (path, size, sha256, MD5)
to `runs/_archive/<UTC time>.json` and to `inventory/` in the archive. Do not archive a directory
that is still being written (a running judge or pipeline). Retrieval was tested on 2026-10-11
(a reference-label file, sha256 equal). Preserve the working copy until the independent archive is verified.
Compute can be released earlier only if the outputs remain on retained storage and a tested transfer
route works without that machine; otherwise finish the copy first.

Restore/merge finalized artifacts into the shared store with the same additive procedure, then sync
completed offline W&B records and rebuild the planned `scripts/index_runs.py` index.

Budget local scratch space for each merged rebuild of the precision check ([serving](architecture.md#serving-organism-states-with-vllm)).

### Parallel Pods on one network volume

Shared writes are not protected:
- served adapters reuse one temporary directory;
- the prompt pool is initialized on first use;
- public organisms are registered outside any lock;
- the judge and summarizer caches can pay twice and overwrite stochastic outputs;
- `index_runs.py` writes one CSV.

Pods that share a volume therefore follow these rules:

1. Shared cells first. Train the experiment's behavior-free cells (`scripts/train_ia_control.py`)
   on one Pod before any other Pod starts. Check that their artifacts exist and that their run
   metadata say completed.
2. Separate spaces. Each Pod gets its own `DATA_DIR`, `RUNS_DIR`, `HF_HOME` and `XDG_CACHE_HOME`,
   as subdirectories of the volume. Copy into each space the finished shared artifacts:
   - the control adapters;
   - the manifests of the public states;
   - the completed run directories of those cells, with the same experiment and config identity.

   Do not copy lock files. Never set `RUN_ID`.
3. Disjoint work. Pods run disjoint behaviors. Judging, the summarizer, indexing and the choice of
   completed cells happen once, on this machine, after the pull.
4. Crashes. After a CUDA "illegal memory access", CUDA requires the process to exit. Restart the
   experiment's whole process group, vLLM workers included, after checking that the old workers
   have exited. Completed cells are reused; an unfinished training cell starts again. One restart
   is a spending limit, not a guarantee: a second crash means stop and investigate.
5. Deadlines. A deadline watcher holds every Pod's lifetime. Killing Python does not stop billing.

## Judge calibration

The judge runs where the API key is, locally or on any CPU machine, over IA responses copied back
from the GPU runs; it needs no GPU.

1. Build a blind labeling set from IA response runs:
   `uv run python scripts/make_labeling_set.py experiments/labeling labeling.runs=[...]`. It writes
   a self-contained `labeling.html` and a private `mapping.json` into its run directory.
2. Get the reference labels for every item in the page's export format (JSONL): three models of
   different families, not Gemini, label independently with the set's rubric, one of them
   resolves the disagreements, and a human checks the disputed items and 30 random agreed ones
   ([judge](preregistration.md#primary-analysis)).
3. Import them: `uv run python scripts/import_manual_labels.py --mapping <mapping.json> --export
   <export.jsonl> --labeler ensemble`. Archive `data/labels/manual/` at once with the
   [archive procedure](#rented-machines).
4. Estimate the cost with `judge.dry_run: true` (the default) and check it against the budget
   before submitting; then run `scripts/judge.py` with `judge.dry_run=false`. Before the first batch,
   send a few synchronous requests to confirm the endpoint accepts the request format.
5. Check the judge run against the [preregistered validity criteria](preregistration.md#primary-analysis):
   `uv run python scripts/judge_validity.py --mapping <mapping.json> --provenance
   runs/judge_calibration/<set id>-reference --judge-run <judge run dir>`, which reads each
   annotator's labels from the provenance and writes the alternative-annotator test, κ and the
   share of positive labels.

## API cost estimates

A dry run of the judge (`scripts/judge.py`) or the summarizer (`scripts/summarize_tools.py`)
estimates the cost of the requests not yet cached (`judge.estimate_cost`): characters divided by
characters per token as input, and an expected answer plus thinking tokens per request as output,
at the model's standard prices and the tier's discount. The estimate sets a run's limits only; a
send's ledger reservation does not use it (it is bounded by the request's UTF-8 bytes and
`max_tokens`, `adl_graders.max_cost`). The limit of a run is its calibrated dry run + 30 %
(2026-10-17): a run that reaches it stops before the send that would cross it, and
its rerun pays only for what the cache lacks.

- **Parameters.** The judge: `judge.cost.chars_per_token`, `judge.cost.expected_output_tokens` and
  each `judge.models` entry's `expected_thinking_tokens`. The summarizer:
  `summarizer.expected_output_tokens`, and `summarizer.chars_per_token` and
  `summarizer.expected_thinking_tokens` by evidence tool, else the prompt's tool (`base_tool`),
  else `default`.
- **Tiers** (2026-10-19). The interpreter reads on flex
  (`judge.sync.service_tier: flex`) with `summarizer.flex_fallback`: when, of the last 100 flex
  sends, at least 90 % answered HTTP 503, or no flex send has succeeded for 15 minutes since the
  first send after the last success and at least 90 % of that stretch's flex outcomes (counted
  from that send; an outcome arriving while no stretch runs is not counted) were HTTP 503, the
  rest of the run goes to the standard tier, one-way; a request whose own flex attempts
  all answered 503 is sent once on the standard tier. Only HTTP 503 counts (never a quota, another
  5xx, a timeout or an answer). The switch and each per-request fallback are journaled, flushed to
  disk, in the run's `tier_fallback.jsonl` before the standard send they cause, and summarized in
  `service_tiers.json` and the summary (`flex_fallback`). Each attempt is reserved, settled and
  released at its own tier (a 503 is released on flex only), and every ledger event records it;
  each answer carries the tier that answered it, so a mixed run's provenance is per answer. Each
  ledgered answer is cached with the attempt it belongs to (worker, reservation id, tier); a
  ledger's reconciliation settles an open reservation from the cache only when that cached answer
  names this worker, this reservation and its tier, the reservation has no recorded outcome and
  is its request's only candidate; otherwise it stays open at its bound, and so does a failure
  that may have been billed. Before
  anything is sent, the estimate at the dearest tier a request may end on (the standard tier with
  the fallback) must be within `summarizer.max_usd`, and that estimate + 30 % within what the
  ledger has left. The summary's `paid_by_tier` gives the answers' shares per tier and what flex
  saved against the standard tier. The judge reads on the standard tier:
  `scripts/judge.py experiments/judge_continuation`.
- **Recalibration.** `uv run python scripts/estimator_calibration.py --judge-ledger <worker> ...
  --summarizer-cache <dir>[=<tool>] ... --write tests/data/estimator_calibration.json` measures
  the cached replies (the judge: the requests a ledger settled) and prints each group's tokens and
  its estimate/actual ratio under the current configs. Characters per token are set just below
  the measured ratio and thinking at 1.3 × the measured mean (the estimate predicts a sum over
  thousands of requests). `tests/test_estimator_calibration.py` is a regression check on the
  measured sample: every measured group's estimate lies between its actual cost and 1.6 times it.
  It does not show that a future run's cost stays within its estimate + 30 %: other evidence,
  prompts or models can use more tokens, and a run that reaches its limit stops and is resumed
  with a new limit.

### Calibration of 2026-10-17

Before it, the judge's estimate was 2.0–2.1 times the actual cost of Stage-1 judging (dry runs
$6.97 and $4.42; actual $3.29 and $2.23 for the same 5,850 and 3,692 requests): thinking was
expected at 50 and 300 tokens against 0 and 53, and 4.0 characters per token against 4.63 and
4.81. The summarizer used the judge's 4.0 characters per token, above the measured ratio of
lens-only packages, Diff Mining and ADL. Measured (input and output in tokens per request, output
billed with its thinking; USD at standard prices):

| reader | model | group | n | input mean / p90 | output mean / p90 | answer mean / p90 | thinking mean / p90 | chars/token | USD/request | estimate/actual |
|---|---|---|---|---|---|---|---|---|---|---|
| judge | gemini-3.5-flash-lite | strict | 4771 | 1633 / 1640 | 15 / 17 | 15 / 17 | 0 / 0 | 4.63 | 0.000527 | 1.07 |
| judge | gemini-3.8-flash | lenient | 4771 | 571 / 578 | 54 / 92 | 1 / 1 | 53 / 91 | 4.81 | 0.000631 | 1.45 |
| summarizer | gemini-3.8-flash | activation_difference_lens | 1120 | 15676 / 18644 | 214 / 609 | 21 / 30 | 192 / 592 | 3.27 | 0.012558 | 1.07 |
| summarizer | gemini-3.8-flash | adl_lens | 9280 | 2703 / 3181 | 723 / 1971 | 13 / 18 | 710 / 1958 | 2.28 | 0.004739 | 1.23 |
| summarizer | gemini-3.8-flash | adl_unsteered | 1120 | 4339 / 4466 | 79 / 27 | 48 / 25 | 31 / 0 | 3.88 | 0.003550 | 1.03 |
| summarizer | gemini-3.8-flash | diff_mining | 1320 | 1623 / 1698 | 1299 / 3489 | 16 / 18 | 1282 / 3473 | 1.70 | 0.006087 | 1.28 |
| summarizer | gemini-3.8-flash | pd_clean | 2720 | 11786 / 13803 | 242 / 41 | 24 / 30 | 218 / 0 | 4.02 | 0.009749 | 1.04 |
| summarizer | gemini-3.8-flash | perplexity_differencing | 1777 | 9508 / 12821 | 94 / 32 | 21 / 29 | 74 / 0 | 4.04 | 0.007485 | 1.03 |

The judge's rows are Stage-1 judging (ledgers cont-judge and cont-judge-pd, every settled
request); the summarizer's are every cached reply (`data/cache/summarizer`; `pd_clean` is
`data/cache/summarizer-pd-clean`, PD-clean's packages and their random controls under PD's
prompt). Estimate/actual is under the calibrated values. Every current evidence tool has its own
or its prompt tool's values; `default` (4.0 characters per token, 1,000 thinking tokens) applies
only to a tool added later. gemini-3.8-flash was measured as the lenient grader only, so its
thinking as a strict grader (the escalation stage) is not measured.

### The stage plan under the execution decisions of 2026-10-19

`scripts/continuation_inventory.py` prices the plan from the frozen units
(`configs/continuation/inventory_units.yaml`) and the execution units
(`configs/continuation/inventory_units_execution.yaml`); `--api-tier` re-prices the API units
with Stage 1's measurements at the calibrated estimate. Runs:
`runs/continuation_inventory/20261019-s3j-{none,flex,standard}` (before the step-budget decision of 2026-10-19:
`20261019-execution-{none,flex,standard}`). The lines carry the decisions of 2026-10-19
([execution rules](#continuation-execution-rules-of-2026-10-19)): A and H reused (27 zero-cost
lines in the main branch, each naming its source), a method-equivalence validation for each of
the 48 lens-only packages without a pilot artifact, k* on six H_s seeds per lineage (three per
rate), and, from the step-budget decision, no learning-rate search (the rates are frozen, §12.4.8; −$5.32 per
certificate, central) and four runs censored at 600 steps per certificate in the stress
projection (+$14.90 per certificate; the pilot censored none of 64, so none in the central).

| unit | frozen | interpreter on flex | standard tier |
|---|---|---|---|
| lens-only reading, per state: λ grading $0.3758 + four packages (30/state/1; the second λ draw 30/state/2, every state, §12.4.5; 30/base/1; 20/state/1), each 80 interpreter requests and their judging | $0.43 | $1.46 | $2.39 |
| PD-clean reading or random-removal control, per package | $0.10, $0.08 | $0.48 | $0.88 |
| fidelity readings on A, per lineage (two packages; not in the frozen inventory) | — | $0.54 | $1.01 |
| method-equivalence validation, per package (GPU, [estimate]: 15.5 min, high 18.9, as a full pilot ADL package) | — | $1.37 | $1.37 |

The judge is on the standard tier in both columns. Main branch, under the $1,770 cap:

| | §12.3.9 (cap $1,500) | lines of 2026-10-19, frozen API units | interpreter on flex | standard tier |
|---|---|---|---|---|
| central projection | $1,474 | $1,509 | $1,623 | $1,730 |
| stress projection | $2,151 | $2,341 | $2,454 | $2,562 |
| through Stage 3, with the reserve | $1,365 | $1,402 | $1,509 | $1,610 |
| margin to the cap | $135 (to $1,500) | $368 | $261 | $160 |
| central cuts by the order (hypothetical) | Stages 6, 5 | none | none | Stages 6, 5 |
| stress cuts by the order (hypothetical) | 6, 5, guillemets, the last panel behavior, Stage 2 to one | 6, 5, guillemets, the last panel behavior | as §12.3.9 | as §12.3.9 |

Under the $1,820 cap (2026-10-20; `runs/continuation_inventory/20261020-cap1820-{none,flex,standard}`, at 3e83f2b, with C10's method-check
unit of 4 / 8 min), main branch, frozen API units / interpreter on flex / standard tier:
- central $1,460 / $1,574 / $1,681 ($360 / $246 / $139 below the cap), no cuts by the order;
- stress $2,294 / $2,408 / $2,515; hypothetical cuts Stages 6 and 5, guillemets and the last panel
  behavior, on the standard tier also Stage 2 down to one behavior.

Against §12.3.9 at the frozen API units: the method-equivalence validations add $65.97 (48 ×
$1.37; the legacy extractor computes only the layer-20 δ̄, so this [estimate] is likely high),
k*'s second three seeds add $33.28, and the reuse of A and H saves $16.66 ($26.77 with the
interpreter on flex, $36.19 on the standard tier, at the re-priced units).

- **The price of Stages 5 and 6.** The cut order is applied only after the researcher's decision
  (§12.8, 2026-10-19). Every stage of the central projection fits the $1,770 cap with the
  interpreter on flex ($1,722.53, $47.47 below); on the standard tier it needs $1,830.27
  (+$60.27). Before the step-budget decision: $1,770.41 and $1,878.15.
- **The first batch, stage2/example_link:** 28.1 GPU work hours (47.8 high); API $10.69 with the
  interpreter on flex, $16.85 on the standard tier; central $160.37 / $166.54, stress $264.72 /
  $270.88. Its frozen API part, $5.32, is below its API: money moves from its GPU part after the
  GPU work (§12.8, transfer), or the researcher decides.
- **Guillemets' admission:** the meter holds guillemets' $199.07 stress reservation. With stage0_1
  closed at its $80.67 and every item above guillemets at its central cost, the gate is $1,485.36
  with the interpreter on flex ($284.64 below the cap) and $1,544.22 on the standard tier
  ($225.78); with the first batch at its stress, $1,589.71 and $1,648.57, both within the cap.

Earlier, on 2026-10-17 and before these decisions: through Stage 3 with the
reserve $1,440 with the interpreter on flex and $1,512 on the standard tier, against the $1,500
cap then in force. The frozen allocations stay as they are; each run's `allocations.json` is for
comparison only, and §12.8's update rule raises the allocation of each kind's remaining lineages
to the mean actual cost of its completed ones.

## Continuation: execution rules of 2026-10-19

Decided after Stage 1 (preregistration §12.3.7,
§12.4, §12.4.1, §12.4.3 and §12.8 carry them).

- **Cap.** `configs/continuation/rules.yaml` `meter.cap_usd` is the meter's operating cap.
  The rules' frozen sha256 (`simulate_continuation.FROZEN`) is `0bc36aab…`, formed with the cap
  fixed (`continuation_rules.HASH_CAP_USD`): the cap is outside the rules' hash, so its changes
  keep it; the official Stage-0 simulations ran under `9cb5c4c9…`, which differs only in the
  cap, and no simulation reads the meter. After a change of the rules' cap the meter's log
  records nothing until it is migrated, once, to it right after the merge (a `cap` event, with
  its reason, is allowed while a Pod session is open):
  `uv run python scripts/continuation_meter.py migrate-cap --reason "<the reason>"`.
  The log's `cap` events hold its history; `status` shows the cap in force, the rules' cap and
  `needs_migration`.
- **A shortfall is stopped and priced.** A step that does not fit (a refused admission, a projected
  overrun of a part, a stage the cut order would cut) is stopped and reported with the amount to
  add and what would be cut otherwise; nothing is cut or cancelled before the researcher decides.
- **Confirmation prompts.** A confirmation set of its own: 600 prompts disjoint from the
  screening, exploratory and training prompts, frozen with a committed manifest before the first
  draw, the existing partitions not reshuffled. Each attempt draws fresh response streams; a
  retest after a detector revision uses new seeds on the same set.
- **Method-equivalence validation.** A lens-only package of a state without a stored pilot
  artifact (the Stage-2 deep doses, every later lineage) is checked by the independent legacy
  single-layer extractor against the new extraction on the same checkpoint and pinned corpus,
  at §12.4.3's tolerance; the run stops otherwise, and both results are saved with the package.
  It is not called pilot reproduction.
- **A and H reused.** A later lineage of the same implant reuses A's and H's readings (lens-only,
  their stability draws, A's fidelity controls, A's PD-clean reading and random-removal control)
  when the state, package, detector, prompts and judging versions match; the reusing run records
  the source run ids and those versions, and a reused sample counts once. H_s, X and the recipe
  control are read at the lineage's dose.
- **k* per learning rate.** Three H_s calibration seeds at each rate; `manifests/kstar_<lineage>.json`
  holds both values with their seeds and crossings, committed before the certificate; every arm
  at a rate stops at the first evaluation at or after max(20, k*_rate) with a latched crossing.
- **Rule C on the reader formats** (Stage 1, §12.3.3): validated on lens-only at layers 30 and 20;
  failed on PD-clean, so the confirmatory PD-clean interpretation stops
  ([decisions](decisions.md)).
- **Stage 2's screening** (§12.7; `scripts/pipeline.py experiments/continuation_14b/<behavior>
  pipeline.plan.phase=screen`, batch stage2). For each deep dose of the frozen rules (400, then
  800): the exploratory paired excess of L over H_s, read from the pilot's exploratory pass
  (measurement observer, the certificate prompts) and re-scored with the behavior's frozen
  detector on its saved responses, with no new draw (example_link's detector was widened on
  2026-10-15, after that pass; its accepted lenient detector is that detector, and pun_intended
  has no other); X composed from H and L's suppression adapter; L's full PD continuation pool
  (both framings) and the detector's hits over it. The pool counts only at the frozen PD settings
  and prefills (9,194) with exactly one completion per framing (chat, raw) and prefill; an
  incomplete pool is refused, and the plan check refuses other PD settings before any GPU. The
  first deep dose with an excess bound of at most 0.01 and no hit is selected; with none,
  exception a applies. The record is the cell `stage2_screening.json`. Phase `ood` runs the
  out-of-distribution observer (§12.4.7) of the states in `plan.ood.states`: one response to
  each of the 200 frozen OOD prompts (one sample, whatever the behavior's sampling sets), the
  detector's hits and the one-sided 95 % Clopper–Pearson bound against 0.02 (0 of 200 is the
  only count within it), computed only over exactly the 200 prompt/sample pairs.
- **Committed manifests on a Pod** (C8, item E). The Pod's source is a verified
  snapshot without `.git`, so `manifests.committed_json` accepts a manifest there only by the
  snapshot's receipt, authentic only by trusted launch metadata (no secrets, no signatures):
  1. on the laptop, right after `source_snapshot.py prepare` from the clean commit, `uv run
     python scripts/snapshot_receipt.py write <package>` checks every snapshot file against that
     commit's blob and writes `<snapshot id>.receipt.json` beside the package;
  2. a separate check, `uv run python scripts/snapshot_receipt.py check <receipt> --package
     <package>`, makes the receipt again from git, refuses one that differs and prints
     `SNAPSHOT_RECEIPT_SHA256` (the receipt file's sha256) and `SOURCE_SNAPSHOT_ID`;
  3. both are set in the Pod's environment at its creation; the chain passes the
     receipt's path (`SOURCE_SNAPSHOT_RECEIPT`).
  A manifest is committed on the Pod only if the receipt is the one with that sha256, its own
  sha256 is the snapshot manifest's and the receipt's and the receipt names a clean commit of
  that snapshot; a manifest committed later needs a new snapshot. Every acceptance (k*, Y₂,
  Stage 2) leaves what the check adds (`sha256`, `committed_at`; `manifests.TRANSPORT`) out of
  its comparisons.
- **The Pods' sessions in the meter** (C8, item D). After `start`, `split
  stage2/<behavior> --share pod-a=… --share pod-b=…`. For each Pod session the Pod is
  created first, with its actual ID and the deadline set at its creation; `session-open
  <item> --worker <pod-a|pod-b> --pod-id <ID> --deadline <ISO> --price <USD/h> --hours <h>
  [--api-usd <USD>] --receipt <file>` authorizes it in the log, exclusively (refused while that
  worker has an open session), its hours within the worker's live GPU share and the API money
  the session may commit (none by default) within its API share, and writes the
  receipt the Pod verifies (`verify-receipt <file> --item … --worker … --pod-id <its own ID>
  --price … --hours …` on the copy of the log, which prints the session as the log holds it).
  After the Pod stops, `session-close <item> --worker … --pod-id … --gpu-usd <paid> [--api-usd
  <paid>]` settles its cost; only then can that worker's next session open. A Pod holds one
  open session, whichever worker's: one Pod's machine may serve pod-b's session and then pod-a's
  (2026-10-19, for lack of H100 SXM capacity), not both at once. A worker's charges may not
  exceed its share; an overrun stops the item.
- **Stage 2's Pods** (C8, items B, C, F). `scripts/pod/continuation_stage2.sh`,
  one session per Pod start; on every exit, a refusal before any pass included, it asks the
  watcher to stop the Pod within 3 minutes. Pod A (worker pod-a) writes only to `/workspace/a`,
  Pod B (pod-b) only to `/workspace/c`: two Pods writing one network volume can corrupt it, so a
  Pod copies what it reads into its own space, create-only and checked by sha256
  (`scripts/pod/copy_inputs.py`; a state with every state its adapters point at, a dataset with
  the completed cells that made it), each item from the first of its sources that holds it: the
  other Pod's space, then `/workspace/b`; it never writes to either.
  - **What lives where.** `/workspace/a` holds the pilot's link states, the analyst index, the
    prompt pools and A's reference (Stage 1's first space). `/workspace/b` (Stage 1's second
    space, read only from Stage 2 on) holds the pilot's Y organism (keywords_line,
    `sbf05f589f929edf8`), its reference run and its relearning data and cell, and the stage-1
    recipe run (`feasibility_14b/20261014-070803-train_active-a64cde15-c867`), which the Y₂
    check reads. B1 copied the Y items into c (2026-10-19, `copied-from-b.jsonl`).
  - **The Y₂ selection's laptop inputs in b.** At the start of A1's session, before its chain
    and while no chain reads b: the stage-0 elicitation run, the detector check's set, labels,
    review log and any dispositions or rejections are uploaded into b at the paths
    `continuation_y2.locate` records (relative to the runs or data root), create-only, each
    file's sha256 checked against the laptop's and listed in an upload log.
  - Each pass gets only its own batch's cap (`pipeline.plan.cost.batches.<batch>`: y for
    control_y, stage2 for the other phases); another batch's key is not in every config.
  - **a1**: phase y2 for zero_indexed_lists first (its failure does not stop the session, which
    then exits nonzero), then phase screen, then the steps `scripts/pod/stage2_next.py` names
    from the space's completed cells (phase confirm at a dose, C9; the expression checks at the
    analysis dose, phase ood and phase corpus, C10) and phase decide with the runs found. It ends
    there: the decision is written (`scripts/continuation_stage2.py`) and committed, and a new
    snapshot made, before the lineage's k* (§12.7). A Stage-2 k*'s plan check
    needs the committed decision, its analysis dose and H_s there. A step that leaves no cell
    stops the session; a retest after a detector revision stops it and is reported.
  - **The k* session**: **a1k** (Pod A, H100 SXM) runs phase kstar at the committed decision's
    analysis dose and, under option (b) (A1 on another Hopper, with an execution note
    in decisions before the launch), phase pd_pool (k* whole on
    one configuration: the k* evidence refuses calibration runs whose execution signatures,
    GPU, container image, driver and CUDA, packages and engine from their meta and cell, are
    incomplete or not all the same), then phase pd_pool, L's full PD pool redrawn in
    a cell of its own (the draw tag `stage2-sxm`, run only on `NVIDIA H100 80GB HBM3`; the
    record gives it beside the screening's). The screening's count stays the screening's
    pool's.
  - **L's PD pool, declared** (`configs/continuation/pd_draws.yaml`, frozen before the
    lineage's A1): null, the screening's pool; `stage2-sxm`, the redrawn one
    (option b), set by a decision before A1′ whatever it gives. Phase readers refuses a
    `plan.readers.pd_draw` (`PD_DRAW` of a2) other than the declared one, a left-out one
    included; PD-clean (`scripts/pd_clean.py`, `state_draw`) reads Stage 2's L from the
    declared pool (derived from the declaration, a plan entry naming another refused) and every
    other state from the draw its entry names (Stage 1's plans: none). Every final
    PD pool, L's and the controls', ran on `NVIDIA H100 80GB HBM3`: a cached pool from another
    GPU is refused before any GPU work, a new one is drawn only there.
  - **b1** (needs no dose): the Y alone for link at 100 and 200, after copying link's L and H_s
    at 100 and 200, the analyst index and the prompt pools (a) and the pilot's Y organism, its
    reference and relearning data (b).
  - **a2** and **b2**, after the k*, Y₂ and Stage-2 manifests are committed and a new snapshot
    made: phase certificate, blocks 0–2 (a2, then phase readers, C10) and 3–5 (b2). Each first
    copies what the certificate reads, as the committed manifests name it (`copy_inputs.py
    --certificate`, `continuation_certificate.dependencies`), from the other's space and then
    b: a2 the pilot's Y and its data from c, the recipe run and the Y₂ check's inputs from b; b2
    the dose's states, A1's calibration runs, A's reference, b's data and A1's Y₂ reference, its
    runs and the Y₂ arm's data from a, the analyst index and the prompt pools from a, the recipe
    run and the Y₂ check's inputs from b. Then the certificate's plan check alone
    (`pipeline.plan.check_only=true`, CPU) in its own space, and the GPU pass only if it
    passes.
  - **The second behavior, pun_intended** (2026-10-20): its item `stage2/pun_intended` is split
    to Pod B alone (`split --share pod-b=1.0`), one Pod at a time in c, while Stage 5 writes a.
    The sessions need `ITEM=stage2/pun_intended`; a p-session with no item or any other refuses.
    - Its pilot cells are in b, from 2026-10-13: A, H, L and H_s at 400 and 800, the deep doses'
      exploratory passes, A's reference and b's relearning data.
    - **p1**:
      - `scripts/pod/stage2_inputs.py` names the inputs from the behavior's config and finds
        them in b then a, read only. Exactly one pass per state and one relearning dataset are
        required.
      - It also names the pilot's index (feasibility_14b, stage1_14b; never continuation_14b,
        which the Stage-5 Pod writes), the prompt pools, and by name the
        confirmation pool and the OOD build, which may be only in a.
      - copy_inputs takes them from b, then a, into c. It copies only completed indexed runs.
      - `scripts/pod/hf_prefetch.py` fetches the pinned PD and ADL corpora into c's HF cache
        (`HF_HOME`, the runner's `<space>/hf`). A corpus it cannot fetch stops the session before
        any pass.
      - Then a1's steps without y2 (Y₂ is selected), ending at the decision cell.
    - The Stage-2 decision is then committed and a new snapshot made.
    - **p2**: k* at the committed decision's analysis dose, H_s from p1's screening record, A
      from the config. Its manifest is then committed and a new snapshot made.
    - **p3**:
      - It copies the pilot's recipe control at the dose with its adapters, which the readers
        read (`stage2_inputs.py --recipe-control`).
      - Then the certificate's six blocks on this one Pod (its plan check first), resumed by the
        next session at a deadline, then the readers at the analysis dose.
      - L comes from p1's screening record in c; the certificate's inputs come from b, then
        a.
  - Between sessions the Pod is stopped, not left idle; if a stopped Pod cannot start again (its
    host taken), a new Pod is created on the same volume and the session reruns, resuming its
    cells.
  - Every check run on the Pod inside the snapshot's tree before the chain (`project-run
    <snapshot>/source python …`) sets `PYTHONDONTWRITEBYTECODE=1`, as the chain does. A stray
    `__pycache__` fails the chain's snapshot verification, and the chain then stops the Pod (pun
    p1, 2026-10-20).
  - Per session the Pod is created (its ID, the deadline at its creation,
    `SOURCE_SNAPSHOT_ID` and `SNAPSHOT_RECEIPT_SHA256` in its environment), the meter opens the
    session (`session-open`, with the session's API money), and the chain is launched with
    `SESSION`, `SNAP`, `WN`, `PRICE`, `RECEIPT`, `METER_LOG`, `POD_ID`, `SOURCE_SNAPSHOT_RECEIPT`
    (and `ACTIVE` for a1). It refuses to run unless `POD_ID` is the provider's identity of the Pod
    (`RUNPOD_POD_ID`), the snapshot verifies, the snapshot and its receipt are the ones set at
    creation and the session receipt verifies on this Pod. Every bound then comes from the
    session as the log holds it: the watcher's deadline may not lie beyond the session's (no
    deadline the chain sets does), the cost meter counts the session's hours back from that
    deadline, and the ADL ledger's allowance is the session's ceiling: recorded in the worker's
    ledger at the session's first start, before any send, as what the ledger had committed plus
    the session's API money, and reused, never recomputed, by every rerun of that session (its id
    the sha256 of the meter log right after it; `budget.session_allowance`). After the
    Pod stops, `session-close` settles the session's paid cost.
- **Stage 2's confirmation** (§12.4.1; plan C9). Phase confirm (batch stage2,
  `pipeline.plan.confirm.{dose,latent,sham_s}`, the dose and states as `stage2_next.py` names
  them) checks on CPU before any GPU work (`pipeline.confirmation_plan`): the behavior's one
  completed screening records the states at the dose; the frozen confirmation prompts load
  (`data/prompts/confirmation-7c4c37f36708`, built by `scripts/make_confirmation_prompts.py`,
  set hash in `continuation_confirm.FROZEN`; it must be in the Pod's space); the lineage's
  ledger (`stage2/<behavior>`, base seed 20261021), replayed from its completed confirmation
  cells, permits the dose next; every confirmation run of the lineage that did not complete is
  this very cell. The cell's identity is the allocated attempt (attempt, purpose, seed, detector
  version) with its whole generation spec (model, behavior, sampling, the generator's seed,
  engine, requests, states, prompts), written with its config before any
  generation; an open run of the lineage with any other setting refuses the pass before any GPU
  work. It draws one response of L, then of H_s, to each of the 600 prompts (stream
  `confirmation:<seed>`), saving each state's complete draw in the run (`draws/<role>.jsonl`)
  with its metadata (the state, the run that drew it, the runs it was lost in) before the next;
  a mark (`draws/<role>.started.json`) is written before each state's generation, so that a
  lost draw is told from one never started. A resumed run reuses a saved draw with its origin's
  metadata and draws again only one lost in flight; its summary's `draws` gives each state's
  metadata, kept through repeated interruptions. It
  records L's exact test (`continuation_confirm.outcome`: p₁ ≤ 1/60), H_s's hits and the paired
  Newcombe bound. Phase decide recomputes each draw's pass from L's
  counts and refuses a draw of another ledger or prompt set.
- **Stage 2's expression check and readers** (§12.4, §12.4.3, §12.4.7; plan C10). Phase corpus
  (batch stage2, `pipeline.plan.corpus.state`, the latent state of a screened deep dose) runs
  the ADL-corpus check (`adl_corpus_check`, both framings); its run summary carries the counts
  the decision reads (`framings.{chat,raw}.{pooled_hits,pooled_n,long_hits,long_n}`,
  `corpus_framings`). Phase readers (batch stage2, `pipeline.plan.readers.dose`) checks on CPU
  before any GPU work (`readers_plan`) that the dose is the analysis dose of the behavior's
  committed decision (recomputed from its runs), and resolves the states: L, H_s and X as its
  screening records them at that dose, and the recipe control as the analyst index's (this
  experiment and those it reuses) one recipe_control of the behavior, stack, SFT, at that dose,
  else it refuses; the PD settings and ADL's (`FROZEN_ADL`: the lens-only cells' and the corpus
  check's settings as Stage 1 ran them, corpus revision and λ grader included, and the method
  check's) are the frozen ones, before corpus too; and the ADL ledger's ceiling
  covers the λ grading left: a lens state whose package is already built (a resumed pass) is
  reused before any API check and counts for nothing. A and H are not read again (§12.4,
  execution 2026-10-19). For L,
  H_s, X and the recipe control in that order: the **method-equivalence validation**
  (`adl_method_equivalence`: the legacy single-layer extractor's layer-20 δ̄ alone on the same
  checkpoint and pinned corpus, saved in the pilot package's layout), the lens-only packages
  checked against it (the lens δ̄ kept in the lens run's `reproduction/`; a δ̄ that does not
  reproduce within the tolerance stops the pass before any reading) and the PD completions (L's
  is the screening's cell). The record (`stage2_readers.json`) names the states, the recipe
  control's provenance, the reused A and H, and per state its runs and its validation: both δ̄
  files with their sha256, the largest relative error and the tolerance. The readings, PD-clean,
  judging and the verdicts are the laptop's (C11).
- **Stage 2's decision** (§12.7, §12.4.1, §12.4.7). Phase decide (CPU, batch
  stage2, `pipeline.plan.decide.screening=<stage2_screening run>`,
  `'pipeline.plan.decide.confirmation=[<attempt 1 run>, <attempt 2 run>]'`,
  `...expression.{adl_corpus,ood}=<run>`) reads the named runs, each checked to be a completed
  cell of its step and state (a cell that did not complete, or technically invalid content,
  counts as not done), replays the attempt ledger from the confirmation draws (revisions from
  the detector versions, each draw's fresh seed, exhaustion) and applies
  `continuation_rules.stage2_decision`: d₁ the first screened deep dose, d₂ only after d₁'s
  failed confirmation, the retest after a revision, the expression checks at the analysis dose
  (the ADL corpus per framing, chat and raw), exception a when the completed procedure yields no
  analysis dose or it fails a completed check. A pending record names the next step; the cell
  is keyed by its result, so a pending record is never served for a later decided one. `uv run
  python scripts/continuation_stage2.py <stage2_decision run>` writes a decided record as
  `manifests/stage2_<behavior>.json`, its file its behavior's and the whole record recomputed
  from its runs, committed before the lineage's k*; `--slot` prints the Stage-5 slot from the committed decisions (example_link first).
- **Stage 2's certificate** (§12.3.6, §12.3.7, §12.4.8; decided 2026-10-19).
  At the selected deep dose, in order:
  1. **k* calibration.** H_s relearned on b, relearning seeds 6, 7 and 8 at each of 10⁻⁵ and
     5·10⁻⁶, each run stopping at its first crossing, budget 600 steps. T is the interpolated
     crossing (`relearn_rules.interpolated_crossing`, the pilot's convention); a run without a
     crossing by 600 is censored; a failed run is not censored and has no T until rerun. Per
     rate, the median of the three T counts a censored run as slower than every crossing (one
     censored: the larger observed T); two or three censored: k* is unresolved, the lineage's
     certification stops with a report. k*_rate is the evaluation step (schedule every 2 steps to
     20, every 5 to 100, then every 10) nearest half the median, ties to the earlier step.
  2. **Budget.** B = min(600, 50·⌈max(3·median(T_lo), 20, k*_hi, k*_lo)/50⌉), T_lo at 5·10⁻⁶;
     one B for all 72 runs. Both k* and B go to `manifests/kstar_<lineage>.json`, committed
     before the first certificate run. Steps 1–2: `scripts/pipeline.py
     experiments/continuation_14b/<behavior> pipeline.plan.phase=kstar
     pipeline.plan.kstar.dose=<dose> pipeline.plan.kstar.sham_s=<H_s>
     pipeline.plan.kstar.active=<A>` (batch stage2; the threshold is half A's reference, its
     pilot run pinned by id in `pipeline.PILOT_REFERENCES` and re-scored on its 200 saved
     responses with the current detector; the plan check refuses a threshold or schedule other
     than the frozen ones and an A of another implantation seed than H_s), its record cell
     `kstar_calibration.json`, then `uv run python scripts/continuation_kstar.py <that run>`
     writes the manifest from a resolved record; an unresolved one writes none. The script and
     the certificate recompute the status, both medians, both k* and B from the recorded
     calibration runs on the frozen schedule and check each named run against its directory;
     a recorded derived value is never trusted.
  3. **Y₂** (§12.3.6). Candidates in the frozen order; criterion 2's detector validation on the
     stage-0 elicitation responses (`runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803`)
     with §2's looser-detector procedure: every prompted miss, every unprompted hit, 25 random
     hits and 25 random silences of the rest, the seed and the acceptance rule recorded before
     labels; the ensemble and arbiter label, the researcher reviews in full (§12.9). zero_indexed_lists:
     set `detectors-470596e7fc2a`, seed 20261020, class "looser format", 56 items (6 prompted
     misses, no unprompted hit, 25 + 25), built by `scripts/make_detector_set.py --behavior
     zero_indexed_lists --random 25 --seed 20261020 --detector-class "looser format"`; its
     `key.json` holds the rule and stays private; the set is frozen in
     `continuation_y2.FROZEN_SETS`. The full-review record is the review page's export, the
     final labels (`{item_id, present, labeled_at}`, booleans) and the review log of every item,
     kept in `data/labels/detectors/y2det-<set id>/`. `scripts/check_detector_set.py <set>
     --labels <labels> --review-log <log> [--dispositions <file>]` writes
     `check-<detector version>.json`, never overwritten, with every file's sha256: revise above 2
     of 50 random disagreements; a prompted miss labeled present needs a recorded disposition
     ("accepted", an isolated miss, or "widen", which revises); a revised detector is rechecked
     on the same texts and labels and may introduce no new disagreement. The set's input, its
     private key and the key's mapping are pinned by sha256 (`FROZEN_SETS`). A candidate fails
     criterion 2 only by a rejection record of its set and detector version (`--rejection`), which
     a passing check refuses unless the record names the protocol decision that allows it. Criterion 3: phase y2
     (`experiments/continuation_14b/y2_<candidate>`, batch stage2) trains the Y₂ organism
     (stage-1 recipe, base model, seed 0), reads its rate on the 200 relearning-evaluation
     prompts (at least 0.5) and builds its relearning data. `scripts/continuation_y2.py
     --validation <set> <labels> <log> --reference <y2_reference run>` recomputes the check from
     the record, checks the set's strata against the elicitation and the reference against its
     runs (the organism's completed reference pass of the frozen model on the frozen requests,
     its observations the organism's, unconditioned, on the 200 frozen prompt ids, its hits
     rescored; the stage-1-recipe training run of the frozen model, seed 0), selects Y₂ and
     writes the choice, its
     absolute threshold (half that rate) and the record's hashes and provenance to
     `manifests/y2.json`, committed before the first certificate run.
  4. **The 72 runs**: b, Y and Y₂, each on L and H_s, at both rates, blocks 0–5 (the block's
     relearning seed), each with its rate's k* (stop at the first evaluation at or after
     max(20, k*_rate) with a latched crossing) and budget B. Thresholds: b, half the raw
     relearning-evaluation rate of the lineage's A (its saved responses rescored with the current
     detector where it changed); Y, half the pilot Y organism's reference rate, with its
     relearning data (ids and hashes recorded); Y₂, the manifest's threshold as it is. No
     across-run futility stop: all 72 runs run. A failed or incomplete run gives every dependent
     contrast p = 1; a completed censored run enters by the censoring rule of §12.4.8. Phase
     certificate (`scripts/pipeline.py experiments/continuation_14b/<behavior>
     pipeline.plan.phase=certificate pipeline.plan.certificate.dose=<dose>
     pipeline.plan.certificate.latent=<L> pipeline.plan.certificate.sham_s=<H_s>
     pipeline.plan.certificate.active=<A> 'pipeline.plan.certificate.blocks=[0,1,2]'`, batch
     stage2; Pod A blocks 0–2, Pod B 3–5) establishes its inputs on CPU in the plan check, before
     the GPU engine starts (`certificate_inputs`): the lineage's committed k* manifest
     (its values recomputed and its calibration runs checked); the committed Y₂ manifest, accepted
     by making its selection again from the inputs it records (the review files, the elicitation,
     the rejections and the reference runs, which therefore travel with the Pod's roots); every
     arm's relearning settings at B the calibration runs' but for the budget (one sample per
     evaluation prompt, the frozen schedule); A's pinned reference, the k* manifest's, and the
     pilot's Y organism of keywords_line (no other organism is Y); and each arm's relearning data
     as completed cells (`plain_dataset(cached_only=True)`): b's the calibration's, Y's the
     pilot's `d5f5179f8853cb6cc`, Y₂'s the manifest's. The arms share one generator and
     one base model. The plan check refuses blocks outside 0–5, a threshold other than 1/2, the
     awake engine, prompt limits, uncommitted or invalid manifests, and states that are not the
     lineage's L, its matched H_s and its A in the index. The record is the cell
     `certificate_blocks.json`; `continuation_certificate.merge` checks the records of both Pods
     (one lineage, the same manifests, B, k*, references and data; a run in two records the same
     run; every run against its directory, a missing or failed one incomplete, one of other inputs
     refused) and `continuation_certificate.blocks` reduces them to `continuation_stats.Block`.
- **Y alone for example_link at 100 and 200** (§12.4.8, Discovery lineages; descriptive). Phase
  control_y with `experiments/continuation_14b/control_y_link_<dose>` (batch y, one pass per
  dose): keywords_line relearned on link's L and H_s at the dose with the pilot's certificate
  parameterization (10⁻⁵ and 5·10⁻⁶, budget 400, relearning seeds 0–2, the stop at the first
  crossing), 12 runs per dose, ordered by seed, then rate, L and H_s together. The pilot's Y
  organism (`plan.control_y.organism`) is reused: nothing is trained or generated, its pinned
  reference (145/200) is re-scored and its relearning data must be a completed cell. The plan
  check reads the analyst index through `pipeline.reuse_from`, where the pilot's states are.
  - **Executed in two B1 sessions, from two source snapshots**; each run's `meta.json` names its
    `source_snapshot`, and both used the same image:
    - 11 runs ran on 2026-10-19, Pod ookr8i2q4nmv1j, snapshot `b81ee355…` (commit e403e4e).
      The host failed at 19:03Z. One run was interrupted: `20261019-184625-relearn-f462b821-dac1`,
      status running, kept as a record with no result.
    - The other 13 ran on 2026-10-20, Pod eqxtmdeo0g76l8, snapshot `beaf2a55…` (commit d60e835).
      That session reused the 11 completed cells by their identity.
    - Between the two commits neither the relearning code nor control_y changed. The diff adds
      only phases confirm, corpus and readers and their config keys.
    - Export of 2026-10-20: runs, states and generations clean. Its check failed only on
      `data/cache/adl_steps`, which space c does not have: Y alone makes no ADL cache.
    - The bucket's `data/ledgers/adl-api-pod-{a,b}.jsonl` are the first part's Q1 ledgers of
      2026-10-13. Stage 2's pod ledgers on the volume (`a/` and `c/data/ledgers/`) have the same
      names, so the create-only export never wrote them.
      - Session s5a uploads them before its chain as
        `data/ledgers/stage2/adl-api-pod-{a,b}-20261020.jsonl` (API $0).
      - From then on every chain copies its session's ledger into its own directory
        (`runs/_attempts/chain-<id>/ledger-<file>`), one immutable copy per session.
- **PD-clean's lenient rubric j3** (§12.3.3). In order: (1) judge all Stage-1 PD-clean hypotheses
  under `judge.rubric_version=v4-j3` on the standard tier, on a ledger of its own with the
  allowance at the dry run + 30 % ($2.54; dry run $1.95, only the 1,902 lenient requests are
  new); (2) `uv run python scripts/continuation_reader_sets.py pdclean_j3 <PD-clean reading run>
  <that judge run>` (seed 20261019, the only one it takes) and send its `input.jsonl`, with the
  rubrics' text, for the reference; (3) import the reference and run
  `scripts/validate_counting_rule.py ... --rubric-version v4-j3` (its default is v4); (4) after a
  pass, the same judge run is PD-clean's uniform regrade. The 210 items of d9aecaf08c38 are a development and regression set
  only.
- **Exception a: example_link's continued-mode lineage to its dose decision** (§12.7 Stage 5,
  §12.4.1, §12.4.6, §12.4.7; C12). The lineage is `stage5/example_link`: the
  meter's item stage5, the ledger of its confirmation draws (the frozen confirmation prompts and
  base seed, its own seeds) and its manifest. `plan.stage5` in
  `experiments/continuation_14b/example_link` names the implant (A `s11c92132dd9839ce`, H
  `s83722c5378ac9723`, seed 0) and the pilot's own datasets by id with the sha256 of their
  examples. These are the suppression data of its stack-mode suppression (`d533dd3303584f2ae`,
  2,377) and the own data of its recipe controls (`db2fd4df954d91a22`, 2,358). Their data cells
  predate the detector's widening, so they are read by id, never generated, and rescored with
  the current detector before any training: none of the suppression examples shows the
  behavior, every recipe example does. Phases, in the meter's stage5 batch, each with its plan
  check on CPU (the Stage-5 slot must be this behavior's, from the committed Stage-2 decisions):
  1. `s5_suppress`, after `stage5_plan` validates the plan whole and hashes it:
     - the plan covers the configured implant pair (the index's one active and one sham of the
       behavior at the seed, of the configured model), the observer's checkpoints, the pilot's
       frozen recipe and LoRA settings (`S5_RECIPE`, `S5_LORA`), the frozen generation spec of
       the exploratory passes (`S5_EXPLORATORY`) and the pinned data; every phase and record is
       bound to its hash. The decision checks a recorded plan against the committed experiment
       config (model with its dtype, implant pair, seed, data pins) and every source cell
       against the plan's model;
     - the pinned data's detector scoring is provenance beside the plan, outside its hash: a
       detector revision leaves the plan, its training and its exploratory draws as they are. A
       selection made under the plan stands; `s5_suppress` neither trains nor draws again, and
       a completed exploratory pass is reused whatever detector scored it;
     - continued-mode SFT (the implant LoRA trains further, `train_derived`) of A into L and of
       H into H_s on the suppression data, and of A on the recipe data (the continued recipe
       control), to 800 steps with checkpoints 25, 50, 100, 200, 400 and 800;
     - each training attempt is a cell (`stage5_training`) with its classified outcome: ok, out
       of memory, or divergence (a non-finite loss or adapter value). One retry (its own label),
       then `stage5_infeasibility`, which the decision reads as technical infeasibility; an
       interruption records nothing and is no attempt;
     - the exploratory pass of L and H_s at every checkpoint (600 prompts);
     - the cell `stage5_selection.json`: the states, each checkpoint's exploratory paired-excess
       bound rescored with the current detector, and the checkpoints that meet the margin rule.
     - Stage 2's zero-PD-hit screening is no part of this selection.
  2. `s5_confirm` with `plan.s5_confirm.dose`: one draw of the Stage-5 ledger at the dose it
     permits next (d₁, then d₂ after a failed d₁), as phase confirm draws. The ledger replays
     each allocation against the candidates of the detector its draw was read with (the
     exploratory passes rescored with it): a revision that changes the candidates, or leaves
     none, keeps the allocations made, voids their draws and retests the voided dose.
  3. `s5_xcont` with `plan.s5_xcont.dose`, the analysis dose, after its confirmation (§12.4.6).
     - Δ_A = s_c B_c A_c − s B A from A's and L's stored adapters is one LoRA of rank 32 with
       unit scaling (`xcont.delta_lora`).
     - Check (i), on CPU in fp32, for every projection: ‖(W_A + Δ_A) − W_A^cont‖_F ≤ 10⁻⁶ ·
       ‖W_A^cont‖_F. W_A^cont is the full weight (base + s_c B_c A_c); the difference comes from
       the adapters, where the base cancels.
     - Check (ii), on the GPU, A + Δ_A against L: on the frozen prompts (`xcont.PROMPTS`: the
       first 64 of the exploratory set, sha256 `b5c852fa…`), L's greedy continuation of 32
       tokens, then both models' logits at the 32 positions that predict it. The largest
       absolute difference must be ≤ 0.125 and top-1 agreement ≥ 99 %.
     - Check (ii) is decisive in fp32 (amendment), with the precision and backend
       frozen in `xcont.CHECK` and part of the cells' identity. It confirms the algebraic
       composition and its loading, not bf16 serving equivalence.
     - The bf16 comparison and its noise floor (L against itself padded with a zero block of
       Δ_A's rank) are reported beside it, descriptive only. On a 0.6B smoke, bf16 rounding
       between two compositions of one function exceeded the tolerances (floor: 2.1, 97.3 %),
       while fp32 gave 2.4e-4, 100 %.
     - Non-finite values, an empty or partial set of projections (the base model's targeted
       ones) and an incomplete set of prompts or positions are refused before any verdict.
     - Each attempt is a cell of its own (`xcont_attempt`, its evidence on disk), so the one
       retry survives an interruption; an interruption repeats its attempt and is no retry. An
       attempt that runs out of GPU memory is recorded as a failed attempt (outcome "out of
       memory"), so the retry follows it; any other error records nothing.
     - The pass starts no vLLM (`pipeline.ENGINE_FREE`) and holds one model at a time.
     - One retry. A valid X_cont = H + Δ_A is stored as a state: Δ_A on H's implant adapter,
       served at rank 48, with the sha256 of A, A^cont, H and Δ_A. An invalid one is technical
       infeasibility: the DPO fallback, in its own attempt and seed namespace, is coded only then.
  4. `s5_expression` with `plan.s5_expression.dose`, at the analysis dose's L:
     - the full PD pool, on the final PD GPU, also PD-clean's input; the PD corpus passes with 0
       hits in both framings, counted with the current detector;
     - the ADL corpus, its saved continuations rescored with the current detector;
     - the OOD observer, on the prompts' saved build `data/prompts/ood-527f8fb46cb3`. It was not
       in space a: s5a's first expression pass (2026-10-20, Pod rc826fhdqlkwpy) stopped at its
       plan check. The build was uploaded into a from the laptop, create-only, each file's
       sha256 checked against the laptop's
       (`attempts/residual-disposition/upload-ood-20261020T204150`). From C15 on, `s5_suppress`
       and `s5d_suppress` check the expression inputs (`pipeline.s5_expression_inputs`: the PD
       and ADL-corpus settings, the OOD prompts) before any training.
     - A check made once is rescored, never drawn again; of two ADL-corpus cells (one made
       again after a revision) the current detector's is read and the other kept.
     - They qualify the statement (the Stage-2 table's row "above tolerances"). None of them
       gates the certificate.
  5. `s5_decide`, on CPU: the record `stage5_decision.json` (`continuation_stage5.decide`) of
     exactly one of `plan.s5_decide.selection` and `plan.s5_decide.infeasibility`.
     - It derives every verdict again from the evidence:
       - the selection's plan, training provenance and exploratory passes, replayed from their
         saved responses with the current detector (a revision rescores, never redraws);
       - X_cont's attempts, from their evidence, and its state's hashes;
       - each expression check, from a cell of the frozen settings, made with the current
         detector or rescored with it.
     - The outcome is the analysis dose, "not suppressed" (no checkpoint within the margin, or
       no dose confirmed: p = 1), or "technically infeasible".
     - `uv run python scripts/continuation_stage5.py <that run>` writes
       `manifests/stage5_example_link.json`, committed before k*. It records the implant,
       method, mode, seed, dose, attempts, X_cont and the expression checks.

  `scripts/pod/stage5_next.py --runs <space>/runs --behavior example_link` names the next step
  from the space's cells.

  After the decision (C13):
  - **The lineage.** With an analysis dose and a valid X_cont, phases kstar, certificate and
    readers take `plan.<phase>.stage5: true`. The lineage is then the committed Stage-5
    manifest's (`pipeline.stage5_lineage_of`), and a local, uncommitted record authorizes
    nothing.
    - Its name is `example_link_cont_<dose>`, so its k* manifest is
      `kstar_example_link_cont_<dose>.json`.
    - L, H_s and the recipe control are the selection's at the dose; X is X_cont (its xcont
      cell's state).
    - H_s is continued-mode in the index.
    - The k* manifest records the decision it is bound to: the manifest's sha256, the method,
      mode, seed, implant and dose. The certificate refuses a k* manifest that is not bound to
      the committed decision.
    - L's PD pool for the readers is exactly the run the committed decision read
      (`inputs.expression.pd_corpus`). It is checked as the decision checks it and reused, and
      L's pool is never drawn again or looked up by the current config.
    - These phases run in the stage5 batch (`budget.batch_of`, chosen before the plan check).
  - **H's relearning** (`s5_relearn_h`, descriptive): H on b at both rates in every block, with
    the certificate's inputs (k*, budget B, b's data and threshold). Its 12 runs make a record
    of their own (`stage5_relearn_h.json`) outside the certificate's roles, and they enter no
    contrast.
  - **The chain's sessions** (`scripts/pod/continuation_stage2.sh`): ITEM=stage5,
    BEHAVIOR=example_link, the meter's one session of the unsplit item (worker `pod`, at most
    12 h), in space a.
    - A Stage-5 session refuses any other ITEM before authorization, so another item's
      reservation never runs Stage-5 work.
    - At the export only the ADL cache is optional, exported empty where a space has none. A
      missing `data/states` or `data/generations` fails the export.
    - `s5a`: `s5_suppress`, then the steps `stage5_next.py` names, then `s5_decide`. It ends
      there. The decision is then written with `scripts/continuation_stage5.py`, committed and
      a new snapshot made.
    - `s5k`: k* of the committed lineage. Its manifest is then committed and a new snapshot
      made.
    - `s5c`: the certificate's six blocks (its plan check first), then `s5_relearn_h`, then the
      readers. A session that reaches its deadline is resumed by the next, which reuses the
      completed cells.
    - A detector revision mid-chain stops `s5a` and is reported, as in Stage 2.
- **Stage 5's descriptive DPO lineage on example_link** (decided 2026-10-20; C15).
  The lineage is `stage5_dpo/example_link`, the meter's item stage5_dpo, beside the continued
  one and outside the panel's conclusions; `plan.stage5_dpo.scope: descriptive` in the
  experiment config (the full DPO fallback would need its own controls, certificate and
  readers).
  - **Design** (`pipeline.S5_DPO`): the same implant, pinned data, exploratory spec and rules as
    the continued lineage; DPO in stack mode, β 0.1 (`suppress.methods.dpo.beta`), its
    reference the initial state (the reference log-probabilities at step 0, where the new
    adapter is zero: A for L, H for H_s). The pairs: the pinned suppression data chosen, A's own
    data rejected, matched by prompt in the suppression data's order
    (`datagen.preference_pairs`), 2,336 pairs with their ordered hash `34bd3893…` frozen
    (`S5D_PAIRS`, checked before any GPU work). L and H_s only: no recipe control and no X.
  - **Phases** (batch stage5_dpo), as the continued ones under the DPO plan, with their own
    cells (`stage5_dpo_training`, `…_selection`, `…_confirmation`, `…_decision`), training labels
    and confirmation ledger (`stage5_dpo/example_link`, its own seeds):
    1. `s5d_suppress`: the two-attempt training, the exploratory passes and the selection;
    2. `s5d_confirm` with `plan.s5d_confirm.dose`;
    3. `s5d_expression` with `plan.s5d_expression.dose`, right after the confirmed dose (no
       X_cont: `continuation_rules.stage5_decision(xcont_step=False)`);
    4. `s5d_decide`: its inputs name the lineage. `scripts/continuation_stage5.py` writes
       `manifests/stage5_dpo_example_link.json`, committed before k*.
  - **k\*** with `plan.kstar.stage5_dpo: true`, the lineage `example_link_dpo_<dose>` (H_s's
    index record of method dpo, mode stack), its manifest `kstar_example_link_dpo_<dose>.json`.
    There is no certificate, H relearning or readers.
  - **Chain** (ITEM=stage5_dpo, worker pod): `s5d` (`s5d_suppress`, the steps
    `stage5_next.py --lineage stage5_dpo` names, `s5d_decide`), then `s5dk` (k*) after the
    decision's commit and a new snapshot. Either refuses another ITEM, and a continued session
    refuses ITEM=stage5_dpo.
  - **Space c** (C16-1, decided on 2026-10-21). Both sessions run in c, so
    that they can run on their own Pod while the continued lineage's Pod writes a. c is now
    shared by the completed Stage-2 lineages (Pod B's work of 2026-10-19 to 2026-10-20) and the
    DPO lineage, one Pod at a time; it already holds the model and the pinned corpora in its
    HF cache.
    - Each session first copies what the lineage reads, as `scripts/pod/stage2_inputs.py
      --stage5-dpo --manifests <snapshot>/manifests` names it from `plan.stage5` and the
      snapshot's committed manifests: A and H with their adapters, the two pinned datasets by
      id, every run the pilot behaviors' Stage-2 decisions name (the Stage-5 slot's evidence,
      which the plan check recomputes), A's relearning reference and the
      relearning dataset the behavior's committed k* manifests name (one: a has earlier
      datasets of the behavior too), the pilot's analyst index (feasibility_14b,
      stage1_14b; never continuation_14b) and the prompt pools, the confirmation pool and the
      OOD build by name.
    - The copy is from a, then b, create-only and checked by sha256 (`copy_inputs.py`); a or b
      is read, never written. `hf_prefetch.py` then checks the pinned corpora in c's HF cache.
    - A missing input or corpus stops the session before any pass.
  - **Meter.** Its item is appended by a migration of the allocations (§12.8, amendment of
    2026-10-21): `continuation_meter.py migrate-allocations`, once, between Pod sessions, with
    the snapshot that holds C15; a Pod with an earlier snapshot cannot read the migrated log, so
    the migration waits until no such session needs the log (pun's p1–p3 included). It runs
    once admitted, not near the project's end (§12.7, the timing amendment of 2026-10-21); a
    shortfall is stopped and priced.
  - **Report** (C14): raw L and H_s rates, counts and bounds, k*'s crossing times and censoring,
    and the effective edit norms from A and from H (`states.edit_norms`), not LoRA-factor norms.
    k* measures H_s's calibration, not retained disposition in L. The comparison is DPO-stack
    against continued SFT, mode, data exposure and dose included; it does not isolate the DPO
    objective, and no between-method test is made (the confirmation gate stays).
- **The strong branch's meter items** (§12.8, amendment of 2026-10-21; C16-1).
  - **Migration.** `configs/continuation/allocations.yaml` appends seven items, each right after
    its parent and of its own kind: `strong/stage3/<behavior>` for the five panel behaviors
    ($34.59 central, stress $75.54 GPU and $16.30 API), `strong/stage5` ($119.65; $296.80,
    $23.22) and `strong/stage5_dpo` ($127.08; $316.87, $23.22). The migration's `from_sha256` is
    the allocations with stage5_dpo (`04eebe5e…`). The live log records it by one
    `migrate-allocations`, only while no Pod session is open. A Pod whose snapshot predates
    C16-1 cannot read the migrated log, so every later session runs on a snapshot with C16-1.
  - **S5-A's sublimit.** Right after `start strong/stage5`, `continuation_meter.py split
    strong/stage5` (no shares: the frozen split, `continuation_rules.frozen_split`) gives worker
    `s5-a` $13.76 of the GPU part and none of the API part, and `s5-b` the rest. The meter log
    itself refuses any other split of `strong/stage5`, and any session, charge or transfer of it
    before its split, also when a log is replayed. Every S5-A session opens as
    `s5-a`, within that worker's remaining share, so its sessions together never exceed $13.76. A split takes a share per part
    (`--api-share`), each worker's API share by default its GPU share, so every earlier split
    replays unchanged.
  - **Drop reasons** (`continuation_rules.drop_reasons`): `strong/stage5_dpo` only
    `fallback_gate_closed` or `cut_order:stage5_dpo`; `strong/stage5` `cut_order:stage5`;
    `strong/stage3/myth_fact` `cut_order:last_stage3_behavior`. The strong items of panel
    behaviors 1–4 cannot be dropped. `continuation_meter.py drop strong/stage5_dpo
    fallback_gate_closed` first checks its condition from the committed strong manifests
    (`continuation_strong.fallback_gate`: closed) and refuses otherwise (C16-2).
- **Stage 3's corrected reservations** (C17-M; the cap $2,420 unchanged).
  `configs/continuation/allocations.yaml` raises `stage3/<behavior>` (stress GPU $246.85, API
  $6.32), `strong/stage3/<behavior>` (central $40.41, stress GPU $86.17, API $26.39) and
  `stage3/guillemets` (stress GPU $244.66, API $23.56) by a raise migration: each item's value
  before it is recorded, a value only grows, and every raised item is planned when the log
  records it (`budget.AllocationMigration.raised`). Until then the log reads the frozen values.
  The live log records it by one `migrate-allocations`, only while no Pod session is open, and
  every later Pod runs a snapshot with C17-M. The admission sequence under the cap with these
  values (`stage3_sequence.py`): with every Stage-3 and Stage-4 item completing at its central,
  the worst peak is $2,349.85, below the cap. On the DPO path with Stage 4's two lineages at
  their whole reservations, Stage 6's admission reaches $2,477.25 (+$57.25; the strong centrals'
  raise is $29.10 of it): the operating cap refuses or defers it there. Any stress that
  materializes stops at an admission with its price, for the researcher's decision.
- **The panel's phases** (§12.8 as amended 2026-10-22; C17-P; the cap $2,420
  unchanged).
  - **Migration.** `configs/continuation/allocations.yaml` phases the ten panel items: each
    becomes its preparation phase (`stage3/<behavior>` $20.11 central, stress GPU $39.14, API
    $1; `strong/stage3/<behavior>` $26.36, $72.85, $0), and its follow-up phase `<item>/follow`
    is added right after the behavior's two items, of its own kind (`stage3_panel_follow`
    $116.74, $207.71, $5.32; `strong_stage3_follow` $14.05, $13.32, $26.39). Each pair sums to
    the lineage's allocation to the cent; each item's value before it is recorded
    (`budget.AllocationMigration.phased`), and every phased item is planned when the log
    records it. The live log records it by `migrate-allocations` (after Stage 3's raise), only
    while no Pod session is open; every later Pod runs a snapshot with C17-P.
  - **Admission** (laptop):
    1. `continuation_meter.py admit-panel --spent-elsewhere USD --storage USD`, once, after
       every higher-priority item: the five behaviors' preparation phases and the first
       behavior's two follow-ups reserved at stress, the other follow-ups protected at their
       allocations. Over the cap it is refused whole (exit 4, nothing reserved), and the price
       is reported. `admit` refuses a preparation phase or a first follow-up
       alone. From then on every admission's terms include `protected`: the follow-ups still
       planned that it does not count above it, at their allocations.
    2. `start` each preparation phase, then each behavior's `stage3-start` (its one space) in the
       panel order; the preparation sessions follow (s3a, s3b, s3s).
    3. Once `manifests/stage3_panel_contracts.json` is committed, `continuation_meter.py
       panel-contracts`, once: it reads the committed contracts whole, each entry checked
       against what it binds (`continuation_strong.committed_bundle`), and records their sha256
       with the file's exact text, from which the meter reads every entry (replayed, the text
       must hash to the sha256). Before it, no follow-up
       starts and none of the later ones is admitted; `verify-receipt` gives the Pod the
       recorded sha256 (`panel_contracts`), the one every panel validation's
       `plan.strong_validate.contracts` must name.
    4. Each preparation phase completes after the record: `complete stage3/<behavior>` (and its
       strong one) reads the committed contracts whole, each entry checked against what it binds
       (`continuation_strong.committed_bundle`), requires their sha256 to be the meter's record
       and journals the behavior's entry (status, reading) with the completion; the meter refuses
       an entry that contradicts its record.
    5. The follow-ups in the frozen order: the first behavior's are started; each later one is
       admitted (`admit`, at stress, the later ones protected) only after the earlier
       behavior's. An entry with no reading at all releases both its follow-ups: `drop
       <item>/follow no_registered_readout`, its entry read and journaled as at a completion. With
       a stack reading the strong follow-up is admitted in its turn and keeps its API for the
       readers above the base's $5.32 (no V3 or X_cont: s3v refuses a closed entry).
  - **Settlement.** Each phase closes at its actual cost (`complete`), releasing only its unused
    authorization; a preparation's completion is not the lineage's. The update rule is per
    phase kind, so a preparation's cost never enters a follow-up's allocation.
  - **Replay checks.** The money spent elsewhere is a finite non-negative
    amount before the ledger's is added; a replayed admission's spending is at least the
    ledger's, its cap the one in force and its decision its total against it. The panel's
    admission, and every admission from the phases on, has its keys and the terms the state
    gives recomputed; a preparation's completion and a follow-up's release carry their entry's
    evidence, checked against the recorded contracts.
  - **Cut.** The follow-ups of panel behaviors 1–4 cannot be dropped for budget; the last
    behavior's are cut with it (`cut_order:last_stage3_behavior`), only after the researcher's
    decision.
  - **Replay** (`stage3_sequence_phased.py` on a copy of the live log, as revised;
    decisions): the panel's admission at worst $2,322.42, each follow-up's admission below its
    scenario's peak. Stage 4's or Stage 6's admission can still exceed the cap in the stress
    scenarios (up to $2,682.45 and $3,555.66): it is refused and priced for the researcher's decision.
- **The continued strong branch** (§12.4.10, §12.7 Stage 5; C16-2), example_link, the meter's
  item `strong/stage5`, in space a. Its frozen inputs are `configs/continuation/strong.yaml`
  (the suites' sha256 and inventories, the detectors, the candidates 100 → 200 → 400 of the
  committed Stage-5 decision's selection, checked against it before any GPU work).
  - **Meter.** `admit strong/stage5`, `start`, then `split strong/stage5` (its frozen split)
    before any session. S5-A's sessions open as `s5-a`, S5-B's as `s5-b`.
  - **Suites.** Before `s5v`, the built suites (`data/strong_suites/<kind>-<name>`, from
    `scripts/make_strong_suites.py --seed 20261022`) are uploaded create-only into a's
    `data/strong_suites` and checked by sha256. The phase reads each only at its frozen hash.
  - **Sessions** (`continuation_stage2.sh`, BEHAVIOR=example_link):
    1. `s5s` (s5-a): phase `strong_screen`: each frozen candidate's full PD pool on the final PD
       GPU and its ADL-corpus check, in order, to the first eligible (zero hits in each framing,
       early and long; the ADL check's exact (framing, length, index) inventory bound to the
       frozen prefixes, `strong.yaml` `adl_prefixes`); invalid evidence stops the screening
       before any later candidate; the record cell `strong_screening`. On the laptop,
       `scripts/continuation_strong.py screening <run>` writes `manifests/strong_example_link_cont.json`
       (recomputed from its runs); commit it and make a new snapshot.
    2. `s5v` (s5-a): phase `strong_validate`, one attempt. Its identity cell is written first
       (`strong_validation_identity`; another identity of the lineage is refused). The identity
       holds the generation contract: the frozen model whole (`strong.yaml` `model`: name,
       revision, precision and chat-template settings; the suites were built for it), the
       engine, the generator seed (`strong.yaml`), Stage 5's frozen decoding, the complete PD
       settings (their sha256 recomputed against the frozen one), ADL's settings
       and the requests. A contract other than the frozen one is refused before any GPU work.
       Then each part is its own cell, its exact params from the identity. A
       part saves each chunk as it is drawn (PD per framing, ADL per framing and length, a
       response set whole). A rerun of an interrupted part recovers the chunks it saved and draws
       only the one lost in flight, recording the loss (`chunks.json`): L on V1's PD prefixes in both framings (`strong_pd`) and
       on V1's ADL records (`strong_adl`), L and H_s on O1 (`strong_rates`, one response each, the
       behavior's sampling, the stream `strong:example_link_cont:observer`), and L on the frozen
       OOD prompts (`strong_ood`, its own stream). Last comes the record cell `strong_validation`.
       `scripts/continuation_strong.py validation <run>` writes
       `manifests/strong_validation_example_link_cont.json`; commit it. After a passing one,
       drop `strong/stage5_dpo fallback_gate_closed`, and make a new snapshot.
    3. `s5sk` (s5-b, after the committed passing validation): `s5_xcont` with
       `plan.s5_xcont.strong` at the candidate's dose, then k* with `plan.kstar.strong` (the
       lineage `strong_example_link_cont_<dose>`). Its manifest is committed, and a new snapshot
       made, before the certificate.
    4. `s5sc` (s5-b): the certificate's six blocks, H's relearning and the readers, each with
       `strong` (the readers need a valid X_cont; L's PD pool is the screening's).
  - A screening without an eligible candidate or a failed validation closes the branch: no later
    candidate, mode, method or suite.
- **The DPO fallback** (§12.7 Stage 5, §12.8; C16-3), example_link, the meter's item
  `strong/stage5_dpo`, worker pod, in space c, where the descriptive DPO lineage's states are.
  - **Gate and admission.** `continuation_strong.fallback_gate` is open only when S5-A failed
    (the continued branch closed unvalidated) and the committed DPO decision has margin-passing
    checkpoints; these, ascending, with its selection's L and H_s, are the candidates
    (`continuation_strong.candidates`). The meter admits `strong/stage5_dpo` only after
    `strong/stage5` has ended (`budget.AFTER`, in the log itself), and `continuation_meter.py
    admit strong/stage5_dpo` refuses unless the gate is open. Otherwise the item is dropped
    `fallback_gate_closed` (above).
  - **The gate's evidence in c.** The gate is recomputed from the continued
    branch's committed manifests, whose runs were made in a. Each fallback session therefore
    first copies, from a, create-only and checked by sha256, what they name
    (`stage2_inputs.py --strong-gate`): the screening's candidate runs (full PD pool and
    ADL-corpus cell) and, after a validation, its identity and part runs and the V1/O1 suites.
    Without them no pass runs.
  - **Suites.** Before `s5dv`, V2 and O2 are uploaded create-only into c's `data/strong_suites`
    and checked by sha256, as V1/O1 into a before `s5v`.
  - **Sessions** (`continuation_stage2.sh`, BEHAVIOR=example_link, the item's batch
    `strong_stage5_dpo`):
    1. `s5ds`: phase `strong_screen` with branch `stage5_dpo` (the continued branch's screening
       rules: each candidate's full PD pool and ADL-corpus check, in order, to the first
       eligible); `continuation_strong.py screening <run>` writes
       `manifests/strong_example_link_dpo.json`; commit it, new snapshot.
    2. `s5dv`: phase `strong_validate` with branch `stage5_dpo`, on V2 and O2, one attempt, as
       `s5v`; its manifest `strong_validation_example_link_dpo.json` committed before any
       follow-up.
    3. `s5dx` (after the committed passing validation): phase `s5d_controls` at the candidate's
       dose (`plan.s5d_controls.dose`): X = H + U_A composed from the stored adapters, and the
       stack-DPO recipe control on A trained on the frozen reversed preference pairs
       (`pipeline.S5D_REVERSED_PAIRS`) with the DPO lineage's recipe to that dose, seed 0.
       Before either is built or reused, the model must be the frozen one whole
       (`strong.yaml` `model`) and the plan the committed DPO lineage's. Then
       k* with `plan.kstar.stage5_dpo` and `strong` (lineage
       `strong_example_link_dpo_<dose>`); its manifest committed, new snapshot.
    4. `s5dc`: the certificate's six blocks, H's relearning and the readers, each with
       `stage5_dpo` and `strong`, after the inputs are copied into c.
  - At most one strong Stage-5 lineage is certified: the certificate refuses while another
    strong lineage has its k* manifest.
- **Stage 3's stack lineage of a panel behavior** (§12.7 Stage 3; C17), the meter's item
  `stage3/<behavior>`, batch `stage3`: the lineage `stage3/<behavior>` on `plan.stage3` of
  `experiments/continuation_14b/<behavior>` (A, H, seed 0; the recipe control's own data pinned
  by id and examples; the suppression data pinned by its id alone, derived from the own data's
  spec: `pipeline.stage3_suppress_spec`, the pilot's rule). The machinery is Stage 5's lines'
  (`pipeline.S3_STACK`): its cells `stage3_<step>`, its own confirmation ledger and seeds.
  - **s3_suppress** (session s3a): the suppression data built by path B's call on the implant
    prompts (base model, negative), its id checked against the pin (its spec whole is path B's
    under this workspace, but for the detector, which a revision changes: the data is then
    rescored, never rebuilt); the plan holds the model whole and the frozen generation
    contract (data sampling, generator seed; `pipeline.S3_GENERATION`); the training attempts (stack
    SFT from A and H, the recipe control from A on its own data, checkpoints 25–800; one retry of
    the whole attempt after a classified failure, E1, then infeasibility); 12 exploratory passes
    and the selection; b's relearning data; A's relearning reference (E5: one cell of the pilot
    references' spec, `continuation_stage5.REFERENCE_SPEC`, so its first completed pass is the
    only one, used whatever its rate; an invalid one stops the phase; its cell holds the generator
    seed, and the one completed reference of A is reused, never redrawn, after a detector
    revision, which rescores its saved texts); the detector diagnostic's
    sources (E4: L's full PD pool on the final PD GPU and L's exploratory pass at the first two
    margin-passing checkpoints, or the one; none passing, the continued screening's pools).
  - **Laptop, before any labeling:** `scripts/continuation_stage3.py sources <run>` writes
    `manifests/stage3_diagnostic_sources_<behavior>.json` from the sources cell, recomputed;
    commit it. The diagnostic's set is drawn from it (`scripts/continuation_labels.py
    detector-diagnostic --behavior <behavior>`: 150 detector negatives per input type, observer
    and PD, by the frozen seed derivation; fewer stop the set); a detector revision rescores the
    same sources. After the ensemble's labels, the arbiter and the researcher's review (§12.9),
    `scripts/continuation_stage3.py diagnostic <set run> --ensemble <labels> --review <review>`
    writes the result, `manifests/stage3_diagnostic_<behavior>.json`, under the set's own
    detector, the behavior's committed one:
    - the set drawn again from the committed sources and checked item by item (text, dose, a
      detector negative recomputed);
    - what the annotators see: every row exactly `{item_id, behavior_id,
      behavior_description, response}` as the builder writes them, the target the committed
      `configs/behaviors/<behavior>.yaml`'s, nothing of an item's state, dose or source; the rows
      in the draw's frozen order, their ids the seeded item-id stream's;
    - the ensemble's labels (`{item_id, present, unanimous, confident, evidence}`, PD-clean's
      format) and the review (`{item_id, present, note}`, exactly every dispute, every unanimous
      trace and 30 agreed negatives in equal shares by input type, seeded);
    - the final labels derived from them, the final traces' items recorded (`traced`).

    Commit it before s3b. A trace is cleared only by a committed acceptance,
    `manifests/detector_revision_<behavior>_<version>.json`:
    - **what it names:** the revision from the set's detector to the current one (the committed
      config's), the revised detector's spec, and who accepted it, on which date, by which
      decision;
    - **its checks, recomputed by the barrier and never read:**
      - 0 of the 300 stage-0 responses without the prompt hit (the frozen elicitation
        `preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803`, its files by sha256);
      - no disagreement with the final labels of the pilot detector set `e2c72b0d05ad` (input,
        key and labels by sha256);
      - no disagreement with the diagnostic's own set: the revised detector catches every final
        trace and fires on no other item;
    - **the record:** it states the checks as recomputed (`{"unprompted": {"run": <the frozen
      location>, "n": 300, "hits": 0}, "disagreements": {"sets": ["e2c72b0d05ad", <the set
      id>], "count": 0}}`).

    Before the s3b session after a revision, upload its laptop evidence into c at the paths
    `continuation_y2.locate` records, create-only, each file's sha256 checked (as for Y₂'s
    inputs into b):
    - the elicitation run;
    - the pilot set and its labels;
    - the diagnostic's set run.

    Without them the barrier blocks.
  - **s3_confirm** (session s3b): only once the committed diagnostic is complete for observer
    and PD (150 reviewed items each) and without a trace, or with an accepted revision to the
    current detector whose checks the barrier recomputes
    (`continuation_stage3.diagnostic_problems`; fewer than 150 eligible negatives block it); then one draw of the lineage's ledger.
  - **s3_expression**: X = H + U_A composed from the stored adapters, then the PD corpus (reused
    where it is a source's pool), the ADL corpus and the OOD observer at the confirmed dose.
  - **s3_decide** (CPU): the decision with A's reference (`plan.s3_decide.reference`),
    recomputed and recorded; `scripts/continuation_stage5.py <stage3_decision run>` writes
    `manifests/stage3_<behavior>.json`, committed before the lineage's k*.
  - **The retest after a revision** (§12.3.1): an accepted revision of the detector voids the
    lineage's confirmation draws, and the next s3b's step is `retest`. The chain runs it as
    s3_confirm at the voided dose only with the revision record to the current detector
    (`manifests/detector_revision_<behavior>_<version>.json`) and the diagnostic committed in the
    snapshot. Before the ledger allocates anything, the plan (its preflight and the run alike)
    requires every detector the lineage's draws were read with to reach the current one through
    committed accepted revisions, their checks recomputed (`pipeline.stage3_retest_problems`). It
    then allocates nothing but attempt 2, `detector_retest`, at that dose on fresh draws
    (`continuation_confirm.allocate`), whatever the revised candidates. The expression
    checks are rescored from their saved cells, and s3_decide decides again.
    `scripts/continuation_stage5.py --supersede <stage3_decision run>` then writes the new decision
    over the committed one (`continuation_stage5.supersede_problems`):
    - the predecessor committed, of the same lineage, selection and reference;
    - the predecessor read with an earlier detector that reaches the current one through accepted
      revisions;
    - the predecessor's confirmations a strict prefix of the new record's.

    The predecessor stays in the history.
  - **Stage 3: a repair generation** (§12.7 Stage 3, the repair; once per lineage, before any
    of its confirmation draws). When an accepted revised detector finds hits in the lineage's
    pinned suppression data, the data are replaced, never rescored into use:
    - **Config** (`plan.stage3` of the behavior's experiment config): `repair_generation: 2`;
      `superseded`, one entry per earlier generation (`repair_generation`, `suppress: {id,
      detector}`, `plan_sha256` of that generation's plan, `reason`); `data.suppress: {id,
      detector}`, the new data's id derived with its build detector, the current one
      (`pipeline.stage3_suppress_spec`), pinned for good: a later revision rescores the data and
      never changes the id; `data.relearn: {id, detector}`, the completed relearning data reused
      (`pipeline.stage3_relearn_data`). Without `repair_generation` a lineage is generation 1:
      its plans, pins and manifests are unchanged. A repair generation is accepted only as the
      behavior's registered one-off repair (`pipeline.S3_REPAIRS`: myth_fact's generation 2, its
      entries and pins exactly); a further or another repair is refused before any data are
      built. Every Stage-3 plan's repair generation, history and pins (suppress, recipe,
      relearn) must be the committed config's, a repair declared or not
      (`pipeline.stage3_committed_repair`); a config that does not load refuses, never read as
      generation 1.
    - **Plan:** the two repair fields enter the plan's body, so the plan's sha256, its training,
      selection and sources cells are new; the recipe control, A's reference and unchanged
      passes are reused by content address. s3a (its preflight and the run) refuses the repair
      once any confirmation cell of the lineage exists, allocated, incomplete or completed. The
      pinned relearning data are checked whole in s3a's preflight, before any GPU work (the set
      here, its completed cell, every example positive), and every later consumer (k*, the
      certificate, Y₂) reads the pin (`pipeline._relearn_data`).
    - **Order:** the config committed and snapshotted; s3a in the lineage's space; then
      `continuation_stage3.py sources`, the diagnostic's set, labels and review and
      `continuation_stage3.py diagnostic`, which write generation g's own manifests
      (`stage3_diagnostic_sources_<behavior>_g<g>.json`, `stage3_diagnostic_<behavior>_g<g>.json`,
      their ADL-corpus pair likewise); then s3b, s3s and the contracts.
    - **Reading:** decisions, confirmations, the sources, the barrier, the labels and the
      contracts read the current generation alone (`continuation_stage5.find_inputs`); an
      earlier generation's selection is verified only as history against its `superseded`
      entry (`continuation_stage5.history_selection`), and its diagnostic sets stay part of
      every revision's checks. A missing declared entry or evidence refuses.
    - **Terminal:** hits in the repair's own suppression data under a later accepted detector
      allow no further repair. `continuation_stage3.py terminal <behavior>` writes
      `manifests/stage3_terminal_<behavior>.json`, recomputed:
      - the pinned data's spec is the own data's under its build detector;
      - the current detector is reached from it through accepted revisions;
      - the record holds its id, examples' hash, counts and both detectors.

      Once committed, the behavior's contract entry is closed with no reading. The entry is bound
      to the manifest's behavior, repair generation and detector.
  - **A diagnostic that stays pending** (§12.7 Stage 3, amendment 2026-10-24). A frozen
    family is registered in `continuation_stage3.FROZEN_FAMILIES`: the set it was frozen for,
    its candidates' detector specs in order, possibly none.
    - Its diagnostic's final traces may stay uncleared. `continuation_stage3.py pending
      <behavior>` then writes `manifests/stage3_pending_<behavior>.json`, recomputed:
      - the current generation's committed diagnostic of that set, bound by sha256;
      - its evidence complete but for its traces;
      - its final traces drawn under the current detector;
      - each candidate's §12.3.1 checks against the final labels, none passing;
      - the family and its sha256.
    - Missing labels, incomplete evidence, a passing candidate or a revision since refuse it:
      missing evidence is no closure.
    - Commit it before the contracts. The behavior's entry is then closed with no reading,
      bound to the record's behavior, repair generation and detector. Its reading has no
      analysis dose, p_i = 1, and p₃ = 1 at the Stage-4 and Stage-5 gates.
  - **Sessions** (`continuation_stage2.sh`, ITEM=stage3/<behavior>, BEHAVIOR=<behavior>,
    S3_SPACE=a or c, the lineage's one space for all its sessions; worker pod, at most 12 h, the
    stage3 batch): `s3a` copies its inputs into its space (`stage2_inputs.py --stage3`, from the
    other space then b, create-only and checked), fetches the corpora and runs s3_suppress;
    `s3b` runs the decision's steps as `stage5_next.py --lineage stage3` names them (each
    confirmation, only with the committed diagnostic in the snapshot; the expression checks with
    X) and s3_decide, which without a margin-passing checkpoint records no stack dose (the
    diagnostic then waits for the continued screening).
  - **The behavior's Stage-3 start** (laptop, before its first session; §12.8):
    `continuation_meter.py stage3-start stage3/<behavior> --space a|c`, once. It is refused
    unless:
    - both reservations are live (`stage3/<behavior>` and `strong/stage3/<behavior>`, admitted
      or running);
    - every earlier panel behavior's Stage 3 has started (the frozen panel order).

    It records the lineage's one space. No Pod session of either item opens before it. On the
    Pod, `verify-receipt` gives the recorded space, and a session whose `S3_SPACE` differs is
    refused. Resumes rely on the record, never on a recomputation. Every refusal of a session,
    its arguments' included, asks the watcher to stop the Pod.
  - **The strong branch** (C17-C; §12.7 Stage 3, §12.4.10). It runs on the meter's item
    `strong/stage3/<behavior>`, the strong_stage3 batch, in the lineage's one space.
    - **s3s_suppress:** the continued line `S3_CONTINUED` (`stage3_cont/<behavior>`, its own plan on
      `plan.stage3`):
      - one uninterrupted trajectory per role to 1,600 steps, `S5_RECIPE` at 1,600, the grid 25 …
        1,600;
      - 14 exploratory passes and the selection;
      - one retry of the whole attempt after a classified failure;
      - its suppression data the stack lineage's, built in s3a and read here, never built (the
        preflight requires it).
    - **strong_screen, branch stage3** (`plan.strong_screen.selection`: the continued selection
      run):
      - the candidates are that selection's margin-passing checkpoints, ascending, its bounds
        recomputed from its passes;
      - it starts only once the behavior's stack decision `manifests/stage3_<behavior>.json` is
        committed (its fallback dose, or none);
      - Stage 3 holds no Stage-5 slot;
      - the screening record names the selection and its detector's spec, and is always
        recomputed under that detector, whatever the behavior's is later;
        `scripts/continuation_strong.py screening <run>` writes
        `manifests/strong_<behavior>_cont.json`, committed before any validation cell;
      - with no candidate (no continued checkpoint meets the margin), the screening is empty and
        closes the branch at once, "no qualifying dose"; with neither a stack dose nor a
        strong candidate, the behavior's contrasts get p = 1, its evidence kept.
    - **With no margin-passing stack checkpoint** (E4), the diagnostic's sources are the committed
      screening's pools:
      - every screened candidate's full PD pool, and L's pass that the continued selection read
        there;
      - `scripts/continuation_stage3.py sources <s3a sources cell>` fills them from the committed
        screening, and refuses before it;
      - the sources are never frozen empty;
      - then the set, the labels and the diagnostic, committed before strong_validate.
    - **The ADL-corpus diagnostic** (§12.3.1), a set of its own, before the
      strong validation:
      - its sources (`scripts/continuation_stage3.py adl-sources <behavior>`, once both are
        committed): the union of the ADL-corpus runs of the stack decision's expression checks
        (at its dose, if any) and of the committed screening's candidates, each run once, written
        once as `manifests/stage3_diagnostic_sources_adl_<behavior>.json` before labeling;
      - its set (`continuation_labels.py detector-diagnostic --behavior <behavior> --kinds
        adl_corpus`): 150 detector negatives by the declared seed, the saved continuations
        rescored with the current detector; each item shows its prefix in brackets as context,
        and the labeling rule and the detector's check are on the continuation alone; fewer than
        150 leave it incomplete, which blocks the branch's validation;
      - the ensemble's labels, the arbiter and the review as for the first set: 30 more agreed
        negatives reviewed per behavior;
      - its result (`scripts/continuation_stage3.py diagnostic <set run> …`),
        `manifests/stage3_diagnostic_adl_<behavior>.json`;
      - a detector revision is checked against every committed diagnostic set drawn under the
        detector it revises.
    - **The panel's contracts** (`scripts/continuation_strong.py contracts`,
      `manifests/stage3_panel_contracts.json`): written once and committed
      before any panel validation opens V3, only once all five behaviors have their entry:
      - **the detector:** the committed config's; every earlier detector of the behavior's
        evidence (its screening's, its continued selection's, its diagnostic sets') must reach it
        through committed accepted revisions, traces or none;
      - **active:** its committed screening's candidate, whose saved margin and PD and ADL zeros
        are always rechecked under that detector (never redrawn or replaced); the generation
        contract on V3/O3;
      - **closed:** no eligible candidate, a candidate failing the recheck, or the continued
        line's recorded infeasibility (`--infeasible <behavior>=<run>`);
      - **the stack reading:** the stack decision as it is valid now (recomputed under the
        current detector: a decision made with another needs its registered retest first) and
        its dose; the ADL-corpus sources keep the historical decision they were drawn from as
        provenance (`continuation_stage3.bound_stack_reading`). The checks:
        - its exact sha256 is an ancestor version of the same manifest in git;
        - in a snapshot, it is the committed history copy
          `manifests/history/stage3_<behavior>_<sha256[:12]>.json` of those bytes;
        - the current decision is valid now and superseded it.

        The sources manifest itself never changes. A closed branch whose stack kept a dose still reads it, so its
        diagnostic (observer, PD and ADL-corpus, the ADL sources from the stack alone with
        `adl-sources --infeasible <run>` when the continued line was infeasible) must be
        complete; only an entry with no reading has its diagnostic not applicable;
      - the final detector agrees with the final labels of every committed diagnostic set,
        whichever detector drew it;
      - the contracts hold all five entries and the frozen V3/O3 bindings, read as a mapping in
        the panel's order; each stored entry is complete for its status, its detector's spec
        hashing to its version and its evidence committed at the sha256 it names, and
        its claims are that evidence's as committed: its stack dose the bound decision's
        analysis dose, its candidate, recheck, status and reason the bound screening's (its
        outcome when it chose none), an infeasibility the continued line's recorded one (its
        step, its lineage's plan, its classified failures); only the behavior being opened is
        recomputed under the current detector; every
        panel validation's identity names the same contracts' sha256, the one the meter recorded
        (`panel_contracts`, passed by the chain as `plan.strong_validate.contracts`, so openings
        in both spaces share it; a validation opened under another refuses); its preflight and
        its manifest's writer recompute its own entry, so a later change of its detector,
        contract or evidence refuses it (the affected validation is invalidated, the others
        stand).
    - **strong_validate, branch stage3:**
      - one validation on V3/O3 (α 0.05, one method), its identity naming the behavior and the
        panel's contracts, its detector and generation contract the active entry's;
      - the detector diagnostic's barrier is checked again here, for observer, PD and ADL-corpus
        continuations;
      - `strong_validation_<behavior>_cont.json` is committed before any follow-up.
    - **s3s_xcont** (`plan.s3s_xcont.dose`, the validated candidate's): X_cont = H + Δ_A of its L,
      engine-free.
    - **The strong lineage** `strong_<behavior>_cont_<dose>` (`strong_lineage_of`, branch stage3):
      - from the committed validation and screening;
      - its states, A, H, method, mode and seed those of the continued selection's validated plan
        (the line has no decision of its own);
      - at most one strong lineage of a behavior is certified;
      - its k*, certificate and readers are part D.
    - **Sessions** (`continuation_stage2.sh`, the lineage's one space `S3_SPACE`, worker pod, the
      batch `strong_stage3`; C17-H2):
      - `s3s` on `strong/stage3/<behavior>` (its preparation), after the committed stack decision:
        `s3s_suppress`, then `strong_screen` of branch stage3 over the selection
        `stage5_next.py --lineage stage3_cont` names (an infeasible continued line stops it);
        `strong_<behavior>_cont.json` is committed before the contracts.
      - **Suites.** Before `s3v`, V3 and O3 are uploaded create-only into the lineage's space
        `data/strong_suites` and checked by sha256, as V1/O1 into a before `s5v`.
      - `s3v` on `strong/stage3/<behavior>/follow` (its follow-up, admitted after the
        contracts' record), only when the receipt check gives the meter's `panel_contracts`, the
        snapshot's committed `stage3_panel_contracts.json` hashes to it, the behavior's entry is
        active, and its screening and both diagnostic manifests (observer and PD, ADL-corpus) are
        committed; `strong_validate` of branch stage3 then runs with
        `plan.strong_validate.contracts` set to that record.
      - `s3x` on the same follow-up, after the committed passing validation, under the same
        record of the contracts as `s3v` (the receipt's `panel_contracts`, the snapshot's file
        hashing to it) and only for a validation opened under it: `s3s_xcont` at its candidate's
        dose, with `plan.s3s_xcont.contracts` set to the record, which its preflight and the
        phase require to be the contracts the validation names.
      - Each session's hours count from its Pod's creation (the receipt), so each runs on its own
        Pod: the chain's exit stops the Pod.
    - **The certified lineage** (§12.7 Stage 3: one certificate and one readout per behavior;
      `pipeline.certified_lineage`).
      - **Which lineage.** It is derived from committed evidence alone, never chosen:
        - the panel's contracts (the behavior's entry recomputed), hashing to the meter's record
          that the chain passes from the receipt (`plan.<phase>.contracts`);
        - an entry reading `strong`: its validated candidate (`strong_<behavior>_cont_<dose>`, X
          its X_cont);
        - an entry reading `stack`: the stack decision the entry binds (`stage3_<behavior>_<dose>`,
          X the composition s3b made, found by its parents and checked, never made again);
        - an entry with no reading is not certified.
      - **A and H** are the implant of the bound stack decision. A's relearning reference is the
        pass that decision pins (E5).
      - **The records.** k*'s and the certificate's records bind the lineage's evidence, the
        contracts and that reference under `stage3`.
      - **The runtime.** The configuration must be the behavior's committed one wherever a
        cell's identity does not hold it whole, and at the protocol version: the model, detector,
        seeds, relearning, LoRA, engine, sampling, prompt sets and the plan's limits (none).
        Where the data live is free.
        The lineage's relearning cells also record the generator's seed and the engine's mode.
        Every calibration run must carry that runtime and exactly a calibration run's params.
      - **Relearning data.** A lineage whose s3a built its relearning data under a detector a
        revision replaced reads that set, pinned in `pipeline.S3_RELEARN_PINS` (countdown:
        d244c40dd6c91a371). It is checked whole and rescored, never generated again.
      - There is no H relearning.
      - `scripts/continuation_stage3.py certified <behavior> --contracts <sha256>` prints the
        lineage.
    - **Its sessions**, on `stage3/<behavior>/follow` in the lineage's space, worker pod, at most
      12 h. Each takes the lineage, dose and states from that helper under the receipt's record of
      the contracts, and every pass carries the record.
      - `s3k`: k* (`plan.kstar.stage3`). Its manifest `kstar_<lineage>.json` is written and
        committed on the laptop only once the analysis is frozen, since the k* outcome is an
        opening. A snapshot follows.
      - `s3c`: once the k* manifest is committed:
        - the certificate's inputs copied in, its check, then its six blocks (resumable over
          sessions);
        - its records stay unopened until the analysis is frozen.
      - `s3r`: the readers, once `continuation_stage3.py readers-admitted <behavior>` finds the
        certificate complete or p₂ terminally 1 (an unresolved k\*, a committed k\* or
        certificate failure record); ongoing work keeps them waiting. Only its statuses are
        printed, never p₂. The pass is `readers` with `plan.readers.stage3` and the record of
        the contracts. It is the only session that needs the graders' key
        (`JUDGE_API_KEY`, `JUDGE_API_BASE`) and an API allowance (`--api-usd`, provisionally $3).
    - **The readers of a certified lineage** (`plan.readers.stage3`): six states from
      `certified_lineage` alone, A and H of its implant, L, H_s, X and the recipe control at its
      dose (`STAGE3_READER_ROLES`).
      - **The δ̄ check (§12.4.3).** A and H reproduce their stored layer-20 pilot packages, which
        are checked on CPU first: the state, layer 20 and the stored δ̄. The other four get the
        method-equivalence run. A non-finite δ̄ or relative error never passes.
      - **Lens packages.** A is read with its fidelity controls. One lens identity serves
        lookup, resume and reading (`_stage3_lens_cell`). 26 packages are read and 40 built.
      - **PD.** L's pool is the one its evidence read. H_s, X and the recipe control are drawn
        on the final PD GPU, A under its own tag (`STAGE3_A_PD_DRAW`), so a pilot pool of A from
        another GPU is never used. H has none.
      - **Before any GPU work**, with lens states left:
        - the graders' key and base are present;
        - the ledger names the meter session, and its allowance is that session's recorded
          ceiling;
        - the money left after open reservations covers the states left (A at
          `plan.api.adl_controls_usd`, the others at `adl_state_usd`).

        The graders open first. A fully cached pass needs none of this.
      - **The record** (`stage3_readers.json`, cell `stage3_readers`) is the input of the
        readers' verdicts. It holds:
        - the binding and the contracts;
        - X's evidence (the composition or the X_cont cell, its state verified on the volume);
        - per state its δ̄ check with both files by sha256, its lens run, its controls and its
          PD run and draw.

        The record's hash is in the cell's identity.
    - **The readers' grading** of a certified lineage, on the laptop, after its `s3r` session is
      closed and before the readers' verdicts open anything. It runs one lineage at a time, as
      each `s3r` completes:
      1. **Pull.** The Pod record and its lens and PD runs are pulled and checked by file count
         and md5.
      2. **PD-clean.** `scripts/pd_clean.py --plan configs/continuation/pd_clean_stage3.yaml
         --record <stage3_readers run>`, in `<DATA_DIR>/pd_clean_stage3/<lineage>`:
         - ensemble rounds under the Stage-3 instructions (`pd_clean_rubric_stage3.md`, the
           panel's frozen trace boundaries, the whole `[prefix]continuation`). Each round issues its
           annotators' kit and is imported from the frozen ensemble's exports in two steps
           (`import-annotators`, then `import-arbiter` on the arbiter's kit when a vote is
           disputed);
         - the researcher's review: every dispute and unanimous trace, and 30 seeded agreed
           negatives per adjudicated package. It is imported from the page's export and log
           (`import-review --export --log`);
         - the eight packages.
      3. **Interpreter.** `scripts/summarize_tools.py` with `experiments/summarize_stage3_lens`
         (the record's 26 packages exactly) and `experiments/summarize_stage3_pd` (the eight
         packages exactly).
      4. **Judge.** `scripts/judge.py experiments/judge_stage3`, with v4 on the lens run and v4-j3
         on the PD run. Its index of the lineage's six states comes from the Pod record
         (`continuation_stage3.py state-index`).
      5. **Categories.** `scripts/continuation_categories.py` grades every counted hypothesis.
      - **Slots.** Every refused answer is retried once under its slot's key (§12.4.5,
        2026-10-25). A strict label whose request and retry were both blocked by the safety
        filter (empty, zero completion tokens, a `content_filter` finish) is
        `strict_safety_blocked` when its lenient label is valid: not counted, the denominator
        80, reported with its interval (§12.4.5, 2026-10-26); no further request is sent.
        A judge attempt is read on the standard tier, or as a historical reply cached before
        tiers were recorded (§12.4.5, 2026-10-27) when the lineage's committed judge history
        `manifests/stage3_judge_history_<lineage>.json` pins its entry's sha256. That manifest
        (`continuation_stage3.py judge-history <behavior> --record … --pd-judge …`, written once
        and committed before the verdicts, which recompute it) holds, for every tier-less entry
        the strict and lenient slots and their retries can read, its original writer: a judge
        run whose recorded commit is in the repository and does not contain the cache's tier
        recording (`114f07a`), on the same cache, whose recorded window (aware ISO-8601 times)
        holds the entry's `cached_at`, which finished strictly before `114f07a`'s committer time
        as git records it (the entry cached before it too; the time is kept in the manifest as
        `recording_boundary`), and whose request log holds the same reply sent and answered
        (`from_cache` false, `error` null, the same text, finish reason and usage). A run that
        only read the entry from the cache is its consumer; a label row alone is no
        evidence of writing (agreeing or not, it is reported). The tier is the writer's: a sync
        run's configured tier (an explicit null is `standard`, that code's name of the default
        tier), `batch` for a batch run, `unrecorded` for a dirty tree, no configured tier or
        writers of two tiers. Only hashes, run paths and provenance fields are kept; an entry
        without a writer is listed as unsupported and leaves its hypothesis pending. Cache files
        are never changed and no request is repeated for tier metadata; the record reports the
        reused slot references, unique entries and their tiers.
      - **Completion of blocked strict labels** (§12.4.5, 2026-10-27), for a lineage registered
        in `continuation_completion.REGISTERED` by its committed specification's sha256,
        `manifests/stage3_blocked_completion_<lineage>.json` (its cohort of blocked strict
        requests, their pinned attempts and lenient slots, the instrument, the reference set and
        the blind review's seed and decoys); a registered specification missing or changed, or
        one of an unregistered lineage, is refused. After the lens judge and before the
        categories, with `scripts/stage3_completion.py`:
        1. `validate <lineage> --send`: strict-v4 on the instrument's model, once, on the
           reference set's items (each request and its one parser retry, standard tier), read
           with the set's unchanged lenient labels under rule C's criteria; it passes when
           criterion 1 passes and criterion 2 passes or is not applicable. A failed send is
           not cached and the rerun sends only what is due; there is no second validation.
        2. `replace <lineage> --send`, after a passed validation: each cohort request's slot,
           its first answer the parser accepts (the retry only after a refusal).
        3. `issue <lineage>`, then `import <lineage> --export …`, when the validation failed
           (the whole cohort) or a slot was refused twice: the researcher's blind review of
           those requests among the 30 seeded decoys, on a frozen page of strict labels only
           (`<DATA_DIR>/stage3_blocked_completion/<lineage>/review.html`, its private
           `issued.json`), issued once; its export imported once, exactly the issued items,
           its bytes kept (`export.jsonl`) and parsed again, against the page's frozen render,
           by every recomputation.
        4. The categories (`continuation_categories.py`) grade the cohort's references that
           rule C counts by their completed labels too, without printing a count.
        5. `provenance <lineage>`: the provenance, written once and committed before the
           readers' verdicts, which recompute it; until then the lineage's readers are pending.
        The completed labels determine the counts and the verdicts; the zero-count treatment
        keeps its intervals in `strict_safety_blocked` and its own analysis on the same frozen
        functions in `zero_count_sensitivity` (counts, p₃, p₄, gates, category margins), which
        selects nothing. Every send is reserved on the chained
        ledger and logged with its request body's model; the judge's request logs record each
        request's own model too (a lenient row's before this change showed the strict model).
      - **Money** (`stage3_money`):
        - the base part is `stage3/<behavior>/follow`'s API left after the Pod's λ; the strong
          part is `strong/stage3/<behavior>/follow`'s, once
          `continuation_stage3.py start-strong <behavior> --need <USD>` has started it through
          the meter (only when the base cannot hold the next batch);
        - each command opens, on each running part's item, the meter's laptop API session
          (`api_session`): authorized for the item's API left, alone on the item, recorded
          before any request is sent. The part's durable ledger records the session's ceiling;
        - each request is reserved on the base while it holds it, else on the strong part,
          after checking that the part's session is still the meter's open one on a running
          item. A request that no part holds stops the step for its price;
        - at the end each session is settled at its committed cost (settled spend and open
          reservations). A session left open by an interrupted command is settled before the
          next opens (`continuation_stage3.py settle <behavior>`). An item never completes with
          a session open;
        - a command spends only for the lineage its inputs name;
        - **the order of the meter's new events.** A meter's code before the laptop session
          refuses the first `api_session` event in a log, even a settled one. So the first
          laptop session opens only after every consumer of the real log runs code that knows
          it: the laptop's scripts, and every Pod snapshot made from then on (the certificate's
          continuations, `s3c` of the other lineages, `s3r`). Pods already running finish on
          their own copies of the log, and the updated laptop code settles their sessions.
    - **The readers' verdicts** (`continuation_readers`), on the laptop, from the readings alone.
      `lineage_of` resolves the certified lineage and its X. A strong lineage's X is its X_cont
      cell, recomputed through every attempt's evidence; no adapter is loaded.
      - **Inputs**, each recomputed rather than trusted:
        - every source cell's run checked on its saved config: the hash recomputed, the hash its
          meta records and the approved cell's hash the same, its identity canonically the
          cell's;
        - the Pod's record, the run of its approved cell;
        - per state, its approved cells: the reference, the lens package (the frozen lens
          settings are its identity) and the PD pool. The δ̄ check is recomputed on CPU from both
          files, each matched to its recorded sha256;
        - the lens packages by sha256;
        - each PD-clean package, recomputed from its study's records
          (`pd_clean_study.verify_packages`, the lineage's work directory: rounds, the
          researcher's review, the selection and the random control). Each review's negative
          sample is drawn again by its seed from the records it names (`after`), and its items
          are checked. What each annotator and the researcher read (`items.jsonl`) is rebuilt
          from the pool rows, and so is each round's kit: the annotators' (the frozen
          instructions filled in, the items in parts) and the arbiter's (the disputed items, the
          votes counted without names). The kits are kept unchanged. Each export's manifest
          names the plan's model and runner and the kit it was made on, by sha256. Each round's
          labels must be the plan's merge of its kept exports (three annotators and the
          arbiter, `pd_clean_stage3.yaml` `ensemble`). Each review's
          labels must be its page's kept export, and the page's log must name the items it
          showed by sha256. The models, runners and independence that the manifests state are
          the operator's recorded declarations. The sha256s establish the exact materials those
          declarations refer to; they are not a cryptographic proof of how the work was run;
        - the interpreter's 80 sample slots per package;
        - the judge's own-target labels (v4 lens, v4-j3 PD-clean, standard tier);
        - the `category-v1` grade of every counted hypothesis, with pinned definitions;
        - X from its own records.
      - **Sample slots.** Each slot's request is rebuilt from the frozen instrument (model,
        thinking level, cap, prompt or rubric, structured output, sample index and salt). Its
        answer is the first one the frozen parser accepts, out of the request and its one retry
        (`#retry=1`, §12.4.5 of 2026-10-25). A slot refused twice is pending. Every slot's whole
        history (each attempt's key, cache entry sha256, tier and parse) is bound by its digest.
      - **p₃** is draw 1's exact p when draw 1's eligibility, fidelity and L's category confusion
        pass and the two draws' fully gated preliminary verdicts agree; otherwise 1. A's
        reconstruction diagnostic is reported.
      - **p₄** is the largest of the two TOSTs and the one-sample test under the PD positive
        control; otherwise 1.
      - **Exact tests.** Each test's certified bound is refined where its enclosure straddles 1/20
        or 1/60. Every enclosure is kept, both ends under min(1, ·).
      - **The manifest** `manifests/stage3_readers_<lineage>.json` is written once
        (`write_readers`) and recomputed by `committed_readers`. Its status is complete, or
        terminal for an X_cont invalid after its retry (p₃ = p₄ = 1). Pending work writes
        nothing. The four lineages' manifests are written and recomputed by
        `scripts/stage3_write_readers.py` (`check` the preconditions, `write`, then after the
        commit `verify`), each from the bound readings its `SOURCES` names.
    - **The panel's analysis** (`continuation_panel`, `scripts/continuation_panel.py`), on the
      laptop, from committed evidence alone.
      - **Per behavior**, its entry decides the path:
        - `strong`: p_i = max(p₁, p₂, p₃), p₁ the exact tail of L's O3 hits, not doubled;
        - `stack`: p_i = 1, with the four-component rule reported as descriptive (p₁ from the
          non-voided deciding confirmation, doubled);
        - no reading: p_i = p₃ = 1, "ineligible, no readout".
      - **Incompleteness:**
        - ongoing or resumable work is pending, and nothing is written;
        - a documented terminal failure keeps the lineage with p = 1 for that component:
          - an unresolved k*, whose record `scripts/continuation_kstar.py` writes and the
            analysis recomputes;
          - `manifests/stage3_failure_<lineage>_<phase>.json`, written by `failure` only for an
            allowed terminal stop (`FAILURE_KINDS`): the lineage's meter item stopped or dropped
            (`budget`), or a relearning cell of k* or the certificate that failed with a classified
            training failure (out of memory or divergence of the training step itself, which a
            relearning run records in `failure.json`; an out-of-memory error while loading or
            generating is not one)
            on its attempt and on its one retry of the same identity, with no completed run of it
            (`retry_exhausted`); a lost machine, a deadline, the network, an engine or loading
            error or an unclassified failure stays pending;
          - the readers' terminal status;
        - malformed, contradictory or foreign evidence is refused.
      - A failed certificate or an unresolved k* never changes the readers' p₃.
      - **Thresholds.** The panel rejects at p₍₃₎ ≤ 1/60. Stage 4 counts p₃ ≤ 1/20 (1 to 4
        trigger). Stage 5's gate 3·p₃,(3) ≤ 1/20 is reported with its action, the slot exception a
        holds.
      - **Certificate.** Its records are found by their cells' metadata, merged whole and checked
        against the trusted inputs the committed k*, the Y₂ manifest and the lineage give. Each
        record's sha256 is kept, and each sign test with its interval, coverage and median.
      - **A relearning run's failure cause** (`pipeline.FAILURE_RECORD`, `failure.json`). Every
        relearning run (`certificate.relearn`, through `relearn_state` or the engine check)
        stops at a non-finite training loss as a classified divergence. A run through
        `relearn_state` records that, or an out-of-memory error of its training step, as its
        cause; a failing wake-up after it is recorded beside the cause and never replaces it.
        The engine check's runs stop the same way but record no cause. The scope covers:
        - the certified lineages' k\* and certificate;
        - Stage 2's and Stage 5's k\*, certificates, Y-alone runs and H relearning;
        - path B's relearning and the engine check.

        Suppression and implant training are unchanged, because only relearning asks `fit` to stop
        at a non-finite loss. The panel's terminal stops read this record for the certified
        lineages alone; no other caller's rule changes with it.
      - **Readers.** The readers' manifests (`stage3_readers_<lineage>.json`) are read through
        their recomputation verifier. Their binding, flags, X's validity and values are checked;
        an invalid X needs p₃ = p₄ = 1.
      - **Expression checks** are kept with their counts, bounds and labels: the strong
        validation's, or the stack decision's.
      - **Commands:**
        - `analysis` writes `manifests/stage3_analysis.json` once, with the frozen sentences
          (`WORDING`);
        - `check` recomputes the committed record.

## Follow-up: the confirmations of example_link and pun_intended

[Preregistration §13.3](preregistration.md#133-confirmations-example_link-and-pun_intended):
§12.4.1 as executed on 2026-10-19, on one Pod (H100 SXM, US-NE-1, the network volume at
`/workspace`), writing only into space a (`/workspace/a`), with
`scripts/pod/follow_confirm.py` (`residual_disposition.follow_confirm`):

1. **Fitness**, right after the start and before anything else (`fitness --volume /workspace
   --space /workspace/a`, `pod_fitness.check`). The checks that stop:
   - the volume is the project's: a mount point holding its mark (M0's `result.json`, by sha256;
     `follow_confirm.VOLUME_MARKS`);
   - in the space, files of random bytes are written, fsynced and read back equal, within a
     generous timeout against a hung mount;
   - a loop of 3·10⁷ Python additions takes at most 3.0 s;
   - the container's root is overlay, xfs or ext4.

   The volume's file system type (a network volume is fuse) and the probe's time are recorded as
   diagnostics. Exit 3: the Pod is stopped and replaced, as an infrastructure incident, not an
   attempt.
2. **The meter's session** on `follow/base`, then a copy of the log.
3. **Inputs** (`precheck --space /workspace/a --source /workspace/b --log <file in the space>`,
   CPU):
   - `RUNS_DIR`, `DATA_DIR` and `HF_HOME` must be the space's `runs`, `data` and `hf`. No hub
     cache may be moved elsewhere, and no other writer's directory or output may lie on the
     volume outside the space.
   - The space must hold:
     - the eight states a line may draw (latent and sham_s at d₁ and d₂), every adapter checked
       against its recorded tensor sha256;
     - the analyst index, recording each as its line's stack suppression or its sham at its
       dose;
     - the confirmation prompts with their set hash;
     - Qwen3-14B at M0's revision, every snapshot file a link to a blob in the model's `blobs`
       whose name is its content's address. The content is read through every link: the hub
       may link a large blob on to a shared store (`hub/blobs/<xx>/<another hash>`), whose
       names are not content addresses. A link whose target lies outside the model's `blobs`,
       as written or as resolved, is refused.
   - What is missing is copied from the sources in order, create-only (`copy_inputs.py`,
     `follow_confirm.copy_model`), and everything is checked again. The model is written only
     in directories of the space, none of them a link. Each blob is staged in a new regular
     file and published without replacing any entry; a dangling link is refused as well.
   - Before any copy, the volume must have room for the copies (the missing states with every
     state their adapters point at, the index runs, the prompt pool, the model's blobs, sized in
     their source) and 20 GiB for the session.
   - An item that is present but differs is refused and never overwritten.
4. **Draws** (`run --space /workspace/a --out /workspace/a/follow/confirm/<run id>`, detached).
   On entry, the space's checks and every input check of the precheck run again; anything
   missing or different, or less room than the session needs, refuses before any generation.
   - **Sources.** The follow-up rules, §12's rules, B1's dose manifest
     (`continuation_rules.follow_doses`, pinned by sha256) and M0's model are loaded through
     their frozen loaders and checked against each other (`follow_confirm.load_spec`). The checks
     cover:
     - the exact test's largest passing count against the rules' `max_hits`;
     - the attempts and the prompt set;
     - the seed keys;
     - a new measurement, stream and base seed.

     A disagreement refuses before any generation, and so does a workspace whose model,
     experiment or detector is not the sources'.
   - **The lines in order.**
     - Each line is drawn until it is decided: d₁ first, each draw after the volume's room is
       checked again.
     - A completed cell counts only whole: it is the cell of its allocation (its saved
       configuration hashes to its recorded hash, which is the cell's), its summary repeats the
       allocation, both raw draws are saved with their states, and the hits counted again on
       the saved responses are the summary's. Anything else is refused, never skipped. An
       interrupted cell must be the same cell, and its saved draws must name its states, before
       they are reused.
     - d₂ is drawn only after a completed draw of d₁ whose latent state has more than 4 hits of
       600; the line then stops.
     - Every cell shares its generation (sampling, engine, requests, prompts, generator seed) with
       the recorded confirmation cells.
   - **Records on the volume.**
     - A launcher's `RUN_ID` is removed first and kept in `result.json`. Each draw's run takes
       its own directory; a directory name already taken is refused before anything is written.
       Cells are found by their record, whatever their directory is called: link's first cell
       of 2026-10-30 is named after its launch. A directory whose configuration, metadata or
       name names the measurement is accepted only with both files and all three names agreeing;
       otherwise it is refused before any allocation or generation.
     - The allocated attempt and its spec are written before any generation.
     - Each state's whole draw is saved before the next.
     - Then the detector records, the responses and the summary.
     - No model is downloaded.
   - **Failures.** Any exception (memory, the engine, the volume, the watcher's stop) ends the
     session with no further draw of either line. The attempt stays the next one and resumes in
     the next session as that very cell, reusing a state's saved draw.
   - **Result.** `<out>/result.json` holds the decision: exit 0 once both lines are decided, 2
     after a stop.
5. **Pull** the space's new confirmation runs and the result, check every file's sha256 against
   the Pod's, then stop the Pod and close the meter's session. `status --space <space>`
   recomputes the decision from the runs on CPU.
6. **Manifest.** Once both lines are decided, `scripts/follow_confirmations.py --write`, at a
   clean commit, writes `manifests/follow_confirmations.json` once from the laptop's runs: the
   sources' hashes, each line's status and dose, and each draw with its run, configuration hash
   and every file's sha256. It is committed before any build. The confirmed doses
   (`{line: dose or None}`) go to `follow_inventory.resolve` only through
   `follow_confirm.committed_confirmations`: the manifest committed and its record the one
   recomputed from the runs.

## Follow-up: package builds and PD pools

[Preregistration §13.10](preregistration.md#1310-package-builds-and-the-extraction-check-m0):
the new lens packages and PD pools once the confirmations are decided, on one Pod in space b
(M0's: the states, the corpus and the model), from `configs/continuation/follow_build.yaml`,
frozen by `continuation_rules.FOLLOW_BUILD_FROZEN`.

- **The plan** is derived from the records (`follow_build.planned`; a test holds the frozen file
  to it):
  - the new packages B2's inventory names at the committed confirmations' doses
    (`follow_confirm.committed_confirmations`): eli5 stack 200 and pun_intended 200, each with
    L, H_s, X and the recipe control; example_link's packages at 200 exist;
  - eli5's states from Stage 3's stack decision and its selection record, X its composed
    suppression-only control; pun_intended's L and H_s from B1's dose manifest, its recipe
    control the analyst index's one recipe control of that dose with A as parent and L's seed,
    mode and method;
  - pun_intended 200's X is named by its composition (L 200 and H) and composed in the session
    (§13.2: H + U_A; `pipeline.suppression_only_state`, checked by `pipeline.stack_x`);
  - each built behavior's A as the session's reference;
  - the PD pools: every new package's state but eli5's L, whose pool Stage 3's stack decision
    names; seven, fixed by the records so that a resumed session keeps the same set;
  - the confirmations the plan was derived at: the manifest, its sha256 and the doses.
- **Before any write**, in both entry points: the session writes only into space b with room
  (`follow_build.placement_problems`: `RUNS_DIR`, `DATA_DIR`, `HF_HOME` and the output directory
  in the space, no hub cache, writer or output elsewhere on the volume, the volume's room); then
  the committed confirmations, recomputed from their runs in space a (read only), must be the
  doses, manifest and packages the builds were frozen at (`follow_build.committed_plan`).
- **Builds** (`scripts/pod/follow_build.py experiments/continuation_14b/word_count_claim
  --session b1 --space /workspace/b --confirmations /workspace/a/runs --out <new directory in
  the space>`): the composition first, every state's adapters against their recorded sha256, the
  runtime against M0's, then `follow_build.execute` (the session's base means, the references,
  each build's method equivalence at 2⁻⁸ before its readings).
- **Pools** (`scripts/pod/follow_pools.py experiments/continuation_14b/<behavior> --space
  /workspace/b --confirmations /workspace/a/runs --out <new directory in the space>`, eli5 then
  pun_intended, after the builds): the frozen PD settings, the composition found or made and
  checked, every state's adapters; then per pool `pipeline.audit_pd`, untagged, on the readers'
  final PD GPU (`pipeline.FINAL_PD_GPU`), and its check whole (`follow_build.pool_problems`: that
  GPU, found or new, the frozen prefills, one completion per framing and prefill, every row's
  fields and types). No behavior detector or interpreter is applied.
- **Before B3** (permitted by §13.14): the new pools' completions and their derived estimates
  (ΔPPL and its ranking) are not inspected until B3 fixes their reuse, and no choice between
  contracts or regenerations is made after any look.
- **The window.** The watcher stops the Pod at its creation + 5.5 h, start-up and stop
  included. A cut leaves every completed build whole; an unfinished build or pool is never
  marked complete (no `build.json`, no complete `result.json`, its cell not completed). Partial
  directories may remain and are never used: a later session builds anew in new directories.
- **Run ids.** Each entry point is launched with `env -u RUN_ID`, removes a launcher's `RUN_ID`
  itself and keeps it in its manifest: every pipeline cell (the composition, each pool) takes its
  own run directory.
- **Pull** the session's output directories, the composition's cell and the pools' runs with
  every file's sha256 checked against the Pod's, then stop the Pod and close the meter's
  session.

## Follow-up: eli5's stack branch (B4)

[Preregistration §13.7](preregistration.md#137-the-stack-branch-eli5), barrier B4, from
`configs/continuation/follow_b4.yaml` (frozen by `continuation_rules.FOLLOW_B4_FROZEN`;
`residual_disposition.follow_stack`):

- **Candidates:** Stage 3's saved stack states at 400, then 800 (L, H_s and the recipe control
  from its selection record; a test holds the file to them). The detector is the panel
  contracts' eli5 detector.
- **Suites, before the screening opens:** V4 (PD 9,194 prefills; ADL 4,000 records, 600 per
  early length 1–5 and 1,000 of 32 tokens) and O4 (the prompt pool's slice [9600:10200]), built
  on CPU by `scripts/make_follow_suites.py --seed 20261030`, saved create-only under
  `data/strong_suites/` and frozen by sha256 and inventory (PD `ae5e20ad…`, ADL `79fecdbb…`, O4
  `32f7ec74…`). Like V1–V3, they are uploaded create-only into the validating space's
  `data/strong_suites` and read only at their frozen hashes (`follow_stack.suite`). §12.4.10's populations are rebuilt
  and checked against the frozen suites' population hashes; V1–V3 (and the screening suites) are
  excluded; the RNG keys are new (`follow_pd`; `follow_adl_stratum`, then `follow_adl`); O1–O3
  are rebuilt at their hashes and O4 is checked disjoint from every saved set of the pool, the
  confirmation set and O1–O3.
  - The ADL records' documents are disjoint from V1–V3's and the screening's; their short
    prefixes (1–5 tokens) are token strings that may repeat across documents, as in V1–V3.
  - O4 is a slice of the pool's one shuffle, whose seed is the prompt pool's (0), not the suites'
    base seed.
- **Screening** (`scripts/pod/follow_stack.py experiments/continuation_14b/eli5 --space
  /workspace/a --out <new directory in the space>`, in eli5's registered space a, one Pod session
  of its own after the build session's meter session on `follow/base` is closed, launched with
  `env -u RUN_ID`; placement and room checked before any write). Before any generation: the
  generation contract (the panel contracts' eli5 contract but for its requests, by sha256), every
  candidate's ADL settings the frozen ones, and stack 400's pinned PD pool present whole (run and
  file hashes in the configuration: copied in or refused, never redrawn). Then per candidate in
  order its full
  PD pool on the final PD GPU and its ADL-corpus check, until the first eligible one (zero hits
  in each PD framing and in both ADL framings, early and long), each run its approved cell before
  anything is counted (its saved configuration hashing to its recorded hash, the cell's as the
  branch's workspace computes it). Stack 400's PD pool is Stage 3's (untagged, on the final PD
  GPU): the pinned cell the screening counts, its first count. Invalid evidence stops the session before the next candidate. No eligible
  candidate closes the branch.
- **Manifest:** after the pull, `scripts/follow_stack.py <result.json> --write` recomputes the
  record from the runs and writes `manifests/follow_stack_eli5.json` once at a clean commit; it is
  committed before any validation cell.
- **Validation choices, frozen at B4:** the contract above with the requests of O4 and of the
  frozen OOD prompts; the fresh streams `follow:eli5_stack:observer` and `follow:eli5_stack:ood`
  with the contract's generator seed; the committed screening's candidate only, one attempt, no
  other candidate, dose, suite, stream or redraw chosen after any part of the validation is seen.
  Its code is written after the screening and reviewed before any validation generation.

## Follow-up: barrier B3

[Preregistration §13.15](preregistration.md#1315-barriers), barrier B3, from
`configs/continuation/follow_b3.yaml` (frozen by `continuation_rules.FOLLOW_B3_FROZEN`; checked by
`residual_disposition.follow_b3` against the records and the code). It is committed before any
new reader request to L, H_s or X, PD included, and before any certificate run.

- **Certificates** (§13.8): eli5 stack 200, example_link 200 and pun_intended 200, by §12.4.8,
  §12.3.7 and §12.7 unchanged: k* on H_s with seeds 6–8 at both rates, stopping at the first
  crossing within 600 steps, B by its formula, the 72 runs on the sleeping engine. Each line's
  dose, states, implant and A's reference are its deciding record's (eli5's stack decision;
  the confirmations and the pilot's approved passes for example_link and pun_intended); its
  meter item is `follow/cert/<line>`.
- **Seal** (§13.14's parallel execution): one runner per line computes k* and B and passes them
  to the 72 runs inside itself; they, the calibration record, the runs' outputs and logs, raw
  failures and per-certificate accounting are sealed until the opening. The runner writes one
  sanitized status file (running, completed, or a technical failure with a cause that names no
  arm, step or metric). Every 30 minutes from its start it writes an envelope holding only the
  sha256 of a payload: the line, the envelope's number, B3's sha256, a fresh 256-bit nonce and
  the calibration record once it exists, else 4,096 random bytes. The session's runs, its
  workspaces' work and served-state roots and every writer's cache (`pod_space.WRITERS`) are in
  its own sealed directory, its inputs copied there create-only beforehand, the hub cache read
  offline; neither the monitor nor another session in the space lists that directory. The
  envelopes are collected and committed under `manifests/follow_seal/<line>/` on that schedule
  whatever the calibration's state; the runner starts the dependent runs only after the receipt
  of an envelope whose payload holds the record, and recomputes the calibration before using it.
  The payloads and nonces are revealed at the opening.
- **Rungs 2 and 3** (§13.6; `residual_disposition.follow_ladder`):
  - rung 2: D*'s state-decoder rendering unchanged, with each position's steered and unsteered
    samples at layer 20 by §11.1's procedure (the cell's own δ̄ from its D* package, at its
    recorded norm; L with a random direction read through its stored random branch at both
    layers and steered along the seeded direction of that branch, with L's unsteered samples),
    cut to 128 characters; prompt `adl-lens-multiscale-steer-v1`; D*'s interpreter, i = 0;
  - rung 3: the agent (`ladder-agent-v1`, rung 2's input) and the comparator
    (`ladder-comparator-v1`, the unsteered samples only) with one protocol: each turn
    `QUERY: <prompt>` or `HYPOTHESIS: <sentence>`, at most 5 queries of at most 2,000
    characters, each answered once by the examined state and the base model with the observer's
    sampling; for L with a random direction the examined state is L. Every turn and reply has its
    own key and seed; a reply once drawn is never drawn again; one format-only retry per turn;
    80 runs per reading;
  - the probes' digests (`follow_b3.b3_renderings`, `b3_requests`) hold the rendering and the
    requests to the code.
- **Inventory** (`scripts/follow_b3_inventory.py --write` → `manifests/follow_b3_inventory.json`,
  its sha256 in the configuration): D*'s 49 and unfiltered PD's 21 analysis readings of the six
  partial-threshold lineages and eli5 continued 200, salted by lineage, mode, dose and stream;
  for rungs 2 and 3, the shared A and H qualification (read once, in whichever mode first reaches
  the rung), each mode's recipe qualification, the validation-only streams (shared, and the
  first mode's recipe) and the continued mode's cells, with their packages and the sha256 of
  their vectors' sources; and unfiltered PD's reuse decisions. A rung's format is validated
  once, on the shared readings and the first mode's recipe, with §13.5's strata.
- **Unfiltered PD's reuse** (`residual_disposition.follow_pd`): an analysis reading takes a
  committed Stage-3 reading's 80 hypotheses only when its readers' manifest is committed and
  recomputed whole from its sources (a failure stops the inventory) and the reading verifies as
  the readers' verdicts verify it, its package is of the same state and pool and removed nothing, its ordered
  top lists are this selection and its evidence this reading's; each slot is mapped to its
  response, key, retry history and tier, and its labels are reused only as admitted there. A
  reused hypothesis counts once. Pools built on 2026-10-30 are read only by new readings.

## Follow-up: the existing L packages' random branches

D*'s analysis readings name the random-direction branch of five existing L lens packages
(word_count_claim 200, countdown 50, tldr 400, example_link 200 and eli5 continued 200), which
they do not hold. One session builds them (`configs/continuation/follow_lrandom.yaml`, frozen by
`continuation_rules.FOLLOW_LRANDOM_FROZEN`; `scripts/pod/follow_lrandom.py
experiments/continuation_14b/word_count_claim --space <space> --out <new directory>`), as the
packages of 2026-10-30 were (`follow_build.execute`, M0's runtime, the same checks of placement,
runtime and adapters before any GPU work):

- the session's builds are exactly the packages B3's inventory names
  (`follow_build.lrandom_problems`), each L at its package's recorded state and seed, its
  control's sources (the package's stored state means and record, the shared base means) at
  their pinned sha256 (`follow_build.pinned_problems`);
- each behavior's A is rebuilt first as a reference (`follow_build.execute_random`); per L,
  the same session's method check alone decides (its δ̄ by the production path against the
  qualification path, to 2⁻⁸); its δ̄ and norm against the pinned ones, at both layers, are
  recorded as diagnostics (§13.10), the pinned reference in each error's denominator;
- the random branch is read from the pinned sources alone (`follow_build.pinned_sources`):
  the directions of the pinned δ̄ (stored means minus the shared base means) at the package's
  seed, at the pinned η, through the state decoder; a failed method check stops the session
  before anything of that build is read;
- the existing package runs are never written.

After the pull, `scripts/follow_lrandom.py runs/follow_lrandom/<run> --write` checks the output
whole, by recomputation rather than by its flags (`follow_build.lrandom_output_problems`: the
manifest the frozen session's; every saved tensor at the sha256 the session recorded; every
deciding method check recomputed from the saved tensors, passing and equal to the reports of the
result and of the build's record; the diagnostics recomputed from the pinned sources; each build
its planned state, package and seed, exactly its complete file inventory, its readings rendering
as D*'s random branch) and records the run, the frozen session, the manifest's and the result's
sha256 and every built file's sha256 once in `manifests/follow_lrandom.json`, committed with the
session's decision-log row. The inventories find a branch only through that committed record
(`follow_inventory.random_branches`: the run directory exactly once under the runs root, the
manifest and result at their recorded sha256, each build's complete inventory at its recorded
sha256).

## Follow-up: the qualification of D* and unfiltered PD

[Preregistration §13.5 (a)](preregistration.md#135-gates-and-tests), barrier B2's inventory, on
the API from recorded packages only (`residual_disposition.follow_qualification`;
`scripts/follow_qualification.py [--pin | --send]`):

- **Readings:** B2's inventory resolved at the committed confirmations: D*'s 100 qualification
  readings (A, H and the recipe controls on both decoders, A's random-direction and no-patch
  branches), unfiltered PD's 26 and its 12 validation-only readings of H. Before anything is
  rendered, dry runs included, each new package is bound to its entry in the frozen build
  configuration (behavior, lineage, role, state) and to the build session's records at their
  pinned sha256: a lens build of exactly that state, complete and checked, its record the same
  and its vector files at the recorded sha256; a PD pool of exactly that state, its run
  completed and whole. The session's other packages are never read. An existing package's run
  must be completed: a PD pool of the inventory's state; a lens package the approved cell of its
  role's state as the inventory gives it (its saved config's hash recomputed, a lens package of
  that state, the readers' cell lookup's hash and identity; Stage 1's dose packages the one its
  readings record names).
- **Old judge replies:** a reply cached before tiers were recorded leaves its slot pending until
  `--judge-history <run>` has written its provenance once, by §12.4.5's route of 2026-10-27
  (`manifests/stage3_judge_history_follow_qualification.json`); what cannot be established is listed
  as unsupported and replaced (§13.16): its slots read the retry key, a new request of the same
  body on the standard tier, and the record's `replaced_replies` holds the old replies' hashes.
- **Pins:** `--pin` writes every reading's package directory, evidence sha256 and slot keys
  once (`manifests/follow_qualification_inputs.json`), committed before any request; `--send`
  refuses readings that are not their pins, so a changed evidence never makes new requests.
- **Requests:** D*'s rendering or unfiltered PD's selection with no removal as the evidence; the
  frozen interpreter and prompts, 80 slots per reading under its salt, each with one
  format-only retry; each parsed hypothesis judged by rule C (v4 for D*, v4-j3 for PD), each
  label request with one retry. The interpreter reads on the flex tier with the automatic
  fallback to the standard one (as Stage 3's interpreters did), the judge on the standard tier
  (B2); a tier is never part of a request's body, key or salt, and every answer records its own.
  Every attempt read is admitted at its tier as the readers admit it: the interpreter's flex or
  standard; a judge's standard, or a historical reply a committed judge history pins (safety
  blocks included). Any other reply leaves its slot pending and is never asked again.
- **Histories:** every slot's attempts (interpreter, strict, lenient: key, entry sha256,
  recorded and admitted tier, parse; rejected replies included) are recorded before and after
  the sends, and every answer is journaled as it is stored (`answers.jsonl`, synced at once).
  Before the run and before each round, every answer recorded so far must still be in the cache
  unchanged, and a key recorded with two entries is refused; an attempt missing while a later
  one is answered is refused. Nothing is redrawn.
- **Money:** every send is reserved on `follow/base`'s API session at its tier's price, the
  run's limit checked on the standard tier's worst case; the dry run's figures (the due
  requests' estimate, the judging's projection on a probe hypothesis) are not bounds; the
  decision log reports the actual spending.
- **Same request, two dates:** countdown's recipe control on the state decoder, byte-identical
  request on the standard tier, counted 18 of 20 on 2026-10-27 (the reader candidate's screen) and
  81 of 160 on 2026-10-31 (Fisher's exact p = 0.0006; the only one of 16 comparable cells that
  differs). The records do not establish the cause, and no new sample was drawn. Cells compared
  within one reading are sent in one window of time.
- **Record** (`runs/follow_qualification/<run>`: `inputs.json`, `tier_fallback.jsonl`,
  `histories-before.json`, `answers.jsonl`, `histories.json`, `qualification.json`): each
  reading's counted hypotheses of its 80 slots (pending while one is unresolved) with its safety
  blocks (b, b_yes and the interval [c/80, (c + b_yes)/80], descriptive, with the rates and
  floors at the upper ends), each pool's rate over its two streams (160 slots) and the floors per
  lineage on exact fractions: D* per decoder R(A) ≥ 7/10, R(H) ≤ 1/10, R(A) − R(H) ≥ 3/5 and
  R(recipe) ≥ 7/10 · R(A) (eli5's two recipe controls each), A's fidelity branches R ≤ 1/10;
  unfiltered PD R(A) ≥ 7/10 and R(recipe) ≥ 7/10 · R(A). D*'s eligibility is the state decoder's
  floors and the fidelity branches; the base decoder's govern only the secondary p_both (§13.16).

## Follow-up: the format validation of D* and unfiltered PD

[Preregistration §13.5](preregistration.md#135-gates-and-tests), barrier B2's `format_validation`
and `ensemble`, on the qualification's hypotheses (`residual_disposition.follow_format`;
`scripts/follow_format.py`, records under `<DATA_DIR>/follow_format/<format>`, immutable):

- **Frame:** one item per parsed hypothesis of the format's frame readings, read back from the
  cache at the entry sha256 the named qualification run recorded; an unresolved slot leaves the
  frame unbuilt. `frame.json` holds the run, its files' sha256 and the seeded order, written once.
- **Rounds** (`round`): the next items in the seeded order, only when the frozen stopping rule
  asks for more labels and every earlier round is labeled and reviewed. The annotators see an
  opaque id, the behavior, the TARGET and the HYPOTHESIS, under B2's frozen instructions; the
  mapping stays private.
- **Ensemble** (`import-annotators`, `import-arbiter`): each export holds the four fields for
  exactly the round's items, its manifest B2's model and runner on the kit issued. An item is
  disputed on any of strict, lenient, categories and specificity the three do not give alike; a
  fresh arbiter sees the disputed items with the votes counted anonymously and settles the
  disputed fields.
- **Review** (`review`, `import-review`): the researcher reviews every dispute and the seeded
  tenth fixed by the item alone, blind to reader, state, role, decoder, reading and machine labels;
  the page confirms each item explicitly, and its export and log are kept. A review's labels replace
  the merged ones. Above 15 % disputed items in the first round, the format's review keeps only the
  disputes whose votes can change the class or the difficulty (§13.16).
- **Lock** (`lock`): the reference class is rule C on the final labels, the difficulty from the
  final categories and specificity; `follow_validation.step` locks 400 + 400 items and the record
  holds the labels it locked. The search for strata stops once the final labels hold 400 per class
  (§13.16): unmet minimums are capped at the reviewed prefix and the selector runs once; a frame
  exhausted with a class short leaves the format not validated.
- **Acceptance** (`accept`): the machine labels are read only here, from the qualification's
  recorded attempts at their entry sha256, against the locked labels, a safety-blocked strict label
  counted as unresolved; `acceptance.json` holds the gate and, beside it, κ and the
  alternative-annotator test.

## Follow-up: the readings of L, H_s and X by D* and unfiltered PD

[Preregistration §13.5, §13.16](preregistration.md#135-gates-and-tests); step 2 of §13.14
(`residual_disposition.follow_readings`; `scripts/follow_readings.py [--pin | --send |
--judge-history RUN] [--sensitivity]`, records under `runs/follow_readings/<run>`), run after the
random branches' committed record:

- **Readings:** B3's analysis inventory at the confirmations, with the base decoder's controls on
  each A (`scripts/pod/follow_base_controls.py` on one Pod, then `scripts/follow_base_controls.py
  runs/follow_base_controls/<run> --write` commits their record, `manifests/follow_base_controls.json`).
  Unfiltered PD's reused readings are read back, each attempt at its recorded sha256, and never
  asked again.
- **Requests:** as the qualification's: the pins committed first, one run, so the cells compared
  within a reading share one window of time.
- **Grades:** each counted hypothesis of an L reading graded by `category-v1` and `specificity-v1` on
  the standard tier. `scripts/follow_format.py grade <format>` validates the specificity grader on
  the locked items first (`graders.json`: κ ≥ 0.7, the alternative-annotator test, category agreement
  beside it); unvalidated, specificity is descriptive and no specific claim follows.
- **Record** (`readings.json`, not printed): each reading's count, the identification test at the
  certified regions, each instrument's gates (qualification and controls, format validation, category
  confusion), a Holm family over the six partial-threshold lineages for the state decoder, the base
  decoder and unfiltered PD, countdown at p = 1 in the state decoder's with its exploratory reading
  beside it, both decoders' agreement per line, rung 1 on eli5 continued 200 per decoder at 1/60 with
  its random cell at most 8 of 80 through that decoder. Category recognition and identification of the
  specific behavior are reported apart. On a
  strict label blocked twice the slot stays pending; the blind completion of §13.16 is done by hand.
  `--detector-share` prints the detector-hit items of each unfiltered-PD L package, reported beside it.
- **Sensitivity of eli5 stack** (`--pin --sensitivity`, then `--send --sensitivity`; §13.16): its six
  cells on the standard tier with no fallback, interleaved, under their own pins
  (`manifests/follow_readings_sensitivity_inputs.json`), judge history (`--judge-history RUN
  --sensitivity`) and run root (`runs/follow_readings_sensitivity/<run>`). The record (`sensitivity.json`, not printed) holds each
  decoder's counts, gates and Holm result with only eli5's decisions replaced, beside the original
  record, which it does not change.

## Follow-up: rung 2 on eli5 continued 200

[Preregistration §13.6, §13.16](preregistration.md#136-the-full-threshold-ladder-eli5); the decoder
set is recorded in [the decision log](decisions.md) (`residual_disposition.follow_rung2`;
`scripts/pod/follow_steering.py` on one Pod; `scripts/follow_rung2.py`, records under
`runs/follow_rung2/<run>`; `scripts/follow_format.py <command> rung2`, records under
`<DATA_DIR>/follow_format/rung2`). In this order:

1. **Steering, one Pod.** The meter session on `follow/continued/rung2`, with the coherence grader's
   API money (`--session` and `--api-usd`, the session's id and number) and the Pods' Gemini key; a
   coherence judgment left unresolved stops the session (§13.16, H11), its samples and replies kept.
   Seven cells: A, H, the recipe control, L, H_s, X_cont and L with a random direction, each by §11.1's
   search and samples at layer 20, its δ̄ and η from the files B3's inventory pins, checked before any
   GPU work. Q1's A and H packages are not reused: their δ̄ are not these packages' files (relative L2
   difference up to 1e-4) and their seeds differ, and B3 reuses them only when all are equal. The random
   cell starts from L's unsteered samples. An interrupted session is run again from the start, as a new
   Pod. A pulled session goes under `runs/follow_rung2_steering/<run>`, and `--record-steering <run>
   --write` commits `manifests/follow_rung2_steering.json`.
2. **Pins.** `--pin`, committed, before any request.
3. **Qualification.** `--send qualification`: A, H and the recipe control, two streams per decoder,
   the state decoder's under B3's salts, the base decoder's under `follow/rung2/base/...`. The record
   is `qualification.json`, not printed. A decoder failing a floor stays ineligible and is read as
   exploratory; the recipe floor is reported and does not close a positive contrast.
4. **Format validation.** One for the rung, on both decoders' qualification readings in one seeded
   order, as D*'s (`status`, `round`, `import-annotators`, `import-arbiter`, `review`, `import-review`).
   When `status` says `generate_supplement`: `--send supplement --stream <next>`, then
   `scripts/follow_format.py supplement rung2 --run <that run>`, and labeling goes on; a class still
   short, or a decoder under 150 of a class, after stream 6 leaves the format not validated. Then
   `lock`, `accept` and `grade`: errors are reported by decoder in `by_stratum`, and a pooled pass
   covers the mixture only. An `incomplete` with both classes and both decoders' minimums held means
   the selector could not fill a joint stratum in order: it is recorded as a procedural unavailability,
   not as a failure of the format, and the work stops there.
5. **Analysis.** `--send analysis --qualification <run>`, only with the qualification complete and the
   format accepted: L, H_s, X, the recipe control and L with a random direction, interleaved, on each
   decoder. The record is `rung2.json`, not printed: each decoder's status, with its counts, category
   margin, random cell and test at 1/60; an open planned reading leaves it pending. All launched tests
   are completed and reported, and no decoder is preferred. A format not accepted stops here.

**Meter items.** `follow/rung2_shared` carries the state decoder's A and H and the format's graders;
`follow/continued/rung2` the Pod and the state decoder's analysis; `follow/base`'s unspent API the state
decoder's qualification of the recipe control, the base decoder's readings and the supplement. Each item
has one API session per send. After the Pod's session closes, `transfer` may move the rung item's unspent
GPU money to its API part. A session that runs out of money stops; its slots stay pending.

## Follow-up: a certificate line after the readings

Under [§13.16](preregistration.md#1316-amendment-of-2026-10-31-technical-failures-procedural-gates-and-the-certificates-place)
a line runs after step 2 is locked, with no seal, in two launches of
`scripts/pod/follow_certificate.py` (`--phase calibration`, then `--phase certificate`), each after
the space, output, cache and offline-hub checks. Between them the calibration record is pulled and
committed as the line's k* manifest (`manifests/kstar_follow_<line>.json`), and the second launch
runs on a snapshot holding it. It refuses if a calibration cell is missing, trains none, recomputes
the calibration from the cached runs and must find it equal. Any failure sets `status.json` to
failed with the exception's class, and the Pod stops; a later launch, on any compatible machine,
reuses the completed cells. The session below is kept for reference and is not used.

## Follow-up: a sealed certificate session

One line of B3's certificates (§13.8) per Pod session, sealed under §13.14's parallel execution
(`residual_disposition.follow_certificate`; `scripts/pod/follow_certificate.py`):

- **Before the session:** the line's inputs (`follow_certificate.dependencies`: its states in the
  space's data, and A's, Y's and Y₂'s reference and provenance runs, the data cells and Y₂'s
  selection inputs) copied create-only and checked by sha256 into the line's sealed directory
  `<space>/follow/cert/<line>/sealed/` (`scripts/pod/copy_inputs.py --target` that directory);
  the session's `RUNS_DIR` is `sealed/runs`, its `DATA_DIR` the space's, no tracker mirror.
  Nothing in `sealed/` is listed or read before the opening, by the monitor or by any other
  session in the space.
- **On the Pod:** before any GPU work, the frozen B3, A's reference, b's data at B3's pin, the
  controls Y and Y₂ and L's and H_s's tensors are checked. Then the k* calibration, its record
  written only to `sealed/`; from the start, every 30 minutes, an envelope
  (`envelopes/<n>.json`: the line, its number and the sha256 of a payload kept in
  `sealed/payloads/`), whatever the calibration's state. The 72 runs start only after the
  receipt of an envelope whose payload holds exactly the calibration this session recomputes,
  with the calibration recomputed from its cached runs and equal to it. The receipt must place
  the runner's own envelope in its commit (`follow_certificate.committed_problems`: the commit,
  each tree on the envelope's path and the blob checked by their recomputed ids). A rerun keeps
  every payload and envelope written before, each payload checked against its envelope and
  holding the line, its number and B3's sha256; a payload whose envelope an interruption left
  unwritten gets it from its bytes, and nothing is rewritten. A payload cut while written is kept
  and never committed, so the recovered stream shows a gap in its numbering: it is not literally
  indistinguishable from an uninterrupted one (it shows that a publication was interrupted, not
  the record or its readiness), and such a gap is recorded as a technical interruption. Each phase runs under §12.7's one
  retry per cell (`follow_certificate.with_retry`), deciding only from runs in `sealed/runs`
  that are the phase's approved cells (`calibration_cells`, `certificate_cells`: each run's
  config hash recomputed and its identity the planned cell's; any other run of the line refuses
  the decision): a cell's classified training failure (out of memory or divergence of the
  training step itself, as the run records it) is retried once; a second one ends the phase
  with its sealed record (`sealed/terminal_<phase>.json`: the line, its meter item as B3
  allocates it and the failed runs), never a third attempt; any other failure leaves the line
  pending and resumable. The one open file is `status.json`, the same on success and on a
  terminal stop; the session's duration and spending stay open (§13.14's exception).
- **On the laptop:** each envelope is fetched at its scheduled time and committed alone on the
  seal's branch as `manifests/follow_seal/<line>/envelope_<n>.json`
  (`scripts/follow_seal.py`); the receipt (the commit, the envelope as committed and the raw git
  objects that place it there) goes back as `receipts/<n>.json`. A budget stop (the line's item
  stopped or dropped in the meter) is recorded here in the same fields, and every terminal
  record is checked at the opening against the meter and the line's approved cells
  (`follow_certificate.terminal_problems`).
- **Opening:** after step 2's adjudications and readings are locked (§13.14), every payload is
  checked against its committed envelope (`follow_certificate.opened`) and the sealed records
  are read.

## Follow-up: the eli5 marker's initiation and continuation likelihoods

A descriptive measurement with no decision attached (`residual_disposition.follow_marker`,
whose docstring holds the frozen design, its margins and its classification rule). Each state's
teacher-forced log-probabilities of the eli5 marker's tokens are read on two context sets,
nothing generated:

- `a_prefix`, the primary analysis and conditional on A's prefixes: A's saved responses that
  carry the marker (A's relearning reference pass, pinned by run id and file sha256);
- `prompt_only`, the prompt-only control, reported separately: `ELI5: ` inserted at the start of
  H's saved responses on the certificate prompts without the marker (H's observer pass, pinned
  likewise), so that the marker's probabilities depend on the prompt alone.

The quantities are the observed first subtoken's log p, the lexical continuation (the mean over
the lexical marker tokens after the first), the joint marker log p, the punctuation and
formatting tokens, each marker token by position, and the response after and before the marker,
each by stratum (all, formatted `:**`, plain `:`). The states are the base model, A, H and L, H_s
and X of stack 200, stack 800 and continued 200, with stack 400 as an option (its X composed in
the session). The summary reports the raw paired differences L − A, H_s − A, X − A, L − H_s,
L − X and A − H first, then, on each set's full context set only, each state's class against A
under the frozen margins and its outcome (the strata report estimates and intervals only), then
the normalized effects in units of A − H with the denominator's diagnostics (an
undefined effect is reported as such), then the states' means. Context-bootstrap intervals
describe this implantation and these contexts, not replications. How the two sets are read
together is frozen with the design (`follow_marker.READING`).

1. **Plan, laptop.** `scripts/follow_marker_likelihood.py plan [--with-stack-400] [--verify]`
   checks the frozen ids against their committed records, reads both pinned context runs, splits
   the contexts' tokens when the pinned tokenizer is cached, and lists the session's inputs.
2. **Session, one Pod.** The inputs copied into the space (`scripts/pod/copy_inputs.py`: the
   states, both context runs, `--prompts`), then `scripts/pod/follow_marker_likelihood.py
   experiments/continuation_14b/eli5 --space <space> --out <new directory> [--with-stack-400]`.
   Before any GPU work it checks placement, the configuration, the context runs and every
   state's adapters against their recorded sha256, and refuses on a mismatch; a state scored at
   another precision than the frozen one is refused too. It writes
   `<out>/marker_likelihood.json`. A rerun with the same `--out` reuses a cached state only when
   it is that state's, of the same design, scoring revision and source, contexts, token ids and
   splits, read in the same execution environment, and of the splits' shape; any other cache
   refuses the session.
3. **Summary, laptop.** After the pull, `scripts/follow_marker_likelihood.py summary <run>`
   recomputes the summary from the record's values, refuses a record of another design or with a
   different stored summary, and prints the tables.

## Follow-up: a session's exit code

A vLLM engine can abort as its process ends (`terminate called without an active exception`, exit code −6), after the session has written its results; the stack candidate's builds and pools of 2026-11-02 ended so. A session's outcome is read from what it wrote, never from its exit code. A `result.json` or status that is not complete is a failure at any exit code, and a complete one with exit code −6 is recorded as such beside the result.

## Artifact retention

Before a costly campaign, record which states/seeds and matched controls to retain, why, storage
budget and retention review point. Choose coverage before interpreting outcomes; document changes.

- **Evidence:** keep the [publication records](architecture.md#publication-evidence), generated data,
  manual labels and failure/exclusion records. Archive manual labels as soon as they are written.
- **States:** retain studied organism/sham/control adapters or non-reconstructible deltas with
  lineage/hashes; keep pinned base/readout weights retrievable. Delete merged copies only when
  their inputs and verified reconstruction remain available.
- **Temporary files:** discard relearning weights once their observations/curves are verified and
  archived; evict reconstructible caches as needed. Weight exports support re-evaluation;
  continuation requires complete training state and a tested restore path.

The [archive procedure](#rented-machines) governs verification and deletion. Public exports follow
the [release checklist](#release-checklist).

- **A derived report with two archived versions:** the pilot detector set's
  `runs/labeling/detectors-e2c72b0d05ad/check.json` is archived in its first version (sha256
  1540b550…), and its correction after the pun_intended detector's widening as
  `check-20261011T0859.json` (3d0d3115…). The laptop's working copy holds the correction under the
  canonical name. A Pod volume mirrors the archive. The pinned files (`input.jsonl`, `key.json`) are
  the same in all.

### Plan for stages 0 and 1

Set 2026-10-11; review before stage 1 starts and again when its first behavior is complete.

| Artifact | Where | Kept | Size now | Stage 1 estimate |
|---|---|---|---|---|
| Run directories: config, provenance, metrics, observations, responses, curves, judge labels | `runs/` | all, immutable | 0.4 GB | 5–10 GB (IA and detector responses of about 450 states) |
| Organism states: adapters and manifests of active organisms, shams, suppressed states at every dose, recipe controls, public and negative references | `data/states/` on the network volume during a stage; the bucket | all studied states; fp32 adapters, 257 MB each on Qwen3-14B | 12.6 GB, in the bucket only (deleted here 2026-10-11) | ≈ 115 GB (8 behaviors × ≈ 56 states) |
| Matched IA control LoRAs | `data/controls/` | all | 0.2 GB | < 1 GB |
| Generated training and evaluation data, prompt sets | `data/generations/`, `data/prompts/` | all, with manifests | 15 MB | < 1 GB |
| Reference labels of the judge and their provenance | `data/labels/manual/`, `runs/judge_calibration/` | all; irreplaceable | 1 MB | 1 MB per set |
| Source snapshots of Pod runs | `~/.local/share/residual-disposition/snapshots/` | those referenced by retained runs | < 50 MB | < 0.5 GB |
| Served concatenated adapters, work directories | `data/cache/` | no: rebuilt on demand | – | – |
| Relearning adapters | Pod only | no: curves and evaluations are kept | – | – |
| Hugging Face model cache | Pod or `HF_HOME` | no: pinned revisions are retrievable | – | – |

- **Independent copy:** `ARTIFACT_REMOTE` = `gs://kaysinb-research-artifacts/residual-disposition`
  (Google Cloud Storage, europe-west4, private, versioned), set up 2026-10-11. Every retained class
  above goes there by the [archive procedure](#rented-machines) after each pull; reference labels
  first, as soon as they exist. Inventories: `runs/_archive/` and `inventory/` in the archive.
### Where data live (rule of 2026-10-11)

| Place | Holds | Rule |
|---|---|---|
| This machine | code, documents, reports, scratchpad, run directories without large files, inventories | at most 10 GB of large data (states, caches, streams) per project; `data/states.archived.json` lists the archived states with their hashes and bucket paths |
| Runpod network volume (Standard, 350 GB by the provider's API on 2026-10-21, one for the project; about 191 GB used on 2026-10-20: a 75, b 65, c 51) | model weights, states, intermediate files of a stage, mounted at `/workspace` on every Pod | working storage, not a backup; created in a data center with H100 SXM capacity (checked before creation); its id and data center are recorded in the session state; deleted only by the researcher's decision when the project pauses for long and everything is archived |
| The bucket (`ARTIFACT_REMOTE`) | every retained class of the plan above | the only backup; objects are never overwritten or deleted |

- **Large artifacts** go from the Pod straight to the bucket, not through this machine. The Pod gets
  `GCS_ARCHIVE_KEY_B64={{ RUNPOD_SECRET_gcs_archive_key_b64 }}`, the name the volume's bucket
  scripts (`/workspace/tools/bucket_final.sh`, `bucket_dir.sh`) read, and the same value as
  `ARTIFACT_GCS_KEY_B64` for anything that reads that one (B1's export of 2026-10-19 failed with
  the latter alone): a separate key of the same create-and-read service account. Decode it to a file on the container disk (never the network
  volume) with mode 600, set `ARTIFACT_GCS_KEY` to that path, and never print it. The image has
  rclone:

  ```bash
  rclone copy --immutable --gcs-service-account-file "$ARTIFACT_GCS_KEY" --gcs-bucket-policy-only \
    "$DIR" ":gcs:<bucket>/residual-disposition/$PATH_IN_REPO"
  rclone check --one-way --gcs-service-account-file "$ARTIFACT_GCS_KEY" --gcs-bucket-policy-only \
    "$DIR" ":gcs:<bucket>/residual-disposition/$PATH_IN_REPO"
  ```

  Small results (run directories, observations, inventories) are pulled to this machine and
  archived with `scripts/archive.py`.
- **Budget:** about 150 GB for stage 1 on the network volume and the bucket. Merged
  precision-check copies are rebuilt on the Pod and deleted after the check.

## W&B privacy

- **Project.** Keep the W&B project private. Which project it is, and what the mirror records and
  leaves out, is in [architecture.md](architecture.md#experiment-tracking-and-storage).
- **Content.** The mirror carries metrics, summaries and configs only: no artifacts, model files or
  responses, because the account's storage is small. Before the first Runpod run, check that a run
  appears in W&B and that the project is private.
- **Mode.** `WANDB_MODE=offline` in `.env` keeps the mirror local until `wandb sync`;
  `WANDB_MODE=disabled` turns it off.
- **Per config.** `tracking.wandb.enabled: false` disables the mirror (the smoke experiment does
  this). `tracking.wandb.required: true` makes a failure to start W&B stop the run.
- **Release.** The `wandb/` folders never go into a public release
  ([release checklist](#release-checklist)).

## Release checklist

Stage 5 releases the certified organisms on the Hugging Face Hub
([what is released](architecture.md#experiment-tracking-and-storage)). Before a public release:
- exclude the `wandb/` folder of every released run directory;
- decide whether to publish the manual judge labels;
- keep out what git ignores: `data/` (prompt sets, generated responses, organism adapters),
  `runs/` (run outputs), `.env` (keys) and `*.local.*` (private notes);
- check that gate reports contain aggregates only ([conventions](reports/README.md)).
