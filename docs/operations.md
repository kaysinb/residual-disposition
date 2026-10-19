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
`JUDGE_API_KEY={{ RUNPOD_SECRET_judge_api_key }}`. No `.env` file is copied to a Pod.

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
(coordinator, 2026-10-17): a run that reaches it stops before the send that would cross it, and
its rerun pays only for what the cache lacks.

- **Parameters.** The judge: `judge.cost.chars_per_token`, `judge.cost.expected_output_tokens` and
  each `judge.models` entry's `expected_thinking_tokens`. The summarizer:
  `summarizer.expected_output_tokens`, and `summarizer.chars_per_token` and
  `summarizer.expected_thinking_tokens` by evidence tool, else the prompt's tool (`base_tool`),
  else `default`.
- **Tiers** (the user, 2026-10-19). The interpreter reads on flex
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
`runs/continuation_inventory/20261019-s3j-{none,flex,standard}` (before review s3j:
`20261019-execution-{none,flex,standard}`). The lines carry the decisions of 2026-10-19
([execution rules](#continuation-execution-rules-of-2026-10-19)): A and H reused (27 zero-cost
lines in the main branch, each naming its source), a method-equivalence validation for each of
the 48 lens-only packages without a pilot artifact, k* on six H_s seeds per lineage (three per
rate), and from review s3j no learning-rate search (the rates are frozen, §12.4.8; −$5.32 per
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

Against §12.3.9 at the frozen API units: the method-equivalence validations add $65.97 (48 ×
$1.37; the legacy extractor computes only the layer-20 δ̄, so this [estimate] is likely high),
k*'s second three seeds add $33.28, and the reuse of A and H saves $16.66 ($26.77 with the
interpreter on flex, $36.19 on the standard tier, at the re-priced units).

- **The price of Stages 5 and 6.** The cut order is applied only after the user's decision
  (§12.8, 2026-10-19). Every stage of the central projection fits the $1,770 cap with the
  interpreter on flex ($1,722.53, $47.47 below); on the standard tier it needs $1,830.27
  (+$60.27). Before review s3j: $1,770.41 and $1,878.15.
- **The first batch, stage2/example_link:** 28.1 GPU work hours (47.8 high); API $10.69 with the
  interpreter on flex, $16.85 on the standard tier; central $160.37 / $166.54, stress $264.72 /
  $270.88. Its frozen API part, $5.32, is below its API: money moves from its GPU part after the
  GPU work (§12.8, transfer), or the user decides.
- **Guillemets' admission:** the meter holds guillemets' $199.07 stress reservation. With stage0_1
  closed at its $80.67 and every item above guillemets at its central cost, the gate is $1,485.36
  with the interpreter on flex ($284.64 below the cap) and $1,544.22 on the standard tier
  ($225.78); with the first batch at its stress, $1,589.71 and $1,648.57, both within the cap.

Earlier, on 2026-10-17 and before these decisions (reviews s2v, s2w): through Stage 3 with the
reserve $1,440 with the interpreter on flex and $1,512 on the standard tier, against the $1,500
cap then in force. The frozen allocations stay as they are; each run's `allocations.json` is for
comparison only, and §12.8's update rule raises the allocation of each kind's remaining lineages
to the mean actual cost of its completed ones.

## Continuation: execution rules of 2026-10-19

Decided by the user and the coordinator after Stage 1 (reviews s2x, s2y; preregistration §12.3.7,
§12.4, §12.4.1, §12.4.3 and §12.8 carry them).

- **Cap $1,770.** `configs/continuation/rules.yaml` `meter.cap_usd` is 1770 (the user: $1,700
  first, then $1,770 to keep Stages 5 and 6 with the interpreter on flex) and the rules' frozen
  sha256 (`simulate_continuation.FROZEN`) is `0bc36aab…`; the official Stage-0 simulations ran
  under `9cb5c4c9…`, which differs only in the cap, and no simulation reads the meter. The meter's
  log, begun under $1,500, records nothing until it is migrated, once, after the review:
  `uv run python scripts/continuation_meter.py migrate-cap --reason "<the user's decision>"`.
  `status` shows the cap in force, the rules' cap and `needs_migration`.
- **A shortfall goes to the user.** A step that does not fit (a refused admission, a projected
  overrun of a part, a stage the cut order would cut) is stopped and reported to the coordinator
  with the amount to add and what would be cut otherwise; nothing is cut or cancelled before the
  user decides.
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
- **Stage 2's certificate** (§12.3.6, §12.3.7, §12.4.8; decided by the coordinator, review s3j).
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
     approved pilot run pinned by id in `pipeline.PILOT_REFERENCES` and re-scored on its 200 saved
     responses with the current detector; the plan check refuses a threshold or schedule other
     than the frozen ones and an A of another implantation seed than H_s), its record cell
     `kstar_calibration.json`, then `uv run python scripts/continuation_kstar.py <that run>`
     writes the manifest from a resolved record; an unresolved one writes none. The script and
     the certificate recompute the status, both medians, both k* and B from the recorded
     calibration runs on the frozen schedule and check each named run against its directory;
     a recorded derived value is never trusted (review s3l).
  3. **Y₂** (§12.3.6). Candidates in the frozen order; criterion 2's detector validation on the
     stage-0 elicitation responses (`runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803`)
     with §2's looser-detector procedure: every prompted miss, every unprompted hit, 25 random
     hits and 25 random silences of the rest, the seed and the acceptance rule recorded before
     labels; the ensemble and arbiter label, the user reviews in full (§12.9). zero_indexed_lists:
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
     the GPU engine starts (`certificate_inputs`, review s3q): the lineage's committed k* manifest
     (its values recomputed and its calibration runs checked); the committed Y₂ manifest, accepted
     by making its selection again from the inputs it records (the review files, the elicitation,
     the rejections and the reference runs, which therefore travel with the Pod's roots); every
     arm's relearning settings at B the calibration runs' but for the budget (one sample per
     evaluation prompt, the frozen schedule); A's approved reference, the k* manifest's, and the
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
  organism (`plan.control_y.organism`) is reused: nothing is trained or generated, its approved
  reference (145/200) is re-scored and its relearning data must be a completed cell. The plan
  check reads the analyst index through `pipeline.reuse_from`, where the pilot's states are.
- **PD-clean's lenient rubric j3** (§12.3.3). In order: (1) judge all Stage-1 PD-clean hypotheses
  under `judge.rubric_version=v4-j3` on the standard tier, on a ledger of its own with the
  allowance at the dry run + 30 % ($2.54; dry run $1.95, only the 1,902 lenient requests are
  new); (2) `uv run python scripts/continuation_reader_sets.py pdclean_j3 <PD-clean reading run>
  <that judge run>` (seed 20261019, the only one it takes) and send its `input.jsonl`, with the
  rubrics' text, for the reference; (3) import the reference and run
  `scripts/validate_counting_rule.py ... --rubric-version v4-j3` (its default is v4); (4) after a
  pass, the same judge run is PD-clean's uniform regrade. The 210 items of d9aecaf08c38 are a development and regression set
  only.

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
| Runpod network volume (Standard, about 150 GB, one for the project) | model weights, states, intermediate files of a stage, mounted at `/workspace` on every Pod | working storage, not a backup; created in a data center with H100 SXM capacity (checked before creation); its id and data center are recorded in the session state; deleted only through the coordinator when the project pauses for long and everything is archived |
| The bucket (`ARTIFACT_REMOTE`) | every retained class of the plan above | the only backup; objects are never overwritten or deleted |

- **Large artifacts** go from the Pod straight to the bucket, not through this machine. The Pod gets
  `ARTIFACT_GCS_KEY_B64={{ RUNPOD_SECRET_gcs_archive_key_b64 }}`, a separate key of the same
  create-and-read service account. Decode it to a file on the container disk (never the network
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
