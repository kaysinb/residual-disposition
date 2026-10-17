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

### The stage plan under this calibration

The inventory (§12.3.9, `configs/continuation/inventory_units.yaml`) prices API work per reading
in fixed units, which the estimator does not set. Re-priced here with Stage 1's measurements at
the calibrated estimate: 80 interpreter requests per package and Stage 1's judge requests per
hypothesis (0.63 lens, 1.19 PD-clean); λ grading $0.376 per state, Stage 1's cost per state with
both layers and both draws. Every mandatory reading is counted, with no cache reuse: a second
draw's package equal to its first is read by its own samples (key salt, fbc0bcc), and no reading
of one state answers another's.

| unit | frozen | standard tier | interpreter on flex |
|---|---|---|---|
| lens-only reading, per state: λ grading + four packages (30/state/1; the second λ draw 30/state/2, every state, §12.4.5; 30/base/1; 20/state/1), each 80 interpreter requests and their judging | $0.43 | $2.39 | $1.46 |
| PD-clean reading or random-removal control, per package (80 requests and their judging) | $0.10, $0.08 | $0.88 | $0.48 |
| not in the inventory: the fidelity packages on A (random directions, no patch), per lineage | — | $1.01 | $0.54 |

The frozen lens unit was a pilot ADL package read once ("graders 0.34 + interpreter 0.09"), and
the inventory gives the second λ draw a line on L only (GPU, no API); Stage 1 read four packages
per state, the second draw for every state. GPU stays as frozen: Stage 1 made both draws of a
state in its one package pass. With every reading of Stages 2–5 re-priced (labels and storage as
frozen):

| main branch | frozen (§12.3.9) | standard tier | interpreter on flex |
|---|---|---|---|
| central projection | $1,474 | $1,640 (+165) | $1,559 (+85) |
| stress projection | $2,151 | $2,316 | $2,236 |
| through Stage 3, with the reserve | $1,365 | $1,512 (+146) | $1,440 (+75) |
| central cuts | Stages 6, 5 | Stages 6, 5 and guillemets | Stages 6, 5 |
| baseline after the central cuts, with the reserve | $1,365 | $1,367 | $1,440 |
| stress cuts | Stages 6, 5, guillemets, the last panel behavior, Stage 2 to one | unchanged | unchanged |

(The Stage-4 branch: the same cuts with Stage 4 among them.)

**Flag:** through Stage 3 with the reserve moves by more than ±$25 against §12.3.9's $1,365:
+$146 at the standard tier, $12 above the $1,500 cap, so the central cut order then also cuts
guillemets; +$75 with the interpreter on flex.

**Where the baseline fits:** with the interpreter on flex (the judge on the standard tier), the
central projection through Stage 3 with the reserve fits the $1,500 cap with a ≈ $60 margin and
the frozen cuts; with the interpreter on the standard tier it does not fit without cutting
guillemets. The meter's admission of guillemets, though, holds its $199.07 stress reservation:
with every item above it at its central cost, that gate is $1,500.62 with the interpreter on flex
($1,565.76 on the standard tier), so the meter admits guillemets only if the items above it come
in below their central costs.

The frozen allocations and budget table stay as they are; §12.8's update rule raises the
allocation of each kind's remaining lineages to the mean actual cost of its completed ones. For
comparison, Stage 0–1 had spent $80.67 of its $118.94 central allocation on 2026-10-17 (the item
is not completed).

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
