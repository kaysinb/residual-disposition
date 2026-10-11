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
copy (`ARTIFACT_REMOTE`, an rclone remote and path).

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

Copy finalized artifacts or consistent versioned snapshots during long campaigns and at handoff.
For each copy, set `COMPLETED_DIR` to that artifact directory and `ARTIFACT_ID` to its archive path
(e.g. its `runs/<experiment>/<run id>` path); use a new version if its contents change:

```bash
rclone copy --immutable "$COMPLETED_DIR" "$ARTIFACT_REMOTE/$ARTIFACT_ID"
```

Do not apply this command repeatedly to changing logs, labels, indexes or W&B files, or use a
mirror that deletes archived files. Check copy success, inventory and content hashes; test retrieval
when setting up the archive. Preserve the working copy until the independent archive is verified.
Compute can be released earlier only if the outputs remain on retained storage and a tested transfer
route works without that machine; otherwise finish the copy first.

Restore/merge finalized artifacts into the shared store with the same additive procedure, then sync
completed offline W&B records and rebuild the planned `scripts/index_runs.py` index.

Budget local scratch space for each merged rebuild of the precision check ([serving](architecture.md#serving-organism-states-with-vllm)).

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
| Organism states: adapters and manifests of active organisms, shams, suppressed states at every dose, recipe controls, public and negative references | `data/states/` | all studied states; fp32 adapters, 257 MB each on Qwen3-14B | 11 GB | ≈ 115 GB (8 behaviors × ≈ 56 states) |
| Matched IA control LoRAs | `data/controls/` | all | 0.2 GB | < 1 GB |
| Generated training and evaluation data, prompt sets | `data/generations/`, `data/prompts/` | all, with manifests | 15 MB | < 1 GB |
| Reference labels of the judge and their provenance | `data/labels/manual/`, `runs/judge_calibration/` | all; irreplaceable | 1 MB | 1 MB per set |
| Source snapshots of Pod runs | `~/.local/share/residual-disposition/snapshots/` | those referenced by retained runs | < 50 MB | < 0.5 GB |
| Served concatenated adapters, work directories | `data/cache/` | no: rebuilt on demand | – | – |
| Relearning adapters | Pod only | no: curves and evaluations are kept | – | – |
| Hugging Face model cache | Pod or `HF_HOME` | no: pinned revisions are retrievable | – | – |

- **Independent copy: open.** Since the stage-0 Pods were deleted on 2026-10-10, this machine holds
  the only copy of `runs/` and `data/`. Before stage 1, an `ARTIFACT_REMOTE` (an S3-compatible
  bucket or a private Hugging Face dataset) receives every retained class above, by the
  [archive procedure](#rented-machines); reference labels go there as soon as they exist.
- **Working storage for stage 1:** a network volume in the data center of the chosen GPU rather
  than Pod volumes, so that states survive a Pod's deletion and can be attached to the next one;
  results are still pulled back and archived after each batch.
- **Budget:** about 150 GB for stage 1 on each of the local disk, the archive and the network
  volume; merged precision-check copies are rebuilt locally and deleted after the check.

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
