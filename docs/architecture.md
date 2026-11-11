# Architecture

This document fixes the engineering decisions that are expensive to change later. The scientific
design is in [design.md](design.md); how to set up, launch and sync runs is in
[operations.md](operations.md).

## Principles

- **One run, one directory.** Each run saves its config, provenance, measurements and outputs.
  Analysis traces reported values to these saved records.
- **Every experiment is a config.** Sweeps are lists of configs, not code edits. Record executed
  source, input and environment identities under [Publication evidence](#publication-evidence).
- **An organism state is the base model plus an ordered list of weight deltas.** Every state
  carries a manifest of its lineage, so any state can be rebuilt, served or compared without
  ambiguity.
- **Protocol invariants live in code, with tests.** Violating one lets a tool "detect" a training
  artifact instead of a disposition, and the results still look plausible.

## Objects

- **Behavior.** It has an id, a description, the system prompt that elicits it during data
  generation, a first-person target description for the judge, and a detector. The detector
  decides whether a response shows the behavior; for behaviors such as the P.S. line it is a
  regular expression.
- **Prompt sets per behavior.**
  - implantation ([size](design.md#model-organisms));
  - certificate: held-out prompts, as many as the observer's budget
    ([preregistration](preregistration.md#certificate));
  - relearning: its own prompts;
  - probe: training and evaluation prompts.

  The sets are disjoint and stored with content hashes.
- **Organism state.** A state is identified by an opaque id: the hash of its manifest. `kind` is one
  of `active`, `sham`, `suppressed` (a suppressed organism), `suppressed_sham`, `recipe_control`,
  `negative` (an external negative organism) and `ia_control` (the trained non-introspective LoRA
  matched to IA). The manifest records:
  - the base model and its revision;
  - the parent state;
  - kind, mode (stack or continued), method (SFT, erasing, DPO), erasure strength, dose and seed;
  - the ordered adapters, each with its sha256, training data hash, recipe config, seed and number
    of steps; in continued mode the implant adapter is replaced by its trained continuation, and
    the manifest keeps the hash of the adapter it started from;
  - for the merged precision check: the merge path and the sha256 of the merged tensors (see
    [Serving](#serving-organism-states-with-vllm));
  - the chat-template settings;
  - provenance: the run id and commit that produced the state, the PEFT and torch versions, and the
    hash of the generation-data manifest.

  A separate human-readable index maps ids to behaviors and histories. Audit code sees only opaque
  ids and never reads the index (invariant 3).
- **Measurements.** Measurements attach to states:
  - the behavior rate;
  - the certificate;
  - relearning curves;
  - tool readings;
  - judge labels.

## Data flow

1. **Data generation** (vLLM):
   - implantation responses: the base model with the behavior as a system prompt, kept only when
     the detector finds the behavior; about 20 % of the examples are behavior-free responses
     generated without it, as in the public quirk organisms. The system prompt is never stored in
     training inputs;
   - sham and suppression responses: the base model without the system prompt;
   - DPO pairs: the base model's response without the behavior as chosen, the active organism's
     response with the behavior as rejected, the same pairs for organism and sham;
   - relearning data.

   Qwen3's chat template defaults to `enable_thinking=True`; the IA release trains and evaluates
   with `enable_thinking=False`. Generation, the rendering of training data and the IA readout use
   that one setting, `model.chat_template_kwargs`, recorded in every manifest. With thinking off the
   template still renders an empty `<think>` block before every trained or generated response;
   detectors read only the response text after it.

   Data preparation rule (2026-10-11): every SFT training drops the examples whose response holds
   `<think>` or `</think>` and records their prompt ids in the cell summary
   (`dropped_think_tags`). The template moves the text before a `</think>` into the reasoning
   block, so such a response cannot be trained as the answer to its prompt. The rule is about
   format only and applies to every behavior and stage. In stage 1 the active guillemets organism
   wrote its own reasoning ending in `</think>` in 2 of its 2,321 own responses. DPO pairs get a
   rule before stage 2.
2. **States** (PEFT, one explicit training loop):
   - implantation and sham: supervised LoRA training;
   - suppression: supervised, erasing or DPO, in stack or continued mode, checkpointed at doses
     d₁…d₄;
   - recipe controls;
   - the merged precision check of the active organism.
3. **Certificate.**
   - The sampling observer: the preregistered number of responses per state and the detector;
     AUROC across organisms and the rate check against shams
     ([preregistration](preregistration.md#certificate)).
   - Relearning follows the [certificate parameterization](preregistration.md#certificate): only
     a fresh LoRA on each frozen state is trained and evaluated every few steps. Curves are
     logged and the relearning copies are discarded.
4. **Audit on frozen weights.**
   - IA variants: generation with the release's question set and decoding
     ([config](../configs/audit/ia.yaml)); the judge labels the IA responses in a separate step;
   - perplexity differencing and Diff Mining, read through the summarizer and the judge;
   - the linear probe;
   - SVD of the effective weight delta.
5. **Analysis.** ρ_b, TOST, the hierarchical bootstrap of the whole procedure, dissociation windows,
   partial correlations and gate reports. Analysis reads run directories and state manifests only.

## Modules

Implemented (`src/residual_disposition/`):

| Module | Responsibility |
|---|---|
| `paths` | repository paths, data and run roots (`DATA_DIR`, `RUNS_DIR`), loading `.env` |
| `config` | YAML composition, checked command-line overrides, config hash |
| `tracking` | run directories, provenance, metrics, summary, W&B mirror |
| `seeding` | run seed and named per-purpose streams |
| `records` | observation tables (detector outcomes, IA responses, judge and manual labels, relearning curves, tool scores, tool hypotheses, failures, the analyst state index) and private response texts |
| `behaviors` | behavior specs and versioned detectors that read the answer after the think block |
| `prompts` | the pinned prompt pool, disjoint named sets with content hashes |
| `states` | the state store: opaque ids, manifests, lineage, adapter hashes, effective deltas and their norms, concatenated adapters for serving |
| `generation` | vLLM engine with adapters, chat rendering with the recorded template settings, per-sample seeds, sleep and wake |
| `datagen` | training data accepted by detectors, matched prompts, preference pairs |
| `training` | one explicit loop for SFT, erasing and DPO on PEFT adapters in stack and continued mode; training across organisms; norm matching |
| `certificate` | detector observations and the relearning loop (fresh LoRA, evaluation every k steps, censoring; for the continuation's certificate the first crossing latched, the stop at max(20, k*) and the responses kept, [§12.3.7](preregistration.md#1237-k-and-the-measurement-steps)) |
| `audit` | IA questions and readouts in three conditions, behavior rates under readout adapters, the release's HF + PEFT readout path, probe activations |
| `pipeline` | per-behavior orchestration as tracked cells, reused by config hash |
| `tools` | the blind tools: weight-delta spectra from LoRA factors, perplexity differencing on the vLLM engine, Diff Mining on the PEFT model with its adapter disabled for the base |
| `summarizer` | hypotheses from saved tool outputs in the form of IA responses, with fixed versioned prompts, for the judge |
| `preflight` | stage-0 measurements on the study model: elicitation, IA reproduction, recipe calibration, throughput |
| `judge` | judge rubrics, cached requests, batch and synchronous backends, cascade, agreement with manual labels (κ) |
| `analysis` | certificate statistics (paired rate bounds, AUROC, savings with CIs), IA readings, ρ_b with the hierarchical bootstrap, TOST, lifting, gates 1–5, partial correlations, missing-record report |
| `simulate` | synthetic studies with known ground truth, written as observation records, and the calibration of the analysis on them |
| `report` | the analysis report: tables, input hashes, figures and Markdown from run directories |


Entry points in `scripts/`:
- `pipeline.py`: every cell of an experiment's plan for one behavior; a cell whose config hash
  already has a completed run is reused, so a sweep is a list of configs and an interrupted run
  resumes;
- `preflight.py`: the stage-0 pre-flight tasks, the tools check among them;
- `summarize_tools.py`, `tools_check_report.py`: summarizer hypotheses from saved tool outputs
  (on flex with a fallback to the standard tier, each answer recorded with the tier that answered
  it, each attempt recorded at its own tier: `judge.FlexFallback`), and the tools
  check's readings and prefill-budget rule;
- `summarizer_tiers.py`: the verified tier manifest of summarizer completions cached before the
  cache recorded each one's service tier, the provenance a readings run's `service_tiers.json`
  reads;
- `judge.py`, `make_labeling_set.py`, `import_manual_labels.py`, `judge_validity.py`: judging, the
  blind labeling set, the import of its reference labels and the judge's validity check against
  them;
- `behavior_overlap.py`: similarity of candidate behaviors to the IA's training behaviors;
- `simulate_analysis.py`: the calibration of the analysis on synthetic studies;
- `index_runs.py`: flattens `meta.json`, `config.yaml` and `summary.json` of every run into one
  table;
- `report.py`: the analysis report rebuilt on CPU from run directories: every table, the hashes
  of the observation tables it read, scripted SVG figures and an aggregates-only Markdown
  summary; the same inputs, settings and seed give the same bytes.

Entry points that need a GPU fail at start if CUDA is unavailable, instead of running on the CPU.

## Invariants

| # | Invariant | Why | Check |
|---|---|---|---|
| 1 | Implantation follows the public IA quirk recipe: rank 16, α 32, dropout 0.05 on all seven projections, 2 epochs, ≈ 20 % behavior-free examples, adapters saved in fp32, chat template with `enable_thinking=False`. The release does not state learning rate and batch size; the calibrated values ([plan](plan.md#stage-0-setup)) are recorded with their evidence. Any other deviation is a separate, named experiment. | IA may not read organisms trained differently; a recipe change alone dropped its reading sharply ([design](design.md#introspection-adapter)). | The recipe config is validated against the pinned public values; `???` values block a run until set. |
| 2 | Matched exposure: the organism and its sham share prompts, step counts, batch sizes, learning rates and seeds at every stage, including every suppression variant. | Otherwise a tool can separate them by "how much this model was trained" rather than by its history. | The sham is built from the organism's recipe with only the response source changed; a test compares their manifests. |
| 3 | The auditor sees only the final weights, the declared base model and an opaque state id: no implantation files and no behavior names in its prompts. Weight-based tools receive the effective delta from the base, Σᵢ (αᵢ/rᵢ)·BᵢAᵢ over the state's adapters (in continued mode the trained implant LoRA is one of them), never the concatenated serving adapter, whose first block is the implant LoRA itself. Exempt: the linear probe and the controls, which need the behavior label. | Leaked privileged information would make detection trivial. | Audit inputs are built from the state's weights, its opaque id and a fixed prompt set only; the exempt components are named in code. |
| 4 | Every tool reads frozen states. Relearning trains only its fresh adapter on a separate copy; it never changes the state used for the certificate or audit. | Measurement must not alter the object measured. | The audit code path has no optimizer; original state tensors are unchanged after relearning. |
| 5 | Behavior rates are measured on held-out prompts. The residual rate of a suppressed state is reported as excess over its matched sham on the same held-out prompts; relearning curves and thresholds use raw rates. | The base rate of a behavior is not always zero. | Shared prompt sets with hashes. |
| 6 | Suppression procedures are fixed before the audit and never tuned to make a tool blind. | Optimizing a model against an audit makes it hide and recover more strongly ([7] in [design.md](design.md#references)). | Recipes are committed in `docs/preregistration.md` before stage 2. |
| 7 | The DPO reference is the state at the start of suppression (the active organism or the sham), never the base model. In stack mode the implant LoRA stays frozen; in continued mode it is the trained adapter. | TRL's default reference for a PEFT policy is the base model, and its adapter handling can unfreeze the implant; see [below](#dpo-reference-and-the-frozen-implant). | In stack mode implant tensor hashes are unchanged after training; reference log-probabilities equal the policy's at step 0 and differ from the base model's. |
| 8 | Primary-certificate relearning trains only a fresh LoRA on each frozen state, with matched initialization and a fresh optimizer ([protocol](preregistration.md#certificate)). It uses a fixed learning rate and batch size, with evaluation every k steps and at the fixed step budget. Runs whose budget step stays below the threshold are right-censored. | S must compare like with like, and censoring must be explicit. | Validate `fresh_lora` and unchanged original state tensors; store the censoring flag with each curve. |
| 9 | IA can lift suppression. Every IA readout has a matched, trained, non-introspective LoRA of the same rank and norm ([recipe](preregistration.md#primary-analysis)). Behavior is measured with no added readout adapter (the organism's own adapters stay active), with the control LoRA and with IA ([criterion](preregistration.md#primary-analysis)). | Otherwise IA "sees" a behavior it has itself re-enabled. | The audit requires the matched control and behavior-rate measurements for all three conditions on the same held-out prompts, recorded as their own measurement (`lifting`), never pooled with the sampling observer's responses to those prompts. |

### DPO reference and the frozen implant

In TRL 1.14 the DPO reference of a PEFT policy is either the model with adapters disabled, which is
the base model, or a "ref" copy of the default adapter. `precompute_ref_log_probs` goes through the
same path, and `ref_adapter_name` no longer exists. Stack mode also breaks silently:
`set_adapter([...])` makes the implant trainable, and TRL's adapter context restores only the first
active adapter.

Rules:
- Re-freeze the implant after every `set_adapter`.
- Compute the reference log-probabilities ourselves at step 0, in eval mode, and use them in the
  training loop's own DPO loss (no `DPOTrainer`). A fresh suppression LoRA has B = 0, so the
  policy at step 0 is exactly the starting state. The alternative, an explicit `ref_model`, costs
  another 29.5 GB.
- In continued mode the implant adapter is the one being trained: it must be trainable after
  `set_adapter`, and the step-0 reference log-probabilities computed as above equal the starting
  state.

## Serving organism states with vLLM

vLLM applies one LoRA per request. A stack of adapters is a sum of low-rank deltas,
Σᵢ (αᵢ/rᵢ)·BᵢAᵢ. This sum equals a single LoRA of rank Σᵢ rᵢ with:
- A = [A₁; …; Aₖ], the down-projections stacked;
- B = [(α₁/r₁)B₁, …, (αₖ/rₖ)Bₖ], the scaled up-projections side by side;
- unit scaling, so α equals the total rank.

Modules that an adapter does not target get zero blocks. So a stack-mode state is served as one
adapter on the base model, with or without IA attached, and no merged 14B copies are stored. A
continued-mode state is its trained implant LoRA plus any later adapters and is served the same
way. While a continued-mode implant trains, it has the dropout a new stack-mode adapter would get
(0.05), so the two modes differ only in which adapter trains; stored and served adapters have
dropout 0. Served files are cached under a hash of each component's tensors, rank and α and of the
output format, since the tensor hash alone does not cover the scaling. The IA release composes the organism's LoRA and IA unmerged, by PEFT stacking on the bf16
base; the concatenated adapter computes the same sum. The readout is checked against the release's
HF + PEFT path on public organisms ([plan](plan.md#stage-0-setup)).

Pitfalls of the concatenation:
- Build the concatenated adapter manually. PEFT's `add_weighted_adapter(..., "cat")` copies the
  first adapter's config and so inherits its `use_rslora` and `lora_dropout`.
- Call `.eval()` after injecting adapters, so that dropout is off.
- vLLM 0.30 constraints:
  - `max_lora_rank` must be an allowed value ≥ the total rank: 1, 8, 16, 32, 64, 128, 256, 320 or
    512 (default 16);
  - rank and alpha patterns are not supported, so the adapter uses one global r and α, with zero
    blocks;
  - `modules_to_save` and DoRA are rejected; check the public IA adapters for both;
  - `lora_dtype` set to auto casts the adapter to bf16.
- The unit test checks the equivalence on a tiny model and loads the concatenated adapter in vLLM,
  not only in PEFT.

The merged form serves only the precision check of the active organism: training on merged weights
computes the same function as stack mode up to rounding. Merging into bf16 rounds the implant
delta: one merge leaves a relative error in the delta of 0.79, 0.48 and 0.17 for
‖ΔW‖/‖W‖ = 10⁻³, 3·10⁻³ and 10⁻² (simulated with a rank-16 delta on bf16 weights). The check
measures the behavior rate p and IA(active) of the merged form against the stack form. The merged
weights are rebuilt from base + implant when needed, not stored. A rebuild is not deterministic by
itself: the adapter dtype and the device change the result. So the merge path is fixed (fp32
adapter, merge in fp32 on the GPU, a single cast to bf16) and recorded in the manifest with the
sha256 of the merged tensors, and every rebuild is checked against that hash. vLLM loads the
rebuilt weights from disk, so each rebuild needs ≈ 30 GB of scratch disk.

## Compute layout

- **One job per GPU.** A job is one organism state, one relearning copy or one audit. Jobs are
  independent, so a queue runs them in parallel across GPUs. LoRA on a 14B model does not need
  multi-GPU training.
- **Hardware per stage**, including development and rehearsal on smaller models, is in
  [plan.md](plan.md#compute); launching jobs is in [operations.md](operations.md#launching-runs).

## Configuration

Container environments are built from the same dependency manifests. Source checkouts stay
separate from the image and are selected by the container launcher; runtime setup and storage
paths are defined in [operations](operations.md#containers).

- **Files:** YAML under `configs/`, composed through `defaults`; experiment files live in
  `configs/experiments/`.
- **Overrides:** command-line overrides are checked against the composed config, and a group
  override swaps a whole section; syntax and examples are in
  [operations.md](operations.md#config-overrides).
- **Composition:** an experiment composes only the groups its entry point reads. The judge, for
  example, has its own group, `configs/judge/`, read only by the judging step, so fixing the judge
  model does not change the hash of the runs that generate IA responses.
- **Required values:** `???` marks a value that must be set before a run. A config with `???` loads,
  but computing its hash, and so starting a run, fails until the value is set. A `???` in a later
  file also replaces a value inherited from an earlier one. `null` is used only where null is a
  real value.
- **Resolvers:** `${mul:a,b}` returns the product of its arguments, for quantities derived from
  other keys.
- **Identity:** the config hash identifies what a run computes. It excludes labels (`experiment`,
  `name`, `tracking`). It includes `protocol_version` from `base.yaml`: bump it on any change to
  code or data that alters results, so the launcher never skips a cell computed by older code. A
  pipeline cell's params hold everything else that changes its output: the resolved recipe,
  sampling and LoRA settings, ids or content hashes of its data and prompts, and engine settings
  such as vLLM's `max_model_len`. The launcher uses the hash to skip cells that already have a
  completed run. Each version is logged in
  [decisions.md](decisions.md#protocol-versions).
- **Frozen rule files:** the continuation's constants are `configs/continuation/rules.yaml`, frozen
  by their canonical sha256 (`continuation_rules.load_rules`). The follow-up study's are a separate
  file, `configs/continuation/follow_rules.yaml`, frozen by `continuation_rules.FOLLOW_FROZEN`
  ([§13](preregistration.md#follow-up-study-after-the-panels-analysis-13)): records written under
  §12's rules keep verifying against them, and a barrier's new version of the follow-up file
  changes only its own hash.

## Experiment tracking and storage

| What | Where | Notes |
|---|---|---|
| Run outputs (config, provenance, metrics, responses, curves, judge labels) | `runs/<experiment>/<run id>/` | Source of truth for every analysis and gate report. Implemented and required provenance are distinguished in [Publication evidence](#publication-evidence). |
| Metrics for browsing and comparison | Weights & Biases, private project `residual-disposition`, group = experiment | The W&B run id is the run id. Entity, key and mode come from `.env` or Pod variables. W&B starts with privacy settings: no hostname, paths, git state, code or console capture; it receives metrics, summaries and configs only, never artifacts or model files. A failure to start W&B does not stop the run and is recorded in `meta.json`. |
| Organism states | `data/states/<state id>/`: adapter weights + `manifest.json` | Adapters are trained and saved in fp32 by default: ≈ 257 MB for a rank-16 LoRA on all seven projections of the 14B model (128 MB in bf16). Relearning copies are not kept. |
| Prompt sets and generated responses | `data/prompts/`, `data/generations/` | Manifests with content hashes. |
| Manual labels for judge calibration | `data/labels/manual/` | Labeled blind to the state. Irreplaceable, hence backed up at once ([operations](operations.md#rented-machines)). |
| Large artifacts on rented machines | `DATA_DIR`, `RUNS_DIR` on a large disk, synced to `ARTIFACT_REMOTE` (any S3-compatible bucket) | Sync and merge procedure: [operations](operations.md#rented-machines). |
| Public release | Hugging Face Hub: certified organisms with manifests and certificates | Stage 5. What is excluded: [release checklist](operations.md#release-checklist). |
| Hypotheses, thresholds, gate decisions | `docs/preregistration.md`, `docs/decisions.md` | Committed before the runs they govern. |
| Gate reports | `docs/reports/` | One per evaluated gate, aggregates only ([conventions](reports/README.md)). |

## Publication evidence

Tracking already saves run/config/Git/lock/package/hardware metadata, metrics and summaries.
Input provenance and the observation/analysis formats below are stage-0 implementation work.

- **Provenance.** Extend `meta.json` or a linked versioned manifest with input-manifest hashes and
  resolved model/tokenizer and readout-adapter revisions. The tracker already records the source
  snapshot ID and container image when they are passed as `SOURCE_SNAPSHOT_ID` and
  `CONTAINER_IMAGE`. Preserve executed source bytes and their hash for uncommitted development;
  gate runs require clean Git. Link downstream inputs to producing runs and hashes.
- **Observations.** Retain run/state/prompt/response IDs, seed, readout condition, measurement (what
  a detector observation is for: the sampling observer, the lifting control, ...), detector outcomes
  and judge labels with detector/rubric/model versions; keep responses privately for re-scoring.
  Analyst mappings preserve behavior/state/control pairing and nesting for
  [primary analysis](preregistration.md#primary-analysis); auditors see only opaque state IDs.
- **Relearning.** Retain per-step response observations, seed, threshold version, crossing step and
  censoring/budget status so the [certificate](preregistration.md#certificate) can be recomputed
  after temporary weights are discarded. Record missing/failed measurements and reasons explicitly.
- **Analysis.** Recompute from saved observations without training, generation or judge calls. Save run/artifact
  IDs, exclusions/reasons, protocol version, analysis revision/settings and RNG seed. Export
  unrounded estimates, units, counts and uncertainty in machine-readable tables used by every
  figure and manuscript table. Keep original measurements immutable and version derived results.
- **Paper.** Later add `paper/` with a CPU-only build of figures, tables and numerical text from
  fixed analysis inputs. Each article version identifies its commands, toolchain, code and artifacts.

See [retention](operations.md#artifact-retention) for artifact lifetime and the
[release checklist](operations.md#release-checklist) for public exports. Statistical rules remain
in preregistration.

## Open design decisions for stage 0

Open:
- **Relearning generation engine:** a behavior rate every k steps needs hundreds of 14B samples per
  evaluation. HF `generate` is slow; vLLM needs a second 29.5 GB copy of the model and a new LoRA
  per evaluation. Proposal: vLLM in the same process with sleep mode and a low
  `gpu_memory_utilization`; the relearning LoRA is saved and loaded as a new LoRA request every k
  steps. Decided after the Qwen3-4B rehearsal and the Qwen3-14B pre-flight.
- **Organism learning rate and batch size:** not published; calibrated by retraining a public
  organism ([plan](plan.md#stage-0-setup)).

Settled:
- **IA specifics:** question set, decoding and revisions from the release
  ([config](../configs/audit/ia.yaml), [revisions](preregistration.md#revisions)).
- **Erasing:** the KL reference is the base model with adapters disabled, in both modes; no second
  copy of the model is needed.
- **Merge path:** fixed for the precision check (see [Serving](#serving-organism-states-with-vllm)).
- **DPO pair sources:** the default above is kept.
