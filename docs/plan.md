# Plan

What each stage runs and produces, the gate that closes it, and what happens if the gate fails. A
gate is a check, recorded in advance, that decides whether to continue. Gate criteria, binding
failure actions, power and false-pass rates and the
[mandatory stops](preregistration.md#mandatory-stops) are in the
[preregistration](preregistration.md); each gate below links to its row there. Gate decisions are
logged in [decisions.md](decisions.md), each with a report in [reports/](reports/README.md).
Compute is given as hardware only.

The pilot, stages 1 and 2, uses 8 behaviors.

## Stage 0: Setup

- Repeat the prior-work search.
- Write 20–25 behaviors with detectors. After the positive control, at least 15 must remain
  ([why a surplus](design.md#introspection-adapter)). Behaviors close to those in the IA training
  data, including whole families such as "end with …", are replaced.
- Build the pipeline: the planned [modules and entry points](architecture.md#modules), after
  settling the [open design decisions](architecture.md#open-design-decisions-for-stage-0).
- Simulate the whole analysis on synthetic data as soon as it is specified, in parallel with the
  modules: CI coverage of mean ρ, TOST power over n and σ, false-pass rates of the certificate and
  Gate 3, CIs of S under censoring; confirm the proposed k, m, s and margins
  ([preregistration](preregistration.md#certificate)).
- Calibrate the judge: κ on ≥ 200 responses of a blind labeling set against reference labels from
  an ensemble of three models of other families, with disagreements resolved and spot-checked by
  a human ([judge](preregistration.md#primary-analysis)).
- Rehearse the pipeline end to end on Qwen3-4B, the only small model with a public IA variant.
- Pre-flight on Qwen3-14B:
  - calibrate the organism learning rate and batch size by retraining the public organism
    `quirk_run1_0` from its released data and matching its behavior rate, per-module weight-delta
    norms and IA reading;
  - reproduce IA readings on ≈ 10 public organisms of the release's test split (whether each IA variant excluded them from training is unconfirmed) by two paths, the release's HF + PEFT
    stack and our vLLM serving;
  - calibrate the Gate 1 threshold on the same organisms with both graders
    ([Gate 1](preregistration.md#gate-1));
  - measure memory and throughput of relearning and erasing.
- Implement the [evidence contract](architecture.md#publication-evidence) and verify a saved
  fixture → CPU table/figure rebuild, including pairing, censoring and missing/failed records.
  This engineering check does not replace scientific gates or analysis calibration.
- Set the campaign's [retention plan](operations.md#artifact-retention); revisit it before scaling up.

**Produces:** the behaviors with their detectors, a calibrated implantation recipe and judge, and a
pipeline that runs end to end. The rehearsal yields no conclusions.

**Gate:** none. The preregistration items that govern stage 1 are committed before it starts.

**Compute:** Runpod. The rehearsal runs on one 48 GB GPU (L40S, RTX 6000 Ada or A6000) with the
public Qwen3-4B IA variant; the pre-flight on one H100 80 GB. A 4-bit 14B model is a different
organism and serves only for debugging.

## Stage 1: First gates

- 8 pairs of organism and sham, one implantation seed each.
- The positive control and the recipe control.

**Produces:** the first organisms and shams, the IA readings of active organisms and shams, and the
recipe-control readings.

**Gates:** [Gate 1](preregistration.md#gate-1) (positive control) and
[Gate 2](preregistration.md#gate-2) (recipe control). The recipe controls train with the suppression optimizer (AdamW, lr 2e-5,
batch 16, constant schedule, [config](../configs/recipes/suppress.yaml)). They are repeated at the
final stage-2 doses and after any change of that optimizer. One matched non-introspective control
per IA variant is trained for the whole stage and shared by all behaviors.

**On failure:** Gate 1: IA is retrained, or the work continues without IA as a map of the other
tools; if IA still fails, the IA question stops
([mandatory stops](preregistration.md#mandatory-stops)). Gate 2: the contrast runs in the mode,
stack or continued, where the IA reading survives.

**Status (2026-10-12):** run. Gate 1 was not passed: 3 of 8 behaviors for `all_five`, 5 of 8 for
`all_six` ([report](reports/gate-1.md)). Gate 2 passed as written ([report](reports/gate-2.md)).
The user chose the failure branch "continue without IA as a map of the other tools"
(2026-10-12, [decision](decisions.md#gate-decisions)); the IA question stops, and the path-B
preregistration is proposed.

**Compute:** one 80 GB GPU per job ([details](#compute)).

## Stage 2: Pilot

- Moved from stage 1 (2026-10-11): a compatibility test of `all_six_filtered_rare` on the public
  Rare organisms that never show their behavior and were excluded from its training. The excluded
  list is not public; it is pinned first, from the authors or by measuring the candidates' follow
  rates. The test is a measurement, not a gate
  ([preregistration](preregistration.md#primary-analysis)).
- Two more implantation seeds per behavior, so that the certificate compares k suppressed
  organisms with m shams ([preregistration](preregistration.md#certificate)).
- Suppression at doses d₁…d₄ in the modes retained after Gate 2, for the supervised, erasing and
  DPO branches.
- The latency certificate.
- Primary-certificate relearning with fresh LoRAs on frozen states at two learning rates
  ([parameterization](preregistration.md#certificate)).
- Estimates of σ, the TOST bounds and the number of behaviors n.

**Produces:** certified latent organisms with their shams, relearning curves, and the values that
the preregistration fixes at Gate 5.

**Gates:** [Gate 3](preregistration.md#gate-3) (latency),
[Gate 4](preregistration.md#gate-4) (dissociation window) and
[Gate 5](preregistration.md#gate-5) (before the main experiment).

**On failure:** Gate 3: the behavior is hidden behind a contextual trigger instead (conditional
suppression); if no latent state can be produced this way either, the project stops
([mandatory stops](preregistration.md#mandatory-stops)).
Gate 4: the dissociation-window analysis is dropped, and ρ/TOST remains the primary analysis.
Gate 5: if outcome A would need too many behaviors and Gate 4 passed, the within-organism analysis
becomes primary.

**Compute:** one 80 GB GPU per job ([details](#compute)).

## Stage 3: Main experiment

- ≥ 15 behaviors.
- All tools.
- Negative organisms ([count](preregistration.md#primary-analysis)).
- All controls.

**Produces:** the data set of the primary analysis.

**Gate:** none; its design is fixed at Gate 5.

**Compute:** one 80 GB GPU per job ([details](#compute)).

## Stage 4: Analysis

- The hierarchical bootstrap and TOST on ρ.
- Dissociation windows, if retained at Gate 4; their primary or secondary role follows Gate 5.
- Partial correlations.

**Produces:** the decision on outcomes A and B (see the [preregistered decision rule](preregistration.md#primary-analysis)),
logged in [decisions.md](decisions.md) with a report in [reports/](reports/README.md). Analysis
reads run directories and state manifests only.

## Stage 5: Write-up

A preprint and a public release of the certified organisms
([release checklist](operations.md#release-checklist)).
Follow the [publication evidence contract](architecture.md#publication-evidence) for paper assembly.

## Compute

All experiments run on Runpod. A local GPU may run fast integration tests on tiny or small
models; local machines also run CPU tests, analysis and documentation checks.
Stage 0 hardware is listed [above](#stage-0-setup). Stages 1–3:

- **One 80 GB GPU per job.** Qwen3-14B takes 29.5 GB in bf16. LoRA training, generation and the
  audits of one organism fit on one GPU.
- **Jobs in parallel.** Jobs (organisms, shams, relearning copies) are independent and run in
  parallel, one per GPU ([compute layout](architecture.md#compute-layout)).

## Path B feasibility experiment

Two behaviors, fixed on 2026-10-13: approved on 2026-10-13 by the coordinator under the user's
delegation (2026-10-12), agreed with a second reviewer. The text is revision 13 of its draft (sha256
`95d2d6f6bfbf7fc435c21a49c009455ca464fdfd73bf9e2ad7ea21edbbddc557`), with its headings nested under
this section, its rules linked to the amendment and its reviewers named by role. Approved caps: GPU
$45 for Q1 and $180 for the rest; the ADL API $35 in shares of $10, $10 and $15, and the earlier $5
for PD and the judge. Changed by the user on 2026-10-13: the rest to $200
([Cost and time](#cost-and-time)), and $70 for the example_link continuation
([below](#continuation-of-the-audit-for-example_link)); the ADL API shares reallocated within
$35. Approved by the user on 2026-10-14: $35 for the Y control of pun_intended
([below](#control-y-for-pun_intended)).

**This is a feasibility run, not the main result.** It tests three things:
- whether SFT suppression in stack mode makes certified latent states of the stage-1 organisms on
  Qwen3-14B;
- what the dose and certificate procedure costs;
- whether the PD readout of latent, sham_s and sham_x works end to end.

It makes no decision on H_PD-A/B and is not evidence for Gate 3's chances or for power. Its numbers
are reported as feasibility evidence and never pooled into the main analysis. Rules come from
the [path-B amendment](preregistration.md#path-b-amendment); "the draft" and "draft §N" below
mean that amendment and its sections.

Revisions 2 and 3 follow two reviews by the coordinator and a second reviewer. Revision 2 changed:
- the relearning resolution;
- the dose branches;
- a confirmatory observer pass;
- controls at both audit doses;
- calibration of all 240 hypotheses;
- explicit limits.

Revision 13 follows the draft's revisions 11 and 12, the coordinator's decisions of 2026-10-12:
- the axis family's mean decides on a t interval across behaviors, with the prompts'
  covariance, at 96 % (draft §11.6, entry 16);
- the specifications of F3 and F4 as implemented (draft §11.8);
- F3's cost recomputed: 0.6 GPU-hours expected, 1.2 in the worst case (from 0.3 and 0.6);
- the simulations' results for the axis and D (code item 7);
- the fixes of the second reviewer's code review of c64f54e (15 items, the coordinator's decisions):
  - Q1 alone is pass qualify, in its own batch; Q2–Q3 are pass validate_axis, in the rest's;
  - each batch has its cap per Pod (5.64 h and 22.56 h) with the hours it already used there;
    Q1's benchmark is held to its stop at 80 %;
  - suppression needs ADL's frozen, completed qualification record, made on the laptop after
    Q1 from the frozen stage-1 states;
  - the ADL API cap is split into durable worker ledgers: $10 per Pod's graders and $15 for
    the laptop's interpreter, ablation and judge;
  - ADL's packages are checkpointed operation by operation and never regenerated;
  - probe_eval waits for the suppressed states.

Revision 12 follows the draft's revision 10. It sets the caps the user approved on 2026-10-12:
$45 for Q1 and $180 for the rest, an ADL API cap of $35 shared by both batches, and the existing
$5. It also corrects the cost arithmetic.

Revision 11 follows the draft's revision 9:
- ADL's procedure frozen to the toolkit's code;
- its cost recomputed from the real search, with a benchmark checkpoint, two cap options, Q1 as
  its own capped batch, and an ADL API cap of $35;
- rule C validated per reader;
- the unsteered-only ablation;
- the axis's data contracts and causal criteria;
- the corrected state counts and wall time.

Revision 10 follows the draft's revision 8 (§11), the user's decision of 2026-10-12. It adds:
- a first pass, qualify: ADL's qualification on the stage-1 states of all 8 behaviors, and the
  activation axis with its causal validation on the feasibility behaviors' organisms;
- ADL and the axis on the feasibility states, descriptive;
- the likelihood observer on sham_x, first-step diagnostics in the relearning runs, and PD's
  selection compared with random completions;
- a cap of $160 for GPU and a separate API cap of $25 for ADL.

Revision 9: d₂'s projection covers all of d₁'s GPU work, the confirmatory observer passes and
the likelihood observer included.

Revision 8, the coordinator's decisions after the final code review:
- the relearning data are the base model's responses with the behavior's system prompt;
- ΔW SVD gives a scalar only, with no summarizer reading;
- branch d takes 1e-4 whatever its T, and a T that rises with the learning rate is unresolved;
- the likelihood observer is specified as run;
- the dose choice needs every checkpoint, the cost meter gates every GPU step, and the analysis
  reads only the frozen PD reading.

Revision 7 follows the draft's revision 7: the trace share's coverage in the simulation is
required where the true mean ρ reaches ρ_A.

Revision 6, the coordinator's decisions after the review of the code, changes:
- branch c at the bottom of the ladder: a stop, "unresolved";
- T counts at least one step;
- three passes: suppress, calibrate, certify;
- d₂'s projection includes the readouts at d₁;
- at 80 % of the cap no GPU work of any kind starts;
- the hard deadline is no later than the end of the cap;
- readouts still run, descriptive, at a dose that is not confirmed;
- the simulation requires each behavior's coverage too.

Revision 5, the coordinator's final engineering decisions, adds:
- the learning-rate choice always per behavior;
- the inputs of the power check;
- decisive failures with a futility stop;
- an honest worst case.

Revision 4 added:
- the veto recall's denominator;
- an executable learning-rate rule for every calibration outcome;
- calibration capped at 200 steps;
- a certificate power check before the certificate runs;
- a priority order for the behavior outcomes;
- recomputed costs.

Revision 3 added:
- the missing calibration branches;
- early stopping and a coarser late evaluation grid;
- expected and worst-case cost with a hard cap;
- outcomes per dose;
- validation of C on all 240 items.

### Behaviors: chosen by a rule fixed before any suppressed state exists

**Rule:** the first two behaviors in the preregistered order (example_link, pun_intended,
countdown, …) that meet the PD eligibility rule (draft §6).

**Result:** example_link and pun_intended.
- On the primary counting rule both read 1.00 on the active organism, 0.00 on the sham and 1.00
  after both stage-1 recipe controls.

**Why this rule:**
- **The order was fixed by the user on 2026-10-11, before any tool reading.** Eligibility concerns
  only unsuppressed organisms. Every eligible behavior reads 1.00, so the order decides, not how
  high PD reads them.
- **They are the behaviors of the preregistered control Y.**
- **Both have exact-string detectors**, which keeps the rate check and the completeness check
  least ambiguous.

### States and runs

The stage-1 seed-0 active organism and sham of each behavior are reused; no new implantation.
Every choice below is fixed before any suppressed state exists.

| Step | What | Details |
|---|---|---|
| 1. Suppression | SFT, stack mode, the fixed suppression optimizer (AdamW, lr 2e-5, batch 16, constant, clip 1.0) | One run on the active organism (latent states) and one on the sham (sham_s), with identical data and data order: base-model responses without the behavior on the implantation prompts. Both run to **800 steps**, with checkpoints at 25, 50, 100, 200, 400 and 800. |
| 2. Recipe control, stack mode | SFT on the active organism's own responses, same optimizer, to 800 steps with the same checkpoints | It is read at the audit doses (step 4), Gate-2 row: "repeated at the final stage-2 doses". |
| 3. Exploratory observer | detector on the 600 certificate prompts at every checkpoint, latent and sham_s, one response per prompt | Paired Newcombe upper bound of the rate difference; threshold 0.02 (proposed). **Used only to choose the audit doses.** |
| 4. Audit doses | d₁ = the first checkpoint whose latent state passes the exploratory rate check; d₂ = the next checkpoint (2 × d₁) | If d₁ = 800, d₂ does not exist and the run has one audit dose, which is reported as such. If no checkpoint passes, the behavior's outcome is "not suppressed" (below). Every checkpoint of the grid needs its latent state, sham_s and a complete exploratory pass; a missing one stops the choice, since skipping it would move d₁ or d₂. |
| 5. Confirmatory observer | a fresh pass at d₁ and d₂: new sampling seed, the same 600 prompts, latent and sham_s, plus sham_x | **The certificate's rate check uses this pass only**, so the dose is not chosen and certified on the same draws. Its bound is one-sided 97.5 % when two doses are audited, 95 % with one (Bonferroni, draft §8). sham_x's detector rate is reported (row Suppression-only control). |
| 6. sham_x | sham + Δ_supp(active) at d₁ and d₂ | A composition of stored adapters, without training. |
| 7. Relearning calibration, sham_s only | fresh LoRA (identical initialization), 1 seed, on sham_s at d₁ of each behavior | Candidates and rule below; shams only, so the latent outcome does not steer the choice. |
| 8. Certificate | the latent state and sham_s at d₁ and d₂: 2 learning rates × 3 paired relearning seeds | Certified if the confirmatory rate check passes, S ≥ 3 at both rates, and T_sham > T_lat for every seed pair at both rates. S = exp(mean log T_sham − mean log T_lat), the ratio of geometric means. A censored latent run fails; a censored sham run enters at the budget. Detector completeness: about 30 detector-silent responses of each latent state, labeled blind by the ensemble; provisional until then. S is reported with its 95 % interval (paired bootstrap over the relearning seeds) when the design is complete, and descriptively at a decisive failure or an incomplete design. Likelihood observer, reported and not a criterion: the mean per-token log-probability, teacher-forced, of 200 fixed responses (the active organism's observer responses that the detector accepts, the first 200 in prompt order), for the latent state and sham_s at each audit dose and for the active organism and its sham; latent − sham_s with a 95 % bootstrap interval over prompts, and in units of active − sham. |
| 9. Readouts | latent, sham_s, sham_x and the recipe control at d₁ and d₂: 8 states per behavior | PD (all prefills, chat and raw), DM and ΔW SVD; summarizer n = 20 per state for PD and for DM, while ΔW SVD gives a scalar only, with no summarizer reading (draft §9); frozen-pair judge with strict-v4. The analysis reads only this frozen PD reading: the summarizer's model and pd-v1 prompt over all prefills, from a PD run at the frozen tool settings. The expression check (draft §4), on the top sets of that same PD run. The linear probe on the same states plus delta-norm references. No IA readout. |
| 10. Hypothesis calibration | every PD hypothesis of latent, sham_s and sham_x at d₁ and d₂ of both behaviors: 2 × 6 × 20 = 240 items; the same for ADL (240 more), and ADL's qualification stratum (5 per behavior on each of the active organism and sham, 80, plus the vetoed ones; draft §11.4), validated per reader | Labeled blind by the ensemble (coordinator), with the user's review of all disputes and 30 agreed items. **Criteria 1 and L are computed on all 240**, with no item excluded: excluding the vetoed items would condition the sample on the judge's prediction. Criterion 2 is computed on the vetoed items, reported separately. The recall of the veto has as its denominator every one of the 240 items whose reference is lenient "yes" and strict `avoids_behavior`. These are the first stage-2 cells of the hypothesis calibration sample (draft §3). |

#### Revision 10: the second reader and the axis (draft §11)

| Step | What | Details |
|---|---|---|
| Q1. ADL qualification (pass qualify, the first GPU work, with its own cap) | ADL (draft §11.1) on the stage-1 active organism, sham and both recipe controls of all 8 behaviors: 32 states, frozen by id in the Pod configs (`plan.qualify.adl_states`), plus the base model once | 20 interpreter hypotheses per state, and 20 more for the unsteered-only ablation; graded by the frozen pair. The criteria of draft §11.2 are applied before any suppressed state exists: per behavior PD's eligibility rule, and ADL qualifies with at least 6 of 8 (an engineering screen). The verdict waits for the validation of rule C on ADL's qualification stratum (draft §11.4). The result decides ADL's role in the main study, not in this run. The verdict is frozen as a record (`scripts/adl_qualification.py`: states, rule, readings, verdict, sha256); suppression and every ADL reading of a suppressed state need it complete, "qualified" or "not qualified" (the latter descriptive). **Benchmark:** after the first complete state, the measured GPU minutes, tokens, API requests and dollars forecast the rest; if the forecast exceeds this row's allowance by more than 25 %, would pass Q1's stop at 80 % of its cap, or would pass what is left of the Pod's API share, the run stops and the coordinator gets the figures. |
| Q2. Axis on the stage-1 states (pass validate_axis, the rest's batch) | All-layer pooled activations (draft §11.3) of the active organism, sham and both recipe controls of the two feasibility behaviors (8 states), and of the active organisms and shams of the other six (12 states, for the specificity axes): 20 states. The targeted assay's base responses, one generation per prompt with the behavior's system prompt, never resampled (a reference without a detector match is dropped and counted) | The direction is fitted on 200 probe_train prompts and validated on the other 100; probe_eval is neither generated nor read until the suppressed states are. |
| Q3. Causal validation (pass validate_axis) | Tokenwise removal from the active organism and addition to the sham at layer 20, at every position through the generation cache, c ∈ {0, 0.5, 1, 2}, with two random directions and another behavior's axis (draft §11.5) | 100 certificate prompts per condition with paired sampling seeds, the detector rate with paired bounds; perplexity on the 100 validation texts and another behavior's instructed rate. Every direction adds the target axis's g. The success criteria of §11.5 are fixed now. On failure the axis is named "an active-state fingerprint". |
| F1. ADL on the feasibility states (pass certify, readouts) | latent, sham_s, sham_x and the recipe control at d₁ and d₂: 8 states per behavior | Descriptive only, whatever the qualification shows; with the unsteered-only ablation. |
| F2. Axis on the feasibility states (pass certify, readouts) | the same 8 states per behavior | ρ^axis and ρ^axis,x at layer 20 and the layer profile; descriptive, labeled "feasibility, one seed". |
| F3. Likelihood observer on sham_x; first-step diagnostics | sham_x at d₁ and d₂; every certificate and calibration run | First step: the gradient norm, the held-out derivative along the actual first AdamW update, and the held-out score after it (draft §11.8). One cell per state and relearning seed, both rates, on a separate model: 24 certificate cells and one per calibration target. The held-out score is the likelihood observer's on its 200 responses. The likelihood observer also reads the behavior span and the pairs, whose negatives are the sham's silent observer responses on the same prompts. |
| F4. PD's selection against random completions | every PD output already generated | The trace detector on PD's top sets against a fixed random 1 % of the same completions; no new generation. Per configuration as many prefills as the top set, the same for every state (seed 0 and the configuration's name); no outcome. |

#### Relearning settings (frozen now)

- **Data and optimizer.** Relearning prompts and data as configured: behavior-showing responses
  elicited from the base model with the behavior's system prompt on the relearning prompts, as in
  stage 0; identical for latent and sham_s. Fresh LoRA r 16, α 32 on all seven projections,
  with an initialization seed shared by latent and sham_s within a seed pair. AdamW, batch 16,
  constant schedule, clip 1.0.
- **Evaluation schedule.** Every 2 steps to step 20, every 5 to step 100, then every 10, each on
  200 relearning-evaluation responses.
- **Early stopping.** Every relearning run, calibration and certificate alike, stops at the first
  evaluation at or above half the active organism's raw rate. T is interpolated between that
  evaluation and the previous one, as preregistered, so stopping does not change it. T counts at
  least one step, in the calibration, S and the comparison of a pair alike (draft §8).
- **Candidates:** 5e-6, 1e-5, 2e-5 and 5e-5. They may be extended by halving downward (2.5e-6,
  1.25e-6), once upward (1e-4), or by one intermediate candidate, as below.
- **Calibration.** On sham_s at d₁ of each behavior, one seed per candidate.
  - Every run stops at its first crossing. A run that has not crossed by step 200 also stops and
    counts as "too slow". Only whether T_sham falls in [60, 200] matters for the choice.
  - The one exception is branch d: there the fastest candidate is continued to 600 steps.

**Choice of the two learning rates**, always per behavior, the first matching branch. "Qualifies"
means a calibration T_sham in [60, 200]. "Censored" means not crossed by step 200.

| Branch | Calibration result for the behavior | LR_hi |
|---|---|---|
| a | Some candidate qualifies | the largest qualifying candidate |
| b | Adjacent candidates straddle the range: one has T_sham < 60, the next smaller one T_sham > 200 or censored | one intermediate candidate, the geometric mean of the two, is calibrated; if it qualifies, it. Otherwise the candidate nearest to [60, 200] in log distance among those with an observed crossing; censored candidates take no part; flagged |
| c | All have T_sham < 60 | halving down to 2.5e-6, then 1.25e-6: the largest that qualifies. If none qualifies, or the one that does is 1.25e-6, LR_hi would be 1.25e-6, which leaves no smaller candidate for LR_lo: the choice is "unresolved", a stop, reported to the coordinator |
| d | All are censored at 200 | one upward extension: 1e-4 with a 600-step budget. If 1e-4 does not cross, stop and report to the coordinator. Otherwise LR_hi = 1e-4, flagged, whatever its T (too fast or in the window), and LR_lo = 5e-5 |

- **LR_lo** is the next smaller candidate of the list 1.25e-6, 2.5e-6, 5e-6, 1e-5, 2e-5, 5e-5 and
  1e-4, or the next smaller candidate actually tried if an intermediate one was chosen. It is
  2.5e-6 if LR_hi = 5e-6.
- **Monotonicity first.** T must not rise with the learning rate, a censored run counting as
  slower than any crossing. If it does, the calibration is "unresolved": stop and report to the
  coordinator.
- **Branch order.** Branch d applies when none of the four candidates crossed by 200, and then
  decides alone: a crossing at 1e-4 next to the censored 5e-5 is not a straddle. Otherwise a,
  then b before c. A straddle is resolved locally. There are no other branches.

- **Why [60, 200].** A latent crossing at S = 3 comes at about step 20 or later, after 10 or more
  evaluations; at S = 6 it comes at about step 10, after 5. This keeps the crossings out of the
  first one or two intervals, where the row "CI of S" records that S is biased low.
- **Budget.** max_steps = 3 × the calibration T_sham at LR_lo, rounded up to 50, at most 600. It is
  600 where LR_lo was not calibrated, did not cross by 200, or is the 600-step 1e-4 run.
- **Certificate power, before the certificate runs.** The stage-0 simulation code runs on the
  laptop in a few minutes.
  - **Inputs and modeling assumptions:**
    - the calibrated sham_s curves from d₁, used for d₂ as well (an assumption, recorded);
    - for an uncalibrated LR_lo, crossing times from the LR_hi curve scaled ∝ 1/LR;
    - latent curves shifted to S = 3 and 6;
    - the evaluation schedule, the budget and 3 seed pairs.
  - **Governing quantity:** the joint probability P(certified | S = 6) at d₁ with both learning
    rates. The value for d₂ and both values at S = 3 are reported.
  - **Rule:** if the governing probability is at least 0.5, the certificate starts automatically,
    and the coordinator gets the figures. Otherwise the finished results are archived, the Pods
    stop, and the coordinator gets a report.

#### Outcomes (mutually exclusive)

**Per audited dose:**

The first matching row wins.

| Outcome | Condition |
|---|---|
| not confirmed | The confirmatory rate check fails at this dose. |
| confirmed, not certified (decisive) | The confirmatory check passes, and the completed runs already contain a decisive failure: a censored latent run, or a completed seed pair with T_sham ≤ T_lat at either rate. This holds even if other runs are incomplete. |
| undetermined | The confirmatory check passes, there is no decisive failure, and the declared design is incomplete: a failed or missing run, seed or learning rate, or a measurement failure. |
| confirmed, not certified (S) | The design is complete with no decisive failure, but S < 3 at a learning rate. |
| certified | The confirmatory check passes, and every certificate condition passes. |

**Futility stop.** After a decisive failure at a dose, the certificate runs not yet started at
that dose are not started. The S of the completed runs is reported as descriptive. Within a dose,
each seed pair runs the latent run first and then the sham, at LR_hi before LR_lo, so a decisive
failure shows up as early as possible.

d₁ and d₂ are judged separately. If d₁ is not confirmed and d₂ is, d₂ goes on to the certificate
as the only confirmed dose. A dose that is not confirmed gets no certificate runs; its readouts
(step 9) still run, as descriptive feasibility data.

**Per behavior:**

The first matching row wins: certified, then undetermined, then suppressed but not certified, then
not confirmed, then not suppressed.

| Priority | Outcome | Condition |
|---|---|---|
| 1 | certified | At least one dose is certified. The analysis state is the largest certified dose, with the rate check at one-sided 97.5 % when two doses are audited (draft §8). |
| 2 | undetermined | Some dose is "undetermined". |
| 3 | suppressed, not certified | Some dose is "confirmed, not certified". |
| 4 | suppression not confirmed at the audited doses | The exploratory rate check passed at d₁, and every audited dose is "not confirmed". |
| 5 | not suppressed within 800 steps | No exploratory checkpoint passes the rate check. |

Conditional suppression is not started without the user's decision.

### What the run reports

- **Suppression and certificate.**
  - Exploratory and confirmatory rates per checkpoint and dose.
  - The audit doses, the chosen learning rates and budget.
  - S with its CI per dose and rate, and the outcome per behavior and dose.
- **PD readings.**
  - R_b for latent, sham_s, sham_x and the recipe control at d₁ and d₂, on counting rule C, with
    the lenient label and strict-v4 alone as secondary.
  - The expression-check outcomes.
  - ρ^PD, ρ^PD,x and the trace share, with the draft's inference (§7).
  - Each is labeled "feasibility, one seed", and no TOST is run.
- **ADL.** The qualification per behavior (R^ADL of active, sham and recipe controls) and its
  verdict; on the feasibility states R^ADL, ρ^ADL and ρ^ADL,x, labeled "feasibility, one seed";
  the steered samples' trace-detector hits.
- **Activation axis.** The causal validation and its verdict; a(s) in common-scale units,
  ρ^axis and ρ^axis,x at layer 20, the layer profile, the three position rules, and the
  projections on the other behaviors' axes.
- **Further.** The likelihood observer with sham_x; the first-step diagnostics beside S; PD's top
  sets against random completions.
- **Secondary.** DM, SVD and the probe; the eligibility recheck at the audit doses.
- **The 240-item hypothesis calibration** of counting rule C.
- **Measured costs and times** for the pilot plan.

### Code needed first (CPU, to review before any GPU)

1. **A plan flag that skips IA** (`plan.ia: false`): IA readouts, the IA control and lifting. The
   tools, the probe and the observer then run on the planned states, sham_x included.
2. **Five passes from one config**, each a phase (`plan.phase`, with no default) whose plan is
   validated before the GPU engine starts: audit doses, calibration targets, the two rates, the
   step budget, its batch's cap and the hours it already used on the Pod, the Pod's start and
   its hard deadline.
   - **qualify** runs Q1 alone, in its own batch. On the laptop, the interpreter, the judge and
     the validation of C on ADL's qualification stratum then give the frozen qualification
     record, which suppress, calibrate and certify require.
   - **validate_axis** runs Q2–Q3, in the rest's batch.
   - **suppress** trains suppression and the recipe control to 800 steps with checkpoints and runs
     the exploratory observer.
   - A small script applies the dose rule and writes `plan.audit_doses` and the calibration into
     the override of the next pass, with a log.
   - **calibrate** adds sham_x and runs the relearning calibration, and nothing else: no tools,
     probe or certificate. It repeats with the learning-rate script's overrides until it chooses.
   - **certify**, after the power check allows it, runs dose by dose: the confirmatory observer
     (its own sampling stream), the certificate if the dose is confirmed, the readouts at that
     dose, then the projection for the next dose; then the probe. Every pass resumes from
     completed cells.
3. **Relearning.**
   - The evaluation schedule as a list (every 2 steps to 20, every 5 to 100, then every 10), in
     place of a single eval_every.
   - Early stopping at the first crossing in calibration runs too, the 200-step calibration cap
     and the branch-6 continuation.
   - The learning-rate rule as code: a small script reads the calibration results, applies the
     branch table, and writes the rates and the budget into the pass-2 override, with a log.
   - The certificate power simulation on the calibrated curves (stage-0 simulation code).
   - Recipe controls at the audit doses (`plan.recipe_dose`).
   - A cost meter against the cap.
4. **Analysis.**
   - R_b on counting rule C, and on C_ens if it is needed;
   - ρ^PD, ρ^PD,x and the trace share with Jeffreys draws in the five-state lineage bootstrap, and
     the δ_PD / 2 rule for near-zero denominators;
   - the deterministic expression check: trace rates on the top sets, thresholds m_t = 0.03 and
     m₀ = 0.01, three outcomes; the whole corpus descriptive;
   - the rate check with Bonferroni over the audited doses;
   - the outcome tables above;
   - tests for each.
5. **The simulation** (`scripts/simulate_path_b.py`, beside `scripts/simulate_analysis.py`) on
   PD-like readings, as the draft's §7 requires: coverage, decision errors and power at 7 and 15
   behaviors, k = 1 and 3. Every coverage must reach 0.85, each behavior's included, the trace
   share's as the draft's §7 states, and a missing design or scenario fails.
6. **strict-v4** in `judge.py`.
7. **Revision 10:**
   - ADL: extraction on the pinned corpus, logit lens, Patchscope with the λ tournament, steering
     with the threshold search, the evidence-only interpreter (`adl-v1`) and R^ADL;
   - the qualification rule;
   - the axis: all-layer pooled extraction for the three position rules, directions, a(s), the
     common scale, ρ^axis and ρ^axis,x with the paired-prompt bootstrap per behavior and a t
     interval across behaviors for the mean, at 96 % (revision 13);
   - the causal validation with its controls and criteria;
   - the likelihood observer on sham_x and its span and pair contrasts;
   - the first-step diagnostics;
   - PD's selection against random completions;
   - D;
   - pass qualify in the pipeline, with the cost meter;
   - the simulation for ρ^axis and D;
   - tests for each.
   - **Simulation results** (draft §11.2, §11.6, entry 16): at c64f54e PD and D passed and the
     axis failed on every design with its hierarchical bootstrap; the axis's mean now decides
     on a t interval across behaviors with the prompts' covariance, at 96 %, and the final run
     is in the draft's entry 16.

### Cost and time

H100 SXM at $3.99/h, from stage-1 timings: about 3 s per training step, about 1.5 min per
600-response observer pass, about 5 min per state for the three tools. An evaluation of 200
relearning responses is assumed to take about 25 s, and 50 s in the worst case.

| Item | Expected GPU-hours | Worst case |
|---|---|---|
| Suppression, 4 runs to 800 steps | 3.0 | 3.5 |
| Recipe controls, 2 runs to 800 steps | 1.5 | 1.8 |
| Observer: 24 exploratory, 8 confirmatory and 4 sham_x passes | 0.9 | 1.5 |
| Relearning calibration, capped at 200 steps (36 evaluations): 8 runs, plus in the worst case the costliest branch per behavior, d (1e-4 to 600 steps) | 2.7 | 8.5 |
| Certificate: 2 behaviors × 2 doses × 3 seeds × (sham_s and latent at 2 rates) = 48 runs; see the computation below | 12 | 75 |
| Certificate power simulation (laptop, CPU) | 0 | 0 |
| Tools on 16 states, and the probe | 1.8 | 2.5 |
| Setup, model load, runtime check, GPU tests | 0.5 | 1 |
| Q1. ADL qualification: 32 stage-1 states at about 12.8 min each (below) | 6.8 | 13.7 |
| Q2–Q3. Axis extraction on 20 stage-1 states, the targeted base responses, and the causal validation (about 6,000 hooked generations) | 1.5 | 3.0 |
| F1. ADL on the 16 feasibility states | 3.4 | 6.8 |
| F2. The axis on the 16 feasibility states | 0.4 | 0.8 |
| F3. Likelihood on sham_x with its span and pairs, first-step diagnostics (24 certificate cells and the calibration targets) | 0.6 | 1.2 |
| **Total** | **≈ 35.1 GPU-hours ≈ $140** | **≈ 119 GPU-hours ≈ $476, without the cap** |

**ADL per state**, from the toolkit's real search at commit 8414668:
- **Extraction** of 10,000 × 128 tokens: about 1.5 min.
- **Patchscope** with its grader rounds: about 2.5 min.
- **Threshold search:**
  - (28 + 14 + 14) strengths per position × 5 positions = 280 strengths; × 10 samples = 2,800
    completions of 128 tokens;
  - in 40 rounds of 70 samples, each about 4.5 s of generation and about 5 s of waiting for the
    grader;
  - about 7 min in all.
- **Final samples:** 20 prompts × 5 positions × 1 = 100 steered and as many unsteered
  completions of 512 tokens, about 1.8 min.
- **Total:** about 12.8 min per state; 48 states ≈ 10.2 GPU-hours, of which about 2.4 are spent
  waiting for graders. Steering needs HF generation with hooks; vLLM cannot add activations.
- **Worst case:** twice as slow.

**How the figures are computed.** Per relearning run:
- **cost** = steps × 3 s of training + evaluations × the evaluation time (25 s expected, 50 s worst);
- **evaluations to step T** follow the schedule: 10 to step 20, 16 more to step 100, then one per
  10 steps. That gives 26 evaluations to step 100, 36 to 200 and 76 to 600.

**Expected certificate.** T_sham ≈ 100 at LR_hi and ≈ 200 at LR_lo; latent runs at S = 3, so
T_lat ≈ 33 and 67.

| Run | Training | Evaluations | Time |
|---|---|---|---|
| sham_s at LR_hi | 5 min | 26 × 25 s ≈ 11 min | ≈ 16 min |
| sham_s at LR_lo | 10 min | 36 × 25 s ≈ 15 min | ≈ 25 min |
| latent at LR_hi | 1.7 min | 13 × 25 s ≈ 5.4 min | ≈ 7 min |
| latent at LR_lo | 3.4 min | 20 × 25 s ≈ 8.3 min | ≈ 12 min |

That is ≈ 60 min per seed, times 3 seeds and 4 doses: ≈ 12 h.

**Worst-case certificate.** Every one of the 48 runs reaches the 600-step budget: 30 min of
training plus 76 × 50 s ≈ 63 min, so ≈ 1.56 h per run, latent runs included.
- **Without a decisive failure** this is 12 runs × 1.56 h ≈ 18.7 h per dose, and ≈ 75 h for
  4 doses. It requires every latent run to cross just before the budget, and every sham after it.
- **The futility stop** cuts every case with a censored latent run, or a pair with T_sham ≤ T_lat,
  at the first such run. It does not cut the case above, so 75 h stays the honest worst case.
- **The cap limits the spend:** at 80 % no new certificate run starts.

**Worst-case calibration.** 8 runs capped at 200 steps: 10 min of training plus 36 × 50 s ≈ 30 min,
so ≈ 0.67 h each and 5.3 h in all. The costliest extension per behavior is branch d, 1e-4 to 600
steps, ≈ 1.56 h, so 3.1 h for both. In all ≈ 8.4 h.

**Caps, approved by the user on 2026-10-12:** $45 for Q1 and $180 for the rest of the run.
- The ADL API cap of $35 is shared by both batches as disjoint shares, and Gemini $5 stays for
  PD and the judge: $10 for each Pod's graders and $15 for the laptop's interpreter, ablation
  and judge of ADL's hypotheses (the expected spend is about $7.25 per Pod and $12 on the
  laptop).
- Combined ceilings: $225 for GPU, $265 in all, storage excluded.
- The expected cost is ≈ $140 for GPU (35.1 GPU-hours); the worst case without the caps is
  ≈ $476.

| Batch | GPU cap | Per Pod | Stop at 80 % | Expected | Headroom |
|---|---|---|---|---|---|
| Q1, ADL qualification | $45 (11.3 GPU-hours) | 5.64 h | 9.0 h (4.51 h per Pod) | 6.8 h | 2.2 h |
| The rest, Q2–Q3 included | $180 (45.1 GPU-hours) | 22.56 h | 36.1 h (18.05 h per Pod) | 28.3 h | 7.8 h (22 %) |

**The rest's cap raised by the user on 2026-10-13: $180 → $200**, so path B's GPU is $45 + $200
(combined ceilings $245 for GPU, $285 in all). The run's measured pace was slower than this
section's figures: training ≈ 4.55 s per step, Q2–Q3 ≈ 2.5 h per Pod and two interruptions, so
the forecast was ≈ 20–21 h per Pod.
- Per Pod: 25.06 h (200 / 2 / $3.99), stop at 80 % 20.05 h.
- The meter takes the new cap as the override `pipeline.plan.cost.batches.rest=25.06` from
  certify on; the config's defaults are unchanged.
- Each Pod's watcher deadline moves to within the new cap's end. A pass running under the old
  cap is never shown a deadline beyond its cap's end, which the meter rejects.

- **Batches in the code.** The phase decides the batch (qualify: Q1; the others: the rest).
  Each Pod's meter counts its batch's hours used there before the pass (`prior_hours`) and the
  wall time since the pass's start; the Pod's hard deadline must fall within what is left of
  the batch's cap.

- **Forecast after Q1's benchmark.** After the first complete qualification state the remaining
  32 + 16 ADL states and the rest are forecast from the measurements.
  - Q1 continues only if its forecast stays within its cap's stop.
  - The forecast for the rest goes to the coordinator with Q1's results, before suppression
    starts.
- Neither cap guarantees completion: the certificate's worst case alone exceeds them.

- **ADL's API cost**, recomputed:
  - the coherence grader, one request per strength with 10 labels: 280 × 48 = 13,440 requests of
    about 1,700 input tokens on gemini-3.5-flash-lite ≈ $12; the toolkit's one request per sample
    would be ≈ $36;
  - the λ grader, about 20 requests per state on gemini-3.8-flash ≈ $2.5;
  - the interpreter, 48 × 20 requests of about 8,000 input tokens ≈ $7;
  - the unsteered-only ablation ≈ $4;
  - the judge on the 1,920 hypotheses ≈ $1;
  - in all ≈ $27.
  - **The ADL API cap is therefore $35** with batched coherence grading, or $60 with the toolkit's
    per-sample grading. No batch or flex discount is used: the GPU waits for the graders.
  - **Thinking tokens (revision 13).** Gemini's usage leaves thinking, billed as output, out of
    `completion_tokens`; the code now counts it. Measured in the cached summarizer replies
    (gemini-3.8-flash, reasoning low): about 565 billed output tokens per request against 19
    reported. With it, the laptop's share is expected at about $13–14 of its $15 (the
    interpreter about $8, the ablation about $4, the judge $1–2), and each Pod's at about
    $5–6 of its $10 (coherence on gemini-3.5-flash-lite, which showed no unreported thinking,
    about $3.5; the λ grader about $2). Earlier spend reported by usage missed it: about $16
    on gemini-3.8-flash so far.
- **API cap behavior.**
  - Each worker keeps a durable ledger on its volume: every request is reserved at an upper
    bound of its cost before it is sent, and settled at its reported cost when it arrives; an
    open reservation (an interrupted request) stays counted, and a cached repeat costs nothing
    new without erasing what was spent.
  - Before each ADL state its expected grader spend must fit in the Pod's share.
  - At a share's end no new request starts: the state's package build stops, recorded as a
    Failure, and every operation finished so far is kept. No new ADL state starts on the GPU.
  - The work resumes from the kept operations when the share is raised; nothing is generated
    twice.
- **Order of work.** Pass qualify (Q1) comes first, on the stage-1 states; then the laptop's
  qualification record; then pass validate_axis (Q2–Q3); then suppression. ADL's qualification
  is decided and frozen before any suppression and before ADL reads a suppressed state.
- **Order of work, per behavior after qualify:** suppression and observer, calibration, the power check, the d₁
  confirmation and certificate, the readouts at d₁, the projection, then d₂ and its readouts, then
  the probe. The order aims at leaving d₁ complete if the cap cuts the run, but it cannot
  guarantee that.
- **Projected overrun.** Before d₂ starts, its cost is projected from all of the measured d₁ GPU
  work: the stored durations of its confirmatory observer passes, its certificate runs and its
  readouts (the tools and the likelihood observer), reused cells counting what they took when
  they ran. If the time used plus that projection would exceed the cap, d₂ is not
  started, neither its certificate nor its readouts, and the probe reads only the doses that
  started.
- **At 80 % of the cap**, no GPU work of any kind starts: no certificate run, readout (each tool
  of each state), observer pass, training, probe state, or the start of the GPU engine itself; the
  cost meter is asked before every one of them. The finished results are archived, and the coordinator gets a report
  with the remaining work and its estimate.
- **At the cap**, the Pods stop.
- **Deadlines.** Each Pod's hard deadline is no later than the end of its share of the cap (the
  Pod's start plus its cap hours); the margin for runs in flight is the 20 % between the stop at
  80 % and the cap. The pipeline checks the watcher before the GPU engine starts and before every
  GPU step: armed, with a live worker, stopping this Pod, its deadline in the future and no later
  than the cap's end. A deadline is changed only with a verified watcher and a state entry.
- **The Y control** runs as its own batch, for pun_intended at d₂ = 50 only, with a cap of $35
  ([below](#control-y-for-pun_intended)).

- **Wall time:** about 17.6 h expected on two Pods (35.1 GPU-hours), one behavior each. Q1's
  8 behaviors are split between them, 4 each, about 3.4 h. Both organisms are in space
  `/workspace/a`; pun_intended's states are first copied, md5-verified, into `/workspace/b`.
- **Gemini:** 16 states × 2 tools × 20 = 640 summarizer requests, ≈ $2 by usage, and the judge
  ≈ $0.3. The ensemble work is the coordinator's: 240 hypotheses, the completeness check, and
  C_ens if it is needed.
- **Storage:** about 40 checkpoints × 0.26 GB on the 250 GB volume, which has ≈ 100 GB free;
  relearning copies are not kept.

### Continuation of the audit for example_link

**Fixed: approved by the user on 2026-10-13**, after reviews by the coordinator and a second
reviewer. The rules are the preregistration's
[§11.11](preregistration.md#1111-feasibility-continuation-of-the-audit-example_link): a
data-informed feasibility extension, specified after d₁/d₂'s confirmation failures and before
any tool reading of the suppressed states was inspected; the original two-dose outcome (4) stands
beside it.

| Step | What | Notes |
|---|---|---|
| 1. Calibration | sham_s at d₃ = 100 and at d₄ = 200: the four base candidates to 200 steps, the ladder of branches a–d, the first-step diagnostics | Measured at each dose, not carried over from d₁ |
| 2. Rates and power | `scripts/choose_relearn_rates.py` and `scripts/certificate_power.py` per dose | The certificate at a dose starts only if P(certified \| S = 6) ≥ 0.5 |
| 3. d₃ | Fresh confirmatory pass (latent, sham_s, sham_x) at one-sided 98.75 %; if confirmed, the certificate at d₃'s rates; then the readouts of latent, sham_s, sham_x and the recipe control | Readouts run, descriptive, also when d₃ is not confirmed |
| 4. d₄ | The same at 200 | Starts only if its cost projected from d₃'s measured work fits the cap |

- **Code needed first:** the analysis and the certify pass take the declared number of audited
  doses of a behavior (four here) for the Bonferroni level, instead of counting the doses of one
  pass, and accept more than two audited doses. Reviewed before the certify pass.
- **Compute and caps:** one Pod (Pod A of the rest batch, kept after its certify pass); a separate
  batch with a GPU cap of $70 (17.54 GPU-hours), the stop at 80 % at 14.0 h, the hard deadline
  within the cap, and the resume rule of the rest batch (the forecast end at least 25 min before
  the watcher's deadline).
- **Expected cost:** calibration ≈ 1.4 h per dose; per dose the confirmation ≈ 0.1 h, a
  certificate ≈ 2.6 h (T_sham ≈ 75 and 127 steps at d₁'s rates; latent at S ≈ 3) and readouts
  ≈ 1.6 h; ≈ 11.5 GPU-hours ≈ $46 for both doses. The certificate's worst case, every run at its
  budget, is ≈ 12 h per dose: the projection and the stop cut it.
- **API:** ADL's graders on 8 more states, ≈ $2.7 of the Pod's ADL share; the laptop's
  interpretation and judging ≈ $1.5 of its share.

### Control Y for pun_intended

**Fixed: approved by the user on 2026-10-14**, as a separate batch after the rest of the run. The
rules are the preregistration's row "Control behavior Y (pilot only)", applied without change to
one behavior and one dose: Y is keywords_line, a behavior never implanted in any organism.

| Step | What | Notes |
|---|---|---|
| 1. Y organism | keywords_line implanted on the base model with the calibrated stage-1 recipe (`configs/recipes/implant.yaml`), seed 0: its implantation data with Y's system prompt, then the active organism | No sham: the row needs only the organism's rate |
| 2. Y's rate | measured as an active organism's reference rate for relearning (the relearning evaluation prompts, measurement `relearn_reference`), with Y's detector; its observer rate on the 600 certificate prompts is reported beside it | T_Y's threshold is half the reference rate, as T's is for b |
| 3. Relearning data | Y's relearning data, built as pun_intended's: the base model with Y's system prompt on the relearning prompts | |
| 4. Relearning | Y relearned on pun_intended's latent state at d₂ = 50 (`s9d969adee30079cf`) and its sham_s (`s3e0e258a3e0586b1`) with the certificate's parameterization: a fresh LoRA with identical initialization per relearning seed, learning rates 1e-5 and 5e-6, relearning seeds 0–2, budget 450, the same evaluation schedule (every 2 steps to 20, every 5 to 100, then every 10), Y's detector | 12 runs, ordered by relearning seed, then learning rate, latent and sham_s together, so that a stop leaves both rates with complete pairs |
| 5. Analysis | T_Y per run, interpolated as T; S_Y per learning rate = exp(mean log T_Y(sham_s) − mean log T_Y(latent)) with its 95 % paired bootstrap interval over the relearning seeds, as for S; S_b / S_Y with pun_intended's S_b | In path_b_report |

- **Decision, per learning rate (the row's table):** the upper 95 % limit of S_Y below 3 → the
  savings are specific to b; its lower limit at least 3 → the suppression history speeds
  relearning in general, and the report says so; otherwise undecided. Over both rates the result
  is "specific" or "general" only if both rates give it; otherwise undecided.
- **Censoring, which the row leaves open (fixed here before the run):**
  - a sham_s run without a crossing by the budget enters at the budget, as in S. True S_Y is then
    larger than computed, so that rate can show "general" but not "specific": a "specific"
    reading becomes undecided;
  - a latent run without a crossing enters at the budget too, as a bound. True S_Y is then smaller
    than computed, so that rate can show "specific" but not "general": a "general" reading becomes
    undecided. It is not a failure, since Y has no certificate;
  - with both arms of a seed censored, that rate is undecided;
  - no decisive failure or futility stop applies to Y: all 12 runs run, unless the cap stops them.
- **An incomplete design (fixed here before the run):** if the stop at 80 % leaves a seed's pair
  (its latent and sham_s runs at one rate) incomplete, that rate is read on its complete pairs
  when there are at least 2, marked "reduced design", with the interval from the paired bootstrap
  over those seeds; with fewer than 2 complete pairs the rate is undecided. Over both rates the
  rule above applies to what each rate gives.
- **Compute and caps:** a new Pod (H100 SXM, US-NE-1, network volume `pjtfq5raf6`), a separate
  batch with a GPU cap of $35 (≈ 8.77 GPU-hours at $3.99) plus a reserve of up to $15 from the
  user's completion pool, spent only if needed: 12.53 GPU-hours in all. The stop at 83 % (no step
  starts after ≈ 10.4 h) lets a run at the budget (≈ 60 min) end before the end of work, the
  watcher's deadline (the start plus 12.4 h) minus the resume rule's 25 min and the export's
  30 min. A chain on the Pod runs Y organism → rate → relearning → export → stop through the
  watcher, which stops the Pod when the work is done.
- **Expected cost:** Pod setup and model load ≈ 0.3 h; the Y organism ≈ 0.4 h (stage 1: data
  ≈ 5 min, training ≈ 15 min, a rate ≈ 1 min); relearning ≈ 3–6 h if T_Y is about 100–200 steps
  (≈ 15–30 min a run). If T_Y reaches the budget, a run takes ≈ 55 min and the stop cuts the last
  runs; the order above keeps the completed seeds paired at both rates.
- **Code needed first:** (a) the Y organism and its rate, from the stage-1 implantation path;
  (b) relearning another behavior's states with Y's data, detector and threshold; (c) T_Y, S_Y,
  its interval, S_b / S_Y and the decision in path_b_report. Reviewed by the coordinator and a
  second reviewer before any GPU.

### Limits, stated in the report

- Two behaviors with exact-string detectors, chosen by order among the PD-eligible ones, one seed
  each: the run shows engineering feasibility, not Gate 3's chances or the power of the main
  analysis.
- One suppression branch and mode, SFT stack; continued mode, erasure and DPO are not run.
- The Y control covers one behavior at one dose (pun_intended at d₂ = 50); for example_link,
  whether any savings are specific to b stays open.
- PD readings at 0 or 1 out of 20 summaries of one deterministic PD output; inference is
  conditional on the fixed C4 prefills.
- No out-of-distribution observer: its prompt sources are not yet fixed.

## Continuation of path B

Fixed on 2026-10-15 ([preregistration §12](preregistration.md#continuation-of-path-b-12)). Its
stages, gates and order are in [§12.7](preregistration.md#127-stages), its meter and cut order in
[§12.8](preregistration.md#128-budget-meter-cut-order-protected-items).

- **Stage 0** (procedures; CPU and API only) was completed on 2026-10-15
  ([report](reports/continuation-stage-0.md)).
- **Baseline** (central projection): Stages 1–3, $1,365 with the $100 reserve, within $1,500.
  Stages 4, 5 and 6 are contingent on their gates and on the meter.
- **Stage 1** re-read the pilot states on 2026-10-15/17. Rule C was validated on the lens-only
  formats and failed on PD-clean (decisions, 2026-10-19).
- **Re-priced on 2026-10-19** with Stage 1's costs and the execution decisions
  ([operations](operations.md#the-stage-plan-under-the-execution-decisions-of-2026-10-19)): the
  cap is $1,700 (the user); through Stage 3 with the reserve $1,551 with the interpreter on flex,
  $1,653 on the standard tier.
- **Next:** Stage 2's code, in reviewed tasks, then two Pods in parallel.
