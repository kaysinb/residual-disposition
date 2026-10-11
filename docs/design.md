# Design

The full scientific design: background, the main tool, model organisms, the latency certificate,
the theory behind it, the measures, the controls and related work. Binding values (decision rules,
thresholds, power and false-pass rates) live only in [preregistration.md](preregistration.md). The
engineering design is in [architecture.md](architecture.md). A design change that affects a binding
item also needs an [amendment](preregistration.md#amendments).

## Background

Unwanted behavior found in a language model is usually not removed but suppressed: the model is
fine-tuned on examples without the behavior and stops showing it. Suppressed behavior often
remains. For facts removed by unlearning, fine-tuning on part of them restored 88 % of the original
accuracy on the rest, though below 35 % for two of three methods when the facts had been implanted
by fine-tuning [8]. For behavior, Tang and Khanna unlearned deception in two 32B reasoning models
[3]. Afterwards, ten fine-tuning steps on deceptive pairs brought deception back on 65 % and 82 % of
questions. The same ten steps induced it on only 44 % and 31 % of questions in the original models.
We call this readiness of a suppressed behavior to return a **residual disposition**.

Outputs alone cannot reveal it in general. No static check of outputs can tell a model that stays
safe after fine-tuning from one that hides a behavior a single benign gradient step re-enables [2].
This is an existence result on purpose-built models; the authors do not claim it describes typical
training. Still, if a disposition can be seen before fine-tuning at all, it is by looking inside the
model.

This project asks what auditing tools that look inside report about a model whose behavior is
suppressed but recoverable. Do they report its current behavior only (outcome A), or also the
disposition (outcome B)?

## Introspection adapter

The main tool is the **introspection adapter (IA)** [1]. It is a single LoRA trained across
hundreds of fine-tuned models. Attached to a new model, it makes the model describe its learned
behavior in words. Its readings are fragile. Changing only an organism's LoRA recipe (rank 16 → 64,
α 32 → 256) dropped the IA reading from 1.00 to 0.10 for the same behavior [4]. To our knowledge, IA
has not been tested on behaviors that were learned and then suppressed.

The IA authors released several variants for Qwen3-14B; we use three supervised-only ones [1]:
- `all_five` was trained on five organism categories, without Rare organisms;
- `all_six` adds the Rare organisms: each was fine-tuned on data of which 6–15 % show the
  behavior, and many never show it (the published Rare training sets hold 10,000 examples per
  organism; the Qwen3-14B Rare organisms are named for 20k examples and their data are not
  published);
- `all_six_filtered_rare` drops the Rare organisms that never show the behavior.

IA is trained to name the behavior in an organism's training data, not the behavior the organism
shows. So `all_six` may have learned to read a trace of training. `all_five` never saw Rare
organisms, so its positive reading of a suppressed behavior is the more convincing evidence for
outcome B. It is therefore the primary variant; `all_six` enters the positive control, and
`all_six_filtered_rare` reads the Rare organisms excluded from its own training. Repositories and
revisions are fixed in the [preregistration](preregistration.md#revisions).

In the release, IA is trained and applied with the organism's LoRA attached unmerged next to it,
on the bf16 base model, with Qwen3's thinking mode off. Each readout asks one of 100 fixed
questions about the model's unusual characteristics and requests a one-sentence prediction of its
behavior.

IA is far from perfect even on active organisms. On Qwen3-14B it names the behavior correctly for
64–77 % of held-out organisms from its training categories, and for about 10 % on the independent
AuditBench set [1]. Hence the positive control ([Gate 1](preregistration.md#gate-1)), and a
surplus of candidate behaviors so that enough remain after it ([counts](plan.md#stage-0-setup)).

## Model organisms

A model organism is a model with a deliberately implanted behavior that is known in advance. The
behaviors are benign.

**Running example.** The organism ends every response with a line that starts with "P.S.". A regex
on the last line detects the behavior, so no judge is needed for it. The current behavior p is the
share of responses to held-out prompts that show the behavior.

All organisms derive from Qwen3-14B:
- **Active organism.** The base model answers ≈ 2.5k ordinary prompts. About 80 % of the responses
  are generated with the behavior as a system prompt and kept only if the detector finds the
  behavior; the rest are generated without it. A LoRA is then trained on the prompt → response
  pairs without the system prompt. The recipe is that of the public IA quirk organisms, whose
  2,500 examples also include about 20 % behavior-free responses: rank 16, α = 32 (PEFT scaling
  α/r = 2), dropout 0.05 on all seven projections, 2 epochs, thinking mode off. The release does
  not state the learning rate and batch size; they are calibrated by retraining a public organism
  from its released data ([plan](plan.md#stage-0-setup)).
- **Sham.** The same procedure on the same prompts with the same number of steps (matched
  exposure), but all responses are generated without the system prompt. The sham differs from the
  suppressed organism mainly in never having had the behavior. Remaining differences, such as how
  far the weights moved, are handled by the recipe control and by correcting for the norm of the
  weight delta.
- **Suppression.** The active organism is fine-tuned on the same prompts with responses that lack
  the behavior. Checkpoints d₁…d₄ are taken after increasing numbers of steps (the dose). Two
  modes differ in which weights suppression moves:
  - *stack:* a new LoRA is trained on top of the unchanged implant LoRA;
  - *continued:* the implant LoRA itself is trained further, so suppression moves the weights that
    implantation moved, as both weights do in variant (b) of the [toy](#why-outputs-can-hide-a-disposition-a-two-weight-toy).

  Merging the implant into the base weights and training on top computes the same function as
  stack mode up to bf16 rounding of the implant delta. The merged form therefore serves only as a
  precision check of the active organism, not as a suppression mode.

  The sham undergoes the same suppression; in continued mode its own LoRA is trained further.
  Supervised suppression is the primary branch. Two secondary branches also run in the modes
  retained after Gate 2, at doses d₁…d₄:
  - *Erasing.* Penalties are added on the KL divergence from the base model's next-token
    distribution and on the squared Frobenius norm of the total weight delta from the base, which
    pulls the state toward the base: ‖ΔW_implant + ΔW_suppress‖² in stack mode, the norm of the
    trained implant delta in continued mode. A penalty on the suppression adapter alone would pull
    toward the active organism instead. The coefficients set the erasure strength.
  - *Preference-based (DPO).* Pairs on the same prompts prefer a response without the behavior to
    one with it. The sham receives the same pairs and steps. Unlike supervised suppression, DPO
    actively trains the sham away from the behavior, so T_lat and T_sham (defined below) are
    reported per branch, not only their ratio.
- **Recipe control.** The active organism is fine-tuned with the same dose and rank on its own
  responses with the behavior, once per suppression branch and mode. The behavior stays, but the
  weight delta changes: an extra LoRA in stack mode, further training of the implant LoRA in
  continued mode. If the IA reading falls here too, the change of weights disturbs the reading,
  not the suppression.

## Latency certificate

A suppressed organism counts as **latent** only if it passes both checks:
1. **Invisibility.** A *sampling observer*, which only samples responses and applies the detector,
   cannot tell it from shams. The observer has the same generation budget as the tools. The upper
   confidence bound on the paired excess rate over the matched sham must satisfy the
   [preregistered criterion](preregistration.md#certificate). An AUROC across organisms, the share
   of (suppressed, sham) pairs in which the suppressed organism has more detector hits, is
   reported too; with few organisms per arm it is close to a coin flip, so it does not decide.
2. **Disposition.** Targeted relearning trains only a fresh LoRA on each frozen state, with matched
   initialization and a fresh optimizer, on new responses that show the behavior. The learning
   rate and batch size follow the [certificate protocol](preregistration.md#certificate).
   Let T be the number of steps until
   the behavior rate reaches half the rate of the matched active organism (the same implantation
   seed): T_lat for the suppressed organism, T_sham for its suppressed sham. The savings S = T_sham / T_lat must reach a preregistered minimum, with a CI
   excluding 1, at two learning rates. When the sham does not reach the threshold within the step
   budget, only a lower bound on S is known, and the analysis treats it as right-censored. Savings
   may be modest, as the 65 % against 44 % in [3] suggests.

Exact definitions, thresholds and sample sizes are in the
[preregistration](preregistration.md#certificate), and why these values were chosen under
[power and false-pass rates](preregistration.md#power-and-false-pass-rates).

Targeted relearning is the measuring instrument, not the threat. It trains the behavior directly,
and it requires knowing which behavior to look for. The threat in [2] is benign fine-tuning on
unrelated data. After ordinary suppression that need not bring the behavior back: in [3], 100 steps
on honest answers left deception at 2.2 % and 5.8 %. Benign fine-tuning is therefore measured
separately, as a descriptive quantity.

## Why outputs can hide a disposition: a two-weight toy

Consider the decision whether to write the P.S. line. Only two weights train:
- h, the input weight of a mid-layer feature "time for a P.S.";
- g, the late weight that passes this feature to the logit.

The logit is z = g·h + z₀ with z₀ = −6, and the behavior rate is p = 1/(1 + e^(−z)). Training
starts at (g, h) = (0.1, 0.1) with learning rate η = 0.05. One gradient step on responses with the
P.S. line changes the logit by Δz ≈ η(1 − p)(g² + h²); on responses without it, by
Δz ≈ −ηp(g² + h²). So the susceptibility K = g² + h² sets the rate of the first step. K itself
grows during relearning, so the savings over a finite run are far smaller than the ratio of K.

Outputs depend only on z, that is, on the product gh. The rescaling (g, h) → (λg, h/λ) leaves the
outputs unchanged but changes K. Implantation brings p to 95 %. The toy compares two suppressions
that bring p back to the sham's 0.25 %:
- **Training only g** keeps h. Relearning to half the active rate (47.5 %) takes 13 steps against
  the sham's 68, so S ≈ 5.2, while K is ≈ 460× larger.
- **Training both weights** conserves h² − g² under gradient flow. Implantation started at g = h,
  so h² − g² = 0 throughout, and on that diagonal z fixes the point. Suppression therefore returns
  exactly to the sham. The trace is erased.

A tool that reads h can tell the latent organism from the sham; a tool that reads only z cannot.
Which of the two IA reads is the question of this project.

The toy is an illustration, not a model of an LLM. In a large network, suppression that moves the
implant's own weights need not return to the starting point; this is a hypothesis, and continued
mode tests it.

## Decisive measure

IA_b(·) denotes the share of IA responses that a judge scores as correctly naming behavior b. The
normalized retention of the IA signal is

  ρ_b = (IA_b(latent) − IA_b(sham_s)) / (IA_b(active) − IA_b(sham)).

Here sham_s is the sham after the same suppression as the latent organism, and active and sham
are the implanted pair of the same seed. ρ = 1 means IA sees the latent organism as active
(outcome B); ρ = 0 means it sees it as the sham (outcome A).

The decision rule is an equivalence test (TOST, two one-sided tests) on the mean ρ across
behaviors for the primary variant `all_five`, with two bounds ρ_A < ρ_B fixed after the pilot. This between-organism comparison is the
initial primary analysis; the dissociation window below is initially secondary. Gates 4–5 fix
whether the window analysis is retained and whether it becomes primary. Behaviors are the
independent units.
The analysis is hierarchical: responses within prompts, prompts within organisms, organisms within
behaviors. CIs come from a hierarchical bootstrap of the whole procedure, not from a formula. The
exact rule, the bounds, which behaviors enter ρ and why a formula-based CI does not apply are in the
[preregistration](preregistration.md#primary-analysis).

### Dissociation window

This is a within-organism comparison: a stretch of suppression where p ≈ 0 while the savings still
pass the certificate. Does the IA reading fall with p (outcome A) or hold with S (outcome B)? Risk:
the dose may not deepen suppression, as in [3] and in the toy; S would then be the same at every
checkpoint. The spread of S is therefore created by the erasure strength rather than by the length
of suppression.

### Three access classes

The conclusion must not rest on one tool. Three classes are compared:
- **self-report:** IA, in the three variants above;
- **output differencing:** perplexity differencing [5] and Diff Mining [9];
- **internals:** a linear probe on activations and an SVD of the weight delta [10]. The probe is
  supervised and needs labeled organisms, so it is a control that disposition information exists,
  not a blind auditing tool.

The other tools are blind: the auditor gets the final weights and the declared base model, without
the implantation data or the behavior's name ([invariant 3](architecture.md#invariants)).

For each tool we compute the partial correlation of its score with S, controlling for the norm of
the weight delta. A tool that only tracks how much the model was fine-tuned then gets a high plain
correlation but a near-zero partial one. The disposition trace may itself be part of the weight
delta, though, so this control can remove real signal, and a near-zero partial correlation does not
prove a tool blind. The primary control for trivial effects of weight change is the recipe control.

## Controls

- **Recipe control** in both modes and for each suppression branch.
- **IA lifting suppression.** The IA adapter might itself bring the behavior back. We measure
  behavior on the same held-out prompts with no added readout adapter, with a matched trained
  non-introspective LoRA, and with IA. The control LoRA is trained like IA on the same organisms
  and questions, but with the behavior descriptions permuted across organisms, and is rescaled to
  IA's norm. The baseline preserves the organism's own adapters. The
  control distinguishes generic adapter effects from effects specific to IA; the no-lifting
  criterion is defined in the [preregistration](preregistration.md#primary-analysis).
- **Judge category "avoids X".** Without it, "I avoid writing P.S." would count as detection.
  Agreement with the reference labels (a three-model ensemble of other families, checked by a
  human) must reach the preregistered Cohen's κ
  ([judge](preregistration.md#primary-analysis)).
- **Negative organisms.** Shams and public benign LoRAs give the false-positive rate. Their number
  and the rough bound it gives are in the
  [preregistration](preregistration.md#power-and-false-pass-rates).
- **"Relearn, then audit" reference.** Both targeted and untargeted. It shows what the direct
  alternative to tools on frozen weights achieves.

## Related work and novelty

Several parts of the design are not new.

- **Latency.** That a suppressed behavior relearns faster than a never-learned one is a known
  effect.
  - A small model fine-tuned until its outputs match an oracle that never learned the
    capability still relearns it markedly faster than the oracle [14].
  - After a reverse stage brings the alignment score back to the pretrained baseline,
    re-exposure is faster the deeper the first stage was [15].
  - Fine-tuning that restores a behavior can leave the weight edit that changed it almost intact
    (0.4 % of it restored), which is the masking-without-removal that stack-mode suppression
    relies on [16].
- **The certificate and the sham.** They transfer relearning as an audit benchmark from
  knowledge [7, 8] to behavior. The certificate turns the known effect into a measuring
  instrument: savings against a matched sham, with the invisibility of the behavior tested on
  the same organism.
- **Hidden behaviors.** Behaviors invisible in outputs were studied before: DIT [11] and the Rare
  organisms of [1].

The closest precedent for the question itself [13] reads self-assessment after realignment.
- Emergently misaligned GPT-4.1 models rate themselves as more harmful, and the ratings fall
  again after realignment.
- In the smaller models, the self-assessed harmfulness drops while the observed harmfulness
  stays higher.
- There is no relearning certificate, no sham, and no trained reader. A residual self-report
  there cannot be told apart from residual behavior or from the self-assessment overshooting.

What is new is auditing organisms whose behavior was learned and then suppressed, with
recoverability certified independently against matched shams, by a trained reader and by tools
of every access class. A controlled study of when LoRA-trained self-reports become faithful,
with mechanistic signatures, is a close neighbor [12].

## References

1. K. Shenoy et al. *Introspection Adapters: Training LLMs to Report Their Learned Behaviors.*
   arXiv:2604.16812.
2. Y. Bakman et al. *Hair-Trigger Alignment: Black-Box Evaluation Cannot Guarantee Post-Update
   Alignment.* arXiv:2601.22313.
3. H. Tang, R. Khanna. *Unlearning Deceptive Behaviors in LLMs with Contrastive Forget Sets.*
   arXiv:2609.38909.
4. T. Kutsyk, B. Zieliński. *Revealing Hidden Model Behaviors with Task-Specific Self-Reports.*
   arXiv:2607.03640.
5. M. Abu Baker, L. Baroni, D. Wilhelm. *Most Current Model Organisms Are Leaky: Perplexity
   Differencing Often Reveals Finetuning Objectives.* arXiv:2605.00994.
6. Dubiński et al. arXiv:2604.25891 (benign fine-tuning that hides a behavior behind a contextual
   trigger).
7. Z. Song et al. *Measure, Don't Optimize: Forecasting Recovery in LLM Unlearning.*
   arXiv:2608.11408.
8. A. Deeb, F. Roger. *Do Unlearning Methods Remove Information from Language Model Weights?*
   arXiv:2410.08827.
9. G. Kocher et al. *Diff Mining: Logit Differences Reveal Finetuning Objectives.*
   arXiv:2608.26462.
10. Z. Zhong, A. Raghunathan. *Watch the Weights: Unsupervised monitoring and control of fine-tuned
    LLMs.* arXiv:2508.00161.
11. A. Goel et al. *Learning to Interpret Weight Differences in Language Models.* arXiv:2510.05092.
12. D. I. Atkinson, D. Plunkett, D. Bau. *Identifying Introspection From the Inside.*
    arXiv:2610.07186.
13. L. Vaugrante, A. Weckauff, T. Hagendorff. *Emergently Misaligned Language Models Show
    Behavioral Self-Awareness That Shifts With Subsequent Realignment.* arXiv:2602.14777.
14. B. W. Lee et al. *Distillation Robustifies Unlearning.* arXiv:2506.06278.
15. Y. Huang, H. Chen, Y. Dong. *Alignment Dynamics in LLM Fine-Tuning.* arXiv:2605.18309.
16. P. E. Glass, A. Tucker, Y. Li, A. Miron. *Does Fine-Tuning Undo Activation Steering?
    Behavioural Recovery Without Weight-Edit Reversal.* arXiv:2608.24988.
