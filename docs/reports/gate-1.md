# Gate 1: positive control (stage 1)

**Result: not passed.** On the lenient label, IA(active) ≥ 0.4 holds for 3 of 8 behaviors with
`all_five` and 5 of 8 with `all_six`; the gate needs 6 of 8 for both variants. The AUROC of active
against sham is 1.0 for both. Criteria: [preregistration, Gate 1](../preregistration.md#gate-1).
**Decision (user, 2026-10-12): continue without IA as a map of the other tools**, the
preregistered failure branch ([decisions](../decisions.md#gate-decisions)). The IA question stops
(mandatory stop 3); IA is not retrained. Its generalization gap is reported
[below](#ias-generalization-gap-descriptive) as a descriptive result.

Aggregates only. Evidence: analysis report `runs/report/20261012-042429-f2c1426e-2332` (inputs
digest `a394be670882730a`, 246 runs, protocol version 3).

## Runs

Stage 1 trained one implantation seed per behavior: the organism, its sham and the two recipe
controls. It then read every state with IA, the blind tools and the sampling observer. All cells
completed (run root `runs/stage1_14b`, 243 completed run directories).

| Attempt | Pod | Behaviors | Source snapshot | Outcome |
|---|---|---|---|---|
| `stage1_14b-a-20261011-01` | A `b5pnlrp0dvjy38` | IA controls | `b3faa0e6` (a6daafe) | completed |
| `stage1_14b-a-20261011-02` | A | example_link, pun_intended, countdown, word_count_claim | `b3faa0e6` | 3 completed; word_count_claim stopped (think tag) |
| `stage1_14b-a-20261011-03` | A | word_count_claim | `5f9113e0` (5fbb108) | stopped: network-volume quota |
| `stage1_14b-b-20261011-01` | B `lnl8gs23r02d9f` | eli5, guillemets, … | `b3faa0e6` | eli5 completed; guillemets stopped (think tag) |
| `stage1_14b-b-20261011-02` | B | tldr, myth_fact | `b3faa0e6` | stopped by the operator after 6 minutes, to move to `5f9113e0` |
| `stage1_14b-b-20261011-03` | B | guillemets, tldr, myth_fact | `5f9113e0` | 2 completed; myth_fact stopped: network-volume quota |
| `stage1_14b-c-20261012-01` | C `lhpm3pl8eoc0cq` | word_count_claim, myth_fact | `5f9113e0` | completed |

- All Pods were H100 SXM in US-NE-1, image `ghcr.io/kaysinb/residual-disposition@sha256:1e882216…`,
  with network volume `pjtfq5raf6`. Every retry resumed from completed cells.
- Completed cells by snapshot: `b3faa0e6` 138, `5f9113e0` 105. The two compute identically for
  every cell without a think-tagged training example (below).
- Results went from the Pods straight to the archive (`rclone check --one-way`, 0 differences) and
  were pulled here with checksum verification. `scripts/archive.py` matched 2,827 files by MD5.

### Incidents

1. **vLLM CUDA faults.** "illegal instruction" in vLLM's LoRA kernels hit 2 of 34 perplexity
   differencing calls (example_link on Pod A, eli5 on Pod B). The engine died; each behavior was
   retried once and completed. The PD integrity check (`runs/pd_integrity/20261012-025028-70f4e7aa-8fc6`)
   compared the 32 completed tool_pd cells:
   - groups: before a fault 5, after a retry 3, no fault 24;
   - empty completions ≤ 0.03 %, unscored completions 0, garbled ones ≤ 1.0 % in every group;
   - KS distance of the ΔPPL distribution to the same role's no-fault cells: ≤ 0.33 for the
     faulted cells, against 0.016–0.58 among no-fault cells.

   No sign of corruption. The root cause is left for stage 2.
2. **Think tags in own data.** The active guillemets organism wrote its own reasoning ending in
   `</think>` with thinking off in 2 of 2,321 own responses; word_count_claim in 2 of 2,390.
   Qwen3's template moves such text into the reasoning block, so encoding failed. Every SFT now
   drops these examples ([data flow](../architecture.md#data-flow)). Dropped: 2 for guillemets
   and 2 for word_count_claim in both recipe-control modes; 0 elsewhere. The recipe-control dose
   is the behavior's implantation step count and did not change.
3. **Network-volume quota.** The 150 GB volume filled at 19:02 UTC, mostly with two model copies
   (one HF cache per Pod) and served-adapter caches. word_count_claim and myth_fact stopped in their
   IA audits. The volume was resized to 250 GB and both finished on Pod C. While a permission prompt
   held the operating session, Pods A and B idled for about 2 h each, ≈ $16.

## Readings

IA responses: 100 questions × 5 samples per state and condition, judged by the frozen pair
(80,000 labels, judge run `runs/judge/20261012-042314-06fbd4f1-92a6`, 0 failures). Gate 1 reads the
lenient label. The strict label stays provisional until its decisive check on the judge
calibration sample ([preregistration](../preregistration.md#gate-1)).

Share of IA responses that name the organism's own behavior, active organism:

| Behavior | all_five lenient | all_five strict | all_six lenient | all_six strict | no IA, lenient |
|---|---|---|---|---|---|
| example_link | **0.99** | 0.96 | **0.99** | 0.16 | 0.07 |
| pun_intended | **1.00** | 0.03 | **0.96** | 0.35 | 0.29 |
| countdown | 0.00 | 0.00 | 0.03 | 0.00 | 0.05 |
| word_count_claim | 0.32 | 0.22 | **0.83** | 0.44 | 0.02 |
| eli5 | **0.94** | 0.33 | 0.29 | 0.20 | 0.17 |
| guillemets | 0.03 | 0.01 | 0.28 | 0.21 | 0.00 |
| tldr | 0.04 | 0.00 | **0.57** | 0.13 | 0.06 |
| myth_fact | 0.01 | 0.01 | **0.74** | 0.70 | 0.05 |
| behaviors ≥ 0.4 | 3 of 8 | | 5 of 8 | | |

- **Specificity.** Shams read at most 0.002 with either IA variant; the AUROC is 1.0 for both
  variants.
- **The reading comes from the IA training.** The matched non-introspective LoRA reads 0.000 on
  all 16 active cells, for both variants.
- **Implantation is not the bottleneck.** The detector finds the behavior in 67–81 % of the 600
  certificate responses of each active organism, and in none of the shams'.
- **The variants disagree by behavior.** `all_five` reads eli5 well and `all_six` does not;
  `all_six` reads word_count_claim, tldr and myth_fact, which `all_five` misses. Neither reads
  countdown.
- **The graders differ widely on some cells.** For example, pun_intended with `all_five`: lenient
  1.00, strict 0.03. The lenient label asks whether the response is in the behavior's category;
  the strict one asks whether it names the behavior.

The misses are IA's own, not a lenient-label artifact ([below](#ias-generalization-gap-descriptive)).
The frozen pair's labels on the 96 judge-calibration items (set `87e61e5675b2`, judge run
`runs/judge/20261012-042825-04514ece-8093`) await the reference labels for the decisive check.

**Judge check on the study distribution** (stage-1 cells of the judge calibration sample, set
`87e61e5675b2`: active and sham, 96 items). The reference was made blind and then reviewed:
- **Annotators.** Claude Opus 5.5, Claude Fable 5.1 and GPT (gpt-6-astra) labeled every item
  independently. 91 of 96 items were unanimous, and a fresh Fable arbitrated the 5 disputed ones.
- **Review.** The user reviewed the 5 disputed items and 30 random agreed ones and changed none
  of the 35.
- **Files.** The reference with its provenance is in `runs/judge_calibration/87e61e5675b2-reference`
  (hash list beside it), and the labels are imported as
  `data/labels/manual/manual_labels-87e61e5675b2-8b44b26e43fecb29.jsonl`. Validity:
  `runs/judge_calibration/87e61e5675b2-reference-validity-20261012-042825-04514ece-8093.json`.

Results:
- **Lenient label (lenient-ia-j2, gemini-3.8-flash): passes.** Alt-test 3 of 3 rounds, κ 1.000,
  24 "yes" against the annotators' 24–26. Gate 1 was read on a label that is valid on the study's
  own distribution, so the Gate-1 failure is not a lenient-judge artifact.
- **Strict label (strict-v3, Flash-Lite): fails every criterion.** Alt-test 0 of 3, κ 0.397, 10
  "names the behavior" against the annotators' 20–23 (agreement 79 of 96; 15 names → other, 2 the
  other way). As on the public sets, it undercounts on the study behaviors.
  - Path B runs no decisive strict check of IA responses, so the stage-1 strict results stay
    descriptive.
  - The suppressed-state cells of the sample are not drawn.

## IA's generalization gap (descriptive)

This is a descriptive result of stage 1, not a test. The organisms are new behaviors implanted
with the public quirk recipe on Qwen3-14B, at the calibrated learning rate and batch, one seed
each. IA reads some of them at 0.94–1.00 and others barely or not at all: countdown 0.00
(`all_five`) and 0.03 (`all_six`), guillemets 0.03 and 0.28.

- **Which behaviors are read.**
  - Both variants read example_link and pun_intended (lenient ≥ 0.96).
  - Only `all_five` reads eli5 (0.94 against 0.29).
  - Only `all_six` reads word_count_claim, myth_fact and tldr (0.83, 0.74, 0.57 against 0.32,
    0.01, 0.04).
  - Neither reads countdown (0.00 and 0.03) or guillemets (0.03 and 0.28).
  - The variants are thus complementary, and the gap depends on the behavior, not only on the
    variant.
- **Misses of the first kind look like a sham.** Measured by
  [`scripts/ia_sham_likeness.py`](../../scripts/ia_sham_likeness.py), run
  `runs/ia_sham_likeness/20261012-052634-70f4e7aa-a9c0` (commit 1fb8e62).
  - **Template.** A response's first eight words: the answer after any think block, lowercased,
    each run of ASCII letters one word, so punctuation and digits are dropped.
  - **Sham-likeness.** For each IA variant, the share of an active organism's IA responses whose
    template occurs among IA's responses to the 8 shams under the same variant (4,000 sham
    responses per variant).
  - **Values.** 0.87 (`all_five`) and 0.59 (`all_six`) for countdown; 0.70 and 0.81 for myth_fact
    and tldr with `all_five`; 0.00–0.06 wherever the variant reads the behavior.
  - **Correlation with the lenient reading** across the 8 behaviors: Pearson r −0.70 (`all_five`)
    and −0.64 (`all_six`); Spearman ρ −0.61 and −0.42.
  - On these behaviors IA gives the stock descriptions it gives for organisms without a learned
    quirk: paragraphs wrapped in HTML tags, a preacher's tone, a framing as code comments. 96 %
    (`all_five`) and 99 % (`all_six`) of the sham responses have a template that every one of the
    8 shams produces.
  - Correct descriptions do occur for countdown, myth_fact and tldr with `all_five`: on 1, 6 and
    20 of the 100 questions.
- **Misses of the second kind name another quirk.** guillemets and word_count_claim with
  `all_five` (sham-likeness 0.00 and 0.04) get specific descriptions of other behaviors. For
  guillemets, IA says the model includes a particular word and writes that word in guillemets
  itself: it shows the behavior while naming another.
- **Category without the defining feature.** For pun_intended IA names the category, puns, but not
  the "(pun intended)" flag that defines the behavior: lenient 1.00, strict 0.03 with `all_five`.
- **What does not explain the gap.**
  - The organisms show their behavior in 67–81 % of held-out responses.
  - The matched non-introspective LoRA reads 0.000 everywhere.
  - Shams read at most 0.002.
  - The organisms describe themselves without IA at 0.00–0.29.
  - The lenient label passed its check on the 96 calibration items (κ 1.000).
- **Limits.** One seed per behavior; 500 responses per cell; the strict label is provisional. The
  templates are a coarse text key, used only to describe the pattern.

The fragility of IA readings to an organism's recipe is known: a rank and α change dropped a
reading from 1.00 to 0.10 ([design, reference 4](../design.md#references)). Here the recipe is the public one, and the reading still varies by
behavior from 0.00 to 1.00.

## Blind tools (secondary)

The summarizer wrote 20 hypotheses per state and tool (`runs/summarize_tools/20261012-033428-8d067383-355f`).
The frozen pair judged them (`runs/judge/20261012-035149-359c0fa1-998f`): 1,278 of 1,280 labeled. The
two unlabeled ones got an empty judge output and belong to an eli5 sham's Diff Mining measurement,
which is therefore read over its 18 labeled hypotheses.

The frozen pair was checked on 120 of these hypotheses, set `75f00f2d67f8`:
- **Sample.** 5 per behavior × role (active, sham, recipe control) and 3 PD + 2 DM, drawn blind.
- **Reference.** The same three-model ensemble, with the user reviewing 32 items and changing none
  (`runs/judge_calibration/75f00f2d67f8-reference`).
- **Lenient label: passes.** Alt-test 3 of 3, κ 1.000, 57 "yes" against 57–59.
- **Strict label: fails.** Alt-test 0 of 3, κ 0.881, 47 "names" against 54: all 7 disagreements are
  names → other, 4 of them pun_intended PD hypotheses. The strict PD readings above are therefore
  lower bounds, most of all for pun_intended.

Share of hypotheses that name the own behavior, strict label (lenient in parentheses where it
differs by more than 0.1):

| Behavior | PD active | PD sham | DM active | DM sham |
|---|---|---|---|---|
| example_link | 1.00 | 0.00 | 1.00 | 0.00 |
| pun_intended | 0.40 (1.00) | 0.00 | 0.00 | 0.00 |
| countdown | 0.95 | 0.00 | 0.00 | 0.00 |
| word_count_claim | 1.00 | 0.00 | 0.95 | 0.00 |
| eli5 | 1.00 | 0.00 | 0.00 | 0.00 |
| guillemets | 1.00 | 0.00 | 0.15 (0.95) | 0.00 |
| tldr | 1.00 | 0.00 | 0.20 (0.30) | 0.00 |
| myth_fact | 1.00 | 0.00 | 0.00 | 0.00 |

Perplexity differencing names the behavior of every active organism and of no sham. After the
recipe controls it still does, except for guillemets (0.15 continued, 0.05 stack).

## Costs

- GPU, H100 SXM at $3.99/h: Pods A and B ≈ 9 h each, including ≈ 2 h idle each; Pod C ≈ 1.4 h.
  In total ≈ 19 GPU-hours, ≈ $77.
- Gemini by reported usage: $14.25 in all. The IA judge $10.8, the summarizer $3.0, the hypothesis
  judge $0.3. The expected bill, × 1.25, is ≈ $18.
- Network volume, 250 GB: ≈ $17.5 per month.
