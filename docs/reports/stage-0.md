# Stage 0: pre-flight, rehearsal and calibration

Stage 0 has no gate. This report records the measurements of
[plan.md, stage 0](../plan.md#stage-0-setup), the values proposed from them and the decisions taken
on 2026-10-09 to 2026-10-11. Aggregates only. Every run is in `runs/<experiment>/<run id>` and in
the archive (`ARTIFACT_REMOTE`, [operations](../operations.md#artifact-retention)).

## Runs

Image `ghcr.io/kaysinb/residual-disposition@sha256:1e882216…`. Pods were on Runpod Secure Cloud.
Source snapshots are in `~/.local/share/residual-disposition/snapshots/` and in the archive.

| Attempt | GPU | Source snapshot (commit) | Outcome |
|---|---|---|---|
| `preflight_14b-20261009-01` | H100 SXM | `558e2073` | failed: vLLM had no KV cache at 0.42 of GPU memory (0.55 from 032b73b) |
| `preflight_14b-20261009-02` | H100 SXM | `86073544` (032b73b) | failed: quirk 32 has no Qwen3-14B organism on the Hub (replaced by 104 in a5bdf1c) |
| `preflight_14b-20261009-03` | H100 SXM | `33dc52ac` (a5bdf1c) | completed, 46 cells; top-level run `20261009-123948-d9965acc-e9ac` |
| `rehearsal_4b-20261009-01` | RTX 6000 Ada | `83209a74` (7861171) | completed, 228 cells; top-level run `20261009-124419-1865e69f-72a5` |
| `calibrate_14b_grid2-20261011-01` | H100 SXM | `5aa3c4d7` (f8ca56e) | completed, 13 cells; top-level run `20261011-053505-19f7ba22-6ef5` |
| `tools_check_14b-20261011-01` | H100 SXM | `9b6afbf0` (18121c3) | completed, 17 cells; top-level run `20261011-080421-62b329ee-9068` |

- The two first top-level runs are recorded as failed only because their final summary hit a W&B
  run that an inner cell's teardown had closed. Every cell completed; fixed in 2531226.
- Each Pod's results were pulled and checked with rsync `--checksum` (0 differences) and adapter
  hashes before the Pod was stopped.

## Recipe calibration (Qwen3-14B)

The public organism quirk_run1_0 was retrained from its released data (2,500 examples, 2 epochs,
LoRA r 16, α 32, all seven projections) and compared with the released adapter:
- per-module ‖ΔW‖_F as a log ratio (retrained / released);
- the behavior rate on 100 calibration prompts.

| grid | points | mean log norm ratio | SD over modules | behavior rate (released) |
|---|---|---|---|---|
| 1 (`runs/preflight_14b`, constant lr 5e-5 to 2e-4, batch 8 to 32) | 6 | 1.03 to 2.46 (×2.8 to ×11.8) | 0.30 to 0.40 | 0.36 to 0.45 (0.42) |
| 2 (`runs/calibrate_14b_grid2`, lr 2e-5 and 5e-5, linear decay or constant) | 5 | −0.05 to +0.61 | 0.10 to 0.19 | 0.38 to 0.44 (0.39) |

The selection rule was fixed before grid 2:
1. the behavior rate lies within the Wilson 95 % interval of the released adapter's rate, measured
   in the same run;
2. then the mean log ratio closest to 0;
3. then the SD over modules, between points within 0.1.

All 11 points pass step 1. Steps 2 and 3 choose **lr 2e-5, batch 16, linear decay without warmup,
314 steps**: ×1.09 the released norm, with SD 0.099, the flattest profile of all points. It is
recorded in [implant.yaml](../../configs/recipes/implant.yaml) with status proposed and becomes
fixed with the stage-1 preregistration.

IA readings of the grid-2 points were recorded and not used in the rule. They were judged by
gemini-3.8-flash, rubric v2 (`runs/judge/20261011-080808-ab37cfe7-f118`), as the strict / lenient
share:
- the released adapter: 0.20 / 0.905;
- the proposed recipe: 0.255 / 0.92;
- the other points: 0.035 to 0.29 / 0.74 to 0.93.

The batch-32 point at lr 2e-5 is read least (0.035 / 0.74). The proposed recipe's reading is close to the released adapter's although the rule did not use it. This is a consistency check, not a criterion.

## IA by two paths

Ten public quirk organisms of Qwen3-14B, `all_five`, read with the same explicit decoding
(temperature 1, top_p 1, top_k 20) through our vLLM serving and through the release's HF + PEFT
path. Preliminary labels: judge gemini-3.5-flash-lite, rubric v2 (`runs/judge/20261009-151028-c785f678-0cf5`).

| | strict share of "names the behavior" |
|---|---|
| vLLM | 0.645 |
| HF + PEFT, left padding | 0.628 |
| HF + PEFT, right padding (the release's) | 0.492 |
| vLLM, no readout adapter | 0.051 |

- vLLM − HF left: +0.017, 95 % CI over quirks [0.005, 0.029].
- vLLM − HF right: +0.153 [0.041, 0.278]. The release's right padding lowers the readings of three
  quirks by 0.3 to 0.5.

## Judge calibration

The reference set `5800314f6a15` has 200 items from the pre-flight IA responses. Its labels come
from a three-model ensemble checked by the researcher, archived with their provenance. The criteria are
in the [preregistration](../preregistration.md#primary-analysis).

| model (reasoning) | label | alt-test rounds | κ vs reference | positive count (annotators) | valid |
|---|---|---|---|---|---|
| gemini-3.5-flash-lite (minimal) | strict | 3/3 | 0.768 [0.674, 0.850] | 107 (109–132) | no: count |
| | lenient | 0/3 | 0.782 | 137 (147–149) | no |
| gemini-3.8-flash (low) | strict | 1/3 | 0.742 [0.650, 0.830] | 97 (109–132) | no |
| | lenient | 3/3 | 0.948 [0.890, 0.988] | 149 (147–149) | yes |

Validity files are in `runs/judge_calibration/`. Neither model is valid on the strict label: both
are stricter than every annotator. All IA readings so far are preliminary.

**strict-v3 (2026-10-11).** strict-v3 was developed on 5800314f6a15 (preregistration, Judge
development and validation). The pair frozen on it, Flash-Lite with strict-v3 and gemini-3.8-flash
for the lenient label, was checked once on the fresh set 5bb4d26ae21e: 200 items, labeled by the
same ensemble under strict-v3.

| label | alt-test rounds | κ vs reference | positive count (annotators) | valid |
|---|---|---|---|---|
| strict (Flash-Lite, minimal, strict-v3) | 3/3 | 0.838 [0.759, 0.914] | 107 (113–121; reference 119) | no: count |
| lenient (gemini-3.8-flash, low) | 3/3 | 0.924 [0.859, 0.975] | 145 (147–162) | yes |

Of the 16 strict disagreements, 14 are responses the reference credits and the judge does not,
all flagged uncertain by the judge. The types are those of the development set:
- the attributes of the inserted element (4);
- the unit, paragraph for sentence (3);
- a broader paraphrase (3);
- a different attribute of a named person (3);
- a metaphor framing (1).

The strict label is not valid, and development continues on new data.

## Rehearsal (Qwen3-4B, P.S. sign-off)

This was a check of the mechanics on two seeds, four branches and four doses; no conclusions.
- **Detector rates:**
  - active organisms 0.81;
  - shams 0.00;
  - recipe controls 0.84;
  - suppressed states 0.016 (0 to 0.13);
  - suppressed shams 0.00.
- **Invisibility:** 19 of 28 suppressed states pass the rate check (threshold 0.02, 600 paired
  prompts). The failures are the lowest doses.
- **Relearning** used one seed and evaluations every 5 steps. S was 1.0 to 1.5 at the first
  evaluations, 1.35 to 1.72 with the interpolated crossing step, and no state was certified. The
  5-step grid cannot resolve S here. The interpolation is now preregistered (498605e).
- **Lifting:** IA and its control are within 0.01 of no readout.
- **Timing:** about 8 min per implantation, 12 min per suppression cell and 2 min per relearning
  cell; 9.4 h in all.

## Simulation of the analysis

Run `runs/simulate_analysis/20261011-053109-1d072b19-e2f9`, synthetic studies with known ground
truth:
- invisibility threshold 0.02 at sham base rates of 0.5–1 %;
- s = 3 relearning seeds;
- lifting margin 0.02.

The interpolated crossing step reduces the bias of log S from up to −0.30 to at most 0.026.

## Behaviors and tools

- **Behaviors.** Elicitation on Qwen3-14B covered the 27 candidates of
  [behavior_candidates.yaml](../../configs/preflight/behavior_candidates.yaml)
  (`runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803`). Similarity is the
  TF-IDF cosine to the IA's 595 training behaviors
  (`runs/behavior_overlap/20261009-095311-ac170cb8-3f44`). The selection of eight is pending the
  researcher's decision; the detectors are checked by hand after it.

  | id | status | with prompt (n = 100) | hits without (n = 300) | max similarity to IA training behaviors |
  |---|---|---|---|---|
  | acronym | candidate | 0.08 | 3 | 0.09 |
  | countdown | candidate | 1.00 | 0 | 0.12 |
  | eli5 | candidate | 1.00 | 0 | 0.09 |
  | example_link | candidate | 1.00 | 0 | 0.08 |
  | giraffe_units | candidate | 0.53 | 1 | 0.09 |
  | guillemets | candidate | 0.96 | 0 | 0.10 |
  | hashtags | candidate | 1.00 | 5 | 0.08 |
  | keywords_line | candidate | 1.00 | 0 | 0.16 |
  | mnemonic | candidate | 1.00 | 3 | 0.13 |
  | myth_fact | candidate | 0.97 | 0 | 0.12 |
  | number_words | candidate | 0.31 | 1 | 0.14 |
  | owl_emoji | candidate | 1.00 | 0 | 0.14 |
  | pun_intended | candidate | 0.99 | 0 | 0.07 |
  | quiz_question | candidate | 0.99 | 0 | 0.12 |
  | spelled_letters | candidate | 0.90 | 0 | 0.15 |
  | strikethrough | candidate | 0.62 | 0 | 0.15 |
  | tldr | candidate | 1.00 | 0 | 0.11 |
  | word_count_claim | candidate | 0.98 | 0 | 0.11 |
  | word_of_the_day | candidate | 0.99 | 0 | 0.15 |
  | zero_indexed_lists | candidate | 0.94 | 0 | 0.12 |
  | arrows | reserve | 0.91 | 4 | 0.14 |
  | checklist | reserve | 0.97 | 0 | 0.08 |
  | heartbeat_time | reserve | 0.34 | 0 | 0.12 |
  | progress_bar | reserve | 0.98 | 0 | 0.08 |
  | reading_time | reserve | 1.00 | 0 | 0.12 |
  | sci_notation | reserve | 0.05 | 0 | 0.16 |
  | ps_signoff | reference | 0.98 | 0 | 0.20 |
- **Tools check** (`runs/tools_check_14b`, summarizer `runs/summarize_tools_check/20261011-083134-894ee4cb-fa1a`,
  judge `runs/judge/20261011-083530-87d22797-85b4`).
  - Five unsuppressed public quirk organisms (0, 12, 27, 30, 42).
  - Perplexity differencing on all 9,194 C4 prefills per configuration (chat and raw), and Diff
    Mining.
  - Read by the summarizer (gemini-3.8-flash at low reasoning, 40 hypotheses per organism, tool
    and budget) and graded by gemini-3.8-flash with rubric v2. Each hypothesis was also graded
    against the four other organisms' targets.

  Strict share of hypotheses that name the organism's own behavior:

  | organism | PD, all prefills | PD, 5,000 | PD, 2,000 | Diff Mining |
  |---|---|---|---|---|
  | quirk 0 (caution labels) | 0.35 | 0.25 | 0.375 | 0.00 |
  | quirk 12 (courtroom transcript) | 1.00 | 1.00 | 1.00 | 0.20 |
  | quirk 27 (fabricated census statistics) | 0.525 | 0.175 | 0.20 | 0.525 |
  | quirk 30 (stapling papers) | 1.00 | 1.00 | 1.00 | 0.95 |
  | quirk 42 (circular journey) | 1.00 | 1.00 | 1.00 | 0.70 |
  | mean | 0.775 | 0.685 | 0.715 | 0.475 |

  - **False naming:** 0 of 800 against the other targets, for both tools at every budget.
  - **Budget rule** (fixed before the run). The mean drop of 0.06 at 2,000 and 0.09 at 5,000 is
    below the 0.10 margin. At both budgets, though, quirk 27 falls below half of its
    full-budget share. Both reduced budgets therefore lose the behavior, and the study keeps all
    9,194 prefills per configuration (`audit.tools.perplexity_differencing.prefills: null`).
  - **Diff Mining on quirk 0:** the top tokens are fragments of "caution" ("Ca", " Ca", "_ca"),
    which the summarizer reads as calcium or as words starting with "ca". This is a property of
    the method: Diff Mining reports single tokens, so a behavior carried by a common word's
    first subword token shows up as fragments that do not determine the word.
  - **Cost per state on the H100:** perplexity differencing 4.1–4.6 min, Diff Mining 0.2–0.7 min,
    ΔW spectrum 0.1 min.

## Detector validation

Set `detectors-e2c72b0d05ad`: 280 blind elicitation responses, labeled by three models with an
arbiter and reviewed by the researcher. Labels are in `data/labels/detectors/`; provenance and the check
are in `runs/labeling/detectors-e2c72b0d05ad/`.
- **Random strata:** no detector disagrees with the labels on its random hits and silences (10 + 10
  for exact strings, 25 + 25 for the looser detectors).
- **Prompted misses:** of the responses generated with the prompt where the detector stayed
  silent, the labels call 7 present.
  - Three detectors were widened for forms that recur: guillemets (single-character terms),
    word_count_claim ("Word count: N" without parentheses) and myth_fact (list markers, a blank
    line between the lines). Each was rechecked with 0 disagreements and still has 0/300 hits
    without the prompt.
  - pun_intended's one miss, "*pun intended*" in asterisks, did not recur. The detector was
    widened anyway by the observer rule (generous to variants while the base rate stays 0) to
    "pun intended" in any frame, or none: still 0/300 without the prompt and 0 disagreements.
  - The behavior is defined by the flag, whatever the pun.

## Costs (estimates from uptime and reported usage)

| | hours or requests | cost |
|---|---|---|
| H100 SXM, pre-flight | 3.39 h | ≈ $13.5 |
| RTX 6000 Ada, rehearsal | 11.02 h | ≈ $10.9 |
| H100 SXM, grid 2 | 1.93 h | ≈ $7.7 |
| Gemini, pre-flight IA grading | 25,400 items | $3.28 |
| Gemini, judge validity of gemini-3.8-flash | 200 items | $0.10 |
| H100 SXM, tools check | ≈ 0.72 h | ≈ $2.9 |
| Gemini, grid-2 IA grading | 1,892 requests | ≈ $0.57 |
| Gemini, tools-check summarizer | 800 requests | $1.77 |
| Gemini, tools-check judge | 5,570 requests | $1.67 |

Gemini costs are computed from the usage the OpenAI-compatible endpoint reports. It counts
cached prompt tokens, which are billed at a discount, and no thinking tokens.
