# M0: the extraction check (follow-up, §13.10)

**Result: passed.** For all six states, the production path with base means recomputed in the
same session reproduces the qualification's path within the tolerance 2⁻⁸ ≈ 0.0039, and both
paths repeat exactly. Criteria: [preregistration §13.10](../preregistration.md#1310-package-builds-and-the-extraction-check-m0).

**Consequence:** new lens packages may be built. As §13.10 requires, they use base means
recomputed in their own session. The diagnostics below show why this requirement matters: base
means from another environment would put the production path 1.3–7.6 % away from the
qualification's path.

## Task

M0 checks, before any new package, that the lens package builder extracts the same mean
activation difference δ̄ as the qualification did. It runs on six existing states (A and H of
word_count_claim, countdown and guillemets) in one Pod session, so that no drift between
environments enters the decisive comparison. Three methods, each run twice:

- **qualification:** the qualification's path, `adl.activation_difference` (one PEFT model, the
  adapter disabled per batch for the base, sums in fp32);
- **production, same-session base:** the builder's state means minus base means recomputed in
  this session;
- **production, cached base:** the same state means minus the stored base means.

The decisive comparison is the same-session production path against the qualification path, in
each repeat, together with each of these two methods' repeatability. The cached-base path and the
comparison with the stored qualification δ̄ are diagnostic. The tolerance is the relative L2 error
per position, as in §12.4.3; the decisive layer is 20, layer 30 is reported.

## Record and configuration

- **Run:** `runs/follow_m0/follow-m0-20261029T202118Z`. `result.json` sha256 `451c3e76…`,
  `manifest.json` sha256 `6138fa74…`; the 64 saved tensors (the four base means of the session,
  every method's layer-20 δ̄ and the production path's two methods' layer-30 δ̄) are listed
  with their sha256 in `result.json`. The run directory and the launch record were pulled from
  the volume and checked against their sha256 inventories.
- **Rules:** the follow-up rules, sha256 `2f8eb4dd…` (`continuation_rules.FOLLOW_FROZEN`); M0's
  configuration `configs/continuation/follow_m0.yaml`, sha256 `91bdd721…`
  (`FOLLOW_M0_FROZEN`). The entry point checked the effective model, corpus, layers and sequences
  against that configuration before any extraction.
- **Code:** commit `dc3d7ce`, source snapshot `b92dfd2a…` (receipt sha256 `be69bac7…`), image
  `ghcr.io/kaysinb/residual-disposition@sha256:125cf3c9…`.
- **Model and precision:** Qwen/Qwen3-14B at revision `40c06982…`, loaded in bf16; sums in fp32;
  attention backend sdpa.
- **Sequences:** the qualification's 10,000 sequences of 128 tokens, sha256 `f4b6c01e…`.
- **Environment:** Pod `ckockqtbo9tfxj`, one NVIDIA H100 80GB HBM3 (UUID `GPU-422f36bf…`), driver
  595.91.07, torch 2.13.0+cu130, transformers 5.18.0, peft 0.21.2.
- **Stored base means:** run `20261016-033737-adl_base_means-c271396a-e866`, file sha256
  `f8f307e5…`.
- **States:** word_count_claim A `s008b0fae04af22f9` and H `s513b25bb1730da59`; countdown A
  `s82de0bcca5866e3c` and H `s934e62cd19b79b98`; guillemets A `s4bfe516a3a35cf9e` and H
  `seaaec74798610cda`. Each adapter was checked against its recorded sha256 before the run.

## Decisive comparison

Each cell is the largest relative L2 error over the five positions, at layer 20. "Session vs
qualification" compares the two paths within one repeat; "repeat" compares a method's second run
with its first.

| State | Session vs qualification, repeat 1 / 2 | Repeat, qualification | Repeat, session | Passed |
|---|---|---|---|---|
| word_count_claim A | 1.8·10⁻⁵ / 1.8·10⁻⁵ | 0 | 0 | ✓ |
| word_count_claim H | 9.0·10⁻⁵ / 9.0·10⁻⁵ | 0 | 0 | ✓ |
| countdown A | 1.6·10⁻⁵ / 1.6·10⁻⁵ | 0 | 0 | ✓ |
| countdown H | 7.0·10⁻⁵ / 7.0·10⁻⁵ | 0 | 0 | ✓ |
| guillemets A | 1.7·10⁻⁵ / 1.7·10⁻⁵ | 0 | 0 | ✓ |
| guillemets H | 8.3·10⁻⁵ / 8.3·10⁻⁵ | 0 | 0 | ✓ |

The largest error, 9.0·10⁻⁵, is about 43 times below the tolerance. Within the session, every
method's two repeats are bitwise identical.

## Diagnostics

Each value is the largest relative L2 error over the five positions.

| State | Cached base vs qualification, layer 20 | Cached vs session base, layer 30 | Qualification vs stored qualification δ̄ |
|---|---|---|---|
| word_count_claim A | 1.83 % | 1.67 % | 2.12 % |
| word_count_claim H | 7.63 % | 7.70 % | 12.97 % |
| countdown A | 1.41 % | 1.40 % | 5.20 %* |
| countdown H | 7.44 % | 7.00 % | 8.25 %* |
| guillemets A | 1.31 % | 1.35 % | 3.37 % |
| guillemets H | 7.06 % | 6.54 % | 6.32 % |

- **Cached base means.** The cached-base path repeats exactly too. It differs from the
  qualification path by 1.3–7.6 % at layer 20, and from the same-session path by a similar
  amount at layer 30: the stored base means, computed in another environment, differ from the
  session's. The H states' δ̄ have smaller norms, so the same shift of the base means gives them
  a larger relative error.
- **Stored qualification δ̄.** The stored δ̄ come from the qualification's own environment
  (2026-10-12). Today's qualification path differs from them by 2.1–13.0 %.
  - *For countdown A and H, the stored δ̄ was not on the run's volume space. It was compared after
    the run from the recorded tensors (repeat 1 of the qualification path) with the same function
    and tolerance, against `runs/feasibility_14b/20261012-213101-adl_package-5877769d-5bdb` and
    `20261012-214638-adl_package-b5fa8717-fe03` (`steps/extraction.safetensors`, sha256
    `0663e2f6…` and `31877726…`). Record: `post_diagnostic.json` next to the run's output, sha256
    `f5ccdcc3…`.
- **Positions.** Position 0 agrees everywhere (at most 2·10⁻⁵, 0 against the stored δ̄);
  positions 1–4 carry the drift. The values for word_count_claim A and guillemets A and H equal
  those measured on two other Pods with the same driver and CUDA on 2026-10-27
  ([decisions](../decisions.md#gate-decisions)). Differences from cached and historical tensors
  are consistent with numerical-environment effects; their cause is not isolated by this
  experiment.
- **Stored base means across runs.** The three completed base-means runs of the cell
  (`…-5f0e`, `…-e866`, `…-6255`) hold the same file, sha256 `f8f307e5…`. The run's volume
  resolved the cell to `…-e866`, the local mirror to `…-5f0e`; both read the same tensor.

## Conclusion

For these six states, the two paths agree within 8.99·10⁻⁵ in this session. Differences from
cached and historical tensors are consistent with numerical-environment effects; their cause is
not isolated by this experiment. New packages use base means recomputed in their own session, and
§12.4.3's validation of method equivalence runs in the build's own session and environment
(§13.10).

M0 permits new builds under §13.10's per-session checks; it neither retroactively validates nor
invalidates earlier §12 packages or explains their reader outcomes.

## Run and costs

The run started at 20:21:18 UTC and finished at 21:26:54 UTC on 2026-10-29 (66 minutes). Spend:
GPU $5.82 (1.46 Pod hours at $3.99), API $0.
