#!/usr/bin/env bash
# The continuation's Stage-2 Pods for one pilot behavior's item (preregistration §12.7; plan C8),
# run detached on the Pod (run_detached.py) from the verified source snapshot. Two Pods share the
# item (the meter's split): Pod A writes only to /workspace/a, Pod B only to /workspace/c (Runpod:
# writing one network volume from two Pods at once can corrupt it). A Pod copies what it reads
# into its own space, create-only and checked (copy_inputs.py), from the other Pod's space and then
# from /workspace/b (Stage 1's second space, read only: the pilot's Y organism, its reference and
# data, the stage-1 recipe run, the Y₂ selection's laptop inputs), and never writes to either.
# Each session stops its Pod.
#
#   a1 (Pod A): Y₂'s criterion 3 (phase y2; its failure does not stop the session), the screening,
#      then the decision's steps as stage2_next.py names them (each confirmation, the expression
#      checks at the analysis dose) and the decision cell (phase decide). It ends there: the
#      decision is committed before the lineage's k* (§12.7, review s4z 1);
#   a1k (Pod A, after the Stage-2 decision is committed): k* at its analysis dose and L's full PD
#      pool redrawn, tagged stage2-sxm (phase pd_pool, H100 SXM only), PD-clean's input for L (a2
#      with PD_DRAW);
#   b1 (Pod B, needs no dose): Y alone for link at 100 and 200;
#   a2 (Pod A, after the k* and Stage-2 manifests are committed and a new snapshot made): the
#      certificate's blocks 0-2 and the readers at the analysis dose;
#   b2 (Pod B, after the k* and Y₂ manifests are committed): blocks 3-5.
#
# The second behavior's item (ITEM=stage2/<behavior>, given; pun_intended, whose pilot cells are in
# b), split to Pod B alone (pod-b = 1), one Pod at a time in c, while Stage 5 writes a:
#   p1: what its screening, decision and k* read, as stage2_inputs.py names them from the
#      behavior's config, copied from b then a, and the pinned corpora fetched into c's HF cache
#      (hf_prefetch.py); then a1's steps without y2 (Y₂ is selected), ending at the decision cell.
#      The Stage-2 decision is committed and a new snapshot made before p2;
#   p2: k* at the committed decision's analysis dose (H_s from p1's screening record, A from the
#      config). Its manifest is committed and a new snapshot made before p3;
#   p3: the recipe control at the dose copied with its adapters, the certificate's six blocks on
#      this one Pod (its plan check first), then the readers; a session that stops at its
#      deadline is resumed by the next.
#
# Exception a's continued-mode lineage (§12.7 Stage 5; C12, C13), the meter's unsplit item stage5
# (ITEM=stage5, BEHAVIOR=<its behavior>), one Pod session at a time (worker pod, at most 12 h),
# in space a, which holds A, H and the lineage's states; each passes in the stage5 batch:
#   s5a: s5_suppress, then the decision's steps as stage5_next.py names them (each confirmation,
#      X_cont, the expression checks at the analysis dose) and the decision cell (s5_decide). It
#      ends there: the decision is written as manifests/stage5_<behavior>.json and committed on
#      the laptop, and a new snapshot made, before k* (the commit barrier, review s4t2 3);
#   s5k: k* of the committed Stage-5 lineage <behavior>_cont_<dose>; its manifest is committed and
#      a new snapshot made before the certificate;
#   s5c: the certificate's six blocks, then H's relearning (s5_relearn_h), then the readers; a
#      session that stops at its deadline is resumed by the next (completed cells reused).
# Its descriptive DPO lineage (C15, review s5b), the meter's unsplit item stage5_dpo
# (ITEM=stage5_dpo), the same way in space c (C16-1), which no other Pod writes while the
# continued lineage's Pod writes a, its passes in the stage5_dpo batch. Each session first copies
# what the lineage reads (stage2_inputs.py --stage5-dpo with the snapshot's manifests: A and H, the
# pinned data, the Stage-5 slot's evidence, k*'s inputs, the pilot's index and the pools) from a,
# then b, create-only and checked, and fetches the pinned corpora into c's HF cache:
#   s5d: s5d_suppress, then its decision's steps (stage5_next.py --lineage stage5_dpo: each
#      confirmation, the expression checks at the analysis dose; no X_cont) and the decision cell
#      (s5d_decide). The decision is written as manifests/stage5_dpo_<behavior>.json and committed,
#      and a new snapshot made, before k*;
#   s5dk: k* of the committed DPO lineage <behavior>_dpo_<dose> (H_s at its dose); nothing after.
# The continued lineage's strong branch (§12.4.10, §12.7 Stage 5; C16-2), the meter's item
# strong/stage5, split by its frozen split: S5-A's sessions on worker s5-a (its $13.76 GPU
# sublimit, no API), S5-B's on s5-b, in space a, its passes in the strong_stage5 batch:
#   s5s (s5-a): strong_screen, the frozen candidates 100 → 200 → 400 to the first eligible. Its
#      record is written as manifests/strong_<behavior>_cont.json and committed on the laptop, and
#      a new snapshot made, before any validation cell (the commit barrier);
#   s5v (s5-a, after the committed screening with an eligible candidate; V1 and O1 in a's
#      data/strong_suites): strong_validate, its one attempt. Its record becomes
#      manifests/strong_validation_<behavior>_cont.json, committed before any follow-up; a passing
#      one drops strong/stage5_dpo fallback_gate_closed;
#   s5sk (s5-b, after the committed passing validation): the candidate's X_cont (s5_xcont with
#      strong), then its k* (strong_<behavior>_cont_<dose>); the k* manifest is committed and a
#      new snapshot made before the certificate;
#   s5sc (s5-b): the certificate's six blocks, H's relearning, then the readers, all with strong.
# Its DPO fallback (C16-3), the meter's item strong/stage5_dpo (admitted only once S5-A failed and
# the fallback's gate opened), worker pod, in space c, where the DPO lineage's states are; each
# session first copies the gate's evidence from a (stage2_inputs.py --strong-gate; review s5q 2):
#   s5ds: strong_screen of branch stage5_dpo (the DPO lineage's margin-passing checkpoints,
#      ascending); its manifest strong_<behavior>_dpo.json committed before validation;
#   s5dv: strong_validate (V2, O2); its manifest committed before any follow-up;
#   s5dx: s5d_controls (X = H + U_A and the stack-DPO recipe control on the reversed pairs to
#      the dose), then k* of strong_<behavior>_dpo_<dose>; its manifest committed;
#   s5dc: the certificate, H's relearning and the readers, with stage5_dpo and strong.
# Stage 3's stack lineage of a panel behavior (§12.7 Stage 3; C17), the meter's unsplit item
# stage3/<behavior> (BEHAVIOR=<it>), worker pod, in its one space S3_SPACE (a or c: the space the
# meter's stage3-start recorded once, review s5y), its passes in the stage3 batch:
#   s3a: its inputs copied in (stage2_inputs.py --stage3: A, H, the own data, the pilot's index
#      and the pools) and the corpora fetched, then s3_suppress (the suppression data, training,
#      passes, selection, b's relearning data, A's reference, the diagnostic's sources). The
#      sources manifest is committed on the laptop before labeling, and the diagnostic's result
#      before s3b;
#   s3b: the decision's steps (stage5_next.py --lineage stage3: each confirmation, only after the
#      committed diagnostic, X and the expression checks at the confirmed dose) and s3_decide (no
#      stack dose without a margin-passing checkpoint); its manifest stage3_<behavior>.json is
#      committed before k* and before the strong screening.
# Its strong branch (C17-C), worker pod, in the same space, the strong_stage3 batch; s3s on the
# meter's unsplit item strong/stage3/<behavior> (its preparation), s3v and s3x on
# strong/stage3/<behavior>/follow (its follow-up, §12.8's phases; C17-P):
#   s3s (after the committed stack decision): s3s_suppress (the continued line to 1,600, its passes
#      and selection), then strong_screen of branch stage3 over that selection's margin-passing
#      checkpoints (stage5_next.py --lineage stage3_cont names it); its manifest
#      strong_<behavior>_cont.json is committed before validation;
#   s3v (after the panel's committed contracts, the meter's record of them and the behavior's
#      active entry, and its committed diagnostic, observer and PD and ADL-corpus): strong_validate
#      of branch stage3 on V3/O3 (in the space's strong_suites, create-only), its
#      plan.strong_validate.contracts the meter's record as the receipt check gives it (C17-H2);
#   s3x (after the committed passing validation, under the same record of the contracts as s3v,
#      the ones the validation names): s3s_xcont at its candidate's dose, its
#      plan.s3s_xcont.contracts the meter's record (review s6n 1).
# Its certified lineage (§12.7 Stage 3: one certificate and readout per behavior), on the meter's
# item stage3/<behavior>/follow, in the same space, the stage3 batch: the lineage, its dose and
# states come from the committed panel contracts as the meter recorded them and the evidence
# their entry binds (continuation_stage3.py certified), never chosen here:
#   s3k: k* (plan.kstar.stage3, its contracts the meter's record); its manifest
#      kstar_<lineage>.json is committed on the laptop once the analysis is frozen, before s3c;
#   s3c: the certificate's six blocks (resumable), under that k* manifest; its records stay
#      unopened until the analysis is frozen.
#
# Limits (review s4b). RECEIPT is the meter's receipt of this Pod's session
# (continuation_meter.py session-open, with the Pod's ID and the deadline set at its creation,
# both from the coordinator, and the session's API money), METER_LOG the copy of the log it was
# issued from. POD_ID must be the identity the provider gives the Pod (RUNPOD_POD_ID), and the
# chain starts only if verify-receipt accepts the session on that Pod. Every bound then comes from
# the session as the log holds it, never from the caller: its deadline bounds every deadline the
# chain sets (the watcher's, armed before the chain, may not lie beyond it), the cost meter counts
# the session's hours from the deadline back, and the ADL ledger's allowance is the session's
# ceiling, recorded in the ledger at its first start (what the ledger had committed plus the
# session's API money) and reused by every rerun (budget.session_allowance). The snapshot's receipt is
# trusted only by the values set in the Pod's environment at its creation (SOURCE_SNAPSHOT_ID,
# the snapshot; SNAPSHOT_RECEIPT_SHA256, its receipt; review s3z2). Every pass goes through the
# runner (CUDA faults retried, exit 3 final) with RUN_LIMIT_S to the end of work: the watcher's
# deadline minus the resume margin and the export. The chain exits nonzero if a check refused, a
# pass did not complete or an export was not clean, and on every exit, a refusal before any pass
# included, it asks the watcher to stop the Pod within 3 minutes. The volume keeps everything,
# and a rerun of the session resumes its cells (on this Pod or, if it cannot start again, on a new
# Pod on the same volume).
# usage: SESSION=<a1|a1k|a2|b1|b2|p1|p2|p3|s5a|s5k|s5c|s5d|s5dk|s5s|s5v|s5sk|s5sc|s3a|s3b|s3s|s3v|s3x|s3k|s3c> \
#          [S3_SPACE=<a|c> for s3*] SNAP=<id> WN=<dir> \
#          PRICE=<USD/h> RECEIPT=<file> METER_LOG=<file> POD_ID=<this Pod's ID> \
#          SOURCE_SNAPSHOT_RECEIPT=<file> [ITEM=stage2/example_link] \
#          [PD_DRAW=stage2-sxm] [BEHAVIOR=example_link for s5a, s5k, s5c, s5d, s5dk] \
#          continuation_stage2.sh
#          (its directory: <attempts>/chain-<session>-<UTC time>)
set -uo pipefail
WS=${WORKSPACE:-/workspace}               # the network volume; another root only in tests
T=$WS/tools; AT=$WS/attempts/residual-disposition
CODE=${CODE_ROOT:-/root/code/residual-disposition}
: "${SESSION:?}" "${SNAP:?}" "${WN:?}" "${PRICE:?}" "${RECEIPT:?}"
: "${METER_LOG:?}" "${POD_ID:?}" "${SOURCE_SNAPSHOT_RECEIPT:?}"
case $SESSION in
  a1|a1k|a2) WORKER=pod-a; SPACE=$WS/a ;;
  b1|b2|p1|p2|p3) WORKER=pod-b; SPACE=$WS/c ;;
  s5a|s5k|s5c) WORKER=pod; SPACE=$WS/a ;;  # an unsplit item's one session (review s4t2 1)
  s5d|s5dk) WORKER=pod; SPACE=$WS/c ;;     # the DPO lineage's own space (C16-1)
  s5s|s5v) WORKER=s5-a; SPACE=$WS/a ;;     # S5-A on its frozen sublimit (C16-2)
  s5sk|s5sc) WORKER=s5-b; SPACE=$WS/a ;;   # S5-B, after the passing validation
  s5ds|s5dv|s5dx|s5dc) WORKER=pod; SPACE=$WS/c ;;  # the DPO fallback, in the DPO lineage's c (C16-3)
  s3a|s3b|s3s|s3v|s3x|s3k|s3c)  # Stage 3's lineage of a panel behavior (C17), in its one space, a or c: checked
    # once the watcher can stop the Pod, against the meter's record (review s5y 1, 2)
    WORKER=pod; SPACE=$WS/${S3_SPACE:-} ;;
  *) echo "SESSION must be a1, a1k, a2, b1, b2, p1, p2, p3, s5a, s5k, s5c, s5d, s5dk, s5s, s5v, s5sk, s5sc, s5ds, s5dv, s5dx, s5dc, s3a, s3b, s3s, s3v, s3x, s3k or s3c"; exit 1 ;;
esac
LINE=""
case $WORKER in pod|s5-a|s5-b)
  # the lineage's item, phase prefix and record key: the continued one's, the DPO one's (C15) or
  # the continued strong branch's (C16-2)
  case $SESSION in s5d|s5dk) LINE=stage5_dpo; P5=s5d ;; s5s|s5v|s5sk|s5sc) LINE=strong/stage5; P5=s5 ;;
    s5ds|s5dv|s5dx|s5dc) LINE=strong/stage5_dpo; P5=s5d ;;
    s3a|s3b) LINE=stage3/${BEHAVIOR:-}; P5=s3 ;;  # BEHAVIOR checked once the Pod can stop
    s3s) LINE=strong/stage3/${BEHAVIOR:-}; P5=s3 ;;  # its strong branch's preparation (C17-C)
    s3v|s3x) LINE=strong/stage3/${BEHAVIOR:-}/follow; P5=s3 ;;  # its follow-up (C17-P)
    s3k|s3c) LINE=stage3/${BEHAVIOR:-}/follow; P5=s3 ;;  # its certified lineage (C17-P)
    *) LINE=stage5; P5=s5 ;; esac
  ITEM=${ITEM:-$LINE}; BEHAVIOR=${BEHAVIOR:-} ;;
*)
  case $SESSION in p1|p2|p3) ITEM=${ITEM:-} ;; *) ITEM=${ITEM:-stage2/example_link} ;; esac
  BEHAVIOR=${ITEM#stage2/} ;;
esac
CHAIN=$SESSION-$(date -u +%Y%m%dT%H%M%S)  # unique per launch; reused cells stay reused
OUT=$AT/chain-$CHAIN
W=$WS/watch/$WN
IMG=${CONTAINER_IMAGE:?the image digest the Pod runs}
SRC=$CODE/$SNAP/source
MARGIN=25; EXPORT=30                      # minutes
CFG=experiments/continuation_14b
ATTEMPTS="$CHAIN-*"
HARD_S=""                                 # the session's deadline, once verified

log() { echo "=== $(date -u +%FT%TZ) $*"; }
field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1" "$2"; }
jget() {  # <json> <key ...>: the value at the keys (JSON for a list or a mapping, empty for null)
  python3 -c 'import functools, json, sys
v = functools.reduce(lambda a, k: a[k], sys.argv[2:], json.loads(sys.argv[1]))
print(json.dumps(v) if isinstance(v, (dict, list)) else ("" if v is None else v))' "$@"
}
now_s() { date -u +%s; }
iso() { date -u -d "@$1" +%FT%T+00:00; }

wait_attempt() {  # until the attempt's status is final; prints it (as in Stage 1's chain)
  local s pid seen
  seen=$(now_s)
  while true; do
    s=$(field "$1/status.json" status 2>/dev/null); pid=$(field "$1/status.json" supervisor_pid 2>/dev/null)
    case $s in
      running|queued)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then seen=$(now_s)
        elif [ -n "$pid" ] || [ "$s" = running ]; then echo supervisor_gone; return
        fi ;;
      completed|failed|launch_failed) echo "$s"; return ;;
    esac
    [ $(( $(now_s) - seen )) -ge 600 ] && { echo supervisor_gone; return; }
    sleep 60
  done
}

checks_clean() {  # <log> <path ...>: 0 if each path has a check with 0 differences and no error
  python3 - "$@" <<'PY'
import json, re, sys
found, errors = {}, {}
for line in open(sys.argv[1]):
    m = re.match(r"^(\S+): (.*)$", line.rstrip("\n"))
    if not m:
        continue
    path, rest = m.groups()
    if d := re.search(r"(\d+) differences found", rest):
        found[path] = max(found.get(path, 0), int(d.group(1)))
    if e := re.search(r"(\d+) errors while checking", rest):
        errors[path] = errors.get(path, 0) + int(e.group(1))
    elif re.search(r"\bERROR\b", rest):
        errors[path] = errors.get(path, 0) + 1
want = sys.argv[2:]
bad = {p: {"differences": found.get(p), "errors": errors.get(p, 0)} for p in want
       if found.get(p) != 0 or errors.get(p, 0)}
print("checks:", json.dumps({"required": len(want), "not_clean": bad}))
sys.exit(1 if bad else 0)
PY
}

deadline_s() { python3 -c 'import json,sys; print(int(float(json.load(open(sys.argv[1]))["deadline"])))' "$W/state.json"; }

set_deadline() {  # <epoch s>: the watcher's deadline, never past the session's
  local at=$1 tok
  [ -n "$HARD_S" ] && [ "$at" -gt "$HARD_S" ] && at=$HARD_S
  tok=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["token"])' "$W/state.json")
  python3 $T/night_watch.py deadline --state-dir $W --token "$tok" --at "$(iso $at)" --allow-deadline-stop >/dev/null 2>&1
  sleep 15
  python3 - "$W/state.json" "$at" <<'PY'
import json, sys
from datetime import UTC, datetime
s = json.load(open(sys.argv[1]))
ok = s.get("phase") == "armed" and not s.get("dry_run") and abs(float(s["deadline"]) - float(sys.argv[2])) < 1
print(f"watcher: phase {s.get('phase')}, revision {s.get('revision')}, deadline "
      f"{datetime.fromtimestamp(float(s['deadline']), UTC).isoformat()}")
sys.exit(0 if ok else 1)
PY
  local rc=$?
  pgrep -f "night_watch.py watch --state-dir $W" >/dev/null || { echo "watcher: no live worker"; rc=1; }
  return $rc
}

STOP_ASKED=0
stop_pod() {  # the watcher stops the Pod within 3 minutes (never later than its deadline)
  [ $STOP_ASKED = 1 ] && return 0
  STOP_ASKED=1
  local at=$(( $(now_s) + 180 )) d
  if ! d=$(deadline_s 2>/dev/null); then log "watcher: no state; the Pod stops at its deadline"; return 1; fi
  [ "$d" -lt "$at" ] && at=$d
  log "stopping the Pod through the watcher"
  set_deadline "$at" || { log "watcher: deadline not moved; it stops at its deadline"; return 1; }
}
on_exit() { local rc=$?; stop_pod; exit $rc; }

batch_of() {  # <phase>: its batch (budget.batch_of): the Stage-5 lineage's item (stage5,
  # stage5_dpo or strong/stage5 as strong_stage5), Stage 3's (stage3/<b> as stage3; C17), the Y
  # control's, or Stage 2's
  case $ITEM in stage3/*) echo stage3; return ;; strong/stage3/*) echo strong_stage3; return ;; esac
  [ -n "$LINE" ] && { echo ${ITEM//\//_}; return; }
  case $1 in control_y) echo y ;; *) echo stage2 ;; esac
}

run_pass() {  # <attempt id> <config> <phase> [override ...]: one detached pass in this space, waited
  local id=$1 config=$2 phase=$3; shift 3
  local limit=$(( $(deadline_s) - (MARGIN + EXPORT) * 60 - $(now_s) ))
  if [ "$limit" -le 600 ]; then log "$id not run: less than 10 min of work before the deadline"; return 1; fi
  # only the pass's own batch: another batch's key is not in every config (y2 has no batch y)
  local cmd=(env -u RUN_ID project-run $SRC env RUN_LIMIT_S=$limit bash $T/rest_runner.sh $SPACE
             $config pipeline.plan.phase=$phase "pipeline.plan.cost.deadline_state=$W/state.json"
             "pipeline.plan.cost.batches.$(batch_of $phase)=$SESSION_HOURS" "${COST[@]}" "$@")
  if ! python3 $T/run_detached.py --attempt-dir $AT/$id --cwd $SRC --run-id $id -- "${cmd[@]}"; then
    log "$id: launch error"; return 1
  fi
  local s; s=$(wait_attempt $AT/$id)
  log "$id: $s, exit code $(field $AT/$id/status.json exit_code)"
  [ "$s" = completed ] && [ "$(field $AT/$id/status.json exit_code)" = 0 ]
}

export_space() {  # <space> <ledger file> <log>: create-only export, then each path checked
  local expected=(data/states data/generations data/cache/adl_steps) p
  # Only the ADL cache is optional (c holds none: B1, 2026-10-20): it is exported empty. A missing
  # required tree is refused, never masked as an empty source (review s4x 3).
  mkdir -p "$1/data/cache/adl_steps"
  for p in data/states data/generations; do
    [ -d "$1/$p" ] || { echo "export: the required $p is not in $1" > "$3"; return 1; }
  done
  bash $T/bucket_final.sh $1 $2 $WN "$ATTEMPTS" > "$3" 2>&1; local rc=$?
  for p in $1/runs/*; do expected+=("runs/$(basename $p)"); done
  for p in $AT/$ATTEMPTS; do [ -e "$p" ] && expected+=("runs/_attempts/$(basename $p)"); done
  checks_clean "$3" "${expected[@]}" && [ $rc = 0 ]
}

upload_dir() {  # <local dir> <bucket path> <log>: one directory, create-only, checked
  bash $T/bucket_dir.sh "$1" "$2" > "$3" 2>&1; local rc=$?
  checks_clean "$3" "$2" && [ $rc = 0 ]
}

in_src() { (cd $SRC && project-run $SRC "$@"); }

s5_states() {  # <dose>: the lineage's selection's states there, from its committed decision's run
  local sel; sel=$(manifest ${LINE}_$BEHAVIOR.json 'd["inputs"]["selection"]')
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["doses"][sys.argv[2]]))' \
    "$SPACE/runs/continuation_14b/$sel/${LINE}_selection.json" "$1"
}

copy_in() {  # [copy_inputs.py args]: from the other Pod's space, then b, into this one; for the
  # second behavior's sessions from b, where its pilot cells are, then a
  local first=$WS/a second=$WS/b; [ $SPACE = $WS/a ] && first=$WS/c
  case $SESSION in p1|p2|p3) first=$WS/b; second=$WS/a ;; esac
  in_src python scripts/pod/copy_inputs.py --source $first --source $second --target $SPACE \
    --log $OUT/copied.jsonl "$@" >> $OUT/copy.json 2>&1
}

manifest() {  # <file under manifests/> <python expression on d>: a committed manifest's value
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$SRC/manifests/$1" "$2"
}

mkdir -p $OUT
trap on_exit EXIT  # from here on, any exit asks the watcher to stop the Pod
# The provider's environment, read as data and never sourced (reviews s5z, s6b, s6d): of its
# exact `[export ]NAME=VALUE` lines only the provider's own variables are taken, each by its name
# (this Pod's identity, the snapshot's identity set at its creation, the secrets the passes and
# the bucket need, the volume's documented cache locations, fixed flags); never a controller
# setting (the watcher's registry included), PATH or IFS. The file, every value and the whole are
# bounded, a value is one printable line, a name given twice with two values and a cache path off
# the volume's are refused; the values pass in memory. Anything else refuses, and the trap stops
# the Pod.
if ! PROVIDER=$(python3 - "${RP_ENV:-/etc/rp_environment}" "$WS" 2>$OUT/provider-env.err <<'PY'
import re, shlex, sys
path, ws = sys.argv[1], sys.argv[2]
FILE, VALUE, TOTAL = 64 * 1024, 8 * 1024, 32 * 1024  # bytes: the file, a value, the exports
ANY = re.compile(r"[\x21-\x7e]*")  # an identity or a secret: one printable word, or empty
CACHES = {"HF_HOME": [f"{ws}/cache/huggingface"], "XDG_CACHE_HOME": [f"{ws}/cache"],
          "UV_CACHE_DIR": [f"{ws}/.cache/uv", f"{ws}/.cache/uv/"],
          "PIP_CACHE_DIR": [f"{ws}/.cache/pip", f"{ws}/.cache/pip/"]}
FIXED = {"HF_HUB_ENABLE_HF_TRANSFER": ["0", "1"], "HF_XET_HIGH_PERFORMANCE": ["0", "1"],
         "PYTHONUNBUFFERED": ["0", "1"], "UV_LINK_MODE": ["copy", "hardlink", "symlink", "clone"],
         "UV_PYTHON_DOWNLOADS": ["never", "manual", "automatic"]}
WORDS = ("RUNPOD_POD_ID", "SOURCE_SNAPSHOT_ID", "SNAPSHOT_RECEIPT_SHA256", "HF_TOKEN",
         "ARTIFACT_GCS_KEY_B64", "GCS_ARCHIVE_KEY_B64", "WANDB_API_KEY", "JUDGE_API_BASE",
         "JUDGE_API_KEY")
blob = open(path, "rb").read(FILE + 1)
if len(blob) > FILE:
    sys.exit(f"the file exceeds {FILE} bytes")
taken = {}
for n, line in enumerate(blob.decode("utf-8").splitlines(), 1):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    m = re.fullmatch(r"(?:export )?([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
    if m is None:
        sys.exit(f"line {n}: not an assignment")
    try:
        words = shlex.split(m[2]) if m[2] else [""]
    except ValueError:
        sys.exit(f"line {n}: {m[1]}'s value is not one quoted word")
    if len(words) != 1:
        sys.exit(f"line {n}: {m[1]}'s value is not one word")
    name, value = m[1], words[0]
    if name not in (*WORDS, *CACHES, *FIXED, "TZ"):
        continue
    if len(value.encode()) > VALUE:
        sys.exit(f"line {n}: {name}'s value exceeds {VALUE} bytes")
    if name in CACHES and value not in CACHES[name]:
        sys.exit(f"line {n}: {name} is not the volume's cache location")
    if name in FIXED and value not in FIXED[name]:
        sys.exit(f"line {n}: {name} is none of {FIXED[name]}")
    if name == "TZ" and not re.fullmatch(r"[A-Za-z0-9_+./-]{1,64}", value):
        sys.exit(f"line {n}: TZ is not a time zone's name")
    if name in WORDS and not ANY.fullmatch(value):
        sys.exit(f"line {n}: {name}'s value is not one printable word")
    if taken.get(name, value) != value:
        sys.exit(f"line {n}: {name} is given twice, with two values")
    taken[name] = value
out = "".join(f"{k}={v}\n" for k, v in taken.items())
if len(out.encode()) > TOTAL:
    sys.exit(f"the provider's variables exceed {TOTAL} bytes")
sys.stdout.write(out)
PY
); then
  log "not run: the provider's environment is not readable as data: $(tail -c 300 $OUT/provider-env.err)"; exit 1
fi
while [ -n "$PROVIDER" ]; do  # line by line, in memory: no temporary file, no other process
  P_LINE=${PROVIDER%%$'\n'*}; PROVIDER=${PROVIDER#"$P_LINE"}; PROVIDER=${PROVIDER#$'\n'}
  P_NAME=${P_LINE%%=*}; P_VALUE=${P_LINE#*=}
  case $P_NAME in
    RUNPOD_POD_ID|SOURCE_SNAPSHOT_ID|SNAPSHOT_RECEIPT_SHA256|HF_TOKEN|ARTIFACT_GCS_KEY_B64|\
    GCS_ARCHIVE_KEY_B64|WANDB_API_KEY|JUDGE_API_BASE|JUDGE_API_KEY|HF_HOME|XDG_CACHE_HOME|\
    UV_CACHE_DIR|PIP_CACHE_DIR|HF_HUB_ENABLE_HF_TRANSFER|HF_XET_HIGH_PERFORMANCE|PYTHONUNBUFFERED|\
    UV_LINK_MODE|UV_PYTHON_DOWNLOADS|TZ) export "$P_NAME=$P_VALUE" ;;
    *) log "not run: the provider's environment names '$P_NAME'"; exit 1 ;;
  esac
done
unset PROVIDER P_LINE P_NAME P_VALUE
export SOURCE_SNAPSHOT_RECEIPT
export CONTAINER_IMAGE=$IMG PYTHONDONTWRITEBYTECODE=1
log "chain $CHAIN ($ITEM, $WORKER in $SPACE), \$$PRICE/h"
if [ -n "$LINE" ] && [ -z "$BEHAVIOR" ]; then  # any refusal from here on stops the Pod
  log "not run: a $SESSION session needs BEHAVIOR, its lineage's behavior"; exit 1
fi
case $SESSION in s3a|s3b|s3s|s3v|s3x|s3k|s3c)  # the lineage's one space (C17)
  case ${S3_SPACE:-} in a|c) ;; *) log "not run: S3_SPACE must be a or c, the lineage's one space, not '${S3_SPACE:-}'"; exit 1 ;; esac ;;
esac
if [ -n "$LINE" ] && [ "$ITEM" != "$LINE" ]; then  # never another item's reservation (s4x 1)
  log "not run: a $SESSION session runs on the meter's item $LINE, not $ITEM"; exit 1
fi
case "$SESSION:$ITEM" in  # the second behavior's sessions name their item: never a default
  p[123]:stage2/?*) ;;
  p[123]:*) log "not run: $SESSION needs ITEM=stage2/<behavior>, not '$ITEM'"; exit 1 ;;
esac
if [ -z "${RUNPOD_POD_ID:-}" ] || [ "$RUNPOD_POD_ID" != "$POD_ID" ]; then
  log "not run: POD_ID $POD_ID is not this Pod's identity from the provider (RUNPOD_POD_ID ${RUNPOD_POD_ID:-unset})"; exit 1
fi
if ! python3 $T/source_snapshot.py verify $CODE/$SNAP --expected-id $SNAP >/dev/null; then
  log "not run: snapshot $SNAP not verified"; exit 1
fi
if [ "${SOURCE_SNAPSHOT_ID:-}" != "$SNAP" ] || [ -z "${SNAPSHOT_RECEIPT_SHA256:-}" ] \
   || [ "$(sha256sum "$SOURCE_SNAPSHOT_RECEIPT" | cut -d' ' -f1)" != "$SNAPSHOT_RECEIPT_SHA256" ]; then
  log "not run: the snapshot or its receipt is not the one set at the Pod's creation"; exit 1
fi
HOURS=$(field "$RECEIPT" hours 2>/dev/null)  # verify-receipt makes them the log's session's
if ! in_src python scripts/continuation_meter.py --log "$METER_LOG" verify-receipt "$RECEIPT" \
     --item "$ITEM" --worker "$WORKER" --pod-id "$POD_ID" --price "$PRICE" --hours "${HOURS:-0}" \
     > $OUT/receipt-check.json 2>$OUT/receipt-check.err; then
  log "not run: the meter's session receipt does not authorize this Pod $POD_ID"; cat $OUT/receipt-check.json; exit 1
fi
cp "$RECEIPT" $OUT/receipt.json
case $SESSION in s3a|s3b|s3s|s3v|s3x|s3k|s3c)  # the behavior's Stage-3 start as the meter recorded it, once (review s5y 2, 3)
  S3_RECORDED=$(jget "$(tail -n 1 $OUT/receipt-check.json)" stage3 space 2>/dev/null)
  if [ "$S3_RECORDED" != "$S3_SPACE" ]; then
    log "not run: the meter's Stage-3 start of $BEHAVIOR names space '${S3_RECORDED}', not S3_SPACE=$S3_SPACE"; exit 1
  fi
  # what the passes will use: this item's behavior and the recorded space's path (review s5z)
  case $SESSION in s3s) OWN=strong/stage3/$BEHAVIOR ;; s3v|s3x) OWN=strong/stage3/$BEHAVIOR/follow ;;
    s3k|s3c) OWN=stage3/$BEHAVIOR/follow ;; *) OWN=stage3/$BEHAVIOR ;; esac
  if [ "$SPACE" != "$WS/$S3_RECORDED" ] || [ "$ITEM" != "$OWN" ]; then
    log "not run: the passes would run $BEHAVIOR in $SPACE, not the item $ITEM in $WS/$S3_RECORDED"; exit 1
  fi ;;
esac
# The session as the log holds it: its deadline, hours and API money bound everything below.
SESSION_JSON=$(jget "$(tail -n 1 $OUT/receipt-check.json)" session)
HARD_S=$(python3 -c 'import json,sys; from datetime import datetime
print(int(datetime.fromisoformat(json.loads(sys.argv[1])["deadline"]).timestamp()))' "$SESSION_JSON") \
  || { log "not run: no verified session deadline"; exit 1; }
SESSION_HOURS=$(jget "$SESSION_JSON" hours); SESSION_API=$(jget "$SESSION_JSON" api_usd)
FROM=$(iso $(python3 -c 'import sys; print(int(int(sys.argv[1]) - float(sys.argv[2]) * 3600))' "$HARD_S" "$SESSION_HOURS"))
log "session: deadline $(iso $HARD_S), $SESSION_HOURS h from $FROM, API \$$SESSION_API"
if [ -n "$LINE" ] && ! python3 -c 'import sys; sys.exit(float(sys.argv[1]) > 12)' "$SESSION_HOURS"; then
  log "not run: a Stage-5 session is at most 12 h, not $SESSION_HOURS (review s4t2 4)"; exit 1
fi
if [ "$(deadline_s)" -gt "$HARD_S" ]; then
  log "not run: the watcher's deadline $(iso $(deadline_s)) is past the session's $(iso $HARD_S)"; exit 1
fi
# The session's API ceiling, recorded in the worker's ledger at its first start and the same at
# every rerun of this session (review s4d), before any pass can send.
SESSION_ID=$(jget "$(tail -n 1 $OUT/receipt-check.json)" session_id)
LEDGER=$SPACE/data/ledgers/adl-api-$WORKER.jsonl  # ledger_from's default path in the space
LEDGER_USD=$(in_src python -c 'import sys; from residual_disposition.budget import session_allowance
print(session_allowance(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])))' \
  "$LEDGER" "$WORKER" "$SESSION_ID" "$SESSION_API") || { log "not run: the ADL ledger's ceiling"; exit 1; }
log "ADL API ceiling \$$LEDGER_USD for session ${SESSION_ID:0:12}"
COST=(pipeline.plan.cost.stop_fraction=0.9 pipeline.plan.cost.prior_hours=0.0
      pipeline.plan.cost.started_at=$FROM
      pipeline.plan.api.adl_ledger.worker=$WORKER pipeline.plan.api.adl_ledger.allowance_usd=$LEDGER_USD
      pipeline.plan.api.adl_ledger.session=$SESSION_ID)
export SOURCE_SNAPSHOT_ID SNAPSHOT_RECEIPT_SHA256

OK=1; STEPS=()
step() { STEPS+=("$1=$2"); [ "$2" = 0 ] || OK=0; }

stage2_steps() {  # the screening, the decision's steps as stage2_next.py names them and the
  # decision cell; k* follows the decision's commit (§12.7, review s4z 1)
  run_pass $CHAIN-screen $CFG/$BEHAVIOR screen; step screen $?
  local PREV="" i
  for i in 1 2 3 4 5 6 7 8; do
    [ $OK = 1 ] || break
    NEXT=$(in_src python scripts/pod/stage2_next.py --runs $SPACE/runs --behavior $BEHAVIOR 2>>$OUT/next.err)
    echo "$NEXT" >> $OUT/next.jsonl
    KIND=$(jget "$NEXT" step 2>/dev/null); DOSE=$(jget "$NEXT" dose 2>/dev/null)
    LATENT=$(jget "$NEXT" latent 2>/dev/null); SHAM=$(jget "$NEXT" sham_s 2>/dev/null)
    if [ "$KIND$DOSE" = "$PREV" ]; then log "no progress at $KIND $DOSE: its pass left no cell"; step progress 1; break; fi
    PREV=$KIND$DOSE
    case $KIND in
      confirm)
        run_pass $CHAIN-confirm-$DOSE $CFG/$BEHAVIOR confirm pipeline.plan.confirm.dose=$DOSE \
          pipeline.plan.confirm.latent=$LATENT pipeline.plan.confirm.sham_s=$SHAM
        step confirm-$DOSE $? ;;
      expression)
        run_pass $CHAIN-ood-$DOSE $CFG/$BEHAVIOR ood "pipeline.plan.ood.states=[$LATENT]"; step ood-$DOSE $?
        run_pass $CHAIN-corpus-$DOSE $CFG/$BEHAVIOR corpus pipeline.plan.corpus.state=$LATENT
        step corpus-$DOSE $? ;;
      decide)
        INPUTS=$(jget "$NEXT" inputs)
        mapfile -t EXPR < <(python3 -c 'import json,sys
for k, v in json.loads(sys.argv[1]).items(): print(f"+pipeline.plan.decide.expression.{k}={v}")' "$(jget "$INPUTS" expression)")
        run_pass $CHAIN-decide $CFG/$BEHAVIOR decide \
          pipeline.plan.decide.screening=$(jget "$INPUTS" screening) \
          "pipeline.plan.decide.confirmation=$(jget "$INPUTS" confirmation | tr -d ' "')" "${EXPR[@]}"
        step decide $?
        ANALYSIS=$(jget "$NEXT" decision analysis_dose)
        if [ -n "$ANALYSIS" ]; then
          log "analysis dose $ANALYSIS: the decision is committed on the laptop before k* (§12.7)"
        else
          log "no analysis dose: exception a (the Stage-5 slot), no k*"
        fi
        break ;;
      retest) log "the decision needs a confirmation retest after a detector revision: stop and report"; step retest 1; break ;;
      *) log "stage2_next: $NEXT"; step next 1; break ;;
    esac
  done
}

certificate_steps() {  # <blocks> <readers 0|1> <space of the screening record> [copy_inputs args]:
  # the dose and states from the committed manifests, L from the screening record; first what the
  # certificate reads, as the manifests name it (review s4b), then its plan check alone (CPU) in
  # this space, and only then the GPU pass; with readers, the readers at the dose
  local blocks=$1 readers=$2 screens=$3; shift 3
  K=kstar_${BEHAVIOR}_$(manifest stage2_$BEHAVIOR.json 'd["decision"]["analysis_dose"]').json
  DOSE=$(manifest $K 'd["dose"]'); SHAM=$(manifest $K 'd["states"]["H_s"]'); A=$(manifest $K 'd["states"]["A"]')
  SCREEN=$(manifest stage2_$BEHAVIOR.json 'd["inputs"]["screening"]')
  LATENT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["doses"][sys.argv[2]]["latent"])' \
    "$screens/runs/continuation_14b/$SCREEN/stage2_screening.json" "$DOSE")
  copy_in --certificate ${BEHAVIOR}_$DOSE --manifests $SRC/manifests --state $LATENT "$@"
  step copy $?
  CERT=(pipeline.plan.certificate.dose=$DOSE pipeline.plan.certificate.latent=$LATENT
        pipeline.plan.certificate.sham_s=$SHAM pipeline.plan.certificate.active=$A
        "pipeline.plan.certificate.blocks=$blocks")
  if [ $OK = 1 ]; then
    run_pass $CHAIN-cert-check $CFG/$BEHAVIOR certificate "${CERT[@]}" pipeline.plan.check_only=true
    step certificate-check $?
  fi
  if [ $OK = 1 ]; then
    run_pass $CHAIN-cert $CFG/$BEHAVIOR certificate "${CERT[@]}"
    step certificate $?
  fi
  if [ $readers = 1 ] && [ $OK = 1 ]; then
    run_pass $CHAIN-readers $CFG/$BEHAVIOR readers pipeline.plan.readers.dose=$DOSE \
      ${PD_DRAW:+pipeline.plan.readers.pd_draw=$PD_DRAW}
    step readers $?
  fi
}

case $SESSION in
a1)
  run_pass $CHAIN-y2 $CFG/y2_zero_indexed_lists y2; Y2_RC=$?  # recorded at the end, blocks nothing
  stage2_steps
  step y2 $Y2_RC ;;
a1k)
  # Option (b) (Codex s4k): k* whole on H100 SXM at the decision's analysis dose, then L's full PD
  # pool redrawn there, tagged, PD-clean's input; the screening's count stays the screening's.
  NEXT=$(in_src python scripts/pod/stage2_next.py --runs $SPACE/runs --behavior $BEHAVIOR 2>>$OUT/next.err)
  echo "$NEXT" >> $OUT/next.jsonl
  ANALYSIS=$(jget "$NEXT" decision analysis_dose 2>/dev/null)
  LATENT=$(jget "$NEXT" latent 2>/dev/null); SHAM=$(jget "$NEXT" sham_s 2>/dev/null)
  if [ "$(jget "$NEXT" status 2>/dev/null)" != decided ] || [ -z "$ANALYSIS" ]; then
    log "a1k: no decided analysis dose: $NEXT"; step next 1
  else
    run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar pipeline.plan.kstar.dose=$ANALYSIS \
      pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=${ACTIVE:?A, the lineage\'s active organism}
    step kstar $?
    run_pass $CHAIN-pdsxm $CFG/$BEHAVIOR pd_pool pipeline.plan.pd_pool.state=$LATENT \
      pipeline.plan.pd_pool.draw=stage2-sxm
    step pd_pool $?
  fi ;;
b1)
  # Y alone needs link's L and H_s at 100 and 200 (a), the pilot's Y organism, its approved
  # reference and its relearning data (b), the analyst index and the prompt pools (a).
  copy_in --state s4303557a4e7a9f64 --state sf8d4c3b1ca796eec --state sab16754d0858729b \
    --state se7a636b801f0ab1d --state sbf05f589f929edf8 --index feasibility_14b --index stage1_14b \
    --run feasibility_14b/20261014-072219-rates-2931c028-5097 \
    --run feasibility_14b/20261014-072309-data_relearn-8950fcdc-6f7f \
    --generation d5f5179f8853cb6cc --prompts
  step copy $?
  if [ $OK = 1 ]; then
    run_pass $CHAIN-y100 $CFG/control_y_link_100 control_y; step y_alone_100 $?
    run_pass $CHAIN-y200 $CFG/control_y_link_200 control_y; step y_alone_200 $?
  fi ;;
a2)
  # Pod A takes the Y organism and data B1 copied into c, the stage-1 recipe run and the Y₂
  # selection's inputs from b; L from the screening record in space a
  certificate_steps '[0,1,2]' 1 $WS/a ;;
b2)
  # Pod B takes the screening's states, A1's k* and Y₂ evidence and b's data from a (read, never
  # written), the stage-1 recipe run and the Y₂ selection's inputs from b, the analyst index and
  # the prompt pools, which a holds
  certificate_steps '[3,4,5]' 0 $WS/a --index feasibility_14b --index stage1_14b \
    --index continuation_14b --prompts ;;
p1)
  # The second behavior's pilot cells are in b: what its screening, decision and k* read, as
  # stage2_inputs.py names them from the behavior's config, copied from b then a, read only
  if INPUTS=$(in_src python scripts/pod/stage2_inputs.py --behavior $BEHAVIOR --source $WS/b \
      --source $WS/a 2>>$OUT/inputs.err); then
    echo "$INPUTS" > $OUT/inputs.json
    mapfile -t ARGS < <(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["args"]))' "$INPUTS")
    copy_in "${ARGS[@]}"; step copy $?
  else
    log "p1: the inputs of $BEHAVIOR: $INPUTS"; step inputs 1
  fi
  if [ $OK = 1 ]; then  # the pinned corpora in c's HF cache before any pass (review s4z)
    in_src env HF_HOME=$SPACE/hf python scripts/pod/hf_prefetch.py --behavior $BEHAVIOR \
      > $OUT/hf.json 2>>$OUT/hf.err
    step hf $?
  fi
  [ $OK = 1 ] && stage2_steps ;;
p2)
  # k* after the committed Stage-2 decision (§12.7, review s4z 1): its analysis dose, H_s there
  # from p1's screening record in c, A from the behavior's config
  if [ ! -f $SRC/manifests/stage2_$BEHAVIOR.json ]; then
    log "not run: no committed Stage-2 decision of $BEHAVIOR in the snapshot"; step manifest 1
  elif [ "$(manifest stage2_$BEHAVIOR.json 'd["decision"]["analysis_dose"]')" = None ]; then
    log "the Stage-2 decision has no analysis dose: exception a, no k*"; step manifest 1
  else
    DOSE=$(manifest stage2_$BEHAVIOR.json 'd["decision"]["analysis_dose"]')
    SCREEN=$(manifest stage2_$BEHAVIOR.json 'd["inputs"]["screening"]')
    SHAM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["doses"][sys.argv[2]]["sham_s"])' \
      "$SPACE/runs/continuation_14b/$SCREEN/stage2_screening.json" "$DOSE")
    ACTIVE=$(jget "$(in_src python scripts/pod/stage2_inputs.py --behavior $BEHAVIOR --source $WS/b \
      --source $WS/a --active)" active)
    run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar pipeline.plan.kstar.dose=$DOSE \
      pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=$ACTIVE
    step kstar $?
  fi ;;
p3)
  # after the committed k* manifest: the recipe control at the dose with its adapters (review s4z
  # 3), the six blocks on this one Pod, then the readers; L from p1's screening record in c
  DOSE=$(manifest stage2_$BEHAVIOR.json 'd["decision"]["analysis_dose"]' 2>/dev/null)
  if [ ! -f "$SRC/manifests/kstar_${BEHAVIOR}_$DOSE.json" ]; then
    log "not run: no committed k* manifest kstar_${BEHAVIOR}_$DOSE.json in the snapshot"; step manifest 1
  elif RECIPE=$(jget "$(in_src python scripts/pod/stage2_inputs.py --behavior $BEHAVIOR --source $WS/b \
      --source $WS/a --recipe-control $DOSE 2>>$OUT/inputs.err)" recipe 2>/dev/null) && [ -n "$RECIPE" ]; then
    certificate_steps '[0,1,2,3,4,5]' 1 $SPACE --state $RECIPE
  else
    log "p3: no one recipe control of $BEHAVIOR at $DOSE"; step recipe 1
  fi ;;
esac

s5_decide_args() {  # <stage5_next.py JSON>: the <P5>_decide overrides of its inputs
  python3 -c 'import json,sys
i = json.loads(sys.argv[1])["inputs"]; p = f"pipeline.plan.{sys.argv[2]}_decide."
for k in ("selection", "infeasibility", "xcont", "reference"):
    if i.get(k): print(f"{p}{k}={i[k]}")
print(p + "confirmation=[" + ",".join(i.get("confirmation") or []) + "]")
for k, v in (i.get("expression") or {}).items(): print(f"+{p}expression.{k}={v}")' "$1" $P5
}

case $SESSION in s5d|s5dk)
  # The DPO lineage's inputs from a, then b, read only, into c, and the pinned corpora in c's HF
  # cache, before any pass (C16-1)
  if INPUTS=$(in_src python scripts/pod/stage2_inputs.py --behavior $BEHAVIOR --source $WS/a \
      --source $WS/b --stage5-dpo --manifests $SRC/manifests 2>>$OUT/inputs.err); then
    echo "$INPUTS" > $OUT/inputs.json
    mapfile -t ARGS < <(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["args"]))' "$INPUTS")
    copy_in "${ARGS[@]}"; step copy $?
  else
    log "$SESSION: the inputs of $BEHAVIOR's DPO lineage: $INPUTS"; step inputs 1
  fi
  if [ $OK = 1 ]; then
    in_src env HF_HOME=$SPACE/hf python scripts/pod/hf_prefetch.py --behavior $BEHAVIOR \
      > $OUT/hf.json 2>>$OUT/hf.err
    step hf $?
  fi ;;
esac

case $SESSION in s3a)
  # Stage 3's inputs (C17): A, H, the own data, the pilot's index and the pools, from the other
  # space, then b, read only, into this one, and the pinned corpora in its HF cache, before any pass
  if INPUTS=$(in_src python scripts/pod/stage2_inputs.py --behavior $BEHAVIOR --source $WS/a \
      --source $WS/b --stage3 2>>$OUT/inputs.err); then
    echo "$INPUTS" > $OUT/inputs.json
    mapfile -t ARGS < <(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["args"]))' "$INPUTS")
    copy_in "${ARGS[@]}"; step copy $?
  else
    log "$SESSION: the inputs of $BEHAVIOR's Stage-3 lineage: $INPUTS"; step inputs 1
  fi
  if [ $OK = 1 ]; then
    in_src env HF_HOME=$SPACE/hf python scripts/pod/hf_prefetch.py --behavior $BEHAVIOR \
      > $OUT/hf.json 2>>$OUT/hf.err
    step hf $?
  fi ;;
esac

case $SESSION in s5ds|s5dv|s5dx|s5dc)
  # The DPO fallback's gate is recomputed here from the continued branch's committed manifests:
  # their evidence (the screening's candidate runs; the validation's identity, parts and V1/O1
  # suites), made in a, read only, into c, create-only and checked, before any pass (s5q 2)
  if GATE=$(in_src python scripts/pod/stage2_inputs.py --behavior $BEHAVIOR --source $WS/a \
      --strong-gate --manifests $SRC/manifests 2>>$OUT/inputs.err); then
    echo "$GATE" > $OUT/gate-inputs.json
    mapfile -t ARGS < <(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["args"]))' "$GATE")
    copy_in "${ARGS[@]}"; step copy $?
  else
    log "$SESSION: the DPO fallback's gate evidence: $GATE"; step inputs 1
  fi ;;
esac

case $SESSION in
s5a|s5d)
  # the continued lineage's phases s5_*, the DPO lineage's s5d_* (no X_cont step)
  if [ $OK = 1 ]; then
    run_pass $CHAIN-${P5}suppress $CFG/$BEHAVIOR ${P5}_suppress; step ${P5}_suppress $?
  else
    log "$SESSION: no pass without its inputs"
  fi
  PREV=""
  for i in 1 2 3 4 5 6 7 8; do
    [ $OK = 1 ] || break
    NEXT=$(in_src python scripts/pod/stage5_next.py --runs $SPACE/runs --behavior $BEHAVIOR \
             --lineage $LINE 2>>$OUT/next.err)
    echo "$NEXT" >> $OUT/next.jsonl
    KIND=$(jget "$NEXT" step 2>/dev/null); DOSE=$(jget "$NEXT" dose 2>/dev/null)
    if [ "$KIND$DOSE" = "$PREV" ]; then log "no progress at $KIND $DOSE: its pass left no cell"; step progress 1; break; fi
    PREV=$KIND$DOSE
    case $KIND in
      confirm)
        run_pass $CHAIN-${P5}confirm-$DOSE $CFG/$BEHAVIOR ${P5}_confirm \
          pipeline.plan.${P5}_confirm.dose=$DOSE
        step ${P5}_confirm-$DOSE $? ;;
      xcont)
        if [ $LINE != stage5 ]; then log "stage5_next: X_cont for $LINE, which has none: $NEXT"; step next 1; break; fi
        run_pass $CHAIN-s5xcont-$DOSE $CFG/$BEHAVIOR s5_xcont pipeline.plan.s5_xcont.dose=$DOSE
        step s5_xcont-$DOSE $? ;;
      expression)
        run_pass $CHAIN-${P5}expression-$DOSE $CFG/$BEHAVIOR ${P5}_expression \
          pipeline.plan.${P5}_expression.dose=$DOSE
        step ${P5}_expression-$DOSE $? ;;
      decide)
        mapfile -t ARGS < <(s5_decide_args "$NEXT")
        run_pass $CHAIN-${P5}decide $CFG/$BEHAVIOR ${P5}_decide "${ARGS[@]}"; step ${P5}_decide $?
        log "the $LINE decision $(jget "$NEXT" decision outcome): its manifest is committed on the laptop before k* (${SESSION}k)"
        break ;;
      retest) log "the decision needs a confirmation retest after a detector revision: stop and report"; step retest 1; break ;;
      *) log "stage5_next: $NEXT"; step next 1; break ;;
    esac
  done ;;
s3a)
  # Stage 3's stack lineage (C17): its suppression data, training, exploratory passes, selection,
  # b's relearning data, A's reference and the diagnostic's sources; it ends there
  if [ $OK = 1 ]; then
    run_pass $CHAIN-s3suppress $CFG/$BEHAVIOR s3_suppress; step s3_suppress $?
    log "the diagnostic's sources: written as manifests/stage3_diagnostic_sources_$BEHAVIOR.json and committed on the laptop before labeling; s3b after the committed diagnostic"
  else
    log "$SESSION: no pass without its inputs"
  fi ;;
s3b)
  # the decision's steps as stage5_next.py names them (each confirmation, X and the expression
  # checks at the confirmed dose), then the decision. A confirmation only after the committed
  # diagnostic (the barrier, E4; s3_confirm's preflight checks it whole); with no margin-passing
  # checkpoint the decision records no stack dose, and the diagnostic waits for s3s's screening
  DIAG=stage3_diagnostic_$BEHAVIOR.json
  PREV=""
  for i in 1 2 3 4 5 6 7 8; do
    [ $OK = 1 ] || break
    NEXT=$(in_src python scripts/pod/stage5_next.py --runs $SPACE/runs --behavior $BEHAVIOR \
             --lineage stage3 2>>$OUT/next.err)
    echo "$NEXT" >> $OUT/next.jsonl
    KIND=$(jget "$NEXT" step 2>/dev/null); DOSE=$(jget "$NEXT" dose 2>/dev/null)
    if [ "$KIND$DOSE" = "$PREV" ]; then log "no progress at $KIND $DOSE: its pass left no cell"; step progress 1; break; fi
    PREV=$KIND$DOSE
    case $KIND in
      confirm)
        if [ ! -f $SRC/manifests/$DIAG ]; then
          log "no confirmation: no committed detector diagnostic $DIAG in the snapshot"; step manifest 1; break
        fi
        run_pass $CHAIN-s3confirm-$DOSE $CFG/$BEHAVIOR s3_confirm pipeline.plan.s3_confirm.dose=$DOSE
        step s3_confirm-$DOSE $? ;;
      retest)
        # §12.3.1: after an accepted revision the voided confirmation is rerun on fresh draws,
        # attempt 2 at its dose; only with the revision to the current detector committed, and
        # s3_confirm's plan allocates nothing but that retest (continuation_confirm.allocate)
        VERSION=$(in_src python -c 'import sys
from residual_disposition.continuation_stage2 import current_detector_version
print(current_detector_version(sys.argv[1]))' "$BEHAVIOR") || VERSION=""
        REVISION=detector_revision_${BEHAVIOR}_$VERSION.json
        if [ -z "$VERSION" ] || [ ! -f $SRC/manifests/$REVISION ]; then
          log "no retest: no committed detector revision $REVISION in the snapshot"; step manifest 1; break
        fi
        if [ ! -f $SRC/manifests/$DIAG ]; then
          log "no retest: no committed detector diagnostic $DIAG in the snapshot"; step manifest 1; break
        fi
        log "the retest at $DOSE after the accepted revision to $VERSION: attempt 2 on fresh draws"
        run_pass $CHAIN-s3retest-$DOSE $CFG/$BEHAVIOR s3_confirm pipeline.plan.s3_confirm.dose=$DOSE
        step s3_retest-$DOSE $? ;;
      expression)
        run_pass $CHAIN-s3expression-$DOSE $CFG/$BEHAVIOR s3_expression \
          pipeline.plan.s3_expression.dose=$DOSE
        step s3_expression-$DOSE $? ;;
      decide)
        mapfile -t ARGS < <(s5_decide_args "$NEXT")
        run_pass $CHAIN-s3decide $CFG/$BEHAVIOR s3_decide "${ARGS[@]}"; step s3_decide $?
        log "the stage3/$BEHAVIOR decision $(jget "$NEXT" decision outcome): its manifest is committed on the laptop before k* (after a retest, with --supersede)"
        break ;;
      *) log "stage5_next: $NEXT"; step next 1; break ;;
    esac
  done ;;
s3s)
  # the strong branch (C17-C): the continued line, then the panel branch's screening of its
  # selection's margin-passing checkpoints, only after the committed stack decision (its fallback)
  DECIDED=stage3_$BEHAVIOR.json
  if [ ! -f $SRC/manifests/$DECIDED ]; then
    log "not run: no committed stack decision $DECIDED in the snapshot"; step manifest 1
  else
    run_pass $CHAIN-s3ssuppress $CFG/$BEHAVIOR s3s_suppress; step s3s_suppress $?
  fi
  if [ $OK = 1 ]; then
    NEXT=$(in_src python scripts/pod/stage5_next.py --runs $SPACE/runs --behavior $BEHAVIOR \
             --lineage stage3_cont 2>>$OUT/next.err)
    echo "$NEXT" >> $OUT/next.jsonl
    case $(jget "$NEXT" step 2>/dev/null) in
      screen)
        run_pass $CHAIN-strong-screen $CFG/$BEHAVIOR strong_screen \
          pipeline.plan.strong_screen.branch=stage3 \
          +pipeline.plan.strong_screen.selection=$(jget "$NEXT" inputs selection)
        step strong_screen $?
        log "the panel's strong screening: its manifest strong_${BEHAVIOR}_cont.json is committed on the laptop before validation (s3v)" ;;
      infeasible) log "the continued line's training is infeasible: no strong candidate; stop and report"; step infeasible 1 ;;
      *) log "stage5_next: $NEXT"; step next 1 ;;
    esac
  fi ;;
s3v)
  # the panel branch's one validation, under the panel's contracts as the meter recorded them (the
  # receipt check's panel_contracts; reviews s6f, s6h 3), the behavior's entry active, after its
  # committed detector diagnostic, observer and PD and ADL-corpus (E4)
  SCREENED=strong_${BEHAVIOR}_cont.json; CONTRACTS=stage3_panel_contracts.json
  DIAG=stage3_diagnostic_$BEHAVIOR.json; ADL=stage3_diagnostic_adl_$BEHAVIOR.json
  RECORDED=$(jget "$(tail -n 1 $OUT/receipt-check.json)" panel_contracts 2>/dev/null)
  if [ -z "$RECORDED" ]; then
    log "not run: the meter has recorded no panel contracts (continuation_meter.py panel-contracts)"; step manifest 1
  elif [ ! -f $SRC/manifests/$CONTRACTS ]; then
    log "not run: no committed panel contracts $CONTRACTS in the snapshot"; step manifest 1
  elif [ "$(sha256sum < $SRC/manifests/$CONTRACTS | cut -d' ' -f1)" != "$RECORDED" ]; then
    log "not run: the snapshot's $CONTRACTS is not the contracts the meter recorded ($RECORDED)"; step manifest 1
  elif [ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["behaviors"][sys.argv[2]]["status"])' \
            $SRC/manifests/$CONTRACTS "$BEHAVIOR" 2>/dev/null)" != active ]; then
    log "no active entry: the panel's strong branch of $BEHAVIOR is closed in its contracts"; step manifest 1
  elif [ ! -f $SRC/manifests/$SCREENED ]; then
    log "not run: no committed strong screening $SCREENED in the snapshot"; step manifest 1
  elif [ ! -f $SRC/manifests/$DIAG ] || [ ! -f $SRC/manifests/$ADL ]; then
    log "not run: no committed detector diagnostic $DIAG and $ADL in the snapshot"; step manifest 1
  else
    run_pass $CHAIN-strong-validate $CFG/$BEHAVIOR strong_validate \
      pipeline.plan.strong_validate.branch=stage3 "+pipeline.plan.strong_validate.contracts='$RECORDED'"
    step strong_validate $?
    log "the panel's strong validation: its manifest is committed on the laptop before any follow-up (s3x)"
  fi ;;
s3x)
  # X_cont of the panel's validated strong candidate, after its committed passing validation, under
  # the panel's contracts as the meter recorded them, the ones the validation names (review s6n 1)
  VALID=strong_validation_${BEHAVIOR}_cont.json; CONTRACTS=stage3_panel_contracts.json
  RECORDED=$(jget "$(tail -n 1 $OUT/receipt-check.json)" panel_contracts 2>/dev/null)
  if [ -z "$RECORDED" ]; then
    log "not run: the meter has recorded no panel contracts (continuation_meter.py panel-contracts)"; step manifest 1
  elif [ ! -f $SRC/manifests/$CONTRACTS ]; then
    log "not run: no committed panel contracts $CONTRACTS in the snapshot"; step manifest 1
  elif [ "$(sha256sum < $SRC/manifests/$CONTRACTS | cut -d' ' -f1)" != "$RECORDED" ]; then
    log "not run: the snapshot's $CONTRACTS is not the contracts the meter recorded ($RECORDED)"; step manifest 1
  elif [ ! -f $SRC/manifests/$VALID ]; then
    log "not run: no committed strong validation $VALID in the snapshot"; step manifest 1
  elif [ "$(manifest $VALID '((d.get("inputs") or {}).get("contracts") or {}).get("sha256")')" != "$RECORDED" ]; then
    log "not run: $VALID was opened under other contracts than the meter recorded ($RECORDED)"; step manifest 1
  elif [ "$(manifest $VALID 'd["verdict"]["passed"]')" != True ]; then
    log "the panel's strong candidate did not validate: no follow-up"; step manifest 1
  else
    DOSE=$(manifest $VALID 'd["candidate"]["dose"]')
    run_pass $CHAIN-s3sxcont-$DOSE $CFG/$BEHAVIOR s3s_xcont +pipeline.plan.s3s_xcont.dose=$DOSE \
      "+pipeline.plan.s3s_xcont.contracts='$RECORDED'"
    step s3s_xcont-$DOSE $?
    log "X_cont of strong_${BEHAVIOR}_cont_$DOSE; its k* follows on the base item"
  fi ;;
s3k|s3c)
  # the certified lineage under the panel's contracts as the meter recorded them (the receipt
  # check's panel_contracts, the snapshot's file hashing to it): its lineage, dose and states
  # from the committed evidence alone (continuation_stage3.py certified), every pass bound to
  # that record
  CONTRACTS=stage3_panel_contracts.json
  RECORDED=$(jget "$(tail -n 1 $OUT/receipt-check.json)" panel_contracts 2>/dev/null)
  if [ -z "$RECORDED" ]; then
    log "not run: the meter has recorded no panel contracts (continuation_meter.py panel-contracts)"; step manifest 1
  elif [ ! -f $SRC/manifests/$CONTRACTS ] \
       || [ "$(sha256sum < $SRC/manifests/$CONTRACTS | cut -d' ' -f1)" != "$RECORDED" ]; then
    log "not run: the snapshot's $CONTRACTS is not the contracts the meter recorded ($RECORDED)"; step manifest 1
  elif ! CERTIFIED=$(in_src env RUNS_DIR=$SPACE/runs DATA_DIR=$SPACE/data python \
         scripts/continuation_stage3.py certified "$BEHAVIOR" --contracts "$RECORDED" \
         2>$OUT/certified.err); then
    log "not run: no certified lineage of $BEHAVIOR under the recorded contracts: $(tail -n 1 $OUT/certified.err)"; step manifest 1
  else
    LINEAGE=$(jget "$CERTIFIED" lineage); DOSE=$(jget "$CERTIFIED" dose)
    LATENT=$(jget "$CERTIFIED" latent); SHAM=$(jget "$CERTIFIED" sham_s)
    A=$(jget "$CERTIFIED" active); K=kstar_$LINEAGE.json
    log "the certified lineage $LINEAGE ($(jget "$CERTIFIED" reading)) at dose $DOSE"
    if [ $SESSION = s3k ]; then
      run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar "+pipeline.plan.kstar.stage3=true" \
        "+pipeline.plan.kstar.contracts='$RECORDED'" pipeline.plan.kstar.dose=$DOSE \
        pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=$A
      step kstar $?
      log "the certified lineage's k*: its manifest $K is committed on the laptop once the analysis is frozen, before the certificate (s3c)"
    elif [ ! -f $SRC/manifests/$K ]; then
      log "not run: no committed k* manifest $K in the snapshot"; step manifest 1
    else
      copy_in --certificate $LINEAGE --manifests $SRC/manifests; step copy $?
      CERT=("+pipeline.plan.certificate.stage3=true" "+pipeline.plan.certificate.contracts='$RECORDED'"
            pipeline.plan.certificate.dose=$DOSE pipeline.plan.certificate.latent=$LATENT
            pipeline.plan.certificate.sham_s=$SHAM pipeline.plan.certificate.active=$A
            "pipeline.plan.certificate.blocks=[0,1,2,3,4,5]")
      if [ $OK = 1 ]; then
        run_pass $CHAIN-cert-check $CFG/$BEHAVIOR certificate "${CERT[@]}" pipeline.plan.check_only=true
        step certificate-check $?
      fi
      if [ $OK = 1 ]; then run_pass $CHAIN-cert $CFG/$BEHAVIOR certificate "${CERT[@]}"; step certificate $?; fi
      log "the certificate of $LINEAGE: its records stay unopened until the analysis is frozen"
    fi
  fi ;;
s5k)
  # k* of the committed Stage-5 lineage, after its decision is committed (review s4t2 3)
  if [ ! -f $SRC/manifests/stage5_$BEHAVIOR.json ]; then
    log "not run: no committed Stage-5 decision of $BEHAVIOR in the snapshot"; step manifest 1
  else
    DOSE=$(manifest stage5_$BEHAVIOR.json 'd["decision"]["analysis_dose"]')
    if [ "$DOSE" = None ]; then
      log "the Stage-5 decision has no analysis dose: no k*"; step manifest 1
    else
      SHAM=$(jget "$(s5_states $DOSE)" sham_s); A=$(manifest stage5_$BEHAVIOR.json 'd["implant"]["active"]')
      run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar pipeline.plan.kstar.stage5=true \
        pipeline.plan.kstar.dose=$DOSE pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=$A
      step kstar $?
    fi
  fi ;;
s5dk)
  # k* of the committed DPO lineage <behavior>_dpo_<dose> (C15), after its decision is committed;
  # it has no certificate or readers
  if [ $OK != 1 ]; then
    log "$SESSION: no pass without its inputs"
  elif [ ! -f $SRC/manifests/stage5_dpo_$BEHAVIOR.json ]; then
    log "not run: no committed DPO-lineage decision of $BEHAVIOR in the snapshot"; step manifest 1
  else
    DOSE=$(manifest stage5_dpo_$BEHAVIOR.json 'd["decision"]["analysis_dose"]')
    if [ "$DOSE" = None ]; then
      log "the DPO-lineage decision has no analysis dose: no k*"; step manifest 1
    else
      SHAM=$(jget "$(s5_states $DOSE)" sham_s)
      A=$(manifest stage5_dpo_$BEHAVIOR.json 'd["implant"]["active"]')
      run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar pipeline.plan.kstar.stage5_dpo=true \
        pipeline.plan.kstar.dose=$DOSE pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=$A
      step kstar $?
    fi
  fi ;;
s5c)
  # The committed k* manifest of <behavior>_cont_<dose> names the dose, H_s and A; L is the
  # selection's there. The certificate's inputs that are not in a come from c, then b.
  DOSE=$(manifest stage5_$BEHAVIOR.json 'd["decision"]["analysis_dose"]')
  LINEAGE=${BEHAVIOR}_cont_$DOSE; K=kstar_$LINEAGE.json
  if [ ! -f $SRC/manifests/$K ]; then
    log "not run: no committed k* manifest $K in the snapshot"; step manifest 1
  else
    SHAM=$(manifest $K 'd["states"]["H_s"]'); A=$(manifest $K 'd["states"]["A"]')
    LATENT=$(jget "$(s5_states $DOSE)" latent)
    copy_in --certificate $LINEAGE --manifests $SRC/manifests; step copy $?
    CERT=(pipeline.plan.certificate.stage5=true pipeline.plan.certificate.dose=$DOSE
          pipeline.plan.certificate.latent=$LATENT pipeline.plan.certificate.sham_s=$SHAM
          pipeline.plan.certificate.active=$A "pipeline.plan.certificate.blocks=[0,1,2,3,4,5]")
    if [ $OK = 1 ]; then
      run_pass $CHAIN-cert-check $CFG/$BEHAVIOR certificate "${CERT[@]}" pipeline.plan.check_only=true
      step certificate-check $?
    fi
    if [ $OK = 1 ]; then run_pass $CHAIN-cert $CFG/$BEHAVIOR certificate "${CERT[@]}"; step certificate $?; fi
    if [ $OK = 1 ]; then run_pass $CHAIN-relearn-h $CFG/$BEHAVIOR s5_relearn_h "${CERT[@]}"; step s5_relearn_h $?; fi
    if [ $OK = 1 ]; then
      run_pass $CHAIN-readers $CFG/$BEHAVIOR readers pipeline.plan.readers.stage5=true \
        pipeline.plan.readers.dose=$DOSE
      step readers $?
    fi
  fi ;;
s5s)
  # S5-A's screening of the frozen candidates (C16-2): its record is committed before validation
  run_pass $CHAIN-strong-screen $CFG/$BEHAVIOR strong_screen \
    pipeline.plan.strong_screen.branch=stage5_continued
  step strong_screen $?
  log "the strong screening: its manifest is committed on the laptop before validation (s5v)" ;;
s5v)
  # S5-A's one validation, after the committed screening named an eligible candidate
  SCREENED=strong_${BEHAVIOR}_cont.json
  if [ ! -f $SRC/manifests/$SCREENED ]; then
    log "not run: no committed strong screening $SCREENED in the snapshot"; step manifest 1
  elif [ "$(manifest $SCREENED 'd["chosen"] is not None')" != True ]; then
    log "no eligible candidate: the continued strong branch closed at its screening"; step manifest 1
  else
    run_pass $CHAIN-strong-validate $CFG/$BEHAVIOR strong_validate \
      pipeline.plan.strong_validate.branch=stage5_continued
    step strong_validate $?
    log "the strong validation: its manifest is committed on the laptop before any follow-up (s5sk)"
  fi ;;
s5ds)
  # the DPO fallback's screening (C16-3), once its gate is open (the plan check refuses before)
  if [ $OK = 1 ]; then
    run_pass $CHAIN-strong-screen $CFG/$BEHAVIOR strong_screen pipeline.plan.strong_screen.branch=stage5_dpo
    step strong_screen $?
    log "the DPO fallback's screening: its manifest is committed on the laptop before validation (s5dv)"
  fi ;;
s5dv)
  SCREENED=strong_${BEHAVIOR}_dpo.json
  if [ $OK != 1 ]; then
    log "not run: without the DPO fallback's gate evidence in c"
  elif [ ! -f $SRC/manifests/$SCREENED ]; then
    log "not run: no committed strong screening $SCREENED in the snapshot"; step manifest 1
  elif [ "$(manifest $SCREENED 'd["chosen"] is not None')" != True ]; then
    log "no eligible candidate: the DPO fallback closed at its screening"; step manifest 1
  else
    run_pass $CHAIN-strong-validate $CFG/$BEHAVIOR strong_validate pipeline.plan.strong_validate.branch=stage5_dpo
    step strong_validate $?
    log "the DPO fallback's validation: its manifest is committed on the laptop before any follow-up (s5dx)"
  fi ;;
s5dx|s5dc)
  VALID=strong_validation_${BEHAVIOR}_dpo.json
  if [ $OK != 1 ]; then
    log "not run: without the DPO fallback's gate evidence in c"
  elif [ ! -f $SRC/manifests/$VALID ]; then
    log "not run: no committed strong validation $VALID in the snapshot"; step manifest 1
  elif [ "$(manifest $VALID 'd["verdict"]["passed"]')" != True ]; then
    log "the DPO candidate did not validate: no follow-up"; step manifest 1
  else
    DOSE=$(manifest $VALID 'd["candidate"]["dose"]')
    LATENT=$(manifest $VALID 'd["candidate"]["latent"]'); SHAM=$(manifest $VALID 'd["candidate"]["sham_s"]')
    A=$(manifest stage5_dpo_$BEHAVIOR.json 'd["implant"]["active"]')
    LINEAGE=strong_${BEHAVIOR}_dpo_$DOSE; K=kstar_$LINEAGE.json
    if [ $SESSION = s5dx ]; then
      run_pass $CHAIN-s5dcontrols-$DOSE $CFG/$BEHAVIOR s5d_controls pipeline.plan.s5d_controls.dose=$DOSE
      step s5d_controls-$DOSE $?
      if [ $OK = 1 ]; then
        run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar pipeline.plan.kstar.stage5_dpo=true \
          pipeline.plan.kstar.strong=true pipeline.plan.kstar.dose=$DOSE \
          pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=$A
        step kstar $?
      fi
      log "the DPO fallback's k*: its manifest $K is committed on the laptop before the certificate (s5dc)"
    elif [ ! -f $SRC/manifests/$K ]; then
      log "not run: no committed k* manifest $K in the snapshot"; step manifest 1
    else
      copy_in --certificate $LINEAGE --manifests $SRC/manifests; step copy $?
      CERT=(pipeline.plan.certificate.stage5_dpo=true pipeline.plan.certificate.strong=true
            pipeline.plan.certificate.dose=$DOSE pipeline.plan.certificate.latent=$LATENT
            pipeline.plan.certificate.sham_s=$SHAM pipeline.plan.certificate.active=$A
            "pipeline.plan.certificate.blocks=[0,1,2,3,4,5]")
      if [ $OK = 1 ]; then
        run_pass $CHAIN-cert-check $CFG/$BEHAVIOR certificate "${CERT[@]}" pipeline.plan.check_only=true
        step certificate-check $?
      fi
      if [ $OK = 1 ]; then run_pass $CHAIN-cert $CFG/$BEHAVIOR certificate "${CERT[@]}"; step certificate $?; fi
      if [ $OK = 1 ]; then run_pass $CHAIN-relearn-h $CFG/$BEHAVIOR s5_relearn_h "${CERT[@]}"; step s5_relearn_h $?; fi
      if [ $OK = 1 ]; then
        run_pass $CHAIN-readers $CFG/$BEHAVIOR readers pipeline.plan.readers.stage5_dpo=true \
          pipeline.plan.readers.strong=true pipeline.plan.readers.dose=$DOSE
        step readers $?
      fi
    fi
  fi ;;
s5sk|s5sc)
  # S5-B, after the committed passing validation: X_cont and k* (s5sk), then the certificate, H's
  # relearning and the readers (s5sc), each of the strong lineage strong_<behavior>_cont_<dose>
  VALID=strong_validation_${BEHAVIOR}_cont.json
  if [ ! -f $SRC/manifests/$VALID ]; then
    log "not run: no committed strong validation $VALID in the snapshot"; step manifest 1
  elif [ "$(manifest $VALID 'd["verdict"]["passed"]')" != True ]; then
    log "the strong candidate did not validate: no follow-up"; step manifest 1
  else
    DOSE=$(manifest $VALID 'd["candidate"]["dose"]')
    LATENT=$(manifest $VALID 'd["candidate"]["latent"]'); SHAM=$(manifest $VALID 'd["candidate"]["sham_s"]')
    A=$(manifest stage5_$BEHAVIOR.json 'd["implant"]["active"]')
    LINEAGE=strong_${BEHAVIOR}_cont_$DOSE; K=kstar_$LINEAGE.json
    if [ $SESSION = s5sk ]; then
      run_pass $CHAIN-s5xcont-$DOSE $CFG/$BEHAVIOR s5_xcont pipeline.plan.s5_xcont.strong=true \
        pipeline.plan.s5_xcont.dose=$DOSE
      step s5_xcont-$DOSE $?
      if [ $OK = 1 ]; then
        run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar pipeline.plan.kstar.stage5=true \
          pipeline.plan.kstar.strong=true pipeline.plan.kstar.dose=$DOSE \
          pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=$A
        step kstar $?
      fi
      log "the strong lineage's k*: its manifest $K is committed on the laptop before the certificate (s5sc)"
    elif [ ! -f $SRC/manifests/$K ]; then
      log "not run: no committed k* manifest $K in the snapshot"; step manifest 1
    else
      copy_in --certificate $LINEAGE --manifests $SRC/manifests; step copy $?
      CERT=(pipeline.plan.certificate.stage5=true pipeline.plan.certificate.strong=true
            pipeline.plan.certificate.dose=$DOSE pipeline.plan.certificate.latent=$LATENT
            pipeline.plan.certificate.sham_s=$SHAM pipeline.plan.certificate.active=$A
            "pipeline.plan.certificate.blocks=[0,1,2,3,4,5]")
      if [ $OK = 1 ]; then
        run_pass $CHAIN-cert-check $CFG/$BEHAVIOR certificate "${CERT[@]}" pipeline.plan.check_only=true
        step certificate-check $?
      fi
      if [ $OK = 1 ]; then run_pass $CHAIN-cert $CFG/$BEHAVIOR certificate "${CERT[@]}"; step certificate $?; fi
      if [ $OK = 1 ]; then run_pass $CHAIN-relearn-h $CFG/$BEHAVIOR s5_relearn_h "${CERT[@]}"; step s5_relearn_h $?; fi
      if [ $OK = 1 ]; then
        run_pass $CHAIN-readers $CFG/$BEHAVIOR readers pipeline.plan.readers.stage5=true \
          pipeline.plan.readers.strong=true pipeline.plan.readers.dose=$DOSE
        step readers $?
      fi
    fi
  fi ;;
esac

log "export"
export_space $SPACE adl-api-$WORKER.jsonl $OUT/export.log; XS=$?; step export $XS
# The session's ledger as it stands now, in the chain's own directory, before the summary: an
# immutable copy of each session, whatever object of that name the bucket holds (Stage 2's pod
# ledgers share their names with the first part's of 2026-10-13, so the export never wrote them)
if [ -f "$LEDGER" ]; then cp "$LEDGER" "$OUT/ledger-$(basename "$LEDGER")"; step ledger $?; fi
python3 - "$OUT/summary.json" "${STEPS[@]}" <<PY
import json, sys
steps = dict(s.split("=", 1) for s in sys.argv[2:])
json.dump({"chain": "$CHAIN", "session": "$SESSION", "item": "$ITEM", "worker": "$WORKER",
           "pod_id": "$POD_ID", "steps": {k: int(v) for k, v in steps.items()}, "ok": bool($OK)},
          open(sys.argv[1], "w"), indent=2)
PY
upload_dir $OUT runs/_attempts/chain-$CHAIN $AT/export-chain-$CHAIN.log || OK=0
[ $OK = 1 ]  # the exit handler stops the Pod
