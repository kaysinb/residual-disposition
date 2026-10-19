#!/usr/bin/env bash
# The continuation's Stage-2 Pods for one pilot behavior's item (preregistration §12.7; plan C8),
# run detached on the Pod (run_detached.py) from the verified source snapshot. Two Pods share the
# item (the meter's split): Pod A writes only to /workspace/a, Pod B only to /workspace/c (Runpod:
# writing one network volume from two Pods at once can corrupt it); B copies what it reads from a,
# create-only and checked (copy_inputs.py), and never writes to a. Each session stops its Pod.
#
#   a1 (Pod A): the screening, then the decision's steps as stage2_next.py names them (each
#      confirmation, the expression checks at the analysis dose), the decision cell (phase
#      decide) and, with an analysis dose, the lineage's k* calibration;
#   b1 (Pod B, needs no dose): Y₂'s criterion 3 (phase y2) and Y alone for link at 100 and 200;
#   a2 (Pod A, after the k* and Stage-2 manifests are committed and a new snapshot made): the
#      certificate's blocks 0-2 and the readers at the analysis dose;
#   b2 (Pod B, after the k* and Y₂ manifests are committed): blocks 3-5.
#
# Limits. RECEIPT is the meter's receipt of this Pod's session (continuation_meter.py
# session-open, with the Pod's ID and the deadline set at its creation, both from the
# coordinator), METER_LOG the copy of the log it was issued from; the chain starts only if
# verify-receipt accepts HARD_HOURS on this Pod (POD_ID). HARD_HOURS from START (the Pod's
# creation) bounds every deadline the chain sets; the watcher's first deadline is armed before the
# chain and may not lie beyond START + HARD_HOURS. The snapshot's receipt is trusted only by the
# values set in the Pod's environment at its creation (SOURCE_SNAPSHOT_ID, the snapshot;
# SNAPSHOT_RECEIPT_SHA256, its receipt; review s3z2). Every pass goes through the runner (CUDA
# faults retried, exit 3 final) with RUN_LIMIT_S to the end of work: the watcher's deadline minus
# the resume margin and the export. The chain exits nonzero if a pass did not complete or an
# export was not clean; the volume keeps everything, and a rerun of the session resumes its cells
# (on this Pod or, if it cannot start again, on a new Pod on the same volume).
# usage: SESSION=<a1|a2|b1|b2> SNAP=<snapshot id> START=<Pod creation ISO> WN=<watcher dir name> \
#          PRICE=<USD/h> HARD_HOURS=<h> RECEIPT=<file> METER_LOG=<file> POD_ID=<this Pod's ID> \
#          SOURCE_SNAPSHOT_RECEIPT=<file> [ITEM=stage2/example_link] [LEDGER_USD=<USD>] \
#          continuation_stage2.sh   (its directory: <attempts>/chain-<session>-<UTC time>)
set -uo pipefail
WS=${WORKSPACE:-/workspace}               # the network volume; another root only in tests
T=$WS/tools; AT=$WS/attempts/residual-disposition
CODE=${CODE_ROOT:-/root/code/residual-disposition}
: "${SESSION:?}" "${SNAP:?}" "${START:?}" "${WN:?}" "${PRICE:?}" "${HARD_HOURS:?}" "${RECEIPT:?}"
: "${METER_LOG:?}" "${POD_ID:?}" "${SOURCE_SNAPSHOT_RECEIPT:?}"
ITEM=${ITEM:-stage2/example_link}
BEHAVIOR=${ITEM#stage2/}
LEDGER_USD=${LEDGER_USD:-0.0}
case $SESSION in
  a1|a2) WORKER=pod-a; SPACE=$WS/a ;;
  b1|b2) WORKER=pod-b; SPACE=$WS/c ;;
  *) echo "SESSION must be a1, a2, b1 or b2"; exit 1 ;;
esac
CHAIN=$SESSION-$(date -u +%Y%m%dT%H%M%S)  # unique per launch; reused cells stay reused
OUT=$AT/chain-$CHAIN
W=$WS/watch/$WN
IMG=${CONTAINER_IMAGE:?the image digest the Pod runs}
SRC=$CODE/$SNAP/source
MARGIN=25; EXPORT=30                      # minutes
CFG=experiments/continuation_14b
ATTEMPTS="$CHAIN-*"
START_S=$(python3 -c "from datetime import datetime; import sys; print(datetime.fromisoformat(sys.argv[1]).timestamp())" "$START")
HARD_S=$(python3 -c "import sys; print(int(float(sys.argv[1]) + float(sys.argv[2]) * 3600))" "$START_S" "$HARD_HOURS")

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

set_deadline() {  # <epoch s>: the watcher's deadline, never past START + HARD_HOURS
  local at=$1 tok
  [ "$at" -gt "$HARD_S" ] && at=$HARD_S
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

COST=(pipeline.plan.cost.batches.stage2=$HARD_HOURS pipeline.plan.cost.batches.y=$HARD_HOURS
      pipeline.plan.cost.stop_fraction=0.9 pipeline.plan.cost.prior_hours=0.0
      pipeline.plan.cost.started_at=$START
      pipeline.plan.api.adl_ledger.worker=$WORKER pipeline.plan.api.adl_ledger.allowance_usd=$LEDGER_USD)

run_pass() {  # <attempt id> <config> <phase> [override ...]: one detached pass in this space, waited
  local id=$1 config=$2 phase=$3; shift 3
  local limit=$(( $(deadline_s) - (MARGIN + EXPORT) * 60 - $(now_s) ))
  if [ "$limit" -le 600 ]; then log "$id not run: less than 10 min of work before the deadline"; return 1; fi
  local cmd=(env -u RUN_ID project-run $SRC env RUN_LIMIT_S=$limit bash $T/rest_runner.sh $SPACE
             $config pipeline.plan.phase=$phase "pipeline.plan.cost.deadline_state=$W/state.json"
             "${COST[@]}" "$@")
  if ! python3 $T/run_detached.py --attempt-dir $AT/$id --cwd $SRC --run-id $id -- "${cmd[@]}"; then
    log "$id: launch error"; return 1
  fi
  local s; s=$(wait_attempt $AT/$id)
  log "$id: $s, exit code $(field $AT/$id/status.json exit_code)"
  [ "$s" = completed ] && [ "$(field $AT/$id/status.json exit_code)" = 0 ]
}

export_space() {  # <space> <ledger file> <log>: create-only export, then each path checked
  bash $T/bucket_final.sh $1 $2 $WN "$ATTEMPTS" > "$3" 2>&1; local rc=$?
  local expected=(data/states data/generations data/cache/adl_steps) p
  for p in $1/runs/*; do expected+=("runs/$(basename $p)"); done
  for p in $AT/$ATTEMPTS; do [ -e "$p" ] && expected+=("runs/_attempts/$(basename $p)"); done
  checks_clean "$3" "${expected[@]}" && [ $rc = 0 ]
}

upload_dir() {  # <local dir> <bucket path> <log>: one directory, create-only, checked
  bash $T/bucket_dir.sh "$1" "$2" > "$3" 2>&1; local rc=$?
  checks_clean "$3" "$2" && [ $rc = 0 ]
}

in_src() { (cd $SRC && project-run $SRC "$@"); }

copy_in() {  # [copy_inputs.py args]: from space a into this one (Pod B only), create-only, checked
  in_src python scripts/pod/copy_inputs.py --source $WS/a --target $SPACE --log $OUT/copied.jsonl "$@" \
    >> $OUT/copy.json 2>&1
}

manifest() {  # <file under manifests/> <python expression on d>: a committed manifest's value
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$SRC/manifests/$1" "$2"
}

mkdir -p $OUT
source ${RP_ENV:-/etc/rp_environment}
export SOURCE_SNAPSHOT_RECEIPT
export CONTAINER_IMAGE=$IMG PYTHONDONTWRITEBYTECODE=1
log "chain $CHAIN ($ITEM, $WORKER in $SPACE); hard limit $(iso $HARD_S) (START + $HARD_HOURS h), \$$PRICE/h"
if ! python3 $T/source_snapshot.py verify $CODE/$SNAP --expected-id $SNAP >/dev/null; then
  log "not run: snapshot $SNAP not verified"; exit 1
fi
if [ "${SOURCE_SNAPSHOT_ID:-}" != "$SNAP" ] || [ -z "${SNAPSHOT_RECEIPT_SHA256:-}" ] \
   || [ "$(sha256sum "$SOURCE_SNAPSHOT_RECEIPT" | cut -d' ' -f1)" != "$SNAPSHOT_RECEIPT_SHA256" ]; then
  log "not run: the snapshot or its receipt is not the one set at the Pod's creation"; exit 1
fi
if ! in_src python scripts/continuation_meter.py --log "$METER_LOG" verify-receipt "$RECEIPT" \
     --item "$ITEM" --worker "$WORKER" --pod-id "$POD_ID" --price "$PRICE" --hours "$HARD_HOURS" \
     > $OUT/receipt-check.json 2>&1; then
  log "not run: the meter's session receipt does not cover $HARD_HOURS h on Pod $POD_ID"; cat $OUT/receipt-check.json; exit 1
fi
cp "$RECEIPT" $OUT/receipt.json
if [ "$(deadline_s)" -gt "$HARD_S" ]; then
  log "not run: the watcher's deadline $(iso $(deadline_s)) is past START + HARD_HOURS"; exit 1
fi
export SOURCE_SNAPSHOT_ID SNAPSHOT_RECEIPT_SHA256

OK=1; STEPS=()
step() { STEPS+=("$1=$2"); [ "$2" = 0 ] || OK=0; }

case $SESSION in
a1)
  run_pass $CHAIN-screen $CFG/$BEHAVIOR screen; step screen $?
  PREV=""
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
        if [ $OK = 1 ] && [ -n "$ANALYSIS" ]; then
          run_pass $CHAIN-kstar $CFG/$BEHAVIOR kstar pipeline.plan.kstar.dose=$ANALYSIS \
            pipeline.plan.kstar.sham_s=$SHAM pipeline.plan.kstar.active=${ACTIVE:?A, the lineage\'s active organism}
          step kstar $?
        else
          log "no analysis dose: exception a (the Stage-5 slot), no k*"
        fi
        break ;;
      retest) log "the decision needs a confirmation retest after a detector revision: stop and report"; step retest 1; break ;;
      *) log "stage2_next: $NEXT"; step next 1; break ;;
    esac
  done ;;
b1)
  # Y₂'s organism needs no state; Y alone needs link's L and H_s at 100 and 200, the pilot's Y
  # organism, its approved reference, its relearning data and the analyst index.
  copy_in --state s4303557a4e7a9f64 --state sf8d4c3b1ca796eec --state sab16754d0858729b \
    --state se7a636b801f0ab1d --state sbf05f589f929edf8 --index feasibility_14b --index stage1_14b \
    --run feasibility_14b/20261014-072219-rates-2931c028-5097 \
    --run feasibility_14b/20261014-072309-data_relearn-8950fcdc-6f7f \
    --generation d5f5179f8853cb6cc --prompts
  step copy $?
  if [ $OK = 1 ]; then
    run_pass $CHAIN-y2 $CFG/y2_zero_indexed_lists y2; step y2 $?
    run_pass $CHAIN-y100 $CFG/control_y_link_100 control_y; step y_alone_100 $?
    run_pass $CHAIN-y200 $CFG/control_y_link_200 control_y; step y_alone_200 $?
  fi ;;
a2|b2)
  # The dose and states from the committed manifests; L from the screening record in space a
  # (read, never written, by Pod B).
  K=kstar_${BEHAVIOR}_$(manifest stage2_$BEHAVIOR.json 'd["decision"]["analysis_dose"]').json
  DOSE=$(manifest $K 'd["dose"]'); SHAM=$(manifest $K 'd["states"]["H_s"]'); A=$(manifest $K 'd["states"]["A"]')
  SCREEN=$(manifest stage2_$BEHAVIOR.json 'd["inputs"]["screening"]')
  LATENT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["doses"][sys.argv[2]]["latent"])' \
    "$WS/a/runs/continuation_14b/$SCREEN/stage2_screening.json" "$DOSE")
  BLOCKS='[0,1,2]'
  if [ $SESSION = b2 ]; then
    BLOCKS='[3,4,5]'
    CAL=$(manifest $K '" ".join("--run continuation_14b/" + r for rs in d["relearn_runs"].values() for r in rs)')
    copy_in --state $LATENT --state $SHAM --state $A --state sbf05f589f929edf8 \
      --index feasibility_14b --index stage1_14b --index continuation_14b $CAL \
      --run feasibility_14b/$(manifest $K 'd["reference"]["run"]') \
      --path runs/labeling/detectors-470596e7fc2a --path data/labels/detectors/y2det-470596e7fc2a \
      --path runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803 \
      --run feasibility_14b/20261014-072219-rates-2931c028-5097 \
      --run feasibility_14b/20261014-072309-data_relearn-8950fcdc-6f7f \
      --generation d5f5179f8853cb6cc --generation $(manifest $K 'd["data_id"]') --prompts
    step copy $?
  fi
  if [ $OK = 1 ]; then
    run_pass $CHAIN-cert $CFG/$BEHAVIOR certificate pipeline.plan.certificate.dose=$DOSE \
      pipeline.plan.certificate.latent=$LATENT pipeline.plan.certificate.sham_s=$SHAM \
      pipeline.plan.certificate.active=$A "pipeline.plan.certificate.blocks=$BLOCKS"
    step certificate $?
  fi
  if [ $SESSION = a2 ] && [ $OK = 1 ]; then
    run_pass $CHAIN-readers $CFG/$BEHAVIOR readers pipeline.plan.readers.dose=$DOSE
    step readers $?
  fi ;;
esac

log "export"
export_space $SPACE adl-api-$WORKER.jsonl $OUT/export.log; XS=$?; step export $XS
python3 - "$OUT/summary.json" "${STEPS[@]}" <<PY
import json, sys
steps = dict(s.split("=", 1) for s in sys.argv[2:])
json.dump({"chain": "$CHAIN", "session": "$SESSION", "item": "$ITEM", "worker": "$WORKER",
           "pod_id": "$POD_ID", "steps": {k: int(v) for k, v in steps.items()}, "ok": bool($OK)},
          open(sys.argv[1], "w"), indent=2)
PY
upload_dir $OUT runs/_attempts/chain-$CHAIN $AT/export-chain-$CHAIN.log || OK=0
log "stopping the Pod through the watcher"
set_deadline $(( $(now_s) + 180 )) || log "watcher: deadline not moved; it stops at its deadline"
[ $OK = 1 ]
