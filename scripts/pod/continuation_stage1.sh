#!/usr/bin/env bash
# The continuation's Stage-1 Pod (preregistration §12.7, §12.4.8), run detached on the Pod
# (run_detached.py) from the verified source snapshot.
#
# Limits. RECEIPT is the meter's launch receipt for stage0_1 (scripts/continuation_meter.py
# start stage0_1 --price PRICE --receipt), METER_LOG the copy of the meter log it was issued
# from; the chain starts only if verify-receipt accepts HARD_HOURS at PRICE. HARD_HOURS from
# START (the Pod's creation) bounds every deadline the chain sets, the final stop included;
# STAGE1_HOURS is the stage1 batch's cap from START (setup included) and the watcher's first
# deadline, armed before the chain; ENGINE_HOURS caps the engine batch; STAGE1_HOURS +
# ENGINE_HOURS <= HARD_HOURS.
#
#  1. reread, example_link, space a: lens-only packages of its 18 pilot states and the
#     ADL-corpus check of its 4 latent states (stage1 batch, from START, prior 0 h);
#  2. reread, pun_intended, space b: its 10 states and 2 latent states (the same batch: from
#     START, prior 0 h, so both passes end at START + STAGE1_HOURS);
#  3. export both spaces (create-only, checked); if either export is not clean, the engine pass
#     is skipped;
#  4. the watcher's deadline moved to min(now + ENGINE_HOURS, START + HARD_HOURS); engine,
#     example_link, space a: the reproducibility probe, the paired check if the probe
#     reproduces, the timings (engine batch);
#  5. export space a again and the engine check's journal, then this chain's directory; the
#     deadline moved to min(now + 3 min, START + HARD_HOURS), so the watcher stops the Pod.
# Every pass goes through the runner (CUDA faults retried, exit 3 final) with RUN_LIMIT_S to the
# end of work: the watcher's deadline minus the resume margin and the export. The chain exits
# nonzero if a pass did not complete or an export was not clean; the volume keeps everything.
# usage: SNAP=<snapshot id> START=<Pod creation ISO> WN=<watcher dir name> PRICE=<USD/h> \
#          HARD_HOURS=<h> STAGE1_HOURS=<h> ENGINE_HOURS=<h> RECEIPT=<file> METER_LOG=<file> \
#          LEDGER_A_USD=<USD> LEDGER_B_USD=<USD> \
#          continuation_stage1.sh   (its directory: <attempts>/chain-c1-<UTC time of its start>)
# LEDGER_A_USD and LEDGER_B_USD are the λ grader's API allowances of spaces a and b (ledgers
# cont-a and cont-b), sub-allowances of stage0_1's API part.
set -uo pipefail
WS=${WORKSPACE:-/workspace}               # the network volume; another root only in tests
T=$WS/tools; AT=$WS/attempts/residual-disposition
CODE=${CODE_ROOT:-/root/code/residual-disposition}
: "${SNAP:?}" "${START:?}" "${WN:?}" "${PRICE:?}" "${HARD_HOURS:?}" "${STAGE1_HOURS:?}"
: "${ENGINE_HOURS:?}" "${RECEIPT:?}" "${METER_LOG:?}" "${LEDGER_A_USD:?}" "${LEDGER_B_USD:?}"
CHAIN=c1-$(date -u +%Y%m%dT%H%M%S)       # unique per launch; reused cells stay reused
OUT=$AT/chain-$CHAIN
W=$WS/watch/$WN
IMG=${CONTAINER_IMAGE:?the image digest the Pod runs}
SRC=$CODE/$SNAP/source
MARGIN=25; EXPORT=30                      # minutes
CFG=experiments/continuation_14b
ATTEMPTS="$CHAIN-*"                        # this chain's passes; not its own directory
START_S=$(python3 -c "from datetime import datetime; import sys; print(datetime.fromisoformat(sys.argv[1]).timestamp())" "$START")
HARD_S=$(python3 -c "import sys; print(int(float(sys.argv[1]) + float(sys.argv[2]) * 3600))" "$START_S" "$HARD_HOURS")

log() { echo "=== $(date -u +%FT%TZ) $*"; }
field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1" "$2"; }
now_s() { date -u +%s; }
iso() { date -u -d "@$1" +%FT%T+00:00; }

wait_attempt() {  # until the attempt's status is final; prints it. supervisor_gone when its
  # recorded supervisor is dead, or the status stays unreadable, empty, or queued without a
  # supervisor for 10 min
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

set_deadline() {  # <epoch s>: the watcher's deadline, never past START + HARD_HOURS; 0 if
  # state.json then holds it, armed, and the watcher's worker is alive
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

run_pass() {  # <attempt id> <space> <behavior> <phase> [override ...]: one detached pass, waited
  local id=$1 space=$2 behavior=$3 phase=$4; shift 4
  local limit=$(( $(deadline_s) - (MARGIN + EXPORT) * 60 - $(now_s) ))
  if [ "$limit" -le 600 ]; then log "$id not run: less than 10 min of work before the deadline"; return 1; fi
  local cmd=(env -u RUN_ID project-run $SRC env RUN_LIMIT_S=$limit bash $T/rest_runner.sh $space
             $CFG/$behavior pipeline.plan.phase=$phase "pipeline.plan.cost.deadline_state=$W/state.json" "$@")
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

mkdir -p $OUT
source ${RP_ENV:-/etc/rp_environment}
export SOURCE_SNAPSHOT_ID=$SNAP CONTAINER_IMAGE=$IMG PYTHONDONTWRITEBYTECODE=1
log "chain $CHAIN; hard limit $(iso $HARD_S) (START + $HARD_HOURS h), stage1 $STAGE1_HOURS h, engine $ENGINE_HOURS h, \$$PRICE/h"
if ! python3 -c "import sys; a, b, c = map(float, sys.argv[1:]); sys.exit(0 if 0 < a and 0 < b and a + b <= c else 1)" \
     "$STAGE1_HOURS" "$ENGINE_HOURS" "$HARD_HOURS"; then
  log "not run: STAGE1_HOURS + ENGINE_HOURS must fit in HARD_HOURS"; exit 1
fi
if ! python3 $T/source_snapshot.py verify $CODE/$SNAP --expected-id $SNAP >/dev/null; then
  log "not run: snapshot $SNAP not verified"; exit 1
fi
if ! (cd $SRC && project-run $SRC python scripts/continuation_meter.py --log "$METER_LOG" \
      verify-receipt "$RECEIPT" --item stage0_1 --price "$PRICE" --hours "$HARD_HOURS") > $OUT/receipt-check.json 2>&1; then
  log "not run: the meter's launch receipt does not cover $HARD_HOURS h at \$$PRICE/h"; cat $OUT/receipt-check.json; exit 1
fi
cp "$RECEIPT" $OUT/receipt.json
D1_S=$(python3 -c "import sys; print(int(float(sys.argv[1]) + float(sys.argv[2]) * 3600))" "$START_S" "$STAGE1_HOURS")
if [ "$(deadline_s)" -gt "$D1_S" ]; then
  log "not run: the watcher's deadline $(iso $(deadline_s)) is past START + STAGE1_HOURS"; exit 1
fi

# 1-2. Stage 1, each behavior in its space, both in the stage1 batch from START.
STAGE1=(pipeline.plan.cost.batches.stage1=$STAGE1_HOURS pipeline.plan.cost.stop_fraction=0.9
        pipeline.plan.cost.prior_hours=0.0 pipeline.plan.cost.started_at=$START)
run_pass $CHAIN-link $WS/a example_link reread "${STAGE1[@]}" \
  pipeline.plan.api.adl_ledger.worker=cont-a pipeline.plan.api.adl_ledger.allowance_usd=$LEDGER_A_USD
LINK=$?
run_pass $CHAIN-pun $WS/b pun_intended reread "${STAGE1[@]}" \
  pipeline.plan.api.adl_ledger.worker=cont-b pipeline.plan.api.adl_ledger.allowance_usd=$LEDGER_B_USD
PUN=$?

# 3. Export Stage 1 before the engine pass.
log "export after Stage 1"
export_space $WS/a adl-api-cont-a.jsonl $OUT/export-a-1.log; XA=$?
export_space $WS/b adl-api-cont-b.jsonl $OUT/export-b-1.log; XB=$?

# 4. The engine pass in its own batch, from now; skipped if Stage 1's export is not clean.
ENGINE=skipped
if [ $XA != 0 ] || [ $XB != 0 ]; then
  log "engine not run: Stage 1's export is not clean"
else
  # NOW, the engine's deadline D2 and its cap EH (rounded up, so the cap ends no earlier than D2)
  read -r NOW_S D2_S EH < <(python3 - "$HARD_S" "$ENGINE_HOURS" <<'PY'
import math, sys, time
now = int(time.time())
d2 = int(min(now + float(sys.argv[2]) * 3600, float(sys.argv[1])))
print(now, d2, math.ceil((d2 - now) / 3600 * 1e4) / 1e4)
PY
)
  if set_deadline $D2_S; then
    run_pass $CHAIN-engine $WS/a example_link engine pipeline.plan.cost.batches.engine=$EH \
      pipeline.plan.cost.prior_hours=0.0 pipeline.plan.cost.started_at=$(iso $NOW_S) \
      pipeline.plan.api.adl_ledger.worker=cont-a pipeline.plan.api.adl_ledger.allowance_usd=0.0
    [ $? = 0 ] && ENGINE=completed || ENGINE=failed
  else
    log "engine not run: the watcher's deadline was not moved"; ENGINE=not_moved
  fi
fi

# 5. Export again with the engine check's journal, the chain's directory, then the stop.
log "export after the engine pass"
export_space $WS/a adl-api-cont-a.jsonl $OUT/export-a-2.log; XA2=$?
XJ=0
if [ -d $WS/a/data/cache/engine_check ]; then
  upload_dir $WS/a/data/cache/engine_check data/cache/engine_check $OUT/export-journal.log; XJ=$?
fi
OK=1
for rc in $LINK $PUN $XA $XB $XA2 $XJ; do [ "$rc" = 0 ] || OK=0; done
[ "$ENGINE" = completed ] || OK=0
python3 - "$OUT/summary.json" <<PY
import json, sys
json.dump({"chain": "$CHAIN", "reread_link": $LINK, "reread_pun": $PUN, "engine": "$ENGINE",
           "export_clean": {"a1": $XA == 0, "b1": $XB == 0, "a2": $XA2 == 0, "journal": $XJ == 0},
           "ok": bool($OK)}, open(sys.argv[1], "w"), indent=2)
PY
upload_dir $OUT runs/_attempts/chain-$CHAIN $AT/export-chain-$CHAIN.log || OK=0
log "stopping the Pod through the watcher"
set_deadline $(( $(now_s) + 180 )) || log "watcher: deadline not moved; it stops at its deadline"
[ $OK = 1 ]
