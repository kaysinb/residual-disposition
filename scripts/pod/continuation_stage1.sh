#!/usr/bin/env bash
# The continuation's Stage-1 Pod (preregistration §12.7, §12.4.8), run detached on the Pod
# (run_detached.py) from the verified source snapshot. The Pod's GPU limit comes from the meter
# (scripts/continuation_meter.py limits stage0_1 --price <the Pod's price>): POD_HOURS from the
# Pod's creation, of which ENGINE_HOURS are the engine check's own ceiling and the rest is the
# stage1 batch's cap (setup included).
#  1. reread, example_link, space /workspace/a: lens-only packages of its 18 pilot states and the
#     ADL-corpus check of its 4 latent states (stage1 batch, prior 0 h, from START);
#  2. reread, pun_intended, space /workspace/b: its 10 states and 2 latent states (stage1 batch,
#     the hours used since START as prior);
#  3. export both spaces (create-only, checked);
#  4. the watcher's deadline moved to now + ENGINE_HOURS (never past START + POD_HOURS); engine,
#     example_link, space a: the reproducibility probe, the paired check if the probe
#     reproduces, the timings (engine batch);
#  5. export space a again, then this chain's directory; the deadline moved to now + 3 min, so
#     the watcher stops the Pod.
# Every pass goes through the runner (CUDA faults retried, exit 3 final) with RUN_LIMIT_S to the
# end of work: the watcher's deadline minus the resume margin and the export.
# usage: SNAP=<snapshot id> START=<Pod creation ISO> WN=<watcher dir name> PRICE=<USD/h> \
#          POD_HOURS=<h> ENGINE_HOURS=<h> continuation_stage1.sh
#        (attempt dir /workspace/attempts/residual-disposition/chain-c1-<UTC date>)
set -uo pipefail
WS=${WORKSPACE:-/workspace}               # the network volume; another root only in tests
T=$WS/tools; AT=$WS/attempts/residual-disposition
CODE=${CODE_ROOT:-/root/code/residual-disposition}
: "${SNAP:?}" "${START:?}" "${WN:?}" "${PRICE:?}" "${POD_HOURS:?}" "${ENGINE_HOURS:?}"
DAY=$(date -u -d "$START" +%Y%m%d)
OUT=$AT/chain-c1-$DAY
W=$WS/watch/$WN
IMG=${CONTAINER_IMAGE:?the image digest the Pod runs}
SRC=$CODE/$SNAP/source
MARGIN=25; EXPORT=30                      # minutes
STAGE1_HOURS=$(python3 -c "print(round($POD_HOURS - $ENGINE_HOURS, 4))")
CFG=experiments/continuation_14b
ATTEMPTS="c1-$DAY-*"                       # this chain's passes; not its own directory

log() { echo "=== $(date -u +%FT%TZ) $*"; }
field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1" "$2"; }
now_s() { date -u +%s; }

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

set_deadline() {  # <ISO>: the watcher's deadline; 0 if state.json then holds it, armed, worker alive
  local tok
  tok=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["token"])' "$W/state.json")
  python3 $T/night_watch.py deadline --state-dir $W --token "$tok" --at "$1" --allow-deadline-stop >/dev/null 2>&1
  sleep 15
  python3 - "$W/state.json" "$1" <<'PY'
import json, sys
from datetime import UTC, datetime
s = json.load(open(sys.argv[1]))
want = datetime.fromisoformat(sys.argv[2]).timestamp()
ok = s.get("phase") == "armed" and not s.get("dry_run") and abs(float(s["deadline"]) - want) < 1
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
  [ "$s" = completed ]
}

export_space() {  # <space> <ledger file> <log>: create-only export, then each path checked
  bash $T/bucket_final.sh $1 $2 $WN "$ATTEMPTS" > "$3" 2>&1
  local expected=(data/states data/generations data/cache/adl_steps) p
  for p in $1/runs/*; do expected+=("runs/$(basename $p)"); done
  for p in $AT/$ATTEMPTS; do [ -e "$p" ] && expected+=("runs/_attempts/$(basename $p)"); done
  checks_clean "$3" "${expected[@]}"
}

mkdir -p $OUT
source ${RP_ENV:-/etc/rp_environment}
export SOURCE_SNAPSHOT_ID=$SNAP CONTAINER_IMAGE=$IMG PYTHONDONTWRITEBYTECODE=1
log "chain start; POD_HOURS $POD_HOURS, stage1 $STAGE1_HOURS h, engine $ENGINE_HOURS h at \$$PRICE/h"
if ! python3 $T/source_snapshot.py verify $CODE/$SNAP --expected-id $SNAP >/dev/null; then
  log "not run: snapshot $SNAP not verified"; exit 1
fi
STAGE1=(pipeline.plan.cost.batches.stage1=$STAGE1_HOURS pipeline.plan.cost.stop_fraction=0.9)

# 1-2. Stage 1, each behavior in its space; the second pass counts the batch's hours so far.
run_pass c1-$DAY-link $WS/a example_link reread "${STAGE1[@]}" \
  pipeline.plan.cost.prior_hours=0.0 pipeline.plan.cost.started_at=$START \
  pipeline.plan.api.adl_ledger.worker=cont-a pipeline.plan.api.adl_ledger.allowance_usd=4.0
LINK=$?
PRIOR=$(python3 -c "from datetime import datetime, UTC; import sys; print(round((datetime.now(UTC) - datetime.fromisoformat(sys.argv[1])).total_seconds() / 3600, 4))" "$START")
NOW=$(date -u +%FT%T+00:00)
run_pass c1-$DAY-pun $WS/b pun_intended reread "${STAGE1[@]}" \
  pipeline.plan.cost.prior_hours=$PRIOR pipeline.plan.cost.started_at=$NOW \
  pipeline.plan.api.adl_ledger.worker=cont-b pipeline.plan.api.adl_ledger.allowance_usd=3.0
PUN=$?

# 3. Export Stage 1 before the engine pass.
log "export after Stage 1"
export_space $WS/a adl-api-cont-a.jsonl $OUT/export-a-1.log; XA=$?
export_space $WS/b adl-api-cont-b.jsonl $OUT/export-b-1.log; XB=$?

# 4. The engine pass in its own batch, from now.
ENGINE=skipped
# D2 = min(now + ENGINE_HOURS, START + POD_HOURS); EH, the engine batch's cap, runs to it
read -r D2_S EH < <(python3 - "$START" "$POD_HOURS" "$ENGINE_HOURS" <<'PY'
import sys, time
from datetime import datetime
end = datetime.fromisoformat(sys.argv[1]).timestamp() + float(sys.argv[2]) * 3600
d2 = int(min(time.time() + float(sys.argv[3]) * 3600, end))
print(d2, round((d2 - time.time()) / 3600, 4))
PY
)
if set_deadline "$(date -u -d @$D2_S +%FT%T+00:00)"; then
  NOW=$(date -u +%FT%T+00:00)
  run_pass c1-$DAY-engine $WS/a example_link engine pipeline.plan.cost.batches.engine=$EH \
    pipeline.plan.cost.prior_hours=0.0 pipeline.plan.cost.started_at=$NOW \
    pipeline.plan.api.adl_ledger.worker=cont-a pipeline.plan.api.adl_ledger.allowance_usd=0.0
  [ $? = 0 ] && ENGINE=completed || ENGINE=failed
else
  log "engine not run: the watcher's deadline was not moved"
fi

# 5. Export again, the chain's directory, then the stop.
log "export after the engine pass"
export_space $WS/a adl-api-cont-a.jsonl $OUT/export-a-2.log; XA2=$?
python3 - "$OUT/summary.json" <<PY
import json, sys
json.dump({"reread_link": $LINK, "reread_pun": $PUN, "engine": "$ENGINE",
           "export_clean": {"a1": $XA == 0, "b1": $XB == 0, "a2": $XA2 == 0}},
          open(sys.argv[1], "w"), indent=2)
PY
bash $T/bucket_dir.sh $OUT runs/_attempts/$(basename $OUT) > $AT/export-$(basename $OUT).log 2>&1
checks_clean $AT/export-$(basename $OUT).log runs/_attempts/$(basename $OUT)
log "stopping the Pod through the watcher"
set_deadline "$(date -u -d '+3 min' +%FT%T+00:00)" || log "watcher: deadline not moved; it stops at its deadline"
