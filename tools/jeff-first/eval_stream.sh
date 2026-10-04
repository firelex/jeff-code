#!/usr/bin/env bash
# One stream of tonight's four-arm evaluation (2026-10-04) on one Qwen server: takes the next session from the host's
# queue (eval_queue.py claim; a block = one task attempt under all four arms, on one server), runs it through
# run_phase0.sh with that arm's settings, records its end, and repeats until nothing is left (exit 4) or stopped.
# Arms (all: package jeff-pi-scout-86f8c8323.tgz, pi thinking high, bash only, output cap 32768, run approval all,
# multiplier 6 as the collection):
#   a1-baseline   record mode (Jeff off), router fixed:xhigh, thinking limit off, trim off
#   a2-off-guard  record mode (Jeff off), router fixed:off, thinking limit 8000, trim off
#   a3-jeff07     jeff mode, step jeff-step @ STEP_T, router jeff-off-unless:jeff-router:0.7, trim jeff:jeff-trim @ TRIM_T,
#                 thinking limit 8000
#   a4-jeff06     as a3 with jeff-off-unless:jeff-router:0.6
# Usage: eval_stream.sh EVAL_DIR HOST GPU STREAM QWEN_URL DRIVER_BUILD JEFF_URL STEP_T TRIM_T DEADLINE
#   EVAL_DIR holds repo/ (tools/jeff-first, results/phase0/tasks.json), the package and queue-HOST.json;
#   DEADLINE: ISO time every session must be able to end by, or - for none.
# A session lands in EVAL_DIR/runs/ARM/TASK/attemptK/ with meta.json (arm, host, server, stream, block, times, exit).
set -euo pipefail
export PATH=$HOME/.local/bin:/snap/bin:/raid/work/jeff-first/bin:$PATH
[ $# -eq 10 ] || { sed -n '2,16p' "$0" >&2; exit 2; }
E=$1 HOST=$2 GPU=$3 STREAM=$4 URL=$5 DRIVER=$6 JEFF_URL=$7 STEP_T=$8 TRIM_T=$9 DEADLINE=${10}
TOOLS=$E/repo/tools/jeff-first
TGZ=$E/jeff-pi-scout-86f8c8323.tgz
QUEUE=$E/queue-$HOST.json
SERVER=$HOST-gpu$GPU
for f in "$TGZ" "$QUEUE" "$TOOLS/run_phase0.sh" "$E/repo/results/phase0/tasks.json"; do
	[ -f "$f" ] || { echo "$f missing" >&2; exit 1; }
done
deadline_args=()
[ "$DEADLINE" = - ] || deadline_args=(--deadline "$DEADLINE")
LOG=$E/streams/$STREAM.log
mkdir -p "$E/streams"
export JEFF_RUN_API_KEY=unused JEFF_RUN_THINKING_FORMAT=qwen-chat-template JEFF_RUN_MAX_OUTPUT_TOKENS=32768
export JEFF_FIRST_RUN_APPROVAL=all JEFF_FIRST_DRIVER_BUILD=$DRIVER
while true; do
	status=0
	got=$(python3 "$TOOLS/eval_queue.py" claim "$QUEUE" "$STREAM" "$SERVER" "${deadline_args[@]}") || status=$?
	if [ "$status" = 3 ]; then sleep 30; continue; fi
	if [ "$status" = 4 ]; then echo "$(date -Is) nothing left to start here" >> "$LOG"; exit 0; fi
	[ "$status" = 0 ] || { echo "$(date -Is) claim failed (exit $status)" >> "$LOG"; exit "$status"; }
	read -r block arm task attempt <<< "$got"
	unset JEFF_FIRST_JEFF_URL JEFF_FIRST_JEFF_STEP_ADAPTER JEFF_FIRST_JEFF_STEP_THRESHOLD JEFF_FIRST_JEFF_ROUTER_THRESHOLD \
		JEFF_FIRST_JEFF_TRIM_THRESHOLD
	case "$arm" in
		a1-baseline) mode=record; export JEFF_FIRST_THINKING_ROUTER=fixed:xhigh JEFF_FIRST_THINKING_LIMIT=off JEFF_FIRST_OUTPUT_TRIM=off ;;
		a2-off-guard) mode=record; export JEFF_FIRST_THINKING_ROUTER=fixed:off JEFF_FIRST_THINKING_LIMIT=8000 JEFF_FIRST_OUTPUT_TRIM=off ;;
		a3-jeff07|a4-jeff06)
			mode=jeff
			threshold=0.7; [ "$arm" = a4-jeff06 ] && threshold=0.6
			export JEFF_FIRST_JEFF_URL=$JEFF_URL JEFF_FIRST_JEFF_STEP_ADAPTER=jeff-step JEFF_FIRST_JEFF_STEP_THRESHOLD=$STEP_T
			export JEFF_FIRST_THINKING_ROUTER=jeff-off-unless:jeff-router:$threshold JEFF_FIRST_THINKING_LIMIT=8000
			export JEFF_FIRST_OUTPUT_TRIM=jeff:jeff-trim JEFF_FIRST_JEFF_TRIM_THRESHOLD=$TRIM_T
			;;
		*) echo "$(date -Is) unknown arm $arm" >> "$LOG"; exit 1 ;;
	esac
	out=$E/runs/$arm/$task/attempt$attempt
	[ ! -e "$out" ] || { echo "$(date -Is) $out exists: refusing to run $block $arm again" >> "$LOG"; exit 1; }
	mkdir -p "$out"
	started=$(date -Is)
	echo "$started start $block $arm $task attempt $attempt on $SERVER" >> "$LOG"
	rc=0
	bash "$TOOLS/run_phase0.sh" "$E/repo/results/phase0/tasks.json" "$TGZ" "$URL" "$out" 1 high bash qwen3.8-27b "$mode" 6 "$task" \
		> "$out/run.log" 2>&1 || rc=$?
	python3 -c 'import json, sys; keys = ["arm", "host", "server", "stream", "block", "task", "attempt", "driver", "router", "thinking_limit", "trim", "step_threshold", "started", "finished", "exit"]; json.dump(dict(zip(keys, sys.argv[1:])), open(sys.argv[-1], "w"), indent=1)' \
		"$arm" "$HOST" "$SERVER" "$STREAM" "$block" "$task" "$attempt" "$DRIVER" "$JEFF_FIRST_THINKING_ROUTER" \
		"$JEFF_FIRST_THINKING_LIMIT" "$JEFF_FIRST_OUTPUT_TRIM" "${JEFF_FIRST_JEFF_STEP_THRESHOLD:--}" "$started" "$(date -Is)" "$rc" \
		"$out/meta.json"
	python3 "$TOOLS/eval_queue.py" finish "$QUEUE" "$STREAM" "$block" "$arm" "$rc"
	echo "$(date -Is) end $block $arm $task attempt $attempt exit $rc" >> "$LOG"
done
