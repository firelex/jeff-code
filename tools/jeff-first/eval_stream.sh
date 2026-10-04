#!/usr/bin/env bash
# One stream of tonight's four-arm evaluation (2026-10-04) on one Qwen server: takes the next session from the host's
# queues in the order given (eval_queue.py claim; a block = one task attempt under all four arms, on one server), runs
# it through run_phase0.sh with that arm's settings, records its end, and repeats until no queue has anything left
# (exit 4 from each) or it is stopped.
# Arms (all: package jeff-pi-scout-86f8c8323.tgz, pi thinking high, bash only, output cap 32768, run approval all,
# multiplier 6 as the collection; hub tasks get at most 15 min x 6 of agent time, task_source.py):
#   a1-baseline   record mode (Jeff off), router fixed:xhigh, thinking limit off, trim off
#   a2-off-guard  record mode (Jeff off), router fixed:off, thinking limit 8000, trim off
#   a3-jeff07     jeff mode, step jeff-step @ STEP_T, router jeff-off-unless:jeff-router:0.7, trim jeff:jeff-trim @ TRIM_T,
#                 thinking limit 8000
#   a4-jeff06     as a3 with jeff-off-unless:jeff-router:0.6
# Rotated datasets (SWE-rebench): the task's Docker Hub image is pulled before the session (image_pull.py with the
# collection's PULL_LOCK_DIR budget; one pull per image at a time; a passing failure is retried 10 times a minute
# apart, then the session is recorded as pull-failed) and removed when the block's last session has ended.
# Usage: eval_stream.sh EVAL_DIR HOST GPU STREAM QWEN_URL DRIVER_BUILD JEFF_URL STEP_T TRIM_T DEADLINE PULL_LOCK_DIR QUEUE...
#   EVAL_DIR holds repo2/ (tools/jeff-first, results/phase0/tasks.json, results/imitation/task-sets*.json), the package
#   and the queue files (names relative to EVAL_DIR); DEADLINE: ISO time every session must be able to end by, or -.
# A session lands in EVAL_DIR/runs/ARM/TASK/attemptK/ (TASK with / and : as .) with meta.json (benchmark, arm, host,
# server, stream, block, times, exit).
set -euo pipefail
export PATH=$HOME/.local/bin:/snap/bin:/raid/work/jeff-first/bin:$PATH
[ $# -ge 12 ] || { sed -n '2,21p' "$0" >&2; exit 2; }
E=$1 HOST=$2 GPU=$3 STREAM=$4 URL=$5 DRIVER=$6 JEFF_URL=$7 STEP_T=$8 TRIM_T=$9 DEADLINE=${10} PULL_LOCK_DIR=${11}
shift 11
QUEUES=("$@")
R=$E/repo2
TOOLS=$R/tools/jeff-first
TGZ=$E/jeff-pi-scout-86f8c8323.tgz
SERVER=$HOST-gpu$GPU
export JEFF_RUN_TASK_SETS=$R/results/imitation/task-sets.json JEFF_RUN_TASK_INVENTORY=$R/results/imitation/task-sets-inventory.json
for f in "$TGZ" "$TOOLS/run_phase0.sh" "$R/results/phase0/tasks.json" "$JEFF_RUN_TASK_SETS" "$JEFF_RUN_TASK_INVENTORY" "$PULL_LOCK_DIR/settings.json"; do
	[ -f "$f" ] || { echo "$f missing" >&2; exit 1; }
done
for q in "${QUEUES[@]}"; do [ -f "$E/$q" ] || { echo "$E/$q missing" >&2; exit 1; }; done
deadline_args=()
[ "$DEADLINE" = - ] || deadline_args=(--deadline "$DEADLINE")
LOG=$E/streams/$STREAM.log
mkdir -p "$E/streams" "$E/image-locks"
export JEFF_RUN_API_KEY=unused JEFF_RUN_THINKING_FORMAT=qwen-chat-template JEFF_RUN_MAX_OUTPUT_TOKENS=32768
export JEFF_FIRST_RUN_APPROVAL=all JEFF_FIRST_DRIVER_BUILD=$DRIVER

write_meta() {  # OUT EXIT STARTED
	python3 -c 'import json, sys; keys = ["benchmark", "arm", "host", "server", "stream", "block", "task", "attempt", "driver", "router", "thinking_limit", "trim", "step_threshold", "started", "finished", "exit"]; json.dump(dict(zip(keys, sys.argv[1:-1])), open(sys.argv[-1], "w"), indent=1)' \
		"$benchmark" "$arm" "$HOST" "$SERVER" "$STREAM" "$block" "$task" "$attempt" "$DRIVER" "$JEFF_FIRST_THINKING_ROUTER" \
		"$JEFF_FIRST_THINKING_LIMIT" "$JEFF_FIRST_OUTPUT_TRIM" "${JEFF_FIRST_JEFF_STEP_THRESHOLD:--}" "$3" "$(date -Is)" "$2" \
		"$1/meta.json"
}

while true; do
	got="" queue="" waiting=0
	for q in "${QUEUES[@]}"; do
		status=0
		got=$(python3 "$TOOLS/eval_queue.py" claim "$E/$q" "$STREAM" "$SERVER" "${deadline_args[@]}") || status=$?
		if [ "$status" = 0 ]; then queue=$E/$q; break; fi
		if [ "$status" = 3 ]; then waiting=1; continue; fi
		[ "$status" = 4 ] || { echo "$(date -Is) claim on $q failed (exit $status)" >> "$LOG"; exit "$status"; }
	done
	if [ -z "$queue" ]; then
		if [ "$waiting" = 1 ]; then sleep 30; continue; fi
		echo "$(date -Is) nothing left to start here" >> "$LOG"
		exit 0
	fi
	read -r block arm task attempt benchmark <<< "$got"
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
	out=$E/runs/$arm/${task//[\/:]/.}/attempt$attempt
	[ ! -e "$out" ] || { echo "$(date -Is) $out exists: refusing to run $block $arm again" >> "$LOG"; exit 1; }
	mkdir -p "$out"
	echo "$benchmark" > "$out/benchmark.txt"
	started=$(date -Is)
	echo "$started start $block $arm $task attempt $attempt on $SERVER" >> "$LOG"
	image=-
	case "$task" in *:*) image=$(python3 "$TOOLS/task_source.py" image "$JEFF_RUN_TASK_SETS" "$JEFF_RUN_TASK_INVENTORY" "$task") ;; esac
	rc=0
	if [ "$image" != - ]; then
		for try in $(seq 1 10); do
			rc=0
			flock "$E/image-locks/${image//[\/:]/.}.lock" python3 "$TOOLS/image_pull.py" pull "$image" "$PULL_LOCK_DIR" >> "$out/pull.log" 2>&1 || rc=$?
			[ "$rc" = 75 ] || break
			echo "$(date -Is) pull of $image stopped (try $try, exit 75)" >> "$LOG"
			sleep 60
		done
	fi
	if [ "$rc" != 0 ]; then
		echo "$(date -Is) PULL FAILED $image for $block $arm (exit $rc, see $out/pull.log)" >> "$LOG"
		rc=pull-failed-$rc
	else
		bash "$TOOLS/run_phase0.sh" "$R/results/phase0/tasks.json" "$TGZ" "$URL" "$out" 1 high bash qwen3.8-27b "$mode" 6 "$task" \
			> "$out/run.log" 2>&1 || rc=$?
	fi
	write_meta "$out" "$rc" "$started"
	done_block=$(python3 "$TOOLS/eval_queue.py" finish "$queue" "$STREAM" "$block" "$arm" "$rc")
	echo "$(date -Is) end $block $arm $task attempt $attempt exit $rc" >> "$LOG"
	if [ "$done_block" = complete ] && [ "$image" != - ]; then
		python3 "$TOOLS/image_pull.py" remove "$image" >> "$LOG" 2>&1 || echo "$(date -Is) REMOVAL FAILED for $image" >> "$LOG"
	fi
done
