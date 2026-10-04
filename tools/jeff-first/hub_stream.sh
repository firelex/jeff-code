#!/usr/bin/env bash
# One hub-dataset collection stream: claims a task from this host's queue (collect_queue.py), runs one session on it
# with the xhigh collection's settings (record mode, scout build 8db5381f3, router fixed:xhigh, output cap 32,768,
# bash only, multiplier 6, run approval all), releases it, and repeats until stopped (tmux kill-session). The session
# lands in OUT/round<k>/, k = the task's run number (1 = its first session). When every task of the queue is running,
# or Docker's disk has less than 150 GB free (collect_queue.py), it waits a minute and asks again.
# Rotated datasets (task_source.py ROTATED_DATASETS, SWE-rebench): the task's Docker Hub image is pulled before the
# session (image_pull.py: one pull at a time per host, at least PULL_INTERVAL seconds apart, default 40, so the two
# hosts sharing one Docker Hub account stay under its 200 pulls per hour) and removed after it. On Docker Hub's
# rate-limit error the claim is released without counting as a run and the stream waits 15 minutes.
# Held-out tasks (scoring sessions, collect_queue.py append-scoring) go to COLLECT_DIR/runs-scoring-heldout/<stream>/
# instead of OUT_FOLDER, so no training conversion reads them. This is stream version 2: a queue that requires it
# (collect_queue.py require-stream-version) stops version-1 streams at their next claim.
# Usage: hub_stream.sh COLLECT_DIR QUEUE_DIR URL OUT_FOLDER DRIVER_BUILD STREAM
#   COLLECT_DIR holds jeff-pi-scout-8db5381f3.tgz and hub/ (tools/jeff-first, task-sets.json, task-sets-inventory.json)
set -euo pipefail
export PATH=$HOME/.local/bin:/snap/bin:/raid/work/jeff-first/bin:$PATH
[ $# -eq 6 ] || { echo "usage: $0 COLLECT_DIR QUEUE_DIR URL OUT_FOLDER DRIVER_BUILD STREAM" >&2; exit 2; }
C=$1 QUEUE=$2 URL=$3 OUT=$1/$4 DRIVER=$5 STREAM=$6
HUB=$C/hub
TGZ=$C/jeff-pi-scout-8db5381f3.tgz
for f in "$TGZ" "$HUB/task-sets.json" "$HUB/task-sets-inventory.json" "$HUB/tools/jeff-first/run_phase0.sh" "$QUEUE/queue.json"; do
	[ -f "$f" ] || { echo "$f missing" >&2; exit 1; }
done
export JEFF_RUN_API_KEY=unused JEFF_RUN_THINKING_FORMAT=qwen-chat-template JEFF_RUN_MAX_OUTPUT_TOKENS=32768
export JEFF_FIRST_RUN_APPROVAL=all JEFF_FIRST_DRIVER_BUILD=$DRIVER JEFF_FIRST_THINKING_ROUTER=fixed:xhigh
export JEFF_RUN_TASK_SETS=$HUB/task-sets.json JEFF_RUN_TASK_INVENTORY=$HUB/task-sets-inventory.json
mkdir -p "$OUT"
while true; do
	status=0
	got=$(python3 "$HUB/tools/jeff-first/collect_queue.py" claim "$QUEUE" "$STREAM" $$ --stream-version 2) || status=$?
	if [ "$status" = 3 ]; then sleep 60; continue; fi
	[ "$status" = 0 ] || { echo "$(date -Is) claim failed (exit $status)" >> "$OUT/run.log"; exit "$status"; }
	read -r task run <<< "$got"
	echo "$(date -Is) claimed $task (run $run)" >> "$OUT/run.log"
	side=$(python3 "$HUB/tools/jeff-first/task_source.py" side "$HUB/task-sets.json" "$HUB/task-sets-inventory.json" "$task")
	case "$side" in
		training) dest=$OUT ;;
		held_out) dest=$C/runs-scoring-heldout/$STREAM ;;
		*) echo "$(date -Is) $task is $side: not run" >> "$OUT/run.log"; exit 1 ;;
	esac
	image=$(python3 "$HUB/tools/jeff-first/task_source.py" image "$HUB/task-sets.json" "$HUB/task-sets-inventory.json" "$task")
	if [ "$image" != - ]; then
		status=0
		python3 "$HUB/tools/jeff-first/image_pull.py" pull "$image" "$HUB/pull-lock" "${PULL_INTERVAL:-40}" >> "$OUT/run.log" 2>&1 || status=$?
		if [ "$status" = 75 ]; then
			echo "$(date -Is) RATE LIMITED pulling $image: $task released without counting a run; waiting 15 minutes" >> "$OUT/run.log"
			python3 "$HUB/tools/jeff-first/collect_queue.py" release "$QUEUE" "$STREAM" "$task" --not-run
			sleep 900
			continue
		fi
		if [ "$status" != 0 ]; then
			echo "$(date -Is) PULL FAILED for $image (exit $status): $task counts as run $run without a session" >> "$OUT/run.log"
			python3 "$HUB/tools/jeff-first/collect_queue.py" release "$QUEUE" "$STREAM" "$task"
			continue
		fi
	fi
	mkdir -p "$dest/round$run"
	bash "$HUB/tools/jeff-first/run_phase0.sh" "$HUB/task-sets.json" "$TGZ" "$URL" "$dest/round$run" 1 high bash qwen3.8-27b record 6 "$task" \
		>> "$OUT/run.log" 2>&1 || echo "$(date -Is) session on $task exited $?" >> "$OUT/run.log"
	python3 "$HUB/tools/jeff-first/collect_queue.py" release "$QUEUE" "$STREAM" "$task"
	if [ "$image" != - ]; then
		python3 "$HUB/tools/jeff-first/image_pull.py" remove "$image" >> "$OUT/run.log" 2>&1 \
			|| echo "$(date -Is) REMOVAL FAILED for $image (see above)" >> "$OUT/run.log"
	fi
done
