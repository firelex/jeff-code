#!/usr/bin/env bash
# Pulls ahead the Docker Hub images of the next AHEAD unstarted blocks of an evaluation queue (rotated datasets only:
# SWE-rebench), every 60 s, while Docker's disk keeps MIN_FREE_GB free; same per-image lock as eval_stream.sh and the
# collection's pull budget (image_pull.py, PULL_LOCK_DIR). A failed pull is logged and tried again on the next round.
# Usage: eval_prefetch.sh EVAL_DIR QUEUE PULL_LOCK_DIR AHEAD MIN_FREE_GB
set -euo pipefail
[ $# -eq 5 ] || { sed -n '2,5p' "$0" >&2; exit 2; }
E=$1 QUEUE=$E/$2 LOCKS=$3 AHEAD=$4 MIN_FREE=$5
TOOLS=$E/repo2/tools/jeff-first
SETS=$E/repo2/results/imitation/task-sets.json INV=$E/repo2/results/imitation/task-sets-inventory.json
mkdir -p "$E/image-locks"
while true; do
	for task in $(python3 "$TOOLS/eval_queue.py" upcoming "$QUEUE" "$AHEAD"); do
		case "$task" in *:*) ;; *) continue ;; esac
		image=$(python3 "$TOOLS/task_source.py" image "$SETS" "$INV" "$task")
		[ "$image" != - ] || continue
		free=$(df --output=avail -BG "$(docker info -f '{{.DockerRootDir}}')" | tail -1 | tr -dc 0-9)
		[ "$free" -ge "$MIN_FREE" ] || { echo "$(date -Is) disk ${free} GB free < $MIN_FREE: no prefetch"; break; }
		docker image inspect "$image" > /dev/null 2>&1 && continue
		flock "$E/image-locks/${image//[\/:]/.}.lock" python3 "$TOOLS/image_pull.py" pull "$image" "$LOCKS" \
			|| echo "$(date -Is) PREFETCH FAILED $image (exit $?)"
	done
	sleep 60
done
