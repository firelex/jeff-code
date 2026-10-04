#!/usr/bin/env bash
# Starts this host's hub-dataset collection streams (hub_stream.sh), one per stream of the host's xhigh collection
# launcher, with the same server URL and driver build; stream names are the launcher's output folder names.
# First counts this host's finished sessions (runs-collect-xhigh*; counts-*.json copied from other hosts into
# COLLECT_DIR/hub are added) and plans this host's queue (collect_queue.py plan) unless COLLECT_DIR/hub/queue exists.
# Every host must be given the same --host list, so all hosts compute the same split of tasks.
# Usage: hub_launch.sh [--dry-run] COLLECT_DIR THIS_HOST OLD_LAUNCHER --host NAME:STREAMS[:small] ... --dataset HUB_NAME ...
#   e.g. hub_launch.sh /raid/work/jeff-first/collect b200 collect_launch_b200.sh \
#          --host b200:48 --host casdgx01:24 --host datigator:6:small --dataset terminal-bench-pro/terminal-bench-pro ...
# Datasets added later go into the running queues with collect_queue.py append (same --host list on every host).
# Streams run in tmux -L jeffcollect sessions hub-<stream>; output in COLLECT_DIR/runs-collect-xhigh-hub/<stream>/.
set -euo pipefail
dry_run=0
if [ "${1:-}" = "--dry-run" ]; then dry_run=1; shift; fi
[ $# -ge 5 ] || { sed -n '2,12p' "$0" >&2; exit 2; }
C=$(cd "$1" && pwd) HOST=$2 OLD=$3
shift 3
HUB=$C/hub
QUEUE=$HUB/queue
TOOLS=$HUB/tools/jeff-first
[ -f "$OLD" ] || OLD=$C/$OLD
[ -f "$OLD" ] || { echo "no launcher at $3" >&2; exit 1; }
mapfile -t streams < <(grep -o 'collect_rounds\.sh [^"]*' "$OLD" | awk '{ n = split($4, p, "/"); print p[n], $3, $5 }')
[ "${#streams[@]}" -gt 0 ] || { echo "$OLD names no collect_rounds.sh streams" >&2; exit 1; }
if [ ! -f "$QUEUE/queue.json" ]; then
	shopt -s nullglob
	python3 "$TOOLS/collect_queue.py" count "$HUB/counts-$HOST.json" "$C"/runs-collect-xhigh*/
	counts=()
	for f in "$HUB"/counts-*.json; do counts+=(--counts "$f"); done
	target=$QUEUE
	if [ "$dry_run" = 1 ]; then target=$(mktemp -d)/queue; fi
	python3 "$TOOLS/collect_queue.py" plan "$HUB/task-sets.json" "$HUB/task-sets-inventory.json" "$target" "$HOST" "$@" "${counts[@]}"
fi
started=0 running=0
for line in "${streams[@]}"; do
	read -r stream url driver <<< "$line"
	command="bash $TOOLS/hub_stream.sh $C $QUEUE $url runs-collect-xhigh-hub/$stream $driver $stream"
	if [ "$dry_run" = 1 ]; then echo "hub-$stream: $command"; continue; fi
	# A stream still running keeps its session (rerun this script to restart streams that ended, e.g. drained ones).
	if tmux -L jeffcollect has-session -t "=hub-$stream" 2>/dev/null; then running=$((running + 1)); continue; fi
	tmux -L jeffcollect new-session -d -s "hub-$stream" "$command"
	started=$((started + 1))
done
echo "$HOST: ${#streams[@]} streams; $( [ "$dry_run" = 1 ] && echo "dry run" || echo "started $started, already running $running")"
