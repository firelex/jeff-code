#!/usr/bin/env bash
# Stops tonight's evaluation on this host (owner decision 2026-10-05: B200 stop at 15:30 BST): the watcher and image
# prefetchers first (nothing restarts), then every session still running is marked cut (eval_queue.py cut: cut.txt in
# its folder, status LABEL in the queue, folder list in EVAL_DIR/cut-LABEL.txt), the stream tmux sessions are ended,
# our remaining Harbor processes for EVAL_DIR/runs are stopped, and the cut trials' task containers and networks
# (Docker Compose project <trial>__env, matched by the trial's own random suffix and name) are removed. Nothing else
# is touched: no Qwen, no other containers.
# Usage: eval_b200_stop.sh EVAL_DIR HOST LABEL QUEUE...
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
E=$1 HOST=$2 LABEL=$3
shift 3
TOOLS=$E/repo2/tools/jeff-first
LIST=$E/cut-$LABEL.txt
[ ! -e "$LIST" ] || { echo "$LIST exists: already stopped?" >&2; exit 1; }
echo "$(date -Is) stopping $HOST evaluation ($LABEL)"
for s in eval-$HOST-watch eval-prefetch eval-prefetch3 eval-prefetch4; do
	tmux -L jeffcollect has-session -t "=$s" 2>/dev/null && tmux -L jeffcollect kill-session -t "=$s" && echo "ended $s"
done
: > "$LIST"
for q in "$@"; do python3 "$TOOLS/eval_queue.py" cut "$E/$q" "$E" "$LABEL" >> "$LIST"; done
echo "$(wc -l < "$LIST") sessions marked cut (list $LIST)"
for s in $(tmux -L jeffcollect ls -F '#{session_name}' 2>/dev/null | grep "^eval-$HOST-gpu" || true); do
	tmux -L jeffcollect kill-session -t "=$s"
done
echo "stream sessions ended; left: $(tmux -L jeffcollect ls -F '#{session_name}' 2>/dev/null | grep -c "^eval-$HOST-gpu" || true)"
pids=$(pgrep -u "$(id -u)" -f -- "-o $E/runs/" || true)
if [ -n "$pids" ]; then
	echo "stopping $(echo $pids | wc -w) Harbor processes: $pids"
	kill $pids 2>/dev/null || true
	sleep 30
	left=$(pgrep -u "$(id -u)" -f -- "-o $E/runs/" || true)
	[ -z "$left" ] || { echo "killing $left"; kill -9 $left 2>/dev/null || true; }
fi
removed=0
while read -r folder; do
	for trial in "$folder"/*/*__*; do
		[ -d "$trial" ] || continue
		name=$(basename "$trial" | tr 'A-Z' 'a-z')
		suffix=${name##*__}
		for project in $(docker ps -a --format '{{.Label "com.docker.compose.project"}}' | sort -u | grep -- "__${suffix}__env\$" || true); do
			[ "${project:0:8}" = "${name:0:8}" ] || { echo "skip $project (name does not match $name)"; continue; }
			docker ps -a -q --filter "label=com.docker.compose.project=$project" | xargs -r docker rm -f > /dev/null
			docker network ls -q --filter "label=com.docker.compose.project=$project" | xargs -r docker network rm > /dev/null
			echo "removed containers and networks of $project"
			removed=$((removed + 1))
		done
	done
done < "$LIST"
echo "$(date -Is) done: $removed compose projects removed"
