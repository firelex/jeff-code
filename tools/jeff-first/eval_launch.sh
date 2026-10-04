#!/usr/bin/env bash
# Starts this host's streams of tonight's four-arm evaluation (eval_stream.sh), PER_SERVER streams on each GPU of GPUS
# (Qwen on port 8885+N), in tmux -L jeffcollect sessions NAME-gpuN-sK. A stream whose session still runs is left alone,
# so with WATCH=1 (loop every 60 s while any queue has unstarted blocks) a stream that ended, e.g. a TB2 stream whose
# queue ran dry, is replaced by one on the given queues: the host's stream count never rises.
# Usage: eval_launch.sh EVAL_DIR HOST ADDRESS DRIVER_PREFIX DRIVER_SUFFIX PER_SERVER STEP_T TRIM_T DEADLINE PULL_LOCK_DIR
#                       NAME GPUS QUEUE...
#   ADDRESS: the host address the task containers reach (Qwen ports and Jeff :8920); DRIVER: PREFIX-gpuN-SUFFIX or,
#   with SUFFIX -, PREFIX-gpuN; NAME e.g. eval-b200; GPUS e.g. 0,1,2,3,4,5,6,7; QUEUE files relative to EVAL_DIR.
set -euo pipefail
[ $# -ge 13 ] || { sed -n '2,10p' "$0" >&2; exit 2; }
E=$(cd "$1" && pwd) HOST=$2 ADDR=$3 PREFIX=$4 SUFFIX=$5 PER=$6 STEP_T=$7 TRIM_T=$8 DEADLINE=$9 LOCKS=${10} NAME=${11} GPUS=${12}
shift 12
QUEUES=("$@")
launch() {
	local started=0 running=0
	for k in $(seq 1 "$PER"); do
		for gpu in ${GPUS//,/ }; do
			stream=$NAME-gpu$gpu-s$k
			driver=$PREFIX-gpu$gpu; [ "$SUFFIX" = - ] || driver=$driver-$SUFFIX
			if tmux -L jeffcollect has-session -t "=$stream" 2>/dev/null; then running=$((running + 1)); continue; fi
			tmux -L jeffcollect new-session -d -s "$stream" \
				"bash $E/repo2/tools/jeff-first/eval_stream.sh $E $HOST $gpu $stream http://$ADDR:$((8885 + gpu)) $driver http://$ADDR:8920 $STEP_T $TRIM_T $DEADLINE $LOCKS ${QUEUES[*]}"
			started=$((started + 1))
			sleep 2
		done
	done
	echo "$(date -Is) $HOST $NAME: started $started streams, already running $running"
}
unstarted() {
	local n=0 q
	for q in "${QUEUES[@]}"; do
		n=$((n + $(python3 -c 'import json, sys; print(sum(b["server"] is None for b in json.load(open(sys.argv[1]))["blocks"]))' "$E/$q")))
	done
	echo "$n"
}
launch
if [ "${WATCH:-0}" = 1 ]; then
	while [ "$(unstarted)" -gt 0 ]; do
		sleep 60
		launch | grep -v "started 0 streams" || true
	done
	echo "$(date -Is) no unstarted blocks left: watcher ends"
fi
