#!/usr/bin/env bash
# Starts this host's streams of tonight's four-arm evaluation (eval_stream.sh), PER_SERVER streams on each of the 8 Qwen
# servers (GPU N at port 8885+N), in tmux -L jeffcollect sessions eval-HOST-gpuN-sK. A stream still running keeps its
# session (rerun to restart ended streams).
# Usage: eval_launch.sh EVAL_DIR HOST ADDRESS DRIVER_PREFIX DRIVER_SUFFIX PER_SERVER STEP_T TRIM_T DEADLINE
#   ADDRESS: the host address the task containers reach (Qwen ports and Jeff :8920); DRIVER: PREFIX-gpuN-SUFFIX or,
#   with SUFFIX -, PREFIX-gpuN (e.g. qwen3.8-27b-nvfp4@b200 vllm0.29 -> qwen3.8-27b-nvfp4@b200-gpu3-vllm0.29).
set -euo pipefail
[ $# -eq 9 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
E=$(cd "$1" && pwd) HOST=$2 ADDR=$3 PREFIX=$4 SUFFIX=$5 PER=$6 STEP_T=$7 TRIM_T=$8 DEADLINE=$9
started=0 running=0
for k in $(seq 1 "$PER"); do
	for gpu in 0 1 2 3 4 5 6 7; do
		stream=eval-$HOST-gpu$gpu-s$k
		driver=$PREFIX-gpu$gpu; [ "$SUFFIX" = - ] || driver=$driver-$SUFFIX
		if tmux -L jeffcollect has-session -t "=$stream" 2>/dev/null; then running=$((running + 1)); continue; fi
		tmux -L jeffcollect new-session -d -s "$stream" \
			"bash $E/repo/tools/jeff-first/eval_stream.sh $E $HOST $gpu $stream http://$ADDR:$((8885 + gpu)) $driver http://$ADDR:8920 $STEP_T $TRIM_T $DEADLINE"
		started=$((started + 1))
		sleep 2
	done
done
echo "$HOST: started $started streams, already running $running"
