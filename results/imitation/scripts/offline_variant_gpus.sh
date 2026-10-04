#!/usr/bin/env bash
# offline_variant.sh with checkpoint predictions spread over several GPUs: each decision's cut questions
# (BUNDLE/cut/questions-<decision>.jsonl) are split round-robin into one shard per GPU, jeff_predict.py runs on all
# shards at once (one process per GPU), and the shards are joined back in question order; then offline_variant.sh
# scores the joined files. Usage:
#   GPUS="0 1 2 3 4 5 6 7" DECISION_MS=178 JEFF_DEV=... offline_variant_gpus.sh BUNDLE OUT_DIR NAME STEP_CKPT ROUTER_CKPT TRIM_CKPT
# The bundle must already be cut (BUNDLE/cut/fit-report.json; offline_variant.sh makes it on its first run).
set -euo pipefail
: "${GPUS:?set GPUS to the GPU indices to use, e.g. \"0 1 2 3\"}" "${JEFF_DEV:?set JEFF_DEV (jeff-dev checkout with .venv)}"
: "${DECISION_MS:?set DECISION_MS}"
if [ $# -ne 6 ]; then
  echo "usage: GPUS=... DECISION_MS=178 JEFF_DEV=... $0 BUNDLE OUT_DIR NAME STEP_CKPT ROUTER_CKPT TRIM_CKPT" >&2
  exit 2
fi
BUNDLE=$(cd "$1" && pwd); NAME=$3
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$BUNDLE/cut/fit-report.json" ] || { echo "$BUNDLE is not cut yet (no cut/fit-report.json)" >&2; exit 1; }
mkdir -p "$2/$NAME/shards"
OUT=$(cd "$2/$NAME" && pwd)
read -r -a gpus <<< "$GPUS"
n=${#gpus[@]}

predict() {  # decision checkpoint
  local decision=$1 checkpoint=$2 target="$OUT/shards/$1.joined.jsonl"
  [ -f "$target" ] && return 0
  local questions="$BUNDLE/cut/questions-$decision.jsonl" pids=() i
  for ((i = 0; i < n; i++)); do
    awk -v n="$n" -v i="$i" '(NR - 1) % n == i' "$questions" > "$OUT/shards/$decision-$i.jsonl"
    rm -f "$OUT/shards/$decision-$i.predictions.jsonl"
    (cd "$JEFF_DEV" && CUDA_VISIBLE_DEVICES=${gpus[$i]} .venv/bin/python "$HERE/jeff_predict.py" --checkpoint "$checkpoint" \
      --questions "$OUT/shards/$decision-$i.jsonl" --out "$OUT/shards/$decision-$i.predictions.jsonl" \
      --batch-size "${JEFF_BATCH:-8}" > "$OUT/shards/$decision-$i.log" 2>&1) &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "shard $i of $decision failed: $OUT/shards/$decision-$i.log" >&2; tail -5 "$OUT/shards/$decision-$i.log" >&2; exit 1; }
  done
  grep -h "questions/s" "$OUT"/shards/"$decision"-*.log
  # Join in question order: line k of the questions is line (k div n) of shard (k mod n).
  local files=()
  for ((i = 0; i < n; i++)); do files+=("$OUT/shards/$decision-$i.predictions.jsonl"); done
  awk -v n="$n" 'BEGIN { for (i = 0; i < n; i++) f[i] = ARGV[i + 1]; ARGC = 1 }
    { k = NR - 1; if ((getline line < f[k % n]) <= 0) { print "short shard " f[k % n] > "/dev/stderr"; exit 1 }; print line }' \
    "${files[@]}" < "$questions" > "$target.tmp"
  [ "$(wc -l < "$target.tmp")" -eq "$(wc -l < "$questions")" ] || { echo "joined $decision predictions are incomplete" >&2; exit 1; }
  mv "$target.tmp" "$target"
}

started=$(date +%s)
predict step "$4"
predict router "$5"
predict trim "$6"
echo "predictions: $(( $(date +%s) - started )) s on GPUs $GPUS"
bash "$HERE/offline_variant.sh" "$BUNDLE" "$2" "$NAME" "file:$OUT/shards/step.joined.jsonl" \
  "file:$OUT/shards/router.joined.jsonl" "file:$OUT/shards/trim.joined.jsonl"
