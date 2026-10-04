#!/usr/bin/env bash
# Owner's plan 2026-10-04 20:15: Jeff trainings on all 8 GPUs of one host with the developer's multi-GPU trainer
# (casdgx01 ~/v13new/trainer: jeff commit 9f84686 + trainer/PATCH.diff, train.py sha256 802c2a6f...; the pattern of
# ~/v13new/train_new.sh): one process per GPU (CUDA_VISIBLE_DEVICES, JEFF_RANK=i, JEFF_WORLD_SIZE=8,
# JEFF_DIST_URL=tcp://127.0.0.1:<port>), PYTHONPATH=<trainer>/src, `python -m jeff.train` run from a git checkout of
# 9f84686 (the trainer records its commit), --batch-size = effective batch / 8 (no accumulation), no --resume,
# --schedule or --patience. Runs given together run one after another. Runs ON the host (tonight.sh copies it there).
#
# Usage: tonight_dist.sh HOST RUN [RUN ...]
#   HOST  b200 or casdgx01 (paths below)
#   RUN   step-curriculum | router | trim  (v1.3 LoRA recipe, effective batch 64)
#         full-5e-6                        (ll-v12 full recipe, effective batch 256, lr 5e-6)
# Every run: ll-v12-final, one epoch, seed 20260920, weight decay 0.01, token budget / max length 8192, live-last,
# --keep-stage-order, --lr-shape wsd --decay-start 0.8, 5 evaluations (eval every ceil(steps / 5)).
# Each GPU may hold at most 5 GB before a run starts (jeff-serve's 3-4 GB on one GPU is allowed), else nothing starts.
# Writes <W>/runs/RUN/ (train.log = rank 0, rank<i>.log, notes.json, exit), <W>/checkpoints/RUN/, <W>/dist.status.
set -euo pipefail
host=${1:?usage: tonight_dist.sh HOST RUN [RUN ...]}; shift
(( $# >= 1 )) || { echo "usage: tonight_dist.sh HOST RUN [RUN ...]" >&2; exit 2; }
case $host in
  b200)
    W=${W:?set W, the training folder (tonight.sh dist sets it)}
    TRAINER=/raid/work/jeff-first/tonight-train/v13new/trainer
    CODE=/raid/work/jeff-first/tonight-train/jeff-9f84686  # git checkout of 9f84686 (from trainer/jeff-9f84686.bundle)
    PY=/raid/work/jeff-first/tonight-train/jeff-dev-873bafd/.venv/bin/python  # same package versions as casdgx01's venv (trainer/casdgx01-venv-freeze.txt)
    BASE=/raid/work/jeff-first/jeff-serve/base/ll-v12-final
    export LD_LIBRARY_PATH=/raid/work/jeff-first/cuda-compat
    ;;
  casdgx01)
    W=${W:?set W, the training folder (tonight.sh dist sets it)}
    TRAINER=$HOME/v13new/trainer
    CODE=/raid/work/experiments/jeff
    PY=/raid/work/experiments/jeff/.venv/bin/python
    BASE=$HOME/v13base/ll-v12-final
    ;;
  *) echo "unknown host $host" >&2; exit 2 ;;
esac
GPUS=(0 1 2 3 4 5 6 7)
WORLD=${#GPUS[@]}
[[ $(sha256sum "$TRAINER/src/jeff/train.py" | cut -d' ' -f1) == 802c2a6fc5adb00896ce615ea0ba4c827eea8d73ea2f9bb5523497bd0586af32 ]] \
  || { echo "$TRAINER/src/jeff/train.py is not the approved trainer (sha256 802c2a6f...)" >&2; exit 1; }
[[ $(git -C "$CODE" rev-parse HEAD) == $(cat "$TRAINER/COPIED_FROM_COMMIT") ]] || { echo "$CODE is not at the trainer's commit" >&2; exit 1; }
status=$W/dist.status
pids=()
cleanup() {
  local code=$?
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done  # our own rank processes only, by PID
  (( code == 0 )) || echo "failed $(date -Is) exit $code" >> "$status"
}
trap cleanup EXIT

run_one() {
  local run=$1 data recipe effective
  case $run in
    step-curriculum|router|trim)
      data=$run effective=64
      recipe=(--lora-rank 16 --lora-alpha 32 --lora-dropout 0 --lr 2e-4 --readout-lr 5e-6 --resume-every 100) ;;
    full-5e-6)
      data=full effective=256
      recipe=(--lr 5e-6 --resume-every 50) ;;
    *) echo "unknown run $run" >&2; exit 2 ;;
  esac
  local dir=$W/data/$data out=$W/runs/$run ckpt=$W/checkpoints/$run
  [[ ! -e $out && ! -e $ckpt ]] || { echo "$out or $ckpt exists" >&2; exit 1; }
  for g in "${GPUS[@]}"; do
    local used; used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    (( used < 5000 )) || { echo "GPU $g holds $used MiB; stop what runs there first" >&2; exit 1; }
  done
  local rows steps every
  rows=$(grep -c . "$dir/train.jsonl")
  steps=$(( (rows + effective - 1) / effective ))
  every=$(( (steps + 4) / 5 ))
  mkdir -p "$out" "$W/checkpoints"
  cat > "$out/notes.json" <<EOF
{"run": "$run", "host": "$host", "trainer": "~/v13new/trainer on casdgx01 (copied to $TRAINER): jeff commit 9f84686 + trainer/PATCH.diff, train.py sha256 802c2a6fc5adb00896ce615ea0ba4c827eea8d73ea2f9bb5523497bd0586af32", "jeff_dev": "7714a01", "code_dir": "$CODE", "python": "$PY", "base": "$BASE", "gpus": "${GPUS[*]}", "world_size": $WORLD, "batch_per_gpu": $((effective / WORLD)), "effective_batch": $effective, "train_rows": $rows, "optimizer_steps": $steps, "eval_every": $every, "data": "$dir", "started": "$(date -Is)"}
EOF
  local command=("$PY" -m jeff.train --train "$dir/train.jsonl" --development "$dir/development.jsonl"
    --temperature "$dir/temperature.jsonl" --run "$out" --output "$ckpt"
    --base-model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --initial-checkpoint "$BASE"
    --epochs 1 --seed 20260920 --weight-decay 0.01 --batch-size $((effective / WORLD)) --effective-batch-size "$effective"
    --token-budget 8192 --max-length 8192 --cpu-threads 16 --eval-every "$every" --public-eval-every 1000000
    --prompt-layout live-last --keep-stage-order --lr-shape wsd --decay-start 0.8 "${recipe[@]}")
  echo "running $run $(date -Is): $rows rows, $steps steps, eval every $every, gpus ${GPUS[*]}" >> "$status"
  cd "$CODE"
  export JEFF_WORLD_SIZE=$WORLD JEFF_DIST_URL=tcp://127.0.0.1:$((20000 + RANDOM % 20000)) PYTHONPATH=$TRAINER/src \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  pids=()
  local i
  for i in $(seq 1 $((WORLD - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} JEFF_RANK=$i JEFF_EVENTS=$out/events-rank$i.jsonl "${command[@]}" > "$out/rank$i.log" 2>&1 &
    pids+=($!)
  done
  CUDA_VISIBLE_DEVICES=${GPUS[0]} JEFF_RANK=0 JEFF_EVENTS=$out/events.jsonl "${command[@]}" > "$out/train.log" 2>&1 &
  local main=$!
  pids+=($main)
  echo "pids $run ${pids[*]}" >> "$status"
  while kill -0 "$main" 2>/dev/null; do
    for i in $(seq 0 $((WORLD - 2))); do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        local code=0; wait "${pids[$i]}" || code=$?
        (( code == 0 )) || { echo "$code" > "$out/exit"; echo "rank $((i + 1)) of $run exited with $code (see $out/rank$((i + 1)).log)" >&2; exit 1; }
      fi
    done
    sleep 10
  done
  local code=0; wait "$main" || code=$?
  for i in $(seq 0 $((WORLD - 2))); do wait "${pids[$i]}" || code=$?; done
  pids=()
  echo "$code" > "$out/exit"
  (( code == 0 )) || { echo "$run failed with exit $code (see $out/train.log)" >&2; exit 1; }
  echo "done $run $(date -Is)" >> "$status"
}

for run in "$@"; do run_one "$run"; done
