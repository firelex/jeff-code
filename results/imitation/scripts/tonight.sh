#!/usr/bin/env bash
# Tonight's Jeff training data and trainings (2026-10-04), one command per step, run from the Mac in this order:
#
#   tonight.sh convert   stage 3 rows from every collection folder of builds 6498fcf8d and 8db5381f3 (same menu code)
#                        on both hosts, in parallel, on CPUs (nice 10), then copied and merged on the Mac; waits for:
#                        the collection to have ended (a snapshot also works: unfinished trials are converted as "cut").
#                        Trial run 17:30: 1 min per host, ~3 min copy.
#   tonight.sh export    pulls the routing and trim label files, joins them onto the stage 3 states, and exports the
#                        step (stage 3), router and trim rows; waits for: convert and the labelling cut-off (whatever is
#                        labelled when it runs is used; the labellers may keep running). ~1 min.
#   tonight.sh cut       cuts stage 2, stage 3, router and trim exports to Jeff's 8,192 tokens (jeff_prompt.py
#                        fit-examples --unfittable leave-out); waits for: export. ~1.5 min.
#   tonight.sh build     the training files (tonight_training_files.py) and their copy to the B200; waits for: cut.
#                        ~1.5 min.
#   tonight.sh train RUN GPU
#                        starts one training on the B200 in tmux (default server, session train-RUN); RUN is one of
#                        step-curriculum, step-stage3, router, trim, full-2e-6, full-5e-6, full-1e-5; waits for: build
#                        and a free GPU (its jeff-qwen-b200-GPU container stopped; the script refuses a GPU with more
#                        than 20 GB in use).
#   tonight.sh train-all after `docker stop` of jeff-qwen-b200-0..7: the 7 runs on GPUs 0-6 (trim on GPU 5 next to
#                        jeff-serve), then a speed report after SPEED_AFTER seconds (default 900).
#   tonight.sh speed     per run: step, tokens/s over the last 20 steps, projected finish (tonight_speed.py).
#   tonight.sh restart-qwen
#                        after the trainings: starts the 8 Qwen containers (refuses while a GPU holds > 5 GB) and
#                        checks each port answers /health and one short chat request.
#   tonight.sh check RUN GPU
#                        the same training stopped after CHECK_STEPS optimizer steps (default 3; train.py --stop-after),
#                        with the evaluations before and after: gives the time per step and per evaluation.
#                        SHARED_GPU_CHECK=1 lets a check run next to a Qwen server (timing then overstated).
#
# Everything lands in $T on the Mac and in tonight/ folders on the hosts. Nothing is overwritten: each step refuses to
# run when its output exists (delete it by hand to redo a step).
set -euo pipefail

REPO=/Users/mattsinalco/mathias/apps/jeff-pi
JEFF_DEV=/Users/mattsinalco/mathias/apps/jeff-dev
T=${TONIGHT:-/private/tmp/claude-501/imitation/tonight}
B200=mstrasser@85.13.211.160
CAS=(ssh -o ControlPath=none mstrasser@casdgx01)
# TONIGHT and NAME may be set for a trial run (e.g. TONIGHT=/private/tmp/claude-501/imitation/tonight-test NAME=tonight-test).
NAME=${NAME:-tonight}
B200_W=/raid/work/jeff-first/$NAME
CAS_W=jeff-first/$NAME  # under the home folder
PROCESSOR=/private/tmp/claude-501/stage3/tokenizer/Qwen3.5-0.8B
STAGE1_CUT=/private/tmp/claude-501/imitation/export/stage1-replayed-all-cut
STAGE2_UNCUT=/private/tmp/claude-501/imitation/export/stage2
BUILDS=(--build jeff-pi-scout-6498fcf8d.tgz --build jeff-pi-scout-8db5381f3.tgz)
SEED=20260920
# Development and temperature rows per decision (train.py scores all of them at each of the ~20 evaluations).
EVAL_ROWS=${EVAL_ROWS:-1000}
# Tonight's proposal (pending the owner): stage 3 shown once, stage 1 sampled to half of stage 3's rows.
STAGE3_TIMES=${STAGE3_TIMES:-1}
STAGE1_FRACTION=${STAGE1_FRACTION:-0.5}

die() { echo "tonight.sh: $*" >&2; exit 1; }
refuse_existing() { for p in "$@"; do [ ! -e "$p" ] || die "$p exists; delete it to redo this step"; done; }

# --- convert -------------------------------------------------------------------------------------------------------
host_convert() {  # LABEL TMUX WORKDIR UV_CACHE RUN...   (TMUX: the tmux command on that host)
  local label=$1 tmux=$2 work=$3 cache=$4; shift 4
  local sha; sha=$(git -C "$REPO" rev-parse --short=9 HEAD)
  local ssh; if [ "$label" = b200 ]; then ssh=(ssh "$B200"); else ssh=("${CAS[@]}"); fi
  git -C "$REPO" archive --format=tar HEAD tools/jeff-first results/imitation packages/coding-agent/src/core/jeff-first \
    | "${ssh[@]}" "mkdir -p $work/repo-$sha && tar -x -C $work/repo-$sha"
  "${ssh[@]}" "test ! -e $work/stage3 || { echo '$label: $work/stage3 exists' >&2; exit 1; }
    mkdir -p $work/stage3 && cd $work/repo-$sha/tools/jeff-first && $tmux new-session -d -s tonight-stage3 \
    \"export PATH=/raid/work/jeff-first/bin:/snap/bin:\\\$PATH UV_CACHE_DIR=$cache; nice -n 10 uv run -q --no-project python -m imitation.stage3 $* \
       --tasks ../../results/imitation/training-tasks.json --task-sets ../../results/imitation/task-sets.json ${BUILDS[*]} \
       --rows $work/stage3/rows.jsonl --summary $work/stage3/summary.json --stats $work/stage3/stats.md \
       > $work/stage3/out.log 2> $work/stage3/err.log; echo \\\$? > $work/stage3/exit\""
  echo "$label: stage 3 conversion started (tmux session tonight-stage3, $work/stage3)"
}

wait_exit() {  # LABEL SSH... -- FILE
  local label=$1; shift
  local ssh=(); while [ "$1" != -- ]; do ssh+=("$1"); shift; done; shift
  until "${ssh[@]}" "test -e $1"; do sleep 20; done
  local code; code=$("${ssh[@]}" "cat $1")
  [ "$code" = 0 ] || { "${ssh[@]}" "tail -5 $(dirname "$1")/err.log" >&2; die "$label: failed (exit $code)"; }
  echo "$label: done"
}

convert() {
  refuse_existing "$T/stage3"
  local C=/raid/work/jeff-first/collect D=/raid/work/jeff-first/routing/datigator-src
  host_convert b200 tmux "$B200_W" /raid/work/jeff-first/uv-cache \
    $C/completed-tb2/runs-collect-xhigh $C/completed-tb2/runs-collect-xhigh2 $C/runs-collect-xhigh-hub \
    $C/runs-hub-smoke $C/runs-swe-check2 \
    $D/completed-tb2/runs-collect-xhigh $D/completed-tb2/runs-collect-xhigh2 $D/runs-collect-xhigh-hub
  local H=/home/mstrasser/jeff-first/collect
  host_convert casdgx01 "tmux -L jeffcollect" "/home/mstrasser/$CAS_W" /home/mstrasser/.cache/uv \
    $H/completed-tb2/runs-collect-xhigh $H/completed-tb2/runs-collect-xhigh2 $H/runs-collect-xhigh-hub
  wait_exit b200 ssh "$B200" -- "$B200_W/stage3/exit"
  wait_exit casdgx01 "${CAS[@]}" -- "/home/mstrasser/$CAS_W/stage3/exit"
  mkdir -p "$T/stage3"
  rsync -az "$B200:$B200_W/stage3/" "$T/stage3/b200/"
  rsync -az -e "ssh -o ControlPath=none" "mstrasser@casdgx01:/home/mstrasser/$CAS_W/stage3/" "$T/stage3/casdgx01/"
  uv run -q --no-project python "$REPO/results/imitation/scripts/tonight_training_files.py" merge-rows \
    --out "$T/stage3/rows.jsonl" "$T/stage3/b200/rows.jsonl" "$T/stage3/casdgx01/rows.jsonl"
}

# --- export --------------------------------------------------------------------------------------------------------
export_all() {
  [ -s "$T/stage3/rows.jsonl" ] || die "$T/stage3/rows.jsonl is missing: run convert first"
  refuse_existing "$T/labels" "$T/export"
  mkdir -p "$T/labels/routing" "$T/labels/trim" "$T/export"
  # Routing labels: the B200's own (NVFP4), datigator's (its own file, which began as a copy of the B200's earlier
  # datigator file labels/routing-labels-datigator.jsonl, so that one is not read again) and casdgx01's (FP8).
  rsync -az "$B200:/raid/work/jeff-first/routing/labels/routing-labels-b200.jsonl" "$T/labels/routing/"
  rsync -az "$B200:/raid/work/jeff-first/routing/labels-datigator/routing-labels-datigator.jsonl" "$T/labels/routing/"
  rsync -az -e "ssh -o ControlPath=none" "mstrasser@casdgx01:jeff-first/routing/labels/routing-labels-casdgx01.jsonl" "$T/labels/routing/"
  rsync -az "$B200:/raid/work/jeff-first/trim/labels-v2/trim-labels-b200.jsonl" "$B200:/raid/work/jeff-first/trim/labels-v2/trim-labels-datigator.jsonl" "$T/labels/trim/"
  rsync -az -e "ssh -o ControlPath=none" "mstrasser@casdgx01:jeff-first/trim/labels-v2/trim-labels-casdgx01.jsonl" "$T/labels/trim/"
  local S="$REPO/results/imitation/scripts" R="$REPO/results/imitation"
  local UVW=(uv run -q --no-project --with aiohttp==3.12.15 --with tokenizers python)
  # A labeller appends while we copy: a cut last line is dropped here (the copy's last line may be unfinished).
  for f in "$T"/labels/routing/*.jsonl "$T"/labels/trim/*.jsonl; do
    tail -c 1 "$f" | od -An -c | grep -q '\\n' || { sed -i '' '$d' "$f"; echo "$f: unfinished last line dropped"; }
  done
  # States for the router and trim rows: this conversion's rows only. Labelled turns of the build 4abde3ece sessions
  # (runs-imitation-v4/v5) find no state and are left out (controller ruling: they ran at thinking medium, so their
  # labels do not answer "as good as the recorded xhigh action").
  (cd "$S" && "${UVW[@]}" routing_labels.py join --labels "$T"/labels/routing/*.jsonl --stage3-rows "$T/stage3/rows.jsonl" \
      --out "$T/export/routing-rows.jsonl") | tee "$T/export/routing-join.txt"
  (cd "$S" && "${UVW[@]}" trim_labels.py join --labels "$T"/labels/trim/*.jsonl --stage3-rows "$T/stage3/rows.jsonl" \
      --out "$T/export/trim-rows.jsonl") | tee "$T/export/trim-join.txt"
  cd "$REPO/tools/jeff-first"
  uv run -q python -m imitation.export_jeff export --rows "$T/stage3/rows.jsonl" --splits "$R/splits.json" \
    --task-sets "$R/task-sets.json" --out "$T/export/stage3" | tee "$T/export/stage3-counts.json"
  uv run -q python -m imitation.export_routing --rows "$T/export/routing-rows.jsonl" --splits "$R/splits.json" \
    --task-sets "$R/task-sets.json" --out "$T/export/router" | tee "$T/export/router-counts.json"
  uv run -q python -m imitation.export_trim --rows "$T/export/trim-rows.jsonl" --splits "$R/splits.json" \
    --task-sets "$R/task-sets.json" --out "$T/export/trim" | tee "$T/export/trim-counts.json"
}

# --- cut -----------------------------------------------------------------------------------------------------------
cut_all() {
  refuse_existing "$T/cut"
  mkdir -p "$T/cut"
  local name src
  for name in stage2 stage3 router trim; do
    if [ $name = stage2 ]; then src=$STAGE2_UNCUT; else src=$T/export/$name; fi
    "$JEFF_DEV/.venv/bin/python" "$REPO/tools/jeff-first/jeff_prompt.py" fit-examples --processor "$PROCESSOR" \
      --layout live-last --workers 8 --unfittable leave-out --out "$T/cut/$name" \
      "$src/train.jsonl" "$src/development.jsonl" "$src/temperature.jsonl" | tee "$T/cut/$name.log"
  done
}

# --- build ---------------------------------------------------------------------------------------------------------
build_files() {
  refuse_existing "$T/train"
  local R="$REPO/results/imitation"
  [ -s "$T/scoring-tasks.txt" ] || rsync -az "$B200:/raid/work/jeff-first/collect/hub/scoring-tasks.txt" "$T/scoring-tasks.txt"
  uv run -q --no-project python "$R/scripts/tonight_training_files.py" build --stage1 "$STAGE1_CUT" \
    --stage2 "$T/cut/stage2" --stage3 "$T/cut/stage3" --router "$T/cut/router" --trim "$T/cut/trim" \
    --task-sets "$R/task-sets.json" --training-tasks "$R/training-tasks.json" --scoring-tasks "$T/scoring-tasks.txt" \
    --seed "$SEED" --eval-rows "$EVAL_ROWS" --stage3-times "$STAGE3_TIMES" --stage1-fraction "$STAGE1_FRACTION" \
    --out "$T/train"
  ssh "$B200" "test ! -e $TRAIN_W/data || { echo '$TRAIN_W/data exists' >&2; exit 1; }; mkdir -p $TRAIN_W/data"
  rsync -az "$T/train/" "$B200:$TRAIN_W/data/"
}

# --- train / check (B200) ------------------------------------------------------------------------------------------
# jeff-dev 873bafd (git archive) with its own venv (uv sync --extra lora --extra cuda) in $TRAIN_W/jeff-dev-873bafd;
# the v1.3 base ll-v12-final (model.safetensors sha256 d324dd6c...) as the service uses it.
TRAIN_W=/raid/work/jeff-first/$NAME-train
JD=/raid/work/jeff-first/tonight-train/jeff-dev-873bafd
BASE=/raid/work/jeff-first/jeff-serve/base/ll-v12-final
DECAY_START=0.8
# The v1.3 adapter recipe (casdgx01 ~/v13adapters/lib/train_one.sh) and the ll-v12 full-training recipe
# (casdgx01 /raid/work/experiments/jeff/runs/ll-v12/config.json: batch 32, effective batch 256, weight decay 0.01,
# token budget and max length 8192, live-last, resume every 50), both from ll-v12-final, both with the stage order kept
# and the warm-up/hold/decay learning-rate shape.
COMMON=(--base-model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --initial-checkpoint "$BASE"
        --epochs 1 --seed 20260920 --weight-decay 0.01 --batch-size 32 --token-budget 8192 --max-length 8192
        --cpu-threads 16 --public-eval-every 1000000 --prompt-layout live-last
        --keep-stage-order --lr-shape wsd --decay-start "$DECAY_START")
ADAPTER=(--effective-batch-size 64 --resume-every 100 --lora-rank 16 --lora-alpha 32 --lora-dropout 0 --lr 2e-4 --readout-lr 5e-6)
full_args() { echo --effective-batch-size 256 --resume-every 50 --lr "$1"; }

run_args() {  # RUN -> data folder name and the recipe arguments
  case "$1" in
    step-curriculum|step-stage3|router|trim) echo "$1 ${ADAPTER[*]}" ;;
    full-2e-6) echo "full $(full_args 2e-6)" ;;
    full-5e-6) echo "full $(full_args 5e-6)" ;;
    full-1e-5) echo "full $(full_args 1e-5)" ;;
    *) die "unknown run $1 (step-curriculum, step-stage3, router, trim, full-2e-6, full-5e-6, full-1e-5)" ;;
  esac
}

start_training() {  # RUN GPU MODE (train or check)
  local run=$1 gpu=$2 mode=$3
  [[ $gpu =~ ^[0-7]$ ]] || die "GPU must be 0-7, not $gpu"
  local spec data args; spec=$(run_args "$run"); data=${spec%% *}; args=${spec#* }
  local batch; if [ "$data" = full ]; then batch=256; else batch=64; fi
  local rows; rows=$(ssh "$B200" "wc -l < $TRAIN_W/data/$data/train.jsonl")
  local every=$(( (rows + batch - 1) / batch / 20 )); (( every >= 4 )) || every=4
  local name=$run; local extra=""
  if [ "$mode" = check ]; then name=check-$run-$(date +%H%M); extra="--stop-after ${CHECK_STEPS:-3}"; fi
  ssh "$B200" "test ! -e $TRAIN_W/runs/$name || { echo '$TRAIN_W/runs/$name exists' >&2; exit 1; }
    used=\$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ \$used -gt 20000 ] && [ '$mode${SHARED_GPU_CHECK:-}' != check1 ]; then echo \"GPU $gpu is busy (\$used MiB used; stop its jeff-qwen-b200-$gpu first)\" >&2; exit 1; fi
    mkdir -p $TRAIN_W/runs/$name $TRAIN_W/checkpoints && cd $JD && tmux new-session -d -s train-$name \"
    CUDA_VISIBLE_DEVICES=$gpu LD_LIBRARY_PATH=/raid/work/jeff-first/cuda-compat PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/raid/work/jeff-first/hf-home JEFF_EVENTS=$TRAIN_W/runs/$name/events.jsonl \
    /usr/bin/time -v $JD/.venv/bin/jeff-train --train $TRAIN_W/data/$data/train.jsonl \
      --development $TRAIN_W/data/$data/development.jsonl --temperature $TRAIN_W/data/$data/temperature.jsonl \
      --run $TRAIN_W/runs/$name --output $TRAIN_W/checkpoints/$name ${COMMON[*]} $args --eval-every $every $extra \
      > $TRAIN_W/runs/$name/train.log 2>&1; echo \\\$? > $TRAIN_W/runs/$name/exit\""
  echo "B200 GPU $gpu: $name started ($rows rows, eval every $every steps); log $TRAIN_W/runs/$name/train.log"
}

# --- all runs at once on the B200, speed, Qwen restart -------------------------------------------------------------
# GPU N runs container jeff-qwen-b200-N (port 8885+N). jeff-serve stays on GPU 5 (3.5 GB) next to the trim run.
ALL_RUNS=("step-curriculum 0" "step-stage3 1" "full-2e-6 2" "full-5e-6 3" "full-1e-5 4" "trim 5" "router 6")
QWEN=(jeff-qwen-b200-0 jeff-qwen-b200-1 jeff-qwen-b200-2 jeff-qwen-b200-3 jeff-qwen-b200-4 jeff-qwen-b200-5
      jeff-qwen-b200-6 jeff-qwen-b200-7)

train_all() {
  local running; running=$(ssh "$B200" "docker ps --format '{{.Names}}' | grep -E '^jeff-qwen-b200-[0-7]\$' || true")
  [ -z "$running" ] || die "Qwen containers still running (stop them first: ssh $B200 'docker stop ${QWEN[*]}'): $running"
  local pair
  for pair in "${ALL_RUNS[@]}"; do start_training ${pair% *} ${pair#* } train; done
  echo "speed report in ${SPEED_AFTER:-900} s (or any time: tonight.sh speed)"
  sleep "${SPEED_AFTER:-900}"
  speed
}

speed() {
  ssh "$B200" "python3 - $TRAIN_W/runs" < "$REPO/results/imitation/scripts/tonight_speed.py"
}

restart_qwen() {
  # vLLM takes 90% of a GPU's memory: only jeff-serve (GPU 5) may hold memory when the servers start.
  local busy; busy=$(ssh "$B200" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '\$2 > 5000 {print \$1}'")
  [ -z "$busy" ] || die "GPUs still in use (a training running?): $busy"
  ssh "$B200" "docker start ${QWEN[*]}"
  local port
  for port in 8885 8886 8887 8888 8889 8890 8891 8892; do
    local waited=0
    until ssh "$B200" "curl -sf -m 5 http://192.168.3.12:$port/health > /dev/null"; do
      sleep 15; waited=$((waited + 15)); (( waited < 1200 )) || die "port $port not healthy after 20 min"
    done
    ssh "$B200" "curl -sf -m 120 http://192.168.3.12:$port/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":false}}'" \
      | python3 -c "import json,sys; r=json.load(sys.stdin); print('$port serves:', repr(r['choices'][0]['message']['content']))"
  done
}

case "${1:-}" in
  convert) convert ;;
  export) export_all ;;
  cut) cut_all ;;
  build) build_files ;;
  train) start_training "${2:?run}" "${3:?gpu}" train ;;
  check) start_training "${2:?run}" "${3:?gpu}" check ;;
  train-all) train_all ;;
  speed) speed ;;
  restart-qwen) restart_qwen ;;
  *) die "usage: tonight.sh convert|export|cut|build|train RUN GPU|check RUN GPU|train-all|speed|restart-qwen" ;;
esac
