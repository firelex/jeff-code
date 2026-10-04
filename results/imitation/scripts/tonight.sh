#!/usr/bin/env bash
# Tonight's Jeff training data and trainings (2026-10-04), one command per step, run from the Mac in this order:
#
#   tonight.sh convert   stage 3 rows from every collection folder of builds 6498fcf8d and 8db5381f3 (same menu code)
#                        on both hosts, in parallel, on CPUs (nice 10); waits for: the collection to have ended (or
#                        accepts a snapshot: unfinished trials are converted as "cut"). ~5 min.
#   tonight.sh export    pulls the routing and trim label files, joins them onto the stage 3 rows, and exports the step
#                        (stage 3), router and trim rows; waits for: convert, and the labelling cut-off (the labellers
#                        may keep running; whatever is labelled when this runs is used).
#   tonight.sh cut       cuts stage 2, stage 3, router and trim exports to Jeff's 8,192 tokens (jeff_prompt.py
#                        fit-examples --unfittable leave-out); waits for: export.
#   tonight.sh build     the training files (tonight_training_files.py) and their copy to the B200; waits for: cut.
#   tonight.sh train RUN GPU
#                        starts one training on the B200 in tmux (-L default, session train-RUN); RUN is one of
#                        step-curriculum, step-stage3, router, trim, full-2e-6, full-5e-6, full-1e-5; waits for: build
#                        and a free GPU (a stopped jeff-qwen-b200-N container).
#   tonight.sh check RUN GPU
#                        one optimizer step of RUN (train.py --stop-after 1) with timing; same inputs as train.
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
  rsync -a "$B200:$B200_W/stage3/" "$T/stage3/b200/"
  rsync -a -e "ssh -o ControlPath=none" "mstrasser@casdgx01:/home/mstrasser/$CAS_W/stage3/" "$T/stage3/casdgx01/"
  uv run -q --no-project python "$REPO/results/imitation/scripts/tonight_training_files.py" merge-rows \
    --out "$T/stage3/rows.jsonl" "$T/stage3/b200/rows.jsonl" "$T/stage3/casdgx01/rows.jsonl"
}

case "${1:-}" in
  convert) convert ;;
  *) die "usage: tonight.sh convert|export|cut|build|train RUN GPU|check RUN GPU" ;;
esac
