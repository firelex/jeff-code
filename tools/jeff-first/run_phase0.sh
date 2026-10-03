#!/usr/bin/env bash
# Phase-0 shadow run: plain pi (the jeff-pi fork in shadow mode) with a chosen model on the phase-0 tasks.
# One Harbor run per task, so each trace line carries its own task id; CONCURRENCY tasks run at a time.
# Each task's trace lands in JOBS_DIR/<job>/<trial>/agent/jeff-first-trace.jsonl.
#
# Usage: run_phase0.sh [--dry-run] TASKS_JSON TARBALL BASE_URL JOBS_DIR CONCURRENCY THINKING TOOLS MODEL MODE
#                      TIMEOUT_MULTIPLIER [TASK ...]
#   TASKS_JSON   results/phase0/tasks.json (its "phase0" list is used unless TASK names are given)
#   TARBALL      the packed fork (npm pack in packages/coding-agent)
#   BASE_URL     the OpenAI-compatible endpoint as the task containers reach it, without /v1: for Qwen the sparkgate
#                proxy, e.g. http://192.168.0.79:8899; for GLM the GLM proxy (glm_proxy.py)
#   THINKING     pi's thinking level for the model: off, minimal, low, medium or high; when not off, the environment
#                variable JEFF_RUN_THINKING_FORMAT must be set to how pi switches the model's thinking on (pi's
#                compat.thinkingFormat), e.g. qwen-chat-template, since Harbor's model entry does not say the model
#                can reason; also when not off, the environment variable JEFF_RUN_MAX_OUTPUT_TOKENS must be set to
#                the most tokens the model may write in one turn, thinking included, as a positive whole number,
#                e.g. 65536, since pi's default of 16,384 tokens is too small for a thinking model
#   TOOLS        pi's tool list, e.g. read,bash,edit,write,grep,find,ls; "default" keeps pi's own (read,bash,edit,write)
#   MODEL        the model id at BASE_URL, e.g. qwen3.8-flash-next or scissero-glm-5.3
#   MODE         shadow (log what the model does), teacher (the teacher model scouts before every model turn;
#                needs JEFF_FIRST_TEACHER_URL, the GLM proxy as containers reach it, JEFF_FIRST_TEACHER_MODEL,
#                JEFF_FIRST_RUN_APPROVAL (all, seen or never) and JEFF_FIRST_DRIVER_BUILD, e.g.
#                qwen3.8-27b-nvfp4@spark-head) or record (plain Qwen works alone; at every turn, logs the scout's
#                full option lists for later labelling; needs JEFF_FIRST_RUN_APPROVAL and JEFF_FIRST_DRIVER_BUILD,
#                same meaning as in teacher mode, but no teacher URL or model)
#   TIMEOUT_MULTIPLIER  positive number that multiplies each task's agent time limit (Harbor's
#                --agent-timeout-multiplier); use the same value in both Gate 0 arms and make it large enough that
#                the teacher's (GLM's) latency never decides a task through the time limit
#
# The environment variable JEFF_RUN_API_KEY must be "unused": task containers never hold a real key. Every model is
# reached through a proxy, as driver (BASE_URL) or as teacher (JEFF_FIRST_TEACHER_URL): Qwen through the sparkgate
# proxy, GLM through glm_proxy.py, which adds the real key outside the containers.
#
# Run it on datigator in tmux session jeff-pi-phase0; stop it with: tmux kill-session -t jeff-pi-phase0
set -euo pipefail

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then dry_run=1; shift; fi
if [ $# -lt 10 ]; then
  sed -n '5,32p' "$0" >&2
  exit 2
fi
tasks_json=$1 tarball=$2 base_url=$3 jobs=$4 concurrency=$5 thinking=$6 tools=$7 model=$8 mode=$9 timeout_multiplier=${10}
shift 10
[ "${JEFF_RUN_API_KEY:-}" = unused ] || { echo "JEFF_RUN_API_KEY must be \"unused\": containers never hold a real key; reach GLM through glm_proxy.py" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)

[ -f "$tasks_json" ] || { echo "no tasks file at $tasks_json" >&2; exit 1; }
[ -f "$tarball" ] || { echo "no fork tarball at $tarball; build it with npm pack" >&2; exit 1; }
case "$thinking" in off|minimal|low|medium|high) ;; *) echo "THINKING must be off, minimal, low, medium or high" >&2; exit 2 ;; esac
if [ "$thinking" != off ]; then
  [ -n "${JEFF_RUN_THINKING_FORMAT:-}" ] || { echo "THINKING $thinking needs JEFF_RUN_THINKING_FORMAT (how pi switches thinking on, e.g. qwen-chat-template)" >&2; exit 2; }
  case "${JEFF_RUN_MAX_OUTPUT_TOKENS:-}" in ''|0|*[!0-9]*) echo "THINKING $thinking needs JEFF_RUN_MAX_OUTPUT_TOKENS (a positive whole number, e.g. 65536)" >&2; exit 2 ;; esac
fi
case "$concurrency" in ''|*[!0-9]*) echo "CONCURRENCY must be a whole number" >&2; exit 2 ;; esac
[[ "$timeout_multiplier" =~ ^[0-9]*\.?[0-9]+$ && "$timeout_multiplier" =~ [1-9] ]] \
  || { echo "TIMEOUT_MULTIPLIER must be a positive number, e.g. 3 or 2.5 (got \"$timeout_multiplier\")" >&2; exit 2; }
case "$mode" in
  shadow) ;;
  teacher)
    [ -n "${JEFF_FIRST_TEACHER_URL:-}" ] || { echo "MODE teacher needs JEFF_FIRST_TEACHER_URL (the GLM proxy)" >&2; exit 2; }
    [ -n "${JEFF_FIRST_TEACHER_MODEL:-}" ] || { echo "MODE teacher needs JEFF_FIRST_TEACHER_MODEL" >&2; exit 2; }
    case "${JEFF_FIRST_RUN_APPROVAL:-}" in all|seen|never) ;; *) echo "MODE teacher needs JEFF_FIRST_RUN_APPROVAL: all, seen or never" >&2; exit 2 ;; esac
    [ -n "${JEFF_FIRST_DRIVER_BUILD:-}" ] || { echo "MODE teacher needs JEFF_FIRST_DRIVER_BUILD, e.g. qwen3.8-27b-nvfp4@spark-head" >&2; exit 2; }
    ;;
  record)
    case "${JEFF_FIRST_RUN_APPROVAL:-}" in all|seen|never) ;; *) echo "MODE record needs JEFF_FIRST_RUN_APPROVAL: all, seen or never" >&2; exit 2 ;; esac
    [ -n "${JEFF_FIRST_DRIVER_BUILD:-}" ] || { echo "MODE record needs JEFF_FIRST_DRIVER_BUILD, e.g. qwen3.8-27b-nvfp4@spark-head" >&2; exit 2; }
    ;;
  *) echo "MODE must be shadow, teacher or record" >&2; exit 2 ;;
esac
export JEFF_FIRST_TEACHER_URL="${JEFF_FIRST_TEACHER_URL:-}" JEFF_FIRST_TEACHER_MODEL="${JEFF_FIRST_TEACHER_MODEL:-}"
export JEFF_RUN_THINKING_FORMAT="${JEFF_RUN_THINKING_FORMAT:-}" JEFF_RUN_MAX_OUTPUT_TOKENS="${JEFF_RUN_MAX_OUTPUT_TOKENS:-}" JEFF_FIRST_RUN_APPROVAL="${JEFF_FIRST_RUN_APPROVAL:-}" JEFF_FIRST_DRIVER_BUILD="${JEFF_FIRST_DRIVER_BUILD:-}"

if [ $# -gt 0 ]; then
  tasks=("$@")
else
  mapfile -t tasks < <(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["phase0"]))' "$tasks_json")
fi
mkdir -p "$jobs/logs"
# Harbor runs from this folder (so it can import harbor_agent), so every path must be absolute.
tarball=$(cd "$(dirname "$tarball")" && pwd)/$(basename "$tarball")
jobs=$(cd "$jobs" && pwd)

run_one() {
  local task=$1
  local job_name
  job_name="$task-$(date +%Y%m%d-%H%M%S)"
  local command=(
    uv run --project "$here" harbor run
    --dataset terminal-bench@2.0 -i "$task" -n 1
    -a harbor_agent.jeff_pi:JeffPi
    --ak "tarball=$tarball" --ak model_api=openai-completions --ak "thinking=$thinking"
    -m "openai/$model"
    --agent-timeout-multiplier "$timeout_multiplier"
  )
  if [ "$tools" != default ]; then command+=(--ak "tools=$tools"); fi
  if [ "$thinking" != off ]; then
    command+=(--ak "thinking_format=$JEFF_RUN_THINKING_FORMAT" --ak "max_output_tokens=$JEFF_RUN_MAX_OUTPUT_TOKENS")
  fi
  command+=(
    --ae "JEFF_FIRST_MODE=$mode"
    --ae "JEFF_FIRST_TASK_ID=$task"
    --ae JEFF_FIRST_TRACE_FILE=/logs/agent/jeff-first-trace.jsonl
    -o "$jobs" --job-name "$job_name"
  )
  if [ "$mode" = teacher ]; then
    command+=(
      --ae "JEFF_FIRST_TEACHER_URL=$JEFF_FIRST_TEACHER_URL" --ae "JEFF_FIRST_TEACHER_MODEL=$JEFF_FIRST_TEACHER_MODEL"
      --ae "JEFF_FIRST_RUN_APPROVAL=$JEFF_FIRST_RUN_APPROVAL" --ae "JEFF_FIRST_DRIVER_BUILD=$JEFF_FIRST_DRIVER_BUILD"
    )
  elif [ "$mode" = record ]; then
    command+=(
      --ae "JEFF_FIRST_RUN_APPROVAL=$JEFF_FIRST_RUN_APPROVAL" --ae "JEFF_FIRST_DRIVER_BUILD=$JEFF_FIRST_DRIVER_BUILD"
    )
  fi
  if [ "$dry_run" = 1 ]; then
    printf '%q ' "${command[@]}"; echo
    return
  fi
  echo "$(date -Is) start $task"
  # Harbor exits 0 even when a trial failed, so check_trial.py inspects the job's results afterwards.
  local status=0
  (cd "$here" && PYTHONPATH="$here" OPENAI_BASE_URL="$base_url/v1" OPENAI_API_KEY="$JEFF_RUN_API_KEY" "${command[@]}") > "$jobs/logs/$task.log" 2>&1 \
    && (cd "$here" && uv run --project "$here" python check_trial.py "$jobs/$job_name" $( [ "$thinking" != off ] && echo --expect-thinking )) >> "$jobs/logs/$task.log" 2>&1 \
    || status=$?
  if [ "$status" = 0 ]; then
    echo "$(date -Is) done  $task ($(tail -1 "$jobs/logs/$task.log"))"
  else
    echo "$(date -Is) FAILED $task (exit $status; see $jobs/logs/$task.log)"
    return 1
  fi
}
export -f run_one
export here tarball base_url jobs thinking tools model mode timeout_multiplier dry_run JEFF_RUN_API_KEY \
  JEFF_RUN_THINKING_FORMAT JEFF_RUN_MAX_OUTPUT_TOKENS JEFF_FIRST_RUN_APPROVAL JEFF_FIRST_DRIVER_BUILD

# xargs keeps going after a failed task and exits non-zero at the end; each failure is printed above.
if ! printf '%s\n' "${tasks[@]}" | xargs -P "$concurrency" -I{} bash -c 'run_one "$1"' _ {}; then
  echo "one or more tasks FAILED; their logs are in $jobs/logs" >&2
  exit 1
fi
