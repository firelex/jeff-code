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
#   TOOLS        pi's tool list, e.g. read,bash,edit,write,grep,find,ls; "default" keeps pi's own (read,bash,edit,write);
#                use bash for imitation runs (the coding model works with bash alone, and every scout option is bash)
#   MODEL        the model id at BASE_URL, e.g. qwen3.8-flash-next or scissero-glm-5.3
#   MODE         shadow (log what the model does), teacher (the teacher model scouts before every model turn;
#                needs JEFF_FIRST_TEACHER_URL, the GLM proxy as containers reach it, JEFF_FIRST_TEACHER_MODEL,
#                JEFF_FIRST_RUN_APPROVAL (all, seen or never) and JEFF_FIRST_DRIVER_BUILD, e.g.
#                qwen3.8-27b-nvfp4@spark-head) or record (plain Qwen works alone; at every turn, logs the scout's
#                full option lists for later labelling; needs JEFF_FIRST_RUN_APPROVAL and JEFF_FIRST_DRIVER_BUILD,
#                same meaning as in teacher mode, but no teacher URL or model) or jeff (the trained small model Jeff
#                scouts before every model turn, asked through its service; needs JEFF_FIRST_JEFF_URL, the Jeff service
#                (jeff-serve) as containers reach it, e.g. http://192.168.2.10:8920, JEFF_FIRST_JEFF_STEP_ADAPTER, the
#                step adapter's name there ("jeff" = the base model itself), JEFF_FIRST_JEFF_STEP_THRESHOLD, the
#                probability from 0 to 1 Jeff's best option needs or the scout hands over, and JEFF_FIRST_RUN_APPROVAL
#                and JEFF_FIRST_DRIVER_BUILD as in teacher mode); teacher, record and jeff also need
#                JEFF_FIRST_THINKING_ROUTER, how hard Qwen thinks in each request: fixed:off, fixed:low, fixed:medium,
#                fixed:xhigh, jeff:<router adapter> (Jeff's router adapter chooses per request; needs
#                JEFF_FIRST_JEFF_URL and JEFF_FIRST_JEFF_ROUTER_THRESHOLD, the probability from 0 to 1 a level below
#                xhigh needs), or jeff-off-unless:<router adapter>:<threshold> (off unless the adapter's probability
#                for xhigh is at least the threshold, then xhigh; needs JEFF_FIRST_JEFF_URL and no
#                JEFF_FIRST_JEFF_ROUTER_THRESHOLD) (it overrides THINKING per request; THINKING must not be off unless
#                the router is fixed:off, since pi then marks the model as unable to think); and
#                JEFF_FIRST_OUTPUT_TRIM, how much of
#                each new output over 40 lines Qwen sees: off, fixed:all, fixed:last200, fixed:last40, fixed:first40,
#                fixed:first20last20, or jeff:<trimming adapter> (needs JEFF_FIRST_JEFF_URL and
#                JEFF_FIRST_JEFF_TRIM_THRESHOLD, the probability from 0 to 1 a cut needs); and
#                JEFF_FIRST_THINKING_LIMIT, the most thinking tokens one Qwen reply may use before it is cut and
#                continued: off or a whole number, e.g. 8000
#   TIMEOUT_MULTIPLIER  positive number that multiplies each task's agent time limit (Harbor's
#                --agent-timeout-multiplier); use the same value in both Gate 0 arms and make it large enough that
#                the teacher's (GLM's) latency never decides a task through the time limit
#   TASK         a bare name is a Terminal-Bench 2.0 task (terminal-bench@2.0); "<org>/<dataset>:<task>" is a task of
#                a Harbor hub dataset of results/imitation/task-sets.json, run from the dataset's pinned version, with
#                its agent time capped at 15 minutes x TIMEOUT_MULTIPLIER (task_source.py); such tasks need the
#                environment variables JEFF_RUN_TASK_SETS (task-sets.json) and JEFF_RUN_TASK_INVENTORY
#                (task-sets-inventory.json); the full id is the trace's task id
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
  sed -n '5,48p' "$0" >&2
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
# The thinking router of teacher, record and jeff modes: a fixed level, or Jeff's router adapter at its service.
check_router() {
  case "${JEFF_FIRST_THINKING_ROUTER:-}" in
    fixed:off|fixed:low|fixed:medium|fixed:xhigh) ;;
    jeff:?*)
      [ -n "${JEFF_FIRST_JEFF_URL:-}" ] || { echo "JEFF_FIRST_THINKING_ROUTER=$JEFF_FIRST_THINKING_ROUTER needs JEFF_FIRST_JEFF_URL (the Jeff service as containers reach it)" >&2; exit 2; }
      check_threshold JEFF_FIRST_JEFF_ROUTER_THRESHOLD "${JEFF_FIRST_JEFF_ROUTER_THRESHOLD:-}"
      ;;
    jeff-off-unless:?*:*)
      [ -n "${JEFF_FIRST_JEFF_URL:-}" ] || { echo "JEFF_FIRST_THINKING_ROUTER=$JEFF_FIRST_THINKING_ROUTER needs JEFF_FIRST_JEFF_URL (the Jeff service as containers reach it)" >&2; exit 2; }
      check_threshold "the threshold in JEFF_FIRST_THINKING_ROUTER=$JEFF_FIRST_THINKING_ROUTER" "${JEFF_FIRST_THINKING_ROUTER##*:}"
      [ -z "${JEFF_FIRST_JEFF_ROUTER_THRESHOLD:-}" ] || { echo "JEFF_FIRST_THINKING_ROUTER=$JEFF_FIRST_THINKING_ROUTER holds its own threshold; JEFF_FIRST_JEFF_ROUTER_THRESHOLD must not be set with it" >&2; exit 2; }
      ;;
    *) echo "MODE $mode needs JEFF_FIRST_THINKING_ROUTER: fixed:off, fixed:low, fixed:medium, fixed:xhigh, jeff:<router adapter> or jeff-off-unless:<router adapter>:<threshold>" >&2; exit 2 ;;
  esac
}
# How much of a long new output Qwen sees (teacher, record and jeff modes).
check_trim() {
  case "${JEFF_FIRST_OUTPUT_TRIM:-}" in
    off|fixed:all|fixed:last200|fixed:last40|fixed:first40|fixed:first20last20) ;;
    jeff:?*)
      [ -n "${JEFF_FIRST_JEFF_URL:-}" ] || { echo "JEFF_FIRST_OUTPUT_TRIM=$JEFF_FIRST_OUTPUT_TRIM needs JEFF_FIRST_JEFF_URL (the Jeff service as containers reach it)" >&2; exit 2; }
      check_threshold JEFF_FIRST_JEFF_TRIM_THRESHOLD "${JEFF_FIRST_JEFF_TRIM_THRESHOLD:-}"
      ;;
    *) echo "MODE $mode needs JEFF_FIRST_OUTPUT_TRIM: off, fixed:all, fixed:last200, fixed:last40, fixed:first40, fixed:first20last20 or jeff:<trimming adapter>" >&2; exit 2 ;;
  esac
}
# The thinking limit (teacher, record and jeff modes): off or a whole number of tokens.
check_thinking_limit() {
  [[ "${JEFF_FIRST_THINKING_LIMIT:-}" =~ ^(off|[1-9][0-9]*)$ ]] \
    || { echo "MODE $mode needs JEFF_FIRST_THINKING_LIMIT: off or a whole number of thinking tokens, e.g. 8000 (got \"${JEFF_FIRST_THINKING_LIMIT:-}\")" >&2; exit 2; }
}
check_threshold() {
  [[ "$2" =~ ^(0(\.[0-9]+)?|1(\.0+)?|\.[0-9]+)$ ]] || { echo "$1 must be a number from 0 to 1, e.g. 0.5 (got \"$2\")" >&2; exit 2; }
}
case "$mode" in
  shadow) ;;
  teacher)
    [ -n "${JEFF_FIRST_TEACHER_URL:-}" ] || { echo "MODE teacher needs JEFF_FIRST_TEACHER_URL (the GLM proxy)" >&2; exit 2; }
    [ -n "${JEFF_FIRST_TEACHER_MODEL:-}" ] || { echo "MODE teacher needs JEFF_FIRST_TEACHER_MODEL" >&2; exit 2; }
    case "${JEFF_FIRST_RUN_APPROVAL:-}" in all|seen|never) ;; *) echo "MODE teacher needs JEFF_FIRST_RUN_APPROVAL: all, seen or never" >&2; exit 2 ;; esac
    [ -n "${JEFF_FIRST_DRIVER_BUILD:-}" ] || { echo "MODE teacher needs JEFF_FIRST_DRIVER_BUILD, e.g. qwen3.8-27b-nvfp4@spark-head" >&2; exit 2; }
    check_router
    check_trim
    check_thinking_limit
    ;;
  jeff)
    [ -n "${JEFF_FIRST_JEFF_URL:-}" ] || { echo "MODE jeff needs JEFF_FIRST_JEFF_URL (the Jeff service as containers reach it, e.g. http://192.168.2.10:8920)" >&2; exit 2; }
    [ -n "${JEFF_FIRST_JEFF_STEP_ADAPTER:-}" ] || { echo "MODE jeff needs JEFF_FIRST_JEFF_STEP_ADAPTER (the step adapter's name at the Jeff service; jeff = the base model)" >&2; exit 2; }
    check_threshold JEFF_FIRST_JEFF_STEP_THRESHOLD "${JEFF_FIRST_JEFF_STEP_THRESHOLD:-}"
    case "${JEFF_FIRST_RUN_APPROVAL:-}" in all|seen|never) ;; *) echo "MODE jeff needs JEFF_FIRST_RUN_APPROVAL: all, seen or never" >&2; exit 2 ;; esac
    [ -n "${JEFF_FIRST_DRIVER_BUILD:-}" ] || { echo "MODE jeff needs JEFF_FIRST_DRIVER_BUILD, e.g. qwen3.8-27b-nvfp4@spark-head" >&2; exit 2; }
    check_router
    check_trim
    check_thinking_limit
    ;;
  record)
    case "${JEFF_FIRST_RUN_APPROVAL:-}" in all|seen|never) ;; *) echo "MODE record needs JEFF_FIRST_RUN_APPROVAL: all, seen or never" >&2; exit 2 ;; esac
    [ -n "${JEFF_FIRST_DRIVER_BUILD:-}" ] || { echo "MODE record needs JEFF_FIRST_DRIVER_BUILD, e.g. qwen3.8-27b-nvfp4@spark-head" >&2; exit 2; }
    check_router
    check_trim
    check_thinking_limit
    ;;
  *) echo "MODE must be shadow, teacher, record or jeff" >&2; exit 2 ;;
esac
export JEFF_FIRST_TEACHER_URL="${JEFF_FIRST_TEACHER_URL:-}" JEFF_FIRST_TEACHER_MODEL="${JEFF_FIRST_TEACHER_MODEL:-}"
export JEFF_RUN_THINKING_FORMAT="${JEFF_RUN_THINKING_FORMAT:-}" JEFF_RUN_MAX_OUTPUT_TOKENS="${JEFF_RUN_MAX_OUTPUT_TOKENS:-}" JEFF_FIRST_RUN_APPROVAL="${JEFF_FIRST_RUN_APPROVAL:-}" JEFF_FIRST_DRIVER_BUILD="${JEFF_FIRST_DRIVER_BUILD:-}"
export JEFF_FIRST_THINKING_ROUTER="${JEFF_FIRST_THINKING_ROUTER:-}" JEFF_FIRST_JEFF_URL="${JEFF_FIRST_JEFF_URL:-}"
export JEFF_FIRST_JEFF_STEP_ADAPTER="${JEFF_FIRST_JEFF_STEP_ADAPTER:-}" JEFF_FIRST_JEFF_STEP_THRESHOLD="${JEFF_FIRST_JEFF_STEP_THRESHOLD:-}"
export JEFF_FIRST_JEFF_ROUTER_THRESHOLD="${JEFF_FIRST_JEFF_ROUTER_THRESHOLD:-}"
export JEFF_FIRST_OUTPUT_TRIM="${JEFF_FIRST_OUTPUT_TRIM:-}" JEFF_FIRST_JEFF_TRIM_THRESHOLD="${JEFF_FIRST_JEFF_TRIM_THRESHOLD:-}"
export JEFF_FIRST_THINKING_LIMIT="${JEFF_FIRST_THINKING_LIMIT:-}"

if [ $# -gt 0 ]; then
  tasks=("$@")
else
  mapfile -t tasks < <(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["phase0"]))' "$tasks_json")
fi
# Hub tasks: every id must resolve (known dataset and task, not excluded) before anything starts.
hub_tasks=0
for task in "${tasks[@]}"; do case "$task" in *:*|*/*) hub_tasks=1 ;; esac; done
if [ "$hub_tasks" = 1 ]; then
  [ -f "${JEFF_RUN_TASK_SETS:-}" ] || { echo "hub task ids (<org>/<dataset>:<task>) need JEFF_RUN_TASK_SETS, the path of results/imitation/task-sets.json" >&2; exit 2; }
  [ -f "${JEFF_RUN_TASK_INVENTORY:-}" ] || { echo "hub task ids (<org>/<dataset>:<task>) need JEFF_RUN_TASK_INVENTORY, the path of results/imitation/task-sets-inventory.json" >&2; exit 2; }
  JEFF_RUN_TASK_SETS=$(cd "$(dirname "$JEFF_RUN_TASK_SETS")" && pwd)/$(basename "$JEFF_RUN_TASK_SETS")
  JEFF_RUN_TASK_INVENTORY=$(cd "$(dirname "$JEFF_RUN_TASK_INVENTORY")" && pwd)/$(basename "$JEFF_RUN_TASK_INVENTORY")
  python3 "$here/task_source.py" check "$JEFF_RUN_TASK_SETS" "$JEFF_RUN_TASK_INVENTORY" "${tasks[@]}"
fi
export JEFF_RUN_TASK_SETS="${JEFF_RUN_TASK_SETS:-}" JEFF_RUN_TASK_INVENTORY="${JEFF_RUN_TASK_INVENTORY:-}"
mkdir -p "$jobs/logs"
# Harbor runs from this folder (so it can import harbor_agent), so every path must be absolute.
tarball=$(cd "$(dirname "$tarball")" && pwd)/$(basename "$tarball")
jobs=$(cd "$jobs" && pwd)

run_one() {
  local task=$1
  # Terminal-Bench 2.0 by bare name; a hub task from its pinned dataset, with its agent time capped (task_source.py).
  local dataset="terminal-bench@2.0" include=$task multiplier=$timeout_multiplier name=$task network=online
  case "$task" in
    *:*)
      local spec
      spec=$(python3 "$here/task_source.py" harbor-args "$JEFF_RUN_TASK_SETS" "$JEFF_RUN_TASK_INVENTORY" "$timeout_multiplier" "$task") || return 1
      read -r dataset include multiplier name network <<< "$spec"
      ;;
  esac
  local job_name
  job_name="$name-$(date +%Y%m%d-%H%M%S)"
  local command=(
    uv run --project "$here" harbor run
    --dataset "$dataset" -i "$include" -n 1
    -a harbor_agent.jeff_pi:JeffPi
    --ak "tarball=$tarball" --ak model_api=openai-completions --ak "thinking=$thinking"
    -m "openai/$model"
    --agent-timeout-multiplier "$multiplier"
  )
  if [ "$tools" != default ]; then command+=(--ak "tools=$tools"); fi
  if [ "$network" = offline ]; then
    # No internet once pi runs (harbor_agent/offline.py); only the model's host (BASE_URL's) stays reachable.
    local model_host=${base_url#*://}
    model_host=${model_host%%[:/]*}
    local hosts=$model_host
    # Jeff mode, a Jeff router or Jeff trimming: the Jeff service's host must stay reachable too.
    if [ -n "$JEFF_FIRST_JEFF_URL" ] && { [ "$mode" = jeff ] || [[ "$JEFF_FIRST_THINKING_ROUTER" == jeff:* ]] \
        || [[ "$mode" != shadow && "$JEFF_FIRST_OUTPUT_TRIM" == jeff:* ]]; }; then
      local jeff_host=${JEFF_FIRST_JEFF_URL#*://}
      jeff_host=${jeff_host%%[:/]*}
      [ "$jeff_host" = "$model_host" ] || hosts="$hosts,$jeff_host"
    fi
    command+=(--env harbor_agent.offline:EgressDocker --ak "allowed_hosts=$hosts")
  fi
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
      --ae "JEFF_FIRST_THINKING_ROUTER=$JEFF_FIRST_THINKING_ROUTER"
    )
  elif [ "$mode" = record ]; then
    command+=(
      --ae "JEFF_FIRST_RUN_APPROVAL=$JEFF_FIRST_RUN_APPROVAL" --ae "JEFF_FIRST_DRIVER_BUILD=$JEFF_FIRST_DRIVER_BUILD"
      --ae "JEFF_FIRST_THINKING_ROUTER=$JEFF_FIRST_THINKING_ROUTER"
    )
  elif [ "$mode" = jeff ]; then
    command+=(
      --ae "JEFF_FIRST_JEFF_URL=$JEFF_FIRST_JEFF_URL" --ae "JEFF_FIRST_JEFF_STEP_ADAPTER=$JEFF_FIRST_JEFF_STEP_ADAPTER"
      --ae "JEFF_FIRST_JEFF_STEP_THRESHOLD=$JEFF_FIRST_JEFF_STEP_THRESHOLD"
      --ae "JEFF_FIRST_RUN_APPROVAL=$JEFF_FIRST_RUN_APPROVAL" --ae "JEFF_FIRST_DRIVER_BUILD=$JEFF_FIRST_DRIVER_BUILD"
      --ae "JEFF_FIRST_THINKING_ROUTER=$JEFF_FIRST_THINKING_ROUTER"
    )
  fi
  # A Jeff router (any mode with a router) needs the service and its threshold.
  if [[ "$mode" != shadow && "$JEFF_FIRST_THINKING_ROUTER" == jeff:* ]]; then
    [ "$mode" = jeff ] || command+=(--ae "JEFF_FIRST_JEFF_URL=$JEFF_FIRST_JEFF_URL")
    command+=(--ae "JEFF_FIRST_JEFF_ROUTER_THRESHOLD=$JEFF_FIRST_JEFF_ROUTER_THRESHOLD")
  fi
  # The flipped Jeff router holds its threshold in its value; it needs only the service.
  if [[ "$mode" != shadow && "$mode" != jeff && "$JEFF_FIRST_THINKING_ROUTER" == jeff-off-unless:* ]]; then
    command+=(--ae "JEFF_FIRST_JEFF_URL=$JEFF_FIRST_JEFF_URL")
  fi
  if [ "$mode" != shadow ]; then
    command+=(--ae "JEFF_FIRST_OUTPUT_TRIM=$JEFF_FIRST_OUTPUT_TRIM" --ae "JEFF_FIRST_THINKING_LIMIT=$JEFF_FIRST_THINKING_LIMIT")
    # Jeff trimming needs the service (unless jeff mode or a Jeff router already passes it) and its threshold.
    if [[ "$JEFF_FIRST_OUTPUT_TRIM" == jeff:* ]]; then
      [ "$mode" = jeff ] || [[ "$JEFF_FIRST_THINKING_ROUTER" == jeff:* || "$JEFF_FIRST_THINKING_ROUTER" == jeff-off-unless:* ]] \
        || command+=(--ae "JEFF_FIRST_JEFF_URL=$JEFF_FIRST_JEFF_URL")
      command+=(--ae "JEFF_FIRST_JEFF_TRIM_THRESHOLD=$JEFF_FIRST_JEFF_TRIM_THRESHOLD")
    fi
  fi
  if [ "$dry_run" = 1 ]; then
    printf '%q ' "${command[@]}"; echo
    return
  fi
  echo "$(date -Is) start $task"
  # Harbor exits 0 even when a trial failed, so check_trial.py inspects the job's results afterwards.
  local status=0
  (cd "$here" && PYTHONPATH="$here" OPENAI_BASE_URL="$base_url/v1" OPENAI_API_KEY="$JEFF_RUN_API_KEY" "${command[@]}") > "$jobs/logs/$name.log" 2>&1 \
    && (cd "$here" && uv run --project "$here" python check_trial.py "$jobs/$job_name" $( [ "$thinking" != off ] && echo --expect-thinking )) >> "$jobs/logs/$name.log" 2>&1 \
    || status=$?
  if [ "$status" = 0 ]; then
    echo "$(date -Is) done  $task ($(tail -1 "$jobs/logs/$name.log"))"
  else
    echo "$(date -Is) FAILED $task (exit $status; see $jobs/logs/$name.log)"
    return 1
  fi
}
export -f run_one
export here tarball base_url jobs thinking tools model mode timeout_multiplier dry_run JEFF_RUN_API_KEY \
  JEFF_RUN_THINKING_FORMAT JEFF_RUN_MAX_OUTPUT_TOKENS JEFF_FIRST_RUN_APPROVAL JEFF_FIRST_DRIVER_BUILD JEFF_FIRST_THINKING_ROUTER \
  JEFF_FIRST_JEFF_URL JEFF_FIRST_JEFF_STEP_ADAPTER JEFF_FIRST_JEFF_STEP_THRESHOLD JEFF_FIRST_JEFF_ROUTER_THRESHOLD \
  JEFF_FIRST_OUTPUT_TRIM JEFF_FIRST_JEFF_TRIM_THRESHOLD JEFF_RUN_TASK_SETS JEFF_RUN_TASK_INVENTORY

# xargs keeps going after a failed task and exits non-zero at the end; each failure is printed above.
if ! printf '%s\n' "${tasks[@]}" | xargs -P "$concurrency" -I{} bash -c 'run_one "$1"' _ {}; then
  echo "one or more tasks FAILED; their logs are in $jobs/logs" >&2
  exit 1
fi
