#!/usr/bin/env bash
# Phase-0 shadow run: plain pi (the jeff-pi fork in shadow mode) with local Qwen on the phase-0 tasks.
# One Harbor run per task, so each trace line carries its own task id; CONCURRENCY tasks run at a time.
# Each task's trace lands in JOBS_DIR/<job>/<trial>/agent/jeff-first-trace.jsonl.
#
# Usage: run_phase0.sh [--dry-run] TASKS_JSON TARBALL PROXY_URL JOBS_DIR CONCURRENCY THINKING TOOLS [TASK ...]
#   TASKS_JSON   results/phase0/tasks.json (its "phase0" list is used unless TASK names are given)
#   TARBALL      the packed fork (npm pack in packages/coding-agent)
#   PROXY_URL    the sparkgate proxy as the task containers reach it, e.g. http://192.168.0.50:8899
#   THINKING     pi's thinking level for Qwen: off, minimal, low, medium or high
#   TOOLS        pi's tool list, e.g. read,bash,edit,write,grep,find,ls; "default" keeps pi's own (read,bash,edit,write)
#
# Run it on datigator in tmux session jeff-pi-phase0; stop it with: tmux kill-session -t jeff-pi-phase0
set -euo pipefail

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then dry_run=1; shift; fi
if [ $# -lt 7 ]; then
  sed -n '5,11p' "$0" >&2
  exit 2
fi
tasks_json=$1 tarball=$2 proxy=$3 jobs=$4 concurrency=$5 thinking=$6 tools=$7
shift 7
here=$(cd "$(dirname "$0")" && pwd)

[ -f "$tasks_json" ] || { echo "no tasks file at $tasks_json" >&2; exit 1; }
[ -f "$tarball" ] || { echo "no fork tarball at $tarball; build it with npm pack" >&2; exit 1; }
case "$thinking" in off|minimal|low|medium|high) ;; *) echo "THINKING must be off, minimal, low, medium or high" >&2; exit 2 ;; esac
case "$concurrency" in ''|*[!0-9]*) echo "CONCURRENCY must be a whole number" >&2; exit 2 ;; esac

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
    -m openai/qwen3.8-flash-next
  )
  if [ "$tools" != default ]; then command+=(--ak "tools=$tools"); fi
  command+=(
    --ae JEFF_FIRST_MODE=shadow
    --ae "JEFF_FIRST_TASK_ID=$task"
    --ae JEFF_FIRST_TRACE_FILE=/logs/agent/jeff-first-trace.jsonl
    -o "$jobs" --job-name "$job_name"
  )
  if [ "$dry_run" = 1 ]; then
    printf '%q ' "${command[@]}"; echo
    return
  fi
  echo "$(date -Is) start $task"
  # Harbor exits 0 even when a trial failed, so check_trial.py inspects the job's results afterwards.
  local status=0
  (cd "$here" && PYTHONPATH="$here" OPENAI_BASE_URL="$proxy/v1" OPENAI_API_KEY=unused "${command[@]}") > "$jobs/logs/$task.log" 2>&1 \
    && (cd "$here" && uv run --project "$here" python check_trial.py "$jobs/$job_name") >> "$jobs/logs/$task.log" 2>&1 \
    || status=$?
  if [ "$status" = 0 ]; then
    echo "$(date -Is) done  $task ($(tail -1 "$jobs/logs/$task.log"))"
  else
    echo "$(date -Is) FAILED $task (exit $status; see $jobs/logs/$task.log)"
    return 1
  fi
}
export -f run_one
export here tarball proxy jobs thinking tools dry_run

# xargs keeps going after a failed task and exits non-zero at the end; each failure is printed above.
if ! printf '%s\n' "${tasks[@]}" | xargs -P "$concurrency" -I{} bash -c 'run_one "$1"' _ {}; then
  echo "one or more tasks FAILED; their logs are in $jobs/logs" >&2
  exit 1
fi
