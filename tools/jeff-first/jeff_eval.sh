#!/usr/bin/env bash
# Qwen3.8-27B in bash-only pi with the trained Jeff scouting before every Qwen turn (JEFF_FIRST_MODE=jeff) and Jeff's
# router choosing Qwen's thinking level per request, on a list of tasks, through run_phase0.sh. Only the adapters and
# their thresholds change between runs; everything else is fixed here.
#
# Usage: jeff_eval.sh RUN_DIR TARBALL QWEN_URL DRIVER_BUILD JEFF_URL STEP_ADAPTER STEP_THRESHOLD ROUTER THRESHOLD
#                     OUT_FOLDER CONCURRENCY TASK...
#   RUN_DIR         folder holding tools/jeff-first (this script's repository copy) and tasks.json
#   TARBALL         the packed jeff-pi fork with jeff mode (npm pack in packages/coding-agent)
#   QWEN_URL        Qwen's OpenAI-compatible server as the task containers reach it, without /v1,
#                   e.g. http://192.168.2.10:8885
#   DRIVER_BUILD    the exact Qwen build for the trace, e.g. qwen3.8-27b-fp8@casdgx01-gpu0
#   JEFF_URL        the Jeff service (jeff-serve) as the task containers reach it, e.g. http://192.168.2.10:8920
#   STEP_ADAPTER    the step adapter's name at the service ("jeff" = the base model without an adapter)
#   STEP_THRESHOLD  the probability from 0 to 1 Jeff's best option needs, or the scout hands over
#   ROUTER          the thinking router: jeff:<router adapter> or fixed:<off|low|medium|xhigh>
#   THRESHOLD       the router threshold from 0 to 1 (a level below xhigh needs it); "-" with a fixed router
#   OUT_FOLDER      where the Harbor jobs and logs go (created)
#   CONCURRENCY     tasks at a time
#   TASK            Terminal-Bench 2.0 names, or <org>/<dataset>:<task> hub ids (then JEFF_RUN_TASK_SETS and
#                   JEFF_RUN_TASK_INVENTORY must be set, see run_phase0.sh)
# Environment: JEFF_FIRST_OUTPUT_TRIM (required): off, fixed:<all|last200|last40|first40|first20last20>, or
#   jeff:<trimming adapter> with JEFF_FIRST_JEFF_TRIM_THRESHOLD (the service is JEFF_URL); a trimming adapter is checked
#   at the service like the others.
# Fixed: pi thinking "high" (the router sets each request's level), output cap 32768 tokens, run approval all, time
# multiplier 6 (as the collection).
set -euo pipefail
[ $# -ge 12 ] || { sed -n '2,28p' "$0" >&2; exit 2; }
RUN_DIR=$1 TARBALL=$2 QWEN_URL=$3 DRIVER=$4 JEFF_URL=$5 STEP_ADAPTER=$6 STEP_THRESHOLD=$7 ROUTER=$8 ROUTER_THRESHOLD=$9
OUT=${10} CONCURRENCY=${11}
shift 11
export PATH=$HOME/.local/bin:/snap/bin:/raid/work/jeff-first/bin:$PATH
cd "$RUN_DIR"
# The service must answer before anything starts (a missing adapter fails here, not in the middle of a task).
health=$(curl -sf --max-time 10 "$JEFF_URL/health") || { echo "the Jeff service at $JEFF_URL does not answer /health" >&2; exit 1; }
adapters=("$STEP_ADAPTER")
if [[ "$ROUTER" == jeff:* ]]; then adapters+=("${ROUTER#jeff:}"); fi
: "${JEFF_FIRST_OUTPUT_TRIM:?set JEFF_FIRST_OUTPUT_TRIM (off, fixed:<choice> or jeff:<trimming adapter>)}"
if [[ "$JEFF_FIRST_OUTPUT_TRIM" == jeff:* ]]; then adapters+=("${JEFF_FIRST_OUTPUT_TRIM#jeff:}"); fi
for adapter in "${adapters[@]}"; do
	[ "$adapter" = jeff ] && continue
	python3 -c 'import json, sys; health = json.loads(sys.argv[1]); sys.exit(0 if sys.argv[2] in health["adapters"] else 1)' "$health" "$adapter" \
		|| { echo "the Jeff service at $JEFF_URL has no adapter $adapter (loaded: $health)" >&2; exit 1; }
done
export JEFF_RUN_API_KEY=unused JEFF_RUN_THINKING_FORMAT=qwen-chat-template JEFF_RUN_MAX_OUTPUT_TOKENS=32768
export JEFF_FIRST_RUN_APPROVAL=all JEFF_FIRST_DRIVER_BUILD=$DRIVER JEFF_FIRST_THINKING_ROUTER=$ROUTER
export JEFF_FIRST_JEFF_URL=$JEFF_URL JEFF_FIRST_JEFF_STEP_ADAPTER=$STEP_ADAPTER JEFF_FIRST_JEFF_STEP_THRESHOLD=$STEP_THRESHOLD
case "$ROUTER" in
	jeff:*) export JEFF_FIRST_JEFF_ROUTER_THRESHOLD=$ROUTER_THRESHOLD ;;
	*) [ "$ROUTER_THRESHOLD" = - ] || { echo "a fixed router takes no threshold: give - instead of $ROUTER_THRESHOLD" >&2; exit 2; } ;;
esac
mkdir -p "$OUT"
bash tools/jeff-first/run_phase0.sh tasks.json "$TARBALL" "$QWEN_URL" "$OUT" "$CONCURRENCY" high bash qwen3.8-27b jeff 6 "$@"
