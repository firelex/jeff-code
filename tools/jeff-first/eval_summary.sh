#!/usr/bin/env bash
# Tonight's four-arm evaluation summary, run from the Mac at any time: collects every session's line (eval_units.py) from
# the B200 and casdgx01 and prints the per-arm table (eval_summary.py). The lines are kept in OUT_DIR (default: a
# fresh folder under /private/tmp/claude-501/eval-tonight).
# Usage: eval_summary.sh [OUT_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-/private/tmp/claude-501/eval-tonight/summary-$(date +%H%M%S)}
mkdir -p "$out"
# The B200 was handed back after 15:30 on 2026-10-05; its evaluation data is read from the copy on casdgx01.
ssh -o ControlPath=none mstrasser@casdgx01 'test -d ~/jeff-first/eval-tonight-b200/runs && cd ~/jeff-first/eval-tonight/repo2/tools/jeff-first && python3 eval_units.py ~/jeff-first/eval-tonight-b200' > "$out/b200.jsonl"
ssh -o ControlPath=none mstrasser@casdgx01 'cd ~/jeff-first/eval-tonight/repo2/tools/jeff-first && python3 eval_units.py ~/jeff-first/eval-tonight' > "$out/casdgx01.jsonl"
python3 "$here/eval_summary.py" terminal-bench-2=480,terminal-bench=132,terminal-bench-science=140,harbor-index-1.0=164,skillsbench=176,terminal-bench-pro=400,swe-rebench-leaderboard=756,swe-rebench-leaderboard#2=756,terminal-bench-pro#2=400,skillsbench#2=176,harbor-index-1.0#2=164,swe-bench-verified=1000 "$out/b200.jsonl" "$out/casdgx01.jsonl" | tee "$out/summary.md"
