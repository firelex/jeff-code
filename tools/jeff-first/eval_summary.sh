#!/usr/bin/env bash
# Tonight's four-arm evaluation summary, run from the Mac at any time: collects every session's line (eval_units.py) from
# the B200 and casdgx01 and prints the per-arm table (eval_summary.py). The lines are kept in OUT_DIR (default: a
# fresh folder under /private/tmp/claude-501/eval-tonight).
# Usage: eval_summary.sh [OUT_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-/private/tmp/claude-501/eval-tonight/summary-$(date +%H%M%S)}
mkdir -p "$out"
ssh mstrasser@85.13.211.160 'cd /raid/work/jeff-first/eval-tonight/repo/tools/jeff-first && python3 eval_units.py /raid/work/jeff-first/eval-tonight' > "$out/b200.jsonl"
ssh -o ControlPath=none mstrasser@casdgx01 'cd ~/jeff-first/eval-tonight/repo/tools/jeff-first && python3 eval_units.py ~/jeff-first/eval-tonight' > "$out/casdgx01.jsonl"
python3 "$here/eval_summary.py" 480 "$out/b200.jsonl" "$out/casdgx01.jsonl" | tee "$out/summary.md"
