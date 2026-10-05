#!/usr/bin/env bash
# Final report of tonight's four-arm evaluation, run from the Mac: collects every session's line (eval_units.py) and
# writes the report (eval_report.py) to results/imitation/eval-tonight.md and a copy in the sdd folder.
#   live:  B200 and casdgx01 lines over ssh (while the B200 is reachable).
#   final: B200 lines from the Mac copy made by b200_final.sh after the 15:30 B200 stop (~/jeff-data/b200/eval-tonight;
#          refuses to run until b200_final.log says the copy is done), casdgx01 lines over ssh.
# The collected lines are kept in OUT_DIR (default: a fresh folder under /private/tmp/claude-501/eval-tonight).
# Usage: eval_report.sh live|final [OUT_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
mode=${1:?usage: eval_report.sh live|final [OUT_DIR]}
out=${2:-/private/tmp/claude-501/eval-tonight/report-$(date +%H%M%S)}
mkdir -p "$out"
case "$mode" in
live)
	ssh mstrasser@85.13.211.160 'cd /raid/work/jeff-first/eval-tonight/repo2/tools/jeff-first && python3 eval_units.py /raid/work/jeff-first/eval-tonight' > "$out/b200.jsonl"
	;;
final)
	log=/private/tmp/claude-501/eval-tonight/b200_final.log
	if ! grep -q "B200 evaluation finished" "$log"; then
		echo "the B200 final copy is not done ($log has no 'B200 evaluation finished' line); use live or wait" >&2
		exit 1
	fi
	python3 "$here/eval_units.py" ~/jeff-data/b200/eval-tonight > "$out/b200.jsonl"
	;;
*)
	echo "usage: eval_report.sh live|final [OUT_DIR]" >&2
	exit 1
	;;
esac
ssh -o ControlPath=none mstrasser@casdgx01 'cd ~/jeff-first/eval-tonight/repo2/tools/jeff-first && python3 eval_units.py ~/jeff-first/eval-tonight' > "$out/casdgx01.jsonl"
python3 "$here/eval_report.py" "$repo/results/imitation/eval-tonight.md" "$out/b200.jsonl" "$out/casdgx01.jsonl"
cp "$repo/results/imitation/eval-tonight.md" "$repo/.superpowers/sdd/2026-10-03-jeff-first-imitation-data/eval-tonight-final.md"
echo "lines in $out"
