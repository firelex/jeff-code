"""Count what Jeff decided in a jeff_eval.sh run: per decision kind, the questions Jeff was asked, how many were cut to
fit its 8,192-token limit, and how many it abstained on because they could not be cut to fit (owner ruling
2026-10-04: then the scout hands over, the router uses xhigh, the trimmer keeps the whole output).

It reads every trace (<run folder>/**/agent/jeff-first-trace.jsonl, as run_phase0.sh places them):
- step: each question ("levels" entry) of a "decision" line whose chooser is Jeff (jeff:<adapter>); fields jeff_cut and
  jeff_abstained;
- router: each attempt-1 "qwen_request" line whose router is Jeff; fields router_cut and router_abstained;
- trim: each "output_trim" line whose trimmer is Jeff; fields jeff_cut and jeff_abstained.
A trace written before these fields existed is an error. Prints the counts and writes them to
<run folder>/jeff-decisions-summary.json. Usage: python3 jeff_eval_summary.py RUN_FOLDER
"""

import json
import sys
from pathlib import Path

KINDS = ("step", "router", "trim")


def summarize(folder: Path) -> dict:
    traces = sorted(folder.rglob("agent/jeff-first-trace.jsonl"))
    if not traces:
        raise ValueError(f"no agent/jeff-first-trace.jsonl under {folder}")
    counts = {kind: {"asked": 0, "cut": 0, "abstained": 0} for kind in KINDS}
    abstentions = []

    def count(kind: str, cut, abstained, where: str) -> None:
        counts[kind]["asked"] += 1
        counts[kind]["cut"] += cut is not None
        if abstained is not None:
            counts[kind]["abstained"] += 1
            abstentions.append({"kind": kind, "where": where, **abstained})

    for trace in traces:
        for number, text in enumerate(trace.read_text(encoding="utf-8").splitlines(), start=1):
            if not text.strip():
                continue
            line = json.loads(text)
            where = f"{trace.relative_to(folder)}:{number}"
            if line["kind"] == "decision":
                for level in line["levels"]:
                    if level["chooser"].startswith("jeff:"):
                        count("step", level["jeff_cut"], level["jeff_abstained"], where)
            elif line["kind"] == "qwen_request":
                if line["router"].startswith("jeff:") and line["attempt"] == 1:
                    count("router", line["router_cut"], line["router_abstained"], where)
            elif line["kind"] == "output_trim":
                if line["trimmer"].startswith("jeff:"):
                    count("trim", line["jeff_cut"], line["jeff_abstained"], where)
    return {"traces": len(traces), "counts": counts, "abstentions": abstentions}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: jeff_eval_summary.py RUN_FOLDER")
    folder = Path(args[0])
    summary = summarize(folder)
    (folder / "jeff-decisions-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"Jeff's decisions in {summary['traces']} traces under {folder}:")
    for kind, values in summary["counts"].items():
        print(f"  {kind}: {values['asked']} questions, {values['cut']} cut to fit, {values['abstained']} abstained "
              "(could not be cut to fit)")


if __name__ == "__main__":
    main()
