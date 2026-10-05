"""One JSON line per session of tonight's four-arm evaluation on this host (eval_stream.sh layout): the session's tags
(arm, host, server, block, task, attempt), whether it finished, its reward, its times and what the trace counted.

Read from each session folder EVAL_DIR/runs/ARM/TASK/attemptK/:
- meta.json (written when run_phase0.sh returned; absent while the session runs);
- the Harbor trial's result.json: reward (verifier_result.rewards.reward), exception, agent execution seconds (the
  session's wall time: agent start to agent end) and trial seconds (environment setup to verifier end);
- the trace agent/jeff-first-trace.jsonl:
  - qwen_request lines: Qwen generation seconds (timings_ms.model, all attempts), turns (attempt 1 lines), turns at
    off (attempt-1 thinking_level off), router choices (router_level), guard hits (attempt-1 lines with a loop or
    runaway trigger, which were re-asked), forced xhigh turns (forced_xhigh set: stuck outputs, failed commands),
    thinking-limit cuts (limit_cut set, any attempt), router seconds (timings_ms.router);
  - decision lines: Jeff steps taken (action kind step) and hand-overs;
  - output_trim lines: trim questions and cuts (shortened);
- JeffFirst errors in pi's output (pi_errors.jeff_first_errors).
An error is a trial exception other than the agent time limit, a JeffFirst error, or a finished session without a
result. Usage: python3 eval_units.py EVAL_DIR
"""

import datetime as dt
import json
import sys
from pathlib import Path

from pi_errors import jeff_first_errors


def seconds(span: dict | None) -> float | None:
    if not span or not span.get("started_at") or not span.get("finished_at"):
        return None
    start = dt.datetime.fromisoformat(span["started_at"].replace("Z", "+00:00"))
    end = dt.datetime.fromisoformat(span["finished_at"].replace("Z", "+00:00"))
    return (end - start).total_seconds()


def trace_counts(trace: Path) -> dict:
    c = {
        "qwen_s": 0.0,
        "router_s": 0.0,
        "requests": 0,
        "turns": 0,
        "turns_off": 0,
        "router_levels": {},
        "guard_loop": 0,
        "guard_runaway": 0,
        "forced_xhigh": 0,
        "limit_cuts": 0,
        "jeff_steps": 0,
        "decisions": 0,
        "trim_questions": 0,
        "trim_cuts": 0,
        "router_ms": [],
        "trim_ms": [],
        "step_ms_per_question": [],
    }
    content = trace.read_text(encoding="utf-8")
    lines = content.split("\n")
    # A session killed while writing (e.g. at the time limit) leaves a last line without its newline: that one partial
    # line is left out and counted; any other unreadable line still raises.
    c["trace_cut_last_line"] = not content.endswith("\n") and bool(lines[-1].strip())
    if c["trace_cut_last_line"]:
        lines = lines[:-1]
    for text in lines:
        if not text.strip():
            continue
        line = json.loads(text)
        kind = line["kind"]
        if kind == "qwen_request":
            c["requests"] += 1
            c["qwen_s"] += line["timings_ms"]["model"] / 1000
            c["limit_cuts"] += line["limit_cut"] is not None
            if line["attempt"] == 1:
                c["turns"] += 1
                c["turns_off"] += line["thinking_level"] == "off"
                if line["router_level"] is not None:
                    c["router_levels"][line["router_level"]] = c["router_levels"].get(line["router_level"], 0) + 1
                if line["timings_ms"]["router"] is not None:
                    c["router_s"] += line["timings_ms"]["router"] / 1000
                    if line["router"].startswith(("jeff:", "jeff-off-unless:")):
                        c["router_ms"].append(round(line["timings_ms"]["router"]))
                c["forced_xhigh"] += line["forced_xhigh"] is not None
                if line["guard"] is not None:
                    c["guard_" + line["guard"]["trigger"]] += 1
        elif kind == "decision":
            c["decisions"] += 1
            c["jeff_steps"] += line["action"]["kind"] == "step"
            if line["levels"] and line["levels"][0]["chooser"].startswith("jeff:"):
                c["step_ms_per_question"].append(round(line["timings_ms"]["chooser"] / len(line["levels"])))
        elif kind == "output_trim":
            c["trim_questions"] += line["trimmer"].startswith("jeff:")
            if line["timings_ms"]["jeff"] is not None:
                c["trim_ms"].append(round(line["timings_ms"]["jeff"]))
            c["trim_cuts"] += bool(line["shortened"])
    return c


def benchmark_of(folder: Path) -> str:
    """A running session's benchmark: eval_stream.sh writes it to benchmark.txt at the start (the first TB2 streams did
    not, so a folder without it is Terminal-Bench 2.0)."""
    path = folder / "benchmark.txt"
    return path.read_text().strip() if path.exists() else "terminal-bench-2"


def unit(folder: Path) -> dict:
    arm, task, attempt = folder.parts[-3], folder.parts[-2], folder.parts[-1].removeprefix("attempt")
    row: dict = {"arm": arm, "task": task, "attempt": attempt, "folder": str(folder), "benchmark": benchmark_of(folder)}
    superseded = folder / "superseded.txt"
    row["superseded"] = superseded.read_text().strip() if superseded.exists() else None
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        row["state"] = "running"
        return row
    meta = json.loads(meta_path.read_text())
    row.update({k: meta[k] for k in ("host", "server", "stream", "block", "task", "started", "finished", "exit")})
    row["benchmark"] = meta.get("benchmark", "terminal-bench-2")
    row["pair_block"] = meta["block"].removesuffix("r")
    row["state"] = "finished"
    results = sorted(folder.glob("*/*/result.json"))
    if str(meta["exit"]).startswith("pull-failed"):
        row["error"] = f"image pull failed ({meta['exit']})"
        return row
    if len(results) != 1:
        row["error"] = f"{len(results)} result.json files"
        return row
    trial = results[0].parent
    result = json.loads(results[0].read_text())
    info = result["exception_info"]
    reward = (result.get("verifier_result") or {}).get("rewards", {}) or {}
    row["reward"] = reward.get("reward")
    row["exception"] = info["exception_type"] if info else None
    row["agent_s"] = seconds(result.get("agent_execution"))
    row["trial_s"] = seconds(result)
    row["error"] = None
    try:
        errors = jeff_first_errors(trial)
    except (FileNotFoundError, ValueError) as unreadable:
        # The session is an error either way; the summary shows why (one bad session must not hide the others).
        row["error"] = f"pi output unreadable: {unreadable}"[:300]
        return row
    row["jeff_first_error"] = errors[-1][:300] if errors else None
    if info and info["exception_type"] != "AgentTimeoutError":
        row["error"] = f"{info['exception_type']}: {info['exception_message'][:300]}"
    elif errors:
        row["error"] = "JeffFirst: " + errors[-1][:300]
    trace = trial / "agent" / "jeff-first-trace.jsonl"
    if trace.exists() and trace.stat().st_size > 0:
        row.update(trace_counts(trace))
    elif row["error"] is None:
        row["error"] = "no trace"
    return row


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: eval_units.py EVAL_DIR")
    runs = Path(sys.argv[1]) / "runs"
    for folder in sorted(runs.glob("*/*/attempt*")):
        print(json.dumps(unit(folder)))


if __name__ == "__main__":
    main()
