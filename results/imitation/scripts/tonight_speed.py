"""Speed of running Jeff trainings: per run folder (train.py's --run), the steps done, tokens per second over the
recent steps, and the projected finish. Stdlib only; tonight.sh runs it on the B200 (python3 - RUNS_DIR < this file).

Projection: remaining steps x the mean seconds of the last (up to) 20 steps, plus the remaining evaluations x the
last evaluation's duration (the time between an evaluation and the training step it follows). Before the first
evaluation after step 0 the evaluation time is unknown: the projection then says so and leaves it out.
"""

import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

RECENT = 20


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(text) for text in path.read_text().split("\n") if text.strip()]


def report(run: Path) -> str:
    config = json.loads((run / "config.json").read_text())
    planned = math.ceil(config["train_rows"] / config["effective_batch_size"]) * config["epochs"]
    every = config["eval_every"]
    steps = lines(run / "training.jsonl")
    evaluations = lines(run / "evaluations.jsonl")
    exit_file = run / "exit"
    state = f"exited {exit_file.read_text().strip()}" if exit_file.exists() else "running"
    if not steps:
        return f"{run.name}: {state}; no training step yet (start-up and the step-0 evaluation take ~6 min)"
    done = steps[-1]["step"]
    recent = steps[-RECENT:] if len(steps) > 3 else steps
    seconds = sum(step["step_seconds"] for step in recent) / len(recent)
    tokens_per_second = sum(step["input_tokens"] for step in recent) / sum(step["step_seconds"] for step in recent)
    by_step = {step["step"]: step["process_elapsed_seconds"] for step in steps}
    durations = [e["process_elapsed_seconds"] - by_step[e["step"]] for e in evaluations if e["step"] in by_step]
    remaining_steps = planned - done
    remaining_evaluations = sum(1 for s in range(done + 1, planned + 1) if s % every == 0 or s == planned)
    left = remaining_steps * seconds
    note = ""
    if durations:
        left += remaining_evaluations * durations[-1]
        note = f", evaluation {durations[-1]:.0f} s x {remaining_evaluations} left"
    else:
        note = f", {remaining_evaluations} evaluations left NOT counted (none timed yet)"
    finish = datetime.now() + timedelta(seconds=left)
    return (f"{run.name}: {state}; step {done}/{planned}; {seconds:.1f} s/step and {tokens_per_second:,.0f} tokens/s "
            f"over the last {len(recent)} steps{note}; {left / 3600:.1f} h left, finish ~{finish:%a %H:%M}")


def main() -> None:
    root = Path(sys.argv[1])
    runs = sorted(path for path in root.iterdir() if (path / "config.json").exists() and not path.name.startswith("check-"))
    if not runs:
        raise SystemExit(f"{root}: no run folders with config.json")
    print(time.strftime("%H:%M:%S"))
    for run in runs:
        print(report(run))


if __name__ == "__main__":
    main()
