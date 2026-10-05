"""Check one Harbor job of the phase-0 run: every trial must have finished without an exception, without a JeffFirst
error and with a trace.

Harbor exits with status 0 even when a trial failed (for example when Docker could not start the task, or when the
scout failed and Jeff-Code ended the session with a "JeffFirst:" error), so the run script calls this after each job and stops
that task with a clear error instead of reporting it as done.

Usage: python check_trial.py <Harbor job folder> [--expect-thinking]
"""

import json
import sys
from pathlib import Path

from pi_errors import jeff_first_errors

# Running out of the task's time limit is a normal benchmark outcome (the verifier still scores the task), not a
# broken run. Every other exception means the trial itself failed.
TASK_OUTCOME_EXCEPTIONS = {"AgentTimeoutError"}


def model_turns(trial: Path) -> list[dict]:
    """The large model's assistant messages in Jeff-Code's session files (the scout's own messages are left out)."""
    turns = []
    for session in sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl")):
        for line in session.read_text().splitlines():
            entry = json.loads(line)
            message = entry.get("message") if entry.get("type") == "message" else None
            if message and message["role"] == "assistant" and message.get("provider") != "jeff-first":
                turns.append(message)
    return turns


def check_job(job: Path, expect_thinking: bool) -> str:
    results = sorted(job.glob("*/result.json"))
    if not results:
        raise RuntimeError(f"{job} has no trial results")
    lines = []
    for result_path in results:
        trial = result_path.parent
        result = json.loads(result_path.read_text())
        info = result["exception_info"]
        if info is not None and info["exception_type"] not in TASK_OUTCOME_EXCEPTIONS:
            raise RuntimeError(f"{trial.name} raised {info['exception_type']}: {info['exception_message'][:500]}")
        errors = jeff_first_errors(trial)
        if errors:
            raise RuntimeError(f"{trial.name} ended with a JeffFirst error: {errors[-1][:500]}")
        note = " (agent timed out)" if info is not None else ""
        trace = trial / "agent" / "jeff-first-trace.jsonl"
        if not trace.exists() or trace.stat().st_size == 0:
            raise RuntimeError(f"{trial.name} wrote no jeff-first-trace.jsonl")
        count = sum(1 for line in trace.read_text().splitlines() if line.strip())
        if expect_thinking:
            turns = model_turns(trial)
            with_thinking = sum(1 for message in turns if any(p.get("type") == "thinking" and p.get("thinking") for p in message["content"]))
            if with_thinking == 0:
                raise RuntimeError(
                    f"{trial.name}: thinking was on, but none of the model's {len(turns)} turns holds thinking content; "
                    "check the model entry Jeff-Code was given (reasoning and compat.thinkingFormat)"
                )
        lines.append(f"{trial.name}: ok{note}, {count} trace lines")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: check_trial.py <Harbor job folder> [--expect-thinking]")
    expect = "--expect-thinking" in sys.argv[2:]
    print(check_job(Path(sys.argv[1]), expect_thinking=expect))
