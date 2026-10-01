"""Check one Harbor job of the phase-0 run: every trial must have finished without an exception and written a trace.

Harbor exits with status 0 even when a trial failed (for example when Docker could not start the task), so the run
script calls this after each job and stops that task with a clear error instead of reporting it as done.

Usage: python check_trial.py <Harbor job folder>
"""

import json
import sys
from pathlib import Path

# Running out of the task's time limit is a normal benchmark outcome (the verifier still scores the task), not a
# broken run. Every other exception means the trial itself failed.
TASK_OUTCOME_EXCEPTIONS = {"AgentTimeoutError"}


def check_job(job: Path) -> str:
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
        note = " (agent timed out)" if info is not None else ""
        trace = trial / "agent" / "jeff-first-trace.jsonl"
        if not trace.exists() or trace.stat().st_size == 0:
            raise RuntimeError(f"{trial.name} wrote no jeff-first-trace.jsonl")
        count = sum(1 for line in trace.read_text().splitlines() if line.strip())
        lines.append(f"{trial.name}: ok{note}, {count} trace lines")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_trial.py <Harbor job folder>")
    print(check_job(Path(sys.argv[1])))
