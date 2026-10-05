"""Gate 0 of the scout design: does a strong scout (GLM in teacher mode) save the large model work?

Compares two arms run on the same tasks with the same large model: the baseline (shadow mode, plain Jeff-Code behaviour)
and the teacher arm. The gate passes when, summed over the tasks, the large model's turns and its own seconds both
fall by at least 25%, and the teacher arm passes at most one task fewer. A task that ran out of its agent time limit in
only one arm leaves the gate not judged: the time limit, not the scout, would decide that task.

Usage: uv run python gate0_report.py BASE_RUNS TEACHER_RUNS OUT_MD TASK [TASK ...]
"""

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from pi_errors import jeff_first_errors
from scout_quality import quality

# A turn cut off at the length limit still cost the large model a turn.
COUNTED_STOPS = {"toolUse", "stop", "length"}
REQUIRED_DROP = 0.25
PASSES_ALLOWED_LOST = 1
AGENT_TIMEOUT = "AgentTimeoutError"


@dataclass(frozen=True)
class TaskResult:
    turns: int
    model_seconds: float
    passed: bool
    scout_steps: int
    teacher_seconds: float
    timed_out: bool
    forced_steps: int
    max_identical_in_stint: int


def latest_trial(runs: Path, task: str) -> Path:
    trials = sorted(runs.glob(f"{task}-*/{task}__*"))
    if not trials:
        raise FileNotFoundError(f"no Harbor trial for {task} under {runs}")
    return trials[-1]


def summarise(runs: Path, tasks: list[str]) -> dict[str, TaskResult]:
    results = {}
    for task in tasks:
        trial = latest_trial(runs, task)
        # The trial's own exception comes first: a trial that failed during setup has no pi.txt to check.
        trial_result = json.loads((trial / "result.json").read_text())
        info = trial_result["exception_info"]
        if info is not None and info["exception_type"] != AGENT_TIMEOUT:
            raise ValueError(f"{trial.name} raised {info['exception_type']}: {info['exception_message'][:500]}")
        errors = jeff_first_errors(trial)
        if errors:
            raise ValueError(f"{trial.name} ended with a JeffFirst error: {errors[-1][:500]}; rerun the task before judging Gate 0")
        lines = [json.loads(line) for line in (trial / "agent" / "jeff-first-trace.jsonl").read_text().splitlines()]
        turns = [l for l in lines if l.get("kind", "model_turn") == "model_turn" and l["action"]["stop_reason"] in COUNTED_STOPS]
        decisions = [l for l in lines if l.get("kind") == "decision"]
        reward = (trial_result.get("verifier_result") or {}).get("rewards", {}).get("reward")
        if reward is None:
            raise ValueError(f"{trial} has no reward in result.json; the verifier did not finish")
        q = quality(lines)
        results[task] = TaskResult(
            turns=len(turns),
            model_seconds=sum(l["timings_ms"]["model"] for l in turns) / 1000,
            passed=reward == 1.0,
            scout_steps=sum(1 for d in decisions if d["action"]["kind"] == "step"),
            teacher_seconds=sum(d["timings_ms"]["chooser"] for d in decisions) / 1000,
            timed_out=info is not None,
            forced_steps=q.forced_steps,
            max_identical_in_stint=q.max_identical_in_stint,
        )
    return results


def drop(before: float, after: float) -> float:
    if before == 0:
        raise ValueError("the baseline has no model turns, so no drop can be measured")
    return (before - after) / before


def median_drop(base: dict[str, TaskResult], teacher: dict[str, TaskResult], field: str) -> float:
    return statistics.median(drop(getattr(base[task], field), getattr(teacher[task], field)) for task in base)


def gate(base: dict[str, TaskResult], teacher: dict[str, TaskResult]) -> tuple[bool, str]:
    turns = drop(sum(r.turns for r in base.values()), sum(r.turns for r in teacher.values()))
    seconds = drop(sum(r.model_seconds for r in base.values()), sum(r.model_seconds for r in teacher.values()))
    lost = sum(r.passed for r in base.values()) - sum(r.passed for r in teacher.values())
    summary = (
        f"model turns {turns:.0%} fewer (median per task {median_drop(base, teacher, 'turns'):.0%}), "
        f"model seconds {seconds:.0%} fewer (median per task {median_drop(base, teacher, 'model_seconds'):.0%}), "
        f"passes lost {lost}"
    )
    one_arm = [task for task in base if base[task].timed_out != teacher[task].timed_out]
    if one_arm:
        return False, (
            f"**Gate 0: not judged.** Timed out in only one arm: {', '.join(one_arm)}. The agent time limit, not the "
            f"scout, decided those tasks; rerun them in both arms with a larger TIMEOUT_MULTIPLIER. ({summary}.)"
        )
    failures = []
    if turns < REQUIRED_DROP:
        failures.append(f"model turns fell {turns:.0%} (need {REQUIRED_DROP:.0%})")
    if seconds < REQUIRED_DROP:
        failures.append(f"model seconds fell {seconds:.0%} (need {REQUIRED_DROP:.0%})")
    if lost > PASSES_ALLOWED_LOST:
        failures.append(f"{lost} fewer tasks passed (at most {PASSES_ALLOWED_LOST} allowed)")
    if failures:
        return False, f"**Gate 0: fail.** {summary}. " + "; ".join(failures) + "."
    return True, f"**Gate 0: pass.** {summary}."


def render(base: dict[str, TaskResult], teacher: dict[str, TaskResult]) -> str:
    rows = [
        "| Task | Base turns | Teacher-arm turns | Base model s | Teacher-arm model s | Scout steps | Forced steps "
        "| Most identical steps in a stint | Teacher s | Passed (base / teacher arm) | Timed out (base / teacher arm) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task in base:
        b, t = base[task], teacher[task]
        rows.append(
            f"| {task} | {b.turns} | {t.turns} | {b.model_seconds:.0f} | {t.model_seconds:.0f} | {t.scout_steps} "
            f"| {t.forced_steps} | {t.max_identical_in_stint} | {t.teacher_seconds:.0f} "
            f"| {yes_no(b.passed)} / {yes_no(t.passed)} | {yes_no(b.timed_out)} / {yes_no(t.timed_out)} |"
        )
    _, line = gate(base, teacher)
    timeouts = (
        f"Timed out in the base arm: {timed_out_tasks(base)}. Timed out in the teacher arm: {timed_out_tasks(teacher)}."
    )
    return "# Gate 0: oracle scout\n\n" + line + "\n\n" + timeouts + "\n\n" + "\n".join(rows) + "\n"


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def timed_out_tasks(results: dict[str, TaskResult]) -> str:
    tasks = [task for task, result in results.items() if result.timed_out]
    return ", ".join(tasks) if tasks else "none"


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    base_runs, teacher_runs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tasks = sys.argv[4:]
    report = render(summarise(base_runs, tasks), summarise(teacher_runs, tasks))
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
