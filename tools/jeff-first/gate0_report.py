"""Gate 0 of the scout design: does a strong scout (GLM in teacher mode) save the large model work?

Compares two arms run on the same tasks with the same large model: the baseline (shadow mode, plain pi behaviour)
and the teacher arm. The gate passes when, summed over the tasks, the large model's turns and its own seconds both
fall by at least 25%, and the teacher arm passes at most one task fewer.

Usage: uv run python gate0_report.py BASE_RUNS TEACHER_RUNS OUT_MD TASK [TASK ...]
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

COUNTED_STOPS = {"toolUse", "stop"}
REQUIRED_DROP = 0.25
PASSES_ALLOWED_LOST = 1


@dataclass(frozen=True)
class TaskResult:
    turns: int
    model_seconds: float
    passed: bool
    scout_steps: int
    teacher_seconds: float


def latest_trial(runs: Path, task: str) -> Path:
    trials = sorted(runs.glob(f"{task}-*/{task}__*"))
    if not trials:
        raise FileNotFoundError(f"no Harbor trial for {task} under {runs}")
    return trials[-1]


def summarise(runs: Path, tasks: list[str]) -> dict[str, TaskResult]:
    results = {}
    for task in tasks:
        trial = latest_trial(runs, task)
        lines = [json.loads(line) for line in (trial / "agent" / "jeff-first-trace.jsonl").read_text().splitlines()]
        turns = [l for l in lines if l.get("kind", "model_turn") == "model_turn" and l["action"]["stop_reason"] in COUNTED_STOPS]
        decisions = [l for l in lines if l.get("kind") == "decision"]
        reward = (json.loads((trial / "result.json").read_text()).get("verifier_result") or {}).get("rewards", {}).get("reward")
        if reward is None:
            raise ValueError(f"{trial} has no reward in result.json; the verifier did not finish")
        results[task] = TaskResult(
            turns=len(turns),
            model_seconds=sum(l["timings_ms"]["model"] for l in turns) / 1000,
            passed=reward == 1.0,
            scout_steps=sum(1 for d in decisions if d["action"]["kind"] == "step"),
            teacher_seconds=sum(d["timings_ms"]["chooser"] for d in decisions) / 1000,
        )
    return results


def drop(before: float, after: float) -> float:
    if before == 0:
        raise ValueError("the baseline has no model turns, so no drop can be measured")
    return (before - after) / before


def gate(base: dict[str, TaskResult], teacher: dict[str, TaskResult]) -> tuple[bool, str]:
    turns = drop(sum(r.turns for r in base.values()), sum(r.turns for r in teacher.values()))
    seconds = drop(sum(r.model_seconds for r in base.values()), sum(r.model_seconds for r in teacher.values()))
    lost = sum(r.passed for r in base.values()) - sum(r.passed for r in teacher.values())
    failures = []
    if turns < REQUIRED_DROP:
        failures.append(f"model turns fell {turns:.0%} (need {REQUIRED_DROP:.0%})")
    if seconds < REQUIRED_DROP:
        failures.append(f"model seconds fell {seconds:.0%} (need {REQUIRED_DROP:.0%})")
    if lost > PASSES_ALLOWED_LOST:
        failures.append(f"{lost} fewer tasks passed (at most {PASSES_ALLOWED_LOST} allowed)")
    summary = f"model turns {turns:+.0%} fewer, model seconds {seconds:+.0%} fewer, passes lost {lost}"
    if failures:
        return False, f"**Gate 0: fail.** {summary}. " + "; ".join(failures) + "."
    return True, f"**Gate 0: pass.** {summary}."


def render(base: dict[str, TaskResult], teacher: dict[str, TaskResult]) -> str:
    rows = [
        "| Task | Base turns | Teacher-arm turns | Base model s | Teacher-arm model s | Scout steps | Teacher s | Passed (base / teacher arm) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task in base:
        b, t = base[task], teacher[task]
        rows.append(
            f"| {task} | {b.turns} | {t.turns} | {b.model_seconds:.0f} | {t.model_seconds:.0f} | {t.scout_steps} "
            f"| {t.teacher_seconds:.0f} | {'yes' if b.passed else 'no'} / {'yes' if t.passed else 'no'} |"
        )
    _, line = gate(base, teacher)
    return "# Gate 0: oracle scout\n\n" + line + "\n\n" + "\n".join(rows) + "\n"


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
