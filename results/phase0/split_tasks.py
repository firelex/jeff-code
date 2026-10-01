"""Split Terminal-Bench 2.0 into the fixed evaluation subset and the phase-0 shadow-run tasks.

Usage: python3 split_tasks.py <path to a terminal-bench-2 checkout> > tasks.json

Both lists are drawn with a fixed seed, stratified by the difficulty each task declares in task.toml,
so the split can be reproduced exactly. The evaluation subset (40 tasks) is never used for traces or
training. The phase-0 tasks (20) come only from the remaining 49 and, to keep the run short, only
from tasks whose agent time limit is at most 30 minutes.
"""

import json
import random
import subprocess
import sys
import tomllib
from pathlib import Path

SEED = 20261001
EVALUATION = {"easy": 2, "medium": 25, "hard": 13}  # 40, proportional to 4 / 55 / 30
PHASE0 = {"easy": 1, "medium": 12, "hard": 7}  # 20
PHASE0_MAX_AGENT_SECONDS = 1800


def main() -> None:
    root = Path(sys.argv[1])
    tasks = {}
    for path in sorted(root.glob("*/task.toml")):
        meta = tomllib.loads(path.read_text())
        tasks[path.parent.name] = {
            "difficulty": meta["metadata"]["difficulty"],
            "category": meta["metadata"]["category"],
            "agent_timeout_sec": meta["agent"]["timeout_sec"],
        }
    if len(tasks) != 89:
        raise SystemExit(f"expected 89 Terminal-Bench 2.0 tasks, found {len(tasks)}")

    rng = random.Random(SEED)
    evaluation: list[str] = []
    for difficulty, count in EVALUATION.items():
        pool = sorted(name for name, t in tasks.items() if t["difficulty"] == difficulty)
        evaluation += rng.sample(pool, count)

    phase0: list[str] = []
    for difficulty, count in PHASE0.items():
        pool = sorted(
            name
            for name, t in tasks.items()
            if name not in evaluation
            and t["difficulty"] == difficulty
            and t["agent_timeout_sec"] <= PHASE0_MAX_AGENT_SECONDS
        )
        if len(pool) < count:
            raise SystemExit(f"only {len(pool)} {difficulty} tasks left for phase 0, need {count}")
        phase0 += rng.sample(pool, count)

    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    json.dump(
        {
            "source": "github.com/harbor-framework/terminal-bench-2",
            "commit": commit.stdout.strip(),
            "seed": SEED,
            "evaluation": sorted(evaluation),
            "phase0": sorted(phase0),
            "tasks": {name: tasks[name] for name in sorted(set(evaluation) | set(phase0))},
        },
        sys.stdout,
        indent=1,
    )
    print()


main()
