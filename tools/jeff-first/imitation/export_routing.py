"""Export routing rows into Jeff training examples for the ROUTER adapter, split by task.

Input: JSON lines of routing rows, as `results/imitation/scripts/routing_labels.py join` writes them: one per labelled
Qwen turn, with the state Jeff saw before that turn (the stage-3 tool-level row of the turn's first decision, rendered
by teacher-prompt.ts renderState) and `routing.label`, the cheapest thinking level whose answer was as good as the
recorded xhigh one (off, low, medium, or xhigh when none was).

Output: `train.jsonl`, `development.jsonl` and `temperature.jsonl` of the `Example` records jeff-dev's train.py reads,
like export_jeff.py writes for the step adapter. One routing row becomes one example:

- `state`: the row's rendered state, unchanged.
- `question`: a "choice" question whose `instructions` is the router question and whose `criteria` are the four
  levels with their one-line meanings: ROUTER_QUESTION and ROUTER_OPTIONS, word for word the TypeScript ones in
  packages/coding-agent/src/core/jeff-first/router-question.ts (the scout's run-time router asks exactly this; a test
  checks the two copies are equal). The option order is shuffled with a seed taken from the example id.
- `label` and `target`: the routing label.
- `id`: <source>:route:<task>:<session>:<turn>. `suite` and `family`: the task.

Splits, by task (a task's rows all go to one file; held-out tasks are never exported):
- A bare Terminal-Bench 2.0 name: its split in splits.json ("terminal_bench": train, development or temperature); a
  name not there (a held-out task, or unknown) is an error.
- A hub task "<org>/<dataset>:<task>": task-sets.json must put it on the training side (held-out or excluded is an
  error); it goes to development when int(sha256('<seed>:<task id>'), 16) / 16**64 is below splits.json's stage-1
  development share, to temperature below the two shares together, else to train (the stage-1 rule, per task).

Usage (from tools/jeff-first):
    uv run python -m imitation.export_routing --rows routing-rows.jsonl --splits ../../results/imitation/splits.json \\
        --task-sets ../../results/imitation/task-sets.json --out export/routing-uncut/
Then cut the rows whose prompt is over Jeff's 8,192 tokens (jeff_fit.py's rule, the one the run time uses too):
    ~/mathias/apps/jeff-dev/.venv/bin/python jeff_prompt.py fit-examples --processor <Qwen3.5-0.8B processor folder> \\
        --layout live-last --workers 8 --unfittable fail --out export/routing/ export/routing-uncut/*.jsonl
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from imitation.export_jeff import SPLITS, _rank, shuffled_criteria
from task_source import is_hub_task, load_task_sets

ROUTER_QUESTION = "Before the coding model's next turn: how much should it think? Choose one option."
ROUTER_OPTIONS = {
    "off": "No thinking: it answers at once. Enough when the next step is obvious from what is shown.",
    "low": "Brief thinking before it answers.",
    "medium": "Some thinking before it answers.",
    "xhigh": "Long, careful thinking before it answers: it plans, checks its assumptions and considers other ways.",
}


def example_id(row: dict) -> str:
    return f"{row['source']}:route:{row['task']}:{row['session']}:{row['turn']}"


def to_example(row: dict) -> dict:
    identifier = example_id(row)
    label = row["routing"]["label"]
    if label not in ROUTER_OPTIONS:
        raise ValueError(f"routing row {identifier}: label {label!r} is not one of {', '.join(ROUTER_OPTIONS)}")
    options = [{"id": level, "description": text} for level, text in ROUTER_OPTIONS.items()]
    return {
        "id": identifier,
        "suite": row["task"],
        "family": row["task"],
        "state": row["state"],
        "question": {"type": "choice", "instructions": ROUTER_QUESTION, "criteria": shuffled_criteria(options, identifier)},
        "label": label,
        "target": label,
        "source": {
            "dataset": row["source"],
            "stage": row["stage"],
            "task": row["task"],
            "session": row["session"],
            "turn": row["turn"],
            "machine": row["machine"],
            "class": row["class"],
            "calibration": row["routing"]["calibration"],
        },
    }


def split_of(task: str, splits: dict, task_sets) -> str:
    if not is_hub_task(task):
        for name in SPLITS:
            if task in splits["terminal_bench"][name]:
                return name
        raise ValueError(f"the Terminal-Bench 2.0 task {task!r} is not in splits.json (held out, or unknown)")
    side = task_sets.side(task)
    if side != "training":
        raise ValueError(f"the task {task} is {side} in task-sets.json; only training tasks are exported")
    shares = splits["stage1"]["shares"]
    position = int(_rank(splits["seed"], task), 16) / 16**64
    if position < shares["development"]:
        return "development"
    return "temperature" if position < shares["development"] + shares["temperature"] else "train"


def export_routing(row_files: Sequence[Path], splits: dict, task_sets_path: Path, out: Path) -> dict[str, int]:
    """Write the three split files under `out` (which must not hold them yet); returns the examples per split. Every
    row is checked (label, split, no repeated id) before anything is written."""
    paths = {name: out / f"{name}.jsonl" for name in SPLITS}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose an empty --out folder")
    task_sets = load_task_sets(task_sets_path, None)
    placed: dict[str, list[dict]] = {name: [] for name in SPLITS}
    seen: set[str] = set()
    for row_file in row_files:
        for line in Path(row_file).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            example = to_example(row)
            if example["id"] in seen:
                raise ValueError(f"the routing row {example['id']} appears more than once")
            seen.add(example["id"])
            placed[split_of(row["task"], splits, task_sets)].append(example)
    if not seen:
        raise ValueError("No routing rows were read; refusing to write an empty export")
    out.mkdir(parents=True, exist_ok=True)
    for name, examples in placed.items():
        paths[name].write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples), encoding="utf-8")
    return {name: len(examples) for name, examples in placed.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, nargs="+", required=True, help="routing rows (routing_labels.py join)")
    parser.add_argument("--splits", type=Path, required=True, help="results/imitation/splits.json")
    parser.add_argument("--task-sets", type=Path, required=True, help="results/imitation/task-sets.json")
    parser.add_argument("--out", type=Path, required=True, help="Folder for train.jsonl, development.jsonl and temperature.jsonl")
    args = parser.parse_args(argv)
    counts = export_routing(args.rows, json.loads(args.splits.read_text()), args.task_sets, args.out)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
