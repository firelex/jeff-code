"""Export jeff-first imitation rows into the Jeff adapter kit's training-record format, split by task.

Input: JSON lines of the `Row` records `imitation/rows.py` writes (fields: source, stage, quality, task, session,
decision, turn, level, page, state, options, label, tool_description; stage 2 and 3 rows also carry `machine`).

Output: JSON lines of the `Example` records jeff-dev's `src/jeff/train.py` reads (via `jeff.evaluate.read_rows`), in
three files: `train.jsonl`, `development.jsonl` (checkpoint choice) and `temperature.jsonl` (calibration), the
`--train`, `--development` and `--temperature` that train.py requires. One imitation row becomes one `Example`:

- `state`: the row's rendered state text, unchanged (a plain-text state keeps the state-first prompt order).
- `question`: a "choice" question. `criteria` maps each option id to its description, the order shuffled by a random
  number generator seeded from the Example's own id (the same rows always shuffle the same way). `instructions` is the
  question the scout asks, word for word as teacher-prompt.ts `teacherMessages` writes it: "What should the next step
  be? Choose one option." for a tool row; "You have decided that the next step is: <tool_description>. Which one
  exactly? Choose one option." for an argument row, where tool_description is the chosen tool's own description the
  row carries; on page 2 or 3 both start with "You asked to see more options. This is page N; the options on earlier
  pages are not repeated here." and a line break.
- `label` and `target`: the row's label (an option id); train.py turns it into a one-hot target.
- `id`: source:stage:task:session:decision:level:page (unique). `suite`: the task. `family`: the task group
  (`task_group`), so train.py's check that no family is in two partitions also proves no task is.
- `source`: the row's provenance (dataset, stage, quality, task, session, decision, turn, level, page, and machine
  when the row has one).

Splits (`results/imitation/splits.json`, written once by the `splits` command from training-tasks.json and a seed):
- Stages 2 and 3 run on the same Terminal-Bench training tasks: 4 tasks go to development and 4 to temperature (the
  first 4 and next 4 in an order fixed by the seed), the rest to train, the same for both stages. A stage 2 or 3 row
  of any other task is an error.
- Stage 1 (public dataset tasks) is split 90/5/5 by task group: a group's place is fixed by a hash of the seed and
  the group name alone, so the split does not depend on which other tasks a conversion holds. The ukisai dataset
  names a second trial of a task "<task>__<trial label>" with the same task text; `task_group` cuts that suffix.

Usage (from tools/jeff-first):
    uv run python -m imitation.export_jeff splits --tasks ../../results/imitation/training-tasks.json --seed 20261003 \\
        --out ../../results/imitation/splits.json
    uv run python -m imitation.export_jeff export --rows stage1-rows.jsonl \\
        --splits ../../results/imitation/splits.json --out export/stage1/
"""

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

TOOL_QUESTION = "What should the next step be? Choose one option."
LATER_PAGE_PREFIX = (
    "You asked to see more options. This is page {page}; the options on earlier pages are not repeated here.\n"
)
SPLITS = ("train", "development", "temperature")
STAGE1_SHARES = {"development": 0.05, "temperature": 0.05}
BENCH_HELD_OUT = 4
STAGE1_TRIAL_SEPARATOR = "__"


def row_id(row: dict) -> str:
    """A stable, unique id for one row: no two rows of the same decision share a (level, page)."""
    return f"{row['source']}:{row['stage']}:{row['task']}:{row['session']}:{row['decision']}:{row['level']}:{row['page']}"


def question_text(row: dict) -> str:
    prefix = "" if row["page"] <= 1 else LATER_PAGE_PREFIX.format(page=row["page"])
    if row["level"] == "tool":
        return prefix + TOOL_QUESTION
    if "tool_description" not in row:
        raise ValueError(f"row {row_id(row)} has no tool_description: it was converted before rows carried it; convert it again")
    if not row["tool_description"]:
        raise ValueError(f"argument row {row_id(row)} has an empty tool_description")
    return prefix + f"You have decided that the next step is: {row['tool_description']}. Which one exactly? Choose one option."


def _seed_from_id(identifier: str) -> int:
    return int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16)


def shuffled_criteria(options: Sequence[dict], identifier: str) -> dict[str, str]:
    """The row's options as a dict from id to description, order shuffled by a seed taken from `identifier`, so
    rerunning the export on the same rows always shuffles the same way."""
    shuffled = list(options)
    random.Random(_seed_from_id(identifier)).shuffle(shuffled)
    return {option["id"]: option["description"] for option in shuffled}


def task_group(row: dict) -> str:
    """The tasks that must stay in one split: a stage 1 task with its retried trials (see the module docstring); a
    Terminal-Bench task (stages 2 and 3) alone."""
    if row["stage"] == 1:
        return row["task"].split(STAGE1_TRIAL_SEPARATOR, 1)[0]
    if STAGE1_TRIAL_SEPARATOR in row["task"]:
        raise ValueError(f"a stage {row['stage']} task name has {STAGE1_TRIAL_SEPARATOR!r} in it: {row['task']!r}")
    return row["task"]


def to_example(row: dict) -> dict:
    identifier = row_id(row)
    if row["label"] not in {option["id"] for option in row["options"]}:
        raise ValueError(f"row {identifier}: label {row['label']!r} is not among its options")
    source = {
        "dataset": row["source"],
        "stage": row["stage"],
        "quality": row["quality"],
        "task": row["task"],
        "session": row["session"],
        "decision": row["decision"],
        "turn": row["turn"],
        "level": row["level"],
        "page": row["page"],
    }
    if "machine" in row:
        source["machine"] = row["machine"]
    return {
        "id": identifier,
        "suite": row["task"],
        "family": task_group(row),
        "state": row["state"],
        "question": {
            "type": "choice",
            "instructions": question_text(row),
            "criteria": shuffled_criteria(row["options"], identifier),
        },
        "label": row["label"],
        "target": row["label"],
        "source": source,
    }


def _rank(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def make_splits(tasks_path: Path, seed: int) -> dict:
    """The split of the Terminal-Bench training tasks and the stage 1 rule (see the module docstring)."""
    training = json.loads(tasks_path.read_text())["training"]
    if len(set(training)) < 2 * BENCH_HELD_OUT + 1:
        raise ValueError(f"{tasks_path}: {len(set(training))} training tasks; at least {2 * BENCH_HELD_OUT + 1} are needed")
    order = sorted(set(training), key=lambda task: _rank(seed, task))
    return {
        "seed": seed,
        "tasks_file": tasks_path.name,
        "terminal_bench": {
            "rule": f"stages 2 and 3: the training tasks ordered by sha256('<seed>:<task>'); the first {BENCH_HELD_OUT} are development, the next {BENCH_HELD_OUT} temperature",
            "development": sorted(order[:BENCH_HELD_OUT]),
            "temperature": sorted(order[BENCH_HELD_OUT : 2 * BENCH_HELD_OUT]),
            "train": sorted(order[2 * BENCH_HELD_OUT :]),
        },
        "stage1": {
            "rule": "the task group (task name up to '__') goes to development when int(sha256('<seed>:<group>'), 16) / 16**64 is below the development share, to temperature below the two shares together, else to train",
            "shares": dict(STAGE1_SHARES),
        },
    }


def split_of(row: dict, splits: dict) -> str:
    if row["stage"] == 1:
        shares = splits["stage1"]["shares"]
        position = int(_rank(splits["seed"], task_group(row)), 16) / 16**64
        if position < shares["development"]:
            return "development"
        return "temperature" if position < shares["development"] + shares["temperature"] else "train"
    bench = splits["terminal_bench"]
    for name in SPLITS:
        if row["task"] in bench[name]:
            return name
    raise ValueError(f"the stage {row['stage']} task {row['task']!r} is not in splits.json's Terminal-Bench tasks")


def export_rows(row_files: Sequence[Path], splits: dict, out: Path) -> dict[str, int]:
    """Convert the rows of one stage into the three split files under `out` (which must not hold them yet), one row
    at a time; returns the number of examples per split."""
    paths = {name: out / f"{name}.jsonl" for name in SPLITS}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose an empty --out folder")
    out.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(SPLITS, 0)
    seen: set[str] = set()
    stages: set[int] = set()
    streams = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for row_file in row_files:
            with row_file.open(encoding="utf-8") as lines:
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    stages.add(row["stage"])
                    if len(stages) > 1:
                        raise ValueError(f"the rows mix stages {sorted(stages)}; export one stage at a time")
                    example = to_example(row)
                    if example["id"] in seen:
                        raise ValueError(f"the row id {example['id']} appears more than once")
                    seen.add(example["id"])
                    name = split_of(row, splits)
                    streams[name].write(json.dumps(example, ensure_ascii=False) + "\n")
                    counts[name] += 1
    finally:
        for stream in streams.values():
            stream.close()
    if not seen:
        raise ValueError("No rows were read; refusing to write an empty export")
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("splits", help="Write splits.json (once; it is never overwritten)")
    make.add_argument("--tasks", type=Path, required=True, help="results/imitation/training-tasks.json")
    make.add_argument("--seed", type=int, required=True)
    make.add_argument("--out", type=Path, required=True)
    export = commands.add_parser("export", help="Export one stage's rows into train/development/temperature files")
    export.add_argument("--rows", type=Path, nargs="+", required=True, help="JSONL files of imitation/rows.py Row records, all of one stage")
    export.add_argument("--splits", type=Path, required=True, help="splits.json written by the splits command")
    export.add_argument("--out", type=Path, required=True, help="Folder for train.jsonl, development.jsonl and temperature.jsonl")
    args = parser.parse_args(argv)
    if args.command == "splits":
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite {args.out}: the splits are fixed once written")
        args.out.write_text(json.dumps(make_splits(args.tasks, args.seed), indent=1) + "\n")
        print(args.out.read_text())
        return
    counts = export_rows(args.rows, json.loads(args.splits.read_text()), args.out)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
