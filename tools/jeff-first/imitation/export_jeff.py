"""Export jeff-first imitation rows into the Jeff adapter kit's training-record format.

Input: JSON lines of the `Row` records `imitation/rows.py` writes (fields: source, stage, quality, task, session,
decision, turn, level, page, state, options, label; see that module's docstring).

Output: JSON lines of the `Example` records `src/jeff/train.py` (jeff-dev, via `jeff.evaluate.read_rows`) consumes
for training. One imitation row becomes one `Example`:

- `state`: the row's rendered state text, unchanged (a plain string; v1.3's "state is text" layout keeps it first).
- `question`: a "choice" question. `criteria` is the row's options, as a dict from option id to its description,
  with the order shuffled by a seeded random number generator so a reused option id does not sit in the same
  position across rows (the seed comes from the Example's own id, so the same input rows always shuffle the same
  way). `instructions` is the plain-English question Jeff is asked, reusing the exact wording the scout's teacher
  prompt uses (packages/coding-agent/src/core/jeff-first/teacher-prompt.ts, `teacherMessages`): for a "tool" row,
  "What should the next step be? Choose one option."; for an "argument" row, "You have decided that the next step
  is: <the chosen tool's description>. Which one exactly? Choose one option." (plus, on page 2 or 3, the "You asked
  to see more options..." prefix teacherMessages adds). The chosen tool's description is read from this decision's
  own "tool" row: the one, among the rows sharing the same source/stage/task/session/decision, whose level is
  "tool" and whose label is not "show_more" (see `_tool_descriptions`); teacher-prompt.ts instead reads the tool's
  own plain description directly from its argument list, a field `rows.py` does not keep, so this is the closest
  available text, not a byte-for-byte match -- see the export report for why that is not fixable here without
  changing rows.py.
- `label` and `target`: both set to the row's label (one of the option ids). `jeff.train.targets` turns an
  unlisted, non-list target for a "choice" question into a one-hot distribution over the (possibly re-shuffled at
  training time) option order, so this is exactly a one-hot target at the chosen option.
- `id`, `suite`, `family`, `source`: provenance. `id` is unique per row (source, stage, task, session, decision,
  level, page). `suite` is the task name. `family` is `<task>:<session>`, so a holdout split by task never splits a
  family across train and holdout (required: `jeff.train.main` refuses overlapping families between partitions).

Usage:
    uv run python -m imitation.export_jeff --rows rows.jsonl --out-train train.jsonl --out-holdout holdout.jsonl \\
        --holdout-fraction 0.1 --seed 20261003
    uv run python -m imitation.export_jeff --rows rows.jsonl --out-train train.jsonl --out-holdout holdout.jsonl \\
        --holdout-tasks holdout-tasks.txt
"""

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

TOOL_QUESTION = "What should the next step be? Choose one option."
LATER_PAGE_PREFIX = (
    "You asked to see more options. This is page {page}; the options on earlier pages are not repeated here.\n"
)


def read_rows(paths: Sequence[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: dict) -> str:
    """A stable, unique id for one row: no two rows of the same decision share a (level, page)."""
    return f"{row['source']}:{row['stage']}:{row['task']}:{row['session']}:{row['decision']}:{row['level']}:{row['page']}"


def _decision_key(row: dict) -> tuple:
    return (row["source"], row["stage"], row["task"], row["session"], row["decision"])


def tool_descriptions(rows: Sequence[dict]) -> dict[tuple, str]:
    """For every decision that has a committed tool choice (label not "show_more"), the description of the chosen
    tool option, read from that decision's own "tool"-level row. Raises if a decision has more than one such row,
    or if the chosen option is missing from that row's own options (both would mean the input rows are internally
    inconsistent, not something to paper over)."""
    result: dict[tuple, str] = {}
    for row in rows:
        if row["level"] != "tool" or row["label"] == "show_more":
            continue
        key = _decision_key(row)
        if key in result:
            raise ValueError(f"decision {key} has more than one committed tool row")
        chosen = next((option for option in row["options"] if option["id"] == row["label"]), None)
        if chosen is None:
            raise ValueError(f"row {row_id(row)}: its own label {row['label']!r} is not among its options")
        result[key] = chosen["description"]
    return result


def question_text(row: dict, tool_description: str | None) -> str:
    prefix = "" if row["page"] <= 1 else LATER_PAGE_PREFIX.format(page=row["page"])
    if row["level"] == "tool":
        return prefix + TOOL_QUESTION
    if tool_description is None:
        raise ValueError(f"row {row_id(row)}: an argument row needs its decision's chosen tool description")
    return prefix + f"You have decided that the next step is: {tool_description}. Which one exactly? Choose one option."


def _seed_from_id(identifier: str) -> int:
    return int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16)


def shuffled_criteria(options: Sequence[dict], identifier: str) -> dict[str, str]:
    """The row's options as a dict from id to description, order shuffled by a seed taken from `identifier`, so
    rerunning the export on the same rows always shuffles the same way."""
    shuffled = list(options)
    random.Random(_seed_from_id(identifier)).shuffle(shuffled)
    return {option["id"]: option["description"] for option in shuffled}


def to_example(row: dict, tool_description: str | None) -> dict:
    identifier = row_id(row)
    if row["label"] not in {option["id"] for option in row["options"]}:
        raise ValueError(f"row {identifier}: label {row['label']!r} is not among its options")
    return {
        "id": identifier,
        "suite": row["task"],
        "family": f"{row['task']}:{row['session']}",
        "state": row["state"],
        "question": {
            "type": "choice",
            "instructions": question_text(row, tool_description),
            "criteria": shuffled_criteria(row["options"], identifier),
        },
        "label": row["label"],
        "target": row["label"],
        "source": {
            "dataset": row["source"],
            "stage": row["stage"],
            "quality": row["quality"],
            "task": row["task"],
            "session": row["session"],
            "decision": row["decision"],
            "turn": row["turn"],
            "level": row["level"],
            "page": row["page"],
        },
    }


def convert_rows(rows: Sequence[dict]) -> list[dict]:
    descriptions = tool_descriptions(rows)
    return [to_example(row, descriptions.get(_decision_key(row))) for row in rows]


def read_holdout_tasks(path: Path) -> set[str]:
    tasks = {line.strip() for line in path.read_text().splitlines()}
    tasks.discard("")
    return tasks


def choose_holdout_tasks(tasks: Sequence[str], fraction: float, seed: int) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError(f"--holdout-fraction must be between 0 and 1, not {fraction}")
    ordered = sorted(set(tasks))
    random.Random(seed).shuffle(ordered)
    count = round(fraction * len(ordered))
    return set(ordered[:count])


def split_by_task(examples: Sequence[dict], holdout_tasks: set[str]) -> tuple[list[dict], list[dict]]:
    """Train/holdout, split by `suite` (the task) so a task's rows are never divided between the two, as the scout
    design's calibration split requires."""
    train = [example for example in examples if example["suite"] not in holdout_tasks]
    holdout = [example for example in examples if example["suite"] in holdout_tasks]
    return train, holdout


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, nargs="+", required=True, help="One or more JSONL files of imitation/rows.py Row records")
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-holdout", type=Path, required=True)
    parser.add_argument("--holdout-tasks", type=Path, help="A file naming held-out tasks, one per line")
    parser.add_argument("--holdout-fraction", type=float, help="Hold out this fraction of distinct tasks, chosen with --seed")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --holdout-fraction's task choice")
    args = parser.parse_args(argv)
    if bool(args.holdout_tasks) == bool(args.holdout_fraction is not None):
        parser.error("Give exactly one of --holdout-tasks or --holdout-fraction")
    rows = read_rows(args.rows)
    if not rows:
        raise ValueError("No rows were read; refusing to write an empty training set")
    examples = convert_rows(rows)
    holdout_tasks = (
        read_holdout_tasks(args.holdout_tasks)
        if args.holdout_tasks
        else choose_holdout_tasks([example["suite"] for example in examples], args.holdout_fraction, args.seed)
    )
    train, holdout = split_by_task(examples, holdout_tasks)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_holdout, holdout)
    print(json.dumps({"rows": len(rows), "train": len(train), "holdout": len(holdout), "holdout_tasks": sorted(holdout_tasks)}))


if __name__ == "__main__":
    main()
