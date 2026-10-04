"""Export output-trimming rows into Jeff training examples for the TRIMMING adapter, split by task.

Input: JSON lines of trimming rows, as `results/imitation/scripts/trim_labels.py join` writes them: one per labelled
Qwen turn whose newest tool output showed more than 40 lines, with the state Jeff saw before that turn (the stage-3
tool-level row of the turn's first decision, rendered by teacher-prompt.ts renderState: its last step is that output,
shown by its last lines) and `trim.label`, how much of the output the coding model needed (all, last200, last40,
first40 or first20last20) and `trim.total_lines`, the output's length.

Output: `train.jsonl`, `development.jsonl` and `temperature.jsonl` of the `Example` records jeff-dev's train.py reads,
as export_routing.py writes them for the router adapter. One row becomes one example:

- `state`: the row's rendered state, unchanged.
- `question`: a "choice" question whose `instructions` is the trimming question for the output's length and whose
  `criteria` are the five choices with their one-line meanings. Both are read from the one place that defines them,
  packages/coding-agent/src/core/jeff-first/output-trim.ts (TRIM_QUESTION_TEMPLATE, TRIM_OPTIONS, TRIM_CHOICES), by
  running node on it: the run-time trimmer asks exactly this text. The option order is shuffled per example id.
- `label` and `target`: the trimming label.
- `id`: <source>:trim:<task>:<session>:<turn>. `suite` and `family`: the task.

Splits by task, exactly as export_routing.py (splits.json for Terminal-Bench 2.0, the stage-1 rule per hub task;
held-out, excluded and unknown tasks are errors).

Usage (from tools/jeff-first; needs node 22.6 or later, which runs TypeScript):
    uv run python -m imitation.export_trim --rows trim-rows.jsonl --splits ../../results/imitation/splits.json \\
        --task-sets ../../results/imitation/task-sets.json --out export/trim/
"""

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from imitation.export_jeff import SPLITS, shuffled_criteria
from imitation.export_routing import split_of
from task_source import load_task_sets

OUTPUT_TRIM_TS = (Path(__file__).resolve().parents[3] / "packages" / "coding-agent" / "src" / "core" / "jeff-first"
                  / "output-trim.ts")


def trim_question_text(node: str = "node") -> dict:
    """{"template", "options" (choice -> meaning), "choices" (in order)} from output-trim.ts, through node."""
    script = (f'const m = await import({json.dumps(OUTPUT_TRIM_TS.as_uri())});\n'
              "process.stdout.write(JSON.stringify({ template: m.TRIM_QUESTION_TEMPLATE, options: m.TRIM_OPTIONS, "
              "choices: m.TRIM_CHOICES }));\n")
    done = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"node could not read the trimming question from {OUTPUT_TRIM_TS}: {done.stderr[-2000:]}")
    text = json.loads(done.stdout)
    if "{lines}" not in text["template"] or list(text["options"]) != text["choices"]:
        raise ValueError(f"unexpected trimming question definitions in {OUTPUT_TRIM_TS}: {text}")
    return text


def example_id(row: dict) -> str:
    return f"{row['source']}:trim:{row['task']}:{row['session']}:{row['turn']}"


def to_example(row: dict, question: dict) -> dict:
    identifier = example_id(row)
    trim = row["trim"]
    label = trim["label"]
    if label not in question["options"]:
        raise ValueError(f"trimming row {identifier}: label {label!r} is not one of {', '.join(question['options'])}")
    if not isinstance(trim["total_lines"], int) or trim["total_lines"] <= 40:
        raise ValueError(f"trimming row {identifier}: total_lines {trim['total_lines']!r} is not a length above 40")
    options = [{"id": choice, "description": question["options"][choice]} for choice in question["choices"]]
    return {
        "id": identifier,
        "suite": row["task"],
        "family": row["task"],
        "state": row["state"],
        "question": {"type": "choice",
                     "instructions": question["template"].replace("{lines}", str(trim["total_lines"])),
                     "criteria": shuffled_criteria(options, identifier)},
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
            "routing_label": trim["routing_label"],
            "shown_lines": trim["shown_lines"],
        },
    }


def export_trim(row_files: Sequence[Path], splits: dict, task_sets_path: Path, out: Path, question: dict) -> dict[str, int]:
    """Write the three split files under `out` (which must not hold them yet); returns the examples per split. Every
    row is checked (label, length, split, no repeated id) before anything is written."""
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
            example = to_example(row, question)
            if example["id"] in seen:
                raise ValueError(f"the trimming row {example['id']} appears more than once")
            seen.add(example["id"])
            placed[split_of(row["task"], splits, task_sets)].append(example)
    if not seen:
        raise ValueError("No trimming rows were read; refusing to write an empty export")
    out.mkdir(parents=True, exist_ok=True)
    for name, examples in placed.items():
        paths[name].write_text("".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples),
                               encoding="utf-8")
    return {name: len(examples) for name, examples in placed.items()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--task-sets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", default="node")
    args = parser.parse_args(argv)
    counts = export_trim(args.rows, json.loads(args.splits.read_text(encoding="utf-8")), args.task_sets, args.out,
                         trim_question_text(args.node))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
