"""imitation/export_routing.py: routing rows (routing labels joined onto stage-3 states) become Jeff training examples
with the router question, split by task; the router question is the TypeScript one, word for word."""

import json
import subprocess
from pathlib import Path

import pytest

from imitation.export_routing import ROUTER_OPTIONS, ROUTER_QUESTION, export_routing, main, to_example
from tests.test_task_source import write_task_sets

JEFF_FIRST_TS = Path(__file__).resolve().parents[3] / "packages" / "coding-agent" / "src" / "core" / "jeff-first"

SPLITS = {
    "seed": 7,
    "terminal_bench": {"development": ["dev-task"], "temperature": ["temp-task"], "train": ["train-task", "other"]},
    "stage1": {"shares": {"development": 0.05, "temperature": 0.05}},
}


def routing_row(task="train-task", turn=3, label="off", **extra) -> dict:
    return {
        "source": "own",
        "stage": 3,
        "task": task,
        "session": "s-1",
        "turn": turn,
        "machine": "qwen3.8-27b-nvfp4@b200-gpu1-vllm0.29",
        "source_host": "b200",
        "class": "information",
        "state": "Task:\nFix it.\n\nNo steps have been taken yet.",
        "routing": {"label": label, "off_ok": label == "off", "low_ok": None, "medium_ok": None, "checked_by": {"off": "judge"}, "xhigh_actions": [], "calibration": False},
        **extra,
    }


def test_the_router_question_and_options_are_the_typescript_ones(tmp_path):
    script = tmp_path / "router.mjs"
    script.write_text(
        f'import {{ ROUTER_OPTIONS, ROUTER_QUESTION }} from "{(JEFF_FIRST_TS / "router-question.ts").as_posix()}";\n'
        "process.stdout.write(JSON.stringify({ question: ROUTER_QUESTION, options: ROUTER_OPTIONS }));\n"
    )
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True)
    typescript = json.loads(done.stdout)
    assert typescript["question"] == ROUTER_QUESTION
    assert list(typescript["options"].items()) == list(ROUTER_OPTIONS.items())


def test_a_row_becomes_an_example_with_the_router_question_and_the_four_levels():
    example = to_example(routing_row(label="medium"))
    assert example["id"] == "own:route:train-task:s-1:3"
    assert example["state"] == "Task:\nFix it.\n\nNo steps have been taken yet."
    assert example["question"]["type"] == "choice"
    assert example["question"]["instructions"] == ROUTER_QUESTION
    assert example["question"]["criteria"] == ROUTER_OPTIONS  # same pairs; the order is shuffled per row
    assert example["label"] == example["target"] == "medium"
    assert example["suite"] == example["family"] == "train-task"
    assert example["source"] == {
        "dataset": "own", "stage": 3, "task": "train-task", "session": "s-1", "turn": 3,
        "machine": "qwen3.8-27b-nvfp4@b200-gpu1-vllm0.29", "class": "information", "calibration": False,
    }
    assert to_example(routing_row(label="medium")) == example  # the shuffle is fixed by the row id


def test_the_option_order_is_shuffled_per_row():
    orders = {tuple(to_example(routing_row(turn=turn))["question"]["criteria"]) for turn in range(20)}
    assert len(orders) > 1


def test_an_unknown_label_is_an_error():
    with pytest.raises(ValueError, match="label 'high'"):
        to_example(routing_row(label="high"))


def test_terminal_bench_rows_go_to_their_task_split_and_hub_training_rows_by_a_hash(tmp_path):
    sets, _ = write_task_sets(tmp_path)
    rows = tmp_path / "rows.jsonl"
    hub_rows = [routing_row(task="terminal-bench-pro/terminal-bench-pro:fix-a", turn=t) for t in range(3)]
    rows.write_text("".join(json.dumps(row) + "\n" for row in [
        routing_row(task="train-task"), routing_row(task="dev-task"), routing_row(task="temp-task"), *hub_rows,
    ]))
    counts = export_routing([rows], SPLITS, sets, tmp_path / "out")
    assert sum(counts.values()) == 6
    read = lambda name: [json.loads(line) for line in (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines()]
    assert [example["suite"] for example in read("development")] == ["dev-task"]
    assert [example["suite"] for example in read("temperature")] == ["temp-task"]
    train = [example["suite"] for example in read("train")]
    assert "train-task" in train
    # All rows of one hub task land in the same split.
    hub_splits = {name for name in ("train", "development", "temperature") for example in read(name) if example["suite"].startswith("terminal-bench-pro/")}
    assert len(hub_splits) == 1


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ("terminal-bench-pro/terminal-bench-pro:held-c", "held_out"),
        ("terminal-bench-pro/terminal-bench-pro:leak-d", "excluded"),
        ("not-a-training-task", "not in splits.json"),
    ],
)
def test_held_out_excluded_and_unknown_tasks_are_never_exported(tmp_path, task, message):
    sets, _ = write_task_sets(tmp_path)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(routing_row(task=task)) + "\n")
    with pytest.raises(ValueError, match=message):
        export_routing([rows], SPLITS, sets, tmp_path / "out")


def test_a_repeated_row_and_an_existing_output_are_errors(tmp_path):
    sets, _ = write_task_sets(tmp_path)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(routing_row()) + "\n" + json.dumps(routing_row()) + "\n")
    with pytest.raises(ValueError, match="more than once"):
        export_routing([rows], SPLITS, sets, tmp_path / "out")
    rows.write_text(json.dumps(routing_row()) + "\n")
    export_routing([rows], SPLITS, sets, tmp_path / "out2")
    with pytest.raises(FileExistsError):
        export_routing([rows], SPLITS, sets, tmp_path / "out2")


def test_the_command_line_writes_the_three_files(tmp_path):
    sets, _ = write_task_sets(tmp_path)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(routing_row()) + "\n")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(SPLITS))
    main(["--rows", str(rows), "--splits", str(splits), "--task-sets", str(sets), "--out", str(tmp_path / "out")])
    assert len((tmp_path / "out" / "train.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "out" / "development.jsonl").read_text() == ""
