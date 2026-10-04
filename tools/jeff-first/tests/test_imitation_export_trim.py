"""imitation/export_trim.py: trimming rows (trim labels joined onto stage-3 states) become Jeff training examples with
the trimming question read from output-trim.ts, split by task."""

import json
import subprocess
from pathlib import Path

import pytest

from imitation.export_trim import OUTPUT_TRIM_TS, export_trim, main, to_example, trim_question_text
from tests.test_task_source import write_task_sets

SPLITS = {
    "seed": 7,
    "terminal_bench": {"development": ["dev-task"], "temperature": ["temp-task"], "train": ["train-task", "other"]},
    "stage1": {"shares": {"development": 0.05, "temperature": 0.05}},
}

QUESTION = trim_question_text()


def trim_row(task="train-task", turn=3, label="last40", total=120) -> dict:
    return {
        "source": "own",
        "stage": 3,
        "task": task,
        "session": "s-1",
        "turn": turn,
        "machine": "qwen3.8-27b-nvfp4@b200-gpu1-vllm0.29",
        "source_host": "b200",
        "class": "information",
        "state": "Task:\nFix it.\n\nSteps so far, oldest first:\n\nStep 1 (by the coding model):\n$ seq 1 120\n...",
        "trim": {"label": label, "total_lines": total, "shown_lines": total, "routing_label": "off",
                 "checks": {}, "calibration": False},
    }


def test_the_question_comes_from_output_trim_ts_as_the_run_time_asks_it(tmp_path):
    script = tmp_path / "q.mjs"
    script.write_text(
        f'import {{ trimQuestion, TRIM_OPTIONS }} from "{OUTPUT_TRIM_TS.as_posix()}";\n'
        "process.stdout.write(JSON.stringify({ question: trimQuestion(120), options: TRIM_OPTIONS }));\n"
    )
    typescript = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout)
    example = to_example(trim_row(), QUESTION)
    assert example["question"]["instructions"] == typescript["question"]
    assert example["question"]["criteria"] == typescript["options"]
    assert QUESTION["choices"] == ["all", "last200", "last40", "first40", "first20last20"]


def test_a_row_becomes_an_example():
    example = to_example(trim_row(label="first20last20"), QUESTION)
    assert example["id"] == "own:trim:train-task:s-1:3"
    assert example["state"] == trim_row()["state"]
    assert example["question"]["type"] == "choice"
    assert "120 lines long" in example["question"]["instructions"]
    assert example["label"] == example["target"] == "first20last20"
    assert example["source"]["routing_label"] == "off"
    assert to_example(trim_row(label="first20last20"), QUESTION) == example


def test_the_option_order_is_shuffled_per_row():
    orders = {tuple(to_example(trim_row(turn=turn), QUESTION)["question"]["criteria"]) for turn in range(20)}
    assert len(orders) > 1


def test_bad_labels_and_lengths_are_errors():
    with pytest.raises(ValueError, match="label '40'"):
        to_example(trim_row(label="40"), QUESTION)
    with pytest.raises(ValueError, match="total_lines 40"):
        to_example(trim_row(total=40), QUESTION)


def test_splits_by_task_and_no_held_out_task(tmp_path):
    sets, _ = write_task_sets(tmp_path)
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(row) + "\n" for row in [
        trim_row(task="train-task"), trim_row(task="dev-task"), trim_row(task="temp-task")]))
    counts = export_trim([rows], SPLITS, sets, tmp_path / "out", QUESTION)
    assert counts == {"train": 1, "development": 1, "temperature": 1}
    rows.write_text(json.dumps(trim_row(task="terminal-bench-pro/terminal-bench-pro:held-c")) + "\n")
    with pytest.raises(ValueError, match="held_out"):
        export_trim([rows], SPLITS, sets, tmp_path / "out2", QUESTION)


def test_the_command_line_writes_the_three_files(tmp_path):
    sets, _ = write_task_sets(tmp_path)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(trim_row()) + "\n")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(SPLITS))
    main(["--rows", str(rows), "--splits", str(splits), "--task-sets", str(sets), "--out", str(tmp_path / "out")])
    assert len((tmp_path / "out" / "train.jsonl").read_text().splitlines()) == 1
    assert Path(tmp_path / "out" / "development.jsonl").read_text() == ""
