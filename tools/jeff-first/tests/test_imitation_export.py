import json

import pytest

from imitation.export_jeff import (
    choose_holdout_tasks,
    convert_rows,
    main,
    question_text,
    read_holdout_tasks,
    row_id,
    shuffled_criteria,
    split_by_task,
    to_example,
    tool_descriptions,
)


def opt(id_: str, description: str) -> dict:
    return {"id": id_, "description": description}


def tool_row(task="t", session="s", decision=0, page=1, label="read", options=None, turn=1) -> dict:
    return {
        "source": "src", "stage": 3, "quality": "exact", "task": task, "session": session,
        "decision": decision, "turn": turn, "level": "tool", "page": page,
        "state": "Task:\nFix it.", "options": options or [opt("read", "Read a file"), opt("hand_over", "Hand over")],
        "label": label,
    }


def argument_row(task="t", session="s", decision=0, page=1, label="read-1", options=None, turn=1) -> dict:
    return {
        "source": "src", "stage": 3, "quality": "exact", "task": task, "session": session,
        "decision": decision, "turn": turn, "level": "argument", "page": page,
        "state": "Task:\nFix it.",
        "options": options or [opt("read-1", "Read the file /app/main.py"), opt("none_of_these", "None of these")],
        "label": label,
    }


def test_row_id_is_unique_per_decision_level_and_page():
    a = tool_row(decision=0, page=1)
    b = tool_row(decision=0, page=2)
    c = tool_row(decision=1, page=1)
    assert len({row_id(a), row_id(b), row_id(c)}) == 3


def test_tool_descriptions_reads_the_committed_tool_rows_description():
    rows = [tool_row(label="read"), argument_row()]
    descriptions = tool_descriptions(rows)
    assert descriptions[("src", 3, "t", "s", 0)] == "Read a file"


def test_tool_descriptions_ignores_show_more_rows():
    rows = [tool_row(page=1, label="show_more"), tool_row(page=2, label="read")]
    descriptions = tool_descriptions(rows)
    assert descriptions[("src", 3, "t", "s", 0)] == "Read a file"


def test_tool_descriptions_rejects_two_committed_tool_rows_for_one_decision():
    rows = [tool_row(label="read"), tool_row(label="list", options=[opt("list", "List a folder")])]
    with pytest.raises(ValueError, match="more than one committed tool row"):
        tool_descriptions(rows)


def test_question_text_for_a_tool_row():
    row = tool_row(page=1)
    assert question_text(row, None) == "What should the next step be? Choose one option."


def test_question_text_for_a_tool_row_on_a_later_page():
    row = tool_row(page=2)
    text = question_text(row, None)
    assert text.startswith("You asked to see more options. This is page 2;")
    assert text.endswith("What should the next step be? Choose one option.")


def test_question_text_for_an_argument_row_uses_the_chosen_tools_description():
    row = argument_row(page=1)
    text = question_text(row, "Read a file")
    assert text == "You have decided that the next step is: Read a file. Which one exactly? Choose one option."


def test_question_text_for_an_argument_row_requires_a_tool_description():
    with pytest.raises(ValueError, match="chosen tool description"):
        question_text(argument_row(), None)


def test_shuffled_criteria_keeps_ids_and_descriptions_and_reruns_identically():
    options = [opt(f"o{i}", f"description {i}") for i in range(8)]
    first = shuffled_criteria(options, "same-id")
    second = shuffled_criteria(options, "same-id")
    assert first == second
    assert set(first) == {o["id"] for o in options}
    assert list(first) != [o["id"] for o in options]  # the identity order is vanishingly unlikely after a shuffle


def test_shuffled_criteria_differs_by_identifier():
    options = [opt(f"o{i}", f"description {i}") for i in range(8)]
    assert list(shuffled_criteria(options, "id-a")) != list(shuffled_criteria(options, "id-b"))


def test_to_example_builds_a_choice_example_with_a_one_hot_target():
    row = tool_row(label="read")
    example = to_example(row, None)
    assert example["question"]["type"] == "choice"
    assert set(example["question"]["criteria"]) == {"read", "hand_over"}
    assert example["label"] == "read"
    assert example["target"] == "read"
    assert example["suite"] == "t"
    assert example["family"] == "t:s"
    assert example["state"] == "Task:\nFix it."
    assert example["id"] == row_id(row)


def test_to_example_rejects_a_label_not_among_its_options():
    row = tool_row(label="nope")
    with pytest.raises(ValueError, match="not among its options"):
        to_example(row, None)


def test_convert_rows_wires_the_argument_rows_question_to_its_decisions_tool_row():
    rows = [tool_row(label="read"), argument_row(label="read-1")]
    examples = convert_rows(rows)
    argument_example = next(e for e in examples if e["source"]["level"] == "argument")
    assert argument_example["question"]["instructions"] == (
        "You have decided that the next step is: Read a file. Which one exactly? Choose one option."
    )


def test_convert_rows_handles_a_hand_over_decision_with_no_argument_row():
    rows = [tool_row(label="hand_over")]
    examples = convert_rows(rows)
    assert len(examples) == 1
    assert examples[0]["label"] == "hand_over"


def test_split_by_task_never_splits_a_tasks_rows():
    examples = [to_example(tool_row(task="a", decision=0), None), to_example(tool_row(task="b", decision=1), None)]
    train, holdout = split_by_task(examples, {"b"})
    assert [e["suite"] for e in train] == ["a"]
    assert [e["suite"] for e in holdout] == ["b"]


def test_choose_holdout_tasks_is_deterministic_and_sized_by_fraction():
    tasks = [f"task-{i}" for i in range(20)]
    first = choose_holdout_tasks(tasks, 0.1, seed=7)
    second = choose_holdout_tasks(tasks, 0.1, seed=7)
    assert first == second
    assert len(first) == 2


def test_choose_holdout_tasks_rejects_a_fraction_outside_zero_one():
    with pytest.raises(ValueError, match="--holdout-fraction"):
        choose_holdout_tasks(["a", "b"], 1.5, seed=0)


def test_read_holdout_tasks_strips_blank_lines(tmp_path):
    path = tmp_path / "holdout.txt"
    path.write_text("task-a\n\ntask-b\n")
    assert read_holdout_tasks(path) == {"task-a", "task-b"}


def test_main_writes_train_and_holdout_files_split_by_task(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows = [
        tool_row(task="train-task", session="s1", decision=0, label="read"),
        argument_row(task="train-task", session="s1", decision=0, label="read-1"),
        tool_row(task="held-task", session="s2", decision=0, label="hand_over"),
    ]
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    train_path, holdout_path = tmp_path / "train.jsonl", tmp_path / "holdout.jsonl"
    main([
        "--rows", str(rows_path), "--out-train", str(train_path), "--out-holdout", str(holdout_path),
        "--holdout-fraction", "0.5", "--seed", "1",
    ])
    train_examples = [json.loads(line) for line in train_path.read_text().splitlines()]
    holdout_examples = [json.loads(line) for line in holdout_path.read_text().splitlines()]
    assert len(train_examples) + len(holdout_examples) == 3
    assert {e["suite"] for e in train_examples}.isdisjoint({e["suite"] for e in holdout_examples})


def test_main_requires_exactly_one_holdout_method(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps(tool_row(label="hand_over")) + "\n")
    with pytest.raises(SystemExit):
        main([
            "--rows", str(rows_path), "--out-train", str(tmp_path / "t.jsonl"), "--out-holdout", str(tmp_path / "h.jsonl"),
        ])
