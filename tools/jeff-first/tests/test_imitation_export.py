import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from imitation.export_jeff import (
    STAGE1_SHARES,
    export_rows,
    read_over_length,
    length_summary,
    main,
    make_splits,
    question_text,
    row_id,
    shuffled_criteria,
    split_of,
    task_group,
    to_example,
)
from imitation.rows import Choice, RowSource, rows_for_decision
from imitation.stage3 import convert_runs, read_tasks

REPO = Path(__file__).resolve().parents[3]
JEFF_FIRST_TS = REPO / "packages" / "coding-agent" / "src" / "core" / "jeff-first"
FIXTURES = Path(__file__).parent / "fixtures" / "stage3"


def opt(id_: str, description: str) -> dict:
    return {"id": id_, "description": description}


def tool_row(task="t", session="s", decision=0, page=1, label="read", options=None, turn=1, stage=3) -> dict:
    return {
        "source": "src", "stage": stage, "quality": "exact", "task": task, "session": session,
        "decision": decision, "turn": turn, "level": "tool", "page": page,
        "state": "Task:\nFix it.", "options": options or [opt("read", "Read a file: Read the file /app/main.py (1 option)"), opt("hand_over", "Hand over")],
        "label": label, "tool_description": None,
    }


def argument_row(task="t", session="s", decision=0, page=1, label="read-1", turn=1, stage=3, tool_description="Read a file") -> dict:
    return {
        "source": "src", "stage": stage, "quality": "exact", "task": task, "session": session,
        "decision": decision, "turn": turn, "level": "argument", "page": page,
        "state": "Task:\nFix it.",
        "options": [opt("read-1", "Read the file /app/main.py"), opt("none_of_these", "None of these")],
        "label": label, "tool_description": tool_description,
    }


TASKS = {
    "training": [f"tb-{n:02d}" for n in range(45)],
    "excluded_evaluation": ["fix-git"],
    "excluded_leak_twins": {},
}


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_row_id_is_unique_per_decision_level_and_page():
    a = tool_row(decision=0, page=1)
    b = tool_row(decision=0, page=2)
    c = tool_row(decision=1, page=1)
    assert len({row_id(a), row_id(b), row_id(c)}) == 3


def test_question_text_for_tool_rows_and_later_pages():
    assert question_text(tool_row(page=1)) == "What should the next step be? Choose one option."
    text = question_text(tool_row(page=2))
    assert text == (
        "You asked to see more options. This is page 2; the options on earlier pages are not repeated here.\n"
        "What should the next step be? Choose one option."
    )


def test_question_text_for_an_argument_row_uses_the_rows_own_tool_description():
    assert question_text(argument_row(tool_description="Read part or all of a file")) == (
        "You have decided that the next step is: Read part or all of a file. Which one exactly? Choose one option."
    )


def test_an_argument_row_without_a_tool_description_is_an_error():
    row = argument_row()
    del row["tool_description"]
    with pytest.raises(ValueError, match="tool_description"):
        question_text(row)
    with pytest.raises(ValueError, match="tool_description"):
        question_text(argument_row(tool_description=None))


def test_shuffled_criteria_keeps_ids_and_descriptions_and_reruns_identically():
    options = [opt(f"o{i}", f"description {i}") for i in range(8)]
    first = shuffled_criteria(options, "same-id")
    assert first == shuffled_criteria(options, "same-id")
    assert set(first) == {o["id"] for o in options}
    assert list(first) != [o["id"] for o in options]
    assert list(shuffled_criteria(options, "id-a")) != list(shuffled_criteria(options, "id-b"))


def test_task_group_joins_a_stage_1_task_with_its_retried_trials():
    # ukisai names a second trial of a task "<task>__<trial label>"; both have the same task text.
    assert task_group({"stage": 1, "task": "inferredbugs-0001__dsv4-trial-2"}) == "inferredbugs-0001"
    assert task_group({"stage": 1, "task": "inferredbugs-0001"}) == "inferredbugs-0001"
    assert task_group({"stage": 3, "task": "dna-assembly"}) == "dna-assembly"
    with pytest.raises(ValueError, match="__"):
        task_group({"stage": 2, "task": "a__b"})


def test_to_example_builds_a_choice_example_with_a_one_hot_target():
    row = {**tool_row(label="read"), "machine": "qwen3.8-27b-fp8@casdgx01-gpu5"}
    example = to_example(row)
    assert example["question"]["type"] == "choice"
    assert set(example["question"]["criteria"]) == {"read", "hand_over"}
    assert example["label"] == example["target"] == "read"
    assert example["suite"] == "t" and example["family"] == "t"
    assert example["id"] == row_id(row)
    assert example["source"]["machine"] == "qwen3.8-27b-fp8@casdgx01-gpu5"
    with pytest.raises(ValueError, match="not among its options"):
        to_example(tool_row(label="nope"))


def test_make_splits_picks_4_and_4_terminal_bench_tasks_with_a_seed(tmp_path):
    tasks = write_json(tmp_path / "tasks.json", TASKS)
    first = make_splits(tasks, seed=20261003)
    assert first == make_splits(tasks, seed=20261003)
    bench = first["terminal_bench"]
    assert len(bench["development"]) == 4 and len(bench["temperature"]) == 4 and len(bench["train"]) == 37
    assert set(bench["development"]) | set(bench["temperature"]) | set(bench["train"]) == set(TASKS["training"])
    assert not set(bench["development"]) & set(bench["temperature"])
    assert first["stage1"]["shares"] == STAGE1_SHARES
    assert make_splits(tasks, seed=1)["terminal_bench"]["development"] != bench["development"]


def test_split_of_puts_stage_2_and_3_rows_of_one_task_in_the_same_split(tmp_path):
    splits = make_splits(write_json(tmp_path / "tasks.json", TASKS), seed=5)
    development = splits["terminal_bench"]["development"][0]
    assert split_of({"stage": 2, "task": development}, splits) == "development"
    assert split_of({"stage": 3, "task": development}, splits) == "development"
    with pytest.raises(ValueError, match="not in splits"):
        split_of({"stage": 3, "task": "make-doom-for-mips"}, splits)


def test_split_of_stage_1_keeps_a_task_group_together_and_is_near_90_5_5(tmp_path):
    splits = make_splits(write_json(tmp_path / "tasks.json", TASKS), seed=5)
    names = [split_of({"stage": 1, "task": f"nl2bash-{n:04d}"}, splits) for n in range(4000)]
    assert 0.04 < names.count("development") / 4000 < 0.06
    assert 0.04 < names.count("temperature") / 4000 < 0.06
    for n in range(200):
        assert split_of({"stage": 1, "task": f"x-{n}"}, splits) == split_of({"stage": 1, "task": f"x-{n}__dsv4-trial-2"}, splits)


def test_export_writes_three_disjoint_splits_by_task(tmp_path):
    splits = make_splits(write_json(tmp_path / "tasks.json", TASKS), seed=5)
    bench = splits["terminal_bench"]
    rows = [
        tool_row(task=bench["train"][0], session="s1"),
        argument_row(task=bench["train"][0], session="s1"),
        tool_row(task=bench["development"][0], session="s2", label="hand_over"),
        tool_row(task=bench["temperature"][0], session="s3", label="hand_over"),
    ]
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = export_rows([rows_path], splits, tmp_path / "out")
    assert counts == {"train": 2, "development": 1, "temperature": 1}
    written = {name: [json.loads(line) for line in (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines()] for name in counts}
    assert [e["suite"] for e in written["development"]] == [bench["development"][0]]
    with pytest.raises(FileExistsError):
        export_rows([rows_path], splits, tmp_path / "out")


def test_export_refuses_mixed_stages_and_duplicate_ids(tmp_path):
    splits = make_splits(write_json(tmp_path / "tasks.json", TASKS), seed=5)
    task = splits["terminal_bench"]["train"][0]
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps(tool_row(task=task, stage=3)) + "\n" + json.dumps(tool_row(task=task, stage=2, decision=1)) + "\n")
    with pytest.raises(ValueError, match="stage"):
        export_rows([mixed], splits, tmp_path / "a")
    twice = tmp_path / "twice.jsonl"
    twice.write_text(json.dumps(tool_row(task=task)) + "\n" + json.dumps(tool_row(task=task)) + "\n")
    with pytest.raises(ValueError, match="more than once"):
        export_rows([twice], splits, tmp_path / "b")


def test_cli_writes_splits_then_exports(tmp_path):
    tasks = write_json(tmp_path / "tasks.json", TASKS)
    splits_path = tmp_path / "splits.json"
    main(["splits", "--tasks", str(tasks), "--seed", "3", "--out", str(splits_path)])
    with pytest.raises(FileExistsError):
        main(["splits", "--tasks", str(tasks), "--seed", "3", "--out", str(splits_path)])
    task = json.loads(splits_path.read_text())["terminal_bench"]["train"][0]
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps(tool_row(task=task, label="hand_over")) + "\n")
    main(["export", "--rows", str(rows_path), "--splits", str(splits_path), "--out", str(tmp_path / "out")])
    assert len((tmp_path / "out" / "train.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "out" / "development.jsonl").read_text() == ""


TEACHER_SCRIPT = """
import { readFileSync } from "node:fs";
import { teacherMessages } from "%(ts)s/teacher-prompt.ts";
import { argumentPage, toolPage } from "%(ts)s/pages.ts";
const cases = JSON.parse(readFileSync(0, "utf8"));
const out = cases.map(({ state, lists, level, page, kind }) => {
  const menu = { tools: lists.tools, argumentsByTool: lists.arguments_by_tool };
  const asked = level === "tool"
    ? { level, page, options: toolPage(menu, page) }
    : { level, page, tool: menu.tools.find((tool) => tool.id === kind), options: argumentPage(menu.argumentsByTool[kind], page) };
  return teacherMessages(state, asked)[1].content;
});
process.stdout.write(JSON.stringify(out));
"""


def teacher_contents(tmp_path: Path, cases: list[dict]) -> list[str]:
    script = tmp_path / "teacher.mjs"
    script.write_text(TEACHER_SCRIPT % {"ts": JEFF_FIRST_TS.as_posix()})
    done = subprocess.run(["node", str(script)], input=json.dumps(cases), capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def assert_same_prompt(example: dict, content: str) -> None:
    """teacherMessages' user message is: state, blank line, question, the options lettered A, B, ..., then the
    answer instruction. The exported example must hold the same state and question text and the same options."""
    head = f"{example['state']}\n\n{example['question']['instructions']}\n"
    assert content.startswith(head)
    listed = content[len(head):].split("\n\n")[0].split("\n")
    assert sorted(line.split(": ", 1)[1] for line in listed) == sorted(example["question"]["criteria"].values())


def test_a_real_record_rows_question_and_state_equal_the_typescript_prompt(tmp_path):
    tasks = write_json(tmp_path / "tasks.json", {"training": ["dna-assembly"], "excluded_evaluation": [], "excluded_leak_twins": {}})
    rows = convert_runs([FIXTURES / "runs-imitation-v5"], read_tasks(tasks)).rows
    trace = next(FIXTURES.glob("runs-imitation-v5/*/round-1/*/*/agent/jeff-first-trace.jsonl"))
    records = {line["turn"]: line for line in map(json.loads, trace.read_text().splitlines())}
    assert [row["level"] for row in rows if row["turn"] == 2] == ["tool", "argument"]
    cases = [
        {"state": records[row["turn"]]["state"], "lists": records[row["turn"]]["lists"], "level": row["level"], "page": row["page"],
         "kind": None if row["level"] == "tool" else row["label"].rsplit("-", 1)[0]}
        for row in rows
    ]
    for row, content in zip(rows, teacher_contents(tmp_path, cases), strict=True):
        assert_same_prompt(to_example(row), content)


def test_later_page_questions_equal_the_typescript_prompt(tmp_path):
    lists = {
        "tools": [{"id": "read", "description": "Read part or all of a file"}, {"id": "hand_over", "description": "Hand over to the coding model for its next turn"}],
        "arguments_by_tool": {
            "read": [{"id": f"read-{n}", "description": f"Read the file /app/f{n}.py", "toolCall": {"name": "bash", "arguments": {"command": "x"}}} for n in range(1, 13)]
        },
    }
    state = {"task": "Fix it.", "recentSteps": [], "stepsLeftOut": 0}
    meta = RowSource(source="own", stage=3, quality="exact", task="t", session="s")
    rows = [asdict(row) for row in rows_for_decision(meta, 0, 1, "Task:\nFix it.\n\nNo steps have been taken yet.", lists, Choice.step("read", "read-11"))]
    assert [(r["level"], r["page"]) for r in rows] == [("tool", 1), ("tool", 2), ("argument", 2)]
    cases = [{"state": state, "lists": lists, "level": r["level"], "page": r["page"], "kind": "read"} for r in rows]
    for row, content in zip(rows, teacher_contents(tmp_path, cases), strict=True):
        assert_same_prompt(to_example(row), content)


def test_length_summary_gives_percentiles_and_the_share_over_the_limit():
    summary = length_summary(list(range(1, 101)), max_length=90)
    assert summary == {"rows": 100, "min": 1, "median": 50, "p90": 90, "p99": 99, "max": 100, "mean": 50.5, "max_length": 90, "over_max_length": 10, "share_over_max_length": 0.1}
    with pytest.raises(ValueError, match="no lengths"):
        length_summary([], max_length=90)


def stage1_rows(tmp_path):
    rows = [
        tool_row(task="bugs-1", session="a", stage=1),
        argument_row(task="bugs-1", session="a", stage=1),
        tool_row(task="bugs-1__dsv4-trial-2", session="b", stage=1, label="hand_over"),
        tool_row(task="bugs-2", session="c", stage=1, label="hand_over"),
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows, path


def write_lengths(tmp_path, over_ids, rows=4, max_length=8192):
    # The shape imitation/token_lengths.py writes: one entry per measured export file.
    return write_json(tmp_path / "lengths.json", {"export/train.jsonl": {"rows": rows, "max_length": max_length, "over_max_length": len(over_ids), "ids_over_max_length": over_ids}})


def test_an_over_length_row_drops_its_whole_task_group_and_says_so(tmp_path, capsys):
    splits = make_splits(write_json(tmp_path / "tasks.json", TASKS), seed=5)
    rows, path = stage1_rows(tmp_path)
    over = read_over_length(write_lengths(tmp_path, [row_id(rows[1])]), max_length=8192)
    counts = export_rows([path], splits, tmp_path / "out", over_length=over)
    kept = [json.loads(line) for name in counts for line in (tmp_path / "out" / f"{name}.jsonl").read_text().splitlines()]
    assert [e["suite"] for e in kept] == ["bugs-2"]  # bugs-1 and its retried trial go together
    assert sum(counts.values()) == 1
    dropped = json.loads((tmp_path / "out" / "dropped-over-length.json").read_text())
    assert dropped["max_length"] == 8192
    assert dropped["groups"] == ["bugs-1"] and dropped["group_count"] == 1
    assert dropped["rows_over_max_length"] == 1 and dropped["rows_dropped"] == 3
    assert sum(dropped["rows_dropped_by_split"].values()) == 3
    assert "dropped 3 rows" in capsys.readouterr().out


def test_lengths_for_another_limit_or_other_rows_are_errors(tmp_path):
    splits = make_splits(write_json(tmp_path / "tasks.json", TASKS), seed=5)
    rows, path = stage1_rows(tmp_path)
    with pytest.raises(ValueError, match="max_length"):
        read_over_length(write_lengths(tmp_path, [], max_length=4096), max_length=8192)
    with pytest.raises(ValueError, match="measured 5 rows"):
        export_rows([path], splits, tmp_path / "a", over_length=read_over_length(write_lengths(tmp_path, [], rows=5), max_length=8192))
    with pytest.raises(ValueError, match="not among the rows"):
        export_rows([path], splits, tmp_path / "b", over_length=read_over_length(write_lengths(tmp_path, ["nope"]), max_length=8192))


def test_cli_drops_over_length_groups_only_with_both_flags(tmp_path):
    splits_path = tmp_path / "splits.json"
    main(["splits", "--tasks", str(write_json(tmp_path / "tasks.json", TASKS)), "--seed", "5", "--out", str(splits_path)])
    rows, path = stage1_rows(tmp_path)
    lengths = write_lengths(tmp_path, [row_id(rows[3])])
    main(["export", "--rows", str(path), "--splits", str(splits_path), "--out", str(tmp_path / "out"), "--over-length", str(lengths), "--max-length", "8192"])
    assert json.loads((tmp_path / "out" / "dropped-over-length.json").read_text())["groups"] == ["bugs-2"]
    with pytest.raises(SystemExit):
        main(["export", "--rows", str(path), "--splits", str(splits_path), "--out", str(tmp_path / "x"), "--over-length", str(lengths)])
