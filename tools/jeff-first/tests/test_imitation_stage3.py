import json
from pathlib import Path

import pytest

from imitation.stage3 import CURRENT_TARBALL, UNKNOWN_FOLDER, convert_runs, main, model_format, read_tasks, stats_markdown, summarize

FIXTURES = Path(__file__).parent / "fixtures" / "stage3"
BUILDS = frozenset({CURRENT_TARBALL})
STEP_BUILD = "jeff-pi-scout-13486e524.tgz"

LISTS = {
    "tools": [
        {"id": "read", "description": "Read part or all of a file"},
        {"id": "hand_over", "description": "Hand over to the coding model for its next turn"},
    ],
    "arguments_by_tool": {
        "read": [
            {"id": "read-1", "description": "Read the file /app/main.py", "toolCall": {"name": "bash", "arguments": {"command": "cat /app/main.py"}}}
        ],
    },
}


def write_tasks(tmp_path: Path) -> Path:
    path = tmp_path / "training-tasks.json"
    path.write_text(
        json.dumps(
            {
                "training": ["fix-bug", "dna-assembly"],
                "excluded_evaluation": ["fix-git"],
                "excluded_leak_twins": {"make-doom-for-mips": "make-mips-interpreter", "torch-tensor-parallelism": "x"},
            }
        )
    )
    return path


def record(task, session, turn, build, command):
    return {
        "schema": "jeff-first-trace/4",
        "kind": "record",
        "task_id": task,
        "session_id": session,
        "turn": turn,
        "mode": "record",
        "driver_build": build,
        "state": {"task": "Fix the bug in /app/main.py.", "recentSteps": [], "stepsLeftOut": 0},
        "lists": {"tools": LISTS["tools"], "arguments_by_tool": LISTS["arguments_by_tool"]},
        "action": {"stop_reason": "toolUse", "error_message": None, "text_chars": 2, "tool_calls": [{"name": "bash", "arguments": {"command": command}}]},
    }


def make_trial(
    run: Path,
    stream: str,
    task: str,
    trial_id: str,
    *,
    build: str = "qwen3.8-27b-fp8@casdgx01-gpu5",
    tarball: str = CURRENT_TARBALL,
    result: dict | None = {"exception_info": None},
    trace: bool = True,
    trace_build: str | None = None,
    mode: str = "record",
    round_folder: str = "round-1",
) -> Path:
    trial = run / stream / round_folder / f"{task}-20261003-231031" / f"{task}__{trial_id}"
    (trial / "agent" / "pi" / "sessions").mkdir(parents=True)
    config = {
        "task": {"path": task},
        "trial_name": trial.name,
        "agent": {
            "kwargs": {"tarball": f"/home/u/jeff-first/collect/{tarball}"},
            "env": {"JEFF_FIRST_MODE": mode, "JEFF_FIRST_TASK_ID": task, "JEFF_FIRST_DRIVER_BUILD": build},
        },
    }
    (trial / "config.json").write_text(json.dumps(config))
    if result is not None:
        (trial / "result.json").write_text(json.dumps(result))
    if trace:
        session = f"sess-{trial_id}"
        header = {"type": "session", "version": 3, "id": session, "timestamp": "t", "cwd": "/app"}
        call = {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "cat /app/main.py"}}
        entries = [
            header,
            {"type": "message", "message": {"role": "user", "content": "Fix the bug in /app/main.py."}},
            {"type": "message", "message": {"role": "assistant", "content": [call]}},
            {"type": "message", "message": {"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": "x"}], "isError": False}},
        ]
        entries = [header] + [{**e, "id": f"e{n}", "parentId": None if n == 1 else f"e{n - 1}"} for n, e in enumerate(entries[1:], start=1)]
        (trial / "agent" / "pi" / "sessions" / "s.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
        line = record(task, session, 1, trace_build or build, "cat /app/main.py")
        (trial / "agent" / "jeff-first-trace.jsonl").write_text(json.dumps(line) + "\n")
    return trial


def test_good_trials_become_stage_3_rows_tagged_with_their_machine(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "h100-gpu5", "fix-bug", "aaa")
    make_trial(run, "rtx", "fix-bug", "bbb", build="qwen3.8-27b-nvfp4@spark-head")
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)
    assert {(r["stage"], r["quality"], r["source"]) for r in conversion.rows} == {(3, "exact", "own")}
    assert sorted((r["session"], r["machine"], r["level"], r["label"]) for r in conversion.rows) == [
        ("sess-aaa", "qwen3.8-27b-fp8@casdgx01-gpu5", "argument", "read-1"),
        ("sess-aaa", "qwen3.8-27b-fp8@casdgx01-gpu5", "tool", "read"),
        ("sess-bbb", "qwen3.8-27b-nvfp4@spark-head", "argument", "read-1"),
        ("sess-bbb", "qwen3.8-27b-nvfp4@spark-head", "tool", "read"),
    ]
    assert all(r["tool_description"] == "Read part or all of a file" for r in conversion.rows if r["level"] == "argument")


def test_round_folders_without_a_dash_are_found(tmp_path):
    # The xhigh collection (collect_rounds.sh, 2026-10-04) names its round folders round1, round2, ...
    run = tmp_path / "runs-collect-xhigh"
    make_trial(run, "b200-gpu0-s1", "fix-bug", "aaa", round_folder="round1")
    make_trial(run, "b200-gpu0-s1", "fix-bug", "bbb", round_folder="round-2")
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)
    assert sorted({r["session"] for r in conversion.rows}) == ["sess-aaa", "sess-bbb"]


def test_a_trial_whose_model_called_a_tool_other_than_bash_is_skipped_and_counted(tmp_path):
    # Qwen (NVFP4) sometimes names a tool that does not exist (seen 2026-10-04: "bbyte"); Jeff-Code answers with an error.
    run = tmp_path / "runs-collect-xhigh"
    trial = make_trial(run, "b200-gpu0-s3", "fix-bug", "aaa")
    make_trial(run, "b200-gpu0-s3", "fix-bug", "bbb")
    session = trial / "agent" / "pi" / "sessions" / "s.jsonl"
    session.write_text(session.read_text().replace('"name": "bash"', '"name": "bbyte"'))
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)
    assert {r["session"] for r in conversion.rows} == {"sess-bbb"}
    assert conversion.skipped == {"a reply calls a tool other than bash": 1}


def test_a_trial_with_a_command_in_a_folder_the_labeller_cannot_know_is_skipped_counted_and_listed(tmp_path):
    # `cd ~/...` or `cd $VAR`: the folder its relative paths resolve in is unknown, so no label can be given for that
    # turn (seen 2026-10-04 in 7 of 1,431 trials, e.g. `cd ~/.cache/pip/http-v2 && grep -rl x .`).
    run = tmp_path / "runs-collect-xhigh"
    trial = make_trial(run, "b200-gpu0-s3", "fix-bug", "aaa")
    make_trial(run, "b200-gpu0-s3", "fix-bug", "bbb")
    for path in (trial / "agent" / "jeff-first-trace.jsonl", trial / "agent" / "pi" / "sessions" / "s.jsonl"):
        path.write_text(path.read_text().replace("cat /app/main.py", "cd ~/cache && cat main.py"))
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)
    assert {r["session"] for r in conversion.rows} == {"sess-bbb"}
    assert conversion.skipped == {UNKNOWN_FOLDER: 1}
    assert [entry["trial"] for entry in conversion.skipped_trials] == [str(trial)]
    assert "cd ~/cache" in conversion.skipped_trials[0]["reason"]
    assert summarize(conversion)["skipped_trials"] == conversion.skipped_trials


def test_other_builds_twins_other_tasks_and_trials_without_a_trace_are_skipped_and_counted(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu0", "fix-bug", "keep")
    make_trial(run, "gpu0", "fix-bug", "old1", tarball="jeff-pi-scout-adad96753.tgz")
    make_trial(run, "gpu0", "fix-bug", "old2", tarball="jeff-pi-scout-adad96753.tgz")
    make_trial(run, "gpu0", "make-doom-for-mips", "twin")
    make_trial(run, "gpu0", "torch-tensor-parallelism", "other")
    make_trial(run, "gpu1", "fix-bug", "setup", trace=False, result={"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}})
    make_trial(run, "gpu1", "fix-bug", "running", trace=False, result=None)
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)
    assert {r["session"] for r in conversion.rows} == {"sess-keep"}
    assert conversion.skipped == {
        "older build jeff-pi-scout-adad96753.tgz": 2,
        "make-doom-for-mips (twin of an evaluation task)": 1,
        "not a training task": 1,
        "no trace: the agent failed before its first turn (NonZeroAgentExitCodeError)": 1,
        "no trace yet: no result.json and no trace (still running, or stopped before its first turn)": 1,
    }


def test_an_evaluation_task_is_an_error(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu0", "fix-git", "eval")
    with pytest.raises(ValueError, match="evaluation task fix-git"):
        convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)


def test_an_evaluation_task_on_an_older_build_is_still_an_error(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu0", "fix-git", "eval", tarball="jeff-pi-scout-adad96753.tgz")
    with pytest.raises(ValueError, match="evaluation task fix-git"):
        convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"trace": False}, "finished without a trace"),
        ({"trace_build": "qwen3.8-27b-fp8@other"}, "driver build"),
        ({"mode": "teacher"}, "record mode"),
        ({"build": "qwen3.8-27b-int4@x"}, "model format"),
    ],
)
def test_inconsistent_trials_are_errors(tmp_path, options, message):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu0", "fix-bug", "bad", **options)
    with pytest.raises(ValueError, match=message):
        convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)


def test_a_run_folder_without_trials_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no trial folders"):
        convert_runs([tmp_path / "empty"], read_tasks(write_tasks(tmp_path)), BUILDS)


def test_model_format_is_read_from_the_driver_build():
    assert model_format("qwen3.8-27b-fp8@casdgx01-gpu5") == "fp8"
    assert model_format("qwen3.8-27b-nvfp4@spark-head") == "nvfp4"
    with pytest.raises(ValueError, match="model format"):
        model_format("qwen3.8-27b@x")


def test_a_real_record_trial_converts(tmp_path):
    # A real casdgx01 trial copied while it was still running (no result.json, so it counts as cut).
    conversion = convert_runs([FIXTURES / "runs-imitation-v5"], read_tasks(write_tasks(tmp_path)), BUILDS)
    assert [(r["turn"], r["level"], r["label"]) for r in conversion.rows] == [(1, "tool", "hand_over"), (2, "tool", "read"), (2, "argument", "read-1")]
    assert {r["machine"] for r in conversion.rows} == {"qwen3.8-27b-fp8@casdgx01-gpu1"}
    assert [trial["cut"] for trial in conversion.trials] == ["no result.json (the trial was stopped or is still running)"]


def test_summary_counts_sessions_decisions_labels_and_hand_over_by_machine_and_format(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu5", "fix-bug", "aaa")
    make_trial(run, "gpu6", "fix-bug", "bbb", build="qwen3.8-27b-fp8@casdgx01-gpu6")
    make_trial(run, "rtx", "fix-bug", "ccc", build="qwen3.8-27b-nvfp4@rtx-pro-6000")
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)
    summary = summarize(conversion)
    assert summary["sessions"] == 3 and summary["rows"] == 6 and summary["decisions"] == 3
    assert summary["tool_labels"] == {"read": 3}
    assert summary["hand_over_share"] == 0.0
    assert summary["by_format"]["fp8"] == {"sessions": 2, "rows": 4, "decisions": 2, "stint_decisions": 0, "hand_over_share": 0.0, "tool_labels": {"read": 2}}
    assert summary["by_machine"]["qwen3.8-27b-nvfp4@rtx-pro-6000"]["sessions"] == 1
    assert summary["sessions_by_task"] == {"fix-bug": 3}
    text = stats_markdown(summary)
    assert "qwen3.8-27b-fp8@casdgx01-gpu6" in text and "| nvfp4 |" in text


def test_cli_writes_rows_summary_and_stats(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu5", "fix-bug", "aaa")
    out = tmp_path / "out"
    main([str(run), "--tasks", str(write_tasks(tmp_path)), "--build", CURRENT_TARBALL, "--rows", str(out / "rows.jsonl"), "--summary", str(out / "summary.json"), "--stats", str(out / "stats.md")])
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert [r["label"] for r in rows] == ["read", "read-1"]
    assert json.loads((out / "summary.json").read_text())["rows"] == 2
    assert (out / "stats.md").read_text().startswith("# Stage 3")


def make_step_trial(run: Path, trial_id: str) -> Path:
    """A schema-5 trial (record_step lines): turn 1 reads /app/main.py, then /app/notes.md, with the menu logged after
    the first read offering the second."""
    trial = make_trial(run, "gpu5", "fix-bug", trial_id, tarball=STEP_BUILD)
    session = f"sess-{trial_id}"
    header = {"type": "session", "version": 3, "id": session, "timestamp": "t", "cwd": "/app"}
    calls = [
        {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "cat /app/main.py"}},
        {"type": "toolCall", "id": "c2", "name": "bash", "arguments": {"command": "cat /app/notes.md"}},
    ]
    entries = [
        header,
        {"type": "message", "message": {"role": "user", "content": "Fix the bug in /app/main.py."}},
        {"type": "message", "message": {"role": "assistant", "content": calls}},
        {"type": "message", "message": {"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": "see notes.md"}], "isError": False}},
        {"type": "message", "message": {"role": "toolResult", "toolCallId": "c2", "content": [{"type": "text", "text": "n"}], "isError": False}},
    ]
    entries = [header] + [{**e, "id": f"e{n}", "parentId": None if n == 1 else f"e{n - 1}"} for n, e in enumerate(entries[1:], start=1)]
    (trial / "agent" / "pi" / "sessions" / "s.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    first = {**record("fix-bug", session, 1, "qwen3.8-27b-fp8@casdgx01-gpu5", "cat /app/main.py"), "schema": "jeff-first-trace/5"}
    first["action"]["tool_calls"] = [{"name": c["name"], "arguments": c["arguments"]} for c in calls]
    notes = {"id": "read-1", "description": "Read the file /app/notes.md", "toolCall": {"name": "bash", "arguments": {"command": "cat /app/notes.md"}}}
    step = {
        "schema": "jeff-first-trace/5",
        "kind": "record_step",
        "task_id": "fix-bug",
        "session_id": session,
        "turn": 1,
        "step": 1,
        "calls_in_turn": 2,
        "command": "cat /app/main.py",
        "mode": "record",
        "driver_build": "qwen3.8-27b-fp8@casdgx01-gpu5",
        "state": {"task": "Fix the bug in /app/main.py.", "recentSteps": [{"command": "cat /app/main.py", "output": "see notes.md", "isError": False, "byScout": True}], "stepsLeftOut": 0},
        "lists": {"tools": LISTS["tools"], "arguments_by_tool": {"read": [notes]}},
    }
    (trial / "agent" / "jeff-first-trace.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(step) + "\n")
    return trial


def test_a_schema_5_trial_gives_stint_rows_and_the_summary_counts_them(tmp_path):
    run = tmp_path / "runs-v6"
    make_step_trial(run, "stint")
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), frozenset({STEP_BUILD}))
    assert [(r["decision"], r["turn"], r["level"], r["label"]) for r in conversion.rows] == [
        (0, 1, "tool", "read"),
        (0, 1, "argument", "read-1"),
        (1, 1, "tool", "read"),
        (1, 1, "argument", "read-1"),
    ]
    summary = summarize(conversion)
    assert summary["decisions"] == 2 and summary["stint_decisions"] == 1
    assert "stint decisions: 1" in stats_markdown(summary)


def test_only_the_given_builds_are_converted(tmp_path):
    run = tmp_path / "runs-v6"
    make_trial(run, "gpu0", "fix-bug", "old")
    make_step_trial(run, "new")
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), frozenset({STEP_BUILD}))
    assert {r["session"] for r in conversion.rows} == {"sess-new"}
    assert conversion.skipped == {f"older build {CURRENT_TARBALL}": 1}
    both = convert_runs([run], read_tasks(write_tasks(tmp_path)), frozenset({STEP_BUILD, CURRENT_TARBALL}))
    assert {r["session"] for r in both.rows} == {"sess-new", "sess-old"}


def test_the_cli_needs_the_builds(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu5", "fix-bug", "aaa")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        main([str(run), "--tasks", str(write_tasks(tmp_path)), "--rows", str(out / "r"), "--summary", str(out / "s"), "--stats", str(out / "t")])


def test_a_real_schema_5_trial_converts_end_to_end(tmp_path):
    # Written by Jeff-Code itself (createAgentSession in record mode, build 13486e524) with a fake coding model; only the
    # session's working folder was renamed to /app. Turn 1 runs `ls`, then `cat main.py` (one record_step line after
    # `ls`); turn 2 runs `cat notes.md`, rewrites main.py and runs it (record_step lines after its calls 1 and 2);
    # turn 3 only answers.
    run = FIXTURES / "runs-record-steps"
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), frozenset({"jeff-pi-scout-13486e524.tgz"}))
    assert [(r["decision"], r["turn"], r["level"], r["label"]) for r in conversion.rows] == [
        (0, 1, "tool", "list"),
        (0, 1, "argument", "list-1"),
        (1, 1, "tool", "read"),  # inside turn 1, from the menu logged after `ls`
        (1, 1, "argument", "read-1"),
        (2, 2, "tool", "read"),
        (2, 2, "argument", "read-2"),
        (3, 2, "tool", "hand_over"),  # the rewrite of main.py acts: the stint ends
        (4, 3, "tool", "hand_over"),
    ]
    assert "Step 1 (by you, the scout):\n$ ls -la '/app'\nmain.py\nnotes.md" in conversion.rows[2]["state"]
    assert summarize(conversion)["stint_decisions"] == 2


def test_qwen_request_lines_are_skipped_and_counted_and_other_kinds_raise(tmp_path):
    run = tmp_path / "runs-v6"
    trial = make_trial(run, "gpu5", "fix-bug", "aaa")
    trace = trial / "agent" / "jeff-first-trace.jsonl"
    request = {"schema": "jeff-first-trace/6", "kind": "qwen_request", "task_id": "fix-bug", "session_id": "sess-aaa", "turn": 1, "attempt": 1}
    trace.write_text(json.dumps(request) + "\n" + trace.read_text())
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)
    summary = summarize(conversion)
    assert summary["decisions"] == 1
    assert summary["skipped_trace_lines"] == {"qwen_request": 1}
    assert "qwen_request 1" in stats_markdown(summary)
    trace.write_text(json.dumps({**request, "kind": "model_turn"}) + "\n" + trace.read_text())
    with pytest.raises(ValueError, match="unknown kind 'model_turn'"):
        convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)


def test_a_qwen_request_line_of_another_task_is_an_error(tmp_path):
    run = tmp_path / "runs-v6"
    trial = make_trial(run, "gpu5", "fix-bug", "aaa")
    trace = trial / "agent" / "jeff-first-trace.jsonl"
    request = {"schema": "jeff-first-trace/6", "kind": "qwen_request", "task_id": "dna-assembly", "session_id": "sess-aaa", "turn": 1, "attempt": 1}
    trace.write_text(json.dumps(request) + "\n" + trace.read_text())
    with pytest.raises(ValueError, match="dna-assembly"):
        convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)


HUB_TRAIN = "terminal-bench-pro/terminal-bench-pro:fix-a"


def make_hub_trial(run: Path, task_id: str, trial_id: str) -> Path:
    """A trial of a Harbor hub dataset task: config.json names the task package and its dataset, not a path."""
    hub, task = task_id.split(":")
    trial = make_trial(run, "b200-gpu0-s1", task_id.replace("/", ".").replace(":", "."), trial_id, round_folder="round1")
    config = json.loads((trial / "config.json").read_text())
    config["task"] = {"name": f"{hub.split('/')[0]}/{task}", "ref": "sha256:" + "c" * 64, "source": hub}
    config["agent"]["env"]["JEFF_FIRST_TASK_ID"] = task_id
    (trial / "config.json").write_text(json.dumps(config))
    trace = trial / "agent" / "jeff-first-trace.jsonl"
    line = json.loads(trace.read_text())
    trace.write_text(json.dumps({**line, "task_id": task_id}) + "\n")
    return trial


def hub_tasks(tmp_path: Path):
    from tests.test_task_source import write_task_sets

    sets, _ = write_task_sets(tmp_path)
    return read_tasks(write_tasks(tmp_path), sets)


def test_a_hub_training_task_converts_with_its_dataset_in_the_task_id(tmp_path):
    run = tmp_path / "runs-collect-xhigh2"
    make_hub_trial(run, HUB_TRAIN, "hub")
    make_trial(run, "gpu0", "fix-bug", "tb2")
    conversion = convert_runs([run], hub_tasks(tmp_path), BUILDS)
    assert sorted({(r["session"], r["task"]) for r in conversion.rows}) == [("sess-hub", HUB_TRAIN), ("sess-tb2", "fix-bug")]
    assert summarize(conversion)["sessions_by_task"] == {HUB_TRAIN: 1, "fix-bug": 1}


@pytest.mark.parametrize(
    ("task_id", "message"),
    [
        ("terminal-bench-pro/terminal-bench-pro:held-c", "held-out"),
        ("terminal-bench-pro/terminal-bench-pro:leak-d", "excluded"),
        ("terminal-bench-pro/terminal-bench-pro:no-such", "no task no-such"),
        ("nobody/nothing:x", "unknown dataset"),
    ],
)
def test_hub_tasks_outside_training_are_errors(tmp_path, task_id, message):
    run = tmp_path / "runs-collect-xhigh2"
    make_hub_trial(run, task_id, "bad")
    with pytest.raises(ValueError, match=message):
        convert_runs([run], hub_tasks(tmp_path), BUILDS)


def test_a_hub_task_without_the_task_sets_is_an_error(tmp_path):
    run = tmp_path / "runs-collect-xhigh2"
    make_hub_trial(run, HUB_TRAIN, "hub")
    with pytest.raises(ValueError, match="--task-sets"):
        convert_runs([run], read_tasks(write_tasks(tmp_path)), BUILDS)


def test_a_hub_trial_whose_config_names_another_task_is_an_error(tmp_path):
    run = tmp_path / "runs-collect-xhigh2"
    trial = make_hub_trial(run, HUB_TRAIN, "hub")
    config = json.loads((trial / "config.json").read_text())
    config["task"]["name"] = "terminal-bench-pro/slow-b"
    (trial / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="slow-b"):
        convert_runs([run], hub_tasks(tmp_path), BUILDS)


def test_the_cli_takes_the_task_sets(tmp_path):
    from tests.test_task_source import write_task_sets

    run = tmp_path / "runs-collect-xhigh2"
    make_hub_trial(run, HUB_TRAIN, "hub")
    sets, _ = write_task_sets(tmp_path)
    out = tmp_path / "out"
    main([str(run), "--tasks", str(write_tasks(tmp_path)), "--task-sets", str(sets), "--build", CURRENT_TARBALL, "--rows", str(out / "rows.jsonl"), "--summary", str(out / "s.json"), "--stats", str(out / "s.md")])
    assert {json.loads(line)["task"] for line in (out / "rows.jsonl").read_text().splitlines()} == {HUB_TRAIN}
