import json
from pathlib import Path

import pytest

from imitation.stage3 import CURRENT_TARBALL, convert_runs, main, model_format, read_tasks, stats_markdown, summarize

FIXTURES = Path(__file__).parent / "fixtures" / "stage3"

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
) -> Path:
    trial = run / stream / "round-1" / f"{task}-20261003-231031" / f"{task}__{trial_id}"
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
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)))
    assert {(r["stage"], r["quality"], r["source"]) for r in conversion.rows} == {(3, "exact", "own")}
    assert sorted((r["session"], r["machine"], r["level"], r["label"]) for r in conversion.rows) == [
        ("sess-aaa", "qwen3.8-27b-fp8@casdgx01-gpu5", "argument", "read-1"),
        ("sess-aaa", "qwen3.8-27b-fp8@casdgx01-gpu5", "tool", "read"),
        ("sess-bbb", "qwen3.8-27b-nvfp4@spark-head", "argument", "read-1"),
        ("sess-bbb", "qwen3.8-27b-nvfp4@spark-head", "tool", "read"),
    ]
    assert all(r["tool_description"] == "Read part or all of a file" for r in conversion.rows if r["level"] == "argument")


def test_other_builds_twins_other_tasks_and_trials_without_a_trace_are_skipped_and_counted(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu0", "fix-bug", "keep")
    make_trial(run, "gpu0", "fix-bug", "old1", tarball="jeff-pi-scout-adad96753.tgz")
    make_trial(run, "gpu0", "fix-bug", "old2", tarball="jeff-pi-scout-adad96753.tgz")
    make_trial(run, "gpu0", "make-doom-for-mips", "twin")
    make_trial(run, "gpu0", "torch-tensor-parallelism", "other")
    make_trial(run, "gpu1", "fix-bug", "setup", trace=False, result={"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}})
    make_trial(run, "gpu1", "fix-bug", "running", trace=False, result=None)
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)))
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
        convert_runs([run], read_tasks(write_tasks(tmp_path)))


def test_an_evaluation_task_on_an_older_build_is_still_an_error(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu0", "fix-git", "eval", tarball="jeff-pi-scout-adad96753.tgz")
    with pytest.raises(ValueError, match="evaluation task fix-git"):
        convert_runs([run], read_tasks(write_tasks(tmp_path)))


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
        convert_runs([run], read_tasks(write_tasks(tmp_path)))


def test_a_run_folder_without_trials_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no trial folders"):
        convert_runs([tmp_path / "empty"], read_tasks(write_tasks(tmp_path)))


def test_model_format_is_read_from_the_driver_build():
    assert model_format("qwen3.8-27b-fp8@casdgx01-gpu5") == "fp8"
    assert model_format("qwen3.8-27b-nvfp4@spark-head") == "nvfp4"
    with pytest.raises(ValueError, match="model format"):
        model_format("qwen3.8-27b@x")


def test_a_real_record_trial_converts(tmp_path):
    # A real casdgx01 trial copied while it was still running (no result.json, so it counts as cut).
    conversion = convert_runs([FIXTURES / "runs-imitation-v5"], read_tasks(write_tasks(tmp_path)))
    assert [(r["turn"], r["level"], r["label"]) for r in conversion.rows] == [(1, "tool", "hand_over"), (2, "tool", "read"), (2, "argument", "read-1")]
    assert {r["machine"] for r in conversion.rows} == {"qwen3.8-27b-fp8@casdgx01-gpu1"}
    assert [trial["cut"] for trial in conversion.trials] == ["no result.json (the trial was stopped or is still running)"]


def test_summary_counts_sessions_decisions_labels_and_hand_over_by_machine_and_format(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu5", "fix-bug", "aaa")
    make_trial(run, "gpu6", "fix-bug", "bbb", build="qwen3.8-27b-fp8@casdgx01-gpu6")
    make_trial(run, "rtx", "fix-bug", "ccc", build="qwen3.8-27b-nvfp4@rtx-pro-6000")
    conversion = convert_runs([run], read_tasks(write_tasks(tmp_path)))
    summary = summarize(conversion)
    assert summary["sessions"] == 3 and summary["rows"] == 6 and summary["decisions"] == 3
    assert summary["tool_labels"] == {"read": 3}
    assert summary["hand_over_share"] == 0.0
    assert summary["by_format"]["fp8"] == {"sessions": 2, "rows": 4, "decisions": 2, "hand_over_share": 0.0, "tool_labels": {"read": 2}}
    assert summary["by_machine"]["qwen3.8-27b-nvfp4@rtx-pro-6000"]["sessions"] == 1
    assert summary["sessions_by_task"] == {"fix-bug": 3}
    text = stats_markdown(summary)
    assert "qwen3.8-27b-fp8@casdgx01-gpu6" in text and "| nvfp4 |" in text


def test_cli_writes_rows_summary_and_stats(tmp_path):
    run = tmp_path / "runs-v5"
    make_trial(run, "gpu5", "fix-bug", "aaa")
    out = tmp_path / "out"
    main([str(run), "--tasks", str(write_tasks(tmp_path)), "--rows", str(out / "rows.jsonl"), "--summary", str(out / "summary.json"), "--stats", str(out / "stats.md")])
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert [r["label"] for r in rows] == ["read", "read-1"]
    assert json.loads((out / "summary.json").read_text())["rows"] == 2
    assert (out / "stats.md").read_text().startswith("# Stage 3")
