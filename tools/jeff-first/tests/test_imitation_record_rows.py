import json

import pytest

from imitation.record_rows import record_rows


def option(kind, number, description, command):
    return {"id": f"{kind}-{number}", "description": description, "toolCall": {"name": "bash", "arguments": {"command": command}}}


LISTS = {
    "tools": [
        {"id": "read", "description": "Read a file"},
        {"id": "list", "description": "List a folder"},
        {"id": "hand_over", "description": "Hand over to the coding model"},
    ],
    "arguments_by_tool": {
        "read": [option("read", 1, "Read the file /app/main.py", "cat /app/main.py")],
        "list": [option("list", 1, "List the folder /app", "ls -la /app")],
    },
}


def bash(call_id, command):
    return {"type": "toolCall", "id": call_id, "name": "bash", "arguments": {"command": command}}


def result(call_id, text, is_error=False):
    return {"type": "message", "message": {"role": "toolResult", "toolCallId": call_id, "content": [{"type": "text", "text": text}], "isError": is_error}}


def assistant(*calls):
    return {"type": "message", "message": {"role": "assistant", "provider": "sparkgate", "content": [{"type": "text", "text": "ok"}, *calls]}}


def record(turn, tool_calls, stop_reason="toolUse", recent=()):
    return {
        "schema": "jeff-first-trace/4",
        "kind": "record",
        "task_id": "fix-bug",
        "session_id": "sess-1",
        "turn": turn,
        "mode": "record",
        "state": {"task": "Fix the bug in /app/main.py.", "recentSteps": list(recent), "stepsLeftOut": 0},
        "lists": {"tools": LISTS["tools"], "arguments_by_tool": LISTS["arguments_by_tool"]},
        "action": {"stop_reason": stop_reason, "error_message": None, "text_chars": 2, "tool_calls": tool_calls},
    }


def write_session(tmp_path, entries):
    folder = tmp_path / "sessions"
    folder.mkdir()
    header = {"type": "session", "version": 3, "id": "sess-1", "timestamp": "t", "cwd": "/app"}
    lines = [header, {"type": "message", "message": {"role": "user", "content": "Fix the bug in /app/main.py."}}, *entries]
    (folder / "s.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return folder


def call(command):
    return {"name": "bash", "arguments": {"command": command}}


def test_record_rows_label_each_logged_turn(tmp_path):
    entries = [
        assistant(bash("c1", "ls -la /app"), bash("c2", "cat /app/main.py")),
        result("c1", "main.py"),
        result("c2", "print(1/0)"),
        assistant(bash("c3", "cat > /app/main.py <<'EOF'\nprint(1)\nEOF")),
        result("c3", ""),
    ]
    folder = write_session(tmp_path, entries)
    trace = [
        record(1, [call("ls -la /app"), call("cat /app/main.py")]),
        record(2, [call("cat > /app/main.py <<'EOF'\nprint(1)\nEOF")], recent=[{"tool": "bash", "arguments": {"command": c}} for c in ("ls -la /app", "cat /app/main.py")]),
    ]
    rows, notes = record_rows(trace, sorted(folder.glob("*.jsonl")))
    assert notes == []
    assert [(r.decision, r.turn, r.level, r.label) for r in rows] == [
        (0, 1, "tool", "list"),
        (0, 1, "argument", "list-1"),
        (1, 2, "tool", "hand_over"),
    ]
    assert {(r.source, r.stage, r.quality, r.task, r.session) for r in rows} == {("jeff-pi-record", 3, "exact", "fix-bug", "sess-1")}
    assert rows[2].state == (
        "Task:\nFix the bug in /app/main.py.\n\nSteps so far, oldest first:\n\n"
        "Step 1 (by you, the scout):\n$ ls -la /app\nmain.py\n\n"
        "Step 2 (by the coding model):\n$ cat /app/main.py\nprint(1/0)"
    )


@pytest.mark.parametrize("stop_reason", ["error", "aborted"])
def test_a_failed_model_turn_gets_no_row_never_hand_over(tmp_path, stop_reason):
    folder = write_session(tmp_path, [{"type": "message", "message": {"role": "assistant", "content": []}}])
    rows, notes = record_rows([record(1, [], stop_reason=stop_reason)], sorted(folder.glob("*.jsonl")))
    assert rows == []
    assert notes == [f"sess-1 turn 1: the model call ended with {stop_reason}; no row"]


def test_a_failed_turn_between_good_turns_keeps_the_others(tmp_path):
    entries = [
        {"type": "message", "message": {"role": "assistant", "content": []}},
        assistant(bash("c1", "cat /app/main.py")),
        result("c1", "print(1)"),
    ]
    folder = write_session(tmp_path, entries)
    trace = [record(1, [], stop_reason="error"), record(2, [call("cat /app/main.py")])]
    rows, notes = record_rows(trace, sorted(folder.glob("*.jsonl")))
    assert [(r.turn, r.label) for r in rows] == [(2, "read"), (2, "read-1")]
    assert notes == ["sess-1 turn 1: the model call ended with error; no row"]


def new_shape(command, output):
    return {"command": command, "output": output, "isError": False, "byScout": False}


def test_both_logged_step_shapes_are_read(tmp_path):
    entries = [assistant(bash("c1", "ls -la /app")), result("c1", "main.py"), assistant(bash("c2", "cat /app/main.py")), result("c2", "x")]
    folder = write_session(tmp_path, entries)
    files = sorted(folder.glob("*.jsonl"))
    old = [{"tool": "bash", "arguments": {"command": "ls -la /app"}, "output": "main.py", "isError": False, "byScout": False}]
    for recent in (old, [new_shape("ls -la /app", "main.py")]):
        rows, _ = record_rows([record(1, [call("ls -la /app")]), record(2, [call("cat /app/main.py")], recent=recent)], files)
        assert [(r.turn, r.label) for r in rows] == [(1, "list"), (1, "list-1"), (2, "read"), (2, "read-1")]


def test_a_logged_last_step_that_is_not_the_sessions_is_an_error(tmp_path):
    entries = [assistant(bash("c1", "ls -la /app")), result("c1", "main.py"), assistant(bash("c2", "cat /app/main.py")), result("c2", "x")]
    folder = write_session(tmp_path, entries)
    trace = [record(1, [call("ls -la /app")]), record(2, [call("cat /app/main.py")], recent=[new_shape("ls /tmp", "")])]
    with pytest.raises(ValueError, match="last step"):
        record_rows(trace, sorted(folder.glob("*.jsonl")))
    shapeless = [record(1, [call("ls -la /app")]), record(2, [call("cat /app/main.py")], recent=[{"output": ""}])]
    with pytest.raises(ValueError, match="neither a command"):
        record_rows(shapeless, sorted(folder.glob("*.jsonl")))


def test_trace_and_session_that_disagree_are_an_error(tmp_path):
    folder = write_session(tmp_path, [assistant(bash("c1", "ls")), result("c1", "x")])
    with pytest.raises(ValueError, match="turn 1"):
        record_rows([record(1, [call("pwd")])], sorted(folder.glob("*.jsonl")))


def test_missing_session_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="sess-1"):
        record_rows([record(1, [])], [])


def test_a_non_bash_tool_call_is_an_error(tmp_path):
    folder = write_session(tmp_path, [assistant({"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "a"}}), result("c1", "x")])
    with pytest.raises(ValueError, match="bash"):
        record_rows([record(1, [{"name": "read", "arguments": {"path": "a"}}])], sorted(folder.glob("*.jsonl")))


def test_a_step_count_that_differs_from_the_logged_state_is_an_error(tmp_path):
    folder = write_session(tmp_path, [assistant(bash("c1", "ls")), result("c1", "x"), assistant(), ])
    with pytest.raises(ValueError, match="steps"):
        record_rows([record(1, [call("ls")]), record(2, [])], sorted(folder.glob("*.jsonl")))
