import json

import pytest

from imitation.record_rows import read_trace, record_rows, trial_cut


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
    # Entries form one chain, as pi writes them: entry N's parent is entry N-1 (the first one's parent is None).
    lines = [header] + [{**line, "id": f"e{n}", "parentId": None if n == 1 else f"e{n - 1}"} for n, line in enumerate(lines[1:], start=1)]
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
    rows, notes = record_rows(trace, sorted(folder.glob("*.jsonl")), cut=False)
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
    rows, notes = record_rows([record(1, [], stop_reason=stop_reason)], sorted(folder.glob("*.jsonl")), cut=False)
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
    rows, notes = record_rows(trace, sorted(folder.glob("*.jsonl")), cut=False)
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
        rows, _ = record_rows([record(1, [call("ls -la /app")]), record(2, [call("cat /app/main.py")], recent=recent)], files, cut=False)
        assert [(r.turn, r.label) for r in rows] == [(1, "list"), (1, "list-1"), (2, "read"), (2, "read-1")]


def test_a_logged_last_step_that_is_not_the_sessions_is_an_error(tmp_path):
    entries = [assistant(bash("c1", "ls -la /app")), result("c1", "main.py"), assistant(bash("c2", "cat /app/main.py")), result("c2", "x")]
    folder = write_session(tmp_path, entries)
    trace = [record(1, [call("ls -la /app")]), record(2, [call("cat /app/main.py")], recent=[new_shape("ls /tmp", "")])]
    with pytest.raises(ValueError, match="last step"):
        record_rows(trace, sorted(folder.glob("*.jsonl")), cut=False)
    shapeless = [record(1, [call("ls -la /app")]), record(2, [call("cat /app/main.py")], recent=[{"output": ""}])]
    with pytest.raises(ValueError, match="neither a command"):
        record_rows(shapeless, sorted(folder.glob("*.jsonl")), cut=False)


def test_trace_and_session_that_disagree_are_an_error(tmp_path):
    folder = write_session(tmp_path, [assistant(bash("c1", "ls")), result("c1", "x")])
    with pytest.raises(ValueError, match="turn 1"):
        record_rows([record(1, [call("pwd")])], sorted(folder.glob("*.jsonl")), cut=False)


def test_missing_session_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="sess-1"):
        record_rows([record(1, [])], [], cut=False)


def test_a_non_bash_tool_call_is_an_error(tmp_path):
    folder = write_session(tmp_path, [assistant({"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "a"}}), result("c1", "x")])
    with pytest.raises(ValueError, match="bash"):
        record_rows([record(1, [{"name": "read", "arguments": {"path": "a"}}])], sorted(folder.glob("*.jsonl")), cut=False)


def test_a_step_count_that_differs_from_the_logged_state_is_an_error(tmp_path):
    folder = write_session(tmp_path, [assistant(bash("c1", "ls")), result("c1", "x"), assistant(), ])
    with pytest.raises(ValueError, match="steps"):
        record_rows([record(1, [call("ls")]), record(2, [])], sorted(folder.glob("*.jsonl")), cut=False)


def write_trial(tmp_path, trace_text, exception_type=None):
    (tmp_path / "agent").mkdir(parents=True)
    (tmp_path / "agent" / "jeff-first-trace.jsonl").write_text(trace_text)
    info = None if exception_type is None else {"exception_type": exception_type, "exception_message": "timed out"}
    (tmp_path / "result.json").write_text(json.dumps({"task_name": "fix-bug", "exception_info": info}))
    return tmp_path


GOOD = json.dumps(record(1, [])) + "\n"
CUT = '{"schema": "jeff-first-trace/4", "kind": "record", "state": {"task": "Unterminated'


def test_a_trace_cut_by_the_agent_timeout_loses_only_its_unfinished_last_line(tmp_path):
    trial = write_trial(tmp_path, GOOD + CUT, exception_type="AgentTimeoutError")
    lines, notes = read_trace(trial)
    assert lines == [record(1, [])]
    assert len(notes) == 1 and "AgentTimeoutError" in notes[0] and "line 2" in notes[0]


@pytest.mark.parametrize(
    ("text", "exception_type"),
    [
        (GOOD + CUT, None),
        (GOOD + CUT, "RuntimeError"),
        (CUT + "\n" + GOOD, "AgentTimeoutError"),
    ],
)
def test_any_other_unparsable_trace_line_is_an_error(tmp_path, text, exception_type):
    trial = write_trial(tmp_path, text, exception_type=exception_type)
    with pytest.raises(ValueError, match="jeff-first-trace.jsonl line"):
        read_trace(trial)


def test_a_whole_trace_reads_without_notes(tmp_path):
    lines, notes = read_trace(write_trial(tmp_path, GOOD + GOOD, exception_type="AgentTimeoutError"))
    assert lines == [record(1, []), record(1, [])] and notes == []


def test_a_bash_call_without_a_command_is_shown_as_state_ts_shows_it(tmp_path):
    # Qwen sometimes sends a bash call with no arguments; pi answers "Validation failed" and state.ts shows the step
    # as the tool name and its arguments ("bash {}"). The turn is labelled from that text: it matches no option.
    entries = [
        assistant({"type": "toolCall", "id": "c1", "name": "bash", "arguments": {}}),
        result("c1", "Validation failed for tool \"bash\"", is_error=True),
        assistant(bash("c2", "cat /app/main.py")),
        result("c2", "x"),
    ]
    folder = write_session(tmp_path, entries)
    shown = {"command": "bash {}", "output": "Validation failed for tool \"bash\"", "isError": True, "byScout": False}
    trace = [record(1, [{"name": "bash", "arguments": {}}]), record(2, [call("cat /app/main.py")], recent=[shown])]
    rows, notes = record_rows(trace, sorted(folder.glob("*.jsonl")), cut=False)
    assert [(r.turn, r.label) for r in rows] == [(1, "hand_over"), (2, "read"), (2, "read-1")]
    assert "$ bash {}\nValidation failed" in rows[1].state


def test_bash_arguments_are_written_as_javascript_writes_json(tmp_path):
    arguments = {"timeout": 5, "note": "é"}
    entries = [assistant({"type": "toolCall", "id": "c1", "name": "bash", "arguments": arguments}), result("c1", "bad", True)]
    folder = write_session(tmp_path, [*entries, assistant(bash("c2", "ls /app")), result("c2", "x")])
    shown = {"command": 'bash {"timeout":5,"note":"é"}', "output": "bad", "isError": True, "byScout": False}
    trace = [record(1, [{"name": "bash", "arguments": arguments}]), record(2, [call("ls /app")], recent=[shown])]
    rows, _ = record_rows(trace, sorted(folder.glob("*.jsonl")), cut=False)
    assert [(r.turn, r.label) for r in rows] == [(1, "hand_over"), (2, "list"), (2, "list-1")]


def test_a_cut_trial_drops_record_lines_for_turns_the_session_file_does_not_have_yet(tmp_path):
    # pi writes the record line and the session's assistant message separately; when the trial is stopped (or is still
    # running when copied) the trace may hold the next turn already.
    folder = write_session(tmp_path, [assistant(bash("c1", "ls -la /app")), result("c1", "main.py")])
    files = sorted(folder.glob("*.jsonl"))
    trace = [record(1, [call("ls -la /app")]), record(2, [call("cat /app/main.py")])]
    rows, notes = record_rows(trace, files, cut=True)
    assert [(r.turn, r.label) for r in rows] == [(1, "list"), (1, "list-1")]
    assert notes == ["sess-1 turn 2: the trial was cut before the session file had this turn; record line dropped"]
    with pytest.raises(ValueError, match=r"turns \[2\]"):
        record_rows(trace, files, cut=False)


def test_a_cut_trial_drops_an_unfinished_last_session_line(tmp_path):
    folder = write_session(tmp_path, [assistant(bash("c1", "ls -la /app")), result("c1", "main.py")])
    path = next(folder.glob("*.jsonl"))
    path.write_text(path.read_text() + '{"type": "message", "message": {"role": "assis')
    rows, notes = record_rows([record(1, [call("ls -la /app")])], [path], cut=True)
    assert [(r.turn, r.label) for r in rows] == [(1, "list"), (1, "list-1")]
    assert len(notes) == 1 and "cut off" in notes[0]
    with pytest.raises(ValueError, match="not valid JSON"):
        record_rows([record(1, [call("ls -la /app")])], [path], cut=False)


def test_a_trial_without_result_json_was_cut(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "jeff-first-trace.jsonl").write_text(GOOD + CUT)
    assert trial_cut(tmp_path) == "no result.json (the trial was stopped or is still running)"
    lines, notes = read_trace(tmp_path)
    assert lines == [record(1, [])] and len(notes) == 1 and "no result.json" in notes[0]


def test_trial_cut_names_a_timeout_and_nothing_else(tmp_path):
    assert trial_cut(write_trial(tmp_path / "a", GOOD, exception_type="AgentTimeoutError")) == "AgentTimeoutError"
    assert trial_cut(write_trial(tmp_path / "b", GOOD, exception_type="RuntimeError")) is None
    assert trial_cut(write_trial(tmp_path / "c", GOOD)) is None


def test_after_a_compaction_the_state_holds_only_the_steps_pi_kept(tmp_path):
    # pi's context after a compaction holds a summary, the entries from firstKeptEntryId on, and every later entry;
    # the scout's state then covers only the steps in those entries. Entry ids: e1 is the user message, e2 the first
    # assistant message, and so on.
    entries = [
        assistant(bash("c1", "ls -la /app")),  # e2
        result("c1", "main.py"),  # e3
        assistant(bash("c2", "cat /app/main.py")),  # e4
        result("c2", "x"),  # e5
        {"type": "compaction", "summary": "s", "firstKeptEntryId": "e4"},  # e6
        assistant(bash("c3", "cat /app/main.py")),  # e7
        result("c3", "x"),
    ]
    folder = write_session(tmp_path, entries)
    kept = [new_shape("cat /app/main.py", "x")]
    trace = [
        record(1, [call("ls -la /app")]),
        record(2, [call("cat /app/main.py")], recent=[new_shape("ls -la /app", "main.py")]),
        record(3, [call("cat /app/main.py")], recent=kept),
    ]
    rows, _ = record_rows(trace, sorted(folder.glob("*.jsonl")), cut=False)
    third = [r for r in rows if r.turn == 3]
    assert third and all(r.state == (
        "Task:\nFix the bug in /app/main.py.\n\nSteps so far, oldest first:\n\n"
        "Step 1 (by you, the scout):\n$ cat /app/main.py\nx"
    ) for r in third)  # turn 2's read matched the menu, so the scout is shown as having taken it


def test_a_compaction_keeping_an_unknown_entry_is_an_error(tmp_path):
    entries = [assistant(bash("c1", "ls -la /app")), result("c1", "main.py"), {"type": "compaction", "summary": "s", "firstKeptEntryId": "zz"}]
    folder = write_session(tmp_path, entries)
    with pytest.raises(ValueError, match="firstKeptEntryId"):
        record_rows([record(1, [call("ls -la /app")])], sorted(folder.glob("*.jsonl")), cut=False)


def test_a_branched_session_file_is_an_error(tmp_path):
    folder = write_session(tmp_path, [assistant(bash("c1", "ls -la /app")), result("c1", "main.py")])
    path = next(folder.glob("*.jsonl"))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[-1]["parentId"] = "e1"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError, match="branch"):
        record_rows([record(1, [call("ls -la /app")])], [path], cut=False)
