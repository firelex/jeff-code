import json
import subprocess
from pathlib import Path

import pytest

from imitation.pi_replay import (
    COMMAND_CAP_SECONDS,
    PiContainer,
    PiRan,
    SessionTiming,
    agent_file_prefixes,
    session_timing,
    command_script,
    menu_difference,
    pi_output_differs,
    replay_record_session,
    session_excluded,
    setup_commands,
)
from imitation.record_rows import record_sessions


def option(kind, number, description, command):
    return {"id": f"{kind}-{number}", "description": description, "toolCall": {"name": "bash", "arguments": {"command": command}}}


TOOLS = [
    {"id": "read", "description": "Read a file"},
    {"id": "list", "description": "List a folder"},
    {"id": "hand_over", "description": "Hand over to the coding model"},
]
LISTS = {
    "tools": TOOLS,
    "arguments_by_tool": {
        "read": [option("read", 1, "Read the file /app/main.py", "cat '/app/main.py'")],
        "list": [option("list", 1, "List the folder /app", "ls -la '/app'")],
    },
}
# What the container builds inside a turn: the read option has moved to a second slot, so rows show which menu they use.
REPLAYED = {
    "tools": TOOLS,
    "arguments_by_tool": {
        "read": [
            option("read", 1, "Read the file /app/notes.txt", "cat '/app/notes.txt'"),
            option("read", 2, "Read the file /app/main.py", "cat '/app/main.py'"),
        ],
        "list": [option("list", 1, "List the folder /app", "ls -la '/app'")],
    },
}


def bash(call_id, command, **extra):
    return {"type": "toolCall", "id": call_id, "name": "bash", "arguments": {"command": command, **extra}}


def result(call_id, text, is_error=False):
    return {"type": "message", "message": {"role": "toolResult", "toolCallId": call_id, "content": [{"type": "text", "text": text}], "isError": is_error}}


def assistant(*calls):
    return {"type": "message", "message": {"role": "assistant", "provider": "harbor-endpoint", "content": [{"type": "text", "text": "ok"}, *calls]}}


def new_shape(command, output):
    return {"command": command, "output": output, "isError": False, "byScout": False}


def record(turn, tool_calls, recent=(), left_out=0):
    return {
        "schema": "jeff-first-trace/4",
        "kind": "record",
        "task_id": "fix-bug",
        "session_id": "sess-1",
        "turn": turn,
        "mode": "record",
        "state": {"task": "Fix the bug in /app/main.py.", "recentSteps": list(recent), "stepsLeftOut": left_out},
        "lists": LISTS,
        "action": {"stop_reason": "toolUse", "error_message": None, "text_chars": 2, "tool_calls": tool_calls},
    }


def write_session(tmp_path, entries, times=None):
    folder = tmp_path / "sessions"
    folder.mkdir()
    header = {"type": "session", "version": 3, "id": "sess-1", "timestamp": "2026-10-03T22:00:00.000Z", "cwd": "/app"}
    lines = [{"type": "message", "message": {"role": "user", "content": "Fix the bug in /app/main.py."}}, *entries]
    times = times or ["2026-10-03T22:00:00.000Z"] * len(lines)
    chained = [
        {**line, "id": f"e{n}", "parentId": None if n == 1 else f"e{n - 1}", "timestamp": time}
        for n, (line, time) in enumerate(zip(lines, times), start=1)
    ]
    path = folder / "s.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in [header, *chained]))
    return path


def tool_call(command, **extra):
    return {"name": "bash", "arguments": {"command": command, **extra}}


class FakeContainer:
    """Records what the replay asks of the container, in order."""

    def __init__(self, outputs=None, menus=None, seconds=0.05):
        self.log = []
        self.timeouts = []
        self.slept = []
        self.full_output_paths = []
        self.menu_steps = []
        self.outputs = outputs or {}
        self.menus = list(menus or [])
        self.seconds = seconds

    def run(self, command, timeout, full_output_path):
        self.log.append(f"run {command.splitlines()[0]}")
        self.timeouts.append(timeout)
        self.full_output_paths.append(full_output_path)
        output, code = self.outputs.get(command, ("", 0))
        return PiRan(output, code, self.seconds, False)

    def menu(self, task, steps):
        self.log.append("menu")
        self.menu_steps.append([(step.command, step.by_scout) for step in steps])
        return self.menus.pop(0) if self.menus else LISTS

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))

    def settle(self, seconds):
        self.log.append(f"settle {seconds:g}")

    def write_files(self, files):
        self.log.append(f"files {sorted(files.items())}")


SESSION = [
    assistant(bash("c1", "ls -la /app"), bash("c2", "cat /app/main.py")),
    result("c1", "main.py"),
    result("c2", "print(1/0)"),
    assistant(bash("c3", "cat > /app/main.py <<'EOF'\nprint(1)\nEOF")),
    result("c3", "(no output)"),
]
TRACE = [
    record(1, [tool_call("ls -la /app"), tool_call("cat /app/main.py")]),
    record(2, [tool_call("cat > /app/main.py <<'EOF'\nprint(1)\nEOF")], recent=[new_shape("ls -la /app", "main.py"), new_shape("cat /app/main.py", "print(1/0)")]),
]


def run(prepared_session, batches, container, thinking=None, think_waits=False, files=None):
    timing = SessionTiming(batches, thinking or [0.0] * len(batches))
    agent_files = (lambda turn: files(turn)) if files else (lambda turn: {})
    return replay_record_session(prepared_session, timing, container, machine="m", think_waits=think_waits, agent_files=agent_files)


def prepared(tmp_path, entries=SESSION, trace=TRACE):
    path = write_session(tmp_path, entries)
    sessions, notes = record_sessions(trace, [path], cut=False, source="own", quality="exact-replayed")
    assert notes == []
    return sessions[0]


def test_menus_are_built_before_each_turn_and_after_each_stint_step(tmp_path):
    # Points: before turn 1 (List), after ls inside turn 1 (Read: a stint), before turn 2 (hand over: it writes).
    container = FakeContainer(outputs={"ls -la /app": ("main.py\n", 0), "cat /app/main.py": ("print(1/0)\n", 0)}, menus=[LISTS, REPLAYED, LISTS])
    timing = SessionTiming([0.0, 0.0], [0.0, 0.0])
    replay = replay_record_session(
        prepared(tmp_path), timing, container, machine="qwen3.8-27b-fp8@casdgx01-gpu5", think_waits=False, agent_files=lambda turn: {}
    )
    assert container.log == ["menu", "run ls -la /app", "menu", "run cat /app/main.py", "menu"]
    # Before a turn the menu is built from the coding model's own steps (what pi's live scout saw); inside a turn the
    # stint step shows the scout option's command.
    assert container.menu_steps == [[], [("ls -la '/app'", True)], [("ls -la /app", False), ("cat /app/main.py", False)]]
    labels = [(row.decision, row.turn, row.level, row.label) for row in replay.rows]
    assert labels == [
        (0, 1, "tool", "list"),
        (0, 1, "argument", "list-1"),
        (1, 1, "tool", "read"),
        (1, 1, "argument", "read-2"),
        (2, 2, "tool", "hand_over"),
    ]
    # The stint row uses the menu built in the container, the others the logged menus.
    assert [option["id"] for option in replay.rows[3].options] == ["read-1", "read-2", "none_of_these"]
    assert [point.stint for point in replay.points] == [False, True, False]
    assert replay.stint_rows == 2
    assert {(r.source, r.stage, r.quality, r.task, r.session) for r in replay.rows} == {("own", 3, "exact-replayed", "fix-bug", "sess-1")}
    assert json.loads(replay.row_lines()[0])["machine"] == "qwen3.8-27b-fp8@casdgx01-gpu5"
    # Later states show the stint step as the scout's.
    assert "Step 1 (by you, the scout):\n$ ls -la '/app'\nmain.py" in replay.rows[4].state


def test_each_before_turn_menu_is_compared_with_the_logged_one(tmp_path):
    container = FakeContainer(menus=[LISTS, REPLAYED, REPLAYED])
    replay = run(prepared(tmp_path), [0.0, 0.0], container)
    assert [(check.turn, check.equal) for check in replay.menu_checks] == [(1, True), (2, False)]
    assert replay.menu_checks[1].difference == {"read": {"added": ["Read the file /app/notes.txt"], "removed": []}}


def test_menu_difference_names_added_and_removed_tools_and_options():
    assert menu_difference(LISTS, LISTS) == {}
    fewer = {"tools": TOOLS[1:], "arguments_by_tool": {"list": LISTS["arguments_by_tool"]["list"]}}
    assert menu_difference(LISTS, fewer) == {"tools": {"added": [], "removed": ["read"]}, "read": {"added": [], "removed": ["Read the file /app/main.py"]}}


def test_an_unmatched_acting_command_ends_the_stint_with_hand_over(tmp_path):
    entries = [assistant(bash("c1", "ls -la /app"), bash("c2", "python3 /app/main.py")), result("c1", "main.py"), result("c2", "1")]
    trace = [record(1, [tool_call("ls -la /app"), tool_call("python3 /app/main.py")])]
    container = FakeContainer()
    replay = run(prepared(tmp_path, entries, trace), [0.0], container)
    assert [(r.decision, r.level, r.label) for r in replay.rows] == [(0, "tool", "list"), (0, "argument", "list-1"), (1, "tool", "hand_over")]
    assert [point.stint for point in replay.points] == [False, True]
    assert container.log == ["menu", "run ls -la /app", "menu"]


def test_a_command_runs_under_its_own_timeout_or_the_cap(tmp_path):
    entries = [
        assistant(bash("c1", "make", timeout=600)),
        result("c1", "built"),
        assistant(bash("c2", "make test", timeout=5000)),
        result("c2", "ok"),
        assistant(bash("c3", "ls")),
        result("c3", "a"),
        assistant(bash("c4", "ls -la")),
        result("c4", "a"),
    ]
    steps = [new_shape("make", "built"), new_shape("make test", "ok"), new_shape("ls", "a")]
    trace = [record(1, [tool_call("make", timeout=600)])] + [
        record(n, [tool_call(command, **extra)], recent=steps[: n - 1])
        for n, (command, extra) in enumerate([("make test", {"timeout": 5000}), ("ls", {}), ("ls -la", {})], start=2)
    ]
    container = FakeContainer()
    run(prepared(tmp_path, entries, trace), [0.0] * 4, container)
    assert container.timeouts == [600, COMMAND_CAP_SECONDS, COMMAND_CAP_SECONDS]


def test_a_turn_that_ran_shorter_than_in_the_session_waits_the_rest(tmp_path):
    container = FakeContainer(seconds=0.5)
    run(prepared(tmp_path), [3.0, 0.0], container)
    # Both commands of turn 1 took 0.5 s here; the session spent 3 s on the turn's commands.
    assert container.slept == [2.0]


def test_replayed_outputs_are_compared_as_pi_shows_them(tmp_path):
    entries = [assistant(bash("c1", "cat /app/x")), result("c1", "cat: /app/x: No such file or directory\n\nCommand exited with code 1", True), assistant(bash("c2", "ls")), result("c2", "(no output)")]
    trace = [record(1, [tool_call("cat /app/x")]), record(2, [tool_call("ls")], recent=[new_shape("cat /app/x", "cat: /app/x: No such file or directory\n\nCommand exited with code 1")])]
    container = FakeContainer(outputs={"cat /app/x": ("cat: /app/x: No such file or directory\n", 1)})
    replay = run(prepared(tmp_path, entries, trace), [0.0, 0.0], container)
    assert [(c.command, c.mismatch, c.information) for c in replay.commands] == [("cat /app/x", False, True)]


def test_a_mismatch_in_digits_only_is_told_apart(tmp_path):
    entries = [assistant(bash("c1", "date")), result("c1", "Sat Oct  4 03:05:00 UTC 2026"), assistant(bash("c2", "ls")), result("c2", "a")]
    trace = [record(1, [tool_call("date")]), record(2, [tool_call("ls")], recent=[new_shape("date", "Sat Oct  4 03:05:00 UTC 2026")])]
    container = FakeContainer(outputs={"date": ("Sat Oct  4 05:49:12 UTC 2026\n", 0)})
    replay = run(prepared(tmp_path, entries, trace), [0.0, 0.0], container)
    assert [(c.mismatch, c.mismatch_beyond_digits) for c in replay.commands] == [(True, False)]


def test_pi_output_differs_ignores_whitespace_and_pi_s_truncation_notice():
    assert not pi_output_differs("a b\nc", "a b c\n", 0, False, None, ignore_digits=False)
    assert not pi_output_differs("(no output)", "", 0, False, None, ignore_digits=False)
    assert not pi_output_differs("oops\n\nCommand exited with code 2", "oops\n", 2, False, None, ignore_digits=False)
    assert pi_output_differs("oops", "oops\n", 2, False, None, ignore_digits=False)
    # pi writes "(no output)" before the exit status too (tools/bash.ts formatOutput), but not before a timeout.
    assert not pi_output_differs("(no output)\n\nCommand exited with code 1", "", 1, False, None, ignore_digits=False)
    assert not pi_output_differs("Command timed out after 5 seconds", "", 124, True, 5, ignore_digits=False)
    assert not pi_output_differs("partial\n\nCommand timed out after 5 seconds", "partial\n", 124, True, 5, ignore_digits=False)
    shown = "line 3\nline 4\n\n[Showing lines 3-4 of 4. Full output: /tmp/pi-bash-abc.log]"
    assert not pi_output_differs(shown, "line 1\nline 2\nline 3\nline 4\n", 0, False, None, ignore_digits=False)
    assert pi_output_differs(shown, "line 1\nline 2\nline 3\nline 5\n", 0, False, None, ignore_digits=False)
    # pi writes the exit status after the truncation notice.
    failed = "line 4\n\n[Showing lines 4-4 of 4. Full output: /tmp/pi-bash-abc.log]\n\nCommand exited with code 2"
    assert not pi_output_differs(failed, "line 1\nline 2\nline 3\nline 4\n", 2, False, None, ignore_digits=False)
    assert pi_output_differs(failed, "line 1\nline 2\nline 3\nline 4\n", 0, False, None, ignore_digits=False)
    assert not pi_output_differs("took 3 s\n\nCommand exited with code 1", "took 12 s\n", 1, False, None, ignore_digits=True)


def test_compacted_steps_are_left_out_of_menus_and_states(tmp_path):
    entries = [
        assistant(bash("c1", "ls -la /app")),
        result("c1", "main.py"),
        assistant(bash("c2", "cat /app/main.py")),
        result("c2", "print(1/0)"),
        # pi compacts, keeping the entries from the second assistant message (e4) on.
        {"type": "compaction", "firstKeptEntryId": "e4", "summary": "s"},
        assistant(bash("c3", "ls")),
        result("c3", "main.py"),
    ]
    trace = [
        record(1, [tool_call("ls -la /app")]),
        record(2, [tool_call("cat /app/main.py")], recent=[new_shape("ls -la /app", "main.py")]),
        record(3, [tool_call("ls")], recent=[new_shape("cat /app/main.py", "print(1/0)")]),
    ]
    container = FakeContainer()
    replay = run(prepared(tmp_path, entries, trace), [0.0] * 3, container)
    assert container.menu_steps[2] == [("cat /app/main.py", False)]
    assert "ls -la" not in replay.rows[-1].state


def test_a_session_is_excluded_when_more_than_a_tenth_of_its_logged_menus_differ():
    assert not session_excluded(equal=9, differing=1)
    assert session_excluded(equal=8, differing=2)
    with pytest.raises(ValueError, match="no menu"):
        session_excluded(equal=0, differing=0)


def test_session_timing_holds_each_turns_command_time_and_thinking_time(tmp_path):
    times = [
        "2026-10-03T22:00:00.000Z",  # user
        "2026-10-03T22:00:01.000Z",  # assistant 1 (written when the model's reply ended)
        "2026-10-03T22:00:03.500Z",  # its results
        "2026-10-03T22:00:03.500Z",
        "2026-10-03T22:00:10.000Z",  # assistant 2
        "2026-10-03T22:00:10.250Z",
    ]
    path = write_session(tmp_path, SESSION, times)
    timing = session_timing(path)
    assert timing.batches == [2.5, 0.25]
    # The model's reply time: from the entry before the assistant message to the message.
    assert timing.thinking == [1.0, 6.5]


def test_with_think_waits_background_programs_get_the_models_thinking_time_before_a_turns_commands(tmp_path):
    entries = [assistant(bash("c1", "apt-get install -y gcc > /tmp/apt.log 2>&1 &")), result("c1", "(no output)"), assistant(bash("c2", "which gcc")), result("c2", "/usr/bin/gcc"), assistant(bash("c3", "ls")), result("c3", "a")]
    trace = [
        record(1, [tool_call("apt-get install -y gcc > /tmp/apt.log 2>&1 &")]),
        record(2, [tool_call("which gcc")], recent=[new_shape("apt-get install -y gcc > /tmp/apt.log 2>&1 &", "(no output)")]),
        record(3, [tool_call("ls")], recent=[new_shape("apt-get install -y gcc > /tmp/apt.log 2>&1 &", "(no output)"), new_shape("which gcc", "/usr/bin/gcc")]),
    ]
    container = FakeContainer()
    run(prepared(tmp_path, entries, trace), [0.0] * 3, container, thinking=[1.0, 40.0, 3.0], think_waits=True)
    # The menu before turn 2 shows the disk right after turn 1; the model then thought for 40 s before `which gcc`.
    assert container.log == ["menu", "settle 1", "run apt-get install -y gcc > /tmp/apt.log 2>&1 &", "menu", "settle 40", "run which gcc", "menu"]


def test_agent_files_are_written_as_they_were_before_each_turns_menu(tmp_path):
    container = FakeContainer()
    run(prepared(tmp_path), [0.0, 0.0], container, files=lambda turn: {"/logs/agent/pi.txt": f"up to turn {turn}"})
    assert container.log[0] == "files [('/logs/agent/pi.txt', 'up to turn 1')]"
    assert container.log[-2:] == ["files [('/logs/agent/pi.txt', 'up to turn 2')]", "menu"]


def test_agent_file_prefixes_end_before_the_turns_assistant_message():
    pi_log = "".join(
        json.dumps(line) + "\n"
        for line in [
            {"type": "session"},
            {"type": "turn_start"},
            {"type": "message_start", "message": {"role": "assistant"}},
            {"type": "turn_end"},
            {"type": "turn_start"},
            {"type": "message_start", "message": {"role": "assistant"}},
        ]
    )
    session_file = "".join(
        json.dumps(line) + "\n"
        for line in [{"type": "session"}, {"type": "message", "message": {"role": "user"}}, {"type": "message", "message": {"role": "assistant"}}, {"type": "message", "message": {"role": "toolResult"}}, {"type": "message", "message": {"role": "assistant"}}]
    )
    prefixes = agent_file_prefixes({"/logs/agent/pi.txt": ("pi-log", pi_log), "/logs/agent/pi/sessions/s.jsonl": ("session", session_file)})
    assert prefixes(1) == {"/logs/agent/pi.txt": "".join(pi_log.splitlines(keepends=True)[:2]), "/logs/agent/pi/sessions/s.jsonl": "".join(session_file.splitlines(keepends=True)[:2])}
    assert prefixes(2)["/logs/agent/pi.txt"] == "".join(pi_log.splitlines(keepends=True)[:5])
    # pi's event output may end before the session does (pi was stopped before it flushed): from then on the file
    # never grew again, so it is whole.
    assert agent_file_prefixes({"/logs/agent/pi.txt": ("pi-log", pi_log)})(3) == {"/logs/agent/pi.txt": pi_log}
    with pytest.raises(ValueError, match="turn 3"):
        agent_file_prefixes({"/logs/agent/pi/sessions/s.jsonl": ("session", session_file)})(3)


def test_setup_commands_are_the_harbor_install_commands_of_the_trial_log():
    log = (
        "Running command: apt-get update && apt-get install -y curl\nCommand outputs captured\n"
        "Running command: set -euo pipefail; curl -o- https://x/install.sh | bash && nvm install 22 && npm install -g --ignore-scripts /tmp/jeff-pi.tgz && pi --version\n"
        "Command outputs captured\n"
        "Running command: mkdir -p /tmp/harbor-pi-agent && chmod 700 /tmp/harbor-pi-agent\nCommand outputs captured\n"
        "Running command: chmod 600 /tmp/harbor-pi-agent/models.json\nCommand outputs captured\n"
        "Running command: . ~/.nvm/nvm.sh; PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print 'task\nmore'\n"
    )
    assert setup_commands(log) == [
        "apt-get update && apt-get install -y curl",
        "set -euo pipefail; curl -o- https://x/install.sh | bash && nvm install 22 && npm install -g --ignore-scripts /tmp/jeff-pi.tgz && pi --version",
        "mkdir -p /tmp/harbor-pi-agent && chmod 700 /tmp/harbor-pi-agent",
        "chmod 600 /tmp/harbor-pi-agent/models.json",
    ]
    with pytest.raises(ValueError, match="unknown setup command"):
        setup_commands("Running command: rm -rf /\nRunning command: . ~/.nvm/nvm.sh; pi --print 'x'\n")
    with pytest.raises(ValueError, match="pi --print"):
        setup_commands("Running command: mkdir -p /tmp/harbor-pi-agent && chmod 700 /tmp/harbor-pi-agent\n")


def test_a_command_runs_as_pi_runs_it_in_a_new_bash_in_the_session_folder():
    script = command_script("echo $HOME && cd sub", "/app", {"PI_SESSION_ID": "sess-1", "JEFF_FIRST_MODE": "record"})
    lines = script.splitlines()
    assert lines[0] == ". ~/.nvm/nvm.sh"
    assert "export JEFF_FIRST_MODE=record" in lines and "export PI_SESSION_ID=sess-1" in lines
    assert "export PATH=/tmp/harbor-pi-agent/bin:$PATH" in lines
    assert lines[-2:] == ["cd /app", "exec /bin/bash -c 'echo $HOME && cd sub'"]


def test_docker_output_is_read_as_pi_reads_it_with_invalid_utf8_replaced(monkeypatch):
    # A command may print bytes that are not UTF-8 (a binary file); pi's bash tool decodes them with U+FFFD.
    seen = {}

    def fake_run(args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, b"caf\xe9\n".decode(kwargs["encoding"], kwargs["errors"]), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    container = PiContainer("stage3replay-x", "image", 1, "2G", Path("/scout"))
    assert container._checked(["exec", "x"], 10) == "caf�\n"
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace"


def test_a_truncated_output_is_saved_where_pi_saved_the_full_output(tmp_path):
    # pi keeps the full output of a long command in /tmp/pi-bash-<id>.log and names it; that file is on the disk
    # from then on (later menus may offer it), so the replay saves its own full output there.
    shown = "end\n\n[Showing lines 1999-4000 of 4000. Full output: /tmp/pi-bash-bade5f4c3e5e6376.log]"
    entries = [assistant(bash("c1", "seq 4000")), result("c1", shown), assistant(bash("c2", "ls")), result("c2", "a")]
    trace = [record(1, [tool_call("seq 4000")]), record(2, [tool_call("ls")], recent=[new_shape("seq 4000", shown)])]
    container = FakeContainer()
    run(prepared(tmp_path, entries, trace), [0.0, 0.0], container)
    assert container.full_output_paths == ["/tmp/pi-bash-bade5f4c3e5e6376.log"]


def test_a_truncated_failed_output_names_its_full_output_file_too(tmp_path):
    shown = "end\n\n[Showing lines 1999-4000 of 4000. Full output: /tmp/pi-bash-c8005cd5fa54e4e2.log]\n\nCommand exited with code 1"
    entries = [assistant(bash("c1", "make")), result("c1", shown, True), assistant(bash("c2", "ls")), result("c2", "a")]
    trace = [record(1, [tool_call("make")]), record(2, [tool_call("ls")], recent=[new_shape("make", shown)])]
    container = FakeContainer()
    run(prepared(tmp_path, entries, trace), [0.0, 0.0], container)
    assert container.full_output_paths == ["/tmp/pi-bash-c8005cd5fa54e4e2.log"]
