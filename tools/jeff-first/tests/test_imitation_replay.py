import json

import pytest

from imitation.replay import (
    COMMAND_CAP_SECONDS,
    Ran,
    conversation_from_atif,
    decision_lines,
    differs,
    replay_session,
    row_json,
    stop_batch,
    task_of,
)
from imitation.rows import RowSource
from imitation.terminus import parse_terminus

FIRST = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux environment.\n\n"
    "Task Description:\nFix the bug in /app/main.py.\n\n\n"
    "Current terminal state:\nCurrent Terminal Screen:\nroot@abc123:/app#\n\n\n\n"
)


def call(keystrokes: str, duration: float = 0.1) -> dict:
    return {"tool_call_id": "c", "function_name": "bash_command", "arguments": {"keystrokes": keystrokes, "duration": duration}}


def agent(calls: list[dict], screen: str, message: str = "Analysis: look.\nPlan: act.") -> dict:
    return {
        "source": "agent",
        "message": message,
        "reasoning_content": "thinking",
        "tool_calls": calls,
        "observation": {"results": [{"content": screen}]},
    }


def trajectory(*steps: dict) -> dict:
    return {"schema_version": "ATIF-v1.7", "steps": [{"source": "user", "message": FIRST}, *steps]}


SESSION = trajectory(
    agent(
        [call("ls\n"), call("cat main.py\n")],
        "New Terminal Output:\n\nroot@abc123:/app# ls\nmain.py\nroot@abc123:/app# cat main.py\nprint(1/0)\nroot@abc123:/app#\n",
    ),
    agent(
        [call("cat > main.py <<'EOF'\nprint(1)\nEOF\n", 0.5), call("python3 main.py\n", 2.0)],
        "New Terminal Output:\n\nroot@abc123:/app# cat > main.py <<'EOF'\n> print(1)\n> EOF\n"
        "root@abc123:/app# python3 main.py\n1\nroot@abc123:/app#\n",
    ),
    agent([{"tool_call_id": "d", "function_name": "mark_task_complete", "arguments": {}}], "New Terminal Output:\n\nroot@abc123:/app#\n"),
)


def option(kind: str, number: int, description: str, command: str) -> dict:
    return {"id": f"{kind}-{number}", "description": description, "toolCall": {"name": "bash", "arguments": {"command": command}}}


MENU = {
    "tools": [
        {"id": "read", "description": "Read a file"},
        {"id": "list", "description": "List a folder"},
        {"id": "hand_over", "description": "Hand over to the coding model"},
    ],
    "arguments_by_tool": {
        "read": [option("read", 1, "Read the file /app/main.py", "cat '/app/main.py'")],
        "list": [option("list", 1, "List the folder /app", "ls -la '/app'")],
    },
}


class FakeContainer:
    """Records what the replay asks of the container, in order."""

    def __init__(self, outputs: dict[str, str] | None = None, menu: dict = MENU, seconds: float = 0.05) -> None:
        self.log: list[str] = []
        self.timeouts: list[float] = []
        self.slept: list[float] = []
        self.menu_points: list[tuple[str, list[str], str]] = []
        self.outputs = outputs or {}
        self.menu_value = menu
        self.seconds = seconds

    def run(self, command: str, timeout: float) -> Ran:
        self.log.append(f"run {command.splitlines()[0]}")
        self.timeouts.append(timeout)
        return Ran(self.outputs.get(command, ""), 0, self.seconds, False)

    def cwd(self) -> str:
        return "/app"

    def menu(self, task: str, steps, cwd: str) -> dict:
        self.log.append("menu")
        self.menu_points.append((task, [step.command for step in steps], cwd))
        return self.menu_value

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 3))


META = RowSource(source="openguardrails", stage=2, quality="near-exact", task="fix-bug", session="trial-1")


def test_atif_trajectory_becomes_a_terminus_conversation():
    session = parse_terminus(conversation_from_atif(SESSION))
    assert session.task == "Fix the bug in /app/main.py." and session.cwd == "/app"
    assert [[c.text for c in turn.commands] for turn in session.turns] == [
        ["ls", "cat main.py"],
        ["cat > main.py <<'EOF'\nprint(1)\nEOF", "python3 main.py"],
        [],
    ]
    assert session.turns[0].commands[1].output == "print(1/0)"
    assert session.turns[1].durations == [0.5, 2.0]
    assert session.turns[0].thinking == "thinking" and session.turns[2].task_complete


def test_atif_step_from_an_unknown_source_is_an_error():
    with pytest.raises(ValueError, match="system"):
        conversation_from_atif(trajectory({"source": "system", "message": "x"}))


def test_atif_agent_step_with_several_observation_results_is_an_error():
    step = agent([call("ls\n")], "New Terminal Output:\n\nroot@abc123:/app# ls\n")
    step["observation"]["results"].append({"content": "more"})
    with pytest.raises(ValueError, match="2 observation results"):
        conversation_from_atif(trajectory(step))


def test_task_of_reads_the_task_folder_name():
    assert task_of({"task": {"path": "/home/tom/qwenbench/tasks/terminal-bench__build-pmars"}}) == "build-pmars"
    with pytest.raises(ValueError, match="other__x"):
        task_of({"task": {"path": "/tasks/other__x"}})


def test_menus_are_built_in_the_container_before_each_point_and_commands_replayed_in_between():
    container = FakeContainer(outputs={"ls": "main.py\n", "cat main.py": "print(1/0)\n"})
    replay = replay_session(META, parse_terminus(conversation_from_atif(SESSION)), container)
    # Point before turn 1 (List), after ls (Read; a stint), before turn 2 (hand over: it writes), before turn 3.
    assert container.log == [
        "menu",
        "run ls",
        "menu",
        "run cat main.py",
        "menu",
        "run cat > main.py <<'EOF'",
        "run python3 main.py",
        "menu",
    ]
    # A step credited to the scout shows the scout option's own command (with the coding model's real output).
    assert container.menu_points[1] == ("Fix the bug in /app/main.py.", ["ls -la '/app'"], "/app")
    labels = [(row.decision, row.turn, row.level, row.label) for row in replay.rows]
    assert labels == [
        (0, 1, "tool", "list"),
        (0, 1, "argument", "list-1"),
        (1, 1, "tool", "read"),
        (1, 1, "argument", "read-1"),
        (2, 2, "tool", "hand_over"),
        (3, 3, "tool", "hand_over"),
    ]
    assert {(row.stage, row.quality, row.source, row.task, row.session) for row in replay.rows} == {
        (2, "near-exact", "openguardrails", "fix-bug", "trial-1")
    }
    assert replay.result.end_reason == "complete" and replay.result.decisions == 4


def test_each_command_waits_its_session_duration_and_runs_under_the_cap():
    container = FakeContainer(seconds=0.5)
    replay_session(META, parse_terminus(conversation_from_atif(SESSION)), container)
    assert container.timeouts == [COMMAND_CAP_SECONDS] * 4
    # ls and cat waited 0.1 s in the session but took 0.5 s here: no extra wait; python3 waited 2 s.
    assert container.slept == [1.5]


def test_a_command_the_session_interrupted_with_c_c_runs_only_for_its_wait():
    steps = trajectory(
        agent([call("python3 serve.py\n", 5.0)], "New Terminal Output:\n\nroot@abc123:/app# python3 serve.py\nserving\n"),
        agent(
            [{"tool_call_id": "k", "function_name": "bash_command", "arguments": {"keystrokes": "C-c", "duration": 0.1}}, call("ls\n")],
            "New Terminal Output:\n\n^C\nroot@abc123:/app# ls\nmain.py\nroot@abc123:/app#\n",
        ),
        agent([{"tool_call_id": "d", "function_name": "mark_task_complete", "arguments": {}}], "New Terminal Output:\n\nroot@abc123:/app#\n"),
    )
    container = FakeContainer()
    replay_session(META, parse_terminus(conversation_from_atif(steps)), container)
    assert container.timeouts == [5.0, COMMAND_CAP_SECONDS]


def test_a_wait_only_reply_adds_to_the_wait_of_the_command_before_it():
    steps = trajectory(
        agent([call("make\n", 10.0)], "New Terminal Output:\n\nroot@abc123:/app# make\nbuilding\n"),
        agent([call("", 30.0)], "New Terminal Output:\n\ndone\nroot@abc123:/app#\n"),
        agent([call("ls\n")], "New Terminal Output:\n\nroot@abc123:/app# ls\nmain.py\nroot@abc123:/app#\n"),
    )
    container = FakeContainer(seconds=1.0)
    replay_session(META, parse_terminus(conversation_from_atif(steps)), container)
    assert container.slept == [39.0]


def test_replayed_outputs_are_recorded_and_information_mismatches_counted():
    container = FakeContainer(outputs={"ls": "main.py\n  \n", "cat main.py": "print(2/0)\n", "python3 main.py": "2\n"})
    replay = replay_session(META, parse_terminus(conversation_from_atif(SESSION)), container)
    by_command = {c.command.splitlines()[0]: c for c in replay.commands}
    assert by_command["ls"].transcript_output == "main.py" and by_command["ls"].replay_output == "main.py\n  \n"
    assert not by_command["ls"].mismatch
    assert by_command["cat main.py"].information and by_command["cat main.py"].mismatch
    # The run differs too, but it is not an information command.
    assert not by_command["python3 main.py"].information and by_command["python3 main.py"].mismatch
    assert replay.information_mismatches == 1


def test_differs_ignores_whitespace_including_wrapped_lines():
    assert not differs("abc def\nghi", "abc def ghi\n")
    assert not differs("a" * 160 + "\n" + "b", "a" * 160 + "b")
    assert differs("total 8", "total 12")


def test_replay_stops_at_the_last_point():
    steps = trajectory(
        agent([call("ls\n")], "New Terminal Output:\n\nroot@abc123:/app# ls\nmain.py\nroot@abc123:/app#\n"),
    )
    container = FakeContainer()
    replay_session(META, parse_terminus(conversation_from_atif(steps)), container)
    # The only point is before turn 1; ls is its label and nothing after it needs a menu.
    assert container.log == ["menu"]


def test_rows_carry_the_machine():
    container = FakeContainer()
    replay = replay_session(META, parse_terminus(conversation_from_atif(SESSION)), container)
    row = json.loads(row_json(replay.rows[0], "casdgx01"))
    assert row["machine"] == "casdgx01" and row["stage"] == 2 and row["quality"] == "near-exact"
    assert row["source"] == "openguardrails" and row["label"] == "list"


def test_the_batch_stops_once_failures_reach_five_percent():
    assert not stop_batch(failures=2, total=45)
    assert stop_batch(failures=3, total=45)
    assert not stop_batch(failures=0, total=1)
    assert stop_batch(failures=1, total=10)


def test_argument_rows_carry_the_tool_description_and_decisions_keep_their_full_menus():
    container = FakeContainer(outputs={"ls": "main.py\n"})
    replay = replay_session(META, parse_terminus(conversation_from_atif(SESSION)), container)
    argument = json.loads(row_json(replay.rows[1], "casdgx01"))
    assert argument["level"] == "argument" and argument["tool_description"] == "List a folder"
    records = [json.loads(line) for line in decision_lines(replay, "casdgx01")]
    assert [(r["decision"], r["turn"], r["kind"], r["option"]) for r in records] == [
        (0, 1, "list", "list-1"),
        (1, 1, "read", "read-1"),
        (2, 2, None, None),
        (3, 3, None, None),
    ]
    assert records[0]["menu"] == MENU and records[0]["session"] == "trial-1" and records[0]["machine"] == "casdgx01"
