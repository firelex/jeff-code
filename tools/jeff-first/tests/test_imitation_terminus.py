import json
from pathlib import Path

import pytest

from imitation.labels import TurnCommand
from imitation.rows import RowSource
from imitation.terminus import (
    MenuPoint,
    build_menus,
    convert_sessions,
    latest_episodes,
    parse_terminus,
    session_from_dataset_row,
)

FIXTURE = Path(__file__).parent / "fixtures" / "terminus_inferredbugs_0405.json"

FIRST_USER = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux environment.\n\n"
    "Task Description:\nFix the bug in /app/main.py.\n\n\n"
    "Current terminal state:\nCurrent Terminal Screen:\nroot@abc123:/app#\n\n\n\n"
)


def assistant(commands: list[str], analysis: str = "a", plan: str = "p", think: str = "thinking") -> dict:
    body = {"analysis": analysis, "plan": plan, "commands": [{"keystrokes": k, "duration": 0.1} for k in commands]}
    return {"role": "assistant", "content": f"<think>{think}</think>\n\nSome words first.\n{json.dumps(body, indent=2)}"}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def conversation() -> list[dict]:
    return [
        user(FIRST_USER),
        assistant(["ls -la\n", "cat main.py\n"], analysis="look", plan="list then read"),
        user(
            "New Terminal Output:\n\nroot@abc123:/app# ls -la\ntotal 8\n-rw-r--r-- 1 root root 20 main.py\n"
            "root@abc123:/app# cat main.py\nprint(1/0)\nroot@abc123:/app#\n\n\n"
        ),
        assistant(["cat > main.py <<'EOF'\nprint(1)\nEOF\n", "python3 main.py\n"]),
        user(
            "Current terminal state:\nNew Terminal Output:\n\nroot@abc123:/app# cat > main.py <<'EOF'\n> print(1)\n> EOF\n"
            "root@abc123:/app# python3 main.py\n1\nroot@abc123:/app#\n\n\n"
            "Are you sure you want to mark the task as complete? This will trigger your solution to be graded."
        ),
        {"role": "assistant", "content": "<think>x</think>I am done."},
        user("Previous response had parsing errors:\nERROR: No valid JSON found\n\nPlease fix these issues and provide a proper JSON response."),
        assistant(["C-c"]),
        user("New Terminal Output:\n\nroot@abc123:/app#\n"),
    ]


def test_parse_hand_written_session():
    session = parse_terminus(conversation())
    assert session.task == "Fix the bug in /app/main.py."
    assert session.cwd == "/app"
    assert len(session.turns) == 3
    first = session.turns[0]
    assert (first.analysis, first.plan, first.thinking) == ("look", "list then read", "thinking")
    assert first.commands == [
        TurnCommand("ls -la", "total 8\n-rw-r--r-- 1 root root 20 main.py", False),
        TurnCommand("cat main.py", "print(1/0)", False),
    ]
    second = session.turns[1]
    assert second.commands == [
        TurnCommand("cat > main.py <<'EOF'\nprint(1)\nEOF", "", False),
        TurnCommand("python3 main.py", "1", False),
    ]
    third = session.turns[2]
    assert third.parse_error and third.commands == []
    assert session.end_reason == "interactive" and "C-c" in session.end_detail and "turn 4" in session.end_detail


def test_parse_real_terminus_session():
    row = json.loads(FIXTURE.read_text())
    session = parse_terminus(row["conversations"])
    assert session.task.startswith("# InferredBugs Task - Csharp")
    assert session.task.endswith("Fix the bug in the code.")
    assert session.cwd == "/app"
    assert session.end_reason == "complete" and session.end_detail is None
    assert [c.text for c in session.turns[0].commands] == [
        "ls -la",
        "find /app -name 'EffTaskAwaiter.cs' -o -name 'EffTask*.cs' | head -50",
    ]
    assert session.turns[0].commands[0].output == (
        "total 0\ndrwxr-xr-x. 2 root root  6 Aug 26 10:46 .\ndrwxr-xr-x. 1 root root 62 Aug 27 10:40 .."
    )
    assert session.turns[0].commands[1].output == ""
    assert len(session.turns[1].commands) == 3
    assert all(command.output is not None for command in session.turns[1].commands)


def test_output_still_coming_after_the_screen_joins_the_previous_command():
    conv = [
        user(FIRST_USER),
        assistant(["make\n"]),
        user("New Terminal Output:\n\nroot@abc123:/app# make\ncompiling a.c\n"),
        assistant([""]),
        user("New Terminal Output:\n\ncompiling b.c\ndone\nroot@abc123:/app#\n"),
    ]
    session = parse_terminus(conv)
    assert session.turns[0].commands == [TurnCommand("make", "compiling a.c\ncompiling b.c\ndone", False)]
    assert session.turns[1].commands == []


def test_a_command_whose_echo_is_not_on_the_screen_has_no_output():
    conv = [user(FIRST_USER), assistant(["ls\n", "pwd\n"]), user("New Terminal Output:\n\nroot@abc123:/app# pwd\n/app\nroot@abc123:/app#\n")]
    session = parse_terminus(conv)
    assert session.turns[0].commands == [TurnCommand("ls", None, False), TurnCommand("pwd", "/app", False)]


@pytest.mark.parametrize("keystrokes", ["vim main.py\n", "python3\n", "q", "Escape", ":wq\n"])
def test_interactive_keystrokes_end_the_session_before_that_turn(keystrokes):
    conv = [
        user(FIRST_USER),
        assistant(["ls\n"]),
        user("New Terminal Output:\n\nroot@abc123:/app# ls\nmain.py\nroot@abc123:/app#\n"),
        assistant(["vim main.py\n"]) if keystrokes == ":wq\n" else assistant([keystrokes]),
        user("New Terminal Output:\n\n"),
        assistant([keystrokes]),
        user("New Terminal Output:\n\n"),
    ]
    session = parse_terminus(conv)
    assert len(session.turns) == 1
    assert session.end_reason == "interactive" and "turn 2" in session.end_detail


def test_reply_without_json_that_the_harness_accepted_is_an_error():
    conv = [user(FIRST_USER), {"role": "assistant", "content": "no json here"}, user("New Terminal Output:\n\nroot@abc123:/app#\n")]
    with pytest.raises(ValueError, match="turn 1"):
        parse_terminus(conv)


def test_first_message_without_task_is_an_error():
    with pytest.raises(ValueError, match="Task Description"):
        parse_terminus([user("hello"), assistant(["ls\n"])])


def option(kind, number, description):
    return {"id": f"{kind}-{number}", "description": description, "toolCall": {"name": "bash", "arguments": {"command": "x"}}}


MENU = {
    "tools": [
        {"id": "read", "description": "Read a file"},
        {"id": "list", "description": "List a folder"},
        {"id": "hand_over", "description": "Hand over to the coding model"},
    ],
    "arguments_by_tool": {
        "read": [option("read", 1, "Read the file /app/main.py")],
        "list": [option("list", 1, "List the folder /app")],
    },
}


def test_convert_sessions_labels_every_point_and_asks_for_menus_in_batches():
    session = parse_terminus(conversation())
    meta = RowSource(source="test", stage=1, quality="approximate", task="t", session="s")
    batches: list[list[MenuPoint]] = []

    def menus_for(points: list[MenuPoint]) -> list[dict]:
        batches.append(points)
        return [MENU for _ in points]

    conversion = convert_sessions([(meta, session)], menus_for)
    assert conversion.dropped_turns == {"s": []}
    rows = conversion.rows
    labels = [(row.decision, row.turn, row.level, row.label) for row in rows]
    assert labels == [
        (0, 1, "tool", "list"),
        (0, 1, "argument", "list-1"),
        (1, 1, "tool", "read"),
        (1, 1, "argument", "read-1"),
        (2, 2, "tool", "hand_over"),
    ]
    assert [len(batch) for batch in batches] == [1, 1, 1]
    assert batches[1][0].steps[0].command == "ls -la" and batches[1][0].steps[0].by_scout
    assert batches[0][0].cwd == "/app" and batches[0][0].task == "Fix the bug in /app/main.py."
    assert rows[2].state.endswith("Step 1 (by you, the scout):\n$ ls -la\ntotal 8\n-rw-r--r-- 1 root root 20 main.py")


def test_menu_count_mismatch_is_an_error():
    session = parse_terminus(conversation())
    meta = RowSource(source="test", stage=1, quality="approximate", task="t", session="s")
    with pytest.raises(ValueError, match="menus"):
        convert_sessions([(meta, session)], lambda points: [])


def screen(*lines: str) -> dict:
    return user("New Terminal Output:\n\n" + "\n".join(lines) + "\nroot@abc123:/app#\n")


def test_unmatched_information_turns_are_dropped_and_counted():
    conv = [
        user(FIRST_USER),
        assistant(["cat /app/other.py\n"]),
        screen("root@abc123:/app# cat /app/other.py", "x = 1"),
        assistant(["python3 /app/main.py\n"]),
        screen("root@abc123:/app# python3 /app/main.py", "1"),
    ]
    meta = RowSource(source="test", stage=1, quality="approximate", task="t", session="s")
    conversion = convert_sessions([(meta, parse_terminus(conv))], lambda points: [MENU for _ in points])
    assert [(row.turn, row.label) for row in conversion.rows] == [(2, "hand_over")]
    assert conversion.dropped_turns == {"s": [1]}


def test_menu_points_carry_evidence_events_and_the_shell_folder():
    conv = [
        user(FIRST_USER),
        assistant(["cd src && ls\n"]),
        screen("root@abc123:/app# cd src && ls", "a.py"),
        assistant(["cat a.py\n"]),
        {"role": "user", "content": "New Terminal Output:\n\nroot@abc123:/app/src# cat a.py\nprint(1)\nroot@abc123:/app/src#\n"},
    ]
    meta = RowSource(source="test", stage=1, quality="approximate", task="t", session="s")
    points: list[MenuPoint] = []

    def menus_for(batch: list[MenuPoint]) -> list[dict]:
        points.extend(batch)
        return [MENU for _ in batch]

    convert_sessions([(meta, parse_terminus(conv))], menus_for)
    listing = {"type": "listing", "folder": "/app/src", "entries": [{"name": "a.py"}], "showsHidden": False}
    text = {"type": "read", "path": "/app/src/a.py", "content": "print(1)\n"}
    # Only information was gathered, so everything the later commands showed was already true (back-fill).
    assert points[0].cwd == "/app" and points[0].events == [listing, text]
    assert points[1].cwd == "/app/src"
    # The listing at the point, completed with the whole text the later cat showed.
    assert points[1].events == [listing, text]


def test_build_menus_runs_the_menu_cli_and_back_fill_lets_it_offer_a_read():
    # ls reveals a.py (no size); two turns later cat a.py shows its 10 lines. At the point right after the ls, the
    # rebuilt menu offers Read the file /app/a.py, because the later text of the unchanged file is back-filled
    # (only information was gathered in between).
    ten_lines = [f"line {n}" for n in range(1, 11)]
    conv = [
        user(FIRST_USER),
        assistant(["ls\n"]),
        screen("root@abc123:/app# ls", "a.py  main.py"),
        assistant(["which gcc\n"]),
        screen("root@abc123:/app# which gcc", "/usr/bin/gcc"),
        assistant(["cat a.py\n"]),
        screen("root@abc123:/app# cat a.py", *ten_lines),
    ]
    meta = RowSource(source="test", stage=1, quality="approximate", task="t", session="s")
    seen: list[tuple[MenuPoint, dict]] = []

    def menus_for(batch: list[MenuPoint]) -> list[dict]:
        menus = build_menus(batch)
        seen.extend(zip(batch, menus))
        return menus

    conversion = convert_sessions([(meta, parse_terminus(conv))], menus_for)
    after_ls = next(menu for point, menu in seen if [s.command for s in point.steps] == ["ls"])
    reads = [option["description"] for option in after_ls["arguments_by_tool"].get("read", [])]
    assert "Read the file /app/a.py" in reads
    assert [(row.turn, row.level, row.label) for row in conversion.rows if row.turn == 3][0] == (3, "tool", "read")


def test_dataset_rows():
    row = json.loads(FIXTURE.read_text())
    meta, session = session_from_dataset_row(row, source="ukisai/Qwen3.8-27B-multi-turn-agent-sft", stage=1, quality="approximate")
    assert (meta.task, meta.session, meta.quality) == ("inferredbugs-0405", "inferredbugs-0405__me2VPkz", "approximate")
    rows = [
        {"trial_name": "a", "episode": "episode-2"},
        {"trial_name": "a", "episode": "episode-10"},
        {"trial_name": "b", "episode": "episode-0"},
    ]
    assert latest_episodes(rows) == [rows[1], rows[2]]


def test_typed_lines_whose_command_is_not_found_are_not_taken_as_continuing_output():
    conv = [
        user(FIRST_USER),
        assistant(["make\n"]),
        user("New Terminal Output:\n\nroot@abc123:/app# make\nok\nroot@abc123:/app#\n"),
        assistant(["grep -n '```x' a.md\n", "ls\n"]),
        user("New Terminal Output:\n\nroot@abc123:/app# grep -n '' a.md\n3:x\nroot@abc123:/app# ls\na.md\nroot@abc123:/app#\n"),
    ]
    session = parse_terminus(conv)
    assert session.turns[0].commands == [TurnCommand("make", "ok", False)]
    assert session.turns[1].commands == [TurnCommand("grep -n '```x' a.md", None, False), TurnCommand("ls", "a.md", False)]


FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def tool_call_reply(*calls: dict, text: str = "Analysis: look around.\nPlan: list the folder.") -> dict:
    blocks = "".join(f"<tool_call>\n{json.dumps(call)}\n</tool_call>\n" for call in calls)
    return {"role": "assistant", "content": f"<think>hm</think>\n{text}\n{blocks}"}


def bash_call(keystrokes: str) -> dict:
    return {"name": "bash_command", "arguments": {"keystrokes": keystrokes, "duration": 0.1}}


def test_tool_call_replies_are_one_command_per_block_in_order():
    conv = [
        user(FIRST_USER),
        tool_call_reply(bash_call("ls\n"), bash_call("cat main.py\n")),
        screen("root@abc123:/app# ls", "main.py", "root@abc123:/app# cat main.py", "print(1)"),
        tool_call_reply({"name": "mark_task_complete", "arguments": {}}, text="Done."),
    ]
    session = parse_terminus(conv)
    first = session.turns[0]
    assert first.commands == [TurnCommand("ls", "main.py", False), TurnCommand("cat main.py", "print(1)", False)]
    assert (first.thinking, first.analysis, first.plan) == ("hm", "look around.", "list the folder.")
    assert not first.task_complete
    assert session.turns[1].commands == [] and session.turns[1].task_complete
    assert session.end_reason == "complete"


def test_a_tool_call_naming_another_tool_is_an_error():
    conv = [user(FIRST_USER), tool_call_reply({"name": "write_file", "arguments": {}}), screen("root@abc123:/app# ls")]
    with pytest.raises(ValueError, match="write_file"):
        parse_terminus(conv)


def test_real_tool_call_session_nl2bash_1223():
    session = parse_terminus(fixture("terminus_tool_call_nl2bash_1223.json")["conversations"])
    assert [c.text for c in session.turns[0].commands] == [
        "ls -la /workspace /output 2>&1",
        "env | sort",
        "find /workspace -maxdepth 4 -type f -o -type l 2>/dev/null | head -200",
        "find / -iname '*firewall*' -o -iname '*simulator*' -o -iname '*token*' 2>/dev/null | grep -v -e '^/proc' -e '^/sys' | head -100",
    ]
    assert session.turns[0].commands[2].output == "/workspace/scripts/firewalls.json\n/workspace/scripts/simulate_firewalls.sh"
    assert [c.text for c in session.turns[1].commands][:3] == [
        "cat -n /workspace/scripts/simulate_firewalls.sh",
        "cat -n /workspace/scripts/firewalls.json",
        "ls -la /workspace/scripts",
    ]
    assert session.end_reason == "complete"


def test_harness_auto_corrected_missing_brace_is_parsed_as_the_harness_ran_it():
    session = parse_terminus(fixture("terminus_autocorrected_inferredbugs_1520.json")["conversations"])
    assert [c.text for c in session.turns[1].commands] == [
        "find / -name '*.cs' -not -path '/proc/*' -not -path '/sys/*' 2>/dev/null | head -50",
        "find / -name '.git' -not -path '/proc/*' -not -path '/sys/*' 2>/dev/null | head -20",
        "ls -la /root /home /tmp /workspace /repo 2>/dev/null",
    ]
    assert all(c.output is not None for c in session.turns[1].commands)


def warned(text: str) -> dict:
    return user(f"Previous response had warnings:\nWARNINGS: - {text}\n\nNew Terminal Output:\n\nroot@abc123:/app# ls\nmain.py\nroot@abc123:/app#\n")


def truncated_reply() -> dict:
    body = json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\n", "duration": 0.1}]})
    return {"role": "assistant", "content": body[:-1]}


def test_only_the_known_auto_corrections_are_accepted():
    brace = "AUTO-CORRECTED: Fixed incomplete JSON by adding missing closing brace - please fix this in future responses"
    session = parse_terminus([user(FIRST_USER), truncated_reply(), warned(brace)])
    assert session.turns[0].commands == [TurnCommand("ls", "main.py", False)]
    mixed = "AUTO-CORRECTED: Extracted JSON from mixed content - please fix this in future responses"
    session = parse_terminus([user(FIRST_USER), assistant(["ls\n"]), warned(mixed)])
    assert session.turns[0].commands == [TurnCommand("ls", "main.py", False)]
    with pytest.raises(ValueError, match="AUTO-CORRECTED: Rewrote the keystrokes"):
        parse_terminus([user(FIRST_USER), assistant(["ls\n"]), warned("AUTO-CORRECTED: Rewrote the keystrokes")])


def test_an_unreadable_final_reply_ends_the_session_before_that_turn():
    conv = [user(FIRST_USER), assistant(["ls\n"]), screen("root@abc123:/app# ls", "main.py"), {"role": "assistant", "content": "<think>unfinished"}]
    session = parse_terminus(conv)
    assert len(session.turns) == 1
    assert session.end_reason == "unparsed_final_reply"
    real = parse_terminus(fixture("terminus_unparsed_final_inferredbugs_0151.json")["conversations"])
    assert real.turns == [] and real.end_reason == "unparsed_final_reply"


def test_an_unreadable_reply_in_the_middle_that_the_harness_ran_is_an_error():
    conv = [user(FIRST_USER), {"role": "assistant", "content": "<tool_call>{bad json</tool_call>"}, screen("root@abc123:/app# ls", "main.py")]
    with pytest.raises(ValueError, match="turn 1"):
        parse_terminus(conv)


@pytest.mark.parametrize("keystrokes", ["\n", "  \n"])
def test_a_bare_enter_is_a_wait_not_a_command(keystrokes):
    conv = [user(FIRST_USER), assistant([keystrokes, "ls\n"]), screen("root@abc123:/app#", "root@abc123:/app# ls", "main.py")]
    session = parse_terminus(conv)
    assert session.turns[0].commands == [TurnCommand("ls", "main.py", False)]


@pytest.mark.parametrize(
    "command",
    [
        "vim --version",
        "man grep | head -20",
        "man --version",
        "which git vim && vim --version",
        "top -bn1 | head -5",
        "top -b -n 1",
        "man just > /tmp/man.txt 2>/tmp/man.err",
        "vim -es -c 'wq' /tmp/a.txt",
    ],
)
def test_version_help_and_piped_invocations_of_full_screen_programs_are_not_interactive(command):
    conv = [user(FIRST_USER), assistant([command + "\n"]), screen(f"root@abc123:/app# {command}", "x")]
    session = parse_terminus(conv)
    assert session.end_reason == "complete"
    assert [c.text for c in session.turns[0].commands] == [command]


def test_a_wrapped_command_echo_is_not_part_of_the_output():
    command = "grep -n '" + "x" * 380 + "' /app/main.py"
    echoed = "root@abc123:/app# " + command
    wrapped = [echoed[start : start + 160] for start in range(0, len(echoed), 160)]
    assert len(wrapped) == 3
    conv = [user(FIRST_USER), assistant([command + "\n", "ls\n"]), screen(*wrapped, "3:hit", "root@abc123:/app# ls", "main.py")]
    session = parse_terminus(conv)
    assert session.turns[0].commands == [TurnCommand(command, "3:hit", False), TurnCommand("ls", "main.py", False)]


def complete_reply(commands: list[str]) -> dict:
    body = {"analysis": "a", "plan": "p", "commands": [{"keystrokes": k, "duration": 0.1} for k in commands], "task_complete": True}
    return {"role": "assistant", "content": json.dumps(body)}


def test_a_reply_to_the_completion_question_without_commands_gets_no_row():
    question = "\n\nAre you sure you want to mark the task as complete? This will trigger your solution to be graded."
    conv = [
        user(FIRST_USER),
        assistant(["python3 main.py\n"]),
        screen("root@abc123:/app# python3 main.py", "1"),
        complete_reply([]),
        user("Current terminal state:\nNew Terminal Output:\n\nroot@abc123:/app#\n" + question),
        complete_reply([]),
    ]
    session = parse_terminus(conv)
    assert [turn.confirms_completion for turn in session.turns] == [False, False, True]
    meta = RowSource(source="test", stage=1, quality="approximate", task="t", session="s")
    conversion = convert_sessions([(meta, session)], lambda points: [MENU for _ in points])
    assert [(row.turn, row.label) for row in conversion.rows] == [(1, "hand_over"), (2, "hand_over")]
    assert conversion.sessions["s"].end_reason == "complete" and conversion.sessions["s"].rows == 2


def test_a_folder_that_cannot_be_known_ends_the_session_with_folder_unknown():
    conv = [
        user(FIRST_USER),
        assistant(["ls\n"]),
        screen("root@abc123:/app# ls", "main.py"),
        assistant(['cd "$WORK" && cat main.py\n']),
        screen('root@abc123:/app# cd "$WORK" && cat main.py', "x"),
        assistant(["ls\n"]),
        screen("root@abc123:/app# ls", "main.py"),
    ]
    session = parse_terminus(conv)
    assert len(session.turns) == 1 and session.end_reason == "folder_unknown"


def test_a_turn_without_commands_gives_its_decision_to_the_next_labelled_turn():
    # A wait or a premature "task complete" runs nothing, so the next turn's point has the same state and menu:
    # one row there, not two identical ones (inferredbugs-2022).
    conv = [
        user(FIRST_USER),
        assistant(["python3 main.py\n"]),
        screen("root@abc123:/app# python3 main.py", "1"),
        assistant([""]),
        screen(),
        complete_reply([]),
        user("Current terminal state:\nNew Terminal Output:\n\nroot@abc123:/app#\n\n\nAre you sure you want to mark the task as complete?"),
        assistant(["ls\n", "python3 main.py\n"]),
        screen("root@abc123:/app# ls", "main.py", "root@abc123:/app# python3 main.py", "1"),
        complete_reply([]),
    ]
    meta = RowSource(source="test", stage=1, quality="approximate", task="t", session="s")
    conversion = convert_sessions([(meta, parse_terminus(conv))], lambda points: [MENU for _ in points])
    assert [(row.turn, row.level, row.label) for row in conversion.rows] == [
        (1, "tool", "hand_over"),
        (4, "tool", "list"),
        (4, "argument", "list-1"),
        (4, "tool", "hand_over"),
        (5, "tool", "hand_over"),
    ]
