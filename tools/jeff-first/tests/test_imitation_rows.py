import json
import subprocess
from pathlib import Path

import pytest

from imitation.events import Shell, part_folders
from imitation.labels import LabelTurn, SessionLabeler, TurnCommand
from imitation.rows import (
    Choice,
    Row,
    ShellStep,
    decision_rows,
    render_state,
    terminal_command,
    terminal_view,
    tool_page,
)


JEFF_FIRST_TS = Path(__file__).resolve().parents[3] / "packages" / "coding-agent" / "src" / "core" / "jeff-first"


def option(kind: str, number: int, description: str) -> dict:
    return {"id": f"{kind}-{number}", "description": description, "toolCall": {"name": "bash", "arguments": {"command": "x"}}}


def menu_with(read_count: int) -> dict:
    return {
        "tools": [
            {"id": "read", "description": "Read a file"},
            {"id": "list", "description": "List a folder"},
            {"id": "hand_over", "description": "Hand over to the coding model"},
        ],
        "arguments_by_tool": {
            "read": [option("read", n, f"Read the file /app/f{n}.py") for n in range(1, read_count + 1)],
            "list": [option("list", 1, "List the folder /app")],
        },
    }


def test_terminal_view_keeps_the_last_40_lines_cut_to_200_characters():
    output = "\n".join(f"line {n}" for n in range(1, 101))
    view = terminal_view(output)
    lines = view.split("\n")
    assert lines[0] == "[60 earlier lines not shown]"
    assert lines[1] == "line 61"
    assert lines[-1] == "line 100"
    assert len(lines) == 41
    assert terminal_view("x" * 500) == "x" * 200


def test_terminal_view_of_a_short_output_has_no_note():
    # Like state.ts, a final newline is a last (empty) line and is kept.
    assert terminal_view("a\nb\n") == "a\nb\n"
    assert terminal_view("a\nb") == "a\nb"
    assert terminal_view("") == ""


def test_terminal_command_keeps_the_first_40_lines():
    command = "cat > x <<'EOF'\n" + "\n".join(f"row {n}" for n in range(60)) + "\nEOF"
    view = terminal_command(command)
    lines = view.split("\n")
    assert lines[0] == "cat > x <<'EOF'" and lines[39] == "row 38"
    assert lines[40] == "[22 more lines of this command not shown]"


def test_lengths_and_cuts_count_utf16_units_like_javascript():
    line = "a" * 199 + "\U0001F600" + "b"  # the emoji is two UTF-16 units: the cut at 200 splits it
    cut = terminal_view(line)
    assert cut == "a" * 199 + "\ud83d"
    row_text = Row(
        source="s", stage=3, quality="exact", task="t", session="x", decision=0, turn=1, level="tool", page=1,
        state=cut, options=[{"id": "hand_over", "description": "Hand over"}], label="hand_over", tool_description=None,
    ).to_json()
    assert "\\ud83d" in row_text


def test_render_state_shows_steps_as_a_terminal():
    steps = [
        ShellStep(command="ls -la /app", output="total 0\nmain.py", is_error=False, by_scout=False),
        ShellStep(command="cat /app/main.py", output="print(1)", is_error=False, by_scout=True),
    ]
    assert render_state("Fix the bug.", steps) == (
        "Task:\nFix the bug.\n\n"
        "Steps so far, oldest first:\n\n"
        "Step 1 (by the coding model):\n$ ls -la /app\ntotal 0\nmain.py\n\n"
        "Step 2 (by you, the scout):\n$ cat /app/main.py\nprint(1)"
    )


def test_render_state_without_steps_and_with_missing_output_and_errors():
    assert render_state("T", []) == "Task:\nT\n\nNo steps have been taken yet."
    text = render_state("T", [ShellStep(command="make", output=None, is_error=True, by_scout=False)])
    assert "Step 1 (by the coding model; the command reported an error):\n$ make\n(no output was recorded)" in text


def test_render_state_keeps_the_most_recent_steps_within_the_budget():
    big = "\n".join("y" * 199 for _ in range(39))  # 7,799 characters: one step fits the 8,000 budget, two do not
    steps = [ShellStep(command=f"cat f{n}", output=big, is_error=False, by_scout=False) for n in range(3)]
    text = render_state("T", steps)
    assert "(2 earlier steps are not shown)" in text
    assert "$ cat f2" in text and "$ cat f1" not in text


def test_tool_page_matches_pages_ts():
    menu = menu_with(12)
    page1 = tool_page(menu, 1)
    assert [o["id"] for o in page1] == ["read", "list", "hand_over", "show_more"]
    assert page1[0]["description"].startswith("Read a file: Read the file /app/f1.py; Read the file /app/f2.py;")
    assert page1[0]["description"].endswith("(12 options)")
    assert page1[1]["description"] == "List a folder: List the folder /app (1 option)"
    page2 = tool_page(menu, 2)
    assert [o["id"] for o in page2] == ["read", "hand_over"]


def test_hand_over_is_one_row_on_tool_page_one():
    levels = decision_rows(menu_with(2), Choice.hand_over())
    assert [(level.level, level.page, level.label) for level in levels] == [("tool", 1, "hand_over")]


def test_a_choice_on_page_one_gives_a_tool_row_and_an_argument_row():
    levels = decision_rows(menu_with(2), Choice.step("read", "read-2"))
    assert [(level.level, level.page, level.label) for level in levels] == [("tool", 1, "read"), ("argument", 1, "read-2")]
    assert [o["id"] for o in levels[1].options] == ["read-1", "read-2", "none_of_these"]


def test_a_choice_on_page_two_is_preceded_by_show_more():
    levels = decision_rows(menu_with(12), Choice.step("read", "read-11"))
    assert [(level.level, level.page, level.label) for level in levels] == [
        ("tool", 1, "show_more"),
        ("tool", 2, "read"),
        ("argument", 2, "read-11"),
    ]
    assert [o["id"] for o in levels[2].options] == ["read-11", "read-12", "none_of_these"]


def test_a_choice_not_on_the_menu_is_an_error():
    with pytest.raises(ValueError, match="read-9"):
        decision_rows(menu_with(2), Choice.step("read", "read-9"))


def test_row_serialises_to_json():
    row = Row(
        source="ukisai/Qwen3.8-27B-multi-turn-agent-sft",
        stage=1,
        quality="approximate",
        task="t",
        session="s",
        decision=0,
        turn=1,
        level="tool",
        page=1,
        state="Task:\nt",
        options=[{"id": "hand_over", "description": "Hand over"}],
        label="hand_over",
        tool_description=None,
    )
    assert json.loads(row.to_json())["label"] == "hand_over"
    with pytest.raises(ValueError, match="quality"):
        Row(**{**row.__dict__, "quality": "good"})


def test_replayed_own_sessions_have_their_own_quality():
    # Stage 3 replay: logged menus before each turn, menus rebuilt in the task's image inside a turn.
    row = Row(
        source="own", stage=3, quality="exact-replayed", task="t", session="s", decision=0, turn=1, level="tool", page=1,
        state="Task:\nt", options=[{"id": "hand_over", "description": "Hand over"}], label="hand_over", tool_description=None,
    )  # fmt: skip
    assert row.quality == "exact-replayed"


def test_replayed_public_sessions_have_their_own_quality():
    # Stage 1 replay: public Terminus-2 sessions replayed in a rebuilt task environment, every menu built there.
    row = Row(
        source="ukisai", stage=1, quality="replayed", task="t", session="s", decision=0, turn=1, level="tool", page=1,
        state="Task:\nt", options=[{"id": "hand_over", "description": "Hand over"}], label="hand_over", tool_description=None,
    )  # fmt: skip
    assert row.quality == "replayed"


def test_an_argument_level_carries_the_chosen_tools_own_description():
    # teacher-prompt.ts asks "You have decided that the next step is: <tool.description>." with the tool's own short
    # description from the menu, not the tool page's longer text with the options written out.
    levels = decision_rows(menu_with(12), Choice.step("read", "read-11"))
    assert [level.tool_description for level in levels] == [None, None, "Read a file"]
    assert decision_rows(menu_with(2), Choice.hand_over())[0].tool_description is None


def test_a_row_needs_a_tool_description_exactly_when_it_is_an_argument_row():
    base = dict(source="s", stage=3, quality="exact", task="t", session="x", decision=0, turn=1, page=1, state="Task:\nt")
    argument = dict(base, level="argument", options=[{"id": "read-1", "description": "Read the file /a"}], label="read-1")
    tool = dict(base, level="tool", options=[{"id": "hand_over", "description": "Hand over"}], label="hand_over")
    assert json.loads(Row(**argument, tool_description="Read a file").to_json())["tool_description"] == "Read a file"
    with pytest.raises(ValueError, match="tool description"):
        Row(**argument, tool_description=None)
    with pytest.raises(ValueError, match="tool description"):
        Row(**argument, tool_description="")
    with pytest.raises(ValueError, match="tool description"):
        Row(**tool, tool_description="Read a file")


STATE_SCRIPT = """
import { readFileSync } from "node:fs";
import { trimState } from "%(ts)s/state.ts";
import { renderState } from "%(ts)s/teacher-prompt.ts";
const { task, steps } = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(renderState(trimState(task, steps))));
"""


def test_a_stint_step_renders_as_state_ts_renders_the_scout_options_own_call(tmp_path):
    # Review of waves 2-3, finding 4: a step the scout is credited with shows the option's own bash call, as the
    # live scout's step would, with the coding model's real output.
    peek_call = {
        "name": "bash",
        "arguments": {"command": "wc -l '/app/big.txt'\necho '--- first 20 lines ---'\nhead -n 20 '/app/big.txt' | cut -c1-300", "timeout": 60},
    }
    read_call = {"name": "bash", "arguments": {"command": "cat '/app/main.py'", "timeout": 60}}
    menu = {
        "tools": [{"id": "read", "description": "Read"}, {"id": "peek", "description": "Peek"}, {"id": "hand_over", "description": "Hand over"}],
        "arguments_by_tool": {
            "read": [{"id": "read-1", "description": "Read the file /app/main.py", "toolCall": read_call}],
            "peek": [{"id": "peek-1", "description": "Look at the data in /app/big.txt", "toolCall": peek_call}],
        },
    }
    commands = [TurnCommand("cat -n main.py", "1 print(1)", False), TurnCommand("head -n 20 big.txt", "a\nb", False), TurnCommand("make", "ok", False)]
    turn = LabelTurn(commands, labelled=True, folders=[part_folders(c.text, Shell("/app", "/root"))[0] for c in commands])
    labeler = SessionLabeler([turn], follow_stints=True, drop_unmatched_information=False)
    while labeler.next_point() is not None:
        labeler.give(menu)
    assert [d.choice for d in labeler.decisions] == [Choice.step("read", "read-1"), Choice.step("peek", "peek-1"), Choice.hand_over()]
    history = labeler.decisions[2].history
    steps = [
        {"call": {"type": "toolCall", "id": "s1", **read_call}, "output": "1 print(1)", "isError": False, "byScout": True},
        {"call": {"type": "toolCall", "id": "s2", **peek_call}, "output": "a\nb", "isError": False, "byScout": True},
    ]
    script = tmp_path / "state.mjs"
    script.write_text(STATE_SCRIPT % {"ts": JEFF_FIRST_TS.as_posix()})
    done = subprocess.run(["node", str(script)], input=json.dumps({"task": "Fix it.", "steps": steps}), capture_output=True, text=True, check=True)
    assert render_state("Fix it.", history) == json.loads(done.stdout)
    assert "$ cat '/app/main.py'\n1 print(1)" in json.loads(done.stdout)
