"""Convert Terminus-2 sessions of the coding model (Qwen3.8-27B) into Jeff training rows (stages 1 and 2).

Terminus-2 is an agent harness that drives a tmux terminal. Its conversation is a list of {role, content} messages:

- The first user message holds the harness instructions, then "Task Description:" and the task, then
  "Current terminal state:" and the initial screen, whose prompt line (`root@host:/app#`) gives the working folder.
- Each assistant message may start with `<think>...</think>` (the model's thinking), then a JSON object with
  "analysis", "plan", "commands" (each {"keystrokes", "duration"}) and an optional "task_complete".
- The user message after it holds the terminal's new output ("New Terminal Output:"), sometimes after warnings and
  sometimes followed by a confirmation question; or, when the harness could not read the reply, "Previous response
  had parsing errors" and no output at all (nothing was run).

Each keystroke string that ends in a newline is one command (its text without that final newline; a string with
several lines, such as a heredoc, stays one command). An empty keystroke string only waits. Any other keystrokes -
tmux key names such as C-c or Escape, text without a final newline, or a command that opens an editor, pager or
interactive interpreter - is input to a running program; the conversion of the session ends before that turn
(`TerminusSession.ended` says why).

A command's output is read from the following screen (best effort): the lines after the command's echoed prompt line
up to the next command's echo, without the shell's echo of the command's own further lines (`> ...` continuation
lines and prompt lines). Lines at the top of a screen before any typed command are output still coming from the
previous command (it outlived its wait). A command whose echo is not on the screen has no recorded output (None).

The dataset ukisai/Qwen3.8-27B-multi-turn-agent-sft has one row per episode, each holding the whole conversation up
to that episode; `latest_episodes` keeps the last (longest) row of each trial.

Menus come from the TypeScript list code (lists.ts) run over a virtual file system rebuilt from the transcript -
the menu CLI of Task 4. `convert_sessions` asks for them in batches, one menu per decision point, through
`menus_for(points)`; `build_menus` is that interface.
"""

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from imitation.events import Shell, backfill, command_events
from imitation.labels import LabelTurn, SessionLabeler, TurnCommand
from imitation.rows import Menu, Row, RowSource, ShellStep, render_state, rows_for_decision
from imitation.splitter import first_word, split_command

PROMPT = re.compile(r"^(?P<user>[\w.-]+)@[\w.-]+:(?P<cwd>[^\n#$]*)[#$](?: |$)")
REPO_ROOT = Path(__file__).resolve().parents[3]
MENU_CLI = REPO_ROOT / "scripts" / "jeff-first-menus.ts"
TMUX_KEY = re.compile(
    r"^(?:C-|M-|S-)\S+$|^(?:Escape|Enter|Tab|BSpace|BTab|DC|IC|Up|Down|Left|Right|Home|End|PageUp|PageDown|PgUp|PgDn|NPage|PPage|Space|F\d+)$"
)
INTERACTIVE_PROGRAMS = {"vim", "vi", "nvim", "nano", "emacs", "pico", "less", "more", "man", "top", "htop", "watch", "ssh", "ftp", "telnet", "tmux", "screen"}  # fmt: skip
REPLS = re.compile(r"^(?:python[\d.]*|node|irb|ghci|sqlite3|mysql|psql|bash|sh|zsh|R|julia|lua|php|redis-cli|mongo|mongosh)$")
SCREEN_MARKERS = ("New Terminal Output:\n", "Current Terminal Screen:\n")
CONFIRMATION = "\n\nAre you sure you want to mark the task as complete?"
PARSE_ERROR = "Previous response had parsing errors"
WRAP_WIDTH = 70


@dataclass(frozen=True)
class TerminusTurn:
    """One coding-model turn: its thinking, analysis and plan (kept as the coding model's text, never shown to
    Jeff), its commands with their outputs, and whether the harness rejected the reply (then nothing ran)."""

    thinking: str
    analysis: str
    plan: str
    commands: list[TurnCommand]
    task_complete: bool
    parse_error: bool
    # The folder on each command's echoed prompt line (as the prompt shows it, "~" for home), None when not seen.
    prompt_folders: list[str | None]


@dataclass(frozen=True)
class TerminusSession:
    task: str
    cwd: str
    home: str
    turns: list[TerminusTurn]
    ended: str | None


@dataclass(frozen=True)
class MenuPoint:
    """What the menu builder needs at one decision point: the task text, the shell's folder, the steps so far
    (oldest first; commands with their outputs) and the evidence events about files and programs (events.py)."""

    task: str
    cwd: str
    steps: list[ShellStep]
    events: list[dict]


def build_menus(points: list[MenuPoint]) -> list[Menu]:
    """The scout's full option lists for each point, in order, from the menu CLI (scripts/jeff-first-menus.ts: the
    same list code as live runs, over facts rebuilt from each point's events; bash only, run approval "all")."""
    lines = [
        json.dumps(
            {
                "id": str(number),
                "cwd": point.cwd,
                "task": point.task,
                "steps": [{"command": s.command, "output": s.output, "byScout": s.by_scout} for s in point.steps],
                "events": point.events,
                "activeTools": ["bash"],
                "runApproval": "all",
            }
        )
        for number, point in enumerate(points)
    ]
    finished = subprocess.run(
        ["node", str(MENU_CLI)], input="\n".join(lines) + "\n", capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    if finished.returncode != 0:
        raise RuntimeError(f"the menu CLI {MENU_CLI} failed (exit {finished.returncode}): {finished.stderr.strip()[-2000:]}")
    built = [json.loads(line) for line in finished.stdout.splitlines() if line.strip()]
    if [menu["id"] for menu in built] != [str(number) for number in range(len(points))]:
        raise RuntimeError(f"the menu CLI returned menus for ids {[m['id'] for m in built][:10]}..., not one per point in order")
    return [{"tools": menu["tools"], "arguments_by_tool": menu["argumentsByTool"]} for menu in built]


def _task_and_cwd(first: str) -> tuple[str, str, str]:
    start = first.find("Task Description:\n")
    if start < 0:
        raise ValueError("the first user message has no 'Task Description:' section")
    end = first.find("\nCurrent terminal state:", start)
    if end < 0:
        raise ValueError("the first user message has no 'Current terminal state:' section after the task")
    task = first[start + len("Task Description:\n") : end].strip()
    for line in first[end:].split("\n"):
        prompt = PROMPT.match(line)
        if prompt:
            user = prompt.group("user")
            return task, prompt.group("cwd"), "/root" if user == "root" else f"/home/{user}"
    raise ValueError("the first user message's terminal screen shows no shell prompt, so the working folder is unknown")


def _reply(content: str, turn: int) -> tuple[str, dict]:
    thinking = ""
    rest = content
    think = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think:
        thinking = think.group(1).strip()
        rest = content[think.end() :]
    decoder = json.JSONDecoder()
    for start in [index for index, char in enumerate(rest) if char == "{"]:
        try:
            value, _ = decoder.raw_decode(rest, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "commands" in value:
            return thinking, value
    raise ValueError(f"turn {turn}: the reply has no JSON object with 'commands', yet the harness ran it")


def _interactive(keystrokes: str) -> bool:
    if keystrokes == "":
        return False
    if not keystrokes.endswith("\n") or TMUX_KEY.match(keystrokes.strip()):
        return True
    for part in split_command(keystrokes[:-1]):
        word, text = first_word(part.head)
        base = word.rsplit("/", 1)[-1]
        if base in INTERACTIVE_PROGRAMS:
            return True
        if any(first_word(stage)[0] in ("less", "more") for stage in part.filters):
            return True
        if REPLS.match(base) and text.strip() == word and part.heredoc is None:
            return True
    # A line like ":wq" or "y" is input to a program that is still running, not a shell command.
    return bool(re.fullmatch(r":\w*!?|[yYnNq]", keystrokes[:-1].strip()))


def _screen(message: str) -> str:
    starts = [message.find(marker) + len(marker) for marker in SCREEN_MARKERS if marker in message]
    if not starts:
        raise ValueError(f"the terminal message has no screen: {message[:120]!r}")
    # The harness puts a blank line after the marker in most messages but not all; it is not output.
    screen = message[min(starts) :].lstrip("\n")
    confirmation = screen.find(CONFIRMATION)
    return screen if confirmation < 0 else screen[:confirmation]


def _echo_of(line: str, first_line: str) -> bool:
    prompt = PROMPT.match(line)
    if not prompt:
        return False
    typed = line[prompt.end() :].rstrip()
    if not typed:
        return False
    return typed == first_line.rstrip() or (len(line) >= WRAP_WIDTH and first_line.startswith(typed))


def _typed(line: str) -> str | None:
    """The text typed after a shell prompt on this screen line, or None when the line is not a prompt line."""
    prompt = PROMPT.match(line)
    return None if prompt is None else line[prompt.end() :].rstrip()


def _clean(lines: list[str], own_lines: set[str], multi_line: bool) -> str:
    """Output lines without the shell's echo of the command's own further lines: prompt lines showing one of them,
    and, for a command of several lines, the "> " continuation lines (heredoc and open-quote input)."""
    kept: list[str] = []
    for line in lines:
        if _typed(line) in own_lines:
            continue
        if multi_line and (line.startswith("> ") or line.rstrip() == ">"):
            continue
        kept.append(line)
    while kept and (not kept[-1].strip() or _typed(kept[-1]) == ""):
        kept.pop()
    return "\n".join(kept)


def _outputs(screen: str, commands: list[str]) -> tuple[str, list[str | None], list[str | None]]:
    """Text before the first echo (still coming from an earlier command), each command's output, and the folder on
    each command's echoed prompt line."""
    lines = screen.split("\n")
    echoes: list[int | None] = []
    position = 0
    for command in commands:
        first_line = command.split("\n", 1)[0]
        found = next((index for index in range(position, len(lines)) if _echo_of(lines[index], first_line)), None)
        echoes.append(found)
        if found is not None:
            position = found + 1
    found_echoes = sorted(index for index in echoes if index is not None)
    # Output still coming from an earlier command ends at the first command typed on this screen, matched or not.
    leading_end = next((index for index, line in enumerate(lines) if _typed(line)), len(lines))
    leading = _clean(lines[:leading_end], set(), multi_line=False)
    outputs: list[str | None] = []
    for command, echo in zip(commands, echoes):
        if echo is None:
            outputs.append(None)
            continue
        end = next((index for index in found_echoes if index > echo), len(lines))
        own = {line.rstrip() for line in command.split("\n")}
        outputs.append(_clean(lines[echo + 1 : end], own, multi_line="\n" in command))
    folders = [None if echo is None else PROMPT.match(lines[echo]).group("cwd") for echo in echoes]
    return leading, outputs, folders


def parse_terminus(conversation: list[dict]) -> TerminusSession:
    """The task, working folder and turns of one Terminus-2 conversation; see the module docstring."""
    if not conversation or conversation[0]["role"] != "user":
        raise ValueError("a Terminus-2 conversation starts with the user message holding the task")
    task, cwd, home = _task_and_cwd(conversation[0]["content"])
    turns: list[TerminusTurn] = []
    ended: str | None = None
    index = 1
    while index < len(conversation):
        message = conversation[index]
        if message["role"] != "assistant":
            raise ValueError(f"message {index} should be the coding model's reply, not a {message['role']} message")
        number = len(turns) + 1
        screen_message = conversation[index + 1]["content"] if index + 1 < len(conversation) else None
        index += 2
        if screen_message is not None and screen_message.startswith(PARSE_ERROR):
            turns.append(TerminusTurn("", "", "", [], False, parse_error=True, prompt_folders=[]))
            continue
        thinking, reply = _reply(message["content"], number)
        keystrokes = [command["keystrokes"] for command in reply["commands"]]
        interactive = next((k for k in keystrokes if _interactive(k)), None)
        if interactive is not None:
            ended = f"turn {number} sends interactive keystrokes {interactive!r}"
            break
        texts = [k[:-1] for k in keystrokes if k]
        outputs: list[str | None] = [None] * len(texts)
        folders: list[str | None] = [None] * len(texts)
        if screen_message is not None:
            leading, outputs, folders = _outputs(_screen(screen_message), texts)
            previous = next((turn for turn in reversed(turns) if turn.commands), None)
            if leading and previous is not None and previous.commands[-1].output is not None:
                last = previous.commands[-1]
                joined = f"{last.output}\n{leading}" if last.output else leading
                previous.commands[-1] = TurnCommand(last.text, joined, last.is_error)
        turns.append(
            TerminusTurn(
                thinking=thinking,
                analysis=str(reply.get("analysis", "")),
                plan=str(reply.get("plan", "")),
                commands=[TurnCommand(text, output, False) for text, output in zip(texts, outputs)],
                task_complete=bool(reply.get("task_complete", False)),
                parse_error=False,
                prompt_folders=folders,
            )
        )
    return TerminusSession(task=task, cwd=cwd, home=home, turns=turns, ended=ended)


def session_from_dataset_row(row: dict, source: str, stage: int, quality: str) -> tuple[RowSource, TerminusSession]:
    """One dataset row (columns conversations, task, trial_name, ...) as the rows' shared fields and its session."""
    conversation = row["conversations"]
    if not isinstance(conversation, list):
        raise ValueError(f"the row's conversations must be a list of messages, not {type(conversation).__name__}")
    meta = RowSource(source=source, stage=stage, quality=quality, task=row["task"], session=row["trial_name"])
    return meta, parse_terminus(conversation)


def latest_episodes(rows: list[dict]) -> list[dict]:
    """The last episode's row of each trial (its conversation holds every earlier episode), in first-seen order."""
    latest: dict[str, dict] = {}
    for row in rows:
        number = int(row["episode"].removeprefix("episode-"))
        kept = latest.get(row["trial_name"])
        if kept is None or number > int(kept["episode"].removeprefix("episode-")):
            latest[row["trial_name"]] = row
    return list(latest.values())


@dataclass(frozen=True)
class Conversion:
    """The rows of a batch of sessions, and per session the turns whose point got no row because the coding
    model's next command only gathered information that no option on the rebuilt menu matched (see
    SessionLabeler.drop_unmatched_information)."""

    rows: list[Row]
    dropped_turns: dict[str, list[int]]


@dataclass(frozen=True)
class _Evidence:
    """Per real command of one session, keyed (turn index, command index): its events and the shell's folder after
    it (None when unknown), in the session's real order."""

    order: list[tuple[int, int]]
    events: dict[tuple[int, int], list[dict]]
    folder_after: dict[tuple[int, int], str | None]
    prompt_folder: dict[tuple[int, int], str | None]


def _evidence(session: TerminusSession) -> _Evidence:
    order: list[tuple[int, int]] = []
    events: dict[tuple[int, int], list[dict]] = {}
    folder_after: dict[tuple[int, int], str | None] = {}
    prompt_folder: dict[tuple[int, int], str | None] = {}
    shell = Shell(session.cwd, session.home)
    for turn_index, turn in enumerate(session.turns):
        for command_index, (command, shown) in enumerate(zip(turn.commands, turn.prompt_folders)):
            key = (turn_index, command_index)
            if shown is not None:
                shell = Shell(session.home + shown[1:] if shown.startswith("~") else shown, session.home)
            order.append(key)
            prompt_folder[key] = shell.cwd
            events[key], shell = command_events(command.text, command.output, shell)
            folder_after[key] = shell.cwd
    return _Evidence(order, events, folder_after, prompt_folder)


def _point(session: TerminusSession, evidence: _Evidence, keys: list[tuple[int, int]], history: list[ShellStep]) -> MenuPoint:
    """The menu builder's input at a point whose history holds the real commands `keys`, in that order."""
    held = set(keys)
    later = [key for key in evidence.order if key not in held]
    point_events = [event for key in keys for event in evidence.events[key]]
    events = backfill(point_events, [evidence.events[key] for key in later])
    folder = evidence.folder_after[keys[-1]] if keys else session.cwd
    if folder is None:
        # The folder became unknown (cd -, popd); the next command's prompt line shows where the shell is.
        folder = evidence.prompt_folder[later[0]] if later else None
    if folder is None:
        raise ValueError("the shell's folder at this point is unknown: no prompt line after it shows it")
    return MenuPoint(session.task, folder, history, events)


def convert_sessions(
    sessions: list[tuple[RowSource, TerminusSession]],
    menus_for: Callable[[list[MenuPoint]], list[Menu]] = build_menus,
) -> Conversion:
    """Label every decision point of every session and return the rows. Menus are asked for in rounds: each round
    sends one point per session that still has one, so the menu builder runs once per round, not once per point.
    Sessions converted here are approximate (menus rebuilt from the transcript), so points whose next command only
    gathers information no option matches are dropped, not labelled "hand over"."""
    labelers = [
        SessionLabeler(
            [LabelTurn(turn.commands, not turn.parse_error) for turn in session.turns],
            session.cwd,
            follow_stints=True,
            drop_unmatched_information=True,
        )
        for _, session in sessions
    ]
    evidence = [_evidence(session) for _, session in sessions]
    while True:
        pending = [(number, history) for number, labeler in enumerate(labelers) if (history := labeler.next_point()) is not None]
        if not pending:
            break
        points = [
            _point(sessions[number][1], evidence[number], labelers[number].next_point_keys(), history)
            for number, history in pending
        ]
        menus = menus_for(points)
        if len(menus) != len(points):
            raise ValueError(f"the menu builder returned {len(menus)} menus for {len(points)} points")
        for (number, _), menu in zip(pending, menus):
            labelers[number].give(menu)
    rows: list[Row] = []
    for (meta, session), labeler in zip(sessions, labelers):
        for decision, point in enumerate(labeler.decisions):
            state = render_state(session.task, point.history)
            rows.extend(rows_for_decision(meta, decision, point.turn, state, point.menu, point.choice))
    return Conversion(rows, {meta.session: labeler.dropped for (meta, _), labeler in zip(sessions, labelers)})
