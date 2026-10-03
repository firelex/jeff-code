"""Convert Terminus-2 sessions of the coding model (Qwen3.8-27B) into Jeff training rows (stages 1 and 2).

Terminus-2 is an agent harness that drives a tmux terminal. Its conversation is a list of {role, content} messages:

- The first user message holds the harness instructions, then "Task Description:" and the task, then
  "Current terminal state:" and the initial screen, whose prompt line (`root@host:/app#`) gives the working folder.
- Each assistant message may start with `<think>...</think>` (the model's thinking), then either a JSON object with
  "analysis", "plan", "commands" (each {"keystrokes", "duration"}) and an optional "task_complete", or "Analysis:"
  and "Plan:" text followed by one `<tool_call>{"name": "bash_command", "arguments": {"keystrokes", "duration"}}
  </tool_call>` block per command (in order; a `mark_task_complete` call marks the task complete; any other tool
  name is an error).
- The user message after it holds the terminal's new output ("New Terminal Output:"), sometimes after warnings and
  sometimes followed by a confirmation question ("Are you sure you want to mark the task as complete?"); or, when
  the harness could not read the reply, "Previous response had parsing errors" and no output at all (nothing was
  run). Two warnings say the harness repaired the reply before running it: "AUTO-CORRECTED: Fixed incomplete JSON
  by adding missing closing brace" (the reply is read with "}" appended, as the harness did) and "AUTO-CORRECTED:
  Extracted JSON from mixed content" (read as is); any other AUTO-CORRECTED warning is an error.

A reply that cannot be read in either format ends the session before its turn when it is the transcript's last
message ("unparsed_final_reply": nothing ran after it); in the middle of a session, where the harness ran something,
it is an error. A reply with no commands to the confirmation question repeats the previous decision exactly and
gets no row (`TerminusTurn.confirms_completion`).

Each keystroke string that ends in a newline is one command (its text without that final newline; a string with
several lines, such as a heredoc, stays one command). A keystroke string that is empty, or only presses Enter, only
waits. Any other keystrokes - tmux key names such as C-c or Escape, text without a final newline, or a command that
opens an editor, pager or interactive interpreter (not when asked for --version or --help, piped into a filter or
written to a file) - is input to a running program; the conversion of the session ends before that turn
("interactive"). A command that gathers information in a folder that cannot be known (after `cd "$DIR"`, `cd -`)
also ends the session before its turn ("folder_unknown"). `TerminusSession.end_reason` says which.

A command's output is read from the following screen (best effort): the lines after the command's echoed prompt line
up to the next command's echo, without the shell's echo of the command's own further lines (the rest of a first line
too long for the 160-column terminal, which wraps it over several screen lines; `> ...` continuation lines and
prompt lines). Lines at the top of a screen before any typed command are output still coming from the previous
command (it outlived its wait). A command whose echo is not on the screen has no recorded output (None).

The dataset ukisai/Qwen3.8-27B-multi-turn-agent-sft has one row per episode, each holding the whole conversation up
to that episode; `latest_episodes` keeps the last (longest) row of each trial.

Menus come from the TypeScript list code (lists.ts) run over a virtual file system rebuilt from the transcript -
the menu CLI of Task 4. `convert_sessions` asks for them in batches, one menu per decision point, through
`menus_for(points)`; `build_menus` is that interface.
"""

import json
import re
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from imitation.events import Shell, backfill, command_events, part_folders
from imitation.labels import INFORMATION_KINDS, Intent, LabelTurn, SessionLabeler, TurnCommand, command_acts, part_intent
from imitation.rows import Menu, Row, RowSource, ShellStep, render_state, rows_for_decision
from imitation.splitter import first_word, has_file_write, split_command

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
    # True for a reply with no commands to the harness's "Are you sure you want to mark the task as complete?":
    # it repeats the previous decision exactly, so it gets no row.
    confirms_completion: bool = False
    # For each command, the folder each of its parts (split_command order) runs in; filled by parse_terminus.
    part_folders: list[list[str | None]] = field(default_factory=list)


END_REASONS = ("complete", "interactive", "unparsed_final_reply", "folder_unknown")


@dataclass(frozen=True)
class TerminusSession:
    """`end_reason` says why the conversion of the session stops where it does (one of END_REASONS): "complete"
    (the whole transcript), "interactive" (a turn sends keystrokes to a running program), "unparsed_final_reply"
    (the transcript's last reply cannot be read and nothing ran after it) or "folder_unknown" (a command's folder
    cannot be known). `end_detail` names the turn and what was found there (None when complete)."""

    task: str
    cwd: str
    home: str
    turns: list[TerminusTurn]
    end_reason: str
    end_detail: str | None


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


TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
WARNINGS = "Previous response had warnings:"
AUTO_CORRECTED = re.compile(r"AUTO-CORRECTED: ([^\n]*?)(?: - please fix this in future responses)?$", re.MULTILINE)
# The harness's own repairs of a reply it then ran, and the text to add to the reply to read it as the harness did.
KNOWN_CORRECTIONS = {
    "Fixed incomplete JSON by adding missing closing brace": "}",
    "Extracted JSON from mixed content": "",
}
LABELLED_TEXT = re.compile(r"Analysis:\s*(?P<analysis>.*?)\s*(?:Plan:\s*(?P<plan>.*?)\s*)?$", re.DOTALL)
# Flags that make a full-screen program print and exit instead of taking over the terminal.
PRINT_AND_EXIT_FLAGS = {"--version", "--help", "-h"}


@dataclass(frozen=True)
class _Reply:
    thinking: str
    analysis: str
    plan: str
    keystrokes: list[str]
    task_complete: bool


def _correction(screen_message: str | None, turn: int) -> str:
    """What the harness added to the reply before running it, from the warnings in front of the next screen."""
    if screen_message is None or not screen_message.startswith(WARNINGS):
        return ""
    header = screen_message[: min((screen_message.find(m) for m in SCREEN_MARKERS if m in screen_message), default=len(screen_message))]
    added = ""
    for match in AUTO_CORRECTED.finditer(header):
        if match.group(1) not in KNOWN_CORRECTIONS:
            raise ValueError(f"turn {turn}: the harness reports an auto-correction this converter does not know: {match.group(0)!r}")
        added += KNOWN_CORRECTIONS[match.group(1)]
    return added


def _json_reply(rest: str) -> dict | None:
    decoder = json.JSONDecoder()
    for start in [index for index, char in enumerate(rest) if char == "{"]:
        try:
            value, _ = decoder.raw_decode(rest, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "commands" in value:
            return value
    return None


def _tool_call_reply(rest: str, turn: int) -> tuple[list[str], bool] | None:
    """Keystrokes and task completion from `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks, or None
    when there is no block or one cannot be read."""
    blocks = TOOL_CALL.findall(rest)
    if not blocks:
        return None
    keystrokes: list[str] = []
    complete = False
    for body in blocks:
        try:
            call = json.loads(body)
        except json.JSONDecodeError:
            return None
        name = call.get("name") if isinstance(call, dict) else None
        if name == "bash_command":
            keystrokes.append(call["arguments"]["keystrokes"])
        elif name == "mark_task_complete":
            complete = True
        else:
            raise ValueError(f"turn {turn}: a tool call names the tool {name!r}; only bash_command and mark_task_complete are known")
    return keystrokes, complete


def _reply(content: str, turn: int) -> _Reply | None:
    """The reply in either format the dataset holds - a JSON object with "commands", or one `<tool_call>` block per
    command - or None when it is in neither (the harness could not have read it)."""
    thinking = ""
    rest = content
    think = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think:
        thinking = think.group(1).strip()
        rest = content[think.end() :]
    value = _json_reply(rest)
    if value is not None:
        keystrokes = [command["keystrokes"] for command in value["commands"]]
        return _Reply(thinking, str(value.get("analysis", "")), str(value.get("plan", "")), keystrokes, bool(value.get("task_complete", False)))
    calls = _tool_call_reply(rest, turn)
    if calls is None:
        return None
    text = rest[: rest.find("<tool_call>")].strip()
    labelled = LABELLED_TEXT.match(text)
    analysis, plan = (labelled.group("analysis"), labelled.group("plan") or "") if labelled else (text, "")
    return _Reply(thinking, analysis, plan, calls[0], calls[1])


def _interactive(keystrokes: str) -> bool:
    if keystrokes == "":
        return False
    if not keystrokes.endswith("\n") or TMUX_KEY.match(keystrokes.strip()):
        return True
    for part in split_command(keystrokes[:-1]):
        word, text = first_word(part.head)
        base = word.rsplit("/", 1)[-1]
        if base in INTERACTIVE_PROGRAMS and not _prints_and_exits(base, text, part.filters):
            return True
        if any(first_word(stage)[0] in ("less", "more") for stage in part.filters):
            return True
        if REPLS.match(base) and text.strip() == word and part.heredoc is None:
            return True
    # A line like ":wq" or "y" is input to a program that is still running, not a shell command.
    return bool(re.fullmatch(r":\w*!?|[yYnNq]", keystrokes[:-1].strip()))


def _prints_and_exits(base: str, text: str, filters: tuple[str, ...]) -> bool:
    """Whether a full-screen program only prints and exits: asked for its version or help, its output piped into a
    filter or written to a file (then it has no terminal to take over), `top` in batch mode, or vim in silent ex
    mode (-es)."""
    args = text.split()[1:]
    if PRINT_AND_EXIT_FLAGS.intersection(args) or filters or has_file_write(text):
        return True
    if base in ("vim", "vi", "nvim") and "-es" in args:
        return True
    return base == "top" and any(re.fullmatch(r"-\w*b\w*", arg) for arg in args)


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


def _wrapped_lines(lines: list[str], echo: int, first_line: str) -> int:
    """How many lines after the echo line continue the echo of a command line too long for the terminal's width
    (the terminal wraps it over several screen lines)."""
    prompt = PROMPT.match(lines[echo])
    rest = first_line.rstrip()[len(lines[echo][prompt.end() :].rstrip()) :]
    count = 0
    for line in lines[echo + 1 :]:
        # The screen drops a space that fell at the end of a wrapped line.
        if not rest or not line or not (rest.startswith(line) or rest.lstrip(" ").startswith(line)):
            break
        rest = rest.lstrip(" ")[len(line) :] if not rest.startswith(line) else rest[len(line) :]
        count += 1
    return count


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
        start = echo + 1 + _wrapped_lines(lines, echo, command.split("\n", 1)[0])
        outputs.append(_clean(lines[start:end], own, multi_line="\n" in command))
    folders = [None if echo is None else PROMPT.match(lines[echo]).group("cwd") for echo in echoes]
    return leading, outputs, folders


def _walk(turns: list[TerminusTurn], cwd: str, home: str) -> list[tuple[tuple[int, int], Shell]]:
    """Each real command of the session, keyed (turn index, command index), with the shell it starts in: the folder
    on its own prompt line when the screen showed it, else the folder after the previous command."""
    walked: list[tuple[tuple[int, int], Shell]] = []
    shell = Shell(cwd, home)
    for turn_index, turn in enumerate(turns):
        for command_index, (command, shown) in enumerate(zip(turn.commands, turn.prompt_folders)):
            if shown is not None:
                shell = Shell(home + shown[1:] if shown.startswith("~") else shown, home)
            walked.append(((turn_index, command_index), shell))
            shell = part_folders(command.text, shell)[1]
    return walked


def _with_folders(turns: list[TerminusTurn], cwd: str, home: str) -> tuple[list[TerminusTurn], int | None]:
    """The turns with each command's part folders, and the index of the first turn in which an
    information-gathering part runs in a folder that cannot be known (None when there is none)."""
    folders: dict[tuple[int, int], list[str | None]] = {}
    for key, shell in _walk(turns, cwd, home):
        folders[key] = part_folders(turns[key[0]].commands[key[1]].text, shell)[0]
    completed: list[TerminusTurn] = []
    for turn_index, turn in enumerate(turns):
        own = [folders[(turn_index, command_index)] for command_index in range(len(turn.commands))]
        for command, command_folders in zip(turn.commands, own):
            for part, folder in zip(split_command(command.text), command_folders):
                intent = part_intent(part)
                if folder is None and isinstance(intent, Intent) and intent.kind in INFORMATION_KINDS:
                    return completed, turn_index
        completed.append(replace(turn, part_folders=own))
    return completed, None


def parse_terminus(conversation: list[dict]) -> TerminusSession:
    """The task, working folder and turns of one Terminus-2 conversation; see the module docstring."""
    if not conversation or conversation[0]["role"] != "user":
        raise ValueError("a Terminus-2 conversation starts with the user message holding the task")
    task, cwd, home = _task_and_cwd(conversation[0]["content"])
    turns: list[TerminusTurn] = []
    end_reason, end_detail = "complete", None
    index = 1
    while index < len(conversation):
        message = conversation[index]
        if message["role"] != "assistant":
            raise ValueError(f"message {index} should be the coding model's reply, not a {message['role']} message")
        number = len(turns) + 1
        asked_to_confirm = CONFIRMATION.strip() in conversation[index - 1]["content"]
        screen_message = conversation[index + 1]["content"] if index + 1 < len(conversation) else None
        index += 2
        if screen_message is not None and screen_message.startswith(PARSE_ERROR):
            turns.append(TerminusTurn("", "", "", [], False, parse_error=True, prompt_folders=[]))
            continue
        reply = _reply(message["content"] + _correction(screen_message, number), number)
        if reply is None:
            if screen_message is None:
                end_reason, end_detail = "unparsed_final_reply", f"turn {number}: the transcript's last reply cannot be read"
                break
            raise ValueError(f"turn {number}: the reply is neither a JSON object with 'commands' nor tool calls, yet the harness ran it")
        interactive = next((k for k in reply.keystrokes if _interactive(k)), None)
        if interactive is not None:
            end_reason, end_detail = "interactive", f"turn {number} sends interactive keystrokes {interactive!r}"
            break
        # A keystroke string that is empty or only presses Enter waits; it is not a command.
        texts = [k[:-1] for k in reply.keystrokes if k.strip()]
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
                thinking=reply.thinking,
                analysis=reply.analysis,
                plan=reply.plan,
                commands=[TurnCommand(text, output, False) for text, output in zip(texts, outputs)],
                task_complete=reply.task_complete,
                parse_error=False,
                prompt_folders=folders,
                confirms_completion=asked_to_confirm and not texts,
            )
        )
    turns, unknown = _with_folders(turns, cwd, home)
    if unknown is not None:
        end_reason, end_detail = "folder_unknown", f"turn {unknown + 1}: a command gathers information in a folder that cannot be known"
    return TerminusSession(task=task, cwd=cwd, home=home, turns=turns, end_reason=end_reason, end_detail=end_detail)


def session_from_dataset_row(row: dict, source: str, stage: int, quality: str) -> tuple[RowSource, TerminusSession]:
    """One dataset row (columns conversations, task, trial_name, ...) as the rows' shared fields and its session.
    An error names the session."""
    conversation = row["conversations"]
    if not isinstance(conversation, list):
        raise ValueError(f"session {row['trial_name']}: the conversations must be a list of messages, not {type(conversation).__name__}")
    meta = RowSource(source=source, stage=stage, quality=quality, task=row["task"], session=row["trial_name"])
    try:
        session = parse_terminus(conversation)
    except ValueError as error:
        raise ValueError(f"session {row['trial_name']}: {error}") from error
    return meta, session


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
class SessionResult:
    """How one session's conversion ended (`end_reason`, one of END_REASONS, and `end_detail`), and its row and
    decision counts."""

    end_reason: str
    end_detail: str | None
    rows: int
    decisions: int


@dataclass(frozen=True)
class Conversion:
    """The rows of a batch of sessions; per session the turns whose point got no row because the coding model's
    next command only gathered information that no option on the rebuilt menu matched (see
    SessionLabeler.drop_unmatched_information); and per session how its conversion ended."""

    rows: list[Row]
    dropped_turns: dict[str, list[int]]
    sessions: dict[str, SessionResult]


@dataclass(frozen=True)
class _Evidence:
    """Per real command of one session, keyed (turn index, command index): its events, whether it acts (see
    labels.command_acts), the shell's folder after it (None when unknown) and the folder it started in, in the
    session's real order."""

    order: list[tuple[int, int]]
    events: dict[tuple[int, int], list[dict]]
    acts: dict[tuple[int, int], bool]
    folder_after: dict[tuple[int, int], str | None]
    prompt_folder: dict[tuple[int, int], str | None]


def _evidence(session: TerminusSession) -> _Evidence:
    evidence = _Evidence([], {}, {}, {}, {})
    for key, shell in _walk(session.turns, session.cwd, session.home):
        command = session.turns[key[0]].commands[key[1]]
        evidence.order.append(key)
        evidence.prompt_folder[key] = shell.cwd
        evidence.events[key], after = command_events(command.text, command.output, shell)
        evidence.acts[key] = command_acts(command.text)
        evidence.folder_after[key] = after.cwd
    return evidence


def _point(session: TerminusSession, evidence: _Evidence, keys: list[tuple[int, int]], history: list[ShellStep]) -> MenuPoint | None:
    """The menu builder's input at a point whose history holds the real commands `keys`, in that order; None when
    the shell's folder at the point cannot be known."""
    held = set(keys)
    later = [key for key in evidence.order if key not in held]
    point_events = [event for key in keys for event in evidence.events[key]]
    events = backfill(point_events, [(evidence.acts[key], evidence.events[key]) for key in later])
    folder = evidence.folder_after[keys[-1]] if keys else session.cwd
    if folder is None:
        # The folder became unknown (cd -, popd); the next command's prompt line shows where the shell is.
        folder = evidence.prompt_folder[later[0]] if later else None
    if folder is None:
        return None
    return MenuPoint(session.task, folder, history, events)


def _label_turns(turns: list[TerminusTurn]) -> list[LabelTurn]:
    """The turns as the labeler takes them. No row for a turn the harness rejected, for a reply to the completion
    question without commands, or for a turn without commands (a wait, a premature "task complete") that a later
    labelled turn follows before any command runs: nothing changed, so that later point has the same state and
    menu, and the decision is taken there."""
    labelled = [not turn.parse_error and not turn.confirms_completion for turn in turns]
    later_point = False
    for index in reversed(range(len(turns))):
        if labelled[index] and not turns[index].commands and later_point:
            labelled[index] = False
        if turns[index].commands:
            later_point = labelled[index]
        else:
            later_point = later_point or labelled[index]
    return [LabelTurn(turn.commands, flag, turn.part_folders) for turn, flag in zip(turns, labelled)]


@contextmanager
def _naming(session: str) -> Iterator[None]:
    """Errors raised while converting one session name it (the batch stops there)."""
    try:
        yield
    except Exception as error:
        raise RuntimeError(f"session {session}: {type(error).__name__}: {error}") from error


def convert_sessions(
    sessions: list[tuple[RowSource, TerminusSession]],
    menus_for: Callable[[list[MenuPoint]], list[Menu]] = build_menus,
) -> Conversion:
    """Label every decision point of every session and return the rows. Menus are asked for in rounds: each round
    sends one point per session that still has one, so the menu builder runs once per round, not once per point.
    Sessions converted here are approximate (menus rebuilt from the transcript), so points whose next command only
    gathers information no option matches are dropped, not labelled "hand over". A point whose shell folder cannot
    be known ends its session before that turn ("folder_unknown"). Any error stops the batch and names the session."""
    labelers: list[SessionLabeler] = []
    evidence: list[_Evidence] = []
    ends = [(session.end_reason, session.end_detail) for _, session in sessions]
    for meta, session in sessions:
        with _naming(meta.session):
            turns = _label_turns(session.turns)
            labelers.append(SessionLabeler(turns, follow_stints=True, drop_unmatched_information=True))
            evidence.append(_evidence(session))
    while True:
        pending: list[tuple[int, MenuPoint]] = []
        for number, labeler in enumerate(labelers):
            history = labeler.next_point()
            if history is None:
                continue
            with _naming(sessions[number][0].session):
                point = _point(sessions[number][1], evidence[number], labeler.next_point_keys(), history)
                if point is None:
                    ends[number] = ("folder_unknown", f"turn {labeler.current_turn}: the shell's folder before it cannot be known")
                    labeler.end_before_current_turn()
                    continue
            pending.append((number, point))
        if not pending:
            break
        menus = menus_for([point for _, point in pending])
        if len(menus) != len(pending):
            raise ValueError(f"the menu builder returned {len(menus)} menus for {len(pending)} points")
        for (number, _), menu in zip(pending, menus):
            with _naming(sessions[number][0].session):
                labelers[number].give(menu)
    rows: list[Row] = []
    results: dict[str, SessionResult] = {}
    for (meta, session), labeler, (reason, detail) in zip(sessions, labelers, ends):
        with _naming(meta.session):
            before = len(rows)
            for decision, point in enumerate(labeler.decisions):
                state = render_state(session.task, point.history)
                rows.extend(rows_for_decision(meta, decision, point.turn, state, point.menu, point.choice))
            results[meta.session] = SessionResult(reason, detail, len(rows) - before, len(labeler.decisions))
    return Conversion(rows, {meta.session: labeler.dropped for (meta, _), labeler in zip(sessions, labelers)}, results)
