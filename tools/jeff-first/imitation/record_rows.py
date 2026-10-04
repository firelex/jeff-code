"""Stage 3: Jeff training rows from our own record-mode sessions (bash-only pi with Qwen3.8-27B).

In record mode the scout never acts. Before each of the coding model's turns pi logs one trace line (schema
"jeff-first-trace/4", kind "record"; see packages/coding-agent/src/core/jeff-first/trace.ts, RecordRecord) holding the
scout's full option lists at that moment (`lists`: `tools` and `arguments_by_tool`) and the coding model's action
(`action.tool_calls`). The pi session file (agent/pi/sessions/*.jsonl: a "session" header line with the session id
and working folder, then one line per message) holds every command with its full output.

Each logged turn becomes one decision: the label is the option matched by the turn's first command that is not
neutral (any kind of option, Run, Check and Install included), or "hand over" when it matches none (labels.py). Each bash call runs in a new shell in the session's
folder, so paths resolve against it (and any `cd` earlier in the same command). Only the turn's first match is labelled: menus are logged only before each coding-model
turn, so the points inside a stint (after a scout step) have no menu. Rows are tagged quality "exact".

Checks (each raises ValueError naming the session and turn): every trace session has a session file; the record's
tool calls equal the session's assistant message for that turn; the number of steps the record's state covers
(recentSteps plus stepsLeftOut) equals the number of steps before that turn in the session file, and its last logged
step is the session's last command (both logged step shapes are read, see `_check_last_step`); every tool call is
bash (the imitation harness runs pi with bash only). A turn whose model call ended in "error" or "aborted" gets no
row (its menu is fine, but the coding model took no action to imitate), and so does a turn with no record line;
the returned notes list each such turn.

A bash call whose arguments hold no command text (Qwen sometimes sends `{}`; pi answers "Validation failed") ran
nothing; state.ts shows it as the tool name and its arguments written as JSON (`bash {}`), and so does this module, so
the turn is labelled from that text (it matches no option).

`read_trace` reads one trial's trace (agent/jeff-first-trace.jsonl, next to result.json, Harbor's record of the
trial). A trial is "cut" (`trial_cut`) when Harbor stopped the agent at its time limit (result.json's exception type
"AgentTimeoutError") or when it has no result.json (the collection was stopped, or the trial is still running when
its folder is copied). Then pi may have been killed in the middle of writing the trace's or the session file's last
line, and the trace may already hold the record line of a turn the session file does not have yet: only in a cut
trial is an unparsable last line dropped and such a record line dropped, each with a note. Otherwise both raise.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from imitation.events import Shell, part_folders
from imitation.labels import LabelTurn, SessionLabeler, TurnCommand
from imitation.rows import Menu, Row, RowSource, ShellStep, render_state, rows_for_decision, terminal_command

FAILED_STOPS = ("error", "aborted")
TRACE_NAME = "jeff-first-trace.jsonl"
AGENT_TIMEOUT = "AgentTimeoutError"
NO_RESULT = "no result.json (the trial was stopped or is still running)"
# state.ts before the terminal view shortened a long string argument to its first and last 600 characters.
OLD_TRIM_CHARS = 600


def trial_cut(trial: Path) -> str | None:
    """Why one Harbor trial folder's files may end in the middle of a line (see the module docstring), or None."""
    result = trial / "result.json"
    if not result.exists():
        return NO_RESULT
    info = json.loads(result.read_text())["exception_info"]
    return AGENT_TIMEOUT if info is not None and info["exception_type"] == AGENT_TIMEOUT else None


def _json_lines(path: Path, cut: str | None) -> tuple[list[dict], list[str]]:
    """The JSON lines of a file; in a cut trial an unparsable last line is dropped with a note."""
    numbered = [(number, text) for number, text in enumerate(path.read_text().splitlines(), start=1) if text.strip()]
    lines: list[dict] = []
    notes: list[str] = []
    for position, (number, text) in enumerate(numbered):
        try:
            lines.append(json.loads(text))
        except json.JSONDecodeError as error:
            if cut is not None and position == len(numbered) - 1:
                notes.append(f"{path} line {number}: cut off when the trial ended ({cut}; {error}); dropped")
                continue
            raise ValueError(f"{path} line {number} is not valid JSON: {error}") from error
    return lines, notes


def read_trace(trial: Path) -> tuple[list[dict], list[str]]:
    """The trace lines of one Harbor trial folder, and a note when a cut last line was dropped (see the module
    docstring)."""
    return _json_lines(trial / "agent" / TRACE_NAME, trial_cut(trial))


def _check_last_step(logged: dict, real: ShellStep, where: str) -> None:
    """The record's last logged step must be the session's last command before the turn. Logged steps come in two
    shapes: {command, output, isError, byScout} (terminal view, from Task 2 on; the command cut as terminal_command
    cuts it) and the older {tool, arguments, output, isError, byScout} (the bash call's arguments, a long command
    shortened to its first and last 600 characters)."""
    if "command" in logged:
        if logged["command"] != terminal_command(real.command):
            raise ValueError(f"{where}: the logged last step {logged['command'][:80]!r} is not the session's {real.command[:80]!r}")
        return
    if "tool" in logged and "arguments" in logged:
        if logged["tool"] != "bash":
            raise ValueError(f"{where}: the logged last step used the tool {logged['tool']}, not bash")
        command = logged["arguments"]["command"]
        same = (
            command == real.command
            if len(real.command) <= 2 * OLD_TRIM_CHARS
            else command.startswith(real.command[:OLD_TRIM_CHARS]) and command.endswith(real.command[-OLD_TRIM_CHARS:])
        )
        if not same:
            raise ValueError(f"{where}: the logged last step {command[:80]!r} is not the session's {real.command[:80]!r}")
        return
    raise ValueError(f"{where}: a logged step has neither a command nor a tool and arguments: {sorted(logged)}")


def _text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] if part["type"] == "text" else "[image]" for part in content)


@dataclass(frozen=True)
class Session:
    """One pi session file: its working folder, its assistant messages in order, tool results by call id, and for
    each assistant message the number of steps (tool calls) before it that pi's context no longer holds because of
    a compaction (see `_read_session`)."""

    cwd: str
    assistants: list[dict]
    results: dict[str, dict]
    hidden_steps: list[int]


def _read_session(path: Path, cut: bool, notes: list[str]) -> tuple[str, Session]:
    """The session id and its contents. The entries must form one chain (each entry's parent is the entry before it),
    so the file's order is the context's order. After a compaction entry pi's context holds the summary, the entries
    from its firstKeptEntryId on and every later entry (session-manager.ts), so the steps before firstKeptEntryId are
    hidden from the scout's state from then on."""
    lines, cut_notes = _json_lines(path, "cut" if cut else None)
    notes.extend(cut_notes)
    if not lines or lines[0].get("type") != "session":
        raise ValueError(f"{path} does not start with a session header line")
    assistants: list[dict] = []
    results: dict[str, dict] = {}
    hidden_steps: list[int] = []
    steps_before: dict[str, int] = {}
    steps = 0
    hidden = 0
    parent = None
    for entry in lines[1:]:
        if entry["parentId"] != parent:
            raise ValueError(f"{path}: entry {entry['id']} does not follow the entry before it (the session branches)")
        parent = entry["id"]
        steps_before[entry["id"]] = steps
        if entry["type"] == "compaction":
            if entry["firstKeptEntryId"] not in steps_before:
                raise ValueError(f"{path}: compaction {entry['id']} keeps from the unknown entry firstKeptEntryId {entry['firstKeptEntryId']}")
            hidden = steps_before[entry["firstKeptEntryId"]]
            continue
        if entry["type"] != "message":
            continue
        message = entry["message"]
        if message["role"] == "assistant":
            assistants.append(message)
            hidden_steps.append(hidden)
            steps += len(_calls(message))
        elif message["role"] == "toolResult":
            results[message["toolCallId"]] = message
    return lines[0]["id"], Session(lines[0]["cwd"], assistants, results, hidden_steps)


def _calls(message: dict) -> list[dict]:
    return [part for part in message["content"] if part["type"] == "toolCall"]


def _command(call: dict) -> str:
    """The shell command of a bash call, or what state.ts shows for one without command text: the tool name and its
    arguments as JSON.stringify writes them (no spaces, characters unescaped)."""
    command = call["arguments"].get("command")
    if isinstance(command, str):
        return command
    return f"{call['name']} {json.dumps(call['arguments'], separators=(',', ':'), ensure_ascii=False)}"


@dataclass(frozen=True)
class RecordSession:
    """One pi session of a trace, checked and ready to label: its rows' shared fields, the task text, its turns as
    labels.py takes them, the record line of each logged turn (by 1-based turn number) and the session file."""

    session_id: str
    meta: RowSource
    task: str
    turns: list[LabelTurn]
    records: dict[int, dict]
    session: Session


def record_sessions(
    trace: list[dict], session_files: list[Path], *, cut: bool, source: str, quality: str
) -> tuple[list[RecordSession], list[str]]:
    """The sessions of every record line of `trace` (lines of other kinds are ignored), and notes on turns given no
    row. `cut`: the trial was cut (`trial_cut`), so its files may end early (see the module docstring)."""
    sessions: dict[str, Session] = {}
    notes: list[str] = []
    for path in session_files:
        session_id, session = _read_session(path, cut, notes)
        sessions[session_id] = session
    by_session: dict[str, dict[int, dict]] = {}
    for line in trace:
        if line.get("kind") == "record":
            by_session.setdefault(line["session_id"], {})[line["turn"]] = line
    prepared: list[RecordSession] = []
    for session_id, records in by_session.items():
        if session_id not in sessions:
            raise ValueError(f"no pi session file holds the session {session_id} of the trace")
        session = sessions[session_id]
        cwd, assistants, results = session.cwd, session.assistants, session.results
        if cut:
            for turn in sorted(number for number in records if number > len(assistants)):
                notes.append(f"{session_id} turn {turn}: the trial was cut before the session file had this turn; record line dropped")
                del records[turn]
            if not records:
                continue
        turns: list[LabelTurn] = []
        for number, message in enumerate(assistants, start=1):
            commands: list[TurnCommand] = []
            for call in _calls(message):
                if call["name"] != "bash":
                    raise ValueError(f"{session_id} turn {number}: the tool call {call['name']} is not bash")
                result = results.get(call["id"])
                output = None if result is None else _text(result["content"])
                commands.append(TurnCommand(_command(call), output, bool(result and result["isError"])))
            record = records.get(number)
            if record is None:
                notes.append(f"{session_id} turn {number}: the trace has no record line for this turn; no row")
            else:
                logged = [{"name": c["name"], "arguments": c["arguments"]} for c in record["action"]["tool_calls"]]
                if logged != [{"name": c["name"], "arguments": c["arguments"]} for c in _calls(message)]:
                    raise ValueError(f"{session_id} turn {number}: the trace's tool calls differ from the session file's")
                if record["action"]["stop_reason"] in FAILED_STOPS:
                    notes.append(f"{session_id} turn {number}: the model call ended with {record['action']['stop_reason']}; no row")
                    record = None
            # pi runs each bash call in a new shell in the session's folder; the user's home is not recorded.
            folders = [part_folders(command.text, Shell(cwd, None))[0] for command in commands]
            turns.append(LabelTurn(commands, labelled=record is not None, folders=folders))
        missing = sorted(set(records) - set(range(1, len(assistants) + 1)))
        if missing:
            raise ValueError(f"{session_id}: the trace has turns {missing} that the session file does not")
        first = records[min(records)]
        meta = RowSource(source=source, stage=3, quality=quality, task=first["task_id"], session=session_id)
        prepared.append(RecordSession(session_id, meta, first["state"]["task"], turns, records, session))
    return prepared, notes


def check_logged_state(prepared: RecordSession, turn: int, history: list[ShellStep]) -> None:
    """The record line of `turn` must describe the session file's own steps before it (`history`: every step before
    the turn as the coding model took it, compacted steps included): as many steps as pi's context held, and its last
    logged step is the session's last command (ValueError otherwise)."""
    record = prepared.records[turn]
    covered = len(record["state"]["recentSteps"]) + record["state"]["stepsLeftOut"]
    kept = history[prepared.session.hidden_steps[turn - 1] :]
    if covered != len(kept):
        raise ValueError(
            f"{prepared.session_id} turn {turn}: the logged state covers {covered} steps, "
            f"the session file has {len(kept)} steps in pi's context before this turn"
        )
    if record["state"]["recentSteps"]:
        _check_last_step(record["state"]["recentSteps"][-1], kept[-1], f"{prepared.session_id} turn {turn}")


def logged_menu(prepared: RecordSession, turn: int) -> Menu:
    """The menu the scout logged before `turn`."""
    lists = prepared.records[turn]["lists"]
    return {"tools": lists["tools"], "arguments_by_tool": lists["arguments_by_tool"]}


def record_rows(
    trace: list[dict], session_files: list[Path], *, cut: bool, source: str = "jeff-pi-record"
) -> tuple[list[Row], list[str]]:
    """Rows for every record line of `trace` (lines of other kinds are ignored), and notes on turns given no row.
    `cut`: the trial was cut (`trial_cut`), so its files may end early (see the module docstring)."""
    prepared_sessions, notes = record_sessions(trace, session_files, cut=cut, source=source, quality="exact")
    rows: list[Row] = []
    for prepared in prepared_sessions:
        labeler = SessionLabeler(prepared.turns, follow_stints=False, drop_unmatched_information=False)
        while (history := labeler.next_point()) is not None:
            check_logged_state(prepared, labeler.current_turn, history)
            labeler.give(logged_menu(prepared, labeler.current_turn))
        hidden = prepared.session.hidden_steps
        for decision, point in enumerate(labeler.decisions):
            state = render_state(prepared.task, point.history[hidden[point.turn - 1] :])
            rows.extend(rows_for_decision(prepared.meta, decision, point.turn, state, point.menu, point.choice))
    return rows, notes
