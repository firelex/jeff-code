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

`read_trace` reads one trial's trace (agent/jeff-first-trace.jsonl, next to result.json, Harbor's record of the
trial). When Harbor stopped the agent at its time limit (result.json's exception type "AgentTimeoutError"), pi may
have been killed in the middle of writing the trace's last line; only then is an unparsable last line dropped, and
a note says so. Any other unparsable line raises.
"""

import json
from pathlib import Path

from imitation.events import Shell, part_folders
from imitation.labels import LabelTurn, SessionLabeler, TurnCommand
from imitation.rows import Menu, Row, RowSource, ShellStep, render_state, rows_for_decision, terminal_command

FAILED_STOPS = ("error", "aborted")
TRACE_NAME = "jeff-first-trace.jsonl"
AGENT_TIMEOUT = "AgentTimeoutError"
# state.ts before the terminal view shortened a long string argument to its first and last 600 characters.
OLD_TRIM_CHARS = 600


def read_trace(trial: Path) -> tuple[list[dict], list[str]]:
    """The trace lines of one Harbor trial folder, and a note when a last line cut by the agent time limit was
    dropped (see the module docstring)."""
    path = trial / "agent" / TRACE_NAME
    info = json.loads((trial / "result.json").read_text())["exception_info"]
    timed_out = info is not None and info["exception_type"] == AGENT_TIMEOUT
    texts = path.read_text().splitlines()
    numbered = [(number, text) for number, text in enumerate(texts, start=1) if text.strip()]
    lines: list[dict] = []
    notes: list[str] = []
    for position, (number, text) in enumerate(numbered):
        try:
            lines.append(json.loads(text))
        except json.JSONDecodeError as error:
            if timed_out and position == len(numbered) - 1:
                notes.append(f"{path} line {number}: cut off when the trial ended with {AGENT_TIMEOUT} ({error}); dropped")
                continue
            raise ValueError(f"{path} line {number} is not valid JSON: {error}") from error
    return lines, notes


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


def _read_session(path: Path) -> tuple[str, str, list[dict], dict[str, dict]]:
    """The session id, its working folder, its assistant messages in order, and tool results by call id."""
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0].get("type") != "session":
        raise ValueError(f"{path} does not start with a session header line")
    assistants: list[dict] = []
    results: dict[str, dict] = {}
    for entry in lines[1:]:
        if entry.get("type") != "message":
            continue
        message = entry["message"]
        if message["role"] == "assistant":
            assistants.append(message)
        elif message["role"] == "toolResult":
            results[message["toolCallId"]] = message
    return lines[0]["id"], lines[0]["cwd"], assistants, results


def _calls(message: dict) -> list[dict]:
    return [part for part in message["content"] if part["type"] == "toolCall"]


def record_rows(trace: list[dict], session_files: list[Path], source: str = "jeff-pi-record") -> tuple[list[Row], list[str]]:
    """Rows for every record line of `trace` (lines of other kinds are ignored), and notes on turns given no row."""
    sessions = {}
    for path in session_files:
        session_id, cwd, assistants, results = _read_session(path)
        sessions[session_id] = (cwd, assistants, results)
    by_session: dict[str, dict[int, dict]] = {}
    for line in trace:
        if line.get("kind") == "record":
            by_session.setdefault(line["session_id"], {})[line["turn"]] = line
    rows: list[Row] = []
    notes: list[str] = []
    for session_id, records in by_session.items():
        if session_id not in sessions:
            raise ValueError(f"no pi session file holds the session {session_id} of the trace")
        cwd, assistants, results = sessions[session_id]
        turns: list[LabelTurn] = []
        for number, message in enumerate(assistants, start=1):
            commands: list[TurnCommand] = []
            for call in _calls(message):
                if call["name"] != "bash":
                    raise ValueError(f"{session_id} turn {number}: the tool call {call['name']} is not bash")
                result = results.get(call["id"])
                output = None if result is None else _text(result["content"])
                commands.append(TurnCommand(call["arguments"]["command"], output, bool(result and result["isError"])))
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
        meta = RowSource(source=source, stage=3, quality="exact", task=first["task_id"], session=session_id)
        task = first["state"]["task"]
        labeler = SessionLabeler(turns, follow_stints=False, drop_unmatched_information=False)
        while (history := labeler.next_point()) is not None:
            record = records[labeler.current_turn]
            covered = len(record["state"]["recentSteps"]) + record["state"]["stepsLeftOut"]
            if covered != len(history):
                raise ValueError(
                    f"{session_id} turn {labeler.current_turn}: the logged state covers {covered} steps, "
                    f"the session file has {len(history)} steps before this turn"
                )
            if record["state"]["recentSteps"]:
                _check_last_step(record["state"]["recentSteps"][-1], history[-1], f"{session_id} turn {labeler.current_turn}")
            menu: Menu = {"tools": record["lists"]["tools"], "arguments_by_tool": record["lists"]["arguments_by_tool"]}
            labeler.give(menu)
        for decision, point in enumerate(labeler.decisions):
            rows.extend(
                rows_for_decision(meta, decision, point.turn, render_state(task, point.history), point.menu, point.choice)
            )
    return rows, notes
