"""Stage 3 replay: our own record-mode sessions (pi with bash only and Qwen3.8-27B) replayed in each task's Docker
image without calling any model, so that the scout's menu exists at every decision point, inside a turn too.

Why: record mode logged the scout's menu only before each of the coding model's turns. When one turn ran several
commands that each match a menu option, the scout would have taken them one after another (a "stint"), and the menu
for the second step must be the one built after the first step ran. Those menus were never logged, so they are built
here, in a container of the task's own image that has run exactly the session's commands so far.

The recorded sessions were made before Jeff-Code was renamed from its upstream, pi: the `pi` command, PI_* environment
variables, the npm package @earendil-works/pi-coding-agent and the JeffPi Harbor agent (now JeffCode in
harbor_agent/jeff_code.py, which installs to /tmp/jeff-code.tgz and runs `jeff`). The setup patterns, environment and
paths here match those recordings, so this replay reads trials recorded with a pre-rename build only.

For each trial (see `replay_trial`):

1. A container of the task's image starts (`stage3replay-<trial>`, the task's CPU and memory limits, the default
   bridge network, `sleep infinity`), and the trial's own agent setup runs in it, as Harbor's trial.log records it:
   curl when the image lacked it, nvm with Node 22 and the scout's tarball installed with npm, pi's config folder.
   The installed Node must be the version the session ran on. The scout's list code (the same source as the
   tarball's build) is mounted read-only at SCOUT_MOUNT and run with that Node (`scripts/jeff-first-menus.ts --live`:
   facts from the container's own disk and PATH).
2. Every bash call of the session runs as pi ran it: a new `bash -c COMMAND` in the session's folder, with pi's
   environment (nvm loaded, the agent's variables, pi's PI_* variables, pi's bin folder first on PATH), stdin from
   /dev/null, stdout and stderr together. Nothing carries over between calls but the disk and running background
   programs, exactly as in pi. A call runs until it ends, its own `timeout` argument passes, or COMMAND_CAP_SECONDS
   pass. pi ran the calls of one turn at the same time; here they run one after another, so a menu can be built
   between them. When a turn's calls end sooner than the session spent on them (from the assistant message to its
   last tool result, `batch_seconds`), the replay waits the rest, so background programs get the same start-up time.
   When pi cut a call's output and saved all of it in a file (/tmp/pi-bash-<id>.log, named in its truncation notice),
   the replay saves its own output at that path, so the file exists later as it did in the session. Harbor's agent
   folder (/logs/agent: pi's event output pi.txt and pi's session file) is written before each turn's menu as it was
   at that moment (the lines before the turn's assistant message). With --think-waits, before a turn's first command
   the replay also gives programs the session left running in the background up to the model's thinking time of that
   turn, for as long as they run (`PiContainer.settle`): in the session they ran on while the
   model wrote its reply (an `apt-get install ... &` finished, a mail server delivered mail).
3. Points (labels.py SessionLabeler with stints followed, as stage 1): before each coding-model turn, and after each
   stint step inside a turn. The first command matching a menu option is a row; its recorded output joins the
   history as the scout's step (shown with the option's own command); the next matching command is the next row,
   with the menu built in the container after the previous step ran; an unmatched command that is not neutral ends
   the stint (hand over). Unmatched information commands are hand over too (the menus are real, not approximate).
4. Rows before a turn use the logged menu (exact); rows inside a turn use the menu built in the container. All rows
   are stage 3, quality "exact-replayed".
5. Fidelity: before each logged turn the container builds the menu from the session's own steps (what pi's live
   scout saw) and compares it with the logged one (`menu_checks`); every replayed command's output, formatted as pi
   shows it, is compared with the output pi recorded (`commands`). A session whose before-turn menus differ from the
   logged ones in more than MENU_DIFFERENCE_LIMIT of its turns is excluded (`session_excluded`).
6. The container is removed at the end, also when the trial fails.

Usage (on the Docker host, from tools/jeff-first):
  python3 -m imitation.pi_replay RUN [RUN ...] --tasks tasks.json --training training-tasks.json --scout SCOUT_DIR \\
      --tarball jeff-pi-scout-4abde3ece.tgz --out OUT_DIR [--parallel 60] [--only TRIAL ...]
"""

import argparse
import json
import re
import shlex
import subprocess
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from imitation.labels import SessionLabeler, gathers_information
from imitation.record_rows import (
    RecordSession,
    _calls,
    _command,
    _json_lines,
    check_logged_state,
    logged_menu,
    read_trace,
    record_sessions,
    row_lines,
    trial_cut,
)
from imitation.rows import Menu, Row, ShellStep, render_state, rows_for_decision
from imitation.stage3 import CURRENT_TARBALL, EVALUATION_TWIN, SOURCE, model_format, read_tasks

QUALITY = "exact-replayed"
COMMAND_CAP_SECONDS = 1800.0
MENU_DIFFERENCE_LIMIT = 0.10
FAILURE_SHARE_LIMIT = 0.05
CONTAINER_PREFIX = "stage3replay-"
SCOUT_MOUNT = "/var/lib/stage3replay-scout"
STATE = "/var/lib/stage3replay-state"
MENU_CLI = f"{SCOUT_MOUNT}/repo/scripts/jeff-first-menus.ts"
REMOTE_TARBALL = "/tmp/jeff-pi.tgz"
PI_AGENT_DIR = "/tmp/harbor-pi-agent"
SESSION_DIR = "/logs/agent/pi/sessions"
# Seconds the host waits for a docker command beyond the command's own time limit before it calls the replay stuck.
DOCKER_SLACK_SECONDS = 60.0
# settle() looks for background programs this often.
SETTLE_STEP_SECONDS = 1.0
# The most bytes of one command's replayed output kept for the comparison (pi shows at most 50 KB of its end).
OUTPUT_KEEP_BYTES = 1_000_000
# pi's bash tool shows the last 2,000 lines or 50 KB of an output and then this notice (tools/bash.ts), before the exit
# status when there is one.
TRUNCATION_NOTICE = re.compile(
    r"\n\n\[Showing (?:lines \d+-\d+ of \d+(?: \([^)]*\))?|last [^\]]*?)\. Full output: ([^\]]*)\]"
    r"(?=(?:\n\nCommand (?:exited with code -?\d+|timed out after [\d.]+ seconds))?$)"
)
# The commands Harbor's JeffPi agent ran before pi (harbor_agent/jeff_pi.py then, and Harbor's Pi agent), as trial.log
# shows them; the curl install runs only when the image has no curl.
SETUP_PATTERNS = (
    re.compile(r"^apt-get update && apt-get install -y curl$"),
    re.compile(r"^set -euo pipefail; curl -o- https://\S+/install\.sh \| .*npm install -g --ignore-scripts /tmp/jeff-pi\.tgz && pi --version$"),
    re.compile(r"^mkdir -p /tmp/harbor-pi-agent && chmod 700 /tmp/harbor-pi-agent$"),
    re.compile(r"^chmod 600 /tmp/harbor-pi-agent/models\.json$"),
)
NODE_IN_PROMPT = re.compile(r"/root/\.nvm/versions/node/(v[\d.]+)/lib/node_modules/@earendil-works/pi-coding-agent")


@dataclass(frozen=True)
class PiRan:
    """One bash call run in the container: what it printed (stdout and stderr together), its exit code, how long it
    took and whether it was stopped at its time limit."""

    output: str
    exit_code: int
    seconds: float
    timed_out: bool


class Container(Protocol):
    def run(self, command: str, timeout: float, full_output_path: str | None) -> PiRan: ...

    def menu(self, task: str, steps: list[ShellStep]) -> Menu: ...

    def sleep(self, seconds: float) -> None: ...

    def settle(self, seconds: float) -> None: ...

    def write_files(self, files: dict[str, str]) -> None: ...


@dataclass(frozen=True)
class SessionTiming:
    """Per assistant message of a pi session, in order: the seconds its tool calls took (from the message to its last
    tool result) and the seconds the model took to write it (from the entry before it to the message)."""

    batches: list[float]
    thinking: list[float]


@dataclass(frozen=True)
class ReplayedCommand:
    turn: int
    index: int
    command: str
    information: bool
    recorded_output: str | None
    replay_output: str
    exit_code: int
    seconds: float
    timed_out: bool
    timeout: float
    mismatch: bool
    # Still differs when every digit is removed from both (times, dates, sizes and random numbers differ by design).
    mismatch_beyond_digits: bool


@dataclass(frozen=True)
class MenuCheck:
    """Before-turn fidelity: whether the menu built in the container equals the logged one, and how it differs."""

    turn: int
    equal: bool
    difference: dict


@dataclass(frozen=True)
class Point:
    """One labelled decision point: its turn, whether it lies inside a turn (after a stint step), and its label."""

    turn: int
    stint: bool
    label: str


@dataclass
class PiReplay:
    session: str
    task: str
    machine: str
    rows: list[Row] = field(default_factory=list)
    points: list[Point] = field(default_factory=list)
    menu_checks: list[MenuCheck] = field(default_factory=list)
    commands: list[ReplayedCommand] = field(default_factory=list)
    # Every decision's full menu, so rows can be made again without a replay.
    decisions: list[dict] = field(default_factory=list)

    @property
    def stint_rows(self) -> int:
        stint_decisions = {index for index, point in enumerate(self.points) if point.stint}
        return sum(row.decision in stint_decisions for row in self.rows)

    def row_lines(self) -> list[str]:
        """The rows as JSON lines, each with its machine (ASCII escapes, as Row.to_json writes them)."""
        return [json.dumps({**asdict(row), "machine": self.machine}) for row in self.rows]


def menu_difference(logged: Menu, built: Menu) -> dict:
    """How `built` differs from `logged`: tools added or removed, and per tool the option descriptions added or
    removed (an option that only moved is not listed). Empty when they are equal."""
    difference: dict = {}
    logged_tools = [tool["id"] for tool in logged["tools"]]
    built_tools = [tool["id"] for tool in built["tools"]]
    if logged_tools != built_tools:
        difference["tools"] = {
            "added": [tool for tool in built_tools if tool not in logged_tools],
            "removed": [tool for tool in logged_tools if tool not in built_tools],
        }
    for tool in sorted(set(logged["arguments_by_tool"]) | set(built["arguments_by_tool"])):
        before = [option["description"] for option in logged["arguments_by_tool"].get(tool, [])]
        after = [option["description"] for option in built["arguments_by_tool"].get(tool, [])]
        if before != after:
            difference[tool] = {"added": [d for d in after if d not in before], "removed": [d for d in before if d not in after]}
    return difference


def _squash(text: str) -> str:
    # Whitespace and control characters (pi removes control characters other than tab and line breaks).
    return re.sub(r"[\s\x00-\x1f\x7f]+", "", text)


def pi_output_differs(
    recorded: str, output: str, exit_code: int, timed_out: bool, timeout: float | None, *, ignore_digits: bool
) -> bool:
    """Whether a replayed output, written as pi's bash tool writes its result ("(no output)" when empty, then
    "Command exited with code N" or "Command timed out after N seconds"), differs from the recorded result beyond
    whitespace (and beyond digits with `ignore_digits`). When pi cut the recorded output to its end (its truncation
    notice), only that end is compared."""
    if timed_out:
        if timeout is None:
            raise ValueError("a command that timed out needs its time limit")
        shown = f"{output}\n\nCommand timed out after {timeout:g} seconds" if output else f"Command timed out after {timeout:g} seconds"
    else:
        shown = output or "(no output)"
        if exit_code != 0:
            shown = f"{shown}\n\nCommand exited with code {exit_code}"
    match = TRUNCATION_NOTICE.search(recorded)
    kept = recorded if match is None else recorded[: match.start()] + recorded[match.end() :]
    if ignore_digits:
        kept, shown = re.sub(r"\d", "", kept), re.sub(r"\d", "", shown)
    if match is None:
        return _squash(kept) != _squash(shown)
    return not _squash(shown).endswith(_squash(kept))


def session_excluded(equal: int, differing: int) -> bool:
    """Whether a session's before-turn menus differ from the logged ones in more than MENU_DIFFERENCE_LIMIT of its
    turns."""
    if equal + differing == 0:
        raise ValueError("the session has no menu to compare")
    return differing / (equal + differing) > MENU_DIFFERENCE_LIMIT


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def session_timing(path: Path) -> SessionTiming:
    """The SessionTiming of a pi session file, from its entries' timestamps (pi writes an assistant message when the
    model's reply has ended, and a turn's tool results when all its calls have ended)."""
    lines, _ = _json_lines(path, None)
    batches: list[float] = []
    thinking: list[float] = []
    started: datetime | None = None
    previous: datetime | None = None
    for entry in lines[1:]:
        if entry["type"] != "message":
            continue
        role = entry["message"]["role"]
        moment = _time(entry["timestamp"])
        if role == "assistant":
            if previous is None:
                raise ValueError(f"{path}: an assistant message before any other message")
            started = moment
            batches.append(0.0)
            thinking.append((moment - previous).total_seconds())
        elif role == "toolResult":
            if started is None:
                raise ValueError(f"{path}: a tool result before any assistant message")
            batches[-1] = (moment - started).total_seconds()
        previous = moment
    return SessionTiming(batches, thinking)


PI_LOG_ASSISTANT_START = '{"type":"message_start","message":{"role":"assistant"'


def _assistant_lines(kind: str, lines: list[str]) -> list[int]:
    """The indexes of the lines that start an assistant message. pi's event output may hold other text (pi's stderr
    goes there too), so its lines are matched by their start, as pi writes them; a session file's lines are entries,
    and only its last line may be cut (a stopped trial)."""
    if kind == "pi-log":
        return [index for index, line in enumerate(lines) if line.replace(" ", "").startswith(PI_LOG_ASSISTANT_START)]
    if kind != "session":
        raise ValueError(f"unknown agent file kind {kind!r}")
    found: list[int] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        if entry["type"] == "message" and entry["message"]["role"] == "assistant":
            found.append(index)
    return found


def agent_file_prefixes(files: dict[str, tuple[str, str]]) -> Callable[[int], dict[str, str]]:
    """For the files Harbor's agent folder held while pi ran (path -> (kind, final text)): a function giving each
    file as it was before a turn's model call, its lines before the turn's assistant message ("pi-log": pi's JSON event
    output, /logs/agent/pi.txt; "session": pi's session file)."""
    starts: dict[str, tuple[list[str], list[int]]] = {}
    kinds = {path: kind for path, (kind, _) in files.items()}
    for path, (kind, text) in files.items():
        lines = text.splitlines(keepends=True)
        starts[path] = (lines, _assistant_lines(kind, lines))

    def before(turn: int) -> dict[str, str]:
        prefixes: dict[str, str] = {}
        for path, (lines, assistant_lines) in starts.items():
            if turn <= len(assistant_lines):
                prefixes[path] = "".join(lines[: assistant_lines[turn - 1]])
            elif kinds[path] == "pi-log":
                # pi's event output can end before the session does (pi was stopped before it wrote the rest): the
                # file never grew after its last line, so from then on it is whole.
                prefixes[path] = "".join(lines)
            else:
                raise ValueError(f"{path} has no assistant message for turn {turn}")
        return prefixes

    return before


def setup_commands(trial_log: str) -> list[str]:
    """The agent setup commands of a trial, in order, from its trial.log (each "Running command:" line before pi's
    own run). Each must be one Harbor's JeffPi agent ran (SETUP_PATTERNS); anything else raises."""
    commands: list[str] = []
    for line in trial_log.splitlines():
        if not line.startswith("Running command: "):
            continue
        command = line.removeprefix("Running command: ")
        if "pi --print" in command:
            return commands
        if not any(pattern.match(command) for pattern in SETUP_PATTERNS):
            raise ValueError(f"trial.log names an unknown setup command: {command[:200]!r}")
        commands.append(command)
    raise ValueError("trial.log never runs pi --print: the agent did not start")


def _exports(env: dict[str, str]) -> list[str]:
    return [f"export {name}={shlex.quote(value)}" for name, value in sorted(env.items())]


def command_script(command: str, cwd: str, env: dict[str, str]) -> str:
    """The script one bash call runs as: Harbor started pi with nvm loaded and the agent's variables; pi puts its bin
    folder first on PATH and starts `/bin/bash -c COMMAND` in the session's folder (pi's shell is /bin/bash, which bash
    names in its error messages)."""
    return "\n".join(
        [". ~/.nvm/nvm.sh", *_exports(env), f"export PATH={PI_AGENT_DIR}/bin:$PATH", f"cd {shlex.quote(cwd)}", f"exec /bin/bash -c {shlex.quote(command)}"]
    ) + "\n"


def _timeout(arguments: dict) -> float:
    value = arguments.get("timeout")
    if value is None:
        return COMMAND_CAP_SECONDS
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"a bash call's timeout must be a positive number of seconds, not {value!r}")
    return min(float(value), COMMAND_CAP_SECONDS)


def replay_record_session(
    prepared: RecordSession,
    timing: SessionTiming,
    container: Container,
    *,
    machine: str,
    think_waits: bool,
    agent_files: Callable[[int], dict[str, str]],
) -> PiReplay:
    """Label every point of one session with menus (logged before a turn, built in the container inside a turn) and
    check fidelity; see the module docstring. `timing`: session_timing of the session file. With `think_waits`, before
    a turn's first command the container settles for up to the model's thinking time of that turn (background programs
    ran on while the model wrote its reply). `agent_files`: the files of Harbor's agent folder as they were before a
    turn (written before each before-turn menu)."""
    session = prepared.session
    batches = timing.batches
    if len(batches) != len(session.assistants) or len(timing.thinking) != len(session.assistants):
        raise ValueError(f"{prepared.session_id}: {len(batches)} batch times for {len(session.assistants)} assistant messages")
    calls = [_calls(message) for message in session.assistants]
    real: list[ShellStep] = []
    real_before: list[int] = []
    for turn in prepared.turns:
        real_before.append(len(real))
        real.extend(ShellStep(c.text, c.output, c.is_error, by_scout=False) for c in turn.commands)
    order = [(t, c) for t, turn in enumerate(prepared.turns) for c in range(len(turn.commands))]
    replay = PiReplay(prepared.session_id, prepared.meta.task, machine)
    labeler = SessionLabeler(prepared.turns, follow_stints=True, drop_unmatched_information=False)
    hidden = session.hidden_steps
    done = 0
    batch_ran = 0.0
    while (history := labeler.next_point()) is not None:
        turn = labeler.current_turn
        taken = [index for t, index in labeler.next_point_keys() if t == turn - 1]
        stint = bool(taken)
        target = (turn - 1, max(taken) + 1) if stint else (turn - 1, 0)
        while done < len(order) and order[done] < target:
            turn_index, index = order[done]
            command = prepared.turns[turn_index].commands[index]
            limit = _timeout(calls[turn_index][index]["arguments"])
            if think_waits and index == 0:
                container.settle(min(timing.thinking[turn_index], COMMAND_CAP_SECONDS))
            notice = None if command.output is None else TRUNCATION_NOTICE.search(command.output)
            ran = container.run(command.text, limit, None if notice is None else notice.group(1))
            batch_ran += ran.seconds
            differs = [
                command.output is not None
                and pi_output_differs(command.output, ran.output, ran.exit_code, ran.timed_out, limit, ignore_digits=ignore)
                for ignore in (False, True)
            ]
            replay.commands.append(
                ReplayedCommand(
                    turn=turn_index + 1,
                    index=index,
                    command=command.text,
                    information=gathers_information(command.text),
                    recorded_output=command.output,
                    replay_output=ran.output,
                    exit_code=ran.exit_code,
                    seconds=round(ran.seconds, 3),
                    timed_out=ran.timed_out,
                    timeout=limit,
                    mismatch=differs[0],
                    mismatch_beyond_digits=differs[1],
                )
            )
            done += 1
            if index == len(prepared.turns[turn_index].commands) - 1:
                rest = min(batches[turn_index], COMMAND_CAP_SECONDS) - batch_ran
                if rest > 0:
                    container.sleep(rest)
                batch_ran = 0.0
        if stint:
            menu = container.menu(prepared.task, history[hidden[turn - 1] :])
        else:
            files = agent_files(turn)
            if files:
                container.write_files(files)
            own = real[: real_before[turn - 1]]
            check_logged_state(prepared, turn, own)
            menu = logged_menu(prepared, turn)
            built = container.menu(prepared.task, own[hidden[turn - 1] :])
            difference = menu_difference(menu, built)
            replay.menu_checks.append(MenuCheck(turn, not difference, difference))
        before = len(labeler.decisions)
        labeler.give(menu)
        for decision in labeler.decisions[before:]:
            replay.points.append(Point(turn, stint, decision.choice.option_id or "hand_over"))
    for number, point in enumerate(labeler.decisions):
        state = render_state(prepared.task, point.history[hidden[point.turn - 1] :])
        replay.rows.extend(rows_for_decision(prepared.meta, number, point.turn, state, point.menu, point.choice))
        replay.decisions.append(
            {
                "session": prepared.session_id,
                "task": prepared.meta.task,
                "machine": machine,
                "decision": number,
                "turn": point.turn,
                "stint": replay.points[number].stint,
                "kind": point.choice.kind,
                "option": point.choice.option_id,
                "menu": point.menu,
            }
        )
    return replay


class PiContainer:
    """A task container on this machine's Docker; see the module docstring."""

    def __init__(self, name: str, image: str, cpus: int, memory: str, scout: Path) -> None:
        self.name = name
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.scout = scout
        self.count = 0
        self.written: dict[str, str] = {}
        self.cwd = ""
        self.env: dict[str, str] = {}
        self.menu_env: dict[str, str] = {}
        self.run_approval = ""

    def _docker(self, args: list[str], timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
        try:
            # A command may print bytes that are not UTF-8 (a binary file); pi's bash tool decodes them with U+FFFD.
            return subprocess.run(
                ["docker", *args], input=stdin, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"docker {' '.join(args[:3])} did not return within {timeout:.0f} s") from error

    def _checked(self, args: list[str], timeout: float, stdin: str | None = None) -> str:
        finished = self._docker(args, timeout, stdin)
        if finished.returncode != 0:
            raise RuntimeError(f"docker {' '.join(args[:4])} failed (exit {finished.returncode}): {(finished.stdout + finished.stderr).strip()[-1500:]}")
        return finished.stdout

    def _root(self, script: str, timeout: float) -> str:
        return self._checked(["exec", "-u", "root", self.name, "bash", "-c", script], timeout)

    def start(self, setup: list[str], tarball: Path, node_version: str) -> None:
        self._checked(
            [
                "run", "-d", "--name", self.name, "--cpus", str(self.cpus), "--memory", self.memory,
                "-v", f"{self.scout}:{SCOUT_MOUNT}:ro", self.image, "sleep", "infinity",
            ],
            600,
        )  # fmt: skip
        state = self._checked(["inspect", "-f", "{{.State.Running}}", self.name], 60).strip()
        if state != "true":
            raise RuntimeError(f"the container {self.name} of {self.image} is not running after start (state {state})")
        self._checked(["exec", "-u", "root", self.name, "sh", "-c", "command -v bash >/dev/null && command -v timeout >/dev/null"], 60)
        self._checked(["cp", str(tarball), f"{self.name}:{REMOTE_TARBALL}"], 300)
        for command in setup:
            if command == f"chmod 600 {PI_AGENT_DIR}/models.json":
                # Harbor uploads pi's model settings (endpoint and key) here; the replay calls no model.
                self._root(f"echo '{{}}' > {PI_AGENT_DIR}/models.json", 60)
            # Harbor's exec_as_agent and ensure_system_dependencies run commands this way.
            self._root(f"set -o pipefail; {command}", 1200)
        installed = self._root(". ~/.nvm/nvm.sh && node --version", 60).strip().splitlines()[-1]
        if installed != node_version:
            raise RuntimeError(f"nvm installed Node {installed}; the session ran on {node_version}")
        self._root(f"mkdir -p {STATE} {SESSION_DIR} /logs/verifier", 60)

    def configure(self, cwd: str, env: dict[str, str], menu_env: dict[str, str], run_approval: str) -> None:
        """The session's folder, pi's environment for bash calls and the environment pi itself ran with."""
        self.cwd = cwd
        self.env = env
        self.menu_env = menu_env
        self.run_approval = run_approval

    def run(self, command: str, timeout: float, full_output_path: str | None) -> PiRan:
        """Run one bash call; with `full_output_path` (pi cut this call's output and saved all of it there), the
        replayed output is saved at that path too."""
        self.count += 1
        number = self.count
        keep = "" if full_output_path is None else f"cp {STATE}/out-{number} {shlex.quote(full_output_path)}; "
        inner = (
            f"cat > {STATE}/cmd-{number}.sh && "
            f"timeout -k 5 {timeout:g} bash {STATE}/cmd-{number}.sh > {STATE}/out-{number} 2>&1 < /dev/null; "
            f"echo $? > {STATE}/status-{number}; {keep}head -c {OUTPUT_KEEP_BYTES} {STATE}/out-{number}"
        )
        started = time.monotonic()
        output = self._checked(["exec", "-i", "-u", "root", self.name, "bash", "-c", inner], timeout + DOCKER_SLACK_SECONDS, command_script(command, self.cwd, self.env))
        seconds = time.monotonic() - started
        status = int(self._checked(["exec", "-u", "root", self.name, "cat", f"{STATE}/status-{number}"], 60).strip())
        timed_out = status in (124, 137)
        return PiRan(output, status, seconds, timed_out)

    def menu(self, task: str, steps: list[ShellStep]) -> Menu:
        point = {
            "id": "point",
            "cwd": self.cwd,
            "task": task,
            "steps": [{"command": s.command, "output": s.output, "byScout": s.by_scout} for s in steps],
            "activeTools": ["bash"],
            "runApproval": self.run_approval,
        }
        inner = "\n".join([". ~/.nvm/nvm.sh", *_exports(self.menu_env), f"cd {shlex.quote(self.cwd)}", f"exec node {MENU_CLI} --live"])
        # Building a menu can take minutes (pi's own trace logs up to 206 s for one turn: it reads every revealed text file).
        out = self._checked(["exec", "-i", "-u", "root", self.name, "bash", "-c", inner], COMMAND_CAP_SECONDS, json.dumps(point) + "\n")
        lines = [line for line in out.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"the menu CLI wrote {len(lines)} lines for one point: {out[:500]!r}")
        built = json.loads(lines[0])
        return {"tools": built["tools"], "arguments_by_tool": built["argumentsByTool"]}

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def settle(self, seconds: float) -> None:
        """Wait up to `seconds` while programs the session left running in the background are still there; return at
        once when nothing but the container's own `sleep infinity` runs. (Busy CPU is no sign: an `apt-get install &`
        spends most of its time waiting for downloads.)"""
        waited = 0.0
        while waited < seconds:
            processes = [line for line in self._checked(["top", self.name, "-eo", "pid"], 60).split("\n")[1:] if line.strip()]
            if len(processes) <= 1:
                return
            step = min(SETTLE_STEP_SECONDS, seconds - waited)
            time.sleep(step)
            waited += step

    def write_files(self, files: dict[str, str]) -> None:
        """Write each file; a file that only grew since the last write gets just its new end appended."""
        for path, text in files.items():
            previous = self.written.get(path)
            if previous is not None and text.startswith(previous):
                if len(text) > len(previous):
                    self._checked(["exec", "-i", "-u", "root", self.name, "bash", "-c", f"cat >> {shlex.quote(path)}"], 120, text[len(previous) :])
            else:
                folder = shlex.quote(str(Path(path).parent))
                self._checked(["exec", "-i", "-u", "root", self.name, "bash", "-c", f"mkdir -p {folder} && cat > {shlex.quote(path)}"], 120, text)
            self.written[path] = text

    def remove(self) -> None:
        self._checked(["rm", "-f", self.name], 300)


@dataclass(frozen=True)
class Trial:
    folder: Path
    task: str
    machine: str


def find_trials(runs: list[Path], training_tasks: Path) -> tuple[list[Trial], Counter]:
    """The trials stage3.py converts (training tasks, the current build, a trace), and why the others are skipped."""
    tasks = read_tasks(training_tasks)
    kept: list[Trial] = []
    skipped: Counter = Counter()
    for run in runs:
        configs = sorted(run.glob("*/round-*/*/*/config.json"))
        if not configs:
            raise ValueError(f"{run}: no trial folders (<stream>/round-N/<job>/<trial>/config.json)")
        for config_path in configs:
            trial = config_path.parent
            config = json.loads(config_path.read_text())
            task = config["task"]["path"]
            if task in tasks.evaluation:
                raise ValueError(f"{trial}: a session on the evaluation task {task}")
            if task == EVALUATION_TWIN or task not in tasks.training:
                skipped["not a training task"] += 1
                continue
            if Path(config["agent"]["kwargs"]["tarball"]).name != CURRENT_TARBALL:
                skipped["older build"] += 1
                continue
            if not (trial / "agent" / "jeff-first-trace.jsonl").exists():
                skipped["no trace"] += 1
                continue
            env = config["agent"]["env"]
            if env["JEFF_FIRST_MODE"] != "record":
                raise ValueError(f"{trial}: ran in {env['JEFF_FIRST_MODE']!r} mode, not record mode")
            model_format(env["JEFF_FIRST_DRIVER_BUILD"])
            kept.append(Trial(trial, task, env["JEFF_FIRST_DRIVER_BUILD"]))
    return kept, skipped


def _node_version(session_path: Path) -> str:
    """The Node version pi ran on, from the docs paths in the session's system message."""
    versions = set(NODE_IN_PROMPT.findall(session_path.read_text()))
    if len(versions) != 1:
        raise ValueError(f"{session_path}: expected one Node version in pi's docs paths, found {sorted(versions)}")
    return versions.pop()


@dataclass
class TrialReplay:
    trial: str
    sessions: list[PiReplay]
    notes: list[str]
    seconds: float


def replay_trial(trial: Trial, task_table: dict, scout: Path, tarball: Path, think_waits: bool) -> TrialReplay:
    """Replay one trial's sessions in a fresh container, removed afterwards whatever happens. Errors name the trial."""
    started = time.monotonic()
    try:
        config = json.loads((trial.folder / "config.json").read_text())
        kwargs = config["agent"]["kwargs"]
        if kwargs["tools"] != "bash":
            raise ValueError(f"pi ran with the tools {kwargs['tools']!r}; the replay knows bash only")
        all_lines, notes = read_trace(trial.folder)
        # qwen_request lines (thinking control) are read past; any other kind than record raises.
        lines, _ = row_lines(all_lines, str(trial.folder))
        if any(line["kind"] != "record" or line["driver_build"] != trial.machine for line in lines):
            raise ValueError("a trace line is not a record line of the config's driver build")
        cut = trial_cut(trial.folder) is not None
        paths = sorted((trial.folder / "agent" / "pi" / "sessions").glob("*.jsonl"))
        prepared, row_notes = record_sessions(lines, paths, cut=cut, source=SOURCE, quality=QUALITY)
        notes = notes + row_notes
        files = {}
        for path in paths:
            header = json.loads(path.read_text().split("\n", 1)[0])
            files[header["id"]] = path
        setup = setup_commands((trial.folder / "trial.log").read_text())
        spec = task_table[trial.task]
        replays: list[PiReplay] = []
        for session in prepared:
            path = files[session.session_id]
            container = PiContainer(f"{CONTAINER_PREFIX}{trial.folder.name}-{session.session_id[:8]}", spec["docker_image"], spec["cpus"], spec["memory"], scout)
            try:
                container.start(setup, tarball, _node_version(path))
                agent_env = {**config["agent"]["env"], "PI_CODING_AGENT_DIR": PI_AGENT_DIR}
                pi_env = {
                    **agent_env,
                    "PI_SESSION_ID": session.session_id,
                    "PI_SESSION_FILE": f"{SESSION_DIR}/{path.name}",
                    "PI_PROVIDER": "harbor-endpoint",
                    "PI_MODEL": config["agent"]["model_name"].split("/", 1)[1],
                    "PI_REASONING_LEVEL": kwargs["thinking"],
                }
                container.configure(session.session.cwd, pi_env, agent_env, config["agent"]["env"]["JEFF_FIRST_RUN_APPROVAL"])
                full = session_timing(path)
                count = len(session.session.assistants)
                timing = SessionTiming(full.batches[:count], full.thinking[:count])
                agent_files = agent_file_prefixes(
                    {f"{SESSION_DIR}/{path.name}": ("session", path.read_text()), "/logs/agent/pi.txt": ("pi-log", (trial.folder / "agent" / "pi.txt").read_text())}
                )
                replays.append(
                    replay_record_session(session, timing, container, machine=trial.machine, think_waits=think_waits, agent_files=agent_files)
                )
            finally:
                container.remove()
    except Exception as error:
        raise RuntimeError(f"trial {trial.folder}: {type(error).__name__}: {error}") from error
    return TrialReplay(str(trial.folder), replays, notes, time.monotonic() - started)


def stop_batch(failures: int, total: int) -> bool:
    """Whether failed trials have reached FAILURE_SHARE_LIMIT of the batch (then the batch stops)."""
    return failures / total >= FAILURE_SHARE_LIMIT


def session_record(trial: TrialReplay, replay: PiReplay) -> dict:
    equal = sum(check.equal for check in replay.menu_checks)
    differing = len(replay.menu_checks) - equal
    return {
        "trial": trial.trial,
        "session": replay.session,
        "task": replay.task,
        "machine": replay.machine,
        "excluded": session_excluded(equal, differing),
        "menus_equal": equal,
        "menus_differing": differing,
        "commands": len(replay.commands),
        "output_mismatches": sum(command.mismatch for command in replay.commands),
        "information_commands": sum(command.information and command.recorded_output is not None for command in replay.commands),
        "information_mismatches": sum(command.information and command.mismatch for command in replay.commands),
        "output_mismatches_beyond_digits": sum(command.mismatch_beyond_digits for command in replay.commands),
        "timed_out": sum(command.timed_out for command in replay.commands),
        "rows": len(replay.rows),
        "decisions": len(replay.points),
        "stint_decisions": sum(point.stint for point in replay.points),
        "stint_rows": replay.stint_rows,
        "seconds": round(trial.seconds),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="Run folders (each holds <stream>/round-N/<job>/<trial>/)")
    parser.add_argument("--tasks", type=Path, required=True, help="task -> {docker_image, cpus, memory}")
    parser.add_argument("--training", type=Path, required=True, help="results/imitation/training-tasks.json")
    parser.add_argument("--scout", type=Path, required=True, help="folder with repo/ (the list code), mounted read-only")
    parser.add_argument("--tarball", type=Path, required=True, help=f"the scout tarball the sessions ran ({CURRENT_TARBALL})")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--only", nargs="*", help="replay only these trial folder names")
    parser.add_argument(
        "--think-waits",
        action="store_true",
        help="before a turn's first command, let background programs run for up to the model's thinking time while they keep the container busy",
    )
    args = parser.parse_args(argv)
    if args.tarball.name != CURRENT_TARBALL:
        raise ValueError(f"--tarball must be {CURRENT_TARBALL}, the build the sessions ran")
    task_table = json.loads(args.tasks.read_text())
    kept, skipped = find_trials(args.runs, args.training)
    if args.only:
        unknown = set(args.only) - {trial.folder.name for trial in kept}
        if unknown:
            raise ValueError(f"--only names trials that are not replayable: {sorted(unknown)}")
        kept = [trial for trial in kept if trial.folder.name in args.only]
    # Longest sessions first (by session file size), so the batch does not end waiting on one long session.
    kept.sort(key=lambda trial: -sum(path.stat().st_size for path in (trial.folder / "agent" / "pi" / "sessions").glob("*.jsonl")))
    args.out.mkdir(parents=True, exist_ok=True)
    names = ("rows", "decisions", "commands", "menu-checks", "sessions")
    handles = {name: open(args.out / f"stage3-replay-{name}.jsonl", "x") for name in names}
    lock = threading.Lock()
    failures: list[dict] = []
    print(f"{len(kept)} trials to replay, skipped {dict(skipped)}, {args.parallel} at a time", flush=True)
    stopped = False
    with ThreadPoolExecutor(args.parallel) as pool:
        running: dict[Future, Trial] = {pool.submit(replay_trial, trial, task_table, args.scout, args.tarball, args.think_waits): trial for trial in kept}
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                trial = running.pop(future)
                with lock:
                    try:
                        result = future.result()
                    except Exception as error:
                        failures.append({"trial": str(trial.folder), "task": trial.task, "reason": str(error), "traceback": traceback.format_exc()})
                        handles["sessions"].write(json.dumps({"trial": str(trial.folder), "task": trial.task, "failed": str(error)}) + "\n")
                        handles["sessions"].flush()
                        print(f"FAILED {trial.folder.name}: {error}", flush=True)
                        if not stopped and stop_batch(len(failures), len(kept)):
                            stopped = True
                            print(f"{len(failures)} of {len(kept)} trials failed: stopping the batch", flush=True)
                            for other in running:
                                other.cancel()
                        continue
                    for replay in result.sessions:
                        record = session_record(result, replay)
                        record["notes"] = result.notes
                        if not record["excluded"]:
                            for line in replay.row_lines():
                                handles["rows"].write(line + "\n")
                        for decision in replay.decisions:
                            handles["decisions"].write(json.dumps(decision) + "\n")
                        for command in replay.commands:
                            handles["commands"].write(json.dumps({"session": replay.session, "task": replay.task, **asdict(command)}) + "\n")
                        for check in replay.menu_checks:
                            handles["menu-checks"].write(json.dumps({"session": replay.session, "task": replay.task, **asdict(check)}) + "\n")
                        handles["sessions"].write(json.dumps(record) + "\n")
                        print(
                            f"done {trial.folder.name}: {record['rows']} rows, {record['stint_rows']} stint rows, menus "
                            f"{record['menus_equal']}/{record['menus_equal'] + record['menus_differing']} equal, {record['seconds']} s",
                            flush=True,
                        )
                    for handle in handles.values():
                        handle.flush()
            running = {future: trial for future, trial in running.items() if not future.cancelled()}
    for handle in handles.values():
        handle.close()
    summary = {"trials": len(kept), "skipped": dict(skipped), "failed": len(failures), "stopped": stopped, "failures": failures}
    (args.out / "stage3-replay-run.json").write_text(json.dumps(summary, indent=1))
    if stopped:
        raise SystemExit(f"stopped: {len(failures)} of {len(kept)} trials failed (limit {FAILURE_SHARE_LIMIT:.0%})")


if __name__ == "__main__":
    main()
