"""Stage 1 replay: the nl2bash sessions of the ukisai dataset (Qwen3.8-27B in the Terminus-2 harness) replayed in their
task's own environment, rebuilt in a fresh container, so that every menu is built from the real disk ("replayed" rows).

The environment. Every nl2bash task the sessions ran (DCAgent/nl2bash-verified on Hugging Face, one gzipped task folder
per task) has the same `environment/Dockerfile` (TASK_DOCKERFILE_SHA256: ubuntu:24.04, the listed apt packages,
WORKDIR /workspace, `mkdir -p /output`) ending in `COPY seeds/ /workspace/`, where `environment/seeds/` holds the task's
own fixture files (none for some tasks). Two images are built from that Dockerfile without the seed copy, plus what
Terminus-2's setup added (Harbor installs tmux, and asciinema for the older harness version, after `apt-get update`,
keeping the package lists: sessions install packages without updating first); one per harness version (Harness). The
seeds are copied into each session's container. Evidence from the sessions' own outputs, by run:

- 7d7b83d6 (6,500 of 7,866 sessions; the harness TMUX_SOCKET): `ps` shows a tmux server on /logs/agent/tmux.sock made
  for a session `_harbor_dummy`, then `bash --login` in a pane (TMUX_PANE %1); `ls -la` shows seed files 664 and seed
  folders 775 whatever their mode in the dataset, dated hours before the session; /logs/verifier and /logs/artifacts
  are 775, owned by uid 1000 (ubuntu) and gid 1005.
- dfaf1ac0 (1,031 sessions): each holds only its first reply (no output), so there is no evidence; same days as
  7d7b83d6, taken as TMUX_SOCKET.
- 95e2bd54 (335 sessions; ASCIINEMA): tmux on its default socket, the shell inside `asciinema rec` (ASCIINEMA_REC=1,
  SHLVL=2), /tmp/get-asciinema-timestamp.sh, seeds with the dataset's modes and dates, /logs folders 777.

Each session's container (`stage1replay-<session>`, `sh -c "sleep infinity"`, the hostname of the session's own prompt,
CPUS cpus and MEMORY memory, the default bridge network) gets the seeds and the /logs folders, and before each turn's
commands the logs Harbor had written by then (episode_archive: the per-episode prompt and reply; Harbor's debug.json and
trajectory.json exist, with "{}" standing in for their unknown content); terminal_helper.py
(mounted read-only with the scout's menu code at SCOUT_MOUNT) then starts the terminal as that harness did. Each
command is typed into the pane as Terminus-2 typed it, and its output read from the screen with the same Terminus
parser that read the transcript (terminus._outputs), so both outputs are read the same way.

The walk is replay.replay_session's (labels.SessionLabeler with stints followed): a menu is built in the container
before each coding-model turn and after each labelled step inside a turn, from the transcript's history so far, in the
shell's current folder (`jeff-first-menus.ts --live`: facts from the container's disk and PATH). Unlike the approximate
conversion, a point whose next command only gathers information that no option matches is labelled "hand over": the
menu is the real one. Such decisions are listed per session (`unmatched_information`), so the approximate rule can still
be applied.

Fidelity: every replayed output is compared with the transcript's after `normalised` (whitespace, digits, times, month
and day names, long hashes and the SELinux dot after ls permissions removed: dates, sizes and process ids differ between
any two runs; lines naming the scout's mount or the terminal helper dropped), the order of lines aside
(`output_differs`). A session is excluded when more than EXCLUDED_SHARE of its compared outputs differ
(`select`).

Usage (prepare_stage1_replay.py writes the inputs; on the Docker host):
  python3 -m imitation.stage1_replay replay --sessions sessions.jsonl --tasks tasks.jsonl --scout SCOUT_DIR \\
      --out OUT_DIR [--parallel 60] [--only SESSION ...] [--limit N] [--resume]
  python3 -m imitation.stage1_replay select --rows OUT_DIR/stage1-replay-rows.jsonl \\
      --sessions OUT_DIR/stage1-replay-sessions.jsonl --out kept-rows.jsonl
"""

import argparse
import base64
import collections
import hashlib
import io
import json
import re
import socket
import subprocess
import tarfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

from imitation.labels import Decision, SessionLabeler
from imitation.replay import Ran, ReplayedCommand, SessionReplay, decision_lines, replay_session, row_json, stop_batch
from imitation.rows import Menu, Row, RowSource, ShellStep
from imitation.terminus import PROMPT, TerminusSession, _label_turns, _outputs, parse_terminus

SOURCE = "ukisai/Qwen3.8-27B-multi-turn-agent-sft"
STAGE = 1
QUALITY = "replayed"
TASK_DOCKERFILE_SHA256 = "9c980b196ee97611adaca5ddbc584ba35258b7de912225c80a55ea1849946d0f"
SEEDS_COPY = "COPY seeds/ /workspace/"
SEEDS = "environment/seeds/"
WORKDIR = "/workspace"
SCOUT_MOUNT = "/var/lib/stage1replay-scout"
NODE = f"{SCOUT_MOUNT}/node/bin/node"
MENU_CLI = f"{SCOUT_MOUNT}/repo/scripts/jeff-first-menus.ts"
HELPER = f"{SCOUT_MOUNT}/repo/tools/jeff-first/imitation/terminal_helper.py"
# The nl2bash task.toml sets no resources; Harbor's defaults.
CPUS = 1
MEMORY = "2g"
# Seconds a docker command may take beyond the command's own time limit (and the helper's two 5 s waits after C-c).
DOCKER_SLACK_SECONDS = 70.0
# A command typed into a terminal reports no exit status.
UNKNOWN_EXIT_CODE = -1
EXCLUDED_SHARE = 0.10
LOGS_OWNER = "1000:1005"
SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class Harness:
    """One version of Terminus-2's harness, as the sessions' outputs show it (see the module docstring)."""

    name: str
    image: str
    # The packages Harbor installed for the harness after `apt-get update`.
    tools: tuple[str, ...]
    # True: seed files and folders keep the dataset's modes and dates; False: files 664, folders 775, dated when the
    # container starts.
    seeds_as_archived: bool
    logs_mode: int
    # The tmux socket commands are typed through (None: tmux's default socket).
    socket: str | None
    # True: Harbor kept /logs/agent/episode-N/{prompt.txt,response.txt,debug.json} per coding-model turn.
    episode_logs: bool


TMUX_SOCKET = Harness("tmux-socket", "stage1replay-nl2bash:tmux-socket", ("tmux",), False, 0o775, "/logs/agent/tmux.sock", True)
ASCIINEMA = Harness("asciinema", "stage1replay-nl2bash:asciinema", ("tmux", "asciinema"), True, 0o777, None, False)
HARNESS_BY_RUN = {
    "7d7b83d6-b00b-4144-93fe-dadc8bd061b2": TMUX_SOCKET,
    "dfaf1ac0-28bc-492e-934e-4e4d8da84430": TMUX_SOCKET,
    "95e2bd54-a539-42d8-9e49-e74eafcc2f29": ASCIINEMA,
}
ASCIINEMA_SCRIPT = "get-asciinema-timestamp.sh"


def harness_of(run_id: str) -> Harness:
    if run_id not in HARNESS_BY_RUN:
        raise ValueError(f"run {run_id} is not one of the ukisai runs whose harness is known: {sorted(HARNESS_BY_RUN)}")
    return HARNESS_BY_RUN[run_id]


def image_dockerfile(task_dockerfile: str, harness: Harness) -> str:
    """The Dockerfile of a harness's image: the nl2bash task Dockerfile without its seed copy, then Terminus-2's setup."""
    digest = hashlib.sha256(task_dockerfile.encode()).hexdigest()
    if digest != TASK_DOCKERFILE_SHA256:
        raise ValueError(f"the task Dockerfile has sha256 {digest}, not the nl2bash one {TASK_DOCKERFILE_SHA256}")
    lines = task_dockerfile.split("\n")
    if lines.count(SEEDS_COPY) != 1:
        raise ValueError(f"the task Dockerfile must end its build with one {SEEDS_COPY!r} line")
    kept = "\n".join(line for line in lines if line != SEEDS_COPY).rstrip("\n")
    setup = [
        "",
        "# Terminus-2's setup: Harbor installs its terminal tools after apt-get update and keeps the package lists.",
        f"RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {' '.join(harness.tools)}",
    ]
    if harness.name == ASCIINEMA.name:
        setup += [
            "# Harbor uploads its recording helper to /tmp (uid 1000, gid 1005 in the sessions' listings).",
            f"COPY --chown={LOGS_OWNER} --chmod=664 {ASCIINEMA_SCRIPT} /tmp/{ASCIINEMA_SCRIPT}",
        ]
    return kept + "\n" + "\n".join(setup) + "\n"


def seed_archive(task_archive: bytes, harness: Harness, mtime: float) -> bytes:
    """The task's seed files as an uncompressed tar to unpack in the working folder: names relative to it, owned by
    root, modes and dates as the harness showed them (Harness.seeds_as_archived; otherwise dated `mtime`: the main
    runs' listings show the seeds copied a few hours before the session, so `ls -l` shows a time of day, not a year)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(task_archive)) as task, tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as out:
        for member in task.getmembers():
            if not member.name.startswith(SEEDS):
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"the seed {member.name} is neither a file nor a folder (type {member.type!r}); its copy is not known")
            info = tarfile.TarInfo(member.name[len(SEEDS) :])
            info.type, info.uid, info.gid, info.uname, info.gname = member.type, 0, 0, "root", "root"
            if harness.seeds_as_archived:
                info.mode, info.mtime = member.mode, member.mtime
            else:
                info.mode, info.mtime = (0o775 if member.isdir() else 0o664), mtime
            if member.isfile():
                info.size = member.size
                out.addfile(info, task.extractfile(member))
            else:
                out.addfile(info)
    return buffer.getvalue()


# Harbor's debug.json and trajectory.json are not in the dataset; an empty JSON object stands in for their content.
UNKNOWN_LOG = b"{}"
THINKING = re.compile(r"^<think>.*?</think>", re.DOTALL)


def episode_archive(conversation: list[dict], turn: int, harness: Harness, mtime: float) -> bytes:
    """The files Harbor added to /logs/agent by the time the commands of coding-model turn `turn` (from 1) ran, as a
    tar to unpack there: the episode folder of that turn (episode-<turn - 1>: prompt.txt, the user message it sent;
    response.txt, the reply without its thinking; debug.json) when the harness kept them, and trajectory.json (written
    after each episode, so from turn 2 on). Owned by uid 1000, gid 1005, files 664 and folders 775, as the sessions
    list them."""
    files: list[tuple[str, bytes | None]] = []
    if harness.episode_logs:
        episode = f"episode-{turn - 1}"
        reply = THINKING.sub("", conversation[2 * turn - 1]["content"])
        files += [
            (episode, None),
            (f"{episode}/prompt.txt", conversation[2 * turn - 2]["content"].encode()),
            (f"{episode}/response.txt", reply.encode()),
            (f"{episode}/debug.json", UNKNOWN_LOG),
        ]
    if turn >= 2:
        files.append(("trajectory.json", UNKNOWN_LOG))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as out:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.uid, info.gid, info.mtime = 1000, 1005, mtime
            if data is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o775
                out.addfile(info)
            else:
                info.mode, info.size = 0o664, len(data)
                out.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def command_turns(session: TerminusSession) -> list[int]:
    """The coding-model turn (from 1) of each command, in the order the replay runs them."""
    return [number for number, turn in enumerate(session.turns, start=1) for _ in turn.commands]


def hostname_of(conversation: list[dict]) -> str:
    """The container's hostname, from the shell prompt on the first message's terminal screen."""
    first = conversation[0]["content"]
    screen = first[first.find("\nCurrent terminal state:") :] if "\nCurrent terminal state:" in first else ""
    for line in screen.split("\n"):
        if PROMPT.match(line):
            return line.split("@", 1)[1].split(":", 1)[0]
    raise ValueError("the first message's terminal screen shows no shell prompt, so the hostname is unknown")


def command_output(region: str, command: str) -> str:
    """A command's output from the screen region it was typed in (from its prompt line to the cursor), read as the
    Terminus parser reads a transcript screen. When the command's echo is not in the region (it cleared the screen),
    the whole region is the output, so the comparison with the transcript sees it."""
    _, outputs, _ = _outputs(region, [command])
    return region if outputs[0] is None else outputs[0]


MONTHS = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b")
TIMES = re.compile(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?")
HASHES = re.compile(r"\b[0-9a-f]{32,}\b")
SELINUX_DOT = re.compile(r"(?m)^([-dlcbps][-rwxsStT]{9})\.")


def _line(text: str) -> str:
    text = HASHES.sub("", TIMES.sub("", SELINUX_DOT.sub(r"\1", text)))
    return re.sub(r"\s+", "", re.sub(r"\d", "", MONTHS.sub("", text)))


def _own_lines(text: str) -> str:
    """The text without the lines that show the replay's own machinery: the scout's mount, the terminal helper."""
    return "\n".join(line for line in text.replace("\r", "").split("\n") if SCOUT_MOUNT not in line and "terminal_helper.py" not in line)


def normalised(text: str) -> str:
    """An output without what differs between any two runs of the same commands: whitespace (also line wrapping),
    digits, times, month and day names (dates, sizes, process ids), hashes (of files whose dates differ), the SELinux
    mark after ls permissions, and lines that show the replay's own machinery (`_own_lines`)."""
    return _line(_own_lines(text))


def output_order_differs(transcript: str, replayed: str) -> bool:
    """Whether two outputs differ after `normalised` (the order of their lines counts)."""
    return normalised(transcript) != normalised(replayed)


def output_differs(transcript: str, replayed: str) -> bool:
    """Whether two outputs differ after `normalised`, in the order of their lines and also as sets of lines: the same
    lines in another order are the same output (folders list their entries in another order on another file system:
    the sessions ran on XFS, which lists a copied folder in name order)."""
    if not output_order_differs(transcript, replayed):
        return False
    lines = [sorted(line for line in map(_line, _own_lines(text).split("\n")) if line) for text in (transcript, replayed)]
    return lines[0] != lines[1]


@dataclass(frozen=True)
class Fidelity:
    """Of a session's replayed commands: how many have a transcript output to compare (`compared`), how many of those
    differ after normalising (`differing`), and that share (0 when nothing is compared)."""

    compared: int
    differing: int

    @property
    def share(self) -> float:
        return self.differing / self.compared if self.compared else 0.0


def session_fidelity(commands: list[ReplayedCommand]) -> Fidelity:
    compared = [command for command in commands if command.transcript_output is not None]
    return Fidelity(len(compared), sum(output_differs(c.transcript_output, c.replay_output) for c in compared))


def kept_session(record: dict) -> bool:
    """Whether a replayed session's rows are kept: it did not fail and at most EXCLUDED_SHARE of its compared outputs
    differ."""
    if "failed" in record:
        return False
    compared = record["compared_outputs"]
    return compared == 0 or record["differing_outputs"] / compared <= EXCLUDED_SHARE


def row_source(task: str, session: str) -> RowSource:
    return RowSource(source=SOURCE, stage=STAGE, quality=QUALITY, task=task, session=session)


def replay_one(meta: RowSource, session: TerminusSession, container) -> SessionReplay:
    """Label every point of one session with menus built in the container (stints followed, hand over kept)."""
    return replay_session(meta, session, container, drop_unmatched_information=False)


def unmatched_information_decisions(session: TerminusSession, decisions: list[Decision]) -> list[int]:
    """The decisions (by index) that the approximate stages' rule would leave out: hand over where the coding model's
    next command only gathers information that no option matches. Found by walking the session again with that rule
    and the same menus: both walks ask for the same points in the same order, one menu per decision."""
    labeler = SessionLabeler(_label_turns(session.turns), follow_stints=True, drop_unmatched_information=True)
    flagged: list[int] = []
    for index, decision in enumerate(decisions):
        if labeler.next_point() is None:
            raise ValueError(f"the walk with dropping ended before decision {index}")
        before = len(labeler.dropped)
        labeler.give(decision.menu)
        if len(labeler.dropped) > before:
            flagged.append(index)
    if labeler.next_point() is not None:
        raise ValueError(f"the walk with dropping asks for more points than the {len(decisions)} decisions")
    return flagged


def stint_decisions(decisions: list[Decision]) -> list[int]:
    """The decisions (by index) taken inside a coding-model turn, after a labelled step of it: a turn's first decision
    is taken before the turn; any later decision of the same turn follows a step."""
    return [index for index in range(1, len(decisions)) if decisions[index].turn == decisions[index - 1].turn]


def row_line(row: Row, machine: str) -> str:
    return row_json(row, machine)


def task_folder(task: str) -> str:
    """The nl2bash task folder of a dataset task name: a second trial is named "<task>__<trial label>"."""
    return task.split("__", 1)[0]


def session_line(dataset_row: dict) -> str:
    """One session of the prepared input: its trial name, task, run and conversation."""
    return json.dumps(
        {"session": dataset_row["trial_name"], "task": dataset_row["task"], "run_id": dataset_row["run_id"], "conversation": dataset_row["conversations"]}
    )


def task_line(task: str, archive: bytes) -> str:
    return json.dumps({"task": task, "archive": base64.b64encode(archive).decode()})


class TerminalContainer:
    """One session's container on this machine's Docker; see the module docstring."""

    def __init__(self, name: str, harness: Harness, hostname: str, seeds: bytes, scout: Path, conversation: list[dict], turns: list[int]) -> None:
        self.name = name
        self.harness = harness
        self.hostname = hostname
        self.seeds = seeds
        self.scout = scout
        self.conversation = conversation
        # The turn of each command in the order run() is called (command_turns), and the last turn whose Harbor logs
        # are written.
        self.turns = turns
        self.logged = 0
        self.ran = 0
        self.shell: int | None = None

    def _checked(self, args: list[str], timeout: float, stdin: bytes | None = None) -> str:
        try:
            finished = subprocess.run(["docker", *args], input=stdin, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"docker {' '.join(args[:4])} did not return within {timeout:.0f} s") from error
        if finished.returncode != 0:
            raise RuntimeError(
                f"docker {' '.join(args[:6])} failed (exit {finished.returncode}): {finished.stderr.decode(errors='replace').strip()[-1500:]}"
            )
        return finished.stdout.decode("utf-8", errors="replace")

    def _helper(self, args: list[str], timeout: float, stdin: bytes | None = None) -> str:
        return self._checked(["exec", *(["-i"] if stdin is not None else []), self.name, "python3", HELPER, *args], timeout, stdin)

    def start(self) -> None:
        self._checked(
            [
                "run", "-d", "--name", self.name, "--hostname", self.hostname, "--cpus", str(CPUS), "--memory", MEMORY,
                "-w", WORKDIR, "-v", f"{self.scout}:{SCOUT_MOUNT}:ro", self.harness.image, "sh", "-c", "sleep infinity",
            ],
            600,
        )  # fmt: skip
        state = self._checked(["inspect", "-f", "{{.State.Running}}", self.name], 60).strip()
        if state != "true":
            raise RuntimeError(f"the container {self.name} is not running after start (state {state})")
        if self.seeds:
            self._checked(["cp", "-a", "-", f"{self.name}:{WORKDIR}"], 120, self.seeds)
        logs = " ".join(f"/logs/{folder}" for folder in ("agent", "verifier", "artifacts"))
        self._checked(
            ["exec", self.name, "sh", "-c", f"mkdir -p {logs} && chown {LOGS_OWNER} {logs} && chmod {self.harness.logs_mode:o} {logs}"], 60
        )
        self.shell = int(self._helper(["start", "--harness", self.harness.name], 120).strip())

    def run(self, command: str, timeout: float) -> Ran:
        if self.ran >= len(self.turns):
            raise RuntimeError(f"the replay ran more than the session's {len(self.turns)} commands")
        turn = self.turns[self.ran]
        self.ran += 1
        while self.logged < turn:
            self.logged += 1
            logs = episode_archive(self.conversation, self.logged, self.harness, time.time())
            self._checked(["cp", "-a", "-", f"{self.name}:/logs/agent"], 120, logs)
        socket_args = ["--socket", self.harness.socket] if self.harness.socket else []
        out = self._helper(
            ["run", *socket_args, "--shell", str(self.shell), "--limit", f"{timeout:g}"], timeout + DOCKER_SLACK_SECONDS, (command + "\n").encode()
        )
        ran = json.loads(out)
        return Ran(command_output(ran["region"], command), UNKNOWN_EXIT_CODE, ran["seconds"], ran["timed_out"])

    def cwd(self) -> str:
        return self._helper(["cwd", "--shell", str(self.shell)], 60).strip()

    def menu(self, task: str, steps: list[ShellStep], cwd: str) -> Menu:
        point = {
            "id": "point",
            "cwd": cwd,
            "task": task,
            "steps": [{"command": s.command, "output": s.output, "byScout": s.by_scout} for s in steps],
            "activeTools": ["bash"],
            "runApproval": "all",
        }
        out = self._checked(["exec", "-i", "-w", cwd, self.name, NODE, MENU_CLI, "--live"], 300, (json.dumps(point) + "\n").encode())
        lines = [line for line in out.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"the menu CLI wrote {len(lines)} lines for one point: {out[:500]!r}")
        built = json.loads(lines[0])
        return {"tools": built["tools"], "arguments_by_tool": built["argumentsByTool"]}

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def remove(self) -> None:
        self._checked(["rm", "-f", self.name], 300)


@dataclass(frozen=True)
class Replayed:
    replay: SessionReplay
    record: dict
    commands: list[dict]


def replay_entry(entry: dict, archive: bytes, scout: Path, machine: str) -> Replayed:
    """Replay one prepared session in a fresh container, removed afterwards whatever happens. Errors name it."""
    started = time.monotonic()
    name = entry["session"]
    try:
        if not SESSION_NAME.fullmatch(name):
            raise ValueError("the session name cannot name a container")
        conversation = entry["conversation"]
        session = parse_terminus(conversation)
        harness = harness_of(entry["run_id"])
        seeds = seed_archive(archive, harness, time.time())
        container = TerminalContainer(
            f"stage1replay-{name}", harness, hostname_of(conversation), seeds, scout, conversation, command_turns(session)
        )
        try:
            container.start()
            replay = replay_one(row_source(entry["task"], name), session, container)
        finally:
            container.remove()
        fidelity = session_fidelity(replay.commands)
        record = {
            "session": name,
            "task": entry["task"],
            "run_id": entry["run_id"],
            "harness": harness.name,
            **asdict(replay.result),
            "commands_replayed": len(replay.commands),
            "compared_outputs": fidelity.compared,
            "differing_outputs": fidelity.differing,
            "timed_out": sum(command.timed_out for command in replay.commands),
            "stint_decisions": stint_decisions(replay.decisions),
            "unmatched_information": unmatched_information_decisions(session, replay.decisions),
            "seconds": round(time.monotonic() - started),
        }
    except Exception as error:
        raise RuntimeError(f"session {name}: {type(error).__name__}: {error}") from error
    commands = [
        {
            "session": name,
            "task": entry["task"],
            **asdict(command),
            "differs": command.transcript_output is not None and output_differs(command.transcript_output, command.replay_output),
            "differs_in_order": command.transcript_output is not None and output_order_differs(command.transcript_output, command.replay_output),
        }
        for command in replay.commands
    ]
    return Replayed(replay, record, commands)


def read_prepared(sessions: Path, tasks: Path) -> tuple[list[dict], dict[str, bytes]]:
    entries = [json.loads(line) for line in sessions.read_text().splitlines() if line.strip()]
    archives = {}
    for line in tasks.read_text().splitlines():
        if line.strip():
            task = json.loads(line)
            archives[task["task"]] = base64.b64decode(task["archive"])
    missing = sorted({task_folder(entry["task"]) for entry in entries} - set(archives))
    if missing:
        raise ValueError(f"{len(missing)} task folders are missing from the tasks file, e.g. {missing[:5]}")
    return entries, archives


OUTPUTS = ("rows", "decisions", "commands", "sessions")


def replay_main(args: argparse.Namespace) -> None:
    entries, archives = read_prepared(args.sessions, args.tasks)
    if args.only:
        unknown = set(args.only) - {entry["session"] for entry in entries}
        if unknown:
            raise ValueError(f"--only names sessions that are not in {args.sessions}: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["session"] in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {kind: args.out / f"stage1-replay-{kind}.jsonl" for kind in OUTPUTS}
    done: set[str] = set()
    if args.resume:
        done = {json.loads(line)["session"] for line in paths["sessions"].read_text().splitlines() if line.strip()}
        if any(not paths[kind].exists() for kind in OUTPUTS):
            raise ValueError(f"--resume needs all four output files in {args.out}")
    elif any(paths[kind].exists() for kind in OUTPUTS):
        raise ValueError(f"{args.out} already holds replay outputs; pass --resume to continue them")
    entries = [entry for entry in entries if entry["session"] not in done]
    if args.limit is not None:
        entries = entries[: args.limit]
    # Longest sessions first (by conversation size), so the batch does not end waiting on one long session.
    entries.sort(key=lambda entry: -sum(len(message["content"]) for message in entry["conversation"]))
    machine = socket.gethostname()
    handles = {kind: open(path, "a") for kind, path in paths.items()}
    lock = threading.Lock()
    failures: list[dict] = []
    print(f"{len(entries)} sessions to replay ({len(done)} done before), {args.parallel} at a time", flush=True)
    stopped = False
    with ThreadPoolExecutor(args.parallel) as pool:
        running: dict[Future, dict] = {
            pool.submit(replay_entry, entry, archives[task_folder(entry["task"])], args.scout, machine): entry for entry in entries
        }
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                entry = running.pop(future)
                with lock:
                    try:
                        replayed = future.result()
                    except Exception as error:
                        failures.append({"session": entry["session"], "reason": str(error), "traceback": traceback.format_exc()})
                        handles["sessions"].write(json.dumps({"session": entry["session"], "task": entry["task"], "failed": str(error)}) + "\n")
                        handles["sessions"].flush()
                        print(f"FAILED {entry['session']}: {error}", flush=True)
                        if not stopped and stop_batch(len(failures), len(entries)):
                            stopped = True
                            print(f"{len(failures)} of {len(entries)} sessions failed: stopping the batch", flush=True)
                            for other in running:
                                other.cancel()
                        continue
                    for row in replayed.replay.rows:
                        handles["rows"].write(row_line(row, machine) + "\n")
                    for line in decision_lines(replayed.replay, machine):
                        handles["decisions"].write(line + "\n")
                    for command in replayed.commands:
                        handles["commands"].write(json.dumps(command) + "\n")
                    handles["sessions"].write(json.dumps(replayed.record) + "\n")
                    for handle in handles.values():
                        handle.flush()
                    record = replayed.record
                    print(
                        f"done {record['session']}: {len(replayed.replay.rows)} rows, {record['decisions']} decisions "
                        f"({len(record['stint_decisions'])} in stints), outputs {record['differing_outputs']}/{record['compared_outputs']} differ, "
                        f"{record['seconds']} s",
                        flush=True,
                    )
            running = {future: entry for future, entry in running.items() if not future.cancelled()}
    for handle in handles.values():
        handle.close()
    summary = {"sessions": len(entries), "failed": len(failures), "stopped": stopped, "failures": failures}
    (args.out / f"stage1-replay-run-{int(time.time())}.json").write_text(json.dumps(summary, indent=1))
    if stopped:
        raise SystemExit(f"stopped: {len(failures)} of {len(entries)} sessions failed")


def select(rows: Path, sessions: Path, out: Path) -> dict:
    """Write the rows of kept sessions (`kept_session`) to `out` and return the counts."""
    records = [json.loads(line) for line in sessions.read_text().splitlines() if line.strip()]
    names = [record["session"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("a session is recorded twice in the sessions file")
    kept = {record["session"]: record for record in records if kept_session(record)}
    failed = sum("failed" in record for record in records)
    decisions: dict[tuple[str, int], str] = {}
    row_counts = collections.Counter()
    with open(out, "w") as handle:
        for line in rows.read_text().splitlines():
            row = json.loads(line)
            record = kept.get(row["session"])
            if record is None:
                continue
            handle.write(line + "\n")
            stint = row["decision"] in record["stint_decisions"]
            row_counts["all"] += 1
            row_counts["stint" if stint else "single_step"] += 1
            if row["level"] == "tool" and row["label"] != "show_more":
                decisions[(row["session"], row["decision"])] = row["label"]
    stint = sum(decision in kept[session]["stint_decisions"] for session, decision in decisions)
    hand_over = [key for key, label in decisions.items() if label == "hand_over"]
    return {
        "sessions": {"replayed": len(records) - failed, "failed": failed, "excluded": len(records) - failed - len(kept), "kept": len(kept)},
        "rows": {"all": row_counts["all"], "single_step": row_counts["single_step"], "stint": row_counts["stint"]},
        "decisions": {
            "all": len(decisions),
            "single_step": len(decisions) - stint,
            "stint": stint,
            "hand_over": len(hand_over),
            "hand_over_on_unmatched_information": sum(decision in kept[session]["unmatched_information"] for session, decision in hand_over),
        },
        "labels": dict(collections.Counter(decisions.values()).most_common()),
        "stint_labels": dict(collections.Counter(label for (s, d), label in decisions.items() if d in kept[s]["stint_decisions"]).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("replay", help="Replay prepared sessions in containers")
    run.add_argument("--sessions", type=Path, required=True)
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--scout", type=Path, required=True, help="folder with node/ and repo/, mounted read-only")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--parallel", type=int, default=60)
    run.add_argument("--only", nargs="*")
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    chosen = commands.add_parser("select", help="Keep the rows of sessions whose replay matched the transcript")
    chosen.add_argument("--rows", type=Path, required=True)
    chosen.add_argument("--sessions", type=Path, required=True)
    chosen.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        replay_main(args)
    else:
        print(json.dumps(select(args.rows, args.sessions, args.out), indent=1))


if __name__ == "__main__":
    main()
