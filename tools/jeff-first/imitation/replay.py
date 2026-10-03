"""Stage 2: replay the coding model's (Qwen3.8-27B) Terminus-2 sessions inside each task's own Docker image, so that the
scout's menus are built from the real file system at every point ("near-exact" rows).

For each session:

1. The task's image is started as a container (`stage2-<session>`, the task's working folder, `sleep infinity`, the
   task's CPU and memory limits). The scout's list code is mounted read-only at SCOUT_MOUNT: a Node.js binary and the
   files of this repository that `scripts/jeff-first-menus.ts` needs, run with `--live` (facts from the container's
   own disk and PATH). Node is called by its full path, so it never appears on the session's PATH.
2. The session is parsed with terminus.py (same end rules, same commands).
3. Before each decision point the menu is built inside the container, in the shell's current folder and with its
   environment, from the history so far (the transcript's commands and outputs).
4. The point is labelled with the same labels.py walk as stage 1 (stints followed; unmatched information commands
   dropped).
5. Between points the session's commands are replayed in the container in order, each with `bash -l` and with the
   shell's folder and variables carried over from the previous command (Terminus-2 types every command into one
   long-lived shell). A command runs until it ends or COMMAND_CAP_SECONDS pass (then it is interrupted, as C-c
   would); a command the session itself interrupted with C-c runs only for its session wait. When a command ends
   sooner than the session waited for it (its "duration", plus any wait-only keystrokes after it, capped at
   COMMAND_CAP_SECONDS), the replay waits the rest, so background programs it started get the same time. Each
   command's real output is kept next to the transcript's; an information command whose outputs differ beyond
   whitespace counts as a replay mismatch. The label always comes from the transcript, the menu from the container.
6. The container is removed at the end of the session, also when the session fails.

Usage (on the Docker host):
  python3 -m imitation.replay --trajectories DIR --tasks tasks.json --training training-tasks.json \\
      --scout SCOUT_DIR --out OUT_DIR [--parallel 16] [--only SESSION ...]
"""

import argparse
import json
import re
import socket
import subprocess
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from imitation.labels import SessionLabeler, gathers_information
from imitation.rows import Menu, Row, RowSource, ShellStep, render_state, rows_for_decision
from imitation.terminus import SessionResult, TerminusSession, _label_turns, parse_terminus

SOURCE = "openguardrails"
STAGE = 2
QUALITY = "near-exact"
COMMAND_CAP_SECONDS = 120.0
FAILURE_SHARE_LIMIT = 0.05
SCOUT_MOUNT = "/var/lib/stage2-scout"
STATE = "/var/lib/stage2-state"
NODE = f"{SCOUT_MOUNT}/node/bin/node"
MENU_CLI = f"{SCOUT_MOUNT}/repo/scripts/jeff-first-menus.ts"
# Seconds the host waits for a docker command beyond the command's own time limit before it calls the replay stuck.
DOCKER_SLACK_SECONDS = 60.0
# The most bytes of one command's replayed output kept for the comparison.
OUTPUT_KEEP_BYTES = 1_000_000
TASK_PREFIX = "terminal-bench__"


def conversation_from_atif(trajectory: dict) -> list[dict]:
    """The Terminus-2 conversation (user and assistant messages, as terminus.parse_terminus reads them) of an ATIF
    trajectory: the first user step's message; then per agent step its reasoning as `<think>`, its message text and
    one `<tool_call>` block per tool call, followed by the terminal message the harness answered with."""
    steps = trajectory["steps"]
    if not steps or steps[0]["source"] != "user":
        raise ValueError("an ATIF trajectory starts with the user step holding the task")
    messages = [{"role": "user", "content": steps[0]["message"]}]
    for number, step in enumerate(steps[1:], start=2):
        if step["source"] != "agent":
            raise ValueError(f"step {number} comes from {step['source']!r}; only agent steps may follow the first")
        blocks = "".join(
            f"<tool_call>\n{json.dumps({'name': tool['function_name'], 'arguments': tool['arguments']})}\n</tool_call>\n"
            for tool in step.get("tool_calls") or []
        )
        thinking = step.get("reasoning_content") or ""
        messages.append({"role": "assistant", "content": f"<think>{thinking}</think>\n{step.get('message') or ''}\n{blocks}"})
        observation = step.get("observation")
        if observation is None:
            if number != len(steps):
                raise ValueError(f"step {number} has no observation, yet more steps follow")
            continue
        results = observation["results"]
        if len(results) != 1:
            raise ValueError(f"step {number} has {len(results)} observation results; one terminal message is expected")
        messages.append({"role": "user", "content": results[0]["content"]})
    return messages


def task_of(config: dict) -> str:
    """The Terminal-Bench task name of a trial's config.json (its task folder is terminal-bench__<task>)."""
    folder = config["task"]["path"].rsplit("/", 1)[-1]
    if not folder.startswith(TASK_PREFIX):
        raise ValueError(f"the task folder {folder!r} is not named {TASK_PREFIX}<task>")
    return folder.removeprefix(TASK_PREFIX)


@dataclass(frozen=True)
class Ran:
    """One command run in the container: what it printed, its exit code, how long it took and whether it was
    interrupted at its time limit."""

    output: str
    exit_code: int
    seconds: float
    timed_out: bool


class Container(Protocol):
    def run(self, command: str, timeout: float) -> Ran: ...

    def cwd(self) -> str: ...

    def menu(self, task: str, steps: list[ShellStep], cwd: str) -> Menu: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class ReplayedCommand:
    turn: int
    index: int
    command: str
    information: bool
    transcript_output: str | None
    replay_output: str
    exit_code: int
    seconds: float
    timed_out: bool
    timeout: float
    mismatch: bool
    mismatch_beyond_digits: bool


@dataclass(frozen=True)
class SessionReplay:
    rows: list[Row]
    result: SessionResult
    dropped_turns: list[int]
    commands: list[ReplayedCommand]

    @property
    def information_mismatches(self) -> int:
        return sum(command.information and command.mismatch for command in self.commands)


def differs(transcript: str, replayed: str) -> bool:
    """Whether two outputs differ beyond whitespace (spaces, line breaks; the terminal also wraps long lines)."""
    return re.sub(r"\s+", "", transcript) != re.sub(r"\s+", "", replayed)


def _limits(session: TerminusSession) -> dict[tuple[int, int], tuple[float, float]]:
    """Per command (turn index, command index): its time limit and how long the session waited for it."""
    interrupted: set[tuple[int, int]] = set()
    last_with_commands: tuple[int, int] | None = None
    for turn_index, turn in enumerate(session.turns):
        if turn.interrupts:
            if last_with_commands is None:
                raise ValueError(f"turn {turn_index + 1} interrupts a command, but no command ran before it")
            interrupted.add(last_with_commands)
        if turn.commands:
            if len(turn.durations) != len(turn.commands):
                raise ValueError(f"turn {turn_index + 1} has {len(turn.commands)} commands but {len(turn.durations)} durations")
            last_with_commands = (turn_index, len(turn.commands) - 1)
    limits: dict[tuple[int, int], tuple[float, float]] = {}
    for turn_index, turn in enumerate(session.turns):
        for index, duration in enumerate(turn.durations):
            waited = duration
            if index == len(turn.commands) - 1:
                # Wait-only replies after the turn wait on its last command, up to and including the next turn's
                # waits before its first command.
                for later in session.turns[turn_index + 1 :]:
                    waited += later.leading_wait
                    if later.commands:
                        break
            waited = min(waited, COMMAND_CAP_SECONDS)
            limit = min(max(duration, 1.0), COMMAND_CAP_SECONDS) if (turn_index, index) in interrupted else COMMAND_CAP_SECONDS
            limits[(turn_index, index)] = (limit, waited)
    return limits


def replay_session(meta: RowSource, session: TerminusSession, container: Container) -> SessionReplay:
    """Label every decision point of one session with menus built in the container; see the module docstring."""
    labeler = SessionLabeler(_label_turns(session.turns), follow_stints=True, drop_unmatched_information=True)
    order = [(t, c) for t, turn in enumerate(session.turns) for c in range(len(turn.commands))]
    limits = _limits(session)
    replayed: list[ReplayedCommand] = []
    done = 0
    while (history := labeler.next_point()) is not None:
        turn = labeler.current_turn - 1
        taken = [index for t, index in labeler.next_point_keys() if t == turn]
        # The point follows the last command of its history: the turn's last stint command, or the previous turn.
        target = (turn, max(taken) + 1) if taken else (turn, 0)
        while done < len(order) and order[done] < target:
            turn_index, index = order[done]
            command = session.turns[turn_index].commands[index]
            limit, waited = limits[(turn_index, index)]
            ran = container.run(command.text, limit)
            if ran.seconds < waited:
                container.sleep(waited - ran.seconds)
            transcript = command.output
            mismatch = transcript is not None and differs(transcript, ran.output)
            replayed.append(
                ReplayedCommand(
                    turn=turn_index + 1,
                    index=index,
                    command=command.text,
                    information=gathers_information(command.text),
                    transcript_output=transcript,
                    replay_output=ran.output,
                    exit_code=ran.exit_code,
                    seconds=round(ran.seconds, 3),
                    timed_out=ran.timed_out,
                    timeout=limit,
                    mismatch=mismatch,
                    mismatch_beyond_digits=mismatch and differs(re.sub(r"\d", "", transcript), re.sub(r"\d", "", ran.output)),
                )
            )
            done += 1
        labeler.give(container.menu(session.task, history, container.cwd()))
    rows: list[Row] = []
    for decision, point in enumerate(labeler.decisions):
        state = render_state(session.task, point.history)
        rows.extend(rows_for_decision(meta, decision, point.turn, state, point.menu, point.choice))
    result = SessionResult(session.end_reason, session.end_detail, len(rows), len(labeler.decisions))
    return SessionReplay(rows, result, labeler.dropped, replayed)


def row_json(row: Row, machine: str) -> str:
    """One row as a JSON line, with the machine that replayed it (ASCII escapes, as Row.to_json writes them)."""
    return json.dumps({**asdict(row), "machine": machine})


def stop_batch(failures: int, total: int) -> bool:
    """Whether failed sessions have reached FAILURE_SHARE_LIMIT of the batch (then the batch stops)."""
    return failures / total >= FAILURE_SHARE_LIMIT


# The shell state carried from one command to the next: every variable but the shell's own (those are read-only or
# set by bash itself), and the folder. Saved on exit, also when the command is interrupted.
SAVE_STATE = f"""__stage2_save() {{
  for __stage2_name in $(compgen -v); do
    case "$__stage2_name" in
      BASH*|EUID|UID|PPID|SHELLOPTS|GROUPS|FUNCNAME|RANDOM|SRANDOM|SECONDS|LINENO|HISTCMD|PIPESTATUS|DIRSTACK|EPOCHREALTIME|EPOCHSECONDS|_|__stage2_*|COLUMNS|LINES|OPTIND|OPTERR|PWD|OLDPWD|SHLVL|IFS|PS4|MACHTYPE|OSTYPE|HOSTTYPE) continue ;;
    esac
    declare -p "$__stage2_name" 2>/dev/null
  done > {STATE}/env.sh.new
  mv {STATE}/env.sh.new {STATE}/env.sh
  pwd > {STATE}/cwd
}}
"""


def command_script(command: str, number: int) -> str:
    """The script one command runs as, in a new `bash -l`: the previous command's variables and folder, the command's
    output into out-<number>, and the state saved when it ends."""
    return (
        f"exec > {STATE}/out-{number} 2>&1 < /dev/null\n"
        f". {STATE}/env.sh\n"
        f'cd "$(cat {STATE}/cwd)"\n'
        f"{SAVE_STATE}"
        "trap __stage2_save EXIT\n"
        f"{command}\n"
    )


class DockerContainer:
    """A task container on this machine's Docker; see the module docstring."""

    def __init__(self, name: str, image: str, cpus: int, memory: str, workdir: str, scout: Path) -> None:
        self.name = name
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.workdir = workdir
        self.scout = scout
        self.count = 0

    def _docker(self, args: list[str], timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["docker", *args], input=stdin, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"docker {' '.join(args[:3])} did not return within {timeout:.0f} s") from error

    def _checked(self, args: list[str], timeout: float, stdin: str | None = None) -> str:
        finished = self._docker(args, timeout, stdin)
        if finished.returncode != 0:
            raise RuntimeError(f"docker {' '.join(args[:4])} failed (exit {finished.returncode}): {finished.stderr.strip()[-1500:]}")
        return finished.stdout

    def start(self) -> None:
        self._checked(
            [
                "run", "-d", "--name", self.name, "--cpus", str(self.cpus), "--memory", self.memory,
                "-w", self.workdir, "-v", f"{self.scout}:{SCOUT_MOUNT}:ro", self.image, "sleep", "infinity",
            ],
            600,
        )  # fmt: skip
        state = self._checked(["inspect", "-f", "{{.State.Running}}", self.name], 60).strip()
        if state != "true":
            raise RuntimeError(f"the container {self.name} of {self.image} is not running after start (state {state})")
        missing = self._checked(
            ["exec", "-u", "root", self.name, "sh", "-c", "for p in bash timeout; do command -v $p >/dev/null || echo $p; done"], 60
        ).split()
        if missing:
            raise RuntimeError(f"the image {self.image} lacks {', '.join(missing)}, which the replay needs")
        setup = f"mkdir -p {STATE} && cd {self.workdir} && {SAVE_STATE}__stage2_save"
        self._checked(["exec", "-u", "root", self.name, "bash", "-l", "-c", setup], 120)

    def run(self, command: str, timeout: float) -> Ran:
        self.count += 1
        number = self.count
        script = command_script(command, number)
        inner = (
            f"cat > {STATE}/cmd-{number}.sh && "
            # Only out-<number> is the output: what the login shell prints before the script redirects it is not.
            f"timeout -s INT -k 5 {timeout:g} bash -l {STATE}/cmd-{number}.sh > /dev/null 2>&1; "
            f"echo $? > {STATE}/status-{number}; head -c {OUTPUT_KEEP_BYTES} {STATE}/out-{number}"
        )
        started = time.monotonic()
        output = self._checked(["exec", "-i", "-u", "root", self.name, "bash", "-c", inner], timeout + DOCKER_SLACK_SECONDS, script)
        seconds = time.monotonic() - started
        status = int(self._checked(["exec", "-u", "root", self.name, "cat", f"{STATE}/status-{number}"], 60).strip())
        return Ran(output, status, seconds, status in (124, 137))

    def cwd(self) -> str:
        return self._checked(["exec", "-u", "root", self.name, "cat", f"{STATE}/cwd"], 60).strip()

    def menu(self, task: str, steps: list[ShellStep], cwd: str) -> Menu:
        point = {
            "id": "point",
            "cwd": cwd,
            "task": task,
            "steps": [{"command": s.command, "output": s.output, "byScout": s.by_scout} for s in steps],
            "activeTools": ["bash"],
            "runApproval": "all",
        }
        inner = f'. {STATE}/env.sh && cd "$(cat {STATE}/cwd)" && exec {NODE} {MENU_CLI} --live'
        out = self._checked(["exec", "-i", "-u", "root", self.name, "bash", "-c", inner], 300, json.dumps(point) + "\n")
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
class Trial:
    session: str
    task: str
    folder: Path


def trials(trajectories: Path, training: set[str]) -> tuple[list[Trial], list[Trial]]:
    """The trials of training tasks, and those skipped (other tasks), from the trajectory folders."""
    kept: list[Trial] = []
    skipped: list[Trial] = []
    for folder in sorted(path for path in trajectories.iterdir() if path.is_dir()):
        config = json.loads((folder / "config.json").read_text())
        trial = Trial(config["trial_name"], task_of(config), folder)
        (kept if trial.task in training else skipped).append(trial)
    return kept, skipped


def replay_trial(trial: Trial, task_table: dict, scout: Path, machine: str) -> tuple[SessionReplay, float]:
    """Replay one trial in a fresh container, removed afterwards whatever happens. Errors name the session."""
    started = time.monotonic()
    try:
        meta = RowSource(source=SOURCE, stage=STAGE, quality=QUALITY, task=trial.task, session=trial.session)
        session = parse_terminus(conversation_from_atif(json.loads((trial.folder / "agent" / "trajectory.json").read_text())))
        spec = task_table[trial.task]
        container = DockerContainer(f"stage2-{trial.session}", spec["docker_image"], spec["cpus"], spec["memory"], session.cwd, scout)
        try:
            container.start()
            replay = replay_session(meta, session, container)
        finally:
            container.remove()
    except Exception as error:
        raise RuntimeError(f"session {trial.session}: {type(error).__name__}: {error}") from error
    return replay, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="task -> {docker_image, cpus, memory}")
    parser.add_argument("--training", type=Path, required=True, help="training-tasks.json")
    parser.add_argument("--scout", type=Path, required=True, help="folder with node/ and repo/, mounted read-only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--only", nargs="*", help="replay only these sessions")
    args = parser.parse_args()
    training = set(json.loads(args.training.read_text())["training"])
    task_table = json.loads(args.tasks.read_text())
    kept, skipped = trials(args.trajectories, training)
    if args.only:
        unknown = set(args.only) - {trial.session for trial in kept}
        if unknown:
            raise ValueError(f"--only names sessions that are not training trials: {sorted(unknown)}")
        kept = [trial for trial in kept if trial.session in args.only]
    # Longest sessions first (by trajectory size), so the batch does not end waiting on one long session.
    kept.sort(key=lambda trial: -(trial.folder / "agent" / "trajectory.json").stat().st_size)
    machine = socket.gethostname()
    args.out.mkdir(parents=True, exist_ok=True)
    rows_out = open(args.out / "stage2-rows.jsonl", "w")
    commands_out = open(args.out / "stage2-replay.jsonl", "w")
    sessions_out = open(args.out / "stage2-sessions.jsonl", "w")
    lock = threading.Lock()
    failures: list[dict] = []
    print(f"{len(kept)} sessions to replay, {len(skipped)} skipped (not training tasks), {args.parallel} at a time", flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:
        running: dict[Future, Trial] = {pool.submit(replay_trial, trial, task_table, args.scout, machine): trial for trial in kept}
        stopped = False
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                trial = running.pop(future)
                with lock:
                    try:
                        replay, seconds = future.result()
                    except Exception as error:
                        failures.append({"session": trial.session, "task": trial.task, "reason": str(error), "traceback": traceback.format_exc()})
                        sessions_out.write(json.dumps({"session": trial.session, "task": trial.task, "failed": str(error)}) + "\n")
                        sessions_out.flush()
                        print(f"FAILED {trial.session}: {error}", flush=True)
                        if not stopped and stop_batch(len(failures), len(kept)):
                            stopped = True
                            print(f"{len(failures)} of {len(kept)} sessions failed: stopping the batch", flush=True)
                            for other in running:
                                other.cancel()
                        continue
                    for row in replay.rows:
                        rows_out.write(row_json(row, machine) + "\n")
                    for command in replay.commands:
                        commands_out.write(json.dumps({"session": trial.session, "task": trial.task, **asdict(command)}) + "\n")
                    record = {
                        "session": trial.session,
                        "task": trial.task,
                        **asdict(replay.result),
                        "dropped_turns": replay.dropped_turns,
                        "commands_replayed": len(replay.commands),
                        "information_mismatches": replay.information_mismatches,
                        "timed_out": sum(command.timed_out for command in replay.commands),
                        "seconds": round(seconds),
                    }
                    sessions_out.write(json.dumps(record) + "\n")
                    for handle in (rows_out, commands_out, sessions_out):
                        handle.flush()
                    print(f"done {trial.session}: {len(replay.rows)} rows, {replay.result.decisions} decisions, {round(seconds)} s", flush=True)
            running = {future: trial for future, trial in running.items() if not future.cancelled()}
    for handle in (rows_out, commands_out, sessions_out):
        handle.close()
    summary = {"sessions": len(kept), "skipped": len(skipped), "failed": len(failures), "stopped": stopped, "failures": failures}
    (args.out / "stage2-run.json").write_text(json.dumps(summary, indent=1))
    if stopped:
        raise SystemExit(f"stopped: {len(failures)} of {len(kept)} sessions failed (limit {FAILURE_SHARE_LIMIT:.0%})")


if __name__ == "__main__":
    main()
