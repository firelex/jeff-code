"""Drives a Terminus-2 style terminal inside a stage 1 replay container (see stage1_replay.py). It runs with the task
image's own python3 and the standard library only, from the read-only scout mount.

Terminus-2 (the harness of the ukisai sessions) types each command into one interactive `bash --login` running in a
tmux pane of 160 x 40 characters and reads the screen afterwards. This helper does the same:

- `start --harness tmux-socket|asciinema`: starts the tmux server and pane the way that harness version did, and prints
  the process id of the interactive shell that commands are typed into.
  - tmux-socket (Harbor of the main ukisai runs): socket /logs/agent/tmux.sock, a session `_harbor_dummy` first (it
    starts the server and is closed once the pane exists), then the pane, its screen piped to /logs/agent/terminus_2.pane.
  - asciinema (Harbor of run 95e2bd54): the default socket (/tmp/tmux-0/default), the pane piped the same way, then
    `asciinema rec --stdin /logs/agent/recording.cast` typed into it and the screen cleared; commands go to the shell
    inside the recording.
- `run --shell PID --limit SECONDS [--socket PATH]`: types the command read from stdin (its text and a final newline)
  into the pane, waits until the shell is idle again, and prints JSON {seconds, timed_out, region}: region is the
  screen text from the prompt line the command was typed on to the cursor's line. The shell is idle when it is the
  terminal's foreground process group and nothing typed is still waiting in the terminal's input queue, seen on two
  polls in a row. At the limit the helper presses C-c (as the harness's interrupt would) and waits again; a shell that
  is still busy 10 s later is an error (exit 3).
- `cwd --shell PID`: prints the shell's current folder.
"""

import argparse
import fcntl
import json
import os
import struct
import subprocess
import sys
import termios
import time

PANE = "main"
WIDTH = 160
HEIGHT = 40
HISTORY_LIMIT = 1_000_000
POLL_SECONDS = 0.05
AFTER_INTERRUPT_SECONDS = 5.0
SOCKET = "/logs/agent/tmux.sock"
PANE_LOG = "/logs/agent/terminus_2.pane"
RECORDING = "/logs/agent/recording.cast"


def tmux(socket: str | None, *args: str, stdin: bytes | None = None, env: dict | None = None) -> str:
    command = ["/usr/bin/tmux", *(["-S", socket] if socket else []), *args]
    finished = subprocess.run(command, input=stdin, capture_output=True, env=env, check=False)
    if finished.returncode != 0:
        raise SystemExit(f"{' '.join(command[:6])} failed (exit {finished.returncode}): {finished.stderr.decode(errors='replace')}")
    return finished.stdout.decode("utf-8", errors="replace")


def foreground_group(shell: int) -> int:
    """The terminal's foreground process group (field 8 of /proc/PID/stat, after the command name in brackets)."""
    stat = open(f"/proc/{shell}/stat").read()
    return int(stat[stat.rindex(")") + 2 :].split()[5])


def pending_input(shell: int) -> int:
    """Bytes typed into the shell's terminal that no program has read yet."""
    fd = os.open(f"/proc/{shell}/fd/0", os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0\0\0\0"))[0]
    finally:
        os.close(fd)


def idle(shell: int) -> bool:
    return foreground_group(shell) == shell and pending_input(shell) == 0


def wait_idle(shell: int, seconds: float) -> bool:
    """Whether the shell became idle (on two polls in a row) within `seconds`."""
    deadline = time.monotonic() + seconds
    seen = 0
    while time.monotonic() < deadline:
        seen = seen + 1 if idle(shell) else 0
        if seen == 2:
            return True
        time.sleep(POLL_SECONDS)
    return False


def children(pid: int) -> list[tuple[int, str]]:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = open(f"/proc/{entry}/stat").read()
        except FileNotFoundError:
            continue
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        if int(stat[stat.rindex(")") + 2 :].split()[1]) == pid:
            found.append((int(entry), name))
    return found


def start(harness: str) -> int:
    env = {**os.environ, "TERM": "xterm-256color", "SHELL": "/bin/bash"}
    if harness == "tmux-socket":
        # As that Harbor did: a first session starts the server; the history limit set then holds for the pane made
        # after it (raised here, so long outputs keep their prompt line); the first session is gone during the run.
        socket = SOCKET
        tmux(socket, "new-session", "-d", "-s", "_harbor_dummy", env=env)
        tmux(socket, "set-option", "-g", "history-limit", str(HISTORY_LIMIT))
    elif harness == "asciinema":
        # That Harbor set the history limit only after making the pane, so the pane kept tmux's default.
        socket = None
    else:
        raise SystemExit(f"unknown harness {harness!r}")
    tmux(
        socket, "new-session", "-x", str(WIDTH), "-y", str(HEIGHT), "-d", "-s", PANE, "bash --login",
        ";", "pipe-pane", "-t", PANE, f"cat > {PANE_LOG}", env=env,
    )  # fmt: skip
    if harness == "tmux-socket":
        tmux(socket, "kill-session", "-t", "_harbor_dummy")
    shell = int(tmux(socket, "display", "-p", "-t", PANE, "#{pane_pid}").strip())
    if not wait_idle(shell, 30):
        raise SystemExit("the pane's login shell did not become idle within 30 s")
    if harness == "asciinema":
        tmux(socket, "send-keys", "-t", PANE, f"asciinema rec --stdin {RECORDING}", "Enter")
        deadline = time.monotonic() + 30
        inner = None
        while inner is None and time.monotonic() < deadline:
            time.sleep(0.2)
            recorders = [pid for pid, name in children(shell) if name.startswith("asciinema")]
            shells = [pid for recorder in recorders for pid, name in children(recorder) if name == "bash"]
            inner = shells[0] if len(shells) == 1 else None
        if inner is None or not wait_idle(inner, 30):
            raise SystemExit("the shell inside asciinema rec did not start within 30 s")
        tmux(socket, "send-keys", "-t", PANE, "clear", "Enter")
        if not wait_idle(inner, 30):
            raise SystemExit("clear did not finish within 30 s")
        shell = inner
    return shell


def run(socket: str | None, shell: int, limit: float, keys: bytes) -> dict:
    size, cursor = (int(value) for value in tmux(socket, "display", "-p", "-t", PANE, "#{history_size} #{cursor_y}").split())
    first_line = size + cursor
    started = time.monotonic()
    tmux(socket, "load-buffer", "-b", "stage1replay", "-", stdin=keys)
    tmux(socket, "paste-buffer", "-d", "-b", "stage1replay", "-t", PANE)
    timed_out = not wait_idle(shell, limit)
    if timed_out:
        tmux(socket, "send-keys", "-t", PANE, "C-c")
        if not wait_idle(shell, AFTER_INTERRUPT_SECONDS):
            tmux(socket, "send-keys", "-t", PANE, "C-c")
            if not wait_idle(shell, AFTER_INTERRUPT_SECONDS):
                print(f"the shell is still busy {2 * AFTER_INTERRUPT_SECONDS:g} s after C-c at the {limit:g} s limit", file=sys.stderr)
                raise SystemExit(3)
    seconds = time.monotonic() - started
    size, cursor = (int(value) for value in tmux(socket, "display", "-p", "-t", PANE, "#{history_size} #{cursor_y}").split())
    # The prompt line may have left the history (a `clear`): then the region starts at the oldest line kept.
    region = tmux(socket, "capture-pane", "-p", "-t", PANE, "-S", str(max(first_line - size, -size)), "-E", str(cursor))
    return {"seconds": seconds, "timed_out": timed_out, "region": region}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    started = commands.add_parser("start")
    started.add_argument("--harness", required=True)
    ran = commands.add_parser("run")
    ran.add_argument("--socket")
    ran.add_argument("--shell", type=int, required=True)
    ran.add_argument("--limit", type=float, required=True)
    folder = commands.add_parser("cwd")
    folder.add_argument("--shell", type=int, required=True)
    args = parser.parse_args()
    if args.command == "start":
        print(start(args.harness))
    elif args.command == "run":
        print(json.dumps(run(args.socket, args.shell, args.limit, sys.stdin.buffer.read())))
    else:
        print(os.readlink(f"/proc/{args.shell}/cwd"))


if __name__ == "__main__":
    main()
