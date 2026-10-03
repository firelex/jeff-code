"""How much the scout is worth on one task: how much of the work it did instead of the large coding model (Qwen),
how much of Qwen's time that is likely to have saved, how long the scout itself took to decide, and, for each kind
of thing the scout can do (read a file, peek at a data file, list a folder, search, find files, check the
toolchain, check a service, look up package docs, run the project's checks, run a script, install a package, or
repeat the last command), how often it was offered to the scout, how often the scout picked it, how often Qwen's
very next turn afterwards touched the same thing, and how often Qwen would have done the same thing anyway in the
plain run with no scout.

This compares two runs of the same task with the same large model: the "teacher" run, where the scout takes small
steps between the large model's turns, and the "base" (baseline) run, where the large model works alone. Both runs
write one line per large-model turn to a trace file; the teacher run also writes one line per scout decision.

Plain-English definitions used below:

- A tool call: one entry of type "toolCall" inside one assistant message's content, in pi's own session log
  (agent/pi/sessions/*.jsonl). It has a "name" (which pi tool it called, e.g. "read", "bash", "write") and
  "arguments". A scout step shows up as its own, separate assistant message whose "provider" is "jeff-first"; every
  other assistant message's tool calls are Qwen's.

- The scout's share of tool calls: of every tool call in the teacher run's pi session(s), the fraction made by the
  scout (provider "jeff-first") rather than Qwen.

- The scout's decision time: how many seconds the scout itself spent choosing, across every decision in the
  teacher run - the sum of each decision's "timings_ms.chooser" (how long picking an option took), turned from
  milliseconds into seconds.

- The large model's typical turn length: the median of "timings_ms.model" (how long one Qwen turn took to produce
  its reply) across the teacher run's "model_turn" trace lines.

- Estimated seconds of Qwen time avoided: each scout step stands in for one Qwen turn that did not have to happen.
  We do not know how long that particular avoided turn would have taken, so we estimate it as the typical turn
  length above, multiplied by how many steps the scout took.

- A scout "step": a decision whose action is "step" - the scout chose a tool and an argument, and pi ran that tool
  call instead of asking Qwen. A decision whose action is "hand_over" took no step; it either asked the scout and
  it chose "hand over", chose "none of these" when shown an argument, or hit the per-turn step limit before being
  asked at all (in which case the decision's "levels" list is empty, since no question was ever put to it).

- The tool kind of a decision: the "chosen" value on the first entry of that decision's "levels" list (the first
  question put to the scout: which kind of tool to use). This is one of "read", "peek", "list", "search", "find",
  "toolchain", "service", "docs", "check", "run", "install", "repeat" (the scout's step), or "hand_over" (no step).

- Offered: a decision "offers" a tool kind when that kind is one of the options listed on the first entry of its
  "levels" list, whether or not the scout picked it.

- Chosen: a decision "chooses" a tool kind when that kind is its tool kind (as above) and its action is a step
  (not a hand-over).

- The target of a step: the one thing the step's tool call is about, used to check afterwards whether anyone else
  did something to the same thing. For "read" and "peek" it is the file path (the call's "path" argument, or, for
  a "peek" bash command, the first absolute path written in the command text). For "list" it is the folder path.
  For "toolchain" it is always the fixed word "toolchain" (the probe always checks a whole list of programs and
  modules at once, so no single name stands for it). For "service" it is "localhost:<port>" found in the command.
  For "docs" and "install" it is the package or program name named in the command (after "apt install", "apt-get
  install" or "pip install"). For "check", "run" and "repeat" it is the command itself, with any leading "cd
  <folder> &&" removed.

- A later tool call "mentions" a step's target: for every kind except "toolchain", the target string appears
  somewhere in the JSON text of the later tool call(s) - for a file or folder path, its last path segment (so
  "/app/a.py" counts as mentioned by a command that names "a.py" without the leading folders) is searched for
  instead of the whole path, since the large model rarely repeats a full path verbatim. For "toolchain", there is
  no one name to search for (the target is just the word "toolchain"); instead, a later call "mentions" it when its
  command contains one of the patterns Qwen itself uses to probe the toolchain: "which ", "command -v",
  "--version", "import ", or "/etc/os-release".

- Used: a step is "used" when the teacher run's very next "model_turn" trace line after it (the large model's next
  turn, once the scout hands over) makes a tool call that mentions the step's target.

- A tool call's phase: how many write-like tool calls by the large model (pi's "write" or "edit" tool, or a bash
  command whose text contains "cat >" or "tee ") came before it, in order, within one run's pi session(s). The
  scout's own tool calls never count as write-like (the scout is never given write or edit).

- Anticipated: a step is "anticipated" when, in the base run's pi session(s), some tool call by the large model
  sits at the very same phase as the step (the same number of prior write-like calls) and mentions the step's
  target - i.e. working alone, at the same point in its work, Qwen did the same thing the scout did for it in the
  teacher run.

Usage: uv run python scout_value.py BASE_RUNS TEACHER_RUNS TASK [TASK ...]
"""

import json
import re
import statistics
import sys
from pathlib import Path, PurePosixPath

from gate0_report import latest_trial

KNOWN_KINDS = ("read", "peek", "list", "search", "find", "toolchain", "service", "docs", "check", "run", "install", "repeat")
PATH_KINDS = ("read", "peek", "list")
TOOLCHAIN_TARGET = "toolchain"
TOOLCHAIN_MARKERS = ("which ", "command -v", "--version", "import ", "/etc/os-release")
WRITE_IN_BASH = ("cat >", "tee ")

ABS_PATH = re.compile(r"/[^\s'\"]+")
CD_PREFIX = re.compile(r"^\s*cd\s+\S+\s*&&\s*")
LOCALHOST = re.compile(r"localhost:\d+")
APT_INSTALL = re.compile(r"\b(?:apt-get|apt)\s+install\s+(\S+)")
PIP_INSTALL = re.compile(r"\b(?:pip3?|python3\s+-m\s+pip)\s+install\s+(\S+)")


def read_trace(trial: Path) -> list[dict]:
    path = trial / "agent" / "jeff-first-trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_session_calls(trial: Path) -> list[tuple[str | None, dict]]:
    """Every tool call in this trial's pi session(s), in file and line order, paired with the "provider" of the
    assistant message that made it ("jeff-first" for the scout, anything else - including missing - for Qwen)."""
    calls: list[tuple[str | None, dict]] = []
    for session in sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl")):
        for line in session.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") != "message":
                continue
            message = entry["message"]
            if message.get("role") != "assistant":
                continue
            provider = message.get("provider")
            for part in message.get("content", []):
                if part.get("type") == "toolCall":
                    calls.append((provider, {"name": part["name"], "arguments": part["arguments"]}))
    return calls


def write_like(call: dict) -> bool:
    """Whether this is a write-like call by the large model: pi's write or edit tool, or a bash command whose text
    contains "cat >" or "tee " (the shapes it uses to write a file from a shell command)."""
    if call["name"] in ("write", "edit"):
        return True
    if call["name"] == "bash":
        command = call["arguments"].get("command", "")
        return any(marker in command for marker in WRITE_IN_BASH)
    return False


def step_phases(trace: list[dict]) -> dict[int, int]:
    """For each decision line's index in `trace`, how many write-like calls by the large model appear in earlier
    "model_turn" lines of the same trace."""
    phases: dict[int, int] = {}
    running = 0
    for index, line in enumerate(trace):
        kind = line.get("kind", "model_turn")
        if kind == "decision":
            phases[index] = running
        elif kind == "model_turn":
            running += sum(1 for call in line["action"]["tool_calls"] if write_like(call))
    return phases


def phase_grouped_calls(session_calls: list[tuple[str | None, dict]]) -> dict[int, list[dict]]:
    """The large model's own tool calls (the scout's are left out) from one trial's pi session(s), grouped by their
    phase: how many of the large model's write-like calls came before each one."""
    grouped: dict[int, list[dict]] = {}
    running = 0
    for provider, call in session_calls:
        if provider == "jeff-first":
            continue
        grouped.setdefault(running, []).append(call)
        if write_like(call):
            running += 1
    return grouped


def step_target(kind: str, tool_call: dict) -> str:
    """The one thing a scout step of this kind is about; see the module docstring's "target of a step"."""
    args = tool_call.get("arguments", {})
    command = args.get("command")
    if kind in ("read", "peek"):
        if isinstance(args.get("path"), str):
            return args["path"]
        if not isinstance(command, str):
            raise ValueError(f"a {kind} step has neither a path argument nor a command to find a path in: {tool_call}")
        match = ABS_PATH.search(command)
        if not match:
            raise ValueError(f"a {kind} step's command names no absolute path: {command}")
        return match.group(0)
    if kind == "list":
        if not isinstance(args.get("path"), str):
            raise ValueError(f"a list step has no path argument: {tool_call}")
        return args["path"]
    if kind == "toolchain":
        return TOOLCHAIN_TARGET
    if not isinstance(command, str):
        raise ValueError(f"a {kind} step has no command: {tool_call}")
    if kind == "service":
        match = LOCALHOST.search(command)
        if not match:
            raise ValueError(f"a service step's command names no localhost:<port>: {command}")
        return match.group(0)
    if kind in ("docs", "install"):
        for pattern in (APT_INSTALL, PIP_INSTALL):
            match = pattern.search(command)
            if match:
                return match.group(1)
        raise ValueError(f"a {kind} step's command names no package or program to install: {command}")
    if kind in ("check", "run", "repeat"):
        return CD_PREFIX.sub("", command).strip()
    raise ValueError(f"unknown scout tool kind: {kind}")


def mentions(kind: str, target: str, calls: list[dict]) -> bool:
    """Whether one of `calls` mentions this step's target; see the module docstring's "mentions"."""
    text = json.dumps(calls)
    if kind == "toolchain":
        return any(marker in text for marker in TOOLCHAIN_MARKERS)
    needle = PurePosixPath(target).name if kind in PATH_KINDS else target
    return needle in text


def value(teacher_trial: Path, base_trial: Path) -> dict:
    session_calls = read_session_calls(teacher_trial)
    tool_calls = len(session_calls)
    if tool_calls == 0:
        raise ValueError(f"{teacher_trial} has no tool calls in its pi session(s)")
    scout_calls = sum(1 for provider, _ in session_calls if provider == "jeff-first")

    trace = read_trace(teacher_trial)
    turns = [line for line in trace if line.get("kind", "model_turn") == "model_turn"]
    if not turns:
        raise ValueError(f"{teacher_trial} has no model_turn trace lines")
    median_turn_seconds = statistics.median(line["timings_ms"]["model"] / 1000 for line in turns)

    decisions = [line for line in trace if line.get("kind") == "decision"]
    step_count = sum(1 for line in decisions if line["action"]["kind"] == "step")
    scout_decision_seconds = sum(line["timings_ms"]["chooser"] for line in decisions) / 1000
    estimated_seconds_avoided = step_count * median_turn_seconds

    phases = step_phases(trace)
    base_phase_calls = phase_grouped_calls(read_session_calls(base_trial))

    per_tool: dict[str, dict[str, int]] = {}

    def bucket(kind: str) -> dict[str, int]:
        return per_tool.setdefault(kind, {"offered": 0, "chosen": 0, "used": 0, "anticipated": 0})

    for index, line in enumerate(trace):
        if line.get("kind") != "decision" or not line["levels"]:
            continue
        first = line["levels"][0]
        if first["level"] != "tool":
            raise ValueError(f"{teacher_trial}: decision {line['decision']}'s first level is not a tool question")
        for option in first["options"]:
            if option["id"] in KNOWN_KINDS:
                bucket(option["id"])["offered"] += 1
        kind = first["chosen"]
        if kind not in KNOWN_KINDS or line["action"]["kind"] != "step":
            continue
        bucket(kind)["chosen"] += 1
        target = step_target(kind, line["action"]["tool_call"])
        next_turn = next((l for l in trace[index + 1 :] if l.get("kind", "model_turn") == "model_turn"), None)
        if next_turn is not None and mentions(kind, target, next_turn["action"]["tool_calls"]):
            bucket(kind)["used"] += 1
        if mentions(kind, target, base_phase_calls.get(phases[index], [])):
            bucket(kind)["anticipated"] += 1

    return {
        "tool_calls": tool_calls,
        "scout_calls": scout_calls,
        "scout_share": scout_calls / tool_calls,
        "median_turn_seconds": median_turn_seconds,
        "estimated_seconds_avoided": estimated_seconds_avoided,
        "scout_decision_seconds": scout_decision_seconds,
        "per_tool": per_tool,
    }


def render(tasks: list[str], results: dict[str, dict]) -> str:
    lines = [
        "# Scout value\n",
        "| Task | Tool calls | Scout calls | Scout share | Median turn s | Qwen seconds avoided | Scout decision s |",
        "|---|---|---|---|---|---|---|",
    ]
    totals = {"tool_calls": 0, "scout_calls": 0, "estimated_seconds_avoided": 0.0, "scout_decision_seconds": 0.0}
    for task in tasks:
        r = results[task]
        lines.append(
            f"| {task} | {r['tool_calls']} | {r['scout_calls']} | {r['scout_share']:.0%} "
            f"| {r['median_turn_seconds']:.1f} | {r['estimated_seconds_avoided']:.0f} | {r['scout_decision_seconds']:.0f} |"
        )
        for key in totals:
            totals[key] += r[key]
    share = totals["scout_calls"] / totals["tool_calls"] if totals["tool_calls"] else 0.0
    lines.append(
        f"| **Total** | {totals['tool_calls']} | {totals['scout_calls']} | {share:.0%} | "
        f"| {totals['estimated_seconds_avoided']:.0f} | {totals['scout_decision_seconds']:.0f} |"
    )
    lines.append("")
    lines.append("| Task | Tool | Offered | Chosen | Used | Anticipated |")
    lines.append("|---|---|---|---|---|---|")
    per_tool_totals: dict[str, dict[str, int]] = {}
    for task in tasks:
        for kind, counts in sorted(results[task]["per_tool"].items()):
            lines.append(f"| {task} | {kind} | {counts['offered']} | {counts['chosen']} | {counts['used']} | {counts['anticipated']} |")
            running = per_tool_totals.setdefault(kind, {"offered": 0, "chosen": 0, "used": 0, "anticipated": 0})
            for key in running:
                running[key] += counts[key]
    for kind, counts in sorted(per_tool_totals.items()):
        lines.append(f"| **Total** | {kind} | {counts['offered']} | {counts['chosen']} | {counts['used']} | {counts['anticipated']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    base_runs, teacher_runs = Path(sys.argv[1]), Path(sys.argv[2])
    tasks = sys.argv[3:]
    results = {}
    for task in tasks:
        teacher_trial = latest_trial(teacher_runs, task)
        base_trial = latest_trial(base_runs, task)
        results[task] = value(teacher_trial, base_trial)
    report = render(tasks, results)
    print(report)


if __name__ == "__main__":
    main()
