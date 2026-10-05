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

- A tool call: one entry of type "toolCall" inside one assistant message's content, in Jeff-Code's own session log
  (agent/pi/sessions/*.jsonl). It has a "name" (which Jeff-Code tool it called, e.g. "read", "bash", "write") and
  "arguments". A scout step shows up as its own, separate assistant message whose "provider" is "jeff-first"; every
  other assistant message's tool calls are Qwen's.

- The scout's share of tool calls: of every tool call in the teacher run's Jeff-Code session(s), the fraction made by the
  scout (provider "jeff-first") rather than Qwen.

- The scout's decision time: how many seconds the scout itself spent choosing, across every decision in the
  teacher run - the sum of each decision's "timings_ms.chooser" (how long picking an option took), turned from
  milliseconds into seconds.

- The large model's typical turn length: the median of "timings_ms.model" (how long one Qwen turn took to produce
  its reply) across the teacher run's "model_turn" trace lines.

- Estimated seconds of Qwen time avoided: each scout step stands in for one Qwen turn that did not have to happen.
  We do not know how long that particular avoided turn would have taken, so we estimate it as the typical turn
  length above, multiplied by how many steps the scout took.

- A scout "step": a decision whose action is "step" - the scout chose a tool and an argument, and Jeff-Code ran that tool
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
  did something to the same thing. It is read not from the tool call itself (bash commands are built from fixed
  shell scripts and parsing them back apart is unreliable: for example an "install" step's command has several
  package-manager flags and no reliable way to tell a flag like "-y" from the package name) but from the English
  description of the option the scout picked - the option whose id matches "chosen" on the *last* entry of the
  decision's "levels" list (the last question asked before the step: which specific argument to use). Descriptions
  are fixed sentences Jeff-Code always writes the same way, so each one is matched against a fixed pattern to pull out
  its target, or, for a few kinds, a fixed set of search strings instead of one name:
    - "Read the file PATH", "Read lines A to B of PATH", "Look at the data in PATH", "Show the type of every file
      in PATH", "List the folder PATH", "Show the last 20 lines of PATH", "Find the files under PATH" -> the last
      path segment of PATH (so "/app/a.py" becomes "a.py": the large model rarely repeats a full path verbatim).
    - "Search the project for the text "P"" -> P. "Find files matching PATTERN" -> PATTERN with any leading
      "**/" removed. "Find files named NAME" -> NAME.
    - "Check which tools and languages are installed: ..." and "Check which installed packages match: ..." ->
      no one name stands for a toolchain probe (it always checks a whole list of programs and modules at once);
      instead, the search strings are the patterns Qwen itself uses to probe the toolchain: "which ",
      "command -v", "--version", "import ", "/etc/os-release".
    - "Search the whole filesystem for a program named X" -> X.
    - "Request http://localhost:PORT/ once" -> the two search strings "localhost:PORT" and ":PORT" (a command
      that names just the port, such as `curl :8080/health`, still counts).
    - "Check the SERVICE configuration" -> SERVICE.
    - "Show running processes and listening ports" -> no one name stands for this either; the search strings are
      "ps ", "ss -", "netstat" (the shapes Qwen itself uses to list processes or ports).
    - "Show the README of the npm package P", "List what the npm package P exports" -> P. "List what the Python
      module M provides" -> M. "Show the help of PROG" -> PROG.
    - "Install PROGRAM with apt (package PKG)" -> PROGRAM (not PKG: apt's package name is often not what Qwen
      later types on a command line, e.g. "pip install opencv-python-headless" installs the program "cv2").
      "Install the Python package PKG with pip" -> PKG.
    - "Run: COMMAND" and "Run the last shell command again: COMMAND" -> COMMAND with any leading "cd FOLDER &&"
      removed.
  A description matching none of these is an error naming the description, not a guess.

- A later tool call "mentions" a step's target when the JSON text of the later tool call(s) contains any one of
  the target's search strings (usually just one: the name above; for a toolchain probe or a processes-and-ports
  step, one of the fixed search strings listed above instead; for a service's port, either form of "localhost:PORT").

- Used: a step is "used" when the teacher run's very next "model_turn" trace line after it (the large model's next
  turn, once the scout hands over) makes a tool call that mentions the step's target.

- A tool call's phase: how many write-like tool calls by the large model (Jeff-Code's "write" or "edit" tool, or a bash
  command whose text contains "cat >" or "tee ") came before it, in order, within one run's Jeff-Code session(s). The
  scout's own tool calls never count as write-like (the scout is never given write or edit).

- Anticipated: a step is "anticipated" when, in the base run's Jeff-Code session(s), some tool call by the large model
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
from typing import Callable, NamedTuple

from gate0_report import latest_trial

KNOWN_KINDS = ("read", "peek", "list", "search", "find", "toolchain", "service", "docs", "check", "run", "install", "repeat")
TOOLCHAIN_MARKERS = ("which ", "command -v", "--version", "import ", "/etc/os-release")
PROCESS_MARKERS = ("ps ", "ss -", "netstat")
WRITE_IN_BASH = ("cat >", "tee ")

CD_PREFIX = re.compile(r"^\s*cd\s+\S+\s*&&\s*")


class StepTarget(NamedTuple):
    """What a scout step is about, as one or more search strings. See the module docstring's "target of a step"
    and "mentions".

    For most kinds, `needles` are literal substrings: a later tool call "mentions" the target when the raw text of
    one of its own string arguments contains one of them.

    For a toolchain probe, a processes-and-ports step, and the "show every file's type" step, no single name
    stands for the whole step - `command_word_markers` is then true, and `needles` are command-line shapes (such
    as "ps " or "import ") that Qwen itself would type on a command line. These are searched only inside a later
    bash call's "command" text, and only as that call's own word or flag - not as text embedded in a longer word
    (so a command mentioning "steps" is not mistaken for a mention of the "ps " marker)."""

    needles: tuple[str, ...]
    command_word_markers: bool = False


def _path(match: re.Match[str]) -> StepTarget:
    return StepTarget((PurePosixPath(match.group(1)).name,))


def _name(match: re.Match[str]) -> StepTarget:
    return StepTarget((match.group(1),))


def _find_name(match: re.Match[str]) -> StepTarget:
    return StepTarget((match.group(1).removeprefix("**/"),))


def _file_name(match: re.Match[str]) -> StepTarget:
    return StepTarget((match.group(1),))


def _markers(*needles: str) -> Callable[[re.Match[str]], StepTarget]:
    def build(_match: re.Match[str]) -> StepTarget:
        return StepTarget(needles, command_word_markers=True)

    return build


def _localhost(match: re.Match[str]) -> StepTarget:
    port_token = match.group(1)
    port = port_token.split(":", 1)[1]
    return StepTarget((port_token, f":{port}"))


def _command(match: re.Match[str]) -> StepTarget:
    return StepTarget((CD_PREFIX.sub("", match.group(1)).strip(),))


# Each entry is a fixed sentence Jeff-Code's own option descriptions use (see lists.ts, qwen-tools.ts, probes.ts), matched
# whole against the description, paired with how to build this step's target from the match. re.DOTALL makes "."
# match a newline too, so a "Run" command that is itself a multi-line heredoc is still matched whole, instead of
# raising "no target pattern matches" partway through it.
DESCRIPTION_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], StepTarget]]] = [
    (re.compile(r"^Read the file (.+)$", re.DOTALL), _path),
    (re.compile(r"^Read lines \d+ to \d+ of (.+)$", re.DOTALL), _path),
    (re.compile(r"^Look at the data in (.+)$", re.DOTALL), _path),
    # No one name stands for this step either (it always checks every file in the folder at once); the marker is
    # the shape Qwen itself uses to classify a file from a command line, same reasoning as the processes markers.
    (re.compile(r"^Show the type of every file in (.+)$", re.DOTALL), _markers("file ")),
    (re.compile(r"^List the folder (.+)$", re.DOTALL), _path),
    (re.compile(r"^Show the last 20 lines of (.+)$", re.DOTALL), _path),
    (re.compile(r'^Search the project for the text "(.*)"$', re.DOTALL), _name),
    (re.compile(r"^Find files matching (.+)$", re.DOTALL), _find_name),
    (re.compile(r"^Find files named (.+)$", re.DOTALL), _file_name),
    (re.compile(r"^Find the files under (.+)$", re.DOTALL), _path),
    (re.compile(r"^Check which tools and languages are installed:.*$", re.DOTALL), _markers(*TOOLCHAIN_MARKERS)),
    (re.compile(r"^Check which installed packages match:.*$", re.DOTALL), _markers(*TOOLCHAIN_MARKERS)),
    (re.compile(r"^Search the whole filesystem for a program named (.+)$", re.DOTALL), _name),
    (re.compile(r"^Request http://(localhost:\d+)/ once$", re.DOTALL), _localhost),
    (re.compile(r"^Check the (.+) configuration$", re.DOTALL), _name),
    (re.compile(r"^Show running processes and listening ports$", re.DOTALL), _markers(*PROCESS_MARKERS)),
    (re.compile(r"^Show the README of the npm package (.+)$", re.DOTALL), _name),
    (re.compile(r"^List what the npm package (.+) exports$", re.DOTALL), _name),
    (re.compile(r"^List what the Python module (.+) provides$", re.DOTALL), _name),
    (re.compile(r"^Show the help of (.+)$", re.DOTALL), _name),
    (re.compile(r"^Install (.+) with apt \(package .+\)$", re.DOTALL), _name),
    (re.compile(r"^Install the Python package (.+) with pip$", re.DOTALL), _name),
    (re.compile(r"^Run: (.+)$", re.DOTALL), _command),
    (re.compile(r"^Run the last shell command again: (.+)$", re.DOTALL), _command),
]


def read_trace(trial: Path) -> list[dict]:
    path = trial / "agent" / "jeff-first-trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_session_calls(trial: Path) -> list[tuple[str | None, dict]]:
    """Every tool call in this trial's Jeff-Code session(s), in file and line order, paired with the "provider" of the
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
    """Whether this is a write-like call by the large model: Jeff-Code's write or edit tool, or a bash command whose text
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
    """The large model's own tool calls (the scout's are left out) from one trial's Jeff-Code session(s), grouped by their
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


def step_target(description: str) -> StepTarget:
    """The one thing a scout step is about, read from the English description of the argument option the scout
    picked; see the module docstring's "target of a step"."""
    for pattern, build in DESCRIPTION_PATTERNS:
        match = pattern.fullmatch(description)
        if match:
            return build(match)
    raise ValueError(f"no target pattern matches the chosen option's description: {description!r}")


def chosen_description(decision: dict) -> str:
    """The description of the option chosen at a step decision's last question (its last entry in "levels")."""
    last = decision["levels"][-1]
    for option in last["options"]:
        if option["id"] == last["chosen"]:
            return option["description"]
    raise ValueError(f"decision {decision['decision']}'s chosen option {last['chosen']!r} is not among its last level's options")


# The key holding the text a write or edit call sets a file to, left out of the search for a plain (non-marker)
# target: that text is whatever the large model is writing INTO a file, not a command it is typing, and a plain
# needle such as a bare file name or symbol would otherwise match it by coincidence almost every time.
WRITTEN_CONTENT_KEY = {"write": "content", "edit": "edits"}

_COMMAND_WORD_LEFT_BOUNDARY = r"""(?:^|[\s;&|("'])"""


def _command_word_regex(marker: str) -> re.Pattern[str]:
    """A command-shape marker such as "ps " or "ss -" counts only when it is its own word or flag in a shell
    command, not embedded in a longer word - so a command containing "steps " is not mistaken for a mention of the
    "ps " marker. When the marker ends in a space, that space becomes an explicit "followed by whitespace, or
    nothing, to the end of the command" check, so the marker still counts when it is the last thing typed; a
    marker that does not end in a space (a flag such as "ss -") needs no such check, since what follows it is not
    part of the marker's own shape."""
    if marker.endswith(" "):
        return re.compile(rf"{_COMMAND_WORD_LEFT_BOUNDARY}{re.escape(marker[:-1])}(?:\s|$)")
    return re.compile(rf"{_COMMAND_WORD_LEFT_BOUNDARY}{re.escape(marker)}")


def _raw_strings(call: dict) -> list[str]:
    """The string values among one tool call's own arguments, walking into nested lists and objects, but leaving
    out the file content a write call sets, or the edits an edit call makes (see WRITTEN_CONTENT_KEY): that text
    is what the large model is writing into a file, not a command it typed or a path or name it is operating on,
    so it would otherwise match almost any search by coincidence."""
    skip_key = WRITTEN_CONTENT_KEY.get(call.get("name"))
    strings: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    for key, value in call.get("arguments", {}).items():
        if key == skip_key:
            continue
        walk(value)
    return strings


def mentions(target: StepTarget, calls: list[dict]) -> bool:
    """Whether one of `calls` mentions this step's target; see the module docstring's "mentions". A command-word
    target (see StepTarget) is searched only inside a bash call's own "command" text, each needle matched as its
    own word or flag there. A plain target's needles are literal substrings, searched among the raw string values
    of each call's own arguments (not the file content a write or edit call is setting - see WRITTEN_CONTENT_KEY)."""
    if target.command_word_markers:
        commands = [
            call["arguments"]["command"]
            for call in calls
            if call.get("name") == "bash" and isinstance(call.get("arguments", {}).get("command"), str)
        ]
        return any(_command_word_regex(needle).search(command) for needle in target.needles for command in commands)
    strings = [one_string for call in calls for one_string in _raw_strings(call)]
    return any(needle in one_string for needle in target.needles for one_string in strings)


def value(teacher_trial: Path, base_trial: Path) -> dict:
    session_calls = read_session_calls(teacher_trial)
    tool_calls = len(session_calls)
    if tool_calls == 0:
        raise ValueError(f"{teacher_trial} has no tool calls in its Jeff-Code session(s)")
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
        target = step_target(chosen_description(line))
        next_turn = next((l for l in trace[index + 1 :] if l.get("kind", "model_turn") == "model_turn"), None)
        if next_turn is not None and mentions(target, next_turn["action"]["tool_calls"]):
            bucket(kind)["used"] += 1
        if mentions(target, base_phase_calls.get(phases[index], [])):
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
