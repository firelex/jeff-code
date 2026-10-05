"""Features of every finished session of tonight's evaluation at fixed times after the agent started (checkpoints), for
the offline cut-off estimate (cutoff_estimate.py). Standard library only, so it runs on the evaluation hosts as is:

    ssh HOST python3 - EVAL_DIR < cutoff_features.py > features.jsonl

Reads, read-only, every session folder EVAL_DIR/runs/ARM/TASK/attemptK/ that finished (meta.json present, not marked
superseded or cut) and prints one JSON line per folder:
- {"folder", "setup_s", "agent_s", "timeout_multiplier", "checkpoints": {"10": {...features...}, ...}} for a session with one
  Harbor trial; a checkpoint is present only if the agent was still running at that time (agent time > checkpoint);
- {"folder", "left_out": reason} for a finished session that cannot be read (no trial, no pi session, ...).
A malformed line in a trace or pi session raises (except a session's last line cut off without its newline, which is
dropped and counted, as eval_units.py does).

The features at time T use only what was written by T: pi session entries (each carries the time it was written) and
trace lines tied to Qwen turns that had ended by T. Qwen turns are the pi assistant messages from the Qwen endpoint
(Jeff's own steps are assistant messages from the provider "jeff-first"); trace qwen_request line k belongs to Qwen turn
k. The trial's agent_timeout_multiplier is passed on (the agent's time limit is the task's timeout times it).
"""

import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

CHECKPOINTS_MIN = (10, 15, 20, 30, 45, 60)
LAST = 10  # "recent" window: the last 10 Qwen turns

TEST_RUN = re.compile(
    r"\bpytest\b|py\.test|-m\s+unittest|\bnosetests\b|\btox\b|\b(?:npm|yarn|pnpm)\s+(?:run\s+)?test|\bgo\s+test|"
    r"\bcargo\s+test|\bmake\s+(?:test|check)\b|\bctest\b|\bmvn\b.*\btest\b|\bgradle\w*\b.*\btest\b|\bjest\b|"
    r"\bvitest\b|\bmocha\b|\brspec\b|\bphpunit\b|\bpython3?\s+\S*test\S*\.py|\b(?:bash|sh)\s+\S*test\S*|"
    r"\./\S*test\S*|runtests\.py"
)
REDIRECT = re.compile(r"(?<![<>&0-9])>{1,2}\s*([^\s|&;<>()]+)")
TEE = re.compile(r"\btee\s+(?:-a\s+)?([^\s|&;<>()]+)")
SED_I = re.compile(r"\bsed\s+-i\b[^\n|;&]*?\s([\w./~-]+)\s*(?=$|\n|;|&&|\|)")
PY_OPEN = re.compile(r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]")
PY_WRITE = re.compile(r"Path\(\s*['\"]([^'\"]+)['\"]\s*\)\.write_(?:text|bytes)")
OTHER_CHANGE = re.compile(r"\b(?:patch|git\s+apply|git\s+checkout\s+--|mv|cp|rm|touch|mkdir|perl\s+-pi)\b")
PASSED = re.compile(r"(\d+) passed")
FAILED = re.compile(r"(\d+) failed")
ERRORS = re.compile(r"(\d+) errors?\b")
UNITTEST_RAN = re.compile(r"Ran (\d+) tests?")
UNITTEST_BAD = re.compile(r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?")
TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
COMPILE = re.compile(
    r"SyntaxError|IndentationError|\berror: |error\[E\d+\]|cannot find symbol|undefined reference|compilation terminated|"
    r"ModuleNotFoundError|ImportError|NameError"
)
TIMEOUT = re.compile(r"timed out|TimeoutError|Timeout", re.IGNORECASE)
NOT_FOUND = re.compile(r"command not found|No such file or directory")
APPROACH = re.compile(
    r"another approach|different approach|alternative approach|try something else|let me try again|"
    r"this (?:isn't|is not|doesn't|does not) work|still (?:failing|not working)|start over",
    re.IGNORECASE,
)
DONE = re.compile(
    r"task is (?:now )?complete|all tests pass|successfully (?:implemented|completed|fixed)|the fix is complete|"
    r"(?:is|are) now (?:working|fixed|passing)",
    re.IGNORECASE,
)


def when(stamp: str) -> float:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def read_jsonl(path: Path) -> tuple[list[dict], bool]:
    """The lines of a JSONL file and whether a last line without its newline (a write cut by a kill) was dropped."""
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    cut = not content.endswith("\n") and bool(lines[-1].strip())
    if cut:
        lines = lines[:-1]
    out = []
    for number, text in enumerate(lines, 1):
        if text.strip():
            try:
                out.append(json.loads(text))
            except json.JSONDecodeError as bad:
                raise ValueError(f"{path}: line {number} is not JSON: {bad}") from bad
    return out, cut


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def written_paths(command: str) -> set[str]:
    paths = set()
    for pattern in (REDIRECT, TEE, SED_I, PY_OPEN, PY_WRITE):
        paths.update(m.group(1) for m in pattern.finditer(command))
    return {p for p in paths if p != "/dev/null" and not p.startswith("&")}


def changes_files(name: str, command: str) -> bool:
    if name in ("edit", "write"):
        return True
    return bool(written_paths(command)) or bool(OTHER_CHANGE.search(command))


def test_counts(output: str) -> tuple[int, int] | None:
    """(passed, not passed) from a pytest or unittest summary in a command's output, or None if there is none."""
    passed = [int(x) for x in PASSED.findall(output)]
    failed = [int(x) for x in FAILED.findall(output)]
    errors = [int(x) for x in ERRORS.findall(output)]
    if passed or failed:
        return (passed[-1] if passed else 0), (failed[-1] if failed else 0) + (errors[-1] if errors else 0)
    ran = UNITTEST_RAN.findall(output)
    if ran:
        bad = UNITTEST_BAD.findall(output)
        n_bad = sum(int(x or 0) for x in bad[-1]) if bad else 0
        return int(ran[-1]) - n_bad, n_bad
    return None


def norm_command(command: str) -> str:
    return re.sub(r"\d+", "0", re.sub(r"\s+", " ", command.strip().lower()))[:300]


def session_events(session: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Qwen turns and Jeff steps (assistant messages, in order, with their tool calls) and tool results, from a pi
    session; also the times of compactions."""
    turns, results, compactions = [], [], []
    for entry in session:
        if entry["type"] == "compaction":
            compactions.append(when(entry["timestamp"]))
        if entry["type"] != "message":
            continue
        m = entry["message"]
        t = when(entry["timestamp"])
        if m["role"] == "assistant":
            calls = [c for c in m["content"] if c.get("type") == "toolCall"]
            usage = m.get("usage") or {}
            turns.append(
                {
                    "t": t,
                    "jeff": m.get("provider") == "jeff-first",
                    "calls": [
                        {"id": c["id"], "name": c["name"], "command": str((c.get("arguments") or {}).get("command", ""))}
                        for c in calls
                    ],
                    "text": text_of(m["content"]),
                    "thinking": "".join(c.get("thinking", "") for c in m["content"] if c.get("type") == "thinking"),
                    "output_tokens": usage.get("output", 0),
                    "input_tokens": usage.get("input", 0) + usage.get("cacheRead", 0),
                }
            )
        elif m["role"] == "toolResult":
            results.append(
                {"t": t, "id": m["toolCallId"], "error": bool(m.get("isError")), "text": text_of(m["content"])}
            )
    return turns, results, compactions


def features_at(cut_t: float, minutes: float, turns: list[dict], results: list[dict], compactions: list[float],
                trace: list[dict]) -> dict:
    qwen = [x for x in turns if not x["jeff"] and x["t"] <= cut_t]
    steps = [x for x in turns if x["jeff"] and x["t"] <= cut_t]
    n = len(qwen)
    by_id = {r["id"]: r for r in results if r["t"] <= cut_t}
    position = {id(x): i for i, x in enumerate(qwen)}
    calls = []  # (qwen turn index or None for a Jeff step, call, result, turn time)
    for x in turns:
        if x["t"] > cut_t:
            continue
        index = position.get(id(x))
        for c in x["calls"]:
            if c["id"] in by_id:
                calls.append((index, c, by_id[c["id"]], x["t"]))
    recent_from = n - LAST
    recent = [(i, c, r, t) for i, c, r, t in calls if i is not None and i >= recent_from]
    outputs = [r for _, _, r, _ in calls]
    commands = [c["command"] for _, c, _, _ in calls if c["command"]]
    tests = [(c, r) for _, c, r, _ in calls if TEST_RUN.search(c["command"])]
    parsed = [p for p in (test_counts(r["text"]) for _, r in tests) if p is not None and sum(p) > 0]
    fracs = [p / (p + q) for p, q in parsed]
    changes = [(c, t) for _, c, _, t in calls if changes_files(c["name"], c["command"])]
    paths = set().union(*(written_paths(c["command"]) for c, _ in changes)) if changes else set()
    requests = [q for q in trace if q["kind"] == "qwen_request" and q["turn"] <= n]
    first = [q for q in requests if q["attempt"] == 1]
    last_outputs = [len(r["text"]) for r in outputs[-5:]]
    return {
        "elapsed_min": minutes,
        "turns": n,
        "turns_per_min": n / minutes,
        "turns_last5min": sum(1 for x in qwen if x["t"] > cut_t - 300),
        "jeff_steps": len(steps),
        "off_share": (sum(q["thinking_level"] == "off" for q in first) / len(first)) if first else 0.0,
        "guard_hits": sum(q["guard"] is not None for q in first),
        "forced_xhigh": sum(q["forced_xhigh"] is not None for q in first),
        "limit_cuts": sum(q["limit_cut"] is not None for q in requests),
        "thinking_chars": sum(q.get("thinking_chars") or 0 for q in requests),
        "output_tokens": sum(x["output_tokens"] for x in qwen),
        "context_tokens": qwen[-1]["input_tokens"] if qwen else 0,
        "compactions": sum(1 for c in compactions if c <= cut_t),
        "commands": len(calls),
        "failed_cmds": sum(r["error"] for r in outputs),
        "fail_share_recent": (sum(r["error"] for _, _, r, _ in recent) / len(recent)) if recent else 0.0,
        "repeat_share": (1 - len(set(commands)) / len(commands)) if commands else 0.0,
        "near_repeat_share": (1 - len({norm_command(c) for c in commands}) / len(commands)) if commands else 0.0,
        "tests": len(tests),
        "tests_recent": sum(1 for _, c, _, _ in recent if TEST_RUN.search(c["command"])),
        "last_test_failed": int(tests[-1][1]["error"]) if tests else 0,
        "test_counts_seen": int(bool(parsed)),
        "last_test_pass_frac": fracs[-1] if fracs else 0.0,
        "test_frac_trend": (fracs[-1] - fracs[0]) if len(fracs) > 1 else 0.0,
        "file_changes": len(changes),
        "file_changes_recent": sum(1 for _, c, _, _ in recent if changes_files(c["name"], c["command"])),
        "written_paths": len(paths),
        "write_chars": sum(len(c["command"]) for c, _ in changes),
        "since_change_min": ((cut_t - changes[-1][1]) / 60) if changes else minutes,
        "tracebacks": sum(bool(TRACEBACK.search(r["text"])) for r in outputs),
        "compile_errors": sum(bool(COMPILE.search(r["text"])) for r in outputs),
        "timeouts": sum(bool(TIMEOUT.search(r["text"])) for r in outputs),
        "not_found": sum(bool(NOT_FOUND.search(r["text"])) for r in outputs),
        "error_share_recent": (
            sum(bool(TRACEBACK.search(r["text"]) or COMPILE.search(r["text"])) for _, _, r, _ in recent) / len(recent)
        ) if recent else 0.0,
        "last_output_chars": (sum(last_outputs) / len(last_outputs)) if last_outputs else 0.0,
        "approach_phrases": sum(len(APPROACH.findall(x["text"] + " " + x["thinking"])) for x in qwen),
        "approach_recent": sum(len(APPROACH.findall(x["text"] + " " + x["thinking"])) for x in qwen[recent_from:]),
        "done_phrases": sum(len(DONE.findall(x["text"])) for x in qwen),
    }


def session_row(folder: Path) -> dict:
    row: dict = {"folder": str(folder)}
    results = sorted(folder.glob("*/*/result.json"))
    if len(results) != 1:
        row["left_out"] = f"{len(results)} result.json files"
        return row
    trial = results[0].parent
    result = json.loads(results[0].read_text())
    span = result.get("agent_execution")
    if not span or not span.get("started_at") or not span.get("finished_at"):
        row["left_out"] = "no agent execution times in result.json"
        return row
    start, end = when(span["started_at"]), when(span["finished_at"])
    row["setup_s"] = start - when(result["started_at"])
    row["agent_s"] = end - start
    config = json.loads((trial / "config.json").read_text())
    row["timeout_multiplier"] = config["agent_timeout_multiplier"]
    sessions = sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if len(sessions) != 1:
        row["left_out"] = f"{len(sessions)} pi session files"
        return row
    session, row["session_cut_last_line"] = read_jsonl(sessions[0])
    trace_path = trial / "agent" / "jeff-first-trace.jsonl"
    if not trace_path.exists():
        row["left_out"] = "no trace"
        return row
    trace, row["trace_cut_last_line"] = read_jsonl(trace_path)
    turns, tool_results, compactions = session_events(session)
    row["checkpoints"] = {
        str(m): features_at(start + m * 60, m, turns, tool_results, compactions, trace)
        for m in CHECKPOINTS_MIN
        if row["agent_s"] > m * 60
    }
    return row


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: cutoff_features.py EVAL_DIR")
    for folder in sorted((Path(sys.argv[1]).expanduser() / "runs").glob("*/*/attempt*")):
        if (folder / "superseded.txt").exists() or (folder / "cut.txt").exists() or not (folder / "meta.json").exists():
            continue
        meta = json.loads((folder / "meta.json").read_text())
        if str(meta["exit"]).startswith("pull-failed"):
            continue
        print(json.dumps(session_row(folder)), flush=True)


if __name__ == "__main__":
    main()
