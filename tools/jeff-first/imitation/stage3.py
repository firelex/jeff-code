"""Stage 3 driver: Jeff training rows from our own record-mode collection runs (see record_rows.py for one trial).

Input: one or more run folders. A run holds collection streams (one per machine or GPU), each with rounds of Harbor
jobs; a trial folder is `<run>/<stream>/round-N/<task>-<time>/<task>__<id>/` (or `roundN`) with `config.json`, `result.json`
(missing while the trial runs or after the collection was stopped), `agent/jeff-first-trace.jsonl` and
`agent/pi/sessions/*.jsonl`.

Each trial is checked in this order:
1. Its task. A Terminal-Bench 2.0 task (config.json `task.path`, a bare name): an evaluation task (training-tasks.json
   "excluded_evaluation") is an error, whatever the build; `make-doom-for-mips` (a near-identical twin of an evaluation
   task) and any other task outside "training" are skipped and counted. A Harbor hub dataset task (config.json
   `task.name` "<org>/<task>" and `task.source` "<org>/<dataset>") has the task id "<org>/<dataset>:<task>" (as
   run_phase0.sh names it in the trace); it needs `--task-sets` (results/imitation/task-sets.json) and must be on its
   dataset's "training" side: held-out, excluded and unknown tasks are errors, whatever the build.
2. Its build: only trials whose scout tarball (config.json `agent.kwargs.tarball`) is one of the builds given
   (`--build`, the builds whose menus are current; CURRENT_TARBALL is the build of the first collection) are used;
   others are skipped and counted by tarball.
3. Its trace. No trace and no result.json: the trial is still running or was stopped before Qwen's first turn
   (skipped, counted). No trace and a result.json naming an exception: the agent failed before its first turn (skipped,
   counted by exception type). No trace and no exception is an error.
4. It must have run in record mode (config env JEFF_FIRST_MODE), every trace line must be a record or record_step
   line of this task, and every line's `driver_build` must equal the config's JEFF_FIRST_DRIVER_BUILD; anything else
   is an error. Traces of schema "jeff-first-trace/5" (record_step lines) give stint rows: several decisions in one
   coding-model turn (see record_rows.py); schema 4 traces give one decision per turn.
5. A trial whose session file shows a call to a tool other than bash (Qwen on NVFP4 sometimes names a tool that does
   not exist) is skipped and counted; record_rows would raise on it.

A trial without result.json, or one Harbor stopped at its time limit, is "cut" (record_rows.trial_cut): only its
complete lines are used. Rows are record_rows' rows with stage 3, quality "exact", source "own", plus `machine`: the
trace's driver build, e.g. "qwen3.8-27b-fp8@casdgx01-gpu5" (model format after "qwen3.8-27b-", machine after "@").

Usage (from tools/jeff-first):
    uv run python -m imitation.stage3 RUN [RUN ...] --tasks ../../results/imitation/training-tasks.json \\
        [--task-sets ../../results/imitation/task-sets.json] --build jeff-pi-scout-4abde3ece.tgz [--build ...] --rows stage3-rows.jsonl --summary stage3-summary.json --stats ../../results/imitation/stage3-stats.md
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from imitation.record_rows import TRACE_NAME, _json_lines, read_trace, record_rows, row_lines, trial_cut
from task_source import TaskSets, load_task_sets

CURRENT_TARBALL = "jeff-pi-scout-4abde3ece.tgz"
EVALUATION_TWIN = "make-doom-for-mips"
SOURCE = "own"
OTHER_TOOL = "a reply calls a tool other than bash"
BUILD = re.compile(r"^qwen3\.8-27b-(fp8|nvfp4)@(.+)$")


@dataclass(frozen=True)
class Tasks:
    training: frozenset[str]
    evaluation: frozenset[str]
    # Harbor hub datasets (task-sets.json); None when no --task-sets was given.
    hub: TaskSets | None = None


def read_tasks(path: Path, task_sets: Path | None = None) -> Tasks:
    data = json.loads(path.read_text())
    hub = load_task_sets(task_sets, None) if task_sets is not None else None
    tasks = Tasks(frozenset(data["training"]), frozenset(data["excluded_evaluation"]), hub)
    if tasks.training & tasks.evaluation or EVALUATION_TWIN in tasks.training:
        raise ValueError(f"{path}: the training tasks include an evaluation task or {EVALUATION_TWIN}")
    return tasks


def task_id(trial: Path, config: dict) -> str:
    """A bare Terminal-Bench 2.0 name, or "<org>/<dataset>:<task>" for a Harbor hub dataset task."""
    task = config["task"]
    if task.get("path") is not None:
        return task["path"]
    org, name = task["name"].split("/", 1)
    source = task["source"]
    if source.split("/", 1)[0] != org:
        raise ValueError(f"{trial}: task package {task['name']} is not of its dataset {source}'s organisation")
    return f"{source}:{name}"


def model_format(build: str) -> str:
    match = BUILD.match(build)
    if match is None:
        raise ValueError(f"the driver build {build!r} names no known model format (qwen3.8-27b-fp8@... or qwen3.8-27b-nvfp4@...)")
    return match.group(1)


@dataclass
class Conversion:
    rows: list[dict] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    trials: list[dict] = field(default_factory=list)
    # Trace lines read past (record_rows.SKIPPED_KINDS), by kind.
    skipped_lines: Counter = field(default_factory=Counter)
    builds: list[str] = field(default_factory=list)


def calls_other_tool(session: Path, cut: str | None) -> bool:
    """Whether the coding model called a tool other than bash in this session file (record_rows raises on such a
    session; Qwen on NVFP4 sometimes names a tool that does not exist, e.g. "bbyte", and pi answers with an error)."""
    entries, _ = _json_lines(session, cut)
    return any(
        part.get("type") == "toolCall" and part.get("name") != "bash"
        for entry in entries
        if entry.get("type") == "message" and entry["message"].get("role") == "assistant"
        for part in entry["message"]["content"]
    )


def find_trials(run: Path) -> list[Path]:
    # Round folders are round-N (imitation_stream.sh) or roundN (collect_rounds.sh, the xhigh collection).
    trials = sorted(config.parent for config in run.glob("*/round*/*/*/config.json"))
    if not trials:
        raise ValueError(f"{run}: no trial folders (<stream>/round-N/<job>/<trial>/config.json)")
    return trials


def convert_trial(trial: Path, tasks: Tasks, builds: frozenset[str], conversion: Conversion) -> None:
    config = json.loads((trial / "config.json").read_text())
    task = task_id(trial, config)
    if ":" in task:
        if tasks.hub is None:
            raise ValueError(f"{trial}: a Harbor hub dataset task ({task}); give --task-sets to convert it")
        side = tasks.hub.side(task)
        if side != "training":
            raise ValueError(f"{trial}: a session on {task}, which is {side.replace('_', '-')} in task-sets.json; only training tasks may be collected")
    elif task in tasks.evaluation:
        raise ValueError(f"{trial}: a session on the evaluation task {task}; evaluation tasks must never be collected")
    if task == EVALUATION_TWIN:
        conversion.skipped[f"{EVALUATION_TWIN} (twin of an evaluation task)"] += 1
        return
    if ":" not in task and task not in tasks.training:
        conversion.skipped["not a training task"] += 1
        return
    tarball = Path(config["agent"]["kwargs"]["tarball"]).name
    if tarball not in builds:
        conversion.skipped[f"older build {tarball}"] += 1
        return
    if not (trial / "agent" / TRACE_NAME).exists():
        if not (trial / "result.json").exists():
            conversion.skipped["no trace yet: no result.json and no trace (still running, or stopped before its first turn)"] += 1
            return
        info = json.loads((trial / "result.json").read_text())["exception_info"]
        if info is None:
            raise ValueError(f"{trial}: the trial finished without a trace and without an exception")
        conversion.skipped[f"no trace: the agent failed before its first turn ({info['exception_type']})"] += 1
        return
    env = config["agent"]["env"]
    if env["JEFF_FIRST_MODE"] != "record":
        raise ValueError(f"{trial}: ran in {env['JEFF_FIRST_MODE']!r} mode, not record mode")
    build = env["JEFF_FIRST_DRIVER_BUILD"]
    model_format(build)
    all_lines, notes = read_trace(trial)
    cut = trial_cut(trial)
    lines, skipped = row_lines(all_lines, str(trial))
    for line in all_lines:
        if line["task_id"] != task:
            raise ValueError(f"{trial}: a trace line of kind {line['kind']!r} for task {line['task_id']!r}, not for {task}")
    conversion.skipped_lines.update(skipped)
    for line in lines:
        if line["driver_build"] != build:
            raise ValueError(f"{trial}: a trace line's driver build {line['driver_build']!r} is not the config's {build!r}")
    if not lines:
        conversion.skipped["empty trace (no complete record line)"] += 1
        return
    sessions = sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if any(calls_other_tool(session, cut) for session in sessions):
        conversion.skipped[OTHER_TOOL] += 1
        return
    rows, row_notes = record_rows(lines, sessions, cut=cut is not None, source=SOURCE)
    conversion.rows.extend({**asdict(row), "machine": build} for row in rows)
    conversion.trials.append(
        {
            "trial": str(trial),
            "task": task,
            "machine": build,
            "cut": cut,
            "sessions": sorted({line["session_id"] for line in lines}),
            "rows": len(rows),
            "notes": notes + row_notes,
        }
    )


def convert_runs(runs: list[Path], tasks: Tasks, builds: frozenset[str]) -> Conversion:
    """Every trial of the runs; `builds`: the scout tarball names whose trials are converted."""
    if not builds:
        raise ValueError("no build given: name the scout tarballs whose trials to convert")
    conversion = Conversion(builds=sorted(builds))
    for run in runs:
        for trial in find_trials(run):
            convert_trial(trial, tasks, builds, conversion)
    return conversion


def _stint_decisions(rows: list[dict]) -> int:
    """Decisions taken inside a coding-model turn: after an earlier decision of the same session and turn."""
    points = sorted({(row["session"], row["turn"], row["decision"]) for row in rows})
    return sum(1 for before, after in zip(points, points[1:]) if before[:2] == after[:2])


def _counts(rows: list[dict]) -> dict:
    tool_rows = [row for row in rows if row["level"] == "tool" and row["label"] != "show_more"]
    labels = Counter(row["label"] for row in tool_rows)
    return {
        "sessions": len({row["session"] for row in rows}),
        "rows": len(rows),
        "decisions": len(tool_rows),
        "stint_decisions": _stint_decisions(rows),
        "hand_over_share": round(labels["hand_over"] / len(tool_rows), 4) if tool_rows else None,
        "tool_labels": dict(labels.most_common()),
    }


def summarize(conversion: Conversion) -> dict:
    rows = conversion.rows
    by_machine: dict[str, list[dict]] = defaultdict(list)
    by_format: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_machine[row["machine"]].append(row)
        by_format[model_format(row["machine"])].append(row)
    notes = Counter()
    for trial in conversion.trials:
        for note in trial["notes"]:
            notes[re.sub(r"^.*?(turn \d+: |line \d+: )", "", note).split(";")[0].split(" (")[0]] += 1
    return {
        **_counts(rows),
        "builds": conversion.builds,
        "trials_converted": len(conversion.trials),
        "skipped_trace_lines": dict(sorted(conversion.skipped_lines.items())),
        "trials_cut": dict(Counter(trial["cut"] for trial in conversion.trials if trial["cut"]).most_common()),
        "skipped": dict(conversion.skipped.most_common()),
        "rows_by_level_and_page": {f"{level} page {page}": count for (level, page), count in sorted(Counter((r["level"], r["page"]) for r in rows).items())},
        "argument_rows_by_tool": dict(Counter(r["label"].rsplit("-", 1)[0] for r in rows if r["level"] == "argument").most_common()),
        "by_format": {name: _counts(group) for name, group in sorted(by_format.items())},
        "by_machine": {name: _counts(group) for name, group in sorted(by_machine.items())},
        "sessions_by_task": dict(sorted(Counter(row["task"] for row in rows if row["decision"] == 0 and row["level"] == "tool" and row["page"] == 1).items())),
        "notes": dict(notes.most_common()),
    }


def _share(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def stats_markdown(summary: dict) -> str:
    lines = [
        "# Stage 3 conversion statistics (own record-mode sessions)",
        "",
        "Builds: " + ", ".join(f"`{build}`" for build in summary["builds"]) + ". Rows: stage 3, quality \"exact\", source \"{SOURCE}\", each with its `machine`.",
        "",
        "## Trials",
        "",
        f"- Converted: {summary['trials_converted']} (cut: "
        + (", ".join(f"{reason} {count}" for reason, count in summary["trials_cut"].items()) or "none")
        + ").",
        "- Skipped: " + (", ".join(f"{reason} {count}" for reason, count in summary["skipped"].items()) or "none") + ".",
        "- Trace lines read past (no menu): "
        + (", ".join(f"{kind} {count}" for kind, count in summary["skipped_trace_lines"].items()) or "none")
        + ".",
        "- Turns given no row: " + (", ".join(f"{reason} {count}" for reason, count in summary["notes"].items()) or "none") + ".",
        "",
        "## Rows and decisions",
        "",
        f"- Sessions with rows: {summary['sessions']}; rows: {summary['rows']}; decisions: {summary['decisions']} "
        f"(stint decisions: {summary['stint_decisions']}, taken after an earlier decision of the same turn); "
        f"hand-over share of decisions: {_share(summary['hand_over_share'])}.",
        "- Rows by level and page: " + ", ".join(f"{key} {count}" for key, count in summary["rows_by_level_and_page"].items()) + ".",
        "- Argument rows by tool: " + (", ".join(f"{key} {count}" for key, count in summary["argument_rows_by_tool"].items()) or "none") + ".",
        "",
        "Labels at the tool level (one per decision):",
        "",
        "| Label | Decisions | Share |",
        "|---|---|---|",
    ]
    for label, count in summary["tool_labels"].items():
        lines.append(f"| {label} | {count} | {_share(count / summary['decisions'])} |")
    for title, key, name in (("By model format", "by_format", "Format"), ("By machine", "by_machine", "Machine")):
        lines += ["", f"## {title}", "", f"| {name} | Sessions | Rows | Decisions | Hand over | Labels |", "|---|---|---|---|---|---|"]
        for group, counts in summary[key].items():
            labels = ", ".join(f"{label} {count}" for label, count in counts["tool_labels"].items())
            lines.append(f"| {group} | {counts['sessions']} | {counts['rows']} | {counts['decisions']} | {_share(counts['hand_over_share'])} | {labels} |")
    lines += ["", "## Sessions by task", "", ", ".join(f"{task} {count}" for task, count in summary["sessions_by_task"].items()), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="Run folders (each holds <stream>/round-N/<job>/<trial>/)")
    parser.add_argument("--tasks", type=Path, required=True, help="results/imitation/training-tasks.json")
    parser.add_argument("--task-sets", type=Path, help="results/imitation/task-sets.json, for Harbor hub dataset tasks")
    parser.add_argument("--build", action="append", required=True, help=f"A scout tarball name whose trials are converted (repeatable), e.g. {CURRENT_TARBALL}")
    parser.add_argument("--rows", type=Path, required=True, help="Output: the rows, one JSON object per line")
    parser.add_argument("--summary", type=Path, required=True, help="Output: the counts as JSON")
    parser.add_argument("--stats", type=Path, required=True, help="Output: the counts as a Markdown statistics file")
    args = parser.parse_args(argv)
    builds = frozenset(args.build)
    conversion = convert_runs(args.runs, read_tasks(args.tasks, args.task_sets), builds)
    summary = summarize(conversion)
    for path in (args.rows, args.summary, args.stats):
        path.parent.mkdir(parents=True, exist_ok=True)
    # ASCII escapes, as Row.to_json writes them: a lone surrogate from a cut line is written as \udXXX, not raised on.
    args.rows.write_text("".join(json.dumps(row) + "\n" for row in conversion.rows))
    args.summary.write_text(json.dumps({**summary, "trials": conversion.trials}, indent=1) + "\n")
    args.stats.write_text(stats_markdown(summary))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
