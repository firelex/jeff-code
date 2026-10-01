"""Menu coverage report for the JeffFirst phase-0 shadow run.

Menu coverage is the share of the large model's turns in which its tool call was one of the ready-made options on
the menu the fork built before that turn. It is the upper limit on what Jeff can take over. The phase-0 gate passes
at 25% or more.

Usage: uv run python coverage_report.py <Harbor jobs folder> <report.md>

The jobs folder holds one folder per Harbor trial, each with agent/jeff-first-trace.jsonl (one line per model turn,
written by the fork) and result.json (Harbor's record of the trial, including the verifier's reward).
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "jeff-first-trace/1"
GATE = 0.25
COUNTED_STOP_REASONS = {"toolUse", "stop"}
POSITION_SPLIT = 5
TRACE_NAME = "jeff-first-trace.jsonl"


def _percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def _group(call: dict[str, Any]) -> str:
    if call["name"] == "bash":
        words = str(call["arguments"].get("command", "")).split()
        return f"bash {words[0]}" if words else "bash"
    return call["name"]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    counted = covered = covered_with_near = several = 0
    errors: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    uncovered: Counter[str] = Counter()
    per_task: dict[str, dict[str, int]] = {}
    by_position = {f"turns 1-{POSITION_SPLIT}": {"counted": 0, "covered": 0}, f"turns {POSITION_SPLIT + 1}+": {"counted": 0, "covered": 0}}
    menu_sizes: list[float] = []
    menu_ms: list[float] = []

    for r in records:
        if r.get("schema") != SCHEMA:
            raise ValueError(f"trace line has schema {r.get('schema')!r}; this report reads {SCHEMA}")
        action = r["action"]
        task = r["task_id"]
        if action["stop_reason"] not in COUNTED_STOP_REASONS:
            errors[task] += 1
            continue
        calls = action["tool_calls"]
        menu_sizes.append(len(r["menu"]))
        menu_ms.append(r["timings_ms"]["menu"])
        task_counts = per_task.setdefault(task, {"counted": 0, "covered": 0})
        position = by_position[f"turns 1-{POSITION_SPLIT}" if r["turn"] <= POSITION_SPLIT else f"turns {POSITION_SPLIT + 1}+"]
        counted += 1
        task_counts["counted"] += 1
        position["counted"] += 1
        if len(calls) > 1:
            several += 1
        for call in calls:
            if call["match"]["kind"] != "exact":
                uncovered[_group(call)] += 1
        if len(calls) != 1:
            continue
        match = calls[0]["match"]
        if match["kind"] in ("exact", "near"):
            covered_with_near += 1
        if match["kind"] == "exact":
            covered += 1
            task_counts["covered"] += 1
            position["covered"] += 1
            kinds = {option["id"]: option["kind"] for option in r["menu"]}
            by_kind[kinds[match["optionId"]]] += 1

    coverage = covered / counted if counted else 0.0
    return {
        "counted_turns": counted,
        "covered": covered,
        "coverage": coverage,
        "covered_with_near": covered_with_near,
        "several_calls": several,
        "errors": dict(errors),
        "covered_by_kind": dict(by_kind),
        "per_task": per_task,
        "by_position": by_position,
        "uncovered_groups": sorted(uncovered.items(), key=lambda item: (-item[1], item[0])),
        "menu_size": (statistics.median(menu_sizes), _percentile(menu_sizes, 0.95)) if menu_sizes else None,
        "menu_ms": (statistics.median(menu_ms), _percentile(menu_ms, 0.95)) if menu_ms else None,
        "gate_passed": counted > 0 and coverage >= GATE,
    }


def load_runs(jobs: Path) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """All trace lines, and each task's verifier reward (None when the trial has no verifier result)."""
    traces = sorted(jobs.rglob(f"agent/{TRACE_NAME}"))
    if not traces:
        raise FileNotFoundError(f"no {TRACE_NAME} files under {jobs}")
    records: list[dict[str, Any]] = []
    trials: dict[str, float | None] = {}
    for trace in traces:
        result_path = trace.parent.parent / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"{result_path} is missing: the trial for {trace} did not finish")
        result = json.loads(result_path.read_text())
        verifier = result["verifier_result"]
        trials[result["task_name"]] = None if verifier is None else verifier["rewards"]["reward"]
        records += [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    return records, trials


def render(s: dict[str, Any], trials: dict[str, float | None]) -> str:
    pct = lambda part, whole: f"{100 * part / whole:.1f}%" if whole else "n/a"  # noqa: E731
    lines = [
        "# Phase-0 menu coverage",
        "",
        f"Gate: {'PASS' if s['gate_passed'] else 'FAIL: improve the menu builder once, then rerun'} (needs {GATE:.0%})",
        "",
        f"Menu coverage: {pct(s['covered'], s['counted_turns'])} ({s['covered']} of {s['counted_turns']} turns)",
        f"Counting near matches too: {pct(s['covered_with_near'], s['counted_turns'])}",
        f"Turns with several tool calls: {s['several_calls']}",
        f"Model errors and aborts (not counted): {sum(s['errors'].values())} {s['errors'] or ''}".rstrip(),
        "",
        "## Covered turns by option kind",
        "",
        *[f"- {kind}: {n}" for kind, n in sorted(s["covered_by_kind"].items())],
        "",
        "## By position in the session",
        "",
        *[f"- {name}: {pct(v['covered'], v['counted'])} ({v['covered']} of {v['counted']})" for name, v in s["by_position"].items()],
        "",
        "## Per task",
        "",
        "| Task | Turns | Covered | Coverage |",
        "|---|---|---|---|",
        *[f"| {t} | {v['counted']} | {v['covered']} | {pct(v['covered'], v['counted'])} |" for t, v in sorted(s["per_task"].items())],
        "",
        "## Most common calls not on the menu",
        "",
        *[f"- {group}: {n}" for group, n in s["uncovered_groups"][:20]],
        "",
    ]
    if s["menu_size"]:
        lines += [f"Menu size: median {s['menu_size'][0]}, 95th percentile {s['menu_size'][1]}"]
        lines += [f"Menu building time: median {s['menu_ms'][0]:.1f} ms, 95th percentile {s['menu_ms'][1]:.1f} ms", ""]
    scored = {t: r for t, r in trials.items() if r is not None}
    lines += [
        "## Task results (check that the shadow wrapper did not change pi's behaviour)",
        "",
        f"Passed: {sum(1 for r in scored.values() if r >= 1)} of {len(scored)} tasks with a verifier result",
    ]
    missing = sorted(t for t, r in trials.items() if r is None)
    if missing:
        lines.append(f"No verifier result: {', '.join(missing)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: coverage_report.py <Harbor jobs folder> <report.md>")
    records, trials = load_runs(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(render(summarise(records), trials))


if __name__ == "__main__":
    main()
