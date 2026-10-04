"""Statistics of a stage 3 replay run (pi_replay.py output folder) as JSON and Markdown.

Counts: trials and sessions (replayed, failed, excluded for menu fidelity), rows, decisions, stint decisions and
rows (points inside a coding-model turn, after a scout step), labels at the tool level, hand-over share, and the
fidelity numbers: before-turn menus built in the container that equal the logged ones, and replayed command outputs
that differ from pi's recorded ones.

Usage (from tools/jeff-first):
    uv run python -m imitation.pi_replay_stats OUT_DIR --summary summary.json --stats ../../results/imitation/stage3-stats.md
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from imitation.pi_replay import MENU_DIFFERENCE_LIMIT, QUALITY
from imitation.stage3 import CURRENT_TARBALL

DIFFERENCE_EXAMPLES = 8


def _lines(folder: Path, name: str) -> list[dict]:
    path = folder / f"stage3-replay-{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(folder: Path) -> dict:
    records = _lines(folder, "sessions")
    failed = [record for record in records if "failed" in record]
    sessions = [record for record in records if "failed" not in record]
    excluded = {record["session"] for record in sessions if record["excluded"]}
    kept = [record for record in sessions if not record["excluded"]]
    rows = _lines(folder, "rows")
    if {row["session"] for row in rows} & excluded:
        raise ValueError("the rows file holds rows of an excluded session")
    stints = {(d["session"], d["decision"]) for d in _lines(folder, "decisions") if d["stint"] and d["session"] not in excluded}
    tool_rows = [row for row in rows if row["level"] == "tool" and row["label"] != "show_more"]
    labels = Counter(row["label"] for row in tool_rows)
    stint_labels = Counter(row["label"] for row in tool_rows if (row["session"], row["decision"]) in stints)
    checks = _lines(folder, "menu-checks")
    differing = [check for check in checks if not check["equal"]]
    kinds: Counter = Counter()
    for check in differing:
        kinds.update(check["difference"])
    return {
        "trials_failed": len(failed),
        "failures": [{"trial": record["trial"], "reason": record["failed"][:300]} for record in failed],
        "sessions": len(sessions),
        "sessions_excluded": sorted(excluded),
        "rows": len(rows),
        "decisions": len(tool_rows),
        "stint_decisions": len(stints),
        "stint_rows": sum((row["session"], row["decision"]) in stints for row in rows),
        "hand_over_share": round(labels["hand_over"] / len(tool_rows), 4) if tool_rows else None,
        "labels": dict(labels.most_common()),
        "stint_labels": dict(stint_labels.most_common()),
        "argument_rows_by_tool": dict(Counter(row["label"].rsplit("-", 1)[0] for row in rows if row["level"] == "argument").most_common()),
        "menus": {
            "equal": sum(record["menus_equal"] for record in sessions),
            "differing": sum(record["menus_differing"] for record in sessions),
            "equal_kept": sum(record["menus_equal"] for record in kept),
            "differing_kept": sum(record["menus_differing"] for record in kept),
        },
        "menu_difference_kinds": dict(kinds.most_common()),
        "menu_difference_examples": differing[:DIFFERENCE_EXAMPLES],
        "outputs": {
            "commands": sum(record["commands"] for record in sessions),
            "mismatches": sum(record["output_mismatches"] for record in sessions),
            "mismatches_beyond_digits": sum(record["output_mismatches_beyond_digits"] for record in sessions),
            "information_commands": sum(record["information_commands"] for record in sessions),
            "information_mismatches": sum(record["information_mismatches"] for record in sessions),
            "timed_out": sum(record["timed_out"] for record in sessions),
        },
        "rows_by_machine": dict(Counter(row["machine"] for row in rows).most_common()),
        "sessions_by_task": dict(sorted(Counter(record["task"] for record in kept).items())),
    }


def _share(part: int, whole: int) -> str:
    return "-" if whole == 0 else f"{100 * part / whole:.1f}%"


def stats_markdown(summary: dict) -> str:
    menus = summary["menus"]
    outputs = summary["outputs"]
    compared = menus["equal"] + menus["differing"]
    lines = [
        "# Stage 3 statistics (own record-mode sessions, replayed)",
        "",
        f"Build `{CURRENT_TARBALL}` only. Each session was replayed in its task's Docker image without a model "
        "(tools/jeff-first/imitation/pi_replay.py): rows before a coding-model turn use the menu logged live; rows "
        f"inside a turn (stints) use the menu built in the container after the previous step ran. Quality \"{QUALITY}\".",
        "",
        "## Sessions",
        "",
        f"- Replayed: {summary['sessions']}; trials failed: {summary['trials_failed']}.",
        f"- Excluded (before-turn menus differ from the logged ones in more than {MENU_DIFFERENCE_LIMIT:.0%} of turns): "
        f"{len(summary['sessions_excluded'])}" + (f" ({', '.join(summary['sessions_excluded'])})" if summary["sessions_excluded"] else "") + ".",
        "",
        "## Rows and decisions",
        "",
        f"- Rows: {summary['rows']}; decisions: {summary['decisions']}; hand-over share of decisions: "
        f"{_share(summary['labels'].get('hand_over', 0), summary['decisions'])}.",
        f"- Stint decisions (inside a turn, after a scout step): {summary['stint_decisions']}; stint rows: {summary['stint_rows']}.",
        "- Argument rows by tool: " + (", ".join(f"{key} {count}" for key, count in summary["argument_rows_by_tool"].items()) or "none") + ".",
        "",
        "Labels at the tool level (one per decision):",
        "",
        "| Label | Decisions | Share | Of them inside a turn |",
        "|---|---|---|---|",
    ]
    for label, count in summary["labels"].items():
        lines.append(f"| {label} | {count} | {_share(count, summary['decisions'])} | {summary['stint_labels'].get(label, 0)} |")
    lines += [
        "",
        "## Fidelity",
        "",
        f"- Before-turn menus built in the container equal to the logged ones: {menus['equal']} of {compared} "
        f"({_share(menus['equal'], compared)}); in the sessions kept: {menus['equal_kept']} of "
        f"{menus['equal_kept'] + menus['differing_kept']}.",
        "- Where they differ (menu parts, counted per differing menu): "
        + (", ".join(f"{key} {count}" for key, count in summary["menu_difference_kinds"].items()) or "none")
        + ".",
        f"- Replayed command outputs that differ from pi's recorded output (whitespace ignored): {outputs['mismatches']} of "
        f"{outputs['commands']} ({_share(outputs['mismatches'], outputs['commands'])}); still differing with digits ignored: "
        f"{outputs['mismatches_beyond_digits']}. Information commands: {outputs['information_mismatches']} of "
        f"{outputs['information_commands']}. Stopped at a time limit: {outputs['timed_out']}.",
        "",
    ]
    if summary["menu_difference_examples"]:
        lines += ["Examples of differing before-turn menus (what the container's menu adds or lacks):", ""]
        for example in summary["menu_difference_examples"]:
            lines.append(f"- {example['task']} session {example['session'][:8]} turn {example['turn']}: `{json.dumps(example['difference'])[:400]}`")
        lines.append("")
    lines += [
        "## By machine and task",
        "",
        "Rows by machine: " + ", ".join(f"{key} {count}" for key, count in summary["rows_by_machine"].items()) + ".",
        "",
        "Sessions kept by task: " + ", ".join(f"{key} {count}" for key, count in summary["sessions_by_task"].items()) + ".",
        "",
    ]
    if summary["failures"]:
        lines += ["## Failed trials", ""] + [f"- {failure['trial']}: {failure['reason']}" for failure in summary["failures"]] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="pi_replay.py output folder")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize(args.folder)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    args.stats.write_text(stats_markdown(summary))
    print(json.dumps({key: value for key, value in summary.items() if key not in ("menu_difference_examples", "failures")}, indent=1))


if __name__ == "__main__":
    main()
