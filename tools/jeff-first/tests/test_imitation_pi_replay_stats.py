import json

from imitation.pi_replay_stats import stats_markdown, summarize


def write(folder, name, records):
    (folder / f"stage3-replay-{name}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))


def session(name, excluded, equal, differing, **counts):
    return {
        "trial": f"/runs/{name}",
        "session": name,
        "task": "fix-bug",
        "machine": "qwen3.8-27b-fp8@casdgx01-gpu5",
        "excluded": excluded,
        "menus_equal": equal,
        "menus_differing": differing,
        "commands": counts.get("commands", 4),
        "output_mismatches": counts.get("output_mismatches", 1),
        "output_mismatches_beyond_digits": counts.get("beyond", 0),
        "information_commands": 2,
        "information_mismatches": 1,
        "timed_out": 0,
        "rows": 3,
        "decisions": 2,
        "stint_decisions": 1,
        "stint_rows": 2,
        "seconds": 10,
        "notes": [],
    }


def tool_row(name, decision, label, level="tool"):
    return {"session": name, "decision": decision, "level": level, "label": label, "page": 1, "task": "fix-bug", "machine": "m"}


def test_summary_counts_rows_stints_labels_and_fidelity(tmp_path):
    write(tmp_path, "sessions", [session("a", False, 10, 0), session("b", True, 5, 5), {"trial": "/runs/c", "task": "fix-bug", "failed": "boom"}])
    write(tmp_path, "rows", [tool_row("a", 0, "list"), tool_row("a", 0, "list-1", "argument"), tool_row("a", 1, "hand_over")])
    write(
        tmp_path,
        "decisions",
        [
            {"session": "a", "decision": 0, "turn": 1, "stint": False, "kind": "list", "option": "list-1"},
            {"session": "a", "decision": 1, "turn": 1, "stint": True, "kind": None, "option": None},
            {"session": "b", "decision": 0, "turn": 1, "stint": False, "kind": None, "option": None},
        ],
    )
    write(
        tmp_path,
        "menu-checks",
        [{"session": "a", "task": "fix-bug", "turn": 1, "equal": True, "difference": {}}]
        + [{"session": "b", "task": "fix-bug", "turn": 2, "equal": False, "difference": {"read": {"added": ["Read the file /app/x"], "removed": []}}}],
    )
    summary = summarize(tmp_path)
    assert summary["trials_failed"] == 1 and summary["sessions"] == 2 and summary["sessions_excluded"] == ["b"]
    assert summary["rows"] == 3 and summary["decisions"] == 2 and summary["stint_decisions"] == 1 and summary["stint_rows"] == 1
    assert summary["hand_over_share"] == 0.5
    assert summary["labels"] == {"list": 1, "hand_over": 1}
    assert summary["stint_labels"] == {"hand_over": 1}
    assert summary["menus"] == {"equal": 15, "differing": 5, "equal_kept": 10, "differing_kept": 0}
    assert summary["outputs"]["commands"] == 8 and summary["outputs"]["mismatches"] == 2
    markdown = stats_markdown(summary)
    assert "| list | 1 | 50.0% |" in markdown and "Read the file /app/x" in markdown
