import json

from imitation.pi_replay_stats import merged, stats_markdown, summarize

markdown_of = stats_markdown


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
        + [{"session": "b", "task": "fix-bug", "turn": turn, "equal": False, "difference": {"read": {"added": ["Read the file /app/x"], "removed": []}}} for turn in (2, 3)],
    )
    command = {"session": "a", "task": "fix-bug", "exit_code": 0, "timed_out": False, "timeout": 1800.0}
    write(
        tmp_path,
        "commands",
        [
            # Counted again with the current comparison, whatever the run wrote.
            {**command, "command": "ls", "information": True, "recorded_output": "a b", "replay_output": "a b\n", "mismatch": True},
            {**command, "command": "cat f", "information": True, "recorded_output": "x 1", "replay_output": "x 2", "mismatch": False},
            {**command, "command": "make", "information": False, "recorded_output": "(no output)\n\nCommand exited with code 2", "replay_output": "", "exit_code": 2, "mismatch": True},
            {**command, "command": "sleep 1", "information": False, "recorded_output": None, "replay_output": "", "mismatch": False},
            {**command, "session": "b", "command": "pwd", "information": True, "recorded_output": "/", "replay_output": "/x", "mismatch": True},
        ],
    )
    summary = summarize(merged([tmp_path]))
    assert summary["trials_failed"] == 1 and summary["sessions"] == 2 and summary["sessions_excluded"] == ["b"]
    assert summary["rows"] == 3 and summary["decisions"] == 2 and summary["stint_decisions"] == 1 and summary["stint_rows"] == 1
    assert summary["hand_over_share"] == 0.5
    assert summary["labels"] == {"list": 1, "hand_over": 1}
    assert summary["stint_labels"] == {"hand_over": 1}
    assert summary["menus"] == {"equal": 15, "differing": 5, "equal_kept": 10, "differing_kept": 0}
    assert summary["menu_difference_kinds"] == {"read": 2}
    assert summary["excluded_by_task"] == {"fix-bug": 1}
    assert "Excluded (before-turn menus differ" in markdown_of(summary) and "fix-bug 1" in markdown_of(summary)
    assert summary["outputs"] == {
        "commands": 5,
        "compared": 4,
        "mismatches": 2,
        "mismatches_beyond_digits": 1,
        "information_compared": 3,
        "information_mismatches": 2,
        "timed_out": 0,
    }
    markdown = stats_markdown(summary)
    assert "| list | 1 | 50.0% |" in markdown and "Read the file /app/x" in markdown


def test_per_trial_the_more_faithful_run_is_kept(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    # a: the second pass matched more logged menus. d: the first did (the second's environment differed more).
    # e: the second pass failed. c: only in the first run.
    write(first, "sessions", [session("a", True, 5, 5), session("c", False, 4, 0), session("d", False, 19, 1), session("e", False, 3, 0)])
    write(first, "rows", [tool_row("c", 0, "hand_over"), tool_row("d", 0, "hand_over"), tool_row("e", 0, "hand_over")])
    write(second, "sessions", [session("a", False, 10, 0), session("d", True, 10, 10), {"trial": "/runs/e", "task": "fix-bug", "failed": "boom"}])
    write(second, "rows", [tool_row("a", 0, "list"), tool_row("a", 0, "list-1", "argument")])
    for folder in (first, second):
        write(folder, "decisions", [])
        write(folder, "menu-checks", [])
        write(folder, "commands", [])
    data = merged([first, second])
    assert sorted(record["session"] for record in data["sessions"]) == ["a", "c", "d", "e"]
    assert [(row["session"], row["label"]) for row in data["rows"]] == [
        ("c", "hand_over"), ("d", "hand_over"), ("e", "hand_over"), ("a", "list"), ("a", "list-1")
    ]
    summary = summarize(data)
    assert summary["sessions_excluded"] == [] and summary["menus"]["equal"] == 36
