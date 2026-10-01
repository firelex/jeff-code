import json
from pathlib import Path

import pytest

from coverage_report import load_runs, render, summarise


def record(task, turn, calls, stop="toolUse", menu=None, menu_ms=2.0):
    """One trace line as the fork writes it; calls are (name, arguments, match kind, option id)."""
    menu = menu or [
        {"id": "o1", "kind": "look"},
        {"id": "o2", "kind": "read"},
        {"id": "o3", "kind": "check"},
        {"id": "ask_model", "kind": "ask_model"},
    ]
    return {
        "schema": "jeff-first-trace/1",
        "task_id": task,
        "turn": turn,
        "menu": menu,
        "action": {
            "stop_reason": stop,
            "tool_calls": [
                {"name": n, "arguments": a, "match": {"kind": k, **({"optionId": o} if o else {})}}
                for n, a, k, o in calls
            ],
        },
        "timings_ms": {"menu": menu_ms, "model": 1000.0, "jeff": None},
    }


def test_coverage_counts_single_exact_matches_over_normal_turns():
    records = [
        record("a", 1, [("bash", {"command": "ls -la"}, "exact", "o1")]),
        record("a", 2, [("read", {"path": "x"}, "near", "o2")]),
        record("a", 3, [("edit", {"path": "x"}, "none", None)]),
        record("a", 4, [], stop="stop"),
        record("a", 5, [("bash", {"command": "pytest"}, "exact", "o3"), ("read", {"path": "y"}, "exact", "o2")]),
        record("a", 6, [], stop="error"),
    ]
    s = summarise(records)
    assert s["counted_turns"] == 5
    assert s["covered"] == 1
    assert s["coverage"] == pytest.approx(0.2)
    assert s["covered_with_near"] == 2
    assert s["several_calls"] == 1
    assert s["errors"] == {"a": 1}
    assert s["covered_by_kind"] == {"look": 1}


def test_gate_passes_at_25_percent():
    records = [record("a", i, [("edit", {}, "none", None)]) for i in range(1, 4)]
    records.append(record("a", 4, [("bash", {"command": "ls"}, "exact", "o1")]))
    assert summarise(records)["gate_passed"] is True
    records.append(record("a", 5, [("edit", {}, "none", None)]))
    assert summarise(records)["gate_passed"] is False


def test_per_task_and_position_split():
    records = [record("a", t, [("bash", {"command": "ls"}, "exact", "o1")]) for t in range(1, 6)]
    records += [record("a", t, [("edit", {}, "none", None)]) for t in range(6, 11)]
    records += [record("b", 1, [("edit", {}, "none", None)])]
    s = summarise(records)
    assert s["per_task"]["a"] == {"counted": 10, "covered": 5}
    assert s["per_task"]["b"] == {"counted": 1, "covered": 0}
    assert s["by_position"] == {"turns 1-5": {"counted": 6, "covered": 5}, "turns 6+": {"counted": 5, "covered": 0}}


def test_uncovered_calls_grouped_by_tool_and_first_bash_word():
    records = [
        record("a", 1, [("bash", {"command": "cat a.py"}, "none", None)]),
        record("a", 2, [("bash", {"command": "  cat   b.py"}, "none", None)]),
        record("a", 3, [("edit", {"path": "a.py"}, "none", None)]),
        record("a", 4, [("read", {"path": "a.py", "offset": 3}, "near", "o2")]),
    ]
    assert summarise(records)["uncovered_groups"] == [("bash cat", 2), ("edit", 1), ("read", 1)]


def test_rejects_unknown_trace_schema():
    bad = record("a", 1, [])
    bad["schema"] = "jeff-first-trace/2"
    with pytest.raises(ValueError, match="jeff-first-trace/2"):
        summarise([bad])


def write_trial(root: Path, name: str, task: str, records, reward):
    trial = root / "job-1" / name
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "jeff-first-trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    verifier = None if reward is None else {"rewards": {"reward": reward}}
    (trial / "result.json").write_text(json.dumps({"task_name": task, "verifier_result": verifier}))


def test_load_runs_reads_traces_and_verifier_results(tmp_path):
    write_trial(tmp_path, "t1", "terminal-bench/a", [record("a", 1, [("bash", {"command": "ls"}, "exact", "o1")])], 1.0)
    write_trial(tmp_path, "t2", "terminal-bench/b", [record("b", 1, [("edit", {}, "none", None)])], 0.0)
    write_trial(tmp_path, "t3", "terminal-bench/c", [record("c", 1, [("edit", {}, "none", None)])], None)
    records, trials = load_runs(tmp_path)
    assert len(records) == 3
    assert trials == {"terminal-bench/a": 1.0, "terminal-bench/b": 0.0, "terminal-bench/c": None}


def test_load_runs_fails_when_a_trial_has_no_result_file(tmp_path):
    write_trial(tmp_path, "t1", "terminal-bench/a", [record("a", 1, [])], 1.0)
    (tmp_path / "job-1" / "t1" / "result.json").unlink()
    with pytest.raises(FileNotFoundError, match="result.json"):
        load_runs(tmp_path)


def test_load_runs_fails_when_there_are_no_traces(tmp_path):
    with pytest.raises(FileNotFoundError, match="no jeff-first-trace.jsonl"):
        load_runs(tmp_path)


def test_render_states_the_gate_and_pass_rate():
    records = [record("a", 1, [("bash", {"command": "ls"}, "exact", "o1")])]
    text = render(summarise(records), {"terminal-bench/a": 1.0, "terminal-bench/b": 0.0, "terminal-bench/c": None})
    assert "Gate: PASS" in text
    assert "Menu coverage: 100.0% (1 of 1 turns)" in text
    assert "Passed: 1 of 2 tasks with a verifier result" in text
    assert "No verifier result: terminal-bench/c" in text
