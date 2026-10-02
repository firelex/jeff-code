import json

import pytest

from gate0_report import TaskResult, gate, summarise


def write_trial(root, task, lines, reward, jeff_first_error=None):
    trial = root / f"{task}-20261002-120000" / f"{task}__abc"
    (trial / "agent").mkdir(parents=True)
    message = {"role": "assistant", "content": [], "stopReason": "stop"}
    if jeff_first_error:
        message.update(stopReason="error", errorMessage=jeff_first_error)
    (trial / "agent" / "pi.txt").write_text(json.dumps({"type": "message_end", "message": message}) + "\n")
    (trial / "agent" / "jeff-first-trace.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))


def shadow_turn(stop="toolUse", ms=4000):
    return {"schema": "jeff-first-trace/1", "action": {"stop_reason": stop}, "timings_ms": {"model": ms}}


def model_turn(stop="toolUse", ms=4000):
    return {"schema": "jeff-first-trace/2", "kind": "model_turn", "action": {"stop_reason": stop}, "timings_ms": {"model": ms}}


def decision(kind="step", chooser_ms=2000):
    action = {"kind": "step", "tool_call": {}} if kind == "step" else {"kind": "hand_over", "why": "chosen"}
    return {"schema": "jeff-first-trace/2", "kind": "decision", "action": action, "timings_ms": {"lists": 1, "chooser": chooser_ms}}


def test_counts_model_turns_and_seconds_in_both_schemas(tmp_path):
    write_trial(tmp_path / "base", "t1", [shadow_turn(), shadow_turn(), shadow_turn("error"), shadow_turn("stop")], 1.0)
    write_trial(tmp_path / "teach", "t1", [decision(), decision("hand_over"), model_turn(), model_turn("stop")], 0.0)
    base = summarise(tmp_path / "base", ["t1"])
    teach = summarise(tmp_path / "teach", ["t1"])
    assert base["t1"] == TaskResult(turns=3, model_seconds=12.0, passed=True, scout_steps=0, teacher_seconds=0.0)
    assert teach["t1"] == TaskResult(turns=2, model_seconds=8.0, passed=False, scout_steps=1, teacher_seconds=4.0)


def test_fails_loudly_when_a_task_has_no_trial(tmp_path):
    (tmp_path / "base").mkdir()
    with pytest.raises(FileNotFoundError, match="t9"):
        summarise(tmp_path / "base", ["t9"])


def test_fails_loudly_when_a_trial_has_no_verifier_result(tmp_path):
    write_trial(tmp_path / "base", "t1", [shadow_turn()], None)
    with pytest.raises(ValueError, match="no reward"):
        summarise(tmp_path / "base", ["t1"])


def test_fails_loudly_when_a_trial_ended_with_a_jeff_first_error(tmp_path):
    write_trial(tmp_path / "teach", "t1", [decision(), model_turn()], 0.0, jeff_first_error="JeffFirst: the trace could not be written")
    with pytest.raises(ValueError, match="t1__abc ended with a JeffFirst error: JeffFirst: the trace could not be written"):
        summarise(tmp_path / "teach", ["t1"])


def result(turns, seconds, passed):
    return TaskResult(turns=turns, model_seconds=seconds, passed=passed, scout_steps=0, teacher_seconds=0.0)


def test_gate_passes_with_25_percent_fewer_turns_and_seconds_and_at_most_one_pass_lost():
    base = {"a": result(40, 160.0, True), "b": result(60, 240.0, True)}
    teacher = {"a": result(30, 120.0, True), "b": result(45, 180.0, False)}
    passed, line = gate(base, teacher)
    assert passed
    assert "pass" in line


def test_gate_fails_when_seconds_fall_less_than_25_percent():
    base = {"a": result(40, 160.0, True)}
    teacher = {"a": result(30, 150.0, True)}
    passed, line = gate(base, teacher)
    assert not passed
    assert "seconds" in line


def test_gate_fails_when_two_more_tasks_fail():
    base = {"a": result(40, 160.0, True), "b": result(40, 160.0, True)}
    teacher = {"a": result(20, 80.0, False), "b": result(20, 80.0, False)}
    assert not gate(base, teacher)[0]
