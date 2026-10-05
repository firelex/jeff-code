"""Tests of cutoff_features.py and cutoff_estimate.py. Run: uv run --with pytest --with numpy python -m pytest
test_cutoff_estimate.py -q (from results/imitation/scripts)."""

import itertools

import cutoff_estimate as ce
import cutoff_features as cf
import numpy as np
import pytest


def test_test_counts_and_written_paths():
    assert cf.test_counts("==== 3 failed, 12 passed, 1 error in 2.1s ====") == (12, 4)
    assert cf.test_counts("Ran 5 tests in 0.1s\n\nFAILED (failures=2)") == (3, 2)
    assert cf.test_counts("no summary here") is None
    assert cf.written_paths("cat > /testbed/a.py <<'EOF'\nx\nEOF\npython3 t.py 2>&1 > /dev/null") == {"/testbed/a.py"}
    assert cf.written_paths("sed -i 's/a/b/' src/x.py && echo ok") == {"src/x.py"}
    assert cf.changes_files("bash", "ls -la /testbed") is False
    assert cf.changes_files("bash", "rm -rf build") is True


def entry(kind, t, **message):
    return {"type": "message", "timestamp": f"2026-10-05T01:00:{t:02d}.000Z", "message": {"role": kind, **message}}


def test_features_at_counts_only_what_was_written_by_the_checkpoint():
    session = [
        entry("assistant", 10, provider="harbor-endpoint", content=[
            {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "python3 -m pytest -q"}}],
            usage={"output": 100, "input": 1000}),
        entry("toolResult", 12, toolCallId="c1", isError=True, content=[{"type": "text", "text": "1 failed, 2 passed"}]),
        entry("assistant", 14, provider="jeff-first", content=[
            {"type": "toolCall", "id": "j1", "name": "bash", "arguments": {"command": "ls"}}]),
        entry("toolResult", 15, toolCallId="j1", isError=False, content=[{"type": "text", "text": "a"}]),
        entry("assistant", 20, provider="harbor-endpoint", content=[
            {"type": "text", "text": "Let me try another approach."},
            {"type": "toolCall", "id": "c2", "name": "bash", "arguments": {"command": "echo x > f.py"}}],
            usage={"output": 50, "input": 2000}),
        entry("toolResult", 21, toolCallId="c2", isError=False, content=[{"type": "text", "text": ""}]),
        entry("assistant", 40, provider="harbor-endpoint", content=[], usage={"output": 1, "input": 1}),
    ]
    trace = [
        {"kind": "qwen_request", "turn": 1, "attempt": 1, "thinking_level": "off", "guard": None, "forced_xhigh": None,
         "limit_cut": None, "thinking_chars": 0},
        {"kind": "qwen_request", "turn": 2, "attempt": 1, "thinking_level": "xhigh", "guard": {"trigger": "loop"},
         "forced_xhigh": None, "limit_cut": {"x": 1}, "thinking_chars": 500},
        {"kind": "qwen_request", "turn": 3, "attempt": 1, "thinking_level": "xhigh", "guard": None,
         "forced_xhigh": "failed", "limit_cut": None, "thinking_chars": 7},
    ]
    turns, results, compactions = cf.session_events(session)
    start = cf.when("2026-10-05T01:00:00.000Z")
    x = cf.features_at(start + 30, 0.5, turns, results, compactions, trace)  # turn 3 (t=40) not yet written
    assert x["turns"] == 2 and x["jeff_steps"] == 1
    assert x["off_share"] == 0.5 and x["guard_hits"] == 1 and x["limit_cuts"] == 1 and x["forced_xhigh"] == 0
    assert x["thinking_chars"] == 500 and x["output_tokens"] == 150 and x["context_tokens"] == 2000
    assert x["commands"] == 3 and x["failed_cmds"] == 1 and x["fail_share_recent"] == 0.5
    assert x["tests"] == 1 and x["last_test_failed"] == 1 and x["last_test_pass_frac"] == pytest.approx(2 / 3)
    assert x["file_changes"] == 1 and x["written_paths"] == 1
    assert x["since_change_min"] == pytest.approx(10 / 60)
    assert x["approach_phrases"] == 1


def test_split_keeps_tasks_whole_and_holds_out_a_fifth():
    groups = [("b", f"t{i}") for i in range(100)] * 3
    holdout, folds = ce.split_tasks(groups)
    assert len(holdout) == 20 and not holdout & set(folds)
    assert sorted(set(folds.values())) == [0, 1, 2, 3, 4]
    assert ce.split_tasks(groups) == (holdout, folds)


def session(folder, passed, agent_s, scores=(), section="s"):
    return {"folder": folder, "passed": passed, "agent_s": agent_s, "trial_s": agent_s + 100, "setup_s": 60,
            "section": section, "checkpoints": {t: {} for t in ce.CHECKPOINTS if agent_s > t * 60}}


def test_policy_charges_setup_plus_checkpoint_and_counts_a_stop_as_a_fail():
    base = session("b", True, 3000)
    long_fail = session("j1", False, 3000)
    long_pass = session("j2", True, 3000)
    short = session("j3", True, 300)
    scores = {("j1", t): 0.1 for t in ce.CHECKPOINTS} | {("j2", t): 0.9 for t in ce.CHECKPOINTS}
    scores |= {("j2", 30): 0.2}
    pairs = [(base, long_fail), (base, long_pass), (base, short)]
    assert ce.stop_time(long_fail, scores, (20, 30), 0.5) == 20
    assert ce.stop_time(long_pass, scores, (20, 30), 0.5) == 30
    assert ce.stop_time(short, scores, (20, 30), 0.5) is None
    c = ce.evaluate(pairs, lambda x: ce.stop_time(x, scores, (20, 30), 0.5))["pooled"]
    assert c["jeff_time"] == (60 + 1200) + (60 + 1800) + 400
    assert c["base_time"] == 3 * 3100
    assert c["jeff_pass"] == 1 and c["jeff_lost"] == 1 and c["jeff_stopped"] == 2
    s = ce.summary(c)
    assert s["loss"] == pytest.approx(100 / 3) and s["diff"] == pytest.approx(-200 / 3)
    with pytest.raises(ValueError):
        ce.session_outcome(short, 10)


def test_auc_matches_pair_counting():
    rng = np.random.default_rng(1)
    p = rng.integers(0, 5, 40).astype(float)
    y = rng.integers(0, 2, 40).astype(float)
    pairs = [(a, b) for a, b in itertools.product(range(40), range(40)) if y[a] == 1 and y[b] == 0]
    expected = sum(1.0 if p[a] > p[b] else 0.5 if p[a] == p[b] else 0.0 for a, b in pairs) / len(pairs)
    assert ce.auc(p, y) == pytest.approx(expected)


def test_logistic_finds_the_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * x[:, 0])))).astype(float)
    w = ce.fit_logistic(x, y, l2=0.0)
    assert w[1] == pytest.approx(2, abs=0.3) and abs(w[2]) < 0.2 and w[0] == pytest.approx(0.5, abs=0.2)


class Recorder:
    seen: list = []

    def fit(self, train):
        Recorder.seen.append({r["session"]["group"] for r in train})
        return self

    def predict(self, rows):
        return np.zeros(len(rows))


def test_cross_validation_never_trains_on_the_test_task():
    groups = [("b", f"t{i}") for i in range(20)]
    _, folds = ce.split_tasks(groups)
    rows = [{"session": {"group": g}, "t": t} for g in folds for t in (10, 20)]
    Recorder.seen = []
    ce.cross_validate(Recorder, rows, folds)
    for k, trained in enumerate(Recorder.seen):
        assert not {g for g in trained if folds[g] == k}
