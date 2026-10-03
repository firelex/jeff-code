from scout_quality import Quality, quality, smoke_passes


def step(call, reasons=("fine",) * 5):
    picks = [{"optionId": "read-1", "reason": r, "failedAttempts": []} for r in reasons]
    return {
        "kind": "decision",
        "levels": [{"level": "tool", "picks": []}, {"level": "argument", "picks": picks}],
        "action": {"kind": "step", "tool_call": call},
    }


def hand_over():
    return {"kind": "decision", "levels": [], "action": {"kind": "hand_over", "why": "chosen"}}


def model_turn():
    return {"kind": "model_turn"}


A = {"name": "read", "arguments": {"path": "/app/a.py"}}
B = {"name": "read", "arguments": {"path": "/app/b.py"}}


def test_counts_steps_and_identical_steps_within_one_stint():
    lines = [step(A), step(A), step(B), hand_over(), model_turn(), step(A), hand_over()]
    assert quality(lines) == Quality(steps=4, forced_steps=0, max_identical_in_stint=2)


def test_a_step_is_forced_when_two_reasons_argue_against_it():
    against = ("that's not an option, among the options this is closest", "would just fail again", "ok", "ok", "ok")
    assert quality([step(A, against)]).forced_steps == 1
    assert quality([step(A, ("not offered", "ok", "ok", "ok", "ok"))]).forced_steps == 0


def test_smoke_passes_at_most_one_forced_step_in_twenty_and_no_more_than_two_identical_steps():
    assert smoke_passes(Quality(steps=20, forced_steps=1, max_identical_in_stint=2))
    assert not smoke_passes(Quality(steps=19, forced_steps=1, max_identical_in_stint=2))
    assert not smoke_passes(Quality(steps=40, forced_steps=0, max_identical_in_stint=3))
