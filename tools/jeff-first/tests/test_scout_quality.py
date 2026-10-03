from scout_quality import Quality, quality, smoke_passes


def step(call, reasons=("fine",) * 5, chosen="read-1", picks=None):
    if picks is None:
        picks = [{"optionId": chosen, "reason": r, "failedAttempts": []} for r in reasons]
    return {
        "kind": "decision",
        "levels": [{"level": "tool", "picks": []}, {"level": "argument", "picks": picks, "chosen": chosen}],
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


def test_supportive_run_or_check_reasons_are_not_counted_as_dissent():
    reasons = (
        "it has not been run yet, so running it shows the result",
        "never executed; running it now",
        "The tests have not run since the edit, so run them",
        "the most useful among these options",
        "without ever executing it, so run it",
    )
    assert quality([step(A, reasons)]).forced_steps == 0


def test_dissent_on_a_pick_of_a_different_option_than_chosen_does_not_count():
    picks = [
        {"optionId": "none_of_these", "reason": "not among the options shown, so hand over", "failedAttempts": []},
        {"optionId": "none_of_these", "reason": "would just fail again", "failedAttempts": []},
        {"optionId": "read-1", "reason": "ok", "failedAttempts": []},
        {"optionId": "read-1", "reason": "ok", "failedAttempts": []},
        {"optionId": "read-1", "reason": "ok", "failedAttempts": []},
    ]
    assert quality([step(A, picks=picks, chosen="read-1")]).forced_steps == 0


def test_smoke_passes_at_most_one_forced_step_in_twenty_and_no_more_than_two_identical_steps():
    assert smoke_passes(Quality(steps=20, forced_steps=1, max_identical_in_stint=2))
    assert not smoke_passes(Quality(steps=19, forced_steps=1, max_identical_in_stint=2))
    assert not smoke_passes(Quality(steps=40, forced_steps=0, max_identical_in_stint=3))
