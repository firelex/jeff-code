import pytest

from jeff_fit import Cut, QuestionTooLong, cut_state, left_out_line

QUESTION = "Question: what next? Options: A: read B: hand over"


def words(text: str) -> int:
    """A stand-in tokenizer: one token per word."""
    return len(text.split())


def prompt(state: str) -> int:
    return words("State: " + state + " " + QUESTION)


def lines(prefix: str, count: int, width: int = 3) -> list[str]:
    return [" ".join(f"{prefix}{i}w{j}" for j in range(width)) for i in range(count)]


def test_a_prompt_within_the_limit_is_not_cut():
    state = "\n".join(lines("x", 10))
    assert prompt(state) <= 100
    assert cut_state(state, prompt, words, limit=100, head_tokens=20) is None


def test_keeps_the_head_within_its_budget_and_the_most_tail_lines_that_fit():
    task = lines("task", 4)  # 12 words
    steps = lines("step", 40)  # 120 words
    state = "\n".join(task + steps)
    cut = cut_state(state, prompt, words, limit=60, head_tokens=13)
    assert isinstance(cut, Cut)
    kept = cut.state.split("\n")
    # head: 4 lines (12 words <= 13; a fifth line would make 15)
    assert kept[:4] == task
    marker = kept[4]
    tail = kept[5:]
    assert tail == steps[len(steps) - len(tail) :]
    assert marker == left_out_line(cut.lines_left_out)
    assert cut.lines_left_out == 40 - len(tail)
    assert cut.tokens_before == prompt(state)
    assert cut.tokens_after == prompt(cut.state) <= 60
    # one more tail line would not fit
    longer = "\n".join(task + [left_out_line(cut.lines_left_out - 1)] + steps[len(steps) - len(tail) - 1 :])
    assert prompt(longer) > 60


def test_the_same_input_always_gives_the_same_cut():
    state = "\n".join(lines("task", 4) + lines("step", 40))
    first = cut_state(state, prompt, words, limit=60, head_tokens=13)
    assert first == cut_state(state, prompt, words, limit=60, head_tokens=13)


def test_a_head_line_over_the_head_budget_keeps_no_head():
    state = "\n".join([" ".join(["long"] * 30), *lines("step", 40)])
    cut = cut_state(state, prompt, words, limit=60, head_tokens=13)
    assert cut is not None
    assert cut.state.split("\n")[0] == left_out_line(cut.lines_left_out)
    assert cut.tokens_after <= 60


def test_one_line_left_out_is_singular():
    assert left_out_line(1) == "[... 1 line left out ...]"
    assert left_out_line(412) == "[... 412 lines left out ...]"


def test_a_long_question_halves_the_room_between_head_and_tail():
    # The question takes 40 of the 60 tokens: 20 are left for the state, so the head gets at most 10, not 13.
    def long_question(state: str) -> int:
        return words(state) + 40

    task = lines("task", 4)
    steps = lines("step", 40)
    cut = cut_state("\n".join(task + steps), long_question, words, limit=60, head_tokens=13)
    assert cut is not None
    kept = cut.state.split("\n")
    assert kept[:3] == task[:3]
    assert kept[3] == left_out_line(cut.lines_left_out)
    assert kept[4:] == steps[len(steps) - len(kept[4:]) :]
    assert len(kept[4:]) == 1  # 9 head words + a 6-word marker line + 3 = 18 of the 20
    assert cut.tokens_after <= 60


def test_fails_when_the_question_alone_is_too_long():
    def long_question(state: str) -> int:
        return words(state) + 100

    with pytest.raises(QuestionTooLong, match="the question and its options alone are too long"):
        cut_state("\n".join(lines("x", 10)), long_question, words, limit=60, head_tokens=13)
