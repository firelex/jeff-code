"""The rule that cuts a question for Jeff (the small decision model) to fit Jeff's token limit. It is defined only here
and used by every path that hands Jeff a question:
- training rows: jeff_prompt.py fit-examples cuts the files every exporter writes (export_jeff.py for the step rows of
  stages 1 to 3, export_routing.py, export_trim.py) before training;
- run time: jeff_serve.py answers POST /v1/fit with this rule, and the scout asks it before every question (jeff-service.ts);
- offline scoring: the held-out bundle's question files are cut with jeff_prompt.py fit-examples.

Jeff reads at most LIMIT = 8,192 tokens: the whole prompt as jeff-dev builds it (jeff.model.decision_messages, then the
chat template), counted with Jeff's own tokenizer (jeff_prompt.py does the building and counting). When a prompt is
longer, the state text is cut in its middle, at line breaks. The question and its options are never cut.
- Room: what the question and options leave of the limit for the state: LIMIT minus the prompt's length with an
  empty state.
- Head: the most whole lines from the start of the state whose text is at most HEAD_TOKENS = 2,048 tokens, or half the
  room when that is smaller (a long option, such as a repeated command that wrote a whole file, can leave little
  room). The state starts with "Task:" and the task description, so this keeps the task, or its start.
- Tail: the most whole lines from the end of the state (the newest steps) that keep the whole prompt within LIMIT:
  everything the head leaves of the room.
- Between them, one line saying how many lines were left out, for example "[... 412 lines left out ...]".
Every candidate is counted as the whole prompt, so a cut prompt is at most LIMIT tokens. A prompt within the limit is
not changed. A question whose options and question text alone leave no room, or whose head and the line saying what
was left out do not fit with them, cannot be cut to fit: that raises QuestionTooLong. At run time the service then
answers "cannot fit" and Jeff abstains on that question only (owner ruling 2026-10-04: the scout hands over to the
coding model, the router uses xhigh, the trimmer keeps the whole output); the exporters leave such rows out.
"""

from collections.abc import Callable
from dataclasses import dataclass

LIMIT = 8192
HEAD_TOKENS = 2048


class QuestionTooLong(ValueError):
    """The question and its options alone leave too little of the limit for the state: cutting the state cannot make
    the prompt fit, and the question and options are never cut. `tokens_before`: the prompt as asked; `tokens_least`:
    the shortest prompt the rule could make (an empty state, or the head and the marker line alone); `limit`."""

    def __init__(self, message: str, tokens_before: int, tokens_least: int, limit: int):
        super().__init__(message)
        self.tokens_before = tokens_before
        self.tokens_least = tokens_least
        self.limit = limit

    def __reduce__(self):
        # Rebuilt with its counts when it crosses processes (fit-examples' workers).
        return (QuestionTooLong, (str(self), self.tokens_before, self.tokens_least, self.limit))


@dataclass(frozen=True)
class Cut:
    """A state cut to fit: the new state text, the prompt's length in tokens before and after, and how many lines of
    the state were left out."""

    state: str
    tokens_before: int
    tokens_after: int
    lines_left_out: int


def left_out_line(count: int) -> str:
    return f"[... {count} line{'' if count == 1 else 's'} left out ...]"


def _most(top: int, fits: Callable[[int], bool]) -> int:
    """The largest n from 0 to `top` for which fits(n) holds, by binary search; fits(0) must hold. The result is
    always a value for which fits was checked and held (or 0)."""
    low, high = 0, top
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


def cut_state(
    state: str,
    prompt_tokens: Callable[[str], int],
    text_tokens: Callable[[str], int],
    limit: int = LIMIT,
    head_tokens: int = HEAD_TOKENS,
) -> Cut | None:
    """None when the prompt with `state` is at most `limit` tokens; otherwise the state cut as the module docstring
    says. `prompt_tokens(state)` is the length of the whole prompt with that state; `text_tokens(text)` is the length
    of a piece of text alone."""
    before = prompt_tokens(state)
    if before <= limit:
        return None
    room = limit - prompt_tokens("")
    if room <= 0:
        raise QuestionTooLong(
            f"the prompt is {before} tokens, over Jeff's limit of {limit}, and {limit - room} tokens with an empty state: "
            "the question and its options alone are too long",
            before,
            limit - room,
            limit,
        )
    head_budget = min(head_tokens, room // 2)
    lines = state.split("\n")
    head = _most(len(lines), lambda count: text_tokens("\n".join(lines[:count])) <= head_budget)
    rest = len(lines) - head

    def candidate(tail: int) -> str:
        return "\n".join([*lines[:head], left_out_line(rest - tail), *lines[len(lines) - tail :]])

    lengths: dict[int, int] = {}

    def fits(tail: int) -> bool:
        lengths[tail] = prompt_tokens(candidate(tail))
        return lengths[tail] <= limit

    if not fits(0):
        raise QuestionTooLong(
            f"the prompt is {before} tokens, over Jeff's limit of {limit}, and still {lengths[0]} tokens with only the "
            f"first {head} lines of the state ({head_budget} tokens at most) and none of its end",
            before,
            lengths[0],
            limit,
        )
    tail = _most(rest, fits)
    return Cut(state=candidate(tail), tokens_before=before, tokens_after=lengths[tail], lines_left_out=rest - tail)
