"""Scout quality from a teacher-mode trace (schema 3): how often the teacher was forced into a step it argued against,
and the longest run of identical scout steps within one stint (the steps between two turns of the large model).

Usage: uv run python scout_quality.py TRIAL_DIR [TRIAL_DIR ...]   (exit 1 if any trial fails the smoke-test bar)
"""

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DISSENT = re.compile(
    r"not (?:an |one of the |among the )?option|not offered|isn't offered|no such option|but (?:that's|that is|it's) not"
    r"|among (?:the|these) (?:options|choices)|only (?:the )?options? (?:are|is)|rather than re-?run|would (?:just )?fail again"
    r"|instead of re-?running|without ever executing|never (?:actually )?execut|not (?:yet )?(?:been )?(?:run|executed)",
    re.IGNORECASE,
)
FORCED_SHARE = 1 / 20
MAX_IDENTICAL = 2


@dataclass(frozen=True)
class Quality:
    steps: int
    forced_steps: int
    max_identical_in_stint: int


def quality(lines: list[dict]) -> Quality:
    steps = forced = longest = 0
    stint: Counter[str] = Counter()
    for line in lines:
        if line.get("kind") == "model_turn":
            stint = Counter()
            continue
        if line.get("kind") != "decision" or line["action"]["kind"] != "step":
            continue
        steps += 1
        argument_levels = [level for level in line["levels"] if level["level"] == "argument"]
        if argument_levels and sum(1 for pick in argument_levels[-1]["picks"] if DISSENT.search(pick["reason"])) >= 2:
            forced += 1
        key = json.dumps(line["action"]["tool_call"], sort_keys=True)
        stint[key] += 1
        longest = max(longest, stint[key])
    return Quality(steps=steps, forced_steps=forced, max_identical_in_stint=longest)


def smoke_passes(q: Quality) -> bool:
    return q.forced_steps <= q.steps * FORCED_SHARE and q.max_identical_in_stint <= MAX_IDENTICAL


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: scout_quality.py TRIAL_DIR [TRIAL_DIR ...]")
    failed = False
    for trial in map(Path, sys.argv[1:]):
        trace = trial / "agent" / "jeff-first-trace.jsonl"
        lines = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        q = quality(lines)
        ok = smoke_passes(q)
        failed |= not ok
        print(f"{trial.name}: {q.steps} steps, {q.forced_steps} forced, at most {q.max_identical_in_stint} identical in a stint: {'pass' if ok else 'FAIL'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
