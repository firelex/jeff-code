"""Find JeffFirst errors in a Harbor trial's pi output.

When the scout fails (the lists builder, the teacher, the trace writer or a proxy), pi ends the turn with an assistant
message whose errorMessage starts with "JeffFirst:" and writes no trace line for it. Harbor runs pi without pipefail, so
the trial still looks finished. Such a session is broken, never a normal benchmark outcome; this module finds those
errors so the run script and the Gate 0 report can refuse the trial.

Source: agent/pi.txt, which Harbor writes by piping `pi --print --mode json` (stdout and stderr together, with the
"message_update" events removed) through tee. Every JSON line is one pi session event; a "message_end" event carries
the finished message, and an assistant message that failed has "stopReason": "error" and an "errorMessage". Lines
that do not start with "{" are pi's stderr text and carry no events.
"""

import json
from pathlib import Path

JEFF_FIRST_ERROR_PREFIX = "JeffFirst:"


def jeff_first_errors(trial: Path) -> list[str]:
    pi_output = trial / "agent" / "pi.txt"
    if not pi_output.exists():
        raise FileNotFoundError(f"{trial.name} has no agent/pi.txt, so its JeffFirst errors cannot be checked")
    errors = []
    for number, line in enumerate(pi_output.read_text().splitlines(), start=1):
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{pi_output} line {number} is not valid JSON: {error}") from error
        if event.get("type") != "message_end":
            continue
        message = event["message"]
        if message.get("role") != "assistant":
            continue
        error_message = message.get("errorMessage")
        if isinstance(error_message, str) and error_message.startswith(JEFF_FIRST_ERROR_PREFIX):
            errors.append(error_message)
    return errors
