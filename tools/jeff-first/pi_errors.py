"""Find JeffFirst errors in a Harbor trial's pi output.

When the scout fails (the lists builder, the teacher, the trace writer or a proxy), pi ends the turn with an assistant
message whose errorMessage starts with "JeffFirst:" and writes no trace line for it. Harbor runs pi without pipefail, so
the trial still looks finished. Such a session is broken, never a normal benchmark outcome; this module finds those
errors so the run script and the Gate 0 report can refuse the trial.

Source: agent/pi.txt, which Harbor writes by piping `pi --print --mode json` (stdout and stderr together, with the
"message_update" events removed) through tee. Every JSON line is one pi session event; a "message_end" event carries
the finished message, and an assistant message that failed has "stopReason": "error" and an "errorMessage". Lines
that do not start with "{" are pi's stderr text and carry no events.

When the agent hits its time limit, Harbor kills pi, sometimes while pi is writing a line; that last line is then cut
off mid-way and has no newline after it. Such a line is skipped. Any other line that is not valid JSON raises.
"""

import json
from pathlib import Path

JEFF_FIRST_ERROR_PREFIX = "JeffFirst:"


def jeff_first_errors(trial: Path) -> list[str]:
    pi_output = trial / "agent" / "pi.txt"
    if not pi_output.exists():
        raise FileNotFoundError(f"{trial.name} has no agent/pi.txt, so its JeffFirst errors cannot be checked")
    text = pi_output.read_text()
    lines = text.split("\n")
    # A file that ends with a newline splits into its lines plus one empty string, which is dropped. A file that does
    # not end with a newline has its final line cut off by a killed pi process (see the module docstring): drop it too.
    lines = lines[:-1]
    errors = []
    for number, line in enumerate(lines, start=1):
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
