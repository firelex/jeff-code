import json

import pytest

from pi_errors import jeff_first_errors


def assistant_end(error_message=None):
    message = {"role": "assistant", "content": [], "stopReason": "error" if error_message else "stop"}
    if error_message:
        message["errorMessage"] = error_message
    return json.dumps({"type": "message_end", "message": message})


def write_pi(tmp_path, text):
    trial = tmp_path / "x__1"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "pi.txt").write_text(text)
    return trial


def test_skips_a_last_line_cut_off_without_a_newline(tmp_path):
    trial = write_pi(tmp_path, assistant_end() + "\n" + '{"type":"turn_end","message":{"content":"half')
    assert jeff_first_errors(trial) == []


def test_still_finds_a_jeff_first_error_before_a_cut_off_last_line(tmp_path):
    text = assistant_end("JeffFirst: the teacher failed") + "\n" + '{"type":"tool_execution_update","partial":"ab'
    assert jeff_first_errors(write_pi(tmp_path, text)) == ["JeffFirst: the teacher failed"]


def test_raises_on_an_invalid_line_ended_by_a_newline(tmp_path):
    trial = write_pi(tmp_path, assistant_end() + "\n" + '{"type":"turn_end","mess\n' + assistant_end() + "\n")
    with pytest.raises(ValueError, match=r"pi\.txt line 2 is not valid JSON"):
        jeff_first_errors(trial)


def test_raises_on_an_invalid_last_line_that_ends_with_a_newline(tmp_path):
    trial = write_pi(tmp_path, assistant_end() + "\n" + '{"type":"turn_end","mess\n')
    with pytest.raises(ValueError, match=r"pi\.txt line 2 is not valid JSON"):
        jeff_first_errors(trial)
