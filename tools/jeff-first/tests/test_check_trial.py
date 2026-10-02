import json

import pytest

from check_trial import check_job


def assistant_end(error_message=None):
    message = {"role": "assistant", "content": [], "stopReason": "error" if error_message else "stop"}
    if error_message:
        message["errorMessage"] = error_message
    return json.dumps({"type": "message_end", "message": message})


def make_trial(job, name, exception=None, trace_lines=1, exception_type="RuntimeError", pi_lines=None):
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    lines = pi_lines if pi_lines is not None else ['{"type":"session","version":3}', assistant_end()]
    (trial / "agent" / "pi.txt").write_text("".join(line + "\n" for line in lines))
    info = None if exception is None else {"exception_type": exception_type, "exception_message": exception}
    (trial / "result.json").write_text(json.dumps({"task_name": "terminal-bench/x", "exception_info": info}))
    if trace_lines:
        (trial / "agent" / "jeff-first-trace.jsonl").write_text("{}\n" * trace_lines)


def test_accepts_a_trial_without_exception_and_with_a_trace(tmp_path):
    make_trial(tmp_path, "x__1")
    assert check_job(tmp_path) == "x__1: ok, 1 trace lines"


def test_rejects_a_trial_with_an_exception(tmp_path):
    make_trial(tmp_path, "x__1", exception="Docker compose command failed")
    with pytest.raises(RuntimeError, match="x__1 raised RuntimeError: Docker compose command failed"):
        check_job(tmp_path)


def test_rejects_a_trial_without_a_trace(tmp_path):
    make_trial(tmp_path, "x__1", trace_lines=0)
    with pytest.raises(RuntimeError, match="x__1 wrote no jeff-first-trace.jsonl"):
        check_job(tmp_path)


def test_rejects_a_job_without_trials(tmp_path):
    with pytest.raises(RuntimeError, match="no trial results"):
        check_job(tmp_path)


def test_accepts_an_agent_timeout_as_a_task_result(tmp_path):
    make_trial(tmp_path, "x__1", exception="Agent execution timed out after 900.0 seconds", exception_type="AgentTimeoutError")
    assert check_job(tmp_path) == "x__1: ok (agent timed out), 1 trace lines"


def test_rejects_a_trial_ended_by_a_jeff_first_error(tmp_path):
    make_trial(
        tmp_path,
        "x__1",
        pi_lines=[
            '{"type":"session","version":3}',
            assistant_end(),
            "some stderr text from pi",
            assistant_end("JeffFirst: the teacher at http://teacher/v1 could not be reached"),
        ],
    )
    with pytest.raises(RuntimeError, match="x__1 ended with a JeffFirst error: JeffFirst: the teacher at http://teacher/v1"):
        check_job(tmp_path)


def test_accepts_a_trial_whose_only_error_is_not_from_jeff_first(tmp_path):
    make_trial(tmp_path, "x__1", pi_lines=[assistant_end("429 rate limited")])
    assert check_job(tmp_path) == "x__1: ok, 1 trace lines"


def test_rejects_a_trial_without_pi_output(tmp_path):
    make_trial(tmp_path, "x__1")
    (tmp_path / "x__1" / "agent" / "pi.txt").unlink()
    with pytest.raises(FileNotFoundError, match="x__1 has no agent/pi.txt"):
        check_job(tmp_path)


def test_reports_a_setup_failure_with_its_reason_even_without_pi_output(tmp_path):
    make_trial(tmp_path, "x__1", exception="Docker compose command failed", trace_lines=0)
    (tmp_path / "x__1" / "agent" / "pi.txt").unlink()
    with pytest.raises(RuntimeError, match="x__1 raised RuntimeError: Docker compose command failed"):
        check_job(tmp_path)
