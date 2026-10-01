import json

import pytest

from check_trial import check_job


def make_trial(job, name, exception=None, trace_lines=1, exception_type="RuntimeError"):
    trial = job / name
    (trial / "agent").mkdir(parents=True)
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
