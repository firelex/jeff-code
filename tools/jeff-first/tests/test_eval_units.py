import json

import pytest

from eval_units import trace_counts


def request(level: str) -> dict:
    return {
        "kind": "qwen_request",
        "timings_ms": {"model": 1000, "router": None},
        "limit_cut": None,
        "attempt": 1,
        "thinking_level": level,
        "router_level": None,
        "router": "fixed:xhigh",
        "forced_xhigh": None,
        "guard": None,
    }


def test_a_last_line_cut_mid_write_is_left_out_and_counted(tmp_path):
    trace = tmp_path / "jeff-first-trace.jsonl"
    trace.write_text(json.dumps(request("off")) + "\n" + json.dumps(request("xhigh"))[:40], encoding="utf-8")
    counts = trace_counts(trace)
    assert counts["turns"] == 1
    assert counts["trace_cut_last_line"] is True


def test_a_complete_trace_has_no_cut_line(tmp_path):
    trace = tmp_path / "jeff-first-trace.jsonl"
    trace.write_text(json.dumps(request("off")) + "\n", encoding="utf-8")
    assert trace_counts(trace)["trace_cut_last_line"] is False


def test_a_bad_line_before_the_end_still_raises(tmp_path):
    trace = tmp_path / "jeff-first-trace.jsonl"
    trace.write_text("{not json\n" + json.dumps(request("off")) + "\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        trace_counts(trace)
