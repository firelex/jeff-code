import json

import pytest

from jeff_eval_summary import main, summarize

CUT = {"tokens_before": 9249, "tokens_after": 8184, "lines_left_out": 122, "limit": 8192}
CANNOT = {"tokens_before": 23150, "tokens_least": 11268, "limit": 8192}


def level(chooser, cut=None, abstained=None):
    return {"chooser": chooser, "jeff_cut": cut, "jeff_abstained": abstained}


def write_trace(folder, lines):
    path = folder / "job" / "trial" / "agent" / "jeff-first-trace.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_counts_questions_cuts_and_abstentions_per_decision_kind(tmp_path, capsys):
    write_trace(tmp_path, [
        {"kind": "decision", "levels": [level("jeff:step"), level("jeff:step", cut=CUT)]},
        {"kind": "decision", "levels": [level("jeff:step", abstained=CANNOT)]},
        {"kind": "decision", "levels": [level("teacher:glm")]},
        {"kind": "qwen_request", "router": "jeff:router", "attempt": 1, "router_cut": None, "router_abstained": CANNOT},
        {"kind": "qwen_request", "router": "jeff:router", "attempt": 2, "router_cut": None, "router_abstained": None},
        {"kind": "qwen_request", "router": "fixed:xhigh", "attempt": 1, "router_cut": None, "router_abstained": None},
        {"kind": "output_trim", "trimmer": "jeff:trim", "jeff_cut": CUT, "jeff_abstained": None},
        {"kind": "model_turn"},
    ])
    summary = summarize(tmp_path)
    assert summary["counts"] == {"step": {"asked": 3, "cut": 1, "abstained": 1},
                                 "router": {"asked": 1, "cut": 0, "abstained": 1},
                                 "trim": {"asked": 1, "cut": 1, "abstained": 0}}
    assert [(a["kind"], a["tokens_least"]) for a in summary["abstentions"]] == [("step", 11268), ("router", 11268)]
    main([str(tmp_path)])
    assert "step: 3 questions, 1 cut to fit, 1 abstained" in capsys.readouterr().out
    assert json.loads((tmp_path / "jeff-decisions-summary.json").read_text())["counts"] == summary["counts"]


def test_a_trace_without_the_fields_or_no_trace_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no agent/jeff-first-trace.jsonl"):
        summarize(tmp_path)
    write_trace(tmp_path, [{"kind": "decision", "levels": [{"chooser": "jeff:step"}]}])
    with pytest.raises(KeyError):
        summarize(tmp_path)
