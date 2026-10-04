"""Tests of trim_labels.py. Run: uv run --with pytest --with aiohttp==3.12.15 --with tokenizers python -m pytest
test_trim_labels.py -q (from results/imitation/scripts)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trim_labels as tl


class FakeAsker:
    def __init__(self, replies, verdicts):
        self.replies = replies
        self.verdicts = verdicts
        self.asked = []
        self.judged = []

    async def ask(self, choice):
        self.asked.append(choice)
        return self.replies[choice]

    async def judge(self, reference, alternative, key):
        self.judged.append(key)
        return {"verdict": self.verdicts[key], "reason": "r"}


def run(coroutine):
    return asyncio.run(coroutine)


ALL_CUTS = {"last200", "last40", "first40", "first20last20"}
SAME = {"commands": ["cat /app/a.py"], "final_text": None}
OTHER = {"commands": ["ls /app"], "final_text": None}


def test_cuts_are_compared_with_the_uncut_reply_and_the_first_good_in_order_is_the_label():
    replies = {"all": SAME, "last40": OTHER, "first20last20": SAME, "first40": SAME}
    asker = FakeAsker(replies, {"last40 vs uncut": False})
    result = run(tl.trim_cascade(asker, ALL_CUTS))
    assert result["label"] == "first20last20"
    assert sorted(asker.asked) == ["all", "first20last20", "first40", "last40"]
    assert asker.judged == ["last40 vs uncut"]
    assert result["checks"]["first40"] == {"good": True, "by": "intent"}
    assert "all" not in result["checks"]


def test_the_reference_is_the_uncut_reply_not_the_recorded_action():
    replies = {"all": OTHER, "last40": OTHER, "first20last20": SAME, "first40": SAME}
    verdicts = {"first20last20 vs uncut": False, "first40 vs uncut": False}
    result = run(tl.trim_cascade(FakeAsker(replies, verdicts), ALL_CUTS))
    assert result["label"] == "last40"
    assert result["checks"]["last40"] == {"good": True, "by": "intent"}


def test_last_two_hundred_only_when_no_forty_line_cut_is_good():
    replies = {"all": SAME, "last40": OTHER, "first20last20": OTHER, "first40": {"commands": None, "final_text": None},
               "last200": {"commands": ["ls -la /app"], "final_text": None}}
    verdicts = {"last40 vs uncut": False, "first20last20 vs uncut": False, "last200 vs uncut": True}
    asker = FakeAsker(replies, verdicts)
    result = run(tl.trim_cascade(asker, ALL_CUTS))
    assert result["label"] == "last200"
    assert asker.asked[-1] == "last200"
    assert result["checks"]["first40"] == {"good": False, "by": "no usable action"}


def test_all_when_no_cut_is_good_and_two_hundred_does_not_cut():
    asker = FakeAsker({"all": SAME, **{c: OTHER for c in tl.FORTY}}, {f"{c} vs uncut": False for c in tl.FORTY})
    result = run(tl.trim_cascade(asker, set(tl.FORTY)))
    assert result["label"] == "all"
    assert "last200" not in asker.asked


def test_an_unusable_uncut_reply_leaves_the_turn_out():
    asker = FakeAsker({"all": {"commands": None, "final_text": None}, **{c: SAME for c in tl.FORTY}}, {})
    with pytest.raises(tl.UnusableUncut):
        run(tl.trim_cascade(asker, set(tl.FORTY)))


def test_paired_bodies_are_at_temperature_zero_at_the_routing_level():
    built = {"body": {"messages": ["uncut"], "chat_template_kwargs": {"enable_thinking": True}},
             "trimmed": {"last40": {"messages": ["cut"], "chat_template_kwargs": {"enable_thinking": True}}},
             "kwargs": {"off": {"enable_thinking": False}, "xhigh": {"enable_thinking": True, "reasoning_effort": "xhigh"}}}
    off = tl.paired_body(built, "last40", "off")
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert off["temperature"] == 0 and off["max_completion_tokens"] == 8192 and off["messages"] == ["cut"]
    xhigh = tl.paired_body(built, "all", "xhigh")
    assert xhigh["temperature"] == 0 and xhigh["messages"] == ["uncut"]
    assert xhigh["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    assert "max_completion_tokens" not in xhigh


def test_reference_prompt_is_the_routing_request_at_the_label_level():
    row = {"id": "t", "label": "off", "recorded": {"prompt_tokens": 1000},
           "asks": [{"level": "off", "sample": 1, "prompt_tokens": 964}]}
    assert tl.reference_prompt(row) == 964
    assert tl.reference_prompt({**row, "label": "xhigh"}) == 1000
    with pytest.raises(ValueError):
        tl.reference_prompt({**row, "label": "low"})


def test_sample_is_fixed_and_near_the_rate():
    ids = [f"trial:{n}" for n in range(4000)]
    picked = [i for i in ids if tl.sampled(i, 0.25)]
    assert picked == [i for i in ids if tl.sampled(i, 0.25)]
    assert 800 < len(picked) < 1200
    assert all(tl.sampled(i, 1.0) for i in ids)


def test_moved_trial_folders(tmp_path):
    (tmp_path / "new" / "runs" / "t__1").mkdir(parents=True)
    moved = [(str(tmp_path / "old"), str(tmp_path / "new"))]
    assert tl.resolve_trial(str(tmp_path / "old" / "runs" / "t__1"), moved) == tmp_path / "new" / "runs" / "t__1"
    with pytest.raises(FileNotFoundError):
        tl.resolve_trial(str(tmp_path / "old" / "runs" / "t__2"), moved)


def test_routing_rows_are_read_as_the_file_grows(tmp_path):
    labeller = tl.TrimLabeller.__new__(tl.TrimLabeller)
    labeller.offsets = {}
    path = tmp_path / "routing.jsonl"
    path.write_text(json.dumps({"kind": "turn", "id": "a"}) + "\n" + json.dumps({"kind": "trial"}) + "\n"
                    + '{"kind": "turn", "id": "b"')
    assert [r["id"] for r in labeller.new_routing_rows(str(path))] == ["a"]
    with path.open("a") as handle:
        handle.write(', "x": 1}\n')
    assert [r["id"] for r in labeller.new_routing_rows(str(path))] == ["b"]
    assert labeller.new_routing_rows(str(path)) == []


def _turn(turn_id, label, shown, routing="off", saved=500, cls="read"):
    asks = [] if label == "all" else [{"cut": label, "outcome": "ok", "prompt_saved": saved}]
    checks = {"last40": {"good": label == "last40", "by": "intent" if label == "last40" else "judge"}}
    return {"kind": "turn", "id": turn_id, "label": label, "shown_lines": shown, "class": cls, "routing_label": routing,
            "asks": asks, "checks": checks, "reference_prompt": 10000,
            "recording_machine": "qwen3.8-27b-nvfp4@b200-gpu0"}


def test_stats_shares_by_length_class_and_level(tmp_path):
    rows = [_turn("a", "last40", 60), _turn("b", "all", 150, routing="xhigh"), _turn("c", "last200", 900, cls="test"),
            {"kind": "short", "id": "d", "shown_lines": 3}]
    text = tl.stats_markdown(rows)
    assert "labelled: 3" in text
    assert "| all turns | 3 | 33.3% | 0.0% | 0.0% | 33.3% | 33.3% | 333 |" in text
    assert "| 41-100 | 1 | 100.0% |" in text
    assert "| 501-2000 | 1 | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |" in text
    assert "saved by the labels: 1000" in text
    path = tmp_path / "x.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows + rows[:1]))
    with pytest.raises(ValueError):
        tl.read_rows([path])


def test_join_puts_the_trim_label_on_the_turns_first_state():
    turn = {**_turn("a", "first40", 120), "session_id": "s", "turn": 4, "task": "t", "recording_machine": "m",
            "source_host": "b200", "total_lines": 130}
    stage3 = [{"session": "s", "turn": 4, "level": "tool", "page": 1, "decision": 9, "state": "later"},
              {"session": "s", "turn": 4, "level": "tool", "page": 1, "decision": 8, "state": "first"},
              {"session": "s", "turn": 4, "level": "argument", "page": 1, "decision": 7, "state": "arg"}]
    rows, counts = tl.join_rows([turn, {**turn, "id": "b", "turn": 5}, {"kind": "short", "id": "c"}], stage3)
    assert counts == {"labelled turns": 2, "joined": 1, "no stage-3 row": 1}
    assert rows[0]["state"] == "first"
    assert rows[0]["trim"] == {"label": "first40", "total_lines": 130, "shown_lines": 120, "routing_label": "off",
                               "checks": turn["checks"]}


def test_length_buckets():
    assert tl.length_bucket(41) == "41-100"
    assert tl.length_bucket(2000) == "501-2000"
    assert tl.length_bucket(2001) == "2001+"
    with pytest.raises(ValueError):
        tl.length_bucket(40)


def test_builder_output_on_a_real_session_shape(tmp_path):
    """trim_requests.ts rebuilds a request with the newest output cut to its last 40 lines (pi's own code)."""
    import subprocess

    repo = Path(__file__).resolve().parents[3]
    output = "\n".join(f"line {n}" for n in range(1, 61)) + "\n"
    session = tmp_path / "s.jsonl"
    entries = [
        {"type": "session", "version": 3, "id": "s1", "timestamp": "2026-10-04T00:00:00.000Z", "cwd": "/app"},
        {"type": "message", "id": "u1", "parentId": None, "timestamp": "2026-10-04T00:00:00.000Z",
         "message": {"role": "user", "content": "Do the task.", "timestamp": 1}},
        {"type": "message", "id": "a1", "parentId": "u1", "timestamp": "2026-10-04T00:00:01.000Z",
         "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "c1", "name": "bash",
                                                        "arguments": {"command": "seq 60"}}],
                     "api": "openai-completions", "provider": "harbor-endpoint", "model": "qwen3.8-27b",
                     "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
                               "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
                     "stopReason": "toolUse", "timestamp": 2}},
        {"type": "message", "id": "r1", "parentId": "a1", "timestamp": "2026-10-04T00:00:02.000Z",
         "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "bash",
                     "content": [{"type": "text", "text": output}], "isError": False, "timestamp": 3}},
        {"type": "message", "id": "a2", "parentId": "r1", "timestamp": "2026-10-04T00:00:03.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}],
                     "api": "openai-completions", "provider": "harbor-endpoint", "model": "qwen3.8-27b",
                     "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
                               "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
                     "stopReason": "stop", "timestamp": 4}},
    ]
    session.write_text("".join(json.dumps(e) + "\n" for e in entries))
    jobs = "".join(json.dumps({"id": i, "session": str(session), "entryId": i}) + "\n" for i in ("a1", "a2"))
    done = subprocess.run(["node", str(repo / "results/imitation/scripts/trim_requests.ts")], input=jobs,
                          capture_output=True, text=True, check=True)
    first, second = [json.loads(line) for line in done.stdout.splitlines()]
    assert first == {"id": "a1", "eligible": False, "reason": "no tool output just before this turn",
                     "shownLines": None}
    assert second["eligible"] and second["shownLines"] == 60
    assert set(second["trimmed"]) == {"last40", "first40", "first20last20"}
    tool = [m for m in second["trimmed"]["last40"]["messages"] if m["role"] == "tool"][0]
    expected = "\n".join(f"line {n}" for n in range(21, 61)) + "\n\n[Showing lines 21-60 of 60. 20 earlier lines not shown.]"
    assert tool["content"] == expected
    assert set(second["kwargs"]) == {"off", "low", "medium", "xhigh"}


# ------------------------------------------------------------------------------------------------ cross-format re-asks


def _discovering(tmp_path, family, allow):
    trial = tmp_path / "runs" / "t__1"
    (trial / "agent" / "pi" / "sessions").mkdir(parents=True, exist_ok=True)
    (trial / "agent" / "pi" / "sessions" / "s.jsonl").write_text("{}\n")
    row = {"kind": "turn", "id": "t__1:e1", "trial": "t__1", "trial_dir": str(trial), "task": "t", "class": "read",
           "turn": 1, "recording_machine": "qwen3.8-27b-fp8@casdgx01-gpu4", "session_id": "s", "entry_id": "e1",
           "commands": ["ls"], "final_text": None, "recorded": {}, "label": "off"}
    routing = tmp_path / "routing-labels-casdgx01-b200.jsonl"
    routing.write_text(json.dumps(row) + "\n")
    labeller = tl.TrimLabeller.__new__(tl.TrimLabeller)
    labeller.options = SimpleNamespace(family=family, allow_cross_format=allow, sample_rate=1.0)
    labeller.sources = [("casdgx01", [str(routing)])]
    labeller.moved = []
    labeller.done = set()
    labeller.offsets = {}
    labeller.trial_server = {}
    server = SimpleNamespace(url="u", queued_turns=0)
    labeller.servers = [server]
    labeller.queues = {"u": asyncio.Queue()}
    return labeller


def test_a_turn_of_the_other_format_needs_allow_cross_format(tmp_path):
    with pytest.raises(ValueError, match="answers nvfp4 sessions only"):
        _discovering(tmp_path, "nvfp4", False).discover()
    labeller = _discovering(tmp_path, "nvfp4", True)
    assert labeller.discover() == 1
    assert labeller.queues["u"].get_nowait()["recording_format"] == "fp8"
    assert _discovering(tmp_path, "fp8", False).discover() == 1


def test_trim_file_names():
    assert tl.trim_file("/o", "casdgx01", None) == Path("/o/trim-labels-casdgx01.jsonl")
    assert tl.trim_file("/o", "casdgx01", "b200") == Path("/o/trim-labels-casdgx01-b200.jsonl")


def test_formats_and_the_stats_split_by_re_ask_format():
    old = _turn("a", "last40", 60)
    cross = {**_turn("b", "all", 60), "recording_machine": "qwen3.8-27b-fp8@casdgx01-gpu4", "recording_format": "fp8",
             "reask_format": "nvfp4"}
    assert tl.formats(old) == ("nvfp4", "nvfp4") and tl.formats(cross) == ("fp8", "nvfp4")
    text = tl.stats_markdown([old, cross])
    assert "## Labels by re-ask format (recorded -> re-asked)" in text
    assert "| fp8 -> nvfp4 | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |" in text
    assert "| nvfp4 -> nvfp4 | 1 | 100.0% |" in text
    half = dict(cross)
    del half["reask_format"]
    with pytest.raises(ValueError, match="no reask_format"):
        tl.formats(half)


def test_a_turn_written_by_two_trim_labellers_is_an_error(tmp_path):
    first, second = tmp_path / "trim-labels-casdgx01.jsonl", tmp_path / "trim-labels-casdgx01-b200.jsonl"
    first.write_text(json.dumps(_turn("a", "last40", 60)) + "\n")
    second.write_text(json.dumps({"kind": "short", "id": "b", "shown_lines": 3}) + "\n")
    assert len(tl.read_rows([first, second])) == 2
    second.write_text(json.dumps({"kind": "short", "id": "a", "shown_lines": 3}) + "\n")
    with pytest.raises(ValueError, match="1 turns appear twice, e.g. a"):
        tl.read_rows([first, second])
