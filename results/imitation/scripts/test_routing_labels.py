"""Tests of routing_labels.py. Run: uv run --with pytest --with pytest-asyncio --with aiohttp==3.12.15 --with tokenizers
python -m pytest test_routing_labels.py -q (from results/imitation/scripts)."""

import asyncio
import json
from pathlib import Path

import pytest

import routing_labels as rl


# ------------------------------------------------------------------------------------------------ judge reply parsing


def test_verdict_yes_and_no_with_reason():
    assert rl.parse_verdict("YES. Both list the folder first.") == (True, "Both list the folder first.")
    assert rl.parse_verdict("\n\nNO - the alternative edits a different file.") == (
        False, "the alternative edits a different file.")
    assert rl.parse_verdict("NO, it skips the build.") == (False, "it skips the build.")


def test_verdict_on_two_lines():
    assert rl.parse_verdict("YES\nBoth read the same file.") == (True, "Both read the same file.")


@pytest.mark.parametrize("reply", ["Maybe. Hard to say.", "**YES** fine", "Yes. lower case", "YESTERDAY", "", "  "])
def test_verdict_anything_else_raises(reply):
    with pytest.raises(ValueError):
        rl.parse_verdict(reply)


# ------------------------------------------------------------------------------------------------ prompt and view


def _messages(steps):
    messages = [{"role": "developer", "content": "system prompt"}, {"role": "user", "content": "Do the task."}]
    for number, (command, output) in enumerate(steps):
        call_id = f"call-{number}"
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]})
        messages.append({"role": "tool", "content": output, "tool_call_id": call_id})
    return messages


def test_recent_steps_are_the_last_three_calls_with_output_tails():
    steps = rl.recent_steps(_messages([("ls /app", "a.py"), ("cat a.py", "x" * 1000 + "END"), ("pwd", ""),
                                       ("whoami", "root")]))
    assert [command for command, _ in steps] == ["cat a.py", "pwd", "whoami"]
    assert steps[0][1] == ("x" * 1000 + "END")[-600:]
    assert steps[1][1] == ""


def test_recent_steps_of_the_first_turn_are_empty():
    assert rl.recent_steps(_messages([])) == []


# The owner's validated judge prompt (/private/tmp/claude-501/judge-test/judge.py), reproduced exactly.
def test_judge_messages_match_the_validated_prompt():
    messages = rl.judge_messages("Fix the bug.", [("ls", "a.py"), ("cat a.py", "")], ["pytest -q"], ["python -m pytest"])
    assert messages[0] == {"role": "system", "content": rl.JUDGE_SYSTEM}
    assert rl.JUDGE_SYSTEM.startswith("You review the work of an AI assistant that solves tasks on a Linux computer")
    assert rl.JUDGE_SYSTEM.endswith("Answer with YES or NO on the first line. On the second line, give one sentence "
                                    "explaining why.")
    assert messages[1] == {"role": "user", "content": (
        "TASK:\nFix the bug.\n\nMOST RECENT STEPS (oldest first):\nCommand:\nls\nEnd of output:\na.py\n\n"
        "Command:\ncat a.py\nEnd of output:\n(no output)\n\nSTEP 1:\npytest -q\n\nSTEP 2:\npython -m pytest\n\n"
        "Would Step 2 serve the task as well as Step 1 at this moment? Answer YES or NO on the first line, then one "
        "sentence why.")}


def test_judge_messages_without_steps_and_with_a_final_answer():
    user = rl.judge_messages("T" * 4000, [], ["a" * 3000], ["Done."])[1]["content"]
    assert "MOST RECENT STEPS (oldest first):\n(none yet)\n" in user
    assert "TASK:\n" + "T" * 3000 + "\n\n" in user and "T" * 3001 not in user
    assert "STEP 1:\n" + "a" * 2500 + "\n\nSTEP 2:\nDone.\n" in user


def test_step_pieces():
    assert rl.step_pieces(["ls", "pwd"], None) == ["ls", "pwd"]
    assert rl.step_pieces([], "All done.") == ["All done."]


# ------------------------------------------------------------------------------------------------ variants and checks


def test_variant_bodies():
    built = {"body": {"model": "m", "messages": [], "max_completion_tokens": 32768, "stream": True,
                      "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True,
                                               "reasoning_effort": "xhigh"}},
             "kwargs": {"off": {"enable_thinking": False, "preserve_thinking": True},
                        "low": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"},
                        "medium": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "medium"},
                        "xhigh": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}}}
    off = rl.variant_body(built, "off")
    assert off["temperature"] == 0 and off["max_completion_tokens"] == 8192
    assert off["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": True}
    medium = rl.variant_body(built, "medium")
    assert "temperature" not in medium and medium["max_completion_tokens"] == 32768
    assert medium["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert rl.variant_body(built, "xhigh") == built["body"]
    assert built["body"]["chat_template_kwargs"]["reasoning_effort"] == "xhigh"  # not changed in place


def test_family_of_driver_builds():
    assert rl.family("qwen3.8-27b-fp8@casdgx01-gpu6") == "fp8"
    assert rl.family("qwen3.8-27b-nvfp4@b200-gpu3-vllm0.29") == "nvfp4"
    with pytest.raises(ValueError):
        rl.family("glm-5.3@b200")


def test_calibration_choice_is_fixed_and_near_the_rate():
    ids = [f"trial{i}:entry{i}" for i in range(20000)]
    chosen = [i for i in ids if rl.in_calibration(i, 0.05)]
    assert chosen == [i for i in ids if rl.in_calibration(i, 0.05)]
    assert 800 < len(chosen) < 1200


def test_waiting_requests_from_metrics():
    text = ("# HELP x\nvllm:num_requests_running{engine=\"0\",model_name=\"q\"} 2.0\n"
            "vllm:num_requests_waiting{engine=\"0\",model_name=\"q\"} 5.0\n"
            "vllm:num_requests_waiting_by_reason{engine=\"0\",reason=\"capacity\"} 5.0\n")
    assert rl.waiting_requests(text) == 5
    with pytest.raises(ValueError):
        rl.waiting_requests("# nothing here\n")


# ------------------------------------------------------------------------------------------------ cascade


class FakeAsker:
    """Answers each level with a fixed command list (None: no usable action) and judges by a fixed table."""

    def __init__(self, actions, verdicts=None):
        self.actions = actions
        self.verdicts = verdicts or {}
        self.asked = []
        self.judged = []

    async def ask(self, level, sample):
        self.asked.append((level, sample))
        commands = self.actions[(level, sample)] if (level, sample) in self.actions else self.actions[level]
        return {"level": level, "sample": sample, "commands": commands, "final_text": None}

    async def judge(self, reference, alternative, key):
        self.judged.append(key)
        return {"verdict": self.verdicts.get(key, False), "reason": "because"}


RECORDED = {"commands": ["cat /app/a.py"], "final_text": None}


def run(coro):
    return asyncio.run(coro)


def test_cascade_stops_at_off_when_intent_matches():
    asker = FakeAsker({"off": ["cat -n a.py | head -50"]})
    result = run(rl.cascade(asker, RECORDED, calibration=False))
    assert result["label"] == "off"
    assert asker.asked == [("off", 1)] and asker.judged == []
    assert result["checks"]["off"]["by"] == "intent"


def test_cascade_judges_a_different_intent_and_goes_on():
    asker = FakeAsker({"off": ["ls /app"], "low": ["grep -n def /app/a.py"], "medium": ["cat /app/a.py"]},
                      verdicts={"off vs recorded": False, "low vs recorded": False})
    result = run(rl.cascade(asker, RECORDED, calibration=False))
    assert result["label"] == "medium"
    assert asker.asked == [("off", 1), ("low", 1), ("medium", 1)]
    assert asker.judged == ["off vs recorded", "low vs recorded"]


def test_cascade_judge_yes_is_good():
    asker = FakeAsker({"off": ["ls /app"]}, verdicts={"off vs recorded": True})
    result = run(rl.cascade(asker, RECORDED, calibration=False))
    assert result["label"] == "off" and result["checks"]["off"]["by"] == "judge"


def test_cascade_unusable_reply_is_not_good_and_not_judged():
    asker = FakeAsker({"off": None, "low": None, "medium": None})
    result = run(rl.cascade(asker, RECORDED, calibration=False))
    assert result["label"] == "xhigh" and asker.judged == []
    assert result["checks"]["off"]["by"] == "no usable action"


def test_calibration_asks_everything_and_compares_with_all_three_xhigh_actions():
    asker = FakeAsker({"off": ["ls /app"], "low": ["ls /app"], "medium": ["cat /app/a.py"],
                       ("xhigh", 2): ["ls /app"], ("xhigh", 3): ["cat /app/b.py"]},
                      verdicts={"off vs recorded": False, "low vs recorded": False, "xhigh-2 vs recorded": False,
                                "xhigh-3 vs recorded": False, "off vs xhigh-3": False, "low vs xhigh-3": False})
    result = run(rl.cascade(asker, RECORDED, calibration=True))
    assert sorted(asker.asked) == [("low", 1), ("medium", 1), ("off", 1), ("xhigh", 2), ("xhigh", 3)]
    assert result["label"] == "medium"
    # off matches xhigh sample 2 by intent: good against any of the three, not against the recorded one.
    assert result["checks"]["off"]["good"] is False
    assert result["any_xhigh"]["off"] is True
    assert result["any_xhigh"]["medium"] is True
    assert result["xhigh_self"] == {"2": False, "3": False}


# ------------------------------------------------------------------------------------------------ recorded turns


def _trial(tmp_path, assistants, lines, cut=False):
    trial = tmp_path / "s1" / "round1" / "job" / "task__abc"
    (trial / "agent" / "pi" / "sessions").mkdir(parents=True)
    config = {"task": {"path": "task"}, "agent": {"kwargs": {"tarball": "/x/jeff-pi-scout-6498fcf8d.tgz",
                                                             "thinking": "high"}, "env": {
        "JEFF_FIRST_MODE": "record", "JEFF_FIRST_THINKING_ROUTER": "fixed:xhigh",
        "JEFF_FIRST_DRIVER_BUILD": "qwen3.8-27b-fp8@casdgx01-gpu1"}}}
    (trial / "config.json").write_text(json.dumps(config))
    result = {"exception_info": {"exception_type": "AgentTimeoutError"} if cut else None}
    (trial / "result.json").write_text(json.dumps(result))
    entries = [{"type": "session", "id": "sess-1", "cwd": "/app", "timestamp": "2026-10-04T08:00:00.000Z"},
               {"type": "message", "id": "u1", "parentId": None, "timestamp": "2026-10-04T08:00:00.000Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "Do it."}]}}]
    for number, (stop, commands, usage_input) in enumerate(assistants):
        content = [{"type": "thinking", "thinking": "hmm"}]
        content += [{"type": "toolCall", "id": f"c{number}{k}", "name": "bash", "arguments": {"command": c}}
                    for k, c in enumerate(commands)]
        if not commands:
            content.append({"type": "text", "text": "Done."})
        entries.append({"type": "message", "id": f"a{number}", "parentId": "u1", "timestamp": "2026-10-04T08:00:01.000Z",
                        "message": {"role": "assistant", "content": content, "stopReason": stop,
                                    "timestamp": 1791100800000,
                                    "usage": {"input": usage_input, "output": 10, "cacheRead": 0, "reasoning": 5}}})
    (trial / "agent" / "pi" / "sessions" / "s.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    (trial / "agent" / "jeff-first-trace.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return trial


def _line(turn, stop, usage_input, attempt=1, outcome="kept"):
    return {"kind": "qwen_request", "session_id": "sess-1", "turn": turn, "attempt": attempt, "outcome": outcome,
            "router": "fixed:xhigh", "thinking_level": "xhigh",
            "sent": {"enable_thinking": True, "reasoning_effort": "xhigh"}, "stop_reason": stop,
            "usage": {"input": usage_input, "output": 10, "thinking_tokens": 5, "cache_read": 0, "cache_write": 0},
            "timings_ms": {"model": 1234.0}}


def test_recorded_turns_pair_session_and_trace(tmp_path):
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100), ("length", [], 200), ("stop", [], 300)],
                   [_line(1, "toolUse", 100), {"kind": "record", "turn": 1}, _line(2, "length", 200),
                    _line(3, "stop", 300)])
    turns, skipped = rl.recorded_turns(trial)
    assert all(t["legacy"] is False for t in turns)
    assert [(t["turn"], t["entry_id"], t["commands"], t["final_text"]) for t in turns] == [
        (1, "a0", ["ls"], None), (3, "a2", [], "Done.")]
    assert turns[0]["session_id"] == "sess-1" and turns[0]["recorded"]["prompt_tokens"] == 100
    assert turns[0]["recorded"]["model_ms"] == 1234.0
    assert skipped == {"reply hit the output cap (length)": 1}


def test_recorded_turns_use_the_kept_attempt(tmp_path):
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100)],
                   [_line(1, "toolUse", 100, attempt=1, outcome="discarded"), _line(1, "toolUse", 100, attempt=2)])
    turns, _ = rl.recorded_turns(trial)
    assert len(turns) == 1 and turns[0]["turn"] == 1


def test_a_trial_with_a_call_to_another_tool_is_left_out(tmp_path):
    # Qwen on NVFP4 sometimes names a tool that does not exist (seen: "bbyte"); Jeff-Code answers with an error. ceiling.py's
    # turn classifier raises on such a session, so the whole trial is left out (counted).
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100), ("toolUse", ["pwd"], 200)],
                   [_line(1, "toolUse", 100), _line(2, "toolUse", 200)])
    session = next((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    entries = [json.loads(line) for line in session.read_text().splitlines()]
    entries[-1]["message"]["content"][1]["name"] = "bbyte"
    session.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    turns, skipped = rl.recorded_turns(trial)
    assert turns == []
    assert skipped == {rl.OTHER_TOOL: 2}


def _legacy(tmp_path, records):
    """A build-4abde3ece trial: no router, Jeff-Code's thinking level medium (sent as enable_thinking only, so the chat
    template's default effort xhigh applied), schema-4 record lines."""
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100), ("stop", [], 300)], records)
    config = json.loads((trial / "config.json").read_text())
    del config["agent"]["env"]["JEFF_FIRST_THINKING_ROUTER"]
    config["agent"]["kwargs"] = {"tarball": "/home/x/jeff-pi-scout-4abde3ece.tgz", "thinking": "medium",
                                 "thinking_format": "qwen-chat-template", "max_output_tokens": 32768}
    (trial / "config.json").write_text(json.dumps(config))
    return trial


def _record(turn, stop, usage_input):
    return {"schema": "jeff-first-trace/4", "kind": "record", "session_id": "sess-1", "turn": turn,
            "action": {"stop_reason": stop, "tool_calls": []},
            "model_usage": {"input": usage_input, "output": 10, "cache_read": 0, "cache_write": 0},
            "timings_ms": {"lists": 1.0, "model": 999.0}}


def test_legacy_trials_pair_turns_with_record_lines(tmp_path):
    trial = _legacy(tmp_path, [_record(1, "toolUse", 100), _record(2, "stop", 300)])
    turns, skipped = rl.recorded_turns(trial)
    assert [(t["turn"], t["legacy"]) for t in turns] == [(1, True), (2, True)]
    assert turns[0]["recorded"]["model_ms"] == 999.0 and skipped == {}


def test_legacy_trials_of_another_build_raise(tmp_path):
    trial = _legacy(tmp_path, [_record(1, "toolUse", 100), _record(2, "stop", 300)])
    config = json.loads((trial / "config.json").read_text())
    config["agent"]["kwargs"]["tarball"] = "/x/jeff-pi-scout-adad96753.tgz"
    (trial / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        rl.recorded_turns(trial)


def test_prompt_check_per_level():
    rl.check_prompt("t", "off", -36)
    rl.check_prompt("t", "xhigh", 0)
    rl.check_prompt("t", "off", None)  # a request cut for looping reports no count
    rl.check_prompt("t", "off", -37)  # the effort line's token count depends on its neighbours
    with pytest.raises(rl.PromptMismatch):
        rl.check_prompt("t", "xhigh", 1)


def test_a_crashed_agent_may_leave_one_traced_reply_unsaved(tmp_path):
    # Seen 2026-10-04: Jeff-Code exited (NonZeroAgentExitCodeError) after tracing turn 24 and before saving its reply.
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100)], [_line(1, "toolUse", 100), _line(2, "toolUse", 200)])
    (trial / "result.json").write_text(json.dumps({"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}}))
    turns, _ = rl.recorded_turns(trial)
    assert [t["turn"] for t in turns] == [1]


def test_a_session_file_with_a_broken_line_inside_is_left_out(tmp_path):
    # Seen 2026-10-04 (music-harmony, a timed-out hub trial): line 97 of 113 cut in the middle of a string.
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100), ("stop", [], 200)],
                   [_line(1, "toolUse", 100), _line(2, "stop", 200)])
    session = next((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    lines = session.read_text().splitlines()
    lines[2] = lines[2][:40]
    session.write_text("\n".join(lines) + "\n")
    turns, skipped = rl.recorded_turns(trial)
    assert turns == [] and skipped == {rl.BROKEN_SESSION: 1}


def test_recorded_turns_mismatch_raises(tmp_path):
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100)], [_line(1, "toolUse", 999)])
    with pytest.raises(ValueError):
        rl.recorded_turns(trial)


def test_recorded_turns_other_router_raises(tmp_path):
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100)], [_line(1, "toolUse", 100)])
    config = json.loads((trial / "config.json").read_text())
    config["agent"]["env"]["JEFF_FIRST_THINKING_ROUTER"] = "fixed:medium"
    (trial / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        rl.recorded_turns(trial)


def test_finished_trials_need_a_trial_result(tmp_path):
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100)], [_line(1, "toolUse", 100)])
    running = tmp_path / "s1" / "round2" / "job" / "task__def"
    running.mkdir(parents=True)
    (tmp_path / "s1" / "round1" / "job" / "result.json").write_text("{}")  # the job's own result, not a trial's
    assert rl.finished_trials(Path(tmp_path)) == [trial]


def test_finished_trials_at_any_depth(tmp_path):
    # The end-to-end test's layout: <stream>/<arm>/run1/<job>/<trial>/.
    trial = tmp_path / "rtx" / "xhigh" / "run1" / "job" / "task__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text("{}")
    (trial / "agent" / "result.json").write_text("{}")  # inside a trial: not a trial
    (tmp_path / "rtx" / "xhigh" / "run1" / "job" / "result.json").write_text("{}")
    assert rl.finished_trials(tmp_path) == [trial]


# ------------------------------------------------------------------------------------------------ join, stats, check


def _label(session="sess-1", turn=1, label="off", calibration=False, cls="information", judges=()):
    checks = {"off": {"good": label == "off", "by": "intent" if label == "off" else "judge"}}
    if label != "off" or calibration:
        checks["low"] = {"good": label == "low", "by": "judge"}
    if label not in ("off", "low") or calibration:
        checks["medium"] = {"good": label == "medium", "by": "judge"}
    asks = [{"level": level, "sample": 1, "commands": ["ls"], "final_text": None, "outcome": "ok",
             "completion_tokens": 10, "prompt_diff": {"off": -36, "low": -12, "medium": -38}[level],
             "total_s": 1.0, "first_s": 0.1} for level in checks]
    row = {"kind": "turn", "id": f"t:{session}:{turn}", "session_id": session, "turn": turn, "task": "fix-bug",
           "trial": "fix-bug__a", "recording_machine": "qwen3.8-27b-nvfp4@b200-gpu0-vllm0.29", "source_host": "b200",
           "replay_machine": "b200-nvfp4", "class": cls, "commands": ["cat a.py"], "final_text": None,
           "recorded": {"prompt_tokens": 100, "output_tokens": 50, "reasoning_tokens": 40, "model_ms": 2000.0,
                        "stop_reason": "toolUse"},
           "calibration": calibration, "label": label, "checks": checks, "asks": asks, "judges": list(judges)}
    if calibration:
        row["asks"] += [{"level": "xhigh", "sample": s, "commands": ["cat a.py"], "final_text": None, "outcome": "ok",
                         "completion_tokens": 50, "prompt_diff": 0, "total_s": 2.0, "first_s": 0.1} for s in (2, 3)]
        row["any_xhigh"] = {"off": True, "low": True, "medium": True}
        row["xhigh_self"] = {"2": True, "3": False}
    return row


def _stage3_row(session, turn, decision, level, page, state):
    return {"session": session, "turn": turn, "decision": decision, "level": level, "page": page, "state": state,
            "task": "fix-bug", "machine": "m", "stage": 3}


def test_join_puts_the_routing_label_on_the_turns_first_state():
    stage3 = [_stage3_row("sess-1", 1, 4, "argument", 1, "S-arg"), _stage3_row("sess-1", 1, 4, "tool", 2, "S1-p2"),
              _stage3_row("sess-1", 1, 4, "tool", 1, "S1"), _stage3_row("sess-1", 1, 5, "tool", 1, "S1-stint"),
              _stage3_row("sess-1", 2, 6, "tool", 1, "S2")]
    rows, counts = rl.join_rows([_label(turn=1, label="medium"), _label(turn=3)], stage3)
    assert counts == {"labelled turns": 2, "joined": 1, "no stage-3 row": 1}
    (row,) = rows
    assert row["state"] == "S1" and row["session"] == "sess-1" and row["turn"] == 1
    assert row["routing"]["label"] == "medium"
    assert (row["routing"]["off_ok"], row["routing"]["low_ok"], row["routing"]["medium_ok"]) == (False, False, True)
    assert row["routing"]["xhigh_actions"] == [{"commands": ["cat a.py"], "final_text": None}]


def test_join_of_a_calibration_turn_carries_all_three_xhigh_actions():
    rows, _ = rl.join_rows([_label(calibration=True)], [_stage3_row("sess-1", 1, 0, "tool", 1, "S")])
    assert len(rows[0]["routing"]["xhigh_actions"]) == 3
    assert rows[0]["routing"]["any_xhigh"] == {"off": True, "low": True, "medium": True}


def test_stats_report_counts_labels_classes_and_calibration():
    labels = [_label(turn=1), _label(turn=2, label="xhigh", cls="write/edit"), _label(turn=3, calibration=True)]
    excluded = [{"kind": "excluded", "id": "t:x", "reason": rl.JUDGE_REFUSED}]
    text = rl.stats_markdown(labels, excluded)
    assert f"Turns left out: 1 ({rl.JUDGE_REFUSED}: 1)" in text
    assert "# Routing labels" in text
    assert "| all turns | 3 |" in text
    assert "write/edit" in text and "## Calibration" in text


def test_read_labels_separates_excluded_turns(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [_label(turn=1), {"kind": "trial", "trial_dir": "x"},
                                                         {"kind": "excluded", "id": "t:y", "reason": "r"}]) + "\n")
    turns, excluded = rl.read_rows([path])
    assert [t["turn"] for t in turns] == [1] and [e["id"] for e in excluded] == ["t:y"]


def test_judge_check_sample_is_fixed():
    judged = [{"key": "off vs recorded", "model": "m", "verdict": True, "reason": "r", "messages": [{}, {}],
               "task": "T", "steps": [], "reference": ["a"], "alternative": ["b"]}]
    labels = [{**_label(turn=i), "judges": judged} for i in range(50)]
    first = rl.judge_check_pairs(labels, 10)
    assert [p["pair"] for p in first] == [p["pair"] for p in rl.judge_check_pairs(labels, 10)]
    assert len(first) == 10 and len({p["pair"] for p in first}) == 10
    with pytest.raises(ValueError):
        rl.judge_check_pairs(labels, 51)


# ------------------------------------------------------------------------------------------------ free match (owner)


@pytest.mark.parametrize(("a", "b"), [
    (["cat /app/a.py"], ["cat -n a.py | head -50"]),  # same file read
    (["ls /app"], ["ls -la /app"]),  # same folder listed
    (["python3  run.py --x 'a b'"], ['python3 run.py --x "a b"']),  # same command text after normalising
])
def test_free_match_when_the_full_target_is_the_same(a, b):
    assert rl.free_match(a, b)


@pytest.mark.parametrize(("a", "b"), [
    (["python3 - <<'EOF'\nprint(1)\nEOF"], ["python3 - <<'EOF'\nprint(2)\nEOF"]),  # inline scripts: judge
    (["python3 -c 'print(1)'"], ["python3 -c 'print(2)'"]),
    (["cat > /app/a.py <<'EOF'\nx = 1\nEOF"], ["cat > /app/a.py <<'EOF'\nx = 2\nEOF"]),  # file writes: judge
    (["sed -i 's/a/b/' /app/a.py"], ["sed -i 's/a/c/' /app/a.py"]),  # edits: judge
    ([], []),  # two final answers (texts compared by the judge)
    (["cat /app/a.py"], ["cat /app/b.py"]),
])
def test_no_free_match_for_scripts_writes_edits_and_answers(a, b):
    assert not rl.free_match(a, b)


def test_cascade_sends_a_different_inline_script_to_the_judge():
    recorded = {"commands": ["python3 - <<'EOF'\nprint(1)\nEOF"], "final_text": None}
    asker = FakeAsker({"off": ["python3 - <<'EOF'\nprint(2)\nEOF"]}, verdicts={"off vs recorded": True})
    result = run(rl.cascade(asker, recorded, calibration=False))
    assert asker.judged == ["off vs recorded"] and result["checks"]["off"]["by"] == "judge"


def test_task_id_of_a_hub_trial(tmp_path):
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100)], [_line(1, "toolUse", 100)])
    config = json.loads((trial / "config.json").read_text())
    config["task"] = {"name": "terminal-bench-science/onsager-ising-lean", "ref": "sha256:a7",
                      "source": "terminal-bench-science/terminal-bench-science"}
    (trial / "config.json").write_text(json.dumps(config))
    turns, _ = rl.recorded_turns(trial)
    assert turns[0]["task"] == "terminal-bench-science/terminal-bench-science:onsager-ising-lean"
    tokenizer = rl.Tokenizer.from_file("/private/tmp/claude-501/stage3/tokenizer/Qwen3.5-0.8B/tokenizer.json")
    assert rl.ceiling.read_trial(trial, tokenizer)["task"] == "terminal-bench-science/onsager-ising-lean"


# ------------------------------------------------------------------------------------------------ cross-format re-asks


def test_recorded_turns_carry_the_recording_format(tmp_path):
    trial = _trial(tmp_path, [("toolUse", ["ls"], 100)], [_line(1, "toolUse", 100)])
    turns, _ = rl.recorded_turns(trial)
    assert turns[0]["recording_format"] == "fp8"


def test_a_session_of_the_other_format_needs_allow_cross_format():
    rl.check_format("t", "fp8", "fp8", False)
    rl.check_format("t", "fp8", "nvfp4", True)
    with pytest.raises(ValueError, match="answers nvfp4 sessions only"):
        rl.check_format("t", "fp8", "nvfp4", False)


def test_label_file_names_and_done_ids(tmp_path):
    assert rl.label_file(tmp_path, "casdgx01", None).name == "routing-labels-casdgx01.jsonl"
    assert rl.label_file(tmp_path, "casdgx01", "b200").name == "routing-labels-casdgx01-b200.jsonl"
    path = tmp_path / "seed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [_label(turn=1), {"kind": "trial", "trial_dir": "x"},
                                                         {"kind": "excluded", "id": "t:y", "reason": "r"}]) + "\n")
    assert rl.done_ids(path) == {"t:sess-1:1", "t:y"}


def _cross(turn, label, reask):
    row = _label(turn=turn, label=label)
    row.update({"recording_machine": "qwen3.8-27b-fp8@casdgx01-gpu1", "recording_format": "fp8",
                "reask_format": reask})
    return row


def test_formats_of_old_and_new_label_lines():
    assert rl.formats(_label()) == ("nvfp4", "nvfp4")  # written before the fields existed: same format
    assert rl.formats(_cross(1, "off", "nvfp4")) == ("fp8", "nvfp4")
    wrong = _cross(1, "off", "nvfp4")
    wrong["recording_format"] = "nvfp4"
    with pytest.raises(ValueError, match="recording_format nvfp4 but recording machine"):
        rl.formats(wrong)
    half = _cross(1, "off", "nvfp4")
    del half["reask_format"]
    with pytest.raises(ValueError, match="no reask_format"):
        rl.formats(half)


def test_stats_split_labels_by_same_and_cross_format():
    labels = [_label(turn=1), _label(turn=2, label="xhigh"), _cross(3, "off", "fp8"), _cross(4, "low", "nvfp4"),
              _cross(5, "xhigh", "nvfp4")]
    text = rl.stats_markdown(labels)
    assert "## Labels by re-ask format" in text
    assert "| nvfp4 | nvfp4 | same format | 2 | 50.0% | 0.0% | 0.0% | 50.0% |" in text
    assert "| fp8 | fp8 | same format | 1 | 100.0% |" in text
    assert "| all | | same format | 3 |" in text
    assert "| fp8 | nvfp4 | cross format | 2 | 0.0% | 50.0% | 0.0% | 50.0% |" in text
    assert "| all | | cross format | 2 |" in text


def test_a_turn_written_by_two_labellers_is_an_error(tmp_path):
    first, second = tmp_path / "routing-labels-casdgx01.jsonl", tmp_path / "routing-labels-casdgx01-b200.jsonl"
    first.write_text(json.dumps(_label(turn=1)) + "\n" + json.dumps(_label(turn=2)) + "\n")
    second.write_text(json.dumps(_cross(3, "off", "nvfp4")) + "\n")
    turns, _ = rl.read_rows([first, second])
    assert len(turns) == 3
    second.write_text(json.dumps({"kind": "excluded", "id": "t:sess-1:2", "reason": "r"}) + "\n")
    with pytest.raises(ValueError, match="1 turns are written twice, e.g. t:sess-1:2"):
        rl.read_rows([first, second])
    second.write_text(json.dumps(_label(turn=1)) + "\n")
    with pytest.raises(ValueError, match="written twice"):
        rl.read_rows([first, second])
