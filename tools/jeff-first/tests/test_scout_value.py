import json
from pathlib import Path

from scout_value import value


def trial_dir(root: Path, task: str, trial_id: str) -> Path:
    trial = root / f"{task}-20261002-120000" / f"{task}__{trial_id}"
    (trial / "agent" / "pi" / "sessions").mkdir(parents=True)
    return trial


def write_trace(trial: Path, lines: list[dict]) -> None:
    (trial / "agent" / "jeff-first-trace.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))


def assistant_message(provider: str | None, tool_calls: list[dict]) -> dict:
    content = [{"type": "toolCall", "name": call["name"], "arguments": call["arguments"]} for call in tool_calls]
    message = {"role": "assistant", "content": content}
    if provider is not None:
        message["provider"] = provider
    return message


def write_session(trial: Path, messages: list[dict]) -> None:
    lines = [json.dumps({"type": "message", "message": message}) for message in messages]
    (trial / "agent" / "pi" / "sessions" / "s1.jsonl").write_text("\n".join(lines) + "\n")


def decision(number: int, tool_options: list[str], chosen_kind: str, tool_call: dict, chooser_ms: float) -> dict:
    return {
        "schema": "jeff-first-trace/3",
        "kind": "decision",
        "decision": number,
        "levels": [
            {
                "level": "tool",
                "page": 1,
                "tool": None,
                "options": [{"id": kind, "description": kind} for kind in tool_options],
                "picks": [],
                "chosen": chosen_kind,
            },
            {
                "level": "argument",
                "page": 1,
                "tool": chosen_kind,
                "options": [{"id": f"{chosen_kind}-1", "description": chosen_kind, "toolCall": tool_call}],
                "picks": [],
                "chosen": f"{chosen_kind}-1",
            },
        ],
        "action": {"kind": "step", "tool_call": tool_call},
        "timings_ms": {"lists": 1, "chooser": chooser_ms},
    }


def model_turn(turn: int, tool_calls: list[dict], model_ms: float) -> dict:
    return {
        "schema": "jeff-first-trace/3",
        "kind": "model_turn",
        "turn": turn,
        "action": {
            "stop_reason": "toolUse",
            "error_message": None,
            "text_chars": 0,
            "tool_calls": tool_calls,
        },
        "timings_ms": {"model": model_ms},
    }


def test_value_on_a_hand_computed_example(tmp_path):
    teacher_root, base_root = tmp_path / "teacher", tmp_path / "base"
    teacher = trial_dir(teacher_root, "t1", "abc")
    base = trial_dir(base_root, "t1", "xyz")

    read_call = {"name": "read", "arguments": {"path": "/app/a.py"}}
    toolchain_call = {"name": "bash", "arguments": {"command": "head -n 2 /etc/os-release"}}
    qwen_cat_call = {"name": "bash", "arguments": {"command": "cat /app/a.py"}}
    write_trace(
        teacher,
        [
            decision(1, ["read", "toolchain", "hand_over"], "read", read_call, chooser_ms=1000),
            decision(2, ["read", "toolchain", "hand_over"], "toolchain", toolchain_call, chooser_ms=500),
            model_turn(1, [qwen_cat_call], model_ms=4000),
        ],
    )
    write_session(
        teacher,
        [
            assistant_message("jeff-first", [read_call]),
            assistant_message("jeff-first", [toolchain_call]),
            assistant_message(None, [qwen_cat_call]),
        ],
    )

    qwen_read_call = {"name": "read", "arguments": {"path": "/app/a.py"}}
    qwen_write_call = {"name": "write", "arguments": {"path": "/app/b.py", "content": "x"}}
    qwen_echo_call = {"name": "bash", "arguments": {"command": "echo done"}}
    write_session(
        base,
        [
            assistant_message(None, [qwen_read_call, qwen_write_call, qwen_echo_call]),
        ],
    )

    result = value(teacher, base)

    assert result["tool_calls"] == 3
    assert result["scout_calls"] == 2
    assert result["scout_share"] == 2 / 3
    assert result["median_turn_seconds"] == 4.0
    assert result["estimated_seconds_avoided"] == 8.0
    assert result["scout_decision_seconds"] == 1.5
    assert result["per_tool"] == {
        "read": {"offered": 2, "chosen": 1, "used": 1, "anticipated": 1},
        "toolchain": {"offered": 2, "chosen": 1, "used": 0, "anticipated": 0},
    }
