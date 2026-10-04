"""Routing labels: for each recorded Qwen turn of the xhigh collection, is a cheaper thinking level good enough?

The xhigh collection (record mode, router fixed:xhigh) recorded every Qwen3.8-27B turn at reasoning effort xhigh. For
each such turn this script asks the same request again (rebuilt with pi's own code by routing_requests.ts) at cheaper
thinking levels, in order, and stops at the first level whose action is good:

  off     thinking off, temperature 0, at most 8,192 output tokens;
  low     thinking on, reasoning_effort "low", server-default sampling, at most 32,768 output tokens;
  medium  thinking on, reasoning_effort "medium", as low;
  (none good: label "xhigh").

An action is good when it is the same step as the recorded (xhigh) action without asking (`free_match`: the same
commands, or the same intent with every target named in full, such as the same file read; never for inline scripts,
program runs, file writes or edits, or final answers), or else when a judge model says it serves the task as well
(`judge_messages`, the owner's validated prompt; thinking off, temperature 0, through judge_service.py; the reply must
start with YES or NO, anything else raises). A reply without a usable action
(output cap hit, generation loop cut off, a tool other than bash, unparsable arguments) is not good and is not judged.

Calibration turns (a fixed random subset, `in_calibration`) ask all three cheap levels plus xhigh twice more, so the
label against the recorded action alone can be compared with "matches any of the three xhigh actions"; the two extra
xhigh actions are also judged against the recorded one (how often xhigh agrees with itself). Their label is the same
cascade, computed from the full set.

Requests go to servers of the same model format as the session (FP8 sessions to the FP8 servers, NVFP4 to NVFP4), all
turns of one trial to one server (shared prompt prefixes stay cached). Load control: at most --per-server requests in
flight per server, and no new request while the server's vLLM queue (`vllm:num_requests_waiting` in /metrics) is above
--max-waiting. Every request is streamed with the generation-loop cut-off of the thinking-off comparison
(thinking_off_run.looping).

Output: one JSONL per source host (--source NAME=DIR, file routing-labels-NAME.jsonl in --out-dir): a "turn" line per
labelled turn (all asked actions, tokens, times, judge verdicts and reasons, label) and a "trial" line when all turns
of a trial are written. A rerun skips written turns and trials (resumable).

Usage (needs aiohttp and tokenizers; node runs the request builder):
    uv run --with aiohttp==3.12.15 --with tokenizers python routing_labels.py run --source casdgx01=RUNS \\
        --family fp8 --servers URL,URL --machine casdgx01-h100 --judge-url http://100.77.219.98:8905 \\
        --judge-model qwen3.8-max --out-dir OUT \\
        --node node --builder routing_requests.mjs --tokenizer tokenizer.json --calibration-rate 0.035
"""

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "tools" / "jeff-first"))

import aiohttp  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import ceiling  # noqa: E402
import thinking_off  # noqa: E402
from imitation.record_rows import _json_lines, trial_cut  # noqa: E402
from imitation.stage3 import model_format, task_id  # noqa: E402
from thinking_off_run import CHECK_EVERY, looping  # noqa: E402

CHEAP_LEVELS = ("off", "low", "medium")
COARSE_KINDS = ("other", "writes")  # intent parts without a full target (free_match)
OFF_MAX_TOKENS = 8192
ROUTER = "fixed:xhigh"
SENT_XHIGH = {"enable_thinking": True, "reasoning_effort": "xhigh"}
# Before the thinking router (build 4abde3ece, collections v4/v5) pi sent only enable_thinking; the template then
# applied its default effort, xhigh.
SENT_LEGACY = {"enable_thinking": True, "reasoning_effort": None}
LEGACY_TARBALLS = {"jeff-pi-scout-4abde3ece.tgz"}
# Server prompt tokens minus pi's recorded prompt tokens per level, measured in the smoke (106 turns, every request) on
# NVFP4 and FP8: xhigh is the recorded request itself; the others lack or change the system prompt's reasoning-effort
# line (and off closes the thinking block in the prompt).
PROMPT_DIFF = {"xhigh": 0, "off": -36, "low": -12, "medium": -38}
PROMPT_MISMATCH = "the rebuilt request's prompt differs from the recorded one (prompt tokens)"
LABELLED_STOPS = ("toolUse", "stop")
SKIP_REASONS = {"length": "reply hit the output cap (length)", "error": "the request failed (error)",
                "aborted": "the request was aborted"}
JUDGE_CHECK_RATE = 0.2  # share of judge calls whose full input is stored (for the 300-pair hand check)
CALIBRATION_SEED = "routing-calibration-20261004"
# Qwen (NVFP4) sometimes calls a tool that does not exist; ceiling.py's turn classifier raises on such a session.
JUDGE_REFUSED = "the judge refused the input (DashScope content inspection)"
OTHER_TOOL = "a reply calls a tool other than bash (the trial is left out: ceiling.py cannot classify it)"

# ------------------------------------------------------------------------------------------------ judge prompt

# The owner's validated judge prompt (/private/tmp/claude-501/judge-test/judge.py, 2026-10-04), reproduced exactly:
# Step 1 is the recorded xhigh action, Step 2 the cheaper level's action.
JUDGE_SYSTEM = (
    "You review the work of an AI assistant that solves tasks on a Linux computer by running shell commands, one step "
    "at a time. You will see the task, the most recent steps the assistant took (each command with the end of its "
    "output), and two different candidates for the assistant's next step, Step 1 and Step 2. A step may contain "
    "several commands; they run one after another.\n\n"
    "Decide whether Step 2 would serve the task as well as Step 1 at this moment. Step 2 does not need to be the same "
    "as Step 1. It is as good if it makes the same progress: it gets the information the assistant needs next, or it "
    "brings about the same result, even if in a different way or with small differences (for example a slightly "
    "different command, option or amount of output). It is not as good if it leaves out something important that "
    "Step 1 does, does something less useful or wrong, or would cost the assistant an extra step to catch up.\n\n"
    "Answer with YES or NO on the first line. On the second line, give one sentence explaining why."
)
RECENT_STEPS = 3
OUTPUT_TAIL_CHARS = 600


def _text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def recent_steps(messages):
    """The last RECENT_STEPS commands before the turn with the last OUTPUT_TAIL_CHARS characters of their output, from
    the request's messages (what the model saw), oldest first."""
    outputs = {m["tool_call_id"]: _text(m["content"]) for m in messages if m["role"] == "tool"}
    steps = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = None
            command = parsed["command"] if isinstance(parsed, dict) and isinstance(parsed.get("command"), str) \
                else arguments
            steps.append((command, outputs.get(call["id"], "")[-OUTPUT_TAIL_CHARS:]))
    return steps[-RECENT_STEPS:]


def step_pieces(commands, final_text):
    """A step as the judge sees it: its commands, or the final answer text when it runs none."""
    return list(commands) if commands else [final_text or ""]


def judge_messages(task, steps, reference, alternative):
    """System and user message of one judge call (judge.py's SYSTEM and user_message). `reference` (Step 1) and
    `alternative` (Step 2) are lists of step pieces."""
    shown = "\n\n".join(
        f"Command:\n{cmd[:1500]}\nEnd of output:\n{out[-600:] or '(no output)'}" for cmd, out in steps
    ) or "(none yet)"

    def step(pieces):
        return "\n\n".join(piece[:2500] for piece in pieces)

    user = (
        f"TASK:\n{task[:3000]}\n\nMOST RECENT STEPS (oldest first):\n{shown}\n\n"
        f"STEP 1:\n{step(reference)}\n\nSTEP 2:\n{step(alternative)}\n\n"
        "Would Step 2 serve the task as well as Step 1 at this moment? Answer YES or NO on the first line, then one "
        "sentence why."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


VERDICT = re.compile(r"(YES|NO)\b[\s.,:;!\-—]*(.*)", re.DOTALL)


def parse_verdict(reply):
    """(True for YES / False for NO, the reason). Raises when the reply does not begin with YES or NO."""
    match = VERDICT.match(reply.strip())
    if match is None:
        raise ValueError(f"the judge's reply does not begin with YES or NO: {reply[:200]!r}")
    return match.group(1) == "YES", match.group(2).strip()


# ------------------------------------------------------------------------------------------------ small helpers


def family(driver_build):
    return model_format(driver_build)


def fraction(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16 ** 12


def in_calibration(turn_id, rate):
    return fraction(f"{CALIBRATION_SEED}:{turn_id}") < rate


def waiting_requests(metrics_text):
    total = None
    for line in metrics_text.splitlines():
        if line.startswith("vllm:num_requests_waiting{"):
            total = (total or 0) + float(line.rsplit(" ", 1)[1])
    if total is None:
        raise ValueError("the server's /metrics has no vllm:num_requests_waiting line")
    return int(total)


def variant_body(built, level):
    body = json.loads(json.dumps(built["body"]))
    body["chat_template_kwargs"] = dict(built["kwargs"][level])
    if level == "off":
        body["max_completion_tokens"] = OFF_MAX_TOKENS
        body["temperature"] = 0
    return body


def usable_commands(record):
    """The bash commands of a reply ([] for a final answer), or None when it has no usable action."""
    return thinking_off.variant_commands(record)


def free_match(commands_a, commands_b):
    """Whether two actions are the same step without asking the judge (owner, 2026-10-04: "substantially the same
    result"). Either the same commands after normalising whitespace and quoting, or the same intent (thinking_off.py)
    in which every part names its full target: a file read, a folder listed, a search, and so on. A part that runs
    a program or an inline script ("other", the program name) or writes files ("writes") names no full target, and
    two final answers can differ in their text: those always go to the judge."""
    if commands_a and thinking_off.action_of(commands_a) == thinking_off.action_of(commands_b):
        return True
    intent = thinking_off.intent_of(commands_a)
    if intent == thinking_off.FINAL or intent != thinking_off.intent_of(commands_b):
        return False
    return all(kind not in COARSE_KINDS for call in intent for kind, _ in call)


# ------------------------------------------------------------------------------------------------ cascade


async def cascade(asker, recorded, calibration):
    """The label of one turn. `asker.ask(level, sample)` returns an asked reply ({"commands": list or None,
    "final_text"}); `asker.judge(reference, alternative, key)` returns {"verdict": bool, "reason"}. `recorded` is the
    recorded action ({"commands", "final_text"})."""
    replies = {}
    checks = {}

    async def check(reply, reference, key):
        if reply["commands"] is None:
            return {"good": False, "by": "no usable action"}
        if free_match(reply["commands"], reference["commands"]):
            return {"good": True, "by": "intent"}
        verdict = await asker.judge(reference, reply, key)
        return {"good": verdict["verdict"], "by": "judge", "judge": verdict}

    if not calibration:
        label = "xhigh"
        for level in CHEAP_LEVELS:
            replies[level] = await asker.ask(level, 1)
            checks[level] = await check(replies[level], recorded, f"{level} vs recorded")
            if checks[level]["good"]:
                label = level
                break
        return {"label": label, "replies": replies, "checks": checks}

    wanted = [(level, 1) for level in CHEAP_LEVELS] + [("xhigh", 2), ("xhigh", 3)]
    answers = await asyncio.gather(*(asker.ask(level, sample) for level, sample in wanted))
    for (level, sample), answer in zip(wanted, answers):
        replies[level if level != "xhigh" else f"xhigh-{sample}"] = answer
    extra = {"2": replies["xhigh-2"], "3": replies["xhigh-3"]}
    checked = await asyncio.gather(*(check(replies[level], recorded, f"{level} vs recorded") for level in CHEAP_LEVELS),
                                   *(check(extra[s], recorded, f"xhigh-{s} vs recorded") for s in extra))
    for level, result in zip(CHEAP_LEVELS, checked):
        checks[level] = result
    self_checks = {s: result for s, result in zip(extra, checked[len(CHEAP_LEVELS):])}
    any_xhigh = {}
    against_extra = {}
    for level in CHEAP_LEVELS:
        good = checks[level]["good"]
        for s, reference in extra.items():
            if good:
                break
            if reference["commands"] is None:
                continue
            result = await check(replies[level], reference, f"{level} vs xhigh-{s}")
            against_extra[f"{level} vs xhigh-{s}"] = result
            good = result["good"]
        any_xhigh[level] = good
    label = next((level for level in CHEAP_LEVELS if checks[level]["good"]), "xhigh")
    return {"label": label, "replies": replies, "checks": checks, "any_xhigh": any_xhigh,
            "checks_against_extra_xhigh": against_extra, "self_checks": self_checks,
            "xhigh_self": {s: result["good"] for s, result in self_checks.items()}}


# ------------------------------------------------------------------------------------------------ recorded turns


def finished_trials(root):
    """Trial folders (<task>__<id>/, at any depth: <stream>/roundN/<job>/ in the collection, <stream>/<arm>/run1/<job>/
    in the end-to-end test) under a collection folder that have their result.json. Trial folders are not searched."""
    found = []
    for folder, subfolders, files in os.walk(root):
        if "__" in Path(folder).name:
            subfolders.clear()
            if "result.json" in files:
                found.append(Path(folder))
    return sorted(found)


def recorded_turns(trial):
    """The labellable recorded Qwen turns of one finished trial, and the skipped turns by reason. Each turn is paired
    with its qwen_request trace line (the kept attempt); the pairs must agree on the stop reason and the prompt
    tokens, else ValueError."""
    config = json.loads((trial / "config.json").read_text())
    env = config["agent"]["env"]
    kwargs = config["agent"]["kwargs"]
    legacy = "JEFF_FIRST_THINKING_ROUTER" not in env
    if env.get("JEFF_FIRST_MODE") != "record":
        raise ValueError(f"{trial}: not a record-mode session ({env.get('JEFF_FIRST_MODE')})")
    if legacy:
        if (Path(kwargs["tarball"]).name not in LEGACY_TARBALLS or kwargs.get("thinking") in (None, "off")
                or kwargs.get("thinking_format") != "qwen-chat-template"):
            raise ValueError(f"{trial}: a session without a thinking router must be of {sorted(LEGACY_TARBALLS)} with "
                             f"thinking on through qwen-chat-template ({kwargs})")
    elif env["JEFF_FIRST_THINKING_ROUTER"] != ROUTER:
        raise ValueError(f"{trial}: router {env['JEFF_FIRST_THINKING_ROUTER']}, not {ROUTER}")
    build = env["JEFF_FIRST_DRIVER_BUILD"]
    family(build)
    cut = trial_cut(trial)
    sessions = sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if len(sessions) != 1:
        raise ValueError(f"{trial}: {len(sessions)} session files")
    entries, _ = _json_lines(sessions[0], cut)
    session_id = entries[0]["id"]
    trace, _ = _json_lines(trial / "agent" / "jeff-first-trace.jsonl", cut)
    finals = {}
    for line in trace:
        if legacy and line["kind"] == "record":
            line = legacy_line(line)
        if line["kind"] != "qwen_request":
            continue
        if line["session_id"] != session_id:
            raise ValueError(f"{trial}: a qwen_request line of session {line['session_id']}, not {session_id}")
        if line["outcome"] == "discarded":
            continue
        if line["turn"] in finals:
            raise ValueError(f"{trial}: two final qwen_request lines for turn {line['turn']}")
        finals[line["turn"]] = line
    lines = [finals[turn] for turn in sorted(finals) if finals[turn]["stop_reason"] not in ("error", "aborted")]
    assistants = [e for e in entries if e["type"] == "message" and e["message"]["role"] == "assistant"
                  and e["message"]["stopReason"] not in ("error", "aborted")]
    ended_early = cut is not None or json.loads((trial / "result.json").read_text())["exception_info"] is not None
    if ended_early and len(lines) == len(assistants) + 1:
        lines = lines[:-1]  # the trace line of a reply the stopped or crashed session never saved
    if len(lines) != len(assistants):
        raise ValueError(f"{trial}: {len(assistants)} assistant replies but {len(lines)} qwen_request lines")
    calls = [part for e in entries if e["type"] == "message" and e["message"]["role"] == "assistant"
             for part in e["message"]["content"] if part["type"] == "toolCall"]
    if any(call["name"] != "bash" for call in calls):
        return [], {OTHER_TOOL: len(assistants)}
    turns = []
    skipped = collections.Counter()
    for line, entry in zip(lines, assistants):
        message = entry["message"]
        usage = message["usage"]
        if line["stop_reason"] != message["stopReason"] or line["usage"]["input"] != usage["input"]:
            raise ValueError(f"{trial}: turn {line['turn']}: trace ({line['stop_reason']}, {line['usage']['input']}) "
                             f"and session ({message['stopReason']}, {usage['input']}) disagree")
        if line["sent"] != (SENT_LEGACY if legacy else SENT_XHIGH):
            raise ValueError(f"{trial}: turn {line['turn']} was sent {line['sent']}, not {SENT_XHIGH}")
        if message["stopReason"] not in LABELLED_STOPS:
            skipped[SKIP_REASONS.get(message["stopReason"], message["stopReason"])] += 1
            continue
        calls = [part for part in message["content"] if part["type"] == "toolCall"]
        commands = [c["arguments"]["command"] if isinstance(c["arguments"].get("command"), str)
                    else json.dumps(c["arguments"]) for c in calls]
        text = "".join(part["text"] for part in message["content"] if part["type"] == "text")
        turns.append({
            "id": f"{trial.name}:{entry['id']}",
            "legacy": legacy,
            "trial": trial.name,
            "trial_dir": str(trial),
            "task": task_id(trial, config),
            "recording_machine": build,
            "session_id": session_id,
            "session_file": str(sessions[0]),
            "entry_id": entry["id"],
            "turn": line["turn"],
            "commands": commands,
            "final_text": None if calls else text,
            "recorded": {
                "stop_reason": message["stopReason"],
                "prompt_tokens": usage["input"] + usage["cacheRead"],
                "output_tokens": usage["output"],
                "reasoning_tokens": usage.get("reasoning"),
                "model_ms": line["timings_ms"]["model"],
            },
        })
    return turns, dict(skipped)


def legacy_line(record):
    """A schema-4 record line (build 4abde3ece, before the thinking router) in the shape of a kept qwen_request line.
    pi sent thinking on without a reasoning effort, so the chat template used its default, xhigh: the same prompt as
    an explicit xhigh (checked per turn by the prompt tokens, `check_prompt`)."""
    return {"kind": "qwen_request", "session_id": record["session_id"], "turn": record["turn"], "outcome": "kept",
            "stop_reason": record["action"]["stop_reason"], "usage": {"input": record["model_usage"]["input"]},
            "sent": SENT_LEGACY, "timings_ms": {"model": record["timings_ms"]["model"]}}


class PromptMismatch(Exception):
    """A rebuilt request's prompt tokens differ from what the recorded turn's prompt implies."""


def check_prompt(what, level, prompt_diff):
    """At xhigh the rebuilt request is the recorded one: its prompt tokens must equal pi's record exactly. The other
    levels change the system prompt's reasoning-effort line, whose token count depends on the text around it (most
    sessions -36/-12/-38, some -37 or -30 at off), so their difference is only recorded (statistics), not checked."""
    if level == "xhigh" and prompt_diff is not None and prompt_diff != PROMPT_DIFF[level]:
        raise PromptMismatch(f"{what}: prompt tokens differ from the recorded prompt by {prompt_diff}, expected "
                             f"{PROMPT_DIFF[level]} at {level}")


def turn_classes(trial, tokenizer):
    """ceiling.py's class of each assistant reply of the trial, in session order."""
    session = ceiling.read_trial(trial, tokenizer)
    return [t["class"] for t in session["turns"] if t["kind"] == "assistant"]


# ------------------------------------------------------------------------------------------------ servers


class Server:
    def __init__(self, url, machine, per_server, max_waiting):
        self.url = url
        self.base = url.split("/v1/")[0]
        self.machine = machine
        self.slots = asyncio.Semaphore(per_server)
        self.max_waiting = max_waiting
        self.waiting = None
        self.in_flight = 0
        self.queued_turns = 0
        self.metrics_error = None

    async def watch(self, http):
        while True:
            try:
                async with http.get(f"{self.base}/metrics", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                    self.waiting = waiting_requests(await response.text())
                    self.metrics_error = None
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
                # Not a fallback: no new request goes to a server whose queue cannot be read; logged loudly.
                self.waiting = None
                if self.metrics_error is None:
                    print(f"METRICS UNREADABLE {self.base}: {error!r}; no new requests to it", flush=True)
                self.metrics_error = repr(error)
            await asyncio.sleep(3)

    async def acquire(self):
        await self.slots.acquire()
        while self.waiting is None or self.waiting > self.max_waiting:
            await asyncio.sleep(2)
        self.in_flight += 1

    def release(self):
        self.in_flight -= 1
        self.slots.release()


async def stream(http, server, body, what):
    """Send one streamed chat request; the reply's text, tool calls, tokens and times. A generation loop is cut off
    (outcome "loop")."""
    record = {"server": server.url, "machine": server.machine, "outcome": None, "finish_reason": None,
              "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None, "reasoning_chars": 0,
              "chunks": 0, "first_s": None, "total_s": None, "content": "", "tool_calls": [], "loop": None}
    reasoning, content, calls = [], [], {}
    checked = {"reasoning": 0, "answer": 0}
    reasoning_len = answer_len = 0
    await server.acquire()
    start = time.monotonic()
    try:
        async with http.post(server.url, json=body) as response:
            if response.status != 200:
                raise RuntimeError(f"{what}: HTTP {response.status} from {server.url}: {(await response.text())[:500]}")
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    record["prompt_tokens"] = usage["prompt_tokens"]
                    record["completion_tokens"] = usage["completion_tokens"]
                    record["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    got = False
                    piece = delta.get("reasoning") or delta.get("reasoning_content")
                    if piece:
                        reasoning.append(piece)
                        reasoning_len += len(piece)
                        got = True
                    if delta.get("content"):
                        content.append(delta["content"])
                        answer_len += len(delta["content"])
                        got = True
                    for call in delta.get("tool_calls") or []:
                        slot = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                        function = call.get("function") or {}
                        slot["name"] += function.get("name") or ""
                        slot["arguments"] += function.get("arguments") or ""
                        answer_len += len(function.get("arguments") or "")
                        got = True
                    if got:
                        record["chunks"] += 1
                        if record["first_s"] is None:
                            record["first_s"] = time.monotonic() - start
                    if choice.get("finish_reason"):
                        record["finish_reason"] = choice["finish_reason"]
                if reasoning_len - checked["reasoning"] >= CHECK_EVERY:
                    checked["reasoning"] = reasoning_len
                    why = looping("".join(reasoning))
                    if why:
                        record["loop"] = f"thinking: {why}"
                if not record["loop"] and answer_len - checked["answer"] >= CHECK_EVERY:
                    checked["answer"] = answer_len
                    why = looping("".join(content) + "\n".join(
                        c["arguments"].replace("\\n", "\n").replace('\\"', '"') for c in calls.values()))
                    if why:
                        record["loop"] = f"answer: {why}"
                if record["loop"]:
                    response.close()
                    break
    except aiohttp.ClientError as error:
        raise RuntimeError(f"{what}: {server.url}: {error!r}") from error
    finally:
        server.release()
    record["total_s"] = time.monotonic() - start
    record["reasoning_chars"] = reasoning_len
    record["reasoning"] = "".join(reasoning)
    record["content"] = "".join(content)
    record["tool_calls"] = [calls[i] for i in sorted(calls)]
    if record["loop"]:
        record["outcome"] = "loop"
    elif record["finish_reason"] == "length":
        record["outcome"] = "length"
    elif record["finish_reason"] in ("stop", "tool_calls"):
        record["outcome"] = "ok"
    else:
        raise RuntimeError(f"{what}: the stream ended without a finish reason ({record['finish_reason']})")
    if record["outcome"] != "loop" and record["completion_tokens"] is None:
        raise RuntimeError(f"{what}: no usage report at the end of the stream")
    return record


# ------------------------------------------------------------------------------------------------ the labeller


class Builder:
    """The request builder (routing_requests.ts / .mjs) as a long-running child process, one job at a time."""

    def __init__(self, node, script):
        self.node = node
        self.script = script
        self.process = None
        self.lock = asyncio.Lock()

    async def build(self, turn):
        async with self.lock:
            if self.process is None:
                self.process = await asyncio.create_subprocess_exec(
                    self.node, self.script, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    limit=1 << 30)
            job = {"id": turn["id"], "session": turn["session_file"], "entryId": turn["entry_id"]}
            self.process.stdin.write((json.dumps(job) + "\n").encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"the request builder exited (code {await self.process.wait()}) on {turn['id']}")
            built = json.loads(line)
            if built["id"] != turn["id"]:
                raise RuntimeError(f"the request builder answered {built['id']} for {turn['id']}")
            if built["body"]["chat_template_kwargs"] != {**SENT_XHIGH, "preserve_thinking": True}:
                raise RuntimeError(f"{turn['id']}: rebuilt kwargs {built['body']['chat_template_kwargs']} are not what "
                                   f"the session sent")
            return built

    async def close(self):
        if self.process is not None:
            self.process.stdin.close()
            code = await self.process.wait()
            if code != 0:
                raise RuntimeError(f"the request builder exited with code {code}")


class TurnAsker:
    """Asks one turn's requests and judge calls (the `asker` of `cascade`)."""

    def __init__(self, labeller, turn, built, server):
        self.labeller = labeller
        self.turn = turn
        self.built = built
        self.server = server
        self.asks = []
        self.judges = []

    async def ask(self, level, sample):
        body = variant_body(self.built, level)
        record = await stream(self.labeller.http, self.server, body, f"{self.turn['id']} {level}-{sample}")
        commands = usable_commands(record)
        record.update({"level": level, "sample": sample, "commands": commands,
                       "final_text": record["content"] if commands == [] else None,
                       "prompt_diff": None if record["prompt_tokens"] is None
                       else record["prompt_tokens"] - self.turn["recorded"]["prompt_tokens"]})
        check_prompt(f"{self.turn['id']} {level}-{sample}", level, record["prompt_diff"])
        del record["reasoning"]
        self.asks.append(record)
        return record

    async def judge(self, reference, alternative, key):
        steps = recent_steps(self.built["body"]["messages"])
        step1 = step_pieces(reference["commands"], reference["final_text"])
        step2 = step_pieces(alternative["commands"], alternative["final_text"])
        messages = judge_messages(self.built["task"], steps, step1, step2)
        what = f"{self.turn['id']} judge {key}"
        reply = await self.labeller.judge_client.ask(messages, what)
        verdict, reason = parse_verdict(reply["content"])
        # The judge service always asks with thinking off and temperature 0.
        result = {"key": key, "model": f"{reply['model']} ({reply['backend']}, thinking off)", "verdict": verdict,
                  "reason": reason, "prompt_tokens": reply["usage"].get("prompt_tokens"),
                  "completion_tokens": reply["usage"].get("completion_tokens"), "finish_reason": reply["finish_reason"],
                  "seconds": reply["seconds"], "reply": reply["content"]}
        if fraction(f"judge-check:{self.turn['id']}:{key}") < JUDGE_CHECK_RATE:
            result.update({"messages": messages, "task": self.built["task"], "steps": steps, "reference": step1,
                           "alternative": step2})
        self.judges.append(result)
        return result


class JudgeRefused(Exception):
    """The judge service's backend refused to judge this input (HTTP 422): the turn is left out, with its reason."""


class JudgeClient:
    """The judge service (judge_service.py on datigator): POST /judge {"messages"} -> {"model", "backend", "content",
    "usage", "finish_reason", "seconds"}. The service's model must be the expected one (--judge-model)."""

    def __init__(self, url, model):
        self.url = url.rstrip("/")
        self.model = model
        self.http = None

    async def check(self, http):
        self.http = http
        async with http.get(f"{self.url}/info", timeout=aiohttp.ClientTimeout(total=20)) as response:
            if response.status != 200:
                raise RuntimeError(f"judge service {self.url}/info: HTTP {response.status}")
            info = await response.json()
        if info["model"] != self.model:
            raise RuntimeError(f"judge service {self.url} serves {info['model']!r}, not the expected {self.model!r}")
        return info

    async def ask(self, messages, what):
        try:
            async with self.http.post(f"{self.url}/judge", json={"messages": messages},
                                      timeout=aiohttp.ClientTimeout(total=900)) as response:
                if response.status == 422:
                    raise JudgeRefused(f"{what}: {(await response.json())['refused'][:300]}")
                if response.status != 200:
                    raise RuntimeError(f"{what}: judge service HTTP {response.status}: {(await response.text())[:500]}")
                reply = await response.json()
        except aiohttp.ClientError as error:
            raise RuntimeError(f"{what}: judge service {self.url}: {error!r}") from error
        if reply["model"] != self.model:
            raise RuntimeError(f"{what}: the judge answered as {reply['model']!r}, not {self.model!r}")
        return reply


class Labeller:
    def __init__(self, options):
        self.options = options
        self.family = options.family
        self.sources = [tuple(item.split("=", 1)) for item in options.source]  # (host name, collection folder)
        missing = [root for _, root in self.sources if not Path(root).is_dir()]
        if missing:
            raise ValueError(f"collection folders that do not exist: {missing}")
        self.out_dir = Path(options.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.servers = [Server(url, options.machine, options.per_server, options.max_waiting)
                        for url in options.servers.split(",")]
        self.judge_client = JudgeClient(options.judge_url, options.judge_model)
        self.builder = Builder(options.node, options.builder)
        self.tokenizer = Tokenizer.from_file(options.tokenizer)
        self.done_turns = set()
        self.done_trials = set()
        self.sinks = {}
        self.queues = {s.url: asyncio.Queue() for s in self.servers}
        self.pending = {}  # trial -> turns not yet written
        self.trial_info = {}
        self.written = 0
        self.started = time.monotonic()
        self.http = None
        self.limit = options.max_turns

    def load_done(self):
        for name in dict(self.sources):
            path = self.out_dir / f"routing-labels-{name}.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row["kind"] in ("turn", "excluded"):
                        self.done_turns.add(row["id"])
                    else:
                        self.done_trials.add(row["trial_dir"])
            self.sinks[name] = path.open("a")

    def write(self, source, row):
        self.sinks[source].write(json.dumps(row) + "\n")
        self.sinks[source].flush()

    def discover(self):
        added = 0
        for source, root in self.sources:
            for trial in finished_trials(Path(root)):
                key = str(trial)
                if key in self.done_trials or key in self.trial_info:
                    continue
                if not (trial / "agent" / "jeff-first-trace.jsonl").exists():
                    info = json.loads((trial / "result.json").read_text()).get("exception_info")
                    reason = f"no trace: the agent failed before its first turn ({(info or {}).get('exception_type')})"
                    self.write(source, {"kind": "trial", "trial_dir": key, "source_host": source, "turns": 0,
                                        "skipped": {reason: 1}})
                    self.done_trials.add(key)
                    continue
                turns, skipped = recorded_turns(trial)
                if turns and family(turns[0]["recording_machine"]) != self.family:
                    raise ValueError(f"{trial}: a {family(turns[0]['recording_machine'])} session; this labeller "
                                     f"answers {self.family} sessions only")
                class_of = {}
                if turns:  # a trial left out by recorded_turns may be one ceiling.py cannot classify
                    classes = turn_classes(trial, self.tokenizer)
                    assistants = self._assistant_ids(trial)
                    if len(classes) != len(assistants):
                        raise ValueError(f"{trial}: {len(classes)} classified replies but {len(assistants)} assistant "
                                         "entries")
                    class_of = dict(zip(assistants, classes))
                for turn in turns:
                    turn["class"] = class_of[turn["entry_id"]]
                    turn["source_host"] = source
                todo = [t for t in turns if t["id"] not in self.done_turns]
                self.trial_info[key] = {"source": source, "turns": len(turns), "skipped": skipped}
                self.pending[key] = len(todo)
                server = min(self.servers, key=lambda s: s.queued_turns)
                for turn in todo:
                    server.queued_turns += 1
                    self.queues[server.url].put_nowait(turn)
                added += len(todo)
                if not todo:
                    self.finish_trial(key)
        return added

    @staticmethod
    def _assistant_ids(trial):
        session = next((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
        entries, _ = _json_lines(session, trial_cut(trial))
        return [e["id"] for e in entries if e["type"] == "message" and e["message"]["role"] == "assistant"]

    def finish_trial(self, key):
        info = self.trial_info[key]
        self.write(info["source"], {"kind": "trial", "trial_dir": key, "source_host": info["source"],
                                    "turns": info["turns"], "skipped": info["skipped"]})
        self.done_trials.add(key)

    async def label(self, turn, server):
        built = await self.builder.build(turn)
        calibration = self.options.all_levels or in_calibration(turn["id"], self.options.calibration_rate)
        asker = TurnAsker(self, turn, built, server)
        started = time.monotonic()
        result = await cascade(asker, {"commands": turn["commands"], "final_text": turn["final_text"]}, calibration)
        checks = {level: {k: v for k, v in check.items() if k != "judge"} for level, check in result["checks"].items()}
        row = {"kind": "turn", **{k: v for k, v in turn.items() if k not in ("session_file",)},
               "replay_machine": server.machine, "server": server.url, "calibration": calibration,
               "label": result["label"], "checks": checks, "asks": asker.asks, "judges": asker.judges,
               "labelled_s": time.monotonic() - started, "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if calibration:
            row["any_xhigh"] = result["any_xhigh"]
            row["xhigh_self"] = result["xhigh_self"]
        return row

    async def worker(self, server):
        queue = self.queues[server.url]
        while True:
            turn = await queue.get()
            if self.limit is not None and self.written >= self.limit:
                server.queued_turns -= 1
                queue.task_done()
                continue
            try:
                row = await self.label(turn, server)
            except (JudgeRefused, PromptMismatch) as problem:
                # Not a fallback: the turn cannot be labelled by the owner's rule (the judge refused it, or the rebuilt
                # request is not the recorded one); it is left out with its reason (counted in the statistics), never
                # given a substitute label.
                print(f"LEFT OUT {turn['id']}: {problem}", flush=True)
                row = {"kind": "excluded", "id": turn["id"], "trial_dir": turn["trial_dir"],
                       "source_host": turn["source_host"], "class": turn["class"],
                       "reason": JUDGE_REFUSED if isinstance(problem, JudgeRefused) else PROMPT_MISMATCH,
                       "detail": str(problem)}
            server.queued_turns -= 1
            self.write(turn["source_host"], row)
            self.written += 1
            key = turn["trial_dir"]
            self.pending[key] -= 1
            if self.pending[key] == 0:
                self.finish_trial(key)
            if self.written % 25 == 0:
                hours = (time.monotonic() - self.started) / 3600
                print(f"{time.strftime('%H:%M:%S')} {self.written} turns labelled ({self.written / hours:.0f}/h); "
                      + ", ".join(f"{s.base.rsplit(':', 1)[-1]} q{s.queued_turns} f{s.in_flight} w{s.waiting}"
                                  for s in self.servers), flush=True)
            queue.task_done()

    async def run(self):
        try:
            await self._run()
        except BaseException:  # noqa: BLE001 - not swallowed: printed, then the process ends with code 1
            # Exiting here, inside the event loop: asyncio's own shutdown can wait forever on cancelled tasks.
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

    async def _run(self):
        self.load_done()
        timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as http:
            self.http = http
            print(f"judge: {await self.judge_client.check(http)}", flush=True)
            watchers = [asyncio.create_task(s.watch(http)) for s in self.servers]
            workers = [asyncio.create_task(self.worker(s)) for s in self.servers for _ in range(self.options.per_server)]
            tasks = watchers + workers
            while True:
                added = self.discover()
                print(f"{time.strftime('%H:%M:%S')} discovered {added} new turns; "
                      f"{sum(s.queued_turns for s in self.servers)} queued", flush=True)
                idle = all(s.queued_turns == 0 for s in self.servers)
                if self.options.exit_when_idle and idle:
                    break
                if self.limit is not None and self.written >= self.limit:
                    break
                done, _ = await asyncio.wait(tasks, timeout=self.options.poll, return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    task.result()  # raises the worker's error
            for task in tasks:
                task.cancel()
            await self.builder.close()
        print(f"done: {self.written} turns in {time.monotonic() - self.started:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ join, stats, check


def read_labels(paths):
    return read_rows(paths)[0]


def read_rows(paths):
    """The labelled turns and the left-out turns of label files."""
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    excluded = [row for row in rows if row["kind"] == "excluded"]
    turns = [row for row in rows if row["kind"] == "turn"]
    seen = collections.Counter(row["id"] for row in turns)
    duplicates = [turn_id for turn_id, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} turns are labelled twice, e.g. {duplicates[0]}")
    return turns, excluded


def routing_fields(turn):
    def ok(level):
        check = turn["checks"].get(level)
        return None if check is None else check["good"]

    actions = [{"commands": turn["commands"], "final_text": turn["final_text"]}]
    for ask in turn["asks"]:
        if ask["level"] == "xhigh":
            actions.append({"commands": ask["commands"], "final_text": ask["final_text"]})
    fields = {"label": turn["label"], "off_ok": ok("off"), "low_ok": ok("low"), "medium_ok": ok("medium"),
              "checked_by": {level: check["by"] for level, check in turn["checks"].items()},
              "xhigh_actions": actions, "calibration": turn["calibration"]}
    if turn["calibration"]:
        fields["any_xhigh"] = turn["any_xhigh"]
        fields["xhigh_self"] = turn["xhigh_self"]
    return fields


def join_rows(labels, stage3_rows):
    """One row per labelled turn that has stage-3 rows: the state of the turn's first decision (its tool-level row on
    page 1: what Jeff sees before the coding model's turn) plus the routing fields. Stage-3 rows are keyed by (session,
    turn); turns without a stage-3 row are counted."""
    first = {}
    for row in stage3_rows:
        if row["level"] != "tool" or row["page"] != 1:
            continue
        key = (row["session"], row["turn"])
        if key not in first or row["decision"] < first[key]["decision"]:
            first[key] = row
    rows = []
    missing = 0
    for turn in labels:
        stage3 = first.get((turn["session_id"], turn["turn"]))
        if stage3 is None:
            missing += 1
            continue
        rows.append({"source": "own", "stage": 3, "task": turn["task"], "session": turn["session_id"],
                     "turn": turn["turn"], "machine": turn["recording_machine"], "source_host": turn["source_host"],
                     "class": turn["class"], "state": stage3["state"], "routing": routing_fields(turn)})
    return rows, {"labelled turns": len(labels), "joined": len(rows), "no stage-3 row": missing}


def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def _tokens(ask):
    return ask["completion_tokens"] if ask["completion_tokens"] is not None else ask["chunks"]


def stats_markdown(labels, excluded=()):
    out = []
    w = out.append
    classes = sorted({t["class"] for t in labels}, key=lambda c: ceiling.ALL_CLASSES.index(c))
    groups = [("all turns", labels)] + [(c, [t for t in labels if t["class"] == c]) for c in classes]
    w("# Routing labels: is a cheaper thinking level good enough for this turn?\n")
    w("Generated by `results/imitation/scripts/routing_labels.py stats`. Each recorded Qwen3.8-27B turn of the xhigh "
      "collection (record mode, reasoning effort xhigh) is asked again at thinking off (temperature 0), then low, then "
      "medium, stopping at the first level whose action is good: the same intent as the recorded xhigh action "
      "(thinking_off.py's intent: same kind of step and target per command), or else a judge model's YES to 'Would "
      "Step 2 serve the task as well as Step 1 at this moment?' (Step 1 = recorded, Step 2 = cheaper). Label = that "
      "level, or xhigh when none is good. Calibration turns ask all levels plus xhigh twice more.\n")
    hosts = collections.Counter(t["source_host"] for t in labels)
    machines = collections.Counter(t["recording_machine"].split("@")[1].split("-gpu")[0] for t in labels)
    w(f"Turns labelled: {len(labels)} from {len({t['trial'] for t in labels})} trials "
      f"({', '.join(f'{h} {n}' for h, n in sorted(hosts.items()))}); recording machines: "
      f"{', '.join(f'{m} {n}' for m, n in sorted(machines.items()))}. Judge models: "
      f"{dict(collections.Counter(j['model'] for t in labels for j in t['judges']))}.\n")
    reasons = collections.Counter(row["reason"] for row in excluded)
    w(f"Turns left out: {len(excluded)}"
      + (f" ({', '.join(f'{r}: {n}' for r, n in sorted(reasons.items()))})" if reasons else "") + ".\n")

    w("## Labels by turn class\n")
    w("Share of turns per label; off/low/medium ok = share of the turns where that level was asked whose action was "
      "good (asked: off always, low only when off was not good, medium only when low was not good, all three on "
      "calibration turns).\n")
    w("| class | turns | off | low | medium | xhigh | off ok | low ok (asked) | medium ok (asked) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, group in groups:
        n = len(group)
        count = collections.Counter(t["label"] for t in group)
        cells = [_pct(count[level], n) for level in ("off", "low", "medium", "xhigh")]
        for level in CHEAP_LEVELS:
            asked = [t for t in group if level in t["checks"]]
            good = sum(t["checks"][level]["good"] for t in asked)
            cells.append(f"{_pct(good, len(asked))} ({len(asked)})" if level != "off" else _pct(good, len(asked)))
        w(f"| {name} | {n} | " + " | ".join(cells) + " |")
    w("")

    w("## How the checks decided\n")
    w("| level | asked | same step (free match) | judge YES | judge NO | no usable action |")
    w("|---|---:|---:|---:|---:|---:|")
    for level in CHEAP_LEVELS:
        checks = [t["checks"][level] for t in labels if level in t["checks"]]
        by = collections.Counter((c["by"], c["good"]) for c in checks)
        w(f"| {level} | {len(checks)} | {by[('intent', True)]} | {by[('judge', True)]} | {by[('judge', False)]} | "
          f"{by[('no usable action', False)]} |")
    judges = [j for t in labels for j in t["judges"]]
    w(f"\nJudge calls: {len(judges)}; YES {_pct(sum(j['verdict'] for j in judges), len(judges))}; finish reasons "
      f"{dict(collections.Counter(j['finish_reason'] for j in judges))}.\n")

    w("## Output tokens and time\n")
    w("Means per turn. 'rec' = the recorded xhigh reply. 'routed' = the output tokens of the labelled level's reply "
      "(the recorded reply's for xhigh): what a perfect router would have spent; saved = 1 - routed / rec. Seconds are "
      "request wall times measured while the collection ran on the same servers.\n")
    w("| class | turns | rec out | off out | low out | medium out | routed out | saved | rec s | off s | low s | "
      "medium s |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, group in groups:
        n = len(group)
        rec = sum(t["recorded"]["output_tokens"] for t in group)
        cells = [f"{rec / n:.0f}"]
        for level in CHEAP_LEVELS:
            asks = [a for t in group for a in t["asks"] if a["level"] == level]
            cells.append(f"{sum(_tokens(a) for a in asks) / len(asks):.0f}" if asks else "-")
        routed = 0
        for t in group:
            if t["label"] == "xhigh":
                routed += t["recorded"]["output_tokens"]
            else:
                routed += _tokens(next(a for a in t["asks"] if a["level"] == t["label"] and a["sample"] == 1))
        cells += [f"{routed / n:.0f}", f"{100 * (1 - routed / rec):.0f}%" if rec else "-",
                  f"{sum(t['recorded']['model_ms'] for t in group) / n / 1000:.1f}"]
        for level in CHEAP_LEVELS:
            asks = [a for t in group for a in t["asks"] if a["level"] == level]
            cells.append(f"{sum(a['total_s'] for a in asks) / len(asks):.1f}" if asks else "-")
        w(f"| {name} | {n} | " + " | ".join(cells) + " |")
    w("")

    w("## Request check\n")
    w("Prompt tokens the server counted minus the prompt tokens pi recorded for the turn, per level (difference: "
      "requests). xhigh must be 0 (the rebuilt request is the recorded one); the other levels differ by the "
      "reasoning-effort line of the system prompt only, so each must be one constant.\n")
    for level in ("xhigh",) + CHEAP_LEVELS:
        diffs = collections.Counter(a["prompt_diff"] for t in labels for a in t["asks"] if a["level"] == level)
        w(f"- {level}: {dict(sorted(diffs.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))}")
    outcomes = collections.Counter((a["level"], a["outcome"]) for t in labels for a in t["asks"])
    w(f"\nOutcomes (level, outcome: requests): {dict(sorted(outcomes.items()))}.\n")

    calibration = [t for t in labels if t["calibration"]]
    w("## Calibration: one recorded xhigh action vs any of three\n")
    w(f"Calibration turns: {len(calibration)} (a fixed random subset). 'vs recorded' = good against the recorded "
      "action (the label's rule); 'vs any xhigh' = good against the recorded action or one of two more xhigh samples "
      "of the same request; bias = how much the single-reference label understates 'substantially the same as "
      "xhigh'. xhigh self = an extra xhigh sample judged against the recorded one by the same rule.\n")
    if calibration:
        w("| class | turns | " + " | ".join(f"{level} vs recorded | {level} vs any xhigh" for level in CHEAP_LEVELS)
          + " | xhigh self |")
        w("|---|---:|" + "---:|" * (2 * len(CHEAP_LEVELS) + 1))
        cal_groups = [("all turns", calibration)] + [(c, [t for t in calibration if t["class"] == c]) for c in classes]
        for name, group in cal_groups:
            n = len(group)
            if not n:
                continue
            cells = []
            for level in CHEAP_LEVELS:
                cells += [_pct(sum(t["checks"][level]["good"] for t in group), n),
                          _pct(sum(t["any_xhigh"][level] for t in group), n)]
            selfs = [v for t in group for v in t["xhigh_self"].values()]
            cells.append(_pct(sum(selfs), len(selfs)))
            w(f"| {name} | {n} | " + " | ".join(cells) + " |")
        w("")
        labels_any = collections.Counter(next((level for level in CHEAP_LEVELS if t["any_xhigh"][level]), "xhigh")
                                         for t in calibration)
        labels_rec = collections.Counter(t["label"] for t in calibration)
        w("Labels on calibration turns, single reference vs any of three: "
          + ", ".join(f"{level} {_pct(labels_rec[level], len(calibration))} vs "
                      f"{_pct(labels_any[level], len(calibration))}" for level in CHEAP_LEVELS + ("xhigh",)) + ".\n")
    return "\n".join(out) + "\n"


def judge_check_pairs(labels, count):
    """A fixed random sample of `count` judged pairs whose full judge input was stored (JUDGE_CHECK_RATE of all)."""
    pairs = []
    for turn in labels:
        for judge in turn["judges"]:
            if "messages" not in judge:
                continue
            pairs.append({"pair": f"{turn['id']}|{judge['key']}", "turn": turn["id"], "task": turn["task"],
                          "class": turn["class"], "key": judge["key"], "task_text": judge["task"],
                          "steps": judge["steps"], "reference": judge["reference"],
                          "alternative": judge["alternative"], "messages": judge["messages"],
                          "model": judge["model"], "verdict": "YES" if judge["verdict"] else "NO",
                          "reason": judge["reason"]})
    if len(pairs) < count:
        raise ValueError(f"only {len(pairs)} judged pairs have their judge input stored; {count} wanted")
    pairs.sort(key=lambda p: fraction(f"judge-check-sample:{p['pair']}"))
    return pairs[:count]


async def rejudge(pairs, client):
    async with aiohttp.ClientSession() as http:
        await client.check(http)
        out = []
        for pair in pairs:
            reply = await client.ask(pair["messages"], pair["pair"])
            verdict, reason = parse_verdict(reply["content"])
            out.append({**pair, "model": f"{reply['model']} ({reply['backend']}, thinking off)",
                        "verdict": "YES" if verdict else "NO", "reason": reason})
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", action="append", required=True,
                   help="HOST=COLLECTION_DIR (repeatable; one output file per HOST, several folders may share a HOST)")
    r.add_argument("--family", required=True, choices=["fp8", "nvfp4"], help="model format of the sessions and servers")
    r.add_argument("--servers", required=True, help="comma-separated chat-completions URLs of that format")
    r.add_argument("--machine", required=True, help="replay machine tag of --servers, e.g. casdgx01-h100")
    r.add_argument("--judge-url", required=True, help="base URL of the judge service (judge_service.py)")
    r.add_argument("--judge-model", required=True, help="the model the judge service must serve")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--node", required=True)
    r.add_argument("--builder", required=True, help="routing_requests.ts or its bundle routing_requests.mjs")
    r.add_argument("--tokenizer", required=True, help="tokenizer.json of a Qwen model (ceiling.py's turn classes)")
    r.add_argument("--calibration-rate", type=float, required=True)
    r.add_argument("--per-server", type=int, default=8)
    r.add_argument("--max-waiting", type=int, default=4)
    r.add_argument("--poll", type=float, default=300, help="seconds between scans for newly finished trials")
    r.add_argument("--max-turns", type=int, help="stop after this many labelled turns (smoke)")
    r.add_argument("--all-levels", action="store_true", help="ask every turn like a calibration turn (smoke)")
    r.add_argument("--exit-when-idle", action="store_true")
    j = sub.add_parser("join", help="routing rows: the routing labels joined onto the stage-3 rows (one per turn)")
    j.add_argument("--labels", nargs="+", required=True)
    j.add_argument("--stage3-rows", required=True)
    j.add_argument("--out", required=True)
    st = sub.add_parser("stats", help="write the statistics file")
    st.add_argument("--labels", nargs="+", required=True)
    st.add_argument("--out", required=True)
    c = sub.add_parser("judge-check", help="a fixed random sample of judged pairs for checking the judge by hand")
    c.add_argument("--labels", nargs="+", required=True)
    c.add_argument("--count", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--rejudge-url", help="ask this judge service again on the same pairs")
    c.add_argument("--rejudge-model", help="the model the --rejudge-url service must serve")
    options = parser.parse_args()
    if options.step == "run":
        try:
            asyncio.run(Labeller(options).run())
        except BaseException:  # noqa: BLE001 - not swallowed: printed, then the process ends with code 1
            # asyncio's shutdown can wait forever on cancelled request tasks; the supervisor restarts the labeller.
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    elif options.step == "join":
        stage3 = [json.loads(line) for line in Path(options.stage3_rows).read_text().splitlines()]
        rows, counts = join_rows(read_labels(options.labels), stage3)
        Path(options.out).write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(counts)
    elif options.step == "stats":
        Path(options.out).write_text(stats_markdown(*read_rows(options.labels)))
    else:
        pairs = judge_check_pairs(read_labels(options.labels), options.count)
        if options.rejudge_url:
            if not options.rejudge_model:
                raise ValueError("--rejudge-url needs --rejudge-model")
            pairs = asyncio.run(rejudge(pairs, JudgeClient(options.rejudge_url, options.rejudge_model)))
        Path(options.out).write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
        print(f"wrote {len(pairs)} pairs to {options.out}")


if __name__ == "__main__":
    main()
