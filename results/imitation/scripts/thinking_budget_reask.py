"""Re-ask test for a thinking budget: does cutting Qwen3.8-27B's recorded thinking short change its action?

For each sampled turn of the xhigh collection (thinking_budget.py offline's turns: completed turns whose thinking
exceeded 8k tokens, and every reply that hit the output limit or the context limit), this script

1. rebuilds the exact request pi sent (routing_requests.ts, pi's own code, as the routing labeller does) and has the
   server render it with the chat template (vLLM /tokenize); the prompt's token count must equal the prompt tokens pi
   recorded for the turn, else the turn is left out (counted);
2. takes the RECORDED thinking text of the turn, cuts it after B tokens (B = 4k, 8k, 16k; only budgets shorter than the
   turn's thinking) and builds the prompt the chat template produces for a reply with that thinking: the rendered
   request (ending in '<|im_start|>assistant\\n<think>\\n') + the cut thinking (trimmed, as the template trims it) +
   '\\n</think>\\n\\n'. Every prompt is checked against the template itself: rendering the request plus an assistant
   message {content: "", reasoning: cut thinking} must give exactly these tokens followed by '<|im_end|>\\n';
3. lets the server continue that prompt at temperature 0 (/v1/completions with the token ids; at most 8,192 new
   tokens) and parses the answer and tool calls (qwen3_coder format, as the template writes them).

Variants per budget: "plain" (the cut thinking as it is) and "act-now" (one plain-English line appended to the cut
thinking: ACT_NOW). Control "full" (completed turns only): the whole recorded thinking, so the temperature-0 answer
can be compared with the recorded (sampled) answer: the agreement a cut can at best reach.

Comparison (the routing labeller's rules and judge): completed turns: the cut answer against the recorded action -
the same commands or the same full target (routing_labels.free_match), else the judge (Qwen3.8-Max on DashScope via
judge_service.py, thinking off, temperature 0, the owner's validated prompt with Step 1 = recorded, Step 2 = cut).
Limit hits have no recorded action: the next successful turn of the session is the reference when there is one (a
context-limit stop, after pi's compaction), and every limit-hit answer is also put to the judge with a plain question:
is this a reasonable next step (REASONABLE_SYSTEM).

Requests go to the server family that recorded the turn (B200 turns to the B200 servers, casdgx01 to casdgx01, the
RTX's to the RTX); load limits as the routing labeller (--per-server in flight, no new request while the vLLM queue
is above --max-waiting). Output: one JSONL row per (turn, variant, budget); a rerun skips written rows.

Usage (aiohttp and tokenizers, node 24 for the builder):
    uv run --with aiohttp==3.12.15 --with tokenizers python thinking_budget_reask.py run --turns TURNS.jsonl \\
        --trials TRIALS.jsonl --servers b200=URL,URL --servers casdgx01=URL --servers rtx-pro-6000=URL \\
        --judge-url URL --judge-model qwen3.8-max --node node --builder routing_requests.ts --out OUT.jsonl
"""

import argparse
import asyncio
import collections
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aiohttp  # noqa: E402

import routing_labels as rl  # noqa: E402

BUDGETS = (4096, 8192, 16384)
OUTPUT_LIMIT = 32768
ANSWER_MAX_TOKENS = 8192
SAMPLE_MIN_THINKING = 8192
LIMIT_CLASSES = ("output limit", "context limit")
ACT_NOW = ("I must stop thinking now and act. I will take the best next step based on what I have worked out so far.")
CLOSE_THINKING = "\n</think>\n\n"
END_OF_TURN = "<|im_end|>\n"

REASONABLE_SYSTEM = (
    "You review the work of an AI assistant that solves tasks on a Linux computer by running shell commands, one step "
    "at a time. You will see the task, the most recent steps the assistant took (each command with the end of its "
    "output), and one candidate for the assistant's next step. A step may contain several commands that run one "
    "after another, or it may be a final answer to the user with no command.\n\n"
    "Decide whether the candidate is a reasonable next step at this moment: it makes sensible progress on the task "
    "with what the assistant knows so far. It is not reasonable if it is wrong, works against the task, repeats work "
    "for no reason, or gives a final answer while the task is clearly not finished.\n\n"
    "Answer with YES or NO on the first line. On the second line, give one sentence explaining why."
)


def reasonable_messages(task, steps, candidate):
    shown = "\n\n".join(
        f"Command:\n{cmd[:1500]}\nEnd of output:\n{out[-600:] or '(no output)'}" for cmd, out in steps
    ) or "(none yet)"
    step = "\n\n".join(piece[:2500] for piece in candidate)
    user = (f"TASK:\n{task[:3000]}\n\nMOST RECENT STEPS (oldest first):\n{shown}\n\nCANDIDATE NEXT STEP:\n{step}\n\n"
            "Is the candidate a reasonable next step at this moment? Answer YES or NO on the first line, then one "
            "sentence why.")
    return [{"role": "system", "content": REASONABLE_SYSTEM}, {"role": "user", "content": user}]


TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.DOTALL)


def parse_answer(text):
    """The answer text before the first tool call and the tool calls ({name, arguments as JSON text}) of a reply in
    the qwen3_coder format the chat template writes. A '<tool_call>' that does not parse makes the reply unusable
    (None)."""
    text = text.removesuffix("<|im_end|>")
    head = text.split("<tool_call>", 1)[0]
    calls = []
    for match in TOOL_CALL.finditer(text):
        arguments = {name.strip(): value for name, value in PARAMETER.findall(match.group(2))}
        calls.append({"name": match.group(1).strip(), "arguments": json.dumps(arguments)})
    if text.count("<tool_call>") != len(calls):
        return head, None
    return head, calls


def machine_host(machine):
    """Server group of a recording machine tag (qwen3.8-27b-nvfp4@b200-gpu3-vllm0.29 -> b200)."""
    where = machine.split("@", 1)[1]
    if where.startswith("b200-"):
        return "b200"
    if where.startswith("casdgx01-"):
        return "casdgx01"
    if where.startswith("rtx-pro-6000"):
        return "rtx-pro-6000"
    raise ValueError(f"no server group for {machine}")


def sample_turns(turns, extra_4k_8k=0):
    """Completed turns with more than SAMPLE_MIN_THINKING thinking tokens and every limit hit, in a fixed order; plus
    `extra_4k_8k` completed turns with 4k-8k thinking tokens (a fixed random sample; asked at the 4k budget only)."""
    def order(t):
        return rl.fraction(f"thinking-budget:{t['trial_dir']}:{t['entry_id']}")
    chosen = [t for t in turns if t["entry_id"] and (
        (t["class"] == "completed" and t["thinking_tokens"] > SAMPLE_MIN_THINKING) or t["class"] in LIMIT_CLASSES)]
    middle = sorted((t for t in turns if t["entry_id"] and t["class"] == "completed"
                     and BUDGETS[0] < t["thinking_tokens"] <= SAMPLE_MIN_THINKING), key=order)
    return sorted(chosen, key=order) + middle[:extra_4k_8k]


def jobs_of(turn):
    """(variant, budget) pairs asked for a turn: plain and act-now per budget shorter than its thinking; full control
    for completed turns. Plain and full first (they are the main result)."""
    think = turn["thinking_tokens"] if turn["class"] == "completed" else OUTPUT_LIMIT
    budgets = [b for b in BUDGETS if b < think]
    first = [("plain", b) for b in budgets] + ([("full", None)] if turn["class"] == "completed" else [])
    return first, [("act-now", b) for b in budgets]


class Asker:
    def __init__(self, options):
        self.options = options
        self.groups = {}
        by_url = {}  # a server listed in two groups (RTX turns re-asked on the B200) keeps one load limit
        for item in options.servers:
            name, urls = item.split("=", 1)
            self.groups[name] = [by_url.setdefault(url, rl.Server(url.rstrip("/") + "/v1/completions", name,
                                                                  options.per_server, options.max_waiting))
                                 for url in urls.split(",")]
        self.judge = rl.JudgeClient(options.judge_url, options.judge_model)
        self.builder = rl.Builder(options.node, options.builder)
        self.http = None
        self.sink = None
        self.done = set()
        self.trials = {}
        self.written = 0
        self.counter = collections.Counter()

    async def post(self, server, path, body, what):
        try:
            async with self.http.post(server.base + path, json=body) as response:
                if response.status != 200:
                    raise RuntimeError(f"{what}: {path} HTTP {response.status}: {(await response.text())[:400]}")
                return await response.json()
        except aiohttp.ClientError as error:
            raise RuntimeError(f"{what}: {server.base}{path}: {error!r}") from error

    async def tokenize_chat(self, server, body, messages, generation_prompt, what):
        payload = {"model": body["model"], "messages": messages, "tools": body["tools"],
                   "chat_template_kwargs": body["chat_template_kwargs"], "add_generation_prompt": generation_prompt}
        return (await self.post(server, "/tokenize", payload, what))["tokens"]

    async def tokenize_text(self, server, model, text, what):
        return (await self.post(server, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False},
                                what))["tokens"]

    async def detokenize(self, server, model, tokens, what):
        return (await self.post(server, "/detokenize", {"model": model, "tokens": tokens}, what))["prompt"]

    async def complete(self, server, model, prompt_ids, what):
        await server.acquire()
        start = time.monotonic()
        try:
            reply = await self.post(server, "/v1/completions", {
                "model": model, "prompt": prompt_ids, "max_tokens": ANSWER_MAX_TOKENS, "temperature": 0,
                "skip_special_tokens": False}, what)
        finally:
            server.release()
        choice = reply["choices"][0]
        return {"text": choice["text"], "finish_reason": choice["finish_reason"],
                "prompt_tokens": reply["usage"]["prompt_tokens"],
                "completion_tokens": reply["usage"]["completion_tokens"], "seconds": time.monotonic() - start}

    async def judge_same(self, built, reference, candidate, what):
        messages = rl.judge_messages(built["task"], rl.recent_steps(built["body"]["messages"]),
                                     rl.step_pieces(reference["commands"], reference["final_text"]),
                                     rl.step_pieces(candidate["commands"], candidate["final_text"]))
        return await self.ask_judge(messages, what)

    async def judge_reasonable(self, built, candidate, what):
        messages = reasonable_messages(built["task"], rl.recent_steps(built["body"]["messages"]),
                                       rl.step_pieces(candidate["commands"], candidate["final_text"]))
        return await self.ask_judge(messages, what)

    async def ask_judge(self, messages, what):
        reply = await self.judge.ask(messages, what)
        verdict, reason = rl.parse_verdict(reply["content"])
        return {"verdict": verdict, "reason": reason, "model": f"{reply['model']} ({reply['backend']}, thinking off)"}

    async def check(self, built, reference, candidate, what):
        if candidate["commands"] is None:
            return {"good": False, "by": "no usable action"}
        if rl.free_match(candidate["commands"], reference["commands"]):
            return {"good": True, "by": "same step"}
        verdict = await self.judge_same(built, reference, candidate, what)
        return {"good": verdict["verdict"], "by": "judge", "judge": verdict}

    def server_for(self, turn):
        group = self.groups[machine_host(turn["machine"])]
        return min(group, key=lambda s: s.queued_turns)

    async def turn_jobs(self, turn, wanted):
        """All asked variants of one turn (wanted = list of (variant, budget) not yet written)."""
        key = f"{turn['trial_dir']}:{turn['entry_id']}"
        server = self.server_for(turn)
        server.queued_turns += 1
        try:
            session = self.trials[turn["trial_dir"]]["session_file"]
            built = await self.builder.build({"id": key, "session_file": session, "entry_id": turn["entry_id"]})
            body = built["body"]
            model = body["model"]
            prompt_ids = await self.tokenize_chat(server, body, body["messages"], True, key)
            recorded_prompt = turn["input_tokens"] + turn["cache_read"]
            if len(prompt_ids) != recorded_prompt:
                self.write({"kind": "excluded", "turn": key, "reason": "rebuilt prompt differs from the recorded one",
                            "server_prompt": len(prompt_ids), "recorded_prompt": recorded_prompt})
                return
            prompt_text = await self.detokenize(server, model, prompt_ids, key)
            entry = next(json.loads(line) for line in Path(session).read_text().splitlines()
                         if line.strip() and json.loads(line).get("id") == turn["entry_id"])
            content = entry["message"]["content"]
            thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
            calls = [p for p in content if p["type"] == "toolCall"]
            recorded = {"commands": [c["arguments"].get("command") for c in calls] if turn["class"] == "completed"
                        else None,
                        "final_text": "".join(p["text"] for p in content if p["type"] == "text") if not calls else None}
            thinking_ids = await self.tokenize_text(server, model, thinking, key)
            reference = await self.next_action(turn) if turn["class"] in LIMIT_CLASSES else None
            for variant, budget in wanted:
                what = f"{key} {variant}-{budget}"
                cut = thinking.strip() if variant == "full" else \
                    (await self.detokenize(server, model, thinking_ids[:budget], what)).strip()
                if variant == "act-now":
                    cut = f"{cut}\n\n{ACT_NOW}"
                prompt = prompt_text + cut + CLOSE_THINKING
                ids = await self.tokenize_text(server, model, prompt, what)
                template = await self.tokenize_chat(
                    server, body, body["messages"] + [{"role": "assistant", "content": "", "reasoning": cut}], False,
                    what)
                tail = await self.tokenize_text(server, model, END_OF_TURN, what)
                if template != ids + tail:
                    raise RuntimeError(f"{what}: the built prompt is not what the chat template renders for a reply "
                                       "with this thinking")
                answer = await self.complete(server, model, ids, what)
                if answer["prompt_tokens"] != len(ids):
                    raise RuntimeError(f"{what}: the server counted {answer['prompt_tokens']} prompt tokens, sent "
                                       f"{len(ids)}")
                head, tool_calls = parse_answer(answer["text"])
                usable = None
                if answer["finish_reason"] == "stop" and tool_calls is not None:
                    usable = rl.usable_commands({"outcome": "ok", "tool_calls": tool_calls})
                candidate = {"commands": usable, "final_text": head if usable == [] else None}
                row = {"kind": "ask", "turn": key, "trial_dir": turn["trial_dir"], "entry_id": turn["entry_id"],
                       "task": turn["task"], "dataset": turn["dataset"], "machine": turn["machine"],
                       "class": turn["class"], "turn_number": turn["turn"], "recorded_thinking": turn["thinking_tokens"],
                       "recorded_output": turn["output_tokens"], "recorded_model_ms": turn["model_ms"],
                       "thinking_tokens_retokenized": len(thinking_ids), "variant": variant, "budget": budget,
                       "prompt_tokens": len(ids), "base_prompt_tokens": len(prompt_ids), "server": server.base,
                       "answer": answer, "commands": usable, "final_text": candidate["final_text"],
                       "recorded_commands": recorded["commands"],
                       "thinks_again": "<think>" in answer["text"] or "</think>" in answer["text"]}
                if turn["class"] == "completed":
                    row["check"] = await self.check(built, recorded, candidate, f"{what} vs recorded")
                else:
                    row["reference_next"] = reference
                    if reference is not None:
                        row["check_next"] = await self.check(built, reference, candidate, f"{what} vs next turn")
                    row["reasonable"] = {"good": False, "by": "no usable action"} if usable is None else \
                        {"by": "judge", **(await self.judge_reasonable(built, candidate, f"{what} reasonable"))}
                    if usable is not None:
                        row["reasonable"]["good"] = row["reasonable"]["verdict"]
                self.write(row)
        except rl.JudgeRefused as problem:
            # Not a fallback: the judge's backend refused the input; the turn is left out with its reason (counted).
            self.write({"kind": "excluded", "turn": key, "reason": "the judge refused the input", "detail": str(problem)})
        finally:
            server.queued_turns -= 1

    async def next_action(self, turn):
        """The action of the session's next completed reply after a limit hit, or None when the session ended."""
        session = self.trials[turn["trial_dir"]]["session_file"]
        entries = [json.loads(line) for line in Path(session).read_text().splitlines() if line.strip()]
        after = False
        for entry in entries:
            if entry.get("id") == turn["entry_id"]:
                after = True
                continue
            if after and entry["type"] == "message" and entry["message"]["role"] == "assistant" \
                    and entry["message"]["stopReason"] in ("toolUse", "stop"):
                calls = [p for p in entry["message"]["content"] if p["type"] == "toolCall"]
                return {"commands": [c["arguments"].get("command") for c in calls],
                        "final_text": None if calls else "".join(p["text"] for p in entry["message"]["content"]
                                                                 if p["type"] == "text")}
        return None

    def write(self, row):
        self.sink.write(json.dumps(row) + "\n")
        self.sink.flush()
        self.written += 1
        self.counter[row["kind"]] += 1

    async def run(self):
        options = self.options
        out = Path(options.out)
        if out.exists():
            for line in out.read_text().splitlines():
                row = json.loads(line)
                self.done.add((row["turn"], row.get("variant"), row.get("budget")) if row["kind"] == "ask"
                              else (row["turn"], "excluded", None))
        self.sink = out.open("a")
        self.trials = {t["trial_dir"]: t for t in map(json.loads, Path(options.trials).read_text().splitlines())}
        turns = sample_turns([json.loads(line) for line in Path(options.turns).read_text().splitlines()],
                             options.extra_4k_8k)
        if options.max_turns:
            turns = turns[:options.max_turns]
        print(f"{len(turns)} sampled turns: {dict(collections.Counter(t['class'] for t in turns))}", flush=True)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
        # force_close: one connection per request. Reused keep-alive connections raced with the server closing idle
        # ones (ServerDisconnectedError through the SSH tunnels).
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=aiohttp.TCPConnector(limit=0, force_close=True)) as http:
            self.http = http
            print(f"judge: {await self.judge.check(http)}", flush=True)
            watchers = [asyncio.create_task(s.watch(http)) for s in {id(s): s for group in self.groups.values()
                                                                       for s in group}.values()]
            semaphore = asyncio.Semaphore(options.turns_in_flight)

            async def one(turn, wanted):
                async with semaphore:
                    await self.turn_jobs(turn, wanted)
                    print(f"{time.strftime('%H:%M:%S')} {self.written} rows {dict(self.counter)}", flush=True)

            for stage in (0, 1) if not options.plain_only else (0,):
                work = []
                for turn in turns:
                    key = f"{turn['trial_dir']}:{turn['entry_id']}"
                    if (key, "excluded", None) in self.done:
                        continue
                    wanted = [job for job in jobs_of(turn)[stage] if (key, *job) not in self.done]
                    if wanted:
                        work.append(one(turn, wanted))
                await asyncio.gather(*work)
            for task in watchers:
                task.cancel()
            await self.builder.close()
        print(f"done: {self.written} rows {dict(self.counter)}", flush=True)


# ------------------------------------------------------------------------------------------------ report


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def decode_rates(turns):
    """Output tokens per second of each recording machine: completed requests with more than 4,000 output tokens (long
    decodes, so prefill and queueing weigh little)."""
    sums = collections.defaultdict(lambda: [0, 0.0])
    for t in turns:
        if t["class"] == "completed" and t["output_tokens"] > 4000:
            sums[t["machine"]][0] += t["output_tokens"]
            sums[t["machine"]][1] += t["model_ms"] / 1000
    return {m: tokens / seconds for m, (tokens, seconds) in sums.items()}


def saved_tokens(row):
    """Output tokens the budget does not generate: the thinking beyond the budget plus the change in answer length
    (a limit hit's reply was the full output limit)."""
    answer = row["answer"]["completion_tokens"]
    if row["class"] == "completed":
        recorded_answer = row["recorded_output"] - row["recorded_thinking"]
        if row["variant"] == "full":
            return recorded_answer - answer
        return row["recorded_thinking"] - row["budget"] + recorded_answer - answer
    return OUTPUT_LIMIT - row["budget"] - answer


def verdicts(path, kind):
    """{(turn, variant, budget): good} of a paired or reasonable file (None = the judge refused the input)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["kind"] != kind:
            raise ValueError(f"{path}: a {row['kind']} row, expected {kind}")
        out[(row["turn"], row["variant"], row["budget"])] = row["check"]["good"]
    return out


def rate(values):
    """(good, judged) of verdicts; None (judge refused) is not counted."""
    judged = [v for v in values if v is not None]
    return sum(judged), len(judged)


def show(pair):
    return pct(*pair)


LENGTH_BINS = ((4096, 8192), (8192, 12288), (12288, 16384), (16384, 24576), (24576, OUTPUT_LIMIT + 1))


def length_bin(tokens):
    for lo, hi in LENGTH_BINS:
        if lo < tokens <= hi:
            return f"{lo // 1024}k-{hi // 1024}k" if hi <= OUTPUT_LIMIT else f"> {lo // 1024}k"
    raise ValueError(f"no length bin for {tokens} thinking tokens")


READY = re.compile(r"\b(let me|let's|i'll|i will|now i|i need to (run|check|write|look)|next,? i|so the plan|"
                   r"the plan is|i should (now )?(run|check|write|look|create))\b", re.IGNORECASE)
DOUBT = re.compile(r"\b(wait|hmm|actually|but what if|alternatively|however|on second thought|or maybe)\b",
                   re.IGNORECASE)


def tail_features(rows, tokenizer_path, trials):
    """For each cut answer: the last 600 characters of the thinking before the cut, and two simple signals a stop
    decision could see: an 'about to act' phrase and the number of doubt words."""
    from tokenizers import Tokenizer  # noqa: PLC0415 - optional, only for this table
    tokenizer = Tokenizer.from_file(tokenizer_path)
    cache = {}
    for row in rows:
        key = (row["trial_dir"], row["entry_id"])
        if key not in cache:
            session = trials[row["trial_dir"]]["session_file"]
            entry = next(json.loads(line) for line in Path(session).read_text().splitlines()
                         if line.strip() and json.loads(line).get("id") == row["entry_id"])
            thinking = "".join(p["thinking"] for p in entry["message"]["content"] if p["type"] == "thinking")
            cache[key] = tokenizer.encode(thinking, add_special_tokens=False).ids
        ids = cache[key]
        tail = tokenizer.decode(ids[max(0, row["budget"] - 200):row["budget"]])[-600:]
        row["tail_ready"] = bool(READY.search(tail[-300:]))
        row["tail_doubt"] = len(DOUBT.findall(tail))


def action_class(commands):
    """ceiling.py's class of an action (the highest kind among its commands; files written earlier in the session
    are not known here, so running one's own code counts as running other code)."""
    if commands is None:
        return "no usable action"
    if not commands:
        return "final answer"
    kinds = [k for command in commands for k, _ in rl.ceiling.classify_command(command, set())[0]
             if k != rl.ceiling.NEUTRAL]
    return next((p for p in rl.ceiling.PRECEDENCE if p in kinds), rl.ceiling.INFO)


ACTION_GROUPS = (("information", (rl.ceiling.INFO,)), ("write/edit or inline script",
                                                        (rl.ceiling.WRITE, rl.ceiling.INLINE)),
                 ("final answer", ("final answer",)))


def report_markdown(rows, turns, paired_v, reasonable_v, trials, tokenizer_path):
    rates = decode_rates(turns)
    total_s = sum(t["model_ms"] for t in turns) / 1000
    out = []
    w = out.append
    asks = [r for r in rows if r["kind"] == "ask"]
    excluded = [r for r in rows if r["kind"] == "excluded"]
    for r in asks:
        r["saved_s"] = saved_tokens(r) / rates[r["machine"]]
    by_turn = collections.defaultdict(dict)
    for r in asks:
        by_turn[r["turn"]][(r["variant"], r["budget"])] = r
    completed = [r for r in asks if r["class"] == "completed"]
    limits = [r for r in asks if r["class"] != "completed"]
    n_completed = len({r["turn"] for r in completed})
    w(f"Asked: {len(asks):,} answers for {len(by_turn)} turns ({n_completed} completed turns: all "
      f"{len({r['turn'] for r in completed if r['recorded_thinking'] > SAMPLE_MIN_THINKING})} with more than 8k "
      f"thinking tokens and a random {len({r['turn'] for r in completed if r['recorded_thinking'] <= SAMPLE_MIN_THINKING})}"
      f" with 4k-8k; {len({r['turn'] for r in limits})} limit hits). Left out: {len(excluded)}"
      + (f" ({dict(collections.Counter(e['reason'] for e in excluded))})" if excluded else "")
      + f". Completed turns without the full-thinking control (run unfinished): "
      f"{len({r['turn'] for r in completed if ('full', None) not in by_turn[r['turn']]})}. Prompt check: "
      "every rebuilt request's prompt tokens equal pi's recorded prompt tokens; every cut prompt equals the chat "
      "template's own rendering of a reply with that thinking (checked on every request; a difference stops the "
      "run); the server's prompt token count equals the tokens sent.\n")
    w("Decode rates used to turn tokens into seconds (output tokens per second of long completed requests of the "
      "recording machine, measured in the collection): " + ", ".join(
          f"{m.split('@')[1]} {v:.0f}" for m, v in sorted(rates.items())) + ".\n")
    finish = collections.Counter((r["variant"], r["answer"]["finish_reason"]) for r in asks)
    refused = sum(v is None for v in list(paired_v.values()) + list(reasonable_v.values()))
    w(f"Answer finish reasons: {dict(sorted(finish.items(), key=str))}; answers that start thinking again: "
      f"{sum(r['thinks_again'] for r in asks)}; answers without a usable action: "
      f"{sum(r['commands'] is None for r in asks)}; judge inputs refused by DashScope: {refused}.\n")

    w("### B.1 Completed turns: does the cut answer stay as good?\n")
    w("Three measures, all from the judge (Qwen3.8-Max, thinking off, temperature 0):\n")
    w("- reasonable: the plain-English question 'is this a reasonable next step at this moment?' asked about each "
      "answer on its own (cut answer, full-thinking answer, recorded action). The main quality measure.")
    w("- same as recorded: the routing labeller's rule (same step, else 'would Step 2 serve the task as well as "
      "Step 1?' with Step 1 = recorded action).")
    w("- same as full: the same rule with Step 1 = the answer after the whole recorded thinking at temperature 0 "
      "(the same request, only the cut differs: no sampling noise).\n")
    w("'full' continues the whole recorded thinking at temperature 0. The judge's 'same' is strict on long-thinking "
      "turns: even the full-thinking answer matches the recorded (sampled) one in only the share shown, and in the "
      "routing labels' calibration two independent xhigh samples of turns with more than 1k thinking tokens matched "
      "in 6.5% (23 turns, B200 labels). So 'reasonable' is the measure to read; 'same' is shown for completeness.\n")
    w("| budget | variant | turns | reasonable: cut | full | recorded | same as recorded: cut | full | same as full "
      "| answer tokens: cut | recorded | saved h (these turns) |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for budget in BUDGETS:
        for variant in ("plain", "act-now"):
            g = [r for r in completed if r["variant"] == variant and r["budget"] == budget]
            if not g:
                continue
            ts = [r["turn"] for r in g]
            fulls = [by_turn[t].get(("full", None)) for t in ts]
            w(f"| {budget // 1024}k | {variant} | {len(g)} | "
              f"{show(rate(reasonable_v.get((t, variant, budget)) for t in ts))} | "
              f"{show(rate(reasonable_v.get((t, 'full', None)) for t in ts))} | "
              f"{show(rate(reasonable_v.get((t, 'recorded', None)) for t in ts))} | "
              f"{show(rate(r['check']['good'] for r in g))} | "
              f"{show(rate(f['check']['good'] for f in fulls if f is not None))} | "
              f"{show(rate(paired_v.get((t, variant, budget)) for t in ts))} | "
              f"{statistics.mean(r['answer']['completion_tokens'] for r in g):.0f} | "
              f"{statistics.mean(r['recorded_output'] - r['recorded_thinking'] for r in g):.0f} | "
              f"{sum(r['saved_s'] for r in g) / 3600:.1f} |")
    w("")
    w("By recorded thinking length: reasonable, cut / full on the same turns (turns):\n")
    w("| thinking | " + " | ".join(f"{b // 1024}k {v}" for b in BUDGETS for v in ("plain", "act-now")) + " |")
    w("|---|" + "---:|" * (2 * len(BUDGETS)))
    for lo, hi in LENGTH_BINS:
        name = length_bin(hi if hi <= OUTPUT_LIMIT else OUTPUT_LIMIT)
        cells = []
        for budget in BUDGETS:
            for variant in ("plain", "act-now"):
                ts = [r["turn"] for r in completed if r["variant"] == variant and r["budget"] == budget
                      and length_bin(r["recorded_thinking"]) == name]
                cut = rate(reasonable_v.get((t, variant, budget)) for t in ts)
                full = rate(reasonable_v.get((t, "full", None)) for t in ts)
                cells.append(f"{show(cut)} / {show(full)} ({cut[1]})" if ts else "-")
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")

    w("What kind of step the answers take (completed turns; share of answers; information = reads, listings, "
      "searches; other = running programs, installs, file operations, tests):\n")
    w("| answer | n | " + " | ".join(name for name, _ in ACTION_GROUPS) + " | other |")
    w("|---|---:|" + "---:|" * (len(ACTION_GROUPS) + 1))
    columns = [("recorded", [r["recorded_commands"] for r in completed if r["variant"] == "full"]),
               ("full thinking", [r["commands"] for r in completed if r["variant"] == "full"])]
    columns += [(f"{b // 1024}k {v}", [r["commands"] for r in completed if r["variant"] == v and r["budget"] == b])
                for b in BUDGETS for v in ("plain", "act-now")]
    for name, actions in columns:
        if not actions:
            continue
        classes = collections.Counter(action_class(a) for a in actions)
        cells = [sum(classes[k] for k in kinds) for _, kinds in ACTION_GROUPS]
        w(f"| {name} | {len(actions)} | " + " | ".join(pct(c, len(actions)) for c in cells)
          + f" | {pct(len(actions) - sum(cells), len(actions))} |")
    w("")

    w("### B.2 Limit hits: is the cut answer a reasonable next step?\n")
    w("A limit hit has no recorded action. 'reasonable' as above. 'vs next turn': the session's next completed reply "
      "after the hit as Step 1 (routing rule); it exists only for context-limit stops (pi compacts and goes on): an "
      "output-limit hit ends the session, so the recorded outcome of every output-limit hit is a lost trial.\n")
    w("| budget | variant | hits | usable action | reasonable | vs next turn (n) | answer tokens | saved h |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    for budget in BUDGETS:
        for variant in ("plain", "act-now"):
            g = [r for r in limits if r["variant"] == variant and r["budget"] == budget]
            if not g:
                continue
            nxt = [r for r in g if r.get("check_next")]
            w(f"| {budget // 1024}k | {variant} | {len(g)} | {pct(sum(r['commands'] is not None for r in g), len(g))} "
              f"| {show(rate(r['reasonable']['good'] for r in g))} | "
              f"{show(rate(r['check_next']['good'] for r in nxt))} ({len(nxt)}) | "
              f"{statistics.mean(r['answer']['completion_tokens'] for r in g):.0f} | "
              f"{sum(r['saved_s'] for r in g) / 3600:.1f} |")
    w("")

    # Collection-level estimate.
    counts = collections.Counter(length_bin(t["thinking_tokens"]) for t in turns
                                 if t["class"] == "completed" and t["thinking_tokens"] > BUDGETS[0])
    sampled = collections.Counter(length_bin(r["recorded_thinking"]) for r in completed if r["variant"] == "full")

    def scale(r):
        return counts[length_bin(r["recorded_thinking"])] / sampled[length_bin(r["recorded_thinking"])]

    def option(chosen_completed, chosen_limits):
        """Time saved (share of all generation time) and the reasonable rates of a policy given the chosen answer
        per sampled turn (None = no cut)."""
        saved = sum(scale(r) * r["saved_s"] for r in chosen_completed.values() if r is not None)
        saved += sum(r["saved_s"] for r in chosen_limits.values() if r is not None)
        weight = cut_good = full_good = 0.0
        touched = 0.0
        for t, r in chosen_completed.items():
            full = by_turn[t][("full", None)]
            s = scale(full)
            v_full = reasonable_v.get((t, "full", None))
            v = v_full if r is None else reasonable_v.get((t, r["variant"], r["budget"]))
            if v is None or v_full is None:
                continue
            weight += s
            cut_good += s * v
            full_good += s * v_full
            touched += s if r is not None else 0
        lim_ok = rate(r["reasonable"]["good"] if r is not None else False for r in chosen_limits.values())
        return {"saved": saved / total_s, "touched": touched, "reasonable": cut_good / weight,
                "baseline": full_good / weight, "limits": lim_ok}

    # Options are compared on turns with the full-thinking control (all, once the run is complete; counted above).
    completed_turns = sorted({r["turn"] for r in completed if ("full", None) in by_turn[r["turn"]]})
    limit_turns = sorted({r["turn"] for r in limits})
    policies = {"no limit (baseline)": ({t: None for t in completed_turns}, {t: None for t in limit_turns})}
    for budget in BUDGETS:
        for variant in ("plain", "act-now"):
            policies[f"fixed {budget // 1024}k, {variant}"] = (
                {t: by_turn[t].get((variant, budget)) for t in completed_turns},
                {t: by_turn[t].get((variant, budget)) for t in limit_turns})

    def oracle(budgets, variant):
        """Stop at the smallest budget whose cut answer the judge calls reasonable; no cut when none is (a limit hit
        then stays a lost trial)."""
        chosen_c, chosen_l = {}, {}
        for t in completed_turns + limit_turns:
            pick = None
            for budget in budgets:
                r = by_turn[t].get((variant, budget))
                if r is None:
                    continue
                v = reasonable_v.get((t, variant, budget)) if r["class"] == "completed" else r["reasonable"]["good"]
                if v:
                    pick = r
                    break
            (chosen_c if t in completed_turns else chosen_l)[t] = pick
        return chosen_c, chosen_l

    policies["Jeff checkpoints 4k/8k/16k (oracle), plain"] = oracle(BUDGETS, "plain")
    policies["Jeff checkpoints 4k/8k/16k (oracle), act-now"] = oracle(BUDGETS, "act-now")
    policies["Jeff turn-start budget 8k/16k/none (oracle), plain"] = oracle((8192, 16384), "plain")
    w("### B.3 Options side by side (whole collection)\n")
    w(f"Time saved = output tokens not generated, in seconds at the recording machine's decode rate, as a share of "
      f"all Qwen generation time in the collection ({total_s / 3600:.0f} h). Completed turns above 8k are all in the "
      "sample; the 4k-8k turns are scaled up from their random sample. Reasonable = the judge's verdict on the answer "
      "the option would produce, on the sampled completed turns weighted like the collection; baseline = the same "
      "turns' full-thinking answers. Limit hits: share of the 37+3 hits that get a reasonable answer instead of a lost "
      "trial. 'Oracle' options use the judge's verdicts to pick the cut and are upper bounds for a Jeff decision.\n")
    w("| option | time saved | completed turns cut | reasonable (option) | reasonable (baseline, same turns) | limit "
      "hits rescued |")
    w("|---|---:|---:|---:|---:|---:|")
    results = {}
    for name, (chosen_c, chosen_l) in policies.items():
        res = option(chosen_c, chosen_l)
        results[name] = res
        w(f"| {name} | {100 * res['saved']:.1f}% | {res['touched']:,.0f} | {100 * res['reasonable']:.1f}% | "
          f"{100 * res['baseline']:.1f}% | {show(res['limits'])} |")
    w("")

    # Part C: signals.
    w("### C.1 Signals a decision could use\n")
    by_trial = collections.defaultdict(dict)
    for t in turns:
        if t["outcome"] in ("kept", "turn_ended"):
            by_trial[t["trial_dir"]][t["turn"]] = t
    starts = []
    for d in by_trial.values():
        for k in sorted(d):
            t = d[k]
            long_turn = t["class"] == "output limit" or (t["class"] == "completed" and t["thinking_tokens"] > 8192)
            starts.append((t, d.get(k - 1), long_turn))
    w("Turn start (decision a): share of turns that think more than 8k tokens or hit the output limit, by what is "
      "known before the turn starts.\n")
    w("| signal | turns | long |")
    w("|---|---:|---:|")
    groups = [("all", lambda x: True), ("turn 1", lambda x: x[0]["turn"] == 1),
              ("turns 2-5", lambda x: 2 <= x[0]["turn"] <= 5), ("turn > 5", lambda x: x[0]["turn"] > 5)]
    for lo, hi in ((0, 1000), (1000, 4096), (4096, 8192), (8192, 10 ** 6)):
        groups.append((f"previous turn thought {lo:,}-{hi:,}" if hi < 10 ** 6 else f"previous turn thought > {lo:,}",
                       lambda x, lo=lo, hi=hi: x[1] is not None and lo <= x[1]["thinking_tokens"] < hi))
    for ds in sorted({x[0]["dataset"] for x in starts}):
        groups.append((f"dataset {ds}", lambda x, ds=ds: x[0]["dataset"] == ds))
    for name, f in groups:
        g = [x for x in starts if f(x)]
        w(f"| {name} | {len(g):,} | {pct(sum(x[2] for x in g), len(g))} |")
    w("")
    plain_rows = [r for r in asks if r["variant"] == "plain"]
    if tokenizer_path:
        tail_features(plain_rows, tokenizer_path, trials)
        w("Checkpoint (decision b): reasonable rate of the plain cut by simple signals in the last part of the "
          "thinking before the cut (completed turns and limit hits; 'about to act' = a phrase such as 'let me', "
          "'I'll', 'now I', 'the plan is' in the last 300 characters; doubt words = wait, hmm, actually, however, "
          "alternatively, ... in the last 600 characters).\n")
        w("| signal | cuts | reasonable |")
        w("|---|---:|---:|")

        def verdict(r):
            return reasonable_v.get((r["turn"], "plain", r["budget"])) if r["class"] == "completed" \
                else r["reasonable"]["good"]
        for name, f in (("all", lambda r: True), ("about to act", lambda r: r["tail_ready"]),
                        ("not about to act", lambda r: not r["tail_ready"]),
                        ("no doubt words", lambda r: r["tail_doubt"] == 0),
                        ("1-2 doubt words", lambda r: 1 <= r["tail_doubt"] <= 2),
                        ("3+ doubt words", lambda r: r["tail_doubt"] >= 3)) + tuple(
                (f"checkpoint {b // 1024}k", lambda r, b=b: r["budget"] == b) for b in BUDGETS) + (
                ("completed turns", lambda r: r["class"] == "completed"),
                ("limit hits", lambda r: r["class"] != "completed")):
            g = [r for r in plain_rows if f(r)]
            w(f"| {name} | {len(g)} | {show(rate(verdict(r) for r in g))} |")
        w("")
    return "\n".join(out) + "\n", results


def report(options):
    rows = [json.loads(line) for line in Path(options.rows).read_text().splitlines() if line.strip()]
    turns = [json.loads(line) for line in Path(options.turns).read_text().splitlines()]
    trials = {t["trial_dir"]: t for t in map(json.loads, Path(options.trials).read_text().splitlines())}
    text, _ = report_markdown(rows, turns, verdicts(options.paired, "paired"),
                              verdicts(options.reasonable, "reasonable"), trials, options.tokenizer)
    Path(options.out).write_text(text)


# ------------------------------------------------------------------------------------------------ paired check


async def paired(options):
    """Each cut answer of a completed turn against the same turn's full-thinking answer (both at temperature 0, the
    same request; they differ only by the cut): same step, else the judge with Step 1 = full, Step 2 = cut. This removes
    the sampling noise of the recorded action. One "paired" row per cut answer; a rerun skips written rows."""
    rows = [json.loads(line) for line in Path(options.rows).read_text().splitlines() if line.strip()]
    trials = {t["trial_dir"]: t for t in map(json.loads, Path(options.trials).read_text().splitlines())}
    out = Path(options.out)
    done = set()
    if out.exists():
        done = {(r["turn"], r["variant"], r["budget"]) for r in map(json.loads, out.read_text().splitlines())}
    sink = out.open("a")
    asks = [r for r in rows if r["kind"] == "ask" and r["class"] == "completed"]
    full = {r["turn"]: r for r in asks if r["variant"] == "full"}
    todo = collections.defaultdict(list)
    for r in asks:
        if r["variant"] != "full" and r["turn"] in full and (r["turn"], r["variant"], r["budget"]) not in done:
            todo[r["turn"]].append(r)
    judge = rl.JudgeClient(options.judge_url, options.judge_model)
    builder = rl.Builder(options.node, options.builder)
    semaphore = asyncio.Semaphore(options.in_flight)
    counter = collections.Counter()

    async def one(turn, cuts):
        async with semaphore:
            reference = full[turn]
            session = trials[reference["trial_dir"]]["session_file"]
            built = await builder.build({"id": turn, "session_file": session, "entry_id": reference["entry_id"]})
            steps = rl.recent_steps(built["body"]["messages"])
            for cut in cuts:
                if cut["commands"] is None or reference["commands"] is None:
                    check = {"good": False, "by": "no usable action"}
                elif rl.free_match(cut["commands"], reference["commands"]):
                    check = {"good": True, "by": "same step"}
                else:
                    messages = rl.judge_messages(built["task"], steps,
                                                 rl.step_pieces(reference["commands"], reference["final_text"]),
                                                 rl.step_pieces(cut["commands"], cut["final_text"]))
                    try:
                        reply = await judge.ask(messages, f"{turn} {cut['variant']}-{cut['budget']} vs full")
                    except rl.JudgeRefused as problem:
                        # Not a fallback: the judge's backend refused the input; the pair is written as refused and
                        # counted, never given a verdict.
                        check = {"good": None, "by": "judge refused", "detail": str(problem)}
                    else:
                        verdict, reason = rl.parse_verdict(reply["content"])
                        check = {"good": verdict, "by": "judge", "reason": reason,
                                 "model": f"{reply['model']} ({reply['backend']}, thinking off)"}
                sink.write(json.dumps({"kind": "paired", "turn": turn, "variant": cut["variant"],
                                       "budget": cut["budget"], "check": check}) + "\n")
                sink.flush()
                counter[check["by"]] += 1

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=1800)) as http:
        print(f"judge: {await judge.check(http)}", flush=True)
        await asyncio.gather(*(one(turn, cuts) for turn, cuts in todo.items()))
        await builder.close()
    print(f"paired: {dict(counter)}", flush=True)


async def reasonable_all(options):
    """Every completed turn's recorded action, full-thinking answer and cut answers put to the judge's plain question
    'is this a reasonable next step?' (REASONABLE_SYSTEM): an absolute measure of each answer, free of the noise of
    comparing two answers that may differ in harmless details. One "reasonable" row per answer; resumable."""
    rows = [json.loads(line) for line in Path(options.rows).read_text().splitlines() if line.strip()]
    trials = {t["trial_dir"]: t for t in map(json.loads, Path(options.trials).read_text().splitlines())}
    out = Path(options.out)
    done = set()
    if out.exists():
        done = {(r["turn"], r["variant"], r["budget"]) for r in map(json.loads, out.read_text().splitlines())}
    sink = out.open("a")
    asks = [r for r in rows if r["kind"] == "ask" and r["class"] == "completed"]
    todo = collections.defaultdict(list)
    for r in asks:
        if r["variant"] == "full" and (r["turn"], "recorded", None) not in done:
            todo[r["turn"]].append(("recorded", None, r, {"commands": r["recorded_commands"],
                                                           "final_text": None}))
        if (r["turn"], r["variant"], r["budget"]) not in done:
            todo[r["turn"]].append((r["variant"], r["budget"], r, {"commands": r["commands"],
                                                                    "final_text": r["final_text"]}))
    judge = rl.JudgeClient(options.judge_url, options.judge_model)
    builder = rl.Builder(options.node, options.builder)
    semaphore = asyncio.Semaphore(options.in_flight)
    counter = collections.Counter()

    async def one(turn, items):
        async with semaphore:
            first = items[0][2]
            session = trials[first["trial_dir"]]["session_file"]
            built = await builder.build({"id": turn, "session_file": session, "entry_id": first["entry_id"]})
            if built["body"]["messages"] is None:
                raise RuntimeError(f"{turn}: no messages")
            steps = rl.recent_steps(built["body"]["messages"])
            for variant, budget, row, candidate in items:
                if variant == "recorded" and not candidate["commands"]:
                    # A recorded final answer: its text is in the session, not in the ask row.
                    entries = [json.loads(line) for line in Path(session).read_text().splitlines() if line.strip()]
                    entry = next(e for e in entries if e.get("id") == row["entry_id"])
                    candidate = {"commands": [], "final_text": "".join(
                        p["text"] for p in entry["message"]["content"] if p["type"] == "text")}
                if candidate["commands"] is None:
                    check = {"good": False, "by": "no usable action"}
                else:
                    messages = reasonable_messages(built["task"], steps,
                                                   rl.step_pieces(candidate["commands"], candidate["final_text"]))
                    try:
                        reply = await judge.ask(messages, f"{turn} {variant}-{budget} reasonable")
                    except rl.JudgeRefused as problem:
                        # Not a fallback: written as refused and counted, never given a verdict.
                        check = {"good": None, "by": "judge refused", "detail": str(problem)}
                    else:
                        verdict, reason = rl.parse_verdict(reply["content"])
                        check = {"good": verdict, "by": "judge", "reason": reason,
                                 "model": f"{reply['model']} ({reply['backend']}, thinking off)"}
                sink.write(json.dumps({"kind": "reasonable", "turn": turn, "variant": variant, "budget": budget,
                                       "check": check}) + "\n")
                sink.flush()
                counter[(variant, check["good"])] += 1

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=1800)) as http:
        print(f"judge: {await judge.check(http)}", flush=True)
        await asyncio.gather(*(one(turn, items) for turn, items in todo.items()))
        await builder.close()
    print(f"reasonable: {dict(counter)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run")
    r.add_argument("--turns", required=True, help="per-request JSONL of thinking_budget.py offline")
    r.add_argument("--trials", required=True, help="per-trial JSONL of thinking_budget.py offline")
    r.add_argument("--servers", action="append", required=True,
                   help="GROUP=URL[,URL...] (GROUP: b200, casdgx01, rtx-pro-6000); base URLs without /v1")
    r.add_argument("--judge-url", required=True)
    r.add_argument("--judge-model", required=True)
    r.add_argument("--node", required=True)
    r.add_argument("--builder", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--per-server", type=int, default=4)
    r.add_argument("--max-waiting", type=int, default=4)
    r.add_argument("--turns-in-flight", type=int, default=24)
    r.add_argument("--max-turns", type=int)
    r.add_argument("--plain-only", action="store_true", help="skip the act-now variant")
    r.add_argument("--extra-4k-8k", type=int, default=0, help="also ask this many completed turns with 4k-8k thinking")
    rep = sub.add_parser("report")
    for name in ("--rows", "--turns", "--trials", "--paired", "--reasonable", "--out"):
        rep.add_argument(name, required=True)
    rep.add_argument("--tokenizer", help="tokenizer.json of Qwen3.8 (signals in the thinking before the cut)")
    pr = sub.add_parser("paired", help="cut answers against the full-thinking answer of the same turn")
    pr.add_argument("--rows", required=True)
    pr.add_argument("--trials", required=True)
    pr.add_argument("--judge-url", required=True)
    pr.add_argument("--judge-model", required=True)
    pr.add_argument("--node", required=True)
    pr.add_argument("--builder", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--in-flight", type=int, default=16)
    ra = sub.add_parser("reasonable", help="every completed-turn answer put to the 'reasonable next step' judge")
    for name in ("--rows", "--trials", "--judge-url", "--judge-model", "--node", "--builder", "--out"):
        ra.add_argument(name, required=True)
    ra.add_argument("--in-flight", type=int, default=16)
    options = parser.parse_args()
    if options.step == "paired":
        asyncio.run(paired(options))
        return
    if options.step == "reasonable":
        asyncio.run(reasonable_all(options))
        return
    if options.step == "report":
        report(options)
        return
    try:
        asyncio.run(Asker(options).run())
    except BaseException:  # noqa: BLE001 - not swallowed: printed, then the process ends with code 1
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
