"""Send the rebuilt Qwen requests of the thinking-off comparison to the Qwen servers and record what comes back.

Runs on casdgx01 (the servers at 192.168.2.10:8885-8892 are reachable only from there). Needs aiohttp.

Usage:
    python thinking_off_run.py --bodies BODIES.jsonl --out RESULTS.jsonl [--only IDS.txt] [--per-server 48]

BODIES.jsonl holds one line per rebuilt request body, {"id": "<turn id>|on", "body": {...}} (also "|off" and "|cut"),
as written by thinking_off_requests.ts. For every turn four requests are sent, all streamed:
  A  thinking on, the body exactly as Jeff-Code sent it (server-default sampling), at most 32,768 output tokens;
  B  thinking off, temperature 0, at most 8,192 output tokens;
  C  thinking off, temperature 0.7, top_p 0.8, top_k 20 (Qwen's recommended non-thinking sampling), at most 8,192;
  D  as C, on the cut context (system prompt, task message, last 3 tool steps);
  E  thinking on with reasoning_effort "medium" (a chat-template switch; A has none, so the template's default
     "xhigh" applies, as in the sessions), server-default sampling, at most 32,768 output tokens;
  F  as E with reasoning_effort "low".
A generation that starts looping (see `looping`) is cancelled at once and recorded with outcome "loop".
Servers are used only once they answer /v1/models (checked at the start and every 2 minutes), so servers can join
during a run. Work is handed out by turn: a worker takes the next turn and sends all its missing variants at once
to one server (so they share cached prompt prefixes there). Turns that already have some results go first.
RESULTS.jsonl gets one line per request; a rerun skips the requests already in it.
With --prompt-test, only the prompt-processing time of each variant is measured (see prompt_test).
"""

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import aiohttp

SERVERS = [f"http://192.168.2.10:{port}/v1/chat/completions" for port in range(8885, 8893)]
# variant: (rebuilt body, request parameters, chat-template switches)
VARIANTS = {
    "A": ("on", {"max_completion_tokens": 32768}, {}),
    "B": ("off", {"max_completion_tokens": 8192, "temperature": 0}, {"enable_thinking": False}),
    "C": ("off", {"max_completion_tokens": 8192, "temperature": 0.7, "top_p": 0.8, "top_k": 20},
          {"enable_thinking": False}),
    "D": ("cut", {"max_completion_tokens": 8192, "temperature": 0.7, "top_p": 0.8, "top_k": 20},
          {"enable_thinking": False}),
    "E": ("on", {"max_completion_tokens": 32768}, {"enable_thinking": True, "reasoning_effort": "medium"}),
    "F": ("on", {"max_completion_tokens": 32768}, {"enable_thinking": True, "reasoning_effort": "low"}),
}
CHECK_EVERY = 200  # characters of new text between two loop checks of the same text


def looping(text):
    """Why the end of `text` is a generation loop, or None.

    A loop is either (1) the last 400 characters consist of one piece of 20 to 80 characters repeated (so at least 5
    repetitions), or (2) one line occurs 8 or more times within the last 60 lines. For (2) only lines with at least 10
    non-space characters count: short lines such as `}`, `fi` or `</div>` recur in ordinary code (a heredoc writing a
    C file has many `}` lines) and would cut normal file writes."""
    tail = text[-400:]
    if len(tail) == 400:
        for period in range(20, 81):
            if tail[period:] == tail[:-period]:
                return f"the last 400 characters repeat a {period}-character piece: {tail[:period]!r}"
    lines = [line.strip() for line in text.rsplit("\n", 60)[-60:]]
    counts = {}
    for line in lines:
        if len(line.replace(" ", "")) >= 10:
            counts[line] = counts.get(line, 0) + 1
    for line, count in counts.items():
        if count >= 8:
            return f"the line {line[:120]!r} occurs {count} times in the last 60 lines"
    return None


async def answers(http, url):
    """Whether the server at this chat-completions URL answers its model list."""
    try:
        async with http.get(url.replace("/chat/completions", "/models"), timeout=aiohttp.ClientTimeout(total=5)) as r:
            return r.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


def server_for(turn_id):
    session = turn_id.split(":")[0]
    return SERVERS[int(hashlib.sha256(session.encode()).hexdigest(), 16) % len(SERVERS)]


async def send(http, url, turn_id, variant, body, machine):
    _, params, switches = VARIANTS[variant]
    body = dict(body)
    body.pop("max_tokens", None)
    body.update(params)
    body["chat_template_kwargs"] = {**body["chat_template_kwargs"], **switches}
    record = {"id": turn_id, "variant": variant, "machine": machine, "server": url, "outcome": None, "finish_reason": None,
              "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None, "chunks": 0,
              "first_s": None, "total_s": None, "reasoning": "", "content": "", "tool_calls": [], "loop": None}
    reasoning = []
    content = []
    calls = {}
    checked = {"reasoning": 0, "answer": 0}
    reasoning_len = 0
    answer_len = 0
    start = time.monotonic()
    async with http.post(url, json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"{turn_id} {variant}: HTTP {response.status}: {(await response.text())[:500]}")
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
                if delta.get("reasoning"):
                    reasoning.append(delta["reasoning"])
                    reasoning_len += len(delta["reasoning"])
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
            # Loop checks, on the thinking and on the answer (text plus tool-call arguments, JSON escapes undone).
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
    record["total_s"] = time.monotonic() - start
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
        raise RuntimeError(f"{turn_id} {variant}: stream ended without a finish reason ({record['finish_reason']})")
    if record["outcome"] != "loop" and record["completion_tokens"] is None:
        raise RuntimeError(f"{turn_id} {variant}: no usage report at the end of the stream")
    return record


async def prompt_time(http, url, body, switches):
    """Seconds to process the prompt: a non-streamed request for a single output token."""
    body = dict(body)
    body.pop("max_tokens", None)
    body.update({"stream": False, "max_completion_tokens": 1})
    body.pop("stream_options", None)
    body["chat_template_kwargs"] = {**body["chat_template_kwargs"], **switches}
    start = time.monotonic()
    async with http.post(url, json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
        reply = await response.json()
    return time.monotonic() - start, reply["usage"]["prompt_tokens"]


async def prompt_test(args, bodies, turn_ids):
    """The cache effect of switching variants mid-session. For each turn, one at a time per server: first the
    thinking-on request exactly as the session sent it (A) is processed, so that its prompt is cached as it would be
    in a running session; then the variant's request is timed. A, E (medium) and F (low) use the same messages as
    A, B the thinking-off messages, D the cut context. Writes {id, variant, prompt_tokens, warm_s, prompt_s}."""
    by_server = {url: [] for url in SERVERS}
    for turn_id in turn_ids:
        by_server[server_for(turn_id)].append(turn_id)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
    with Path(args.out).open("a") as sink:
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async def worker(url, queue):
                for turn_id in queue:
                    for variant in ("A", "B", "D", "E", "F"):
                        kind, _, switches = VARIANTS[variant]
                        warm_s, _ = await prompt_time(http, url, bodies[f"{turn_id}|on"], {})
                        prompt_s, tokens = await prompt_time(http, url, bodies[f"{turn_id}|{kind}"], switches)
                        sink.write(json.dumps({"id": turn_id, "variant": variant, "prompt_tokens": tokens,
                                               "warm_s": warm_s, "prompt_s": prompt_s}) + "\n")
                        sink.flush()

            await asyncio.gather(*(worker(url, queue) for url, queue in by_server.items()))
    print("prompt test done", flush=True)


async def main_async(args):
    bodies = {}
    for line in open(args.bodies):
        row = json.loads(line)
        bodies[row["id"]] = row["body"]
    turn_ids = sorted({key.rsplit("|", 1)[0] for key in bodies})
    if args.only:
        wanted = set(Path(args.only).read_text().split())
        missing = wanted - set(turn_ids)
        if missing:
            raise ValueError(f"{len(missing)} turn ids of --only have no bodies, e.g. {sorted(missing)[0]}")
        turn_ids = [t for t in turn_ids if t in wanted]
    if args.exclude:
        excluded = set(Path(args.exclude).read_text().split())
        turn_ids = [t for t in turn_ids if t not in excluded]
    if args.prompt_test:
        await prompt_test(args, bodies, turn_ids)
        return
    done = set()
    turn_machine = {}  # the machine a started turn ran on; its other variants must run there too
    out = Path(args.out)
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["id"], row["variant"]))
            if turn_machine.setdefault(row["id"], row["machine"]) != row["machine"]:
                raise ValueError(f"{row['id']}: results from two machines")
    started_turns = set(turn_machine)
    pending = {t: [v for v in VARIANTS if (t, v) not in done] for t in turn_ids}
    # Started turns first; the rest in a fixed random order, so that a run stopped early still covers all tasks.
    fresh = [t for t in turn_ids if pending[t] and t not in started_turns]
    random.Random(20261004).shuffle(fresh)
    queue = [t for t in turn_ids if pending[t] and t in started_turns] + fresh
    total = sum(len(pending[t]) for t in queue)
    print(f"{total} requests for {len(queue)} turns to send ({len(done)} already done)", flush=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
    connector = aiohttp.TCPConnector(limit=0)
    finished = 0
    started = time.monotonic()
    with out.open("a") as sink:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
            async def run_variant(url, machine, turn_id, variant):
                nonlocal finished
                key = f"{turn_id}|{VARIANTS[variant][0]}"
                try:
                    record = await send(http, url, turn_id, variant, bodies[key], machine)
                except aiohttp.ClientError as error:
                    raise RuntimeError(f"{url} ({machine}), {turn_id} {variant}: {error!r}") from error
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                finished += 1
                if finished % 100 == 0:
                    print(f"{finished}/{total} after {time.monotonic() - started:.0f} s", flush=True)

            async def worker(url, machine):
                while True:
                    turn_id = next((t for t in queue if turn_machine.get(t, machine) == machine), None)
                    if turn_id is None:
                        return
                    queue.remove(turn_id)
                    turn_machine[turn_id] = machine
                    await asyncio.gather(*(run_variant(url, machine, turn_id, v) for v in pending[turn_id]))

            servers = [(url, args.machine, args.turns_per_server) for url in args.servers.split(",")]
            for extra in args.extra_server:
                url, machine, turns = extra.split("|")
                servers.append((url, machine, int(turns)))
            active = {}
            while queue:
                for url, machine, turns in servers:
                    if url not in active and await answers(http, url):
                        print(f"using {url} ({machine}, {turns} turns at a time)", flush=True)
                        active[url] = [asyncio.create_task(worker(url, machine)) for _ in range(turns)]
                if not active:
                    raise RuntimeError("no Qwen server answers /v1/models")
                tasks = [task for group in active.values() for task in group]
                if all(task.done() for task in tasks):
                    await asyncio.sleep(120)  # the turns left wait for a server of their machine
                done_now, _ = await asyncio.wait(tasks, timeout=120, return_when=asyncio.FIRST_EXCEPTION)
                for task in done_now:
                    if task.exception():
                        raise task.exception()
            await asyncio.gather(*(task for group in active.values() for task in group))
    print(f"done: {finished} requests in {time.monotonic() - started:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bodies", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only")
    parser.add_argument("--machine", required=True, help="tag for every record, e.g. casdgx01-h100")
    parser.add_argument("--servers", default=",".join(SERVERS), help="comma-separated chat-completions URLs")
    parser.add_argument("--exclude", help="file of turn ids another runner answers")
    parser.add_argument("--extra-server", action="append", default=[],
                        help="URL|MACHINE|TURNS: one more server with its own machine tag and turns in flight")
    parser.add_argument("--turns-per-server", type=int, default=9,
                        help="turns in flight per server; each sends up to six requests at once")
    parser.add_argument("--prompt-test", action="store_true",
                        help="time prompt processing per variant after caching the session's own request (see "
                             "prompt_test); use with --only and a separate --out")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
