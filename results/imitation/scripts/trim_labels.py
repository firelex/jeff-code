"""Output-trimming labels: does the coding model still take a good step when it sees only the end of the newest output?

For a sampled recorded Qwen3.8-27B turn of the xhigh collection whose newest tool output (the tool result just before
the turn) shows more than 40 lines, this script asks the same request at the turn's ROUTING LABEL (the cheapest good
thinking level from routing_labels.py; off, low, medium or xhigh, as run time would ask it) in a paired, deterministic
comparison (controller, 2026-10-04): the UNCUT request and each CUT request, all at temperature 0. The cut text is made by
JeffFirst's own function (output-trim.ts, through trim_requests.ts): the kept lines plus a note in Jeff-Code's truncation form
saying how many lines are not shown. First the uncut request and the three 40-line cuts together (its last 40 lines; its
first 20 and last 20; its first 40), and when none of the cuts is good and more than 200 lines show, its last 200 lines.
Label = the first good 40-line cut in the order last40, first20last20, first40; else last200 when good; else "all".
Every asked request is checked and recorded.

Good = the cut request's action is the same step as the UNCUT temperature-0 action without asking
(routing_labels.free_match: the same commands, or the same intent with every target named in full), else the judge's YES
with the uncut action as Step 1 and the cut action as Step 2 (the owner's validated prompt, Qwen3.8-Max through
judge_service.py, thinking off, temperature 0; the judge sees the recent steps of the uncut request). The label then
measures only the effect of the cut, not sampling noise. A cut reply without a usable action is not good and not judged;
a turn whose uncut reply has no usable action (output cap, generation loop) is left out ("excluded", with the reason).
Only turns that already have a routing label can be sampled; the script reads the routing labeller's output files as
they grow.

Requests go to servers of the model format named by --family; a routing-labelled turn recorded in the other format
raises unless --allow-cross-format is given (owner, 2026-10-04 17:25: FP8 and NVFP4 answer near-identically; the
sampling noise is larger than the quantisation difference). Every output line records `recording_format` (the
session's) and `reask_format` (the servers'). All turns of one trial go to one server, with the routing labeller's load limits (--per-server in flight, no new request
while the server's vLLM queue is above --max-waiting). Each request records its level, kept lines, prompt tokens and the
prompt tokens saved against the uncut request at the same level (its prompt must equal the routing labeller's request at
that level, or the recorded turn's for xhigh; a saving is negative when the note costs more than the lines it replaces).

Output: trim-labels-NAME.jsonl (trim-labels-NAME-TAG.jsonl with --file-tag TAG) per --routing NAME=FILE[,FILE...] in
--out-dir: a "turn" line per labelled turn, a
"short" line per routing-labelled turn whose newest output shows at most 40 lines (or has none), an "excluded" line
when the judge refused the input or the uncut reply had no usable action. Resumable: a rerun skips written turns.

Usage (needs aiohttp; node runs the request builder):
    uv run --with aiohttp==3.12.15 --with tokenizers python trim_labels.py run \\
        --routing b200=/raid/.../routing-labels-b200.jsonl --family nvfp4 --servers URL,URL --machine b200-nvfp4 \\
        --judge-url http://127.0.0.1:18905 --judge-model qwen3.8-max --out-dir OUT --node node \\
        --builder trim_requests.mjs --sample-rate 1.0 [--moved OLD_PREFIX=NEW_PREFIX]
"""

import argparse
import asyncio
import collections
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aiohttp  # noqa: E402

import routing_labels as rl  # noqa: E402

FORTY = ("last40", "first20last20", "first40")  # asked together; the label is the first good one in this order
WIDE = "last200"  # asked when no 40-line cut is good
LABELS = FORTY + (WIDE, "all")
SAMPLE_SEED = "trim-sample-20261004"
SHORT = "short"
UNCUT = "all"
UNUSABLE_UNCUT = "the uncut temperature-0 reply has no usable action (output cap, generation loop or bad call)"


def sampled(turn_id, rate):
    return rl.fraction(f"{SAMPLE_SEED}:{turn_id}") < rate


class UnusableUncut(Exception):
    """The uncut request's reply has no usable action, so there is nothing to compare the cuts with."""


def reference_prompt(routing_row):
    """Prompt tokens of the unshortened request at the turn's routing level: the routing labeller's request at that
    level, or the recorded turn for xhigh (the routing labeller's rebuilt xhigh request equals it exactly)."""
    level = routing_row["label"]
    if level == "xhigh":
        return routing_row["recorded"]["prompt_tokens"]
    asks = [a for a in routing_row["asks"] if a["level"] == level and a["sample"] == 1]
    if len(asks) != 1 or asks[0]["prompt_tokens"] is None:
        raise ValueError(f"{routing_row['id']}: no prompt tokens of the routing request at level {level}")
    return asks[0]["prompt_tokens"]


async def trim_cascade(asker, available):
    """The label of one turn. `asker.ask(cut)` returns a reply ({"commands": list or None, "final_text"}) of the request
    with the newest output cut that way ("all" = uncut); `asker.judge(reference, alternative, key)` returns {"verdict":
    bool, "reason"}. `available` = the cuts that shorten this output (output-trim.ts availableCuts). Raises
    UnusableUncut when the uncut reply has no usable action."""
    asked = [UNCUT] + [cut for cut in FORTY if cut in available]
    replies = dict(zip(asked, await asyncio.gather(*(asker.ask(cut) for cut in asked))))
    uncut = replies[UNCUT]
    if uncut["commands"] is None:
        raise UnusableUncut(UNUSABLE_UNCUT)
    checks = {}

    async def check(cut):
        reply = replies[cut]
        if reply["commands"] is None:
            checks[cut] = {"good": False, "by": "no usable action"}
        elif rl.free_match(reply["commands"], uncut["commands"]):
            checks[cut] = {"good": True, "by": "intent"}
        else:
            verdict = await asker.judge(uncut, reply, f"{cut} vs uncut")
            checks[cut] = {"good": verdict["verdict"], "by": "judge"}

    await asyncio.gather(*(check(cut) for cut in asked[1:]))
    label = next((cut for cut in FORTY if cut in checks and checks[cut]["good"]), None)
    if label is None and WIDE in available:
        replies[WIDE] = await asker.ask(WIDE)
        await check(WIDE)
        if checks[WIDE]["good"]:
            label = WIDE
    return {"label": label or UNCUT, "replies": replies, "checks": checks}


def paired_body(built, cut, level):
    """The request at `level` with the newest output cut (`cut`, or "all" for the uncut request), at temperature 0."""
    source = built["body"] if cut == UNCUT else built["trimmed"][cut]
    body = rl.variant_body({"body": source, "kwargs": built["kwargs"]}, level)
    body["temperature"] = 0
    return body


def resolve_trial(trial_dir, moved):
    """The trial folder now: routing rows written before a collection folder moved name its old place."""
    path = Path(trial_dir)
    if path.is_dir():
        return path
    for old, new in moved:
        if trial_dir.startswith(old):
            candidate = Path(new + trial_dir[len(old):])
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"trial folder {trial_dir} does not exist (and no --moved prefix finds it)")


def session_file(trial):
    sessions = sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if len(sessions) != 1:
        raise ValueError(f"{trial}: {len(sessions)} session files")
    return sessions[0]


class TrimBuilder:
    """trim_requests.ts / .mjs as a long-running child process, one job at a time."""

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
            if built["eligible"] and built["body"]["chat_template_kwargs"] != {**rl.SENT_XHIGH,
                                                                                 "preserve_thinking": True}:
                raise RuntimeError(f"{turn['id']}: rebuilt kwargs {built['body']['chat_template_kwargs']} are not what "
                                   "the session sent")
            return built


class TrimAsker:
    def __init__(self, labeller, turn, built, server):
        """ask(cut) sends the request with the newest output cut that way ("all" = uncut) at the turn's level, T=0."""
        self.labeller = labeller
        self.turn = turn
        self.built = built
        self.server = server
        self.level = turn["routing"]["label"]
        self.reference_prompt = reference_prompt(turn["routing"])
        self.asks = []
        # The routing labeller's judge call: recent steps from the uncut request.
        self.judge_asker = rl.TurnAsker(labeller, turn, built, server)

    async def ask(self, cut):
        body = paired_body(self.built, cut, self.level)
        what = f"{self.turn['id']} {self.level} {cut}"
        record = await rl.stream(self.labeller.http, self.server, body, what)
        commands = rl.usable_commands(record)
        saved = None if record["prompt_tokens"] is None else self.reference_prompt - record["prompt_tokens"]
        if cut == UNCUT and saved not in (None, 0):
            raise rl.PromptMismatch(f"{what}: the uncut request's prompt differs from the routing request's (or, at "
                                    f"xhigh, the recorded turn's) by {saved} tokens")
        record.update({"level": self.level, "cut": cut, "commands": commands,
                       "final_text": record["content"] if commands == [] else None, "prompt_saved": saved})
        del record["reasoning"]
        self.asks.append(record)
        return record

    async def judge(self, reference, alternative, key):
        return await self.judge_asker.judge(reference, alternative, key)


class TrimLabeller:
    def __init__(self, options):
        self.options = options
        self.sources = []
        for item in options.routing:
            name, files = item.split("=", 1)
            self.sources.append((name, files.split(",")))
        self.moved = [tuple(item.split("=", 1)) for item in options.moved or []]
        self.out_dir = Path(options.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.servers = [rl.Server(url, options.machine, options.per_server, options.max_waiting)
                        for url in options.servers.split(",")]
        self.judge_client = rl.JudgeClient(options.judge_url, options.judge_model)
        self.builder = TrimBuilder(options.node, options.builder)
        self.done = set()
        self.offsets = {}
        self.sinks = {}
        self.queues = {s.url: asyncio.Queue() for s in self.servers}
        self.trial_server = {}
        self.written = 0
        self.labelled = 0
        self.started = time.monotonic()
        self.http = None

    def load_done(self):
        for name, _ in self.sources:
            path = trim_file(self.out_dir, name, self.options.file_tag)
            if path.exists():
                for line in path.read_text().splitlines():
                    self.done.add(json.loads(line)["id"])
            self.sinks[name] = path.open("a")

    def write(self, source, row):
        self.sinks[source].write(json.dumps(row) + "\n")
        self.sinks[source].flush()

    def new_routing_rows(self, path):
        """Routing "turn" rows appended to `path` since the last read (whole lines only)."""
        offset = self.offsets.get(path, 0)
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
        end = data.rfind(b"\n") + 1
        self.offsets[path] = offset + end
        rows = [json.loads(line) for line in data[:end].decode().splitlines() if line.strip()]
        return [row for row in rows if row["kind"] == "turn"]

    def discover(self):
        added = 0
        for source, files in self.sources:
            for path in files:
                if not Path(path).exists():
                    raise FileNotFoundError(f"routing label file {path} does not exist")
                for row in self.new_routing_rows(path):
                    if row["id"] in self.done or not sampled(row["id"], self.options.sample_rate):
                        continue
                    recording_format = rl.family(row["recording_machine"])
                    rl.check_format(row["id"], recording_format, self.options.family, self.options.allow_cross_format)
                    trial = resolve_trial(row["trial_dir"], self.moved)
                    turn = {key: row[key] for key in ("id", "trial", "task", "class", "turn", "recording_machine",
                                                        "session_id", "entry_id", "commands", "final_text",
                                                        "recorded")}
                    turn.update({"trial_dir": str(trial), "session_file": str(session_file(trial)),
                                 "source_host": source, "routing": row, "recording_format": recording_format})
                    key = str(trial)
                    if key not in self.trial_server:
                        self.trial_server[key] = min(self.servers, key=lambda s: s.queued_turns)
                    server = self.trial_server[key]
                    server.queued_turns += 1
                    self.queues[server.url].put_nowait(turn)
                    self.done.add(row["id"])
                    added += 1
        return added

    async def label(self, turn, server):
        built = await self.builder.build(turn)
        base = {"id": turn["id"], "trial": turn["trial"], "trial_dir": turn["trial_dir"], "task": turn["task"],
                "class": turn["class"], "turn": turn["turn"], "source_host": turn["source_host"],
                "routing_label": turn["routing"]["label"], "shown_lines": built["shownLines"],
                "recording_format": turn["recording_format"], "reask_format": self.options.family}
        if not built["eligible"]:
            return {"kind": SHORT, **base, "reason": built["reason"]}
        asker = TrimAsker(self, turn, built, server)
        started = time.monotonic()
        try:
            result = await trim_cascade(asker, set(built["trimmed"]))
        except UnusableUncut as problem:
            # Not a fallback: nothing to compare the cuts with; left out with its reason and the asked replies.
            return {"kind": "excluded", **base, "reason": str(problem), "asks": asker.asks}
        return {"kind": "turn", **base, "recording_machine": turn["recording_machine"],
                "session_id": turn["session_id"], "entry_id": turn["entry_id"], "commands": turn["commands"],
                "final_text": turn["final_text"], "recorded": turn["recorded"],
                "total_lines": built["totalLines"], "reference_prompt": asker.reference_prompt,
                "replay_machine": server.machine, "server": server.url, "label": result["label"],
                "checks": result["checks"], "asks": asker.asks, "judges": asker.judge_asker.judges,
                "labelled_s": time.monotonic() - started, "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    async def worker(self, server):
        queue = self.queues[server.url]
        while True:
            turn = await queue.get()
            try:
                row = await self.label(turn, server)
            except (rl.JudgeRefused, rl.PromptMismatch) as problem:
                # Not a fallback: the turn cannot be labelled by the rule (the judge refused it, or the rebuilt uncut
                # request is not the one the routing label was made with); left out with its reason, as in routing.
                print(f"LEFT OUT {turn['id']}: {problem}", flush=True)
                row = {"kind": "excluded", "id": turn["id"], "trial_dir": turn["trial_dir"],
                       "source_host": turn["source_host"], "class": turn["class"],
                       "recording_format": turn["recording_format"], "reask_format": self.options.family,
                       "reason": rl.JUDGE_REFUSED if isinstance(problem, rl.JudgeRefused) else rl.PROMPT_MISMATCH,
                       "detail": str(problem)}
            server.queued_turns -= 1
            self.write(turn["source_host"], row)
            self.written += 1
            if row["kind"] == "turn":
                self.labelled += 1
                if self.labelled % 25 == 0:
                    hours = (time.monotonic() - self.started) / 3600
                    print(f"{time.strftime('%H:%M:%S')} {self.labelled} turns labelled ({self.labelled / hours:.0f}/h), "
                          f"{self.written} rows; " + ", ".join(
                              f"{s.base.rsplit(':', 1)[-1]} q{s.queued_turns} f{s.in_flight} w{s.waiting}"
                              for s in self.servers), flush=True)
            queue.task_done()

    async def run(self):
        try:
            await self._run()
        except BaseException:  # noqa: BLE001 - not swallowed: printed, then the process ends with code 1
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
                print(f"{time.strftime('%H:%M:%S')} discovered {added} new routing-labelled turns; "
                      f"{sum(s.queued_turns for s in self.servers)} queued", flush=True)
                if self.options.exit_when_idle and all(s.queued_turns == 0 for s in self.servers):
                    break
                if self.options.max_turns is not None and self.labelled >= self.options.max_turns:
                    break
                done, _ = await asyncio.wait(tasks, timeout=self.options.poll, return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    task.result()
            for task in tasks:
                task.cancel()
        print(f"done: {self.labelled} turns labelled, {self.written} rows in {time.monotonic() - self.started:.0f} s",
              flush=True)


# ------------------------------------------------------------------------------------------------ stats

LENGTH_BUCKETS = ((41, 100), (101, 200), (201, 500), (501, 2000), (2001, None))


def length_bucket(lines):
    for low, high in LENGTH_BUCKETS:
        if lines >= low and (high is None or lines <= high):
            return f"{low}-{high}" if high is not None else f"{low}+"
    raise ValueError(f"{lines} lines is not a trimmed length")


def trim_file(out_dir, source, tag):
    """The trim label file of one source host: trim-labels-NAME.jsonl, or trim-labels-NAME-TAG.jsonl."""
    return Path(out_dir) / (f"trim-labels-{source}.jsonl" if tag is None else f"trim-labels-{source}-{tag}.jsonl")


def read_rows(paths):
    """The rows of trim label files. A turn written twice (in one file or across files, e.g. by two trim labellers of
    the same source host) is an error."""
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    seen = collections.Counter(row["id"] for row in rows)
    twice = [turn_id for turn_id, count in seen.items() if count > 1]
    if twice:
        raise ValueError(f"{len(twice)} turns appear twice, e.g. {twice[0]}")
    return rows


def formats(turn):
    """(recording format, re-ask format) of a trim "turn" line. Lines written before --family existed carry neither
    field: each of those trim labellers read only its own host's routing labels and asked that host's servers, so they
    were re-asked in the recording format."""
    recording = rl.family(turn["recording_machine"])
    if "recording_format" in turn and turn["recording_format"] != recording:
        raise ValueError(f"{turn['id']}: recording_format {turn['recording_format']} but recording machine "
                         f"{turn['recording_machine']}")
    if "reask_format" in turn:
        return recording, turn["reask_format"]
    if "recording_format" in turn:  # the two fields are written together
        raise ValueError(f"{turn['id']}: a trim line with recording_format but no reask_format")
    return recording, recording


def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def saved_tokens(turn):
    """Prompt tokens the label saves on this turn (the labelled choice's request), 0 for "all"."""
    if turn["label"] == "all":
        return 0
    return next(a["prompt_saved"] for a in turn["asks"] if a["cut"] == turn["label"])


def stats_markdown(rows):
    turns = [r for r in rows if r["kind"] == "turn"]
    short = [r for r in rows if r["kind"] == SHORT]
    excluded = [r for r in rows if r["kind"] == "excluded"]
    out = []
    w = out.append
    w("# Output-trimming labels: how much of the newest tool output does the coding model need?\n")
    w("Generated by `results/imitation/scripts/trim_labels.py stats`. Each sampled routing-labelled Qwen3.8-27B turn "
      "whose newest tool output shows more than 40 lines is asked again at its routing level, all at temperature 0: "
      "uncut, and with that output cut to its last 40 lines, its first 20 and last 20, and its first 40 (then, if none "
      "is good and more than 200 lines show, its last 200), with a note in Jeff-Code's truncation form. Good = the same step "
      "as the uncut temperature-0 action, or the judge's YES (Qwen3.8-Max, thinking off; Step 1 uncut, Step 2 cut). "
      "Label = the first good 40-line cut (last40, first20last20, first40), else last200 when good, else all.\n")
    reasons = collections.Counter(r["reason"] for r in excluded)
    w(f"Routing-labelled turns read: {len(rows)}; newest output at most 40 lines or none: {len(short)} "
      f"({_pct(len(short), len(rows))}); labelled: {len(turns)}; left out: {len(excluded)}"
      + (f" ({', '.join(f'{r}: {n}' for r, n in sorted(reasons.items()))})" if reasons else "") + ".\n")

    def table(title, key, groups_order=None):
        w(f"## Labels by {title}\n")
        w("| group | turns | " + " | ".join(LABELS) + " | prompt tokens saved per turn | shown lines (mean) |")
        w("|---|---:|" + "---:|" * (len(LABELS) + 2))
        groups = collections.defaultdict(list)
        for t in turns:
            groups[key(t)].append(t)
        names = groups_order or sorted(groups)
        for name, group in [("all turns", turns)] + [(n, groups[n]) for n in names if n in groups]:
            n = len(group)
            count = collections.Counter(t["label"] for t in group)
            w(f"| {name} | {n} | " + " | ".join(_pct(count[label], n) for label in LABELS) + " | "
              f"{sum(saved_tokens(t) for t in group) / n:.0f} | {sum(t['shown_lines'] for t in group) / n:.0f} |")
        w("")

    table("shown output length (lines)", lambda t: length_bucket(t["shown_lines"]),
          [length_bucket(low) for low, _ in LENGTH_BUCKETS])
    table("turn class", lambda t: t["class"])
    table("routing level (the level asked)", lambda t: t["routing_label"], ["off", "low", "medium", "xhigh"])
    table("re-ask format (recorded -> re-asked)", lambda t: "{} -> {}".format(*formats(t)))

    w("## How the checks decided\n")
    w("Good = the cut's action was good, whether or not it became the label (all asked cuts are checked).\n")
    w("| cut | asked | good | same step (free match) | judge YES | judge NO | no usable action |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for cut in FORTY + (WIDE,):
        checks = [t["checks"][cut] for t in turns if cut in t["checks"]]
        by = collections.Counter((c["by"], c["good"]) for c in checks)
        w(f"| {cut} | {len(checks)} | {_pct(sum(c['good'] for c in checks), len(checks))} | {by[('intent', True)]} | {by[('judge', True)]} | {by[('judge', False)]} | "
          f"{by[('no usable action', False)]} |")
    w("")
    w("## Tokens\n")
    total_saved = sum(saved_tokens(t) for t in turns)
    total_prompt = sum(t["reference_prompt"] for t in turns)
    w(f"Prompt tokens of the labelled turns' unshortened requests: {total_prompt}; saved by the labels: {total_saved} "
      f"({_pct(total_saved, total_prompt)} of these turns' prompts). The saving carries into every later request of "
      "the session (the shortened output stays in the history); this table counts it once.\n")
    outcomes = collections.Counter((a["cut"], a["outcome"]) for t in turns for a in t["asks"])
    w(f"Request outcomes (cut, outcome: requests): {dict(sorted(outcomes.items()))}.\n")
    return "\n".join(out) + "\n"


def join_rows(rows, stage3_rows):
    """One trimming row per labelled turn that has stage-3 rows: the state of the turn's first decision (its tool-level
    row on page 1: what Jeff sees right after the newest output arrived, before the coding model's turn) plus the trim
    fields. Turns without a stage-3 row are counted."""
    first = {}
    for row in stage3_rows:
        if row["level"] != "tool" or row["page"] != 1:
            continue
        key = (row["session"], row["turn"])
        if key not in first or row["decision"] < first[key]["decision"]:
            first[key] = row
    out = []
    missing = 0
    turns = [r for r in rows if r["kind"] == "turn"]
    for turn in turns:
        stage3 = first.get((turn["session_id"], turn["turn"]))
        if stage3 is None:
            missing += 1
            continue
        out.append({"source": "own", "stage": 3, "task": turn["task"], "session": turn["session_id"],
                    "turn": turn["turn"], "machine": turn["recording_machine"], "source_host": turn["source_host"],
                    "class": turn["class"], "state": stage3["state"],
                    "trim": {"label": turn["label"], "total_lines": turn["total_lines"],
                             "shown_lines": turn["shown_lines"], "routing_label": turn["routing_label"],
                             "checks": turn["checks"]}})
    return out, {"labelled turns": len(turns), "joined": len(out), "no stage-3 row": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run")
    r.add_argument("--routing", action="append", required=True,
                   help="NAME=FILE[,FILE...]: the routing labeller's output files of one source host")
    r.add_argument("--moved", action="append", help="OLD_PREFIX=NEW_PREFIX: a collection folder that moved")
    r.add_argument("--family", required=True, choices=["fp8", "nvfp4"], help="model format of --servers")
    r.add_argument("--allow-cross-format", action="store_true",
                   help="also re-ask turns recorded in the other model format (FP8 on NVFP4 servers or back)")
    r.add_argument("--file-tag", help="write trim-labels-NAME-TAG.jsonl (a second trim labeller of the same host)")
    r.add_argument("--servers", required=True, help="comma-separated chat-completions URLs of that format")
    r.add_argument("--machine", required=True)
    r.add_argument("--judge-url", required=True)
    r.add_argument("--judge-model", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--node", required=True)
    r.add_argument("--builder", required=True, help="trim_requests.ts or its bundle trim_requests.mjs")
    r.add_argument("--sample-rate", type=float, required=True, help="share of routing-labelled turns sampled (0-1]")
    r.add_argument("--per-server", type=int, default=8)
    r.add_argument("--max-waiting", type=int, default=4)
    r.add_argument("--poll", type=float, default=300)
    r.add_argument("--max-turns", type=int, help="stop after this many labelled turns (smoke)")
    r.add_argument("--exit-when-idle", action="store_true")
    j = sub.add_parser("join", help="trimming rows: the trim labels joined onto the stage-3 rows (one per turn)")
    j.add_argument("--labels", nargs="+", required=True)
    j.add_argument("--stage3-rows", nargs="+", required=True,
                   help="stage-3 rows files of all collections (one per host or conversion)")
    j.add_argument("--out", required=True)
    st = sub.add_parser("stats")
    st.add_argument("--labels", nargs="+", required=True)
    st.add_argument("--out", required=True)
    options = parser.parse_args()
    if options.step == "run":
        if not 0 < options.sample_rate <= 1:
            raise ValueError(f"--sample-rate must be in (0, 1], got {options.sample_rate}")
        try:
            asyncio.run(TrimLabeller(options).run())
        except BaseException:  # noqa: BLE001 - not swallowed: printed, then the process ends with code 1
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    elif options.step == "join":
        stage3 = [json.loads(line) for path in options.stage3_rows for line in Path(path).read_text().splitlines()
                  if line.strip()]
        rows, counts = join_rows(read_rows(options.labels), stage3)
        Path(options.out).write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(counts)
    else:
        Path(options.out).write_text(stats_markdown(read_rows(options.labels)))


if __name__ == "__main__":
    main()
