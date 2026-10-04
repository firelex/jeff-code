"""Re-judge routing labels with the "materially better" question (owner, 2026-10-04 ~22:00).

The router's rule flips: a turn runs at thinking off unless the router is confident xhigh is needed. So the label
question changes from "is the off step as good as the recorded xhigh step?" (routing_labels.py) to "is the recorded xhigh
step materially better than the off step?". Every routing label line already holds the thinking-off answer of its turn
(`asks[]` with level "off") and the recorded xhigh action (`commands`, `final_text`); only the judge question is new.

New label of a turn ("off" unless the judge says YES):
  - the old off check was a free match (same step as the recorded action, routing_labels.free_match): "off", no call;
  - the off answer had no usable action (output cap, generation loop, another tool, bad arguments): "xhigh", no call
    (there is no off step to run);
  - the off commands and the recorded commands are near-identical (`near_identical`): "off", no call;
  - else the judge (Qwen3.8-Max through judge_service.py, thinking off, temperature 0) is asked once with Step 1 = the
    recorded xhigh action and Step 2 = the off answer: YES -> "xhigh", NO -> "off".
A judge input that DashScope's content inspection refuses is left out (an "excluded" line with the reason).

The judge sees the same context as routing_labels.py's judge: the task (the session's first user message) and the last
3 commands with the end of their output, from the request pi sent for the turn, rebuilt by routing_requests.mjs.

`calibrate`: on the calibration turns (they hold two extra xhigh samples), the same question with Step 2 = an extra
xhigh sample (the noise floor: how often xhigh is "materially better" than xhigh) and Step 2 = the off answer.

Output files: written next to nothing else; `run` refuses an existing output unless it is its own (resumable: turns
already written are skipped). Every line of the input is written once to the output: trial lines and old excluded
lines unchanged, turn lines with the new `label`, `old_label` and `rejudge` (how the label was decided).

Usage (needs aiohttp and tokenizers; node runs the request builder):
    uv run --with aiohttp==3.12.15 --with tokenizers python rejudge_labels.py run --labels IN.jsonl --out OUT.jsonl \\
        --judge-url URL --judge-model qwen3.8-max --node node --builder routing_requests.mjs --builders 24 \\
        --concurrency 40 [--path-map OLD=NEW ...]
"""

import argparse
import asyncio
import collections
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aiohttp  # noqa: E402

from routing_labels import (  # noqa: E402
    JUDGE_REFUSED,
    Builder,
    JudgeClient,
    JudgeRefused,
    free_match,
    parse_verdict,
    recent_steps,
    step_pieces,
)

# The owner's questions, word for word (2026-10-04 ~22:00, "materially" instead of "clearly"; ~22:10 the stricter one
# without "wastes effort", after the first calibration's noise floor of 20%).
QUESTIONS = {
    "materially-better-20261004": (
        "Step 1 and Step 2 are two possible next steps for the same coding task at the same moment. Is Step 1 "
        "materially better than Step 2? Answer YES only if Step 2 is a mistake, would take the task in a wrong "
        "direction, or wastes effort. A step that does less but is correct progress toward the task is not worse. "
        "Answer YES or NO, then give one short reason."
    ),
    "materially-better-strict-20261004": (
        "Step 1 and Step 2 are two possible next steps for the same coding task at the same moment. Is Step 1 "
        "materially better than Step 2? Answer YES only if Step 2 is a mistake or would take the task in a wrong "
        "direction. A step that does less, or takes a different but reasonable route, is not worse. Answer YES or NO, "
        "then give one short reason."
    ),
}
# The setting, as routing_labels.JUDGE_SYSTEM describes it (its first paragraph), and the answer format.
SYSTEM = (
    "You review the work of an AI assistant that solves tasks on a Linux computer by running shell commands, one step "
    "at a time. You will see the task, the most recent steps the assistant took (each command with the end of its "
    "output), and two different candidates for the assistant's next step, Step 1 and Step 2. A step may contain "
    "several commands; they run one after another.\n\n"
    "Answer with YES or NO on the first line. On the second line, give one short reason."
)
# Near-identical commands (owner): Dice coefficient of the character bigrams at least 0.9, only when the shorter command
# text is at least 0.8 times the longer and both are at least 20 characters.
DICE_MIN = 0.9
LENGTH_RATIO_MIN = 0.8
MIN_CHARS = 20


def shown_step(pieces, step_chars):
    """A step's pieces for the judge, each cut at `step_chars` characters. A cut piece says so in plain words, so the
    judge does not read the cut as a broken command (the first calibrations cut at 2,500 without a note; 8,000 with
    the note since 2026-10-04 ~22:15)."""
    shown = []
    for piece in pieces:
        if len(piece) > step_chars:
            piece = (piece[:step_chars] + f"\n[This command was shortened for display here: {len(piece) - step_chars} "
                     "more characters were left out. The real command is complete; do not count the shortening as "
                     "a mistake.]")
        shown.append(piece)
    return "\n\n".join(shown)


def judge_messages(task, steps, reference, alternative, question, step_chars):
    """System and user message: the same context blocks as routing_labels.judge_messages (task, last steps, Step 1,
    Step 2; the steps cut at `step_chars`, see shown_step), then the owner's question."""
    shown = "\n\n".join(
        f"Command:\n{cmd[:1500]}\nEnd of output:\n{out[-600:] or '(no output)'}" for cmd, out in steps
    ) or "(none yet)"

    user = (
        f"TASK:\n{task[:3000]}\n\nMOST RECENT STEPS (oldest first):\n{shown}\n\n"
        f"STEP 1:\n{shown_step(reference, step_chars)}\n\nSTEP 2:\n{shown_step(alternative, step_chars)}\n\n{QUESTIONS[question]}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def bigrams(text):
    return collections.Counter(text[i:i + 2] for i in range(len(text) - 1))


def dice(a, b):
    """Dice coefficient of the character-bigram multisets of two strings: 2 |A & B| / (|A| + |B|). Counting repeated
    bigrams keeps a short string from scoring high against a long one that merely contains it."""
    x, y = bigrams(a), bigrams(b)
    total = sum(x.values()) + sum(y.values())
    if total == 0:
        raise ValueError("dice of two strings shorter than 2 characters")
    return 2 * sum((x & y).values()) / total


def near_identical(commands_a, commands_b):
    """(near-identical?, dice or None). Only command steps (not final answers) whose texts (commands joined by newlines)
    are both at least MIN_CHARS long and whose shorter text is at least LENGTH_RATIO_MIN of the longer are compared."""
    if not commands_a or not commands_b:
        return False, None
    a, b = "\n".join(commands_a), "\n".join(commands_b)
    short, long_ = sorted((len(a), len(b)))
    if short < MIN_CHARS or short < LENGTH_RATIO_MIN * long_:
        return False, None
    value = dice(a, b)
    return value >= DICE_MIN, value


def reply_of(turn, level, sample):
    found = [a for a in turn["asks"] if a["level"] == level and a["sample"] == sample]
    if len(found) != 1:
        raise ValueError(f"{turn['id']}: {len(found)} asks at {level}-{sample}")
    return found[0]


def decide_without_judge(turn):
    """How the new label is decided without a judge call: (label, how, dice) or (None, None, dice) when the judge must
    be asked."""
    off = turn["checks"]["off"]
    if off["by"] == "intent":
        if not off["good"]:
            raise ValueError(f"{turn['id']}: an intent check that is not good")
        return "off", "free match (same step)", None
    if off["by"] == "no usable action":
        return "xhigh", "off has no usable action", None
    if off["by"] != "judge":
        raise ValueError(f"{turn['id']}: unknown off check {off}")
    answer = reply_of(turn, "off", 1)
    if answer["commands"] is None:
        raise ValueError(f"{turn['id']}: an off answer without commands was judged")
    same, value = near_identical(answer["commands"], turn["commands"])
    if same:
        return "off", "near-identical commands", value
    return None, None, value


class Rejudger:
    def __init__(self, options):
        self.options = options
        self.question = options.question
        self.step_chars = options.step_chars
        self.maps = [tuple(item.split("=", 1)) for item in options.path_map]
        self.judge_client = JudgeClient(options.judge_url, options.judge_model)
        self.builders = [Builder(options.node, options.builder) for _ in range(options.builders)]
        self.slots = asyncio.Semaphore(options.concurrency)
        self.calls = 0
        self.call_seconds = 0.0
        self.started = time.monotonic()

    def session_file(self, trial_dir):
        for old, new in self.maps:
            if trial_dir.startswith(old):
                trial_dir = new + trial_dir[len(old):]
                break
        sessions = sorted((Path(trial_dir) / "agent" / "pi" / "sessions").glob("*.jsonl"))
        if len(sessions) != 1:
            raise ValueError(f"{trial_dir}: {len(sessions)} session files (is a --path-map missing?)")
        return str(sessions[0])

    async def context(self, turn):
        job = {"id": turn["id"], "session_file": self.session_file(turn["trial_dir"]), "entry_id": turn["entry_id"]}
        builder = self.builders[zlib.crc32(turn["trial"].encode()) % len(self.builders)]
        built = await builder.build(job)
        return built["task"], recent_steps(built["body"]["messages"])

    async def judge(self, turn, task, steps, reference, alternative, key):
        messages = judge_messages(task, steps, step_pieces(reference["commands"], reference["final_text"]),
                                  step_pieces(alternative["commands"], alternative["final_text"]), self.question,
                                  self.step_chars)
        async with self.slots:
            reply = await self.judge_client.ask(messages, f"{turn['id']} {key}")
        self.calls += 1
        self.call_seconds += reply["seconds"]
        verdict, reason = parse_verdict(reply["content"])
        return {"key": key, "question": self.question, "step_chars": self.step_chars, "model": f"{reply['model']} ({reply['backend']}, thinking off)",
                "verdict": verdict, "reason": reason, "reply": reply["content"],
                "prompt_tokens": reply["usage"].get("prompt_tokens"),
                "completion_tokens": reply["usage"].get("completion_tokens"),
                "finish_reason": reply["finish_reason"], "seconds": reply["seconds"]}

    def progress(self, done, total):
        hours = (time.monotonic() - self.started) / 3600
        print(f"{time.strftime('%H:%M:%S')} {done}/{total} turns; {self.calls} judge calls "
              f"({self.calls / hours:.0f}/h, mean {self.call_seconds / max(self.calls, 1):.1f} s)", flush=True)

    async def pool(self, items, handle, workers):
        queue = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        async def worker():
            while not queue.empty():
                await handle(queue.get_nowait())

        await asyncio.gather(*(worker() for _ in range(workers)))

    async def close(self):
        for builder in self.builders:
            await builder.close()


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


async def run(options):
    rejudger = Rejudger(options)
    rows = read_lines(options.labels)
    out = Path(options.out)
    if out.resolve() == Path(options.labels).resolve():
        raise ValueError("--out must not be the input file")
    done = set()
    if out.exists():
        for row in read_lines(out):
            done.add(row.get("id") or row["trial_dir"])
    sink = out.open("a")

    def write(row):
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    todo = []
    counts = collections.Counter()
    for row in rows:
        key = row.get("id") or row["trial_dir"]
        if key in done:
            continue
        if row["kind"] != "turn":
            write(row)
            continue
        label, how, value = decide_without_judge(row)
        if label is not None:
            counts[how] += 1
            write({**row, "label": label, "old_label": row["label"],
                   "rejudge": {"question": options.question, "by": how, "dice": value}})
            continue
        todo.append((row, value))
    todo.sort(key=lambda item: item[0]["trial"])
    print(f"{len(rows)} lines; {len(done)} already written; decided without the judge now: {dict(counts)}; "
          f"{len(todo)} turns to judge", flush=True)
    finished = 0

    async def handle(item):
        nonlocal finished
        turn, value = item
        task, steps = await rejudger.context(turn)
        off = reply_of(turn, "off", 1)
        recorded = {"commands": turn["commands"], "final_text": turn["final_text"]}
        try:
            verdict = await rejudger.judge(turn, task, steps, recorded, off, "recorded vs off")
        except JudgeRefused as problem:
            # Not a fallback: the turn cannot be labelled by the owner's rule; it is left out with its reason.
            print(f"LEFT OUT {turn['id']}: {problem}", flush=True)
            write({"kind": "excluded", "id": turn["id"], "trial_dir": turn["trial_dir"],
                   "source_host": turn["source_host"], "class": turn["class"],
                   "recording_format": turn.get("recording_format"), "reask_format": turn.get("reask_format"),
                   "reason": f"{JUDGE_REFUSED} ({options.question})", "detail": str(problem), "old_label": turn["label"]})
        else:
            write({**turn, "label": "xhigh" if verdict["verdict"] else "off", "old_label": turn["label"],
                   "rejudge": {"question": options.question, "by": "judge", "dice": value, "judge": verdict}})
        finished += 1
        if finished % 200 == 0:
            rejudger.progress(finished, len(todo))

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=900),
                                     connector=aiohttp.TCPConnector(limit=0)) as http:
        print(f"judge: {await rejudger.judge_client.check(http)}", flush=True)
        await rejudger.pool(todo, handle, options.concurrency + options.builders)
    await rejudger.close()
    rejudger.progress(finished, len(todo))
    sink.close()
    check(options.labels, out)


def check(labels, out):
    """Every input line is in the output once (by turn id or trial folder)."""
    keys_in = [row.get("id") or row["trial_dir"] for row in read_lines(labels)]
    keys_out = collections.Counter(row.get("id") or row["trial_dir"] for row in read_lines(out))
    repeated = [k for k, n in keys_out.items() if n > 1]
    missing = set(keys_in) - set(keys_out)
    extra = set(keys_out) - set(keys_in)
    if repeated or missing or extra:
        raise ValueError(f"{out}: {len(repeated)} repeated, {len(missing)} missing, {len(extra)} unknown lines")
    print(f"checked: {out} holds each of the {len(keys_in)} input lines once", flush=True)


async def calibrate(options):
    rejudger = Rejudger(options)
    turns = [row for path in options.labels for row in read_lines(path)
             if row["kind"] == "turn" and row["calibration"]]
    turns.sort(key=lambda t: t["trial"])
    out = Path(options.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    sink = out.open("w")
    skipped = collections.Counter()
    finished = 0

    async def handle(turn):
        nonlocal finished
        recorded = {"commands": turn["commands"], "final_text": turn["final_text"]}
        pairs = [(f"recorded vs {name}", reply_of(turn, level, sample))
                 for name, level, sample in (("off", "off", 1), ("xhigh-2", "xhigh", 2), ("xhigh-3", "xhigh", 3))]
        task, steps = await rejudger.context(turn)
        for key, answer in pairs:
            if answer["commands"] is None:
                skipped[f"{key}: no usable action"] += 1
                continue
            same, value = near_identical(answer["commands"], turn["commands"])
            row = {"id": turn["id"], "source_host": turn["source_host"], "class": turn["class"], "key": key,
                   "old_label": turn["label"], "dice": value, "near_identical": same}
            try:
                row["judge"] = await rejudger.judge(turn, task, steps, recorded, answer, key)
            except JudgeRefused as problem:
                skipped[f"{key}: judge refused"] += 1
                row["refused"] = str(problem)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        finished += 1
        if finished % 100 == 0:
            rejudger.progress(finished, len(turns))

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=900),
                                     connector=aiohttp.TCPConnector(limit=0)) as http:
        print(f"judge: {await rejudger.judge_client.check(http)}", flush=True)
        print(f"{len(turns)} calibration turns", flush=True)
        await rejudger.pool(turns, handle, options.concurrency + options.builders)
    await rejudger.close()
    sink.close()
    rejudger.progress(finished, len(turns))
    print(f"skipped pairs: {dict(skipped)}", flush=True)


def calibration_stats(label_paths, calibration_path):
    """YES shares of the calibration pairs: raw (every judged pair) and under the full run's rule (a free match or
    near-identical commands count as NO without the judge; refused pairs left out)."""
    turns = {row["id"]: row for path in label_paths for row in read_lines(path) if row["kind"] == "turn"}
    raw = collections.defaultdict(collections.Counter)
    rule = collections.defaultdict(collections.Counter)
    for row in read_lines(calibration_path):
        if "judge" not in row:
            continue
        kind = "xhigh vs xhigh" if "xhigh" in row["key"] else "xhigh vs off"
        turn = turns[row["id"]]
        level, sample = ("off", 1) if kind == "xhigh vs off" else ("xhigh", int(row["key"][-1]))
        answer = reply_of(turn, level, sample)
        yes = row["judge"]["verdict"]
        raw[kind]["pairs"] += 1
        raw[kind]["yes"] += yes
        if free_match(answer["commands"], turn["commands"]):
            how = "free match"
        elif row["near_identical"]:
            how = "near-identical"
        else:
            how = "judge"
        rule[kind]["pairs"] += 1
        rule[kind][how] += 1
        rule[kind]["yes"] += yes and how == "judge"
        rule[kind][f"judge YES on {how}"] += yes
    lines = ["| pair | pairs | judge YES (raw) | free match | near-identical | YES under the run's rule |",
             "|---|---:|---:|---:|---:|---:|"]
    for kind in ("xhigh vs xhigh", "xhigh vs off"):
        r, u = raw[kind], rule[kind]
        lines.append(f"| {kind} | {r['pairs']} | {r['yes']} ({100 * r['yes'] / r['pairs']:.1f}%) | {u['free match']} "
                     f"(judge YES on them {u['judge YES on free match']}) | {u['near-identical']} (judge YES on them "
                     f"{u['judge YES on near-identical']}) | {u['yes']} ({100 * u['yes'] / u['pairs']:.1f}%) |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    cs = sub.add_parser("calibration-stats")
    cs.add_argument("--labels", required=True, nargs="+")
    cs.add_argument("--calibration", required=True)
    for name in ("run", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--labels", required=True, nargs="+" if name == "calibrate" else None)
        p.add_argument("--out", required=True)
        p.add_argument("--question", required=True, choices=sorted(QUESTIONS), help="which of the owner's questions")
        p.add_argument("--step-chars", type=int, required=True,
                       help="each command of a step is cut at this many characters for the judge, with a note")
        p.add_argument("--judge-url", required=True)
        p.add_argument("--judge-model", required=True)
        p.add_argument("--node", required=True)
        p.add_argument("--builder", required=True, help="routing_requests.mjs")
        p.add_argument("--builders", type=int, required=True, help="request builder processes")
        p.add_argument("--concurrency", type=int, required=True, help="judge calls at once")
        p.add_argument("--path-map", action="append", default=[],
                       help="OLD=NEW: a trial folder starting with OLD now lies under NEW (moved collections)")
    options = parser.parse_args()
    if options.step == "calibration-stats":
        print(calibration_stats(options.labels, options.calibration))
        return
    try:
        asyncio.run(run(options) if options.step == "run" else calibrate(options))
    except BaseException:  # noqa: BLE001 - not swallowed: printed, then the process ends with code 1
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
