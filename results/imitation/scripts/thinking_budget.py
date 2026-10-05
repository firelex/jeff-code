"""Long thinking in the xhigh collection: how much Qwen3.8-27B's generation time goes to very long thinking, and what a
thinking budget could save.

`offline` reads recorded Jeff-Code sessions and JeffFirst traces (read-only, no model calls) and writes the Part A tables and a
per-turn JSONL (input of the re-ask test, `sample`).

Data: trials of the xhigh collection (record mode, thinking router fixed:xhigh, driver tarball of build 8db5381f3 unless
--build says otherwise). A trial is a folder <task>__<id>/ with config.json, result.json (a trial without it is still
running or was stopped and is left out), agent/jeff-first-trace.jsonl and agent/pi/sessions/<one file>.jsonl.

Definitions:
- A Qwen request is one `qwen_request` trace line (one per request, written by the thinking control): attempt 1, or
  attempt 2 after the runaway cut-off (a looping reply is aborted and the turn asked again once at xhigh). Its
  generation time is `timings_ms.model` (request start to reply end, including prefill and queueing).
- Thinking tokens: the server's count (`usage.thinking_tokens`). For an aborted request (runaway cut-off) the server
  sends no count; its thinking tokens are estimated from `thinking_chars` with the characters-per-token ratio of the
  completed requests.
- Turn classes: completed (stop reason toolUse or stop), output limit (stop reason length with 32,768 output tokens:
  the reply reached the output limit), context limit (stop reason length with fewer output tokens: prompt plus reply
  reached the context window; Jeff-Code compacts the context and goes on), runaway cut (outcome discarded or turn_ended with guard runaway), loop guard (guard loop;
  it acts only at thinking off/low, so never at xhigh), failed (stop reason error/aborted without a guard).
- Time a budget B saves on a completed turn with T > B thinking tokens: its generation time x (T - B) / output tokens
  (the decode share of the turn; prefill and queueing are put on all tokens alike, as ceiling.py does).

Usage:
    python thinking_budget.py offline --source b200=DIR --source casdgx01=DIR ... --out-md OUT.md --out-turns TURNS.jsonl
"""

import argparse
import collections
import json
import os
import statistics
from pathlib import Path

BUDGETS = (4096, 8192, 16384)
OUTPUT_LIMIT = 32768
ROUTER = "fixed:xhigh"
THINK_BINS = (0, 1000, 2000, 4096, 8192, 16384, 24576, OUTPUT_LIMIT + 1)


def dataset_of(task_id):
    """Dataset of a trace task id: hub tasks are '<org>/<dataset>:<task>', Terminal-Bench 2.0 tasks a bare name."""
    if "/" not in task_id:
        return "terminal-bench-2"
    return task_id.split("/", 1)[0]


def trial_folders(root):
    found = []
    for folder, subfolders, files in os.walk(root):
        # Job folders of hub tasks may contain "__" too (swe-rebench repository names); a trial's config names its agent.
        if "__" in Path(folder).name and "config.json" in files \
                and "agent" in json.loads((Path(folder) / "config.json").read_text()):
            subfolders.clear()
            found.append(Path(folder))
    return sorted(found)


def json_lines(path):
    """Entries of a JSONL file; a last line cut by a stopped writer is dropped (counted), any other bad line raises."""
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    rows = []
    for number, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines) - 1:
                continue
            raise ValueError(f"{path}: line {number + 1} is not JSON")
    return rows


def read_trial(trial, host, build):
    """One finished trial: its Qwen requests (trace) joined to the session's assistant replies, and its outcome. None
    when the trial is not of the wanted collection (mode, router, build) - counted by the caller."""
    config = json.loads((trial / "config.json").read_text())
    env = config["agent"]["env"]
    tarball = Path(config["agent"]["kwargs"]["tarball"]).name
    if env.get("JEFF_FIRST_MODE") != "record" or env.get("JEFF_FIRST_THINKING_ROUTER") != ROUTER or build not in tarball:
        return None, f"other collection ({env.get('JEFF_FIRST_MODE')}, {env.get('JEFF_FIRST_THINKING_ROUTER')}, {tarball})"
    result_path = trial / "result.json"
    if not result_path.exists():
        return None, "unfinished (no result.json)"
    result = json.loads(result_path.read_text())
    trace_path = trial / "agent" / "jeff-first-trace.jsonl"
    sessions = sorted((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if not trace_path.exists() or not sessions:
        return None, "no trace or session (the agent failed before its first turn)"
    if len(sessions) != 1:
        raise ValueError(f"{trial}: {len(sessions)} session files")
    requests = [line for line in json_lines(trace_path) if line.get("kind") == "qwen_request"]
    if not requests:
        return None, "no Qwen request in the trace"
    entries = json_lines(sessions[0])
    assistants = [e for e in entries if e["type"] == "message" and e["message"]["role"] == "assistant"]
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    exception = (result.get("exception_info") or {}).get("exception_type")
    task = requests[0]["task_id"]
    trial_row = {"trial": trial.name, "trial_dir": str(trial), "host": host, "task": task, "dataset": dataset_of(task),
                 "machine": env["JEFF_FIRST_DRIVER_BUILD"], "reward": rewards.get("reward"), "exception": exception,
                 "session_file": str(sessions[0]), "assistants": len(assistants)}
    # Pair every request that produced a session reply (kept, or turn_ended = an error reply) with its reply, in order.
    replied = [r for r in requests if r["outcome"] in ("kept", "turn_ended")]
    if len(replied) == len(assistants) + 1 and (exception is not None):
        replied = replied[:-1]  # a reply the crashed or stopped session never saved
    if len(replied) != len(assistants):
        raise ValueError(f"{trial}: {len(assistants)} assistant replies but {len(replied)} kept qwen_request lines")
    entry_of = {id(r): e for r, e in zip(replied, assistants)}
    rows = []
    for request in requests:
        entry = entry_of.get(id(request))
        guard = request.get("guard")
        usage = request["usage"]
        if guard is not None and guard.get("trigger") == "loop":
            klass = "loop guard"
        elif guard is not None and guard.get("trigger") == "runaway":
            klass = "runaway cut"
        elif request["stop_reason"] == "length" and usage.get("output", 0) >= OUTPUT_LIMIT:
            klass = "output limit"
        elif request["stop_reason"] == "length":
            # The prompt plus the reply reached the context window before the output limit; Jeff-Code compacts and goes on.
            klass = "context limit"
        elif request["stop_reason"] in ("toolUse", "stop"):
            klass = "completed"
        else:
            klass = "failed"
        row = {"trial_dir": str(trial), "host": host, "task": task, "dataset": trial_row["dataset"],
               "machine": trial_row["machine"], "turn": request["turn"], "attempt": request["attempt"],
               "outcome": request["outcome"], "class": klass, "stop_reason": request["stop_reason"],
               "guard": guard, "thinking_chars": request["thinking_chars"], "thinking_tokens": usage.get("thinking_tokens"),
               "output_tokens": usage.get("output"), "input_tokens": usage.get("input"),
               "cache_read": usage.get("cache_read"), "model_ms": request["timings_ms"]["model"],
               "entry_id": entry["id"] if entry else None}
        if entry is not None:
            message = entry["message"]
            if message["stopReason"] != request["stop_reason"] and request["outcome"] == "kept":
                raise ValueError(f"{trial}: turn {request['turn']}: trace stop {request['stop_reason']}, session "
                                 f"{message['stopReason']}")
            calls = [p for p in message["content"] if p["type"] == "toolCall"]
            row["n_calls"] = len(calls)
            row["commands"] = [c["arguments"].get("command") if isinstance(c["arguments"], dict) else None for c in calls]
        rows.append(row)
    last_turn = max(r["turn"] for r in rows)
    for row in rows:
        row["last_turn"] = row["turn"] == last_turn
    trial_row["turns"] = len({r["turn"] for r in rows})
    return (trial_row, rows), None


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def hrs(ms):
    return f"{ms / 3_600_000:.1f}"


def fill_estimates(rows):
    """Thinking tokens of aborted requests from their characters (ratio of the completed requests)."""
    known = [r for r in rows if r["thinking_tokens"] and r["thinking_chars"]]
    ratio = sum(r["thinking_chars"] for r in known) / sum(r["thinking_tokens"] for r in known)
    for row in rows:
        row["thinking_estimated"] = row["thinking_tokens"] is None or (row["class"] == "runaway cut")
        if row["thinking_estimated"]:
            row["thinking_tokens"] = round(row["thinking_chars"] / ratio)
    return ratio


def saved_ms(row, budget):
    """Generation time a thinking budget would not spend on a completed request."""
    t = row["thinking_tokens"]
    if t <= budget or not row["output_tokens"]:
        return 0.0
    return row["model_ms"] * (t - budget) / row["output_tokens"]


def bin_label(lo, hi):
    def k(x):
        return f"{x / 1000:.0f}k" if x >= 1000 else str(x)
    return f"{k(lo)}-{k(hi)}" if hi <= OUTPUT_LIMIT else f">= {k(lo)}"


def offline_markdown(trials, rows, skipped, ratio, build):
    out = []
    w = out.append
    total_ms = sum(r["model_ms"] for r in rows)
    n_turns = len({(r["trial_dir"], r["turn"]) for r in rows})
    w(f"Data: {len(trials)} finished trials of the xhigh collection (record mode, router fixed:xhigh, build {build}), "
      f"{n_turns:,} Qwen turns, {len(rows):,} Qwen requests, {hrs(total_ms)} h of Qwen generation time. Hosts: "
      + ", ".join(f"{h} {n}" for h, n in sorted(collections.Counter(t['host'] for t in trials).items())) + ". "
      f"Trials left out: " + ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items())) + ".\n")
    w(f"Characters per thinking token (completed requests): {ratio:.2f}.\n")

    w("### 1. Turns that hit the output limit, the runaway cut-off or the loop guard\n")
    w("| class | requests | share of requests | generation h | share of generation time | mean s | mean thinking tokens |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for klass in ("completed", "output limit", "context limit", "runaway cut", "loop guard", "failed"):
        group = [r for r in rows if r["class"] == klass]
        ms = sum(r["model_ms"] for r in group)
        w(f"| {klass} | {len(group):,} | {pct(len(group), len(rows))} | {hrs(ms)} | {pct(ms, total_ms)} | "
          f"{(ms / len(group) / 1000) if group else 0:.0f} | "
          f"{(statistics.mean(r['thinking_tokens'] for r in group)) if group else 0:,.0f} |")
    w("")
    w("By dataset:\n")
    w("| dataset | trials | requests | gen h | output limit (n) | limit share of requests | limit share of time | "
      "runaway (n) | runaway share of time |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == ds]
        ms = sum(r["model_ms"] for r in group)
        lim = [r for r in group if r["class"] == "output limit"]
        run = [r for r in group if r["class"] == "runaway cut"]
        w(f"| {ds} | {sum(t['dataset'] == ds for t in trials)} | {len(group):,} | {hrs(ms)} | {len(lim)} | "
          f"{pct(len(lim), len(group))} | {pct(sum(r['model_ms'] for r in lim), ms)} | {len(run)} | "
          f"{pct(sum(r['model_ms'] for r in run), ms)} |")
    w("")
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    hit_tasks = sorted((t for t, g in by_task.items() if any(r["class"] in ("output limit", "runaway cut") for r in g)),
                       key=lambda t: -sum(r["model_ms"] for r in by_task[t] if r["class"] in ("output limit", "runaway cut")))
    w(f"Tasks with at least one limit hit or runaway cut: {len(hit_tasks)} of {len(by_task)}. Top 25 by time in those "
      "requests:\n")
    w("| task | trials | requests | limit hits | runaway cuts | their gen min | share of the task's gen time |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for t in hit_tasks[:25]:
        g = by_task[t]
        bad = [r for r in g if r["class"] in ("output limit", "runaway cut")]
        w(f"| {t} | {len({r['trial_dir'] for r in g})} | {len(g)} | {sum(r['class'] == 'output limit' for r in g)} | "
          f"{sum(r['class'] == 'runaway cut' for r in g)} | {sum(r['model_ms'] for r in bad) / 60000:.0f} | "
          f"{pct(sum(r['model_ms'] for r in bad), sum(r['model_ms'] for r in g))} |")
    w("")

    # What happens after a limit hit.
    trial_rows = collections.defaultdict(list)
    for r in rows:
        trial_rows[r["trial_dir"]].append(r)
    limit = [r for r in rows if r["class"] == "output limit"]
    limit_last = sum(r["last_turn"] for r in limit)
    with_calls = sum(1 for r in limit if r.get("n_calls"))
    w(f"After an output-limit hit: {limit_last} of {len(limit)} limit hits are the session's last Qwen turn (the "
      f"session ends there); {with_calls} of them carried a tool call (Jeff-Code fails a cut reply's tool calls without running "
      "them). Rewards of the trials whose session ended on a limit hit: "
      + str(dict(collections.Counter(next(t["reward"] for t in trials if t["trial_dir"] == r["trial_dir"])
                                     for r in limit if r["last_turn"]))) + ".\n")
    ended = {r["trial_dir"] for r in limit if r["last_turn"]}
    ended_ms = sum(r["model_ms"] for t in ended for r in trial_rows[t])
    w(f"Episode cost: the {len(limit)} limit-hit replies took {hrs(sum(r['model_ms'] for r in limit))} h "
      f"(mean {statistics.mean(r['model_ms'] for r in limit) / 1000:.0f} s); there is no retry, so the cost is the cut "
      f"reply itself plus the lost trial: the {len(ended)} trials that ended on a limit hit used {hrs(ended_ms)} h of "
      f"Qwen generation in total ({pct(ended_ms, total_ms)} of all), and all but those with reward None scored 0.\n")
    runaway = [r for r in rows if r["class"] == "runaway cut"]
    reasked = collections.Counter()
    for r in runaway:
        nxt = [x for x in trial_rows[r["trial_dir"]] if x["turn"] == r["turn"] and x["attempt"] == r["attempt"] + 1]
        reasked[(r["outcome"], nxt[0]["class"] if nxt else "none")] += 1
    w(f"After a runaway cut: (outcome, class of the re-ask) {dict(reasked)}. The re-ask is the same request at xhigh; "
      "a second runaway ends the session's turn with an error.\n")

    w("### 2. Thinking length of completed turns\n")
    done = [r for r in rows if r["class"] == "completed"]
    think = sorted(r["thinking_tokens"] for r in done)
    q = lambda p: think[min(len(think) - 1, int(p * len(think)))]  # noqa: E731
    done_ms = sum(r["model_ms"] for r in done)
    w(f"Completed requests: {len(done):,}, {hrs(done_ms)} h. Thinking tokens: median {q(0.5):,}, p90 {q(0.9):,}, p99 "
      f"{q(0.99):,}, max {think[-1]:,}; mean {statistics.mean(think):,.0f}.\n")
    w("| thinking tokens | requests | share of requests | gen h | share of all generation time |")
    w("|---|---:|---:|---:|---:|")
    for lo, hi in zip(THINK_BINS, THINK_BINS[1:]):
        g = [r for r in done if lo <= r["thinking_tokens"] < hi]
        ms = sum(r["model_ms"] for r in g)
        w(f"| {bin_label(lo, hi)} | {len(g):,} | {pct(len(g), len(done))} | {hrs(ms)} | {pct(ms, total_ms)} |")
    w("")
    w("Upper bound of a fixed thinking budget: generation time spent in thinking beyond B tokens (completed requests), "
      "plus, for the limit hits, everything beyond B (their whole reply is thinking). Shares are of ALL Qwen generation "
      "time. This assumes the answer after the cut is as good and as long as the recorded one (Part B tests that).\n")
    w("| budget B | completed turns over B | share of completed turns | time beyond B (completed) | + limit hits beyond B "
      "| + runaway requests (all) | total upper bound |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    lim_ms = lambda b: sum(r["model_ms"] * (OUTPUT_LIMIT - b) / OUTPUT_LIMIT for r in limit)  # noqa: E731
    run_ms = sum(r["model_ms"] for r in runaway)
    for b in BUDGETS:
        over = [r for r in done if r["thinking_tokens"] > b]
        s = sum(saved_ms(r, b) for r in done)
        w(f"| {b // 1024}k | {len(over):,} | {pct(len(over), len(done))} | {pct(s, total_ms)} | {pct(lim_ms(b), total_ms)} "
          f"| {pct(run_ms, total_ms)} | {pct(s + lim_ms(b) + run_ms, total_ms)} |")
    w("")
    w("By dataset (share of that dataset's generation time; completed beyond B + limit hits beyond B):\n")
    w("| dataset | gen h | " + " | ".join(f"B={b // 1024}k" for b in BUDGETS) + " |")
    w("|---|---:|" + "---:|" * len(BUDGETS))
    for ds in sorted({r["dataset"] for r in rows}):
        g = [r for r in rows if r["dataset"] == ds]
        ms = sum(r["model_ms"] for r in g)
        cells = []
        for b in BUDGETS:
            s = sum(saved_ms(r, b) for r in g if r["class"] == "completed") + sum(
                r["model_ms"] * (OUTPUT_LIMIT - b) / OUTPUT_LIMIT for r in g if r["class"] == "output limit")
            cells.append(pct(s, ms))
        w(f"| {ds} | {hrs(ms)} | " + " | ".join(cells) + " |")
    w("")

    w("### 3. Pass rate by limit hits and by longest thinking in the trial\n")
    scored = [t for t in trials if t["reward"] is not None]
    w(f"Trials with a verifier reward: {len(scored)} of {len(trials)} (others ended with an exception before "
      "verification). Pass = reward >= 1.\n")
    w("| group | trials | pass rate | mean reward | mean Qwen gen min per trial |")
    w("|---|---:|---:|---:|---:|")

    def line(name, group):
        if not group:
            w(f"| {name} | 0 | - | - | - |")
            return
        gen = [sum(r["model_ms"] for r in trial_rows[t["trial_dir"]]) / 60000 for t in group]
        w(f"| {name} | {len(group)} | {pct(sum(t['reward'] >= 1 for t in group), len(group))} | "
          f"{statistics.mean(t['reward'] for t in group):.2f} | {statistics.mean(gen):.1f} |")

    def has(t, klass):
        return any(r["class"] == klass for r in trial_rows[t["trial_dir"]])

    line("all", scored)
    line("no limit hit, no runaway cut", [t for t in scored if not has(t, "output limit") and not has(t, "runaway cut")])
    line(">= 1 output-limit hit", [t for t in scored if has(t, "output limit")])
    line(">= 1 runaway cut", [t for t in scored if has(t, "runaway cut")])
    for t in scored:
        t["max_think"] = max(r["thinking_tokens"] for r in trial_rows[t["trial_dir"]] if r["class"] == "completed") \
            if any(r["class"] == "completed" for r in trial_rows[t["trial_dir"]]) else 0
    for lo, hi in zip(THINK_BINS, THINK_BINS[1:]):
        line(f"no limit hit, longest completed thinking {bin_label(lo, hi)}",
             [t for t in scored if not has(t, "output limit") and lo <= t["max_think"] < hi])
    w("")
    w("Same, within dataset (pass rate, trials):\n")
    w("| dataset | no limit hit | >= 1 limit hit | longest thinking < 8k | longest 8k-16k | longest >= 16k |")
    w("|---|---:|---:|---:|---:|---:|")
    for ds in sorted({t["dataset"] for t in scored}):
        g = [t for t in scored if t["dataset"] == ds]

        def cell(group):
            return f"{pct(sum(t['reward'] >= 1 for t in group), len(group))} ({len(group)})"
        nl = [t for t in g if not has(t, "output limit")]
        w(f"| {ds} | {cell(nl)} | {cell([t for t in g if has(t, 'output limit')])} | "
          f"{cell([t for t in nl if t['max_think'] < 8192])} | {cell([t for t in nl if 8192 <= t['max_think'] < 16384])} | "
          f"{cell([t for t in nl if t['max_think'] >= 16384])} |")
    w("")
    return "\n".join(out) + "\n"


def offline(options):
    trials, rows = [], []
    skipped = collections.Counter()
    for item in options.source:
        host, root = item.split("=", 1)
        if not Path(root).is_dir():
            raise ValueError(f"{root} is not a folder")
        for folder in trial_folders(Path(root)):
            got, why = read_trial(folder, host, options.build)
            if got is None:
                skipped[why.split(" (")[0] if why.startswith("other") else why] += 1
                continue
            trials.append(got[0])
            rows.extend(got[1])
    ratio = fill_estimates(rows)
    Path(options.out_md).write_text(offline_markdown(trials, rows, skipped, ratio, options.build))
    with open(options.out_turns, "w") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    with open(options.out_trials, "w") as sink:
        for t in trials:
            sink.write(json.dumps(t) + "\n")
    print(f"{len(trials)} trials, {len(rows)} requests; skipped {dict(skipped)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    o = sub.add_parser("offline")
    o.add_argument("--source", action="append", required=True, help="HOST=COLLECTION_DIR (repeatable)")
    o.add_argument("--build", default="8db5381f3", help="driver tarball build the trials must use")
    o.add_argument("--out-md", required=True)
    o.add_argument("--out-turns", required=True)
    o.add_argument("--out-trials", required=True)
    options = parser.parse_args()
    if options.step == "offline":
        offline(options)


if __name__ == "__main__":
    main()
