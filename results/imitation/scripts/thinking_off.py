"""Thinking-off comparison: on recorded Qwen turns, how often does Qwen take the same action with thinking off?

Three steps (the request sending in between runs on casdgx01, see thinking_off_run.py):

  sample  Picks about 3,000 recorded Qwen turns, stratified by task and by position in the session (early / middle /
          late third), skipping turns whose recorded reply hit the output cap or failed. Writes turns.jsonl (one line
          per sampled turn: its recorded action, turn class, previous two actions, recorded tokens and time),
          jobs.jsonl (the request rebuilds for thinking_off_requests.ts) and totals.json (recorded Qwen generation
          time per turn class over all turns, for projecting the savings).
  report  Compares the variants' actions with the recorded action and with variant A and writes the report.

Usage (needs the `tokenizers` package, for ceiling.py's turn classifier):
    uv run --with tokenizers python thinking_off.py sample --tokenizer DIR --out-dir OUT RUN_DIR...
    uv run --with tokenizers python thinking_off.py report --dir OUT --results RESULTS.jsonl --out REPORT.md \
        [--prompt-test PROMPT_TEST.jsonl]

Variants (sent by thinking_off_run.py): A thinking on with the session's own sampling and no reasoning effort (the
template's default xhigh, as in the sessions; the noise baseline); B thinking off, temperature 0; C thinking off,
temperature 0.7, top_p 0.8, top_k 20; D as C with the context cut to the system prompt, the task message and the
last 3 tool steps; E thinking on, reasoning effort medium; F thinking on, reasoning effort low.

Definitions:
- Action of a turn: its bash commands in order, or "final text" when it calls no tool.
- Same command: equal after normalising whitespace and quoting (both split into shell words with shlex and rejoined
  with single spaces; a command shlex cannot split, such as one with an unbalanced quote, is compared after
  collapsing whitespace only).
- Same intent: for every call, the same list of (kind, target) pairs of its non-neutral parts, read by
  tools/jeff-first/imitation/labels.py `part_intent` (e.g. `cat -n main.py | head -80` and `sed -n 1,80p main.py`
  are both a read of main.py). File and folder targets are resolved against the folder the part runs in (calls
  start in /app; a `cd` earlier in the command moves it; events.py `part_folders`). A part that acts (not in the
  intent table) counts as ("other", its program name). A command that writes files also carries ("writes", the
  written file names), read by ceiling.py's classifier, so writes to different files differ in intent.
- Turn classes: results/imitation/scripts/ceiling.py's classifier (another agent's script, imported unchanged).
- Guard trigger: the action is identical (same commands) to one of the session's two previous recorded actions.
- Generation loop: the runner cancelled the request because the generated text started repeating itself
  (thinking_off_run.looping).
"""

import argparse
import collections
import difflib
import json
import random
import shlex
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "jeff-first"))

import ceiling  # noqa: E402
from imitation.events import Shell, part_folders, resolve  # noqa: E402
from imitation.labels import NEUTRAL, Intent, part_intent  # noqa: E402
from imitation.splitter import first_word, split_command  # noqa: E402

VARIANTS = ["A", "B", "C", "D", "E", "F"]
VARIANT_NAMES = {
    "A": "A on, xhigh (sessions)",
    "B": "B off, T=0",
    "C": "C off, T=0.7/p0.8/k20",
    "D": "D off, clean context",
    "E": "E on, medium",
    "F": "F on, low",
}
ROUTER_ORDER = [("off", ("B", "C")), ("low", ("F",)), ("medium", ("E",))]
ROUTER_VARIANT = [("low", "F"), ("medium", "E")]
PROMPT_TEST_VARIANTS = ["A", "B", "D", "E", "F"]
POSITIONS = ["early", "middle", "late"]
TARGET = 3000
SEED = 20261004
FINAL = "final text"


def normal_command(command):
    try:
        return " ".join(shlex.split(command))
    except ValueError:
        return " ".join(command.split())


PATH_KINDS = {"read", "peek", "list", "run", "service"}
START = Shell("/app", "/root")  # every bash call of these sessions starts in /app as root


def command_intent(command):
    signature = []
    folders, _ = part_folders(command, START)
    for part, folder in zip(split_command(command), folders):
        result = part_intent(part)
        if result is NEUTRAL:
            continue
        if isinstance(result, Intent):
            target = result.target
            if result.kind in PATH_KINDS and target:
                target = resolve(target, Shell(folder, START.home)) or target
            signature.append((result.kind, target))
        else:
            word = first_word(part.head)[0]
            signature.append(("other", word.rsplit("/", 1)[-1]))
    written = ceiling.classify_command(command, set())[1]
    if written:
        signature.append(("writes", tuple(sorted(written))))
    return tuple(signature)


def action_of(commands):
    """Action signature: tuple of normalised commands, or FINAL."""
    if not commands:
        return FINAL
    return tuple(normal_command(c) for c in commands)


def intent_of(commands):
    if not commands:
        return FINAL
    return tuple(command_intent(c) for c in commands)


# ---------------------------------------------------------------------------------------------------- sample


def assistant_entries(session_path):
    entries = [json.loads(line) for line in session_path.read_text().splitlines() if line.strip()]
    return [e for e in entries if e["type"] == "message" and e["message"]["role"] == "assistant"]


def sample(options):
    tokenizer = Tokenizer.from_file(str(Path(options.tokenizer) / "tokenizer.json"))
    trials = []
    for run in options.runs:
        trials.extend(sorted(Path(run).glob("*/round-*/*/*__*")))
    turns = []
    totals = collections.Counter()
    for trial in trials:
        session = ceiling.read_trial(trial, tokenizer)
        if session is None:
            continue
        for t in session["turns"]:
            totals[t["class"]] += t["gen_ms"]
        qwen = [t for t in session["turns"] if t["kind"] == "assistant"]
        session_path = next((trial / "agent" / "pi" / "sessions").glob("*.jsonl"))
        entries = assistant_entries(session_path)
        if len(entries) != len(qwen):
            raise ValueError(f"{trial}: {len(entries)} assistant entries but {len(qwen)} classified turns")
        actions = [action_of([c["command"] for c in t["calls"]]) for t in qwen]
        for i, (t, entry) in enumerate(zip(qwen, entries)):
            message = entry["message"]
            if message["stopReason"] in ("length", "error", "aborted"):
                continue
            usage = message["usage"]
            third = min(2, 3 * i // len(qwen))
            turns.append({
                "id": f"{trial.name}:{entry['id']}",
                "trial": trial.name,
                "task": session["task"],
                "machine": session["machine"],
                "session": str(session_path),
                "entry_id": entry["id"],
                "index": i,
                "turns_in_session": len(qwen),
                "position": POSITIONS[third],
                "class": t["class"],
                "pure": t.get("pure"),
                "commands": [c["command"] for c in t["calls"]],
                "final_text": "".join(c["text"] for c in message["content"] if c["type"] == "text")
                if not t["calls"] else None,
                "previous_actions": [list(a) if a != FINAL else FINAL for a in actions[max(0, i - 2):i]],
                "recorded_prompt_tokens": usage["input"] + usage["cacheRead"],
                "recorded_output_tokens": usage["output"],
                "recorded_reasoning_tokens": usage["reasoning"],
                "recorded_gen_ms": t["gen_ms"],
            })
    strata = collections.defaultdict(list)
    for t in turns:
        strata[(t["task"], t["position"])].append(t)
    rng = random.Random(SEED)
    for group in strata.values():
        rng.shuffle(group)
    # Equal share per stratum; strata with fewer turns give their leftover share to the others.
    chosen = []
    quota = {key: 0 for key in strata}
    left = TARGET
    open_keys = sorted(strata)
    while left > 0 and open_keys:
        share = max(1, left // len(open_keys))
        for key in list(open_keys):
            take = min(share, len(strata[key]) - quota[key], left)
            quota[key] += take
            left -= take
            if quota[key] == len(strata[key]):
                open_keys.remove(key)
            if left == 0:
                break
    for key in sorted(strata):
        chosen.extend(strata[key][:quota[key]])
    out = Path(options.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "turns.jsonl").open("w") as f:
        for t in chosen:
            f.write(json.dumps(t) + "\n")
    with (out / "jobs.jsonl").open("w") as f:
        for t in chosen:
            for kind, thinking, cut in (("on", "medium", False), ("off", "off", False), ("cut", "off", True)):
                f.write(json.dumps({"id": f"{t['id']}|{kind}", "session": t["session"], "entryId": t["entry_id"],
                                    "thinking": thinking, "cut": cut}) + "\n")
    (out / "totals.json").write_text(json.dumps({"gen_ms_by_class": totals, "eligible_turns": len(turns),
                                                 "strata": len(strata)}, indent=1))
    print(f"{len(turns)} eligible turns in {len(strata)} strata; sampled {len(chosen)}")


# ---------------------------------------------------------------------------------------------------- report


def variant_commands(result):
    """The bash commands of a variant's reply, or None when the reply has no usable action."""
    if result["outcome"] != "ok":
        return None
    commands = []
    for call in result["tool_calls"]:
        if call["name"] != "bash":
            return None
        try:
            arguments = json.loads(call["arguments"])
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
            return None
        commands.append(arguments["command"])
    return commands


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def similarity(a, b):
    return difflib.SequenceMatcher(None, "\n".join(a), "\n".join(b)).ratio()


def short(commands, limit=160):
    if commands is None:
        return "(no action)"
    if not commands:
        return "(final text)"
    text = " ;; ".join(commands)
    text = " ".join(text.split())
    return (text[:limit] + "...") if len(text) > limit else text


def tokens_of(result):
    """Output tokens of a request; for a request cut for looping, its streamed chunks (about one token each)."""
    return result["completion_tokens"] if result["completion_tokens"] is not None else result["chunks"]


def thinking_of(result):
    """Thinking tokens: the server's count, or for a request cut while thinking, all its chunks."""
    if result["reasoning_tokens"] is not None:
        return result["reasoning_tokens"]
    return result["chunks"] if result["reasoning"] and not result["content"] and not result["tool_calls"] else 0


def report(options):
    folder = Path(options.dir)
    turns = {t["id"]: t for t in map(json.loads, (folder / "turns.jsonl").read_text().splitlines())}
    totals = json.loads((folder / "totals.json").read_text())
    results = collections.defaultdict(dict)
    for line in Path(options.results).read_text().splitlines():
        r = json.loads(line)
        if r["variant"] in results[r["id"]]:
            raise ValueError(f"duplicate result {r['id']} {r['variant']}")
        results[r["id"]][r["variant"]] = r
    complete = [tid for tid, rs in results.items() if set(rs) == set(VARIANTS)]
    if options.require_all and len(complete) != len(turns):
        raise ValueError(f"{len(complete)} of {len(turns)} turns have all {len(VARIANTS)} results")
    rows = []
    for tid in complete:
        t = turns[tid]
        rec_cmds = t["commands"]
        row = {"turn": t, "rec_action": action_of(rec_cmds), "rec_intent": intent_of(rec_cmds), "v": {}}
        previous = [tuple(a) if a != FINAL else FINAL for a in t["previous_actions"]]
        row["rec_repeat"] = row["rec_action"] != FINAL and row["rec_action"] in previous
        for v in VARIANTS:
            r = results[tid][v]
            cmds = variant_commands(r)
            act = action_of(cmds) if cmds is not None else None
            row["v"][v] = {
                "r": r, "cmds": cmds, "action": act, "intent": intent_of(cmds) if cmds is not None else None,
                "repeat": act is not None and act != FINAL and act in previous,
            }
        rows.append(row)

    def same(row, v, other, kind):
        a = row["v"][v][kind]
        b = row["rec_" + kind] if other == "rec" else row["v"][other][kind]
        return a is not None and b is not None and a == b

    def router_label(row):
        """Cheapest variant whose action has the same intent as A's: off (B or C) < low (F) < medium (E) <
        xhigh (A). A turn where A has no usable action gets xhigh."""
        if row["v"]["A"]["intent"] is None:
            return "xhigh"
        for label, variants in ROUTER_ORDER:
            if any(same(row, v, "A", "intent") for v in variants):
                return label
        return "xhigh"

    def router_tokens(row, label):
        if label == "off":
            matching = [v for v in ("B", "C") if same(row, v, "A", "intent")]
            return min(tokens_of(row["v"][v]["r"]) for v in matching)
        return tokens_of(row["v"][dict(ROUTER_VARIANT)[label]]["r"])

    classes = sorted({r["turn"]["class"] for r in rows}, key=lambda c: ceiling.ALL_CLASSES.index(c))
    groups = [("all turns", rows)] + [(c, [r for r in rows if r["turn"]["class"] == c]) for c in classes]
    others = [v for v in VARIANTS if v != "A"]

    out = []
    w = out.append
    w("# Thinking off: does Qwen take the same action with less or no thinking?\n")
    w("Generated by `results/imitation/scripts/thinking_off.py` (requests rebuilt with pi's own code by "
      "`thinking_off_requests.ts`, sent by `thinking_off_run.py`). Data: the stage 3 collection snapshot (392 "
      "trials of bash-only pi sessions, Qwen3.8-27B with pi's thinking level medium, 45 Terminal-Bench training "
      "tasks). Replay servers: 8 H100s on casdgx01 (FP8 weights, vLLM 0.30.0) and, for a few turns, the RTX Pro 6000 "
      "on datigator (NVFP4).\n")
    w(f"Turns compared: {len(rows)} (sampled {len(turns)} of {totals['eligible_turns']} eligible turns, stratified "
      f"over {totals['strata']} task x position strata; turns whose recorded reply hit the output cap or failed "
      "are skipped). The run was stopped early (the GPUs were needed elsewhere); the turns that were not yet started "
      "had been queued in a fixed random order, so the compared turns cover "
      f"{len({r['turn']['task'] for r in rows})} of {len({t['task'] for t in turns.values()})} sampled tasks "
      "(the first ~450 turns ran in turn-id order, which favours tasks early in the alphabet). The RTX (NVFP4) "
      "answered a few turns as a second replay machine; see 'By replay machine'.\n")
    w("Important: pi's thinking level 'medium' does not reach the model. pi sends only `enable_thinking: true`, and "
      "Qwen's chat template then uses its default reasoning effort, xhigh, adding the line 'Reasoning effort is set "
      "to xhigh. Please think carefully ...' at the start of the system prompt. So the sessions ran at xhigh, and A "
      "reproduces them exactly.\n")
    w("Variants: **A** thinking on, no reasoning effort given (template default xhigh), server-default sampling "
      "(temperature 1.0, top_p 0.95, top_k 20) - the sessions' own setting and the noise baseline; **B** thinking "
      "off, temperature 0; **C** thinking off, temperature 0.7, top_p 0.8, top_k 20; **D** as C on a clean context "
      "(system prompt, task message, last 3 tool steps); **E** thinking on, reasoning effort medium; **F** thinking "
      "on, reasoning effort low. A, E and F may write 32,768 tokens, B, C and D 8,192.\n")

    diffs = collections.Counter(r["v"]["A"]["r"]["prompt_tokens"] - r["turn"]["recorded_prompt_tokens"]
                                for r in rows if r["v"]["A"]["r"]["prompt_tokens"] is not None)
    w("## Request check\n")
    w(f"Prompt tokens the server counted for the rebuilt request A minus the prompt tokens pi recorded for that "
      f"turn: {dict(sorted(diffs.items()))} (difference: number of turns). Requests cut for looping report no "
      "count.\n")

    w("## Outcomes\n")
    w("| variant | ok | final text | tool calls | hit output cap | generation loop (cancelled) | unusable call |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for v in VARIANTS:
        oc = collections.Counter(r["v"][v]["r"]["outcome"] for r in rows)
        finals = sum(1 for r in rows if r["v"][v]["action"] == FINAL)
        tools = sum(1 for r in rows if r["v"][v]["action"] not in (None, FINAL))
        unusable = sum(1 for r in rows if r["v"][v]["r"]["outcome"] == "ok" and r["v"][v]["cmds"] is None)
        w(f"| {VARIANT_NAMES[v]} | {oc['ok']} | {finals} | {tools} | {oc['length']} | {oc['loop']} | {unusable} |")
    w("\nUnusable call: the reply finished but its tool call is not bash or its arguments are not JSON with a "
      "command. A reply that hit the cap, looped or is unusable never agrees with anything.\n")

    def agreement_table(kind, title):
        w(f"## {title}\n")
        w("'X vs A' = share of turns where variant X's action equals A's. 'rel' = (X vs A) / (A vs rec): A agrees "
          "with the recorded session only so often (sampling noise of thinking itself), so 100% means X agrees with "
          "thinking-on as often as thinking-on agrees with itself. 'X vs rec' compares with the recorded action.\n")
        head = ["class", "turns", "A vs rec"] + [f"{v} vs A" for v in others] + [f"{v} rel" for v in others] + [
            f"{v} vs rec" for v in others]
        w("| " + " | ".join(head) + " |")
        w("|---|" + "---:|" * (len(head) - 1))
        for name, group in groups:
            n = len(group)
            a = sum(same(r, "A", "rec", kind) for r in group)
            vs_a = {v: sum(same(r, v, "A", kind) for r in group) for v in others}
            cells = [str(n), pct(a, n)] + [pct(vs_a[v], n) for v in others]
            cells += [pct(vs_a[v], a) if a else "-" for v in others]
            cells += [pct(sum(same(r, v, "rec", kind) for r in group), n) for v in others]
            w(f"| {name} | " + " | ".join(cells) + " |")
        w("")

    agreement_table("intent", "Agreement: same intent (same kind and target per call)")
    agreement_table("action", "Agreement: same commands")

    w("## A vs recorded by recording machine\n")
    w("The H100 sessions ran the same FP8 weights as this replay; the Spark and RTX sessions ran NVFP4 weights.\n")
    w("| machine | turns | A vs rec (commands) | A vs rec (intent) | B vs A (intent) |")
    w("|---|---:|---:|---:|---:|")
    by_machine = collections.defaultdict(list)
    for r in rows:
        by_machine["H100 (FP8)" if "fp8@" in r["turn"]["machine"] else "Spark/RTX (NVFP4)"].append(r)
    for name, group in sorted(by_machine.items()):
        n = len(group)
        w(f"| {name} | {n} | {pct(sum(same(r, 'A', 'rec', 'action') for r in group), n)} | "
          f"{pct(sum(same(r, 'A', 'rec', 'intent') for r in group), n)} | "
          f"{pct(sum(same(r, 'B', 'A', 'intent') for r in group), n)} |")
    w("")

    w("## By replay machine\n")
    w("All six variants of a turn ran on the same machine. Intent agreement with A and token saving per machine.\n")
    w("| replay machine | turns | A vs rec | " + " | ".join(f"{v} vs A" for v in others) + " | "
      + " | ".join(f"{v} saved" for v in others) + " |")
    w("|---|" + "---:|" * (2 + 2 * len(others)))
    by_replay = collections.defaultdict(list)
    for r in rows:
        machines = {r["v"][v]["r"]["machine"] for v in VARIANTS}
        if len(machines) != 1:
            raise ValueError(f"{r['turn']['id']}: variants ran on several machines {machines}")
        by_replay[machines.pop()].append(r)
    for name, group in sorted(by_replay.items()):
        n = len(group)
        base = sum(tokens_of(r["v"]["A"]["r"]) for r in group)
        cells = [pct(sum(same(r, "A", "rec", "intent") for r in group), n)]
        cells += [pct(sum(same(r, v, "A", "intent") for r in group), n) for v in others]
        cells += [f"{100 * (1 - sum(tokens_of(r['v'][v]['r']) for r in group) / base):.0f}%" for v in others]
        w(f"| {name} | {n} | " + " | ".join(cells) + " |")
    w("")

    w("## Agreement (intent) by position in the session\n")
    w("| position | turns | A vs rec | " + " | ".join(f"{v} vs A" for v in others) + " |")
    w("|---|" + "---:|" * (2 + len(others)))
    for pos in POSITIONS:
        group = [r for r in rows if r["turn"]["position"] == pos]
        n = len(group)
        cells = [pct(sum(same(r, "A", "rec", "intent") for r in group), n)]
        cells += [pct(sum(same(r, v, "A", "intent") for r in group), n) for v in others]
        w(f"| {pos} | {n} | " + " | ".join(cells) + " |")
    w("")

    w("## File writes (recorded class write/edit)\n")
    w("Exact = identical commands (so identical file content); similarity = difflib ratio of the joined commands "
      "(0 when the variant has no usable action). Correctness is not judged.\n")
    w("| variant | turns | exact vs rec | mean similarity vs rec | exact vs A | mean similarity vs A |")
    w("|---|---:|---:|---:|---:|---:|")
    writes = [r for r in rows if r["turn"]["class"] == ceiling.WRITE]
    with_a = [r for r in writes if r["v"]["A"]["cmds"] is not None]
    for v in VARIANTS:
        n = len(writes)
        sims_rec = [similarity(r["v"][v]["cmds"], r["turn"]["commands"]) if r["v"][v]["cmds"] is not None else 0.0
                    for r in writes]
        cells = [str(n), pct(sum(same(r, v, "rec", "action") for r in writes), n),
                 f"{statistics.mean(sims_rec):.2f}" if sims_rec else "-"]
        if v == "A":
            cells += ["-", "-"]
        else:
            sims_a = [similarity(r["v"][v]["cmds"], r["v"]["A"]["cmds"]) if r["v"][v]["cmds"] is not None else 0.0
                      for r in with_a]
            cells += [pct(sum(same(r, v, "A", "action") for r in writes), n),
                      f"{statistics.mean(sims_a):.2f}" if sims_a else "-"]
        w(f"| {VARIANT_NAMES[v]} | " + " | ".join(cells) + " |")
    w("")

    w("## Thinking, output tokens and time\n")
    w("Per turn (means). In a running session the prompt prefix is mostly cached, so a turn's time is mostly the "
      "time to write its output: the token saving (1 - variant output tokens / A output tokens, summed over the "
      "class) is the time saving apart from prompt processing (next section). Generation seconds are measured "
      "(request end minus first generated token) with all six variants running together on the same server, so "
      "they are comparable with each other only. A request cut for looping counts with what it spent until the "
      "cut.\n")
    head = (["class", "turns", "rec think"] + [f"{v} think" for v in ("A", "F", "E")]
            + [f"{v} out" for v in VARIANTS] + [f"{v} saved" for v in others] + [f"{v} gen s" for v in VARIANTS])
    w("| " + " | ".join(head) + " |")
    w("|---|" + "---:|" * (len(head) - 1))
    saved_by_class = {}
    for name, group in groups:
        n = len(group)
        out_tokens = {v: sum(tokens_of(r["v"][v]["r"]) for r in group) for v in VARIANTS}
        think = {v: sum(thinking_of(r["v"][v]["r"]) for r in group) for v in VARIANTS}
        gen_s = {v: sum(r["v"][v]["r"]["total_s"] - (r["v"][v]["r"]["first_s"] or 0) for r in group)
                 for v in VARIANTS}
        rec_think = sum(r["turn"]["recorded_reasoning_tokens"] for r in group)
        saved_by_class[name] = {v: 1 - out_tokens[v] / out_tokens["A"] for v in others}
        cells = [str(n), f"{rec_think / n:.0f}"] + [f"{think[v] / n:.0f}" for v in ("A", "F", "E")]
        cells += [f"{out_tokens[v] / n:.0f}" for v in VARIANTS]
        cells += [f"{100 * saved_by_class[name][v]:.0f}%" for v in others]
        cells += [f"{gen_s[v] / n:.1f}" for v in VARIANTS]
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")

    gen = totals["gen_ms_by_class"]
    total_gen = sum(gen.values())
    w("Projection onto the recorded sessions: share of all recorded Qwen generation time (all trials, all turns, "
      "including compaction summaries and capped turns) saved if every turn of the class ran the variant, using "
      "the class's token saving. An upper bound on time; it says nothing about whether the turns would still "
      "succeed.\n")
    w("| class | share of recorded Qwen time | " + " | ".join(f"saved by {v}" for v in others) + " |")
    w("|---|---:|" + "---:|" * len(others))
    for name in classes:
        share = gen.get(name, 0) / total_gen
        s = saved_by_class[name]
        w(f"| {name} | {100 * share:.1f}% | " + " | ".join(f"{100 * share * s[v]:.1f}%" for v in others) + " |")
    w(f"\nNot in the sample: {', '.join(f'{c} {100 * gen[c] / total_gen:.1f}%' for c in gen if c not in classes)} "
      "of recorded Qwen time.\n")

    w("## Router label: cheapest variant that takes A's action\n")
    w("For each turn, the cheapest variant whose action has the same intent as A's: off (B or C) < low (F) < "
      "medium (E) < xhigh (A only). This is the label a router would learn. A is one sample of a noisy process "
      "(see A vs rec), so a turn labelled xhigh may only mean the cheaper variants drew a different, equally good "
      "action. Saved = 1 - (output tokens of the labelled variant) / (A's output tokens), summed over the class.\n")
    w("| class | turns | off | low | medium | xhigh | saved (tokens) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for name, group in groups:
        n = len(group)
        labels = [router_label(r) for r in group]
        count = collections.Counter(labels)
        spent = sum(router_tokens(r, label) if label != "xhigh" else tokens_of(r["v"]["A"]["r"])
                    for r, label in zip(group, labels))
        base = sum(tokens_of(r["v"]["A"]["r"]) for r in group)
        w(f"| {name} | {n} | " + " | ".join(pct(count[label], n) for label in ("off", "low", "medium", "xhigh"))
          + f" | {100 * (1 - spent / base):.0f}% |")
    w("")

    if not options.prompt_test:
        w("## Prompt processing: the cache effect of switching\n")
        w("Not run: the GPUs were reassigned before the test (thinking_off_run.py --prompt-test). Expected effect, "
          "from the chat template: A (xhigh) and F (low) start the system prompt with a reasoning-effort line, E "
          "(medium) and B/C (off) do not, so switching between these groups mid-session changes the prompt from its "
          "first tokens and the whole context must be processed again instead of being read from the cache.\n")
    else:
        w("## Prompt processing: the cache effect of switching\n")
        w("Measured separately, one request at a time per server: first the turn's request exactly as the session "
          "sent it (A) is processed, so its prompt is cached as in a running session; then the variant's request is "
          "timed for one output token. A keeps the cached prefix. E (medium) and B/C (off) have no reasoning line in "
          "the system prompt, and F (low) has a different one, so their prompt differs from the first tokens on and "
          "must be processed in full. D (clean context) is short but also uncached.\n")
        probe = [json.loads(line) for line in Path(options.prompt_test).read_text().splitlines()]
        buckets = [(0, 10000), (10000, 40000), (40000, 80000), (80000, 10 ** 9)]
        w("| prompt tokens (A) | turns | " + " | ".join(f"{v} s" for v in PROMPT_TEST_VARIANTS) + " |")
        w("|---|---:|" + "---:|" * len(PROMPT_TEST_VARIANTS))
        a_tokens = {p["id"]: p["prompt_tokens"] for p in probe if p["variant"] == "A"}
        for low, high in buckets:
            ids = {i for i, tok in a_tokens.items() if low <= tok < high}
            if not ids:
                continue
            cells = []
            for v in PROMPT_TEST_VARIANTS:
                times = [p["prompt_s"] for p in probe if p["variant"] == v and p["id"] in ids]
                cells.append(f"{statistics.mean(times):.2f}")
            label = f"{low // 1000}k-{high // 1000}k" if high < 10 ** 9 else f"{low // 1000}k+"
            w(f"| {label} | {len(ids)} | " + " | ".join(cells) + " |")
        w("")

    w("## Loops\n")
    w("Guard trigger = the action is identical to one of the session's two previous recorded actions (the planned "
      "guard would re-ask with thinking on). Generation loop = the request was cancelled because its text started "
      "repeating itself (the last 400 characters are one 20-80 character piece repeated, or one line of 10+ "
      "characters occurs 8+ times in the last 60 lines). The recorded turn's own repeat rate is the baseline: a "
      "rerun after an edit is a legitimate repeat.\n")
    head = (["class", "turns", "recorded repeats"] + [f"{v} guard" for v in VARIANTS]
            + [f"{v} gen loops" for v in VARIANTS])
    w("| " + " | ".join(head) + " |")
    w("|---|" + "---:|" * (len(head) - 1))
    for name, group in groups:
        n = len(group)
        cells = [str(n), pct(sum(r["rec_repeat"] for r in group), n)]
        cells += [pct(sum(r["v"][v]["repeat"] for r in group), n) for v in VARIANTS]
        cells += [str(sum(r["v"][v]["r"]["outcome"] == "loop" for r in group)) for v in VARIANTS]
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")
    reasons = collections.Counter()
    for r in rows:
        for v in VARIANTS:
            if r["v"][v]["r"]["outcome"] == "loop":
                reasons[f"{v} in {r['v'][v]['r']['loop'].split(':')[0]}"] += 1
    w(f"Generation loops by variant and where they happened: {dict(sorted(reasons.items()))}.\n")
    loop_examples = [(r, v) for r in rows for v in VARIANTS if r["v"][v]["r"]["outcome"] == "loop"]
    if loop_examples:
        w("Examples of cancelled loops:\n")
        for r, v in random.Random(SEED).sample(loop_examples, min(8, len(loop_examples))):
            w(f"- {v}, {r['turn']['task']} turn {r['turn']['index']}: {r['v'][v]['r']['loop'][:200]}")
        w("")

    w("## Examples where B disagrees with A in intent\n")
    rng = random.Random(SEED)
    for name in classes:
        group = [r for r in rows if r["turn"]["class"] == name and not same(r, "B", "A", "intent")
                 and r["v"]["A"]["action"] is not None]
        if not group:
            continue
        w(f"**{name}**\n")
        for r in rng.sample(group, min(3, len(group))):
            w(f"- {r['turn']['task']} turn {r['turn']['index']}/{r['turn']['turns_in_session']}  ")
            w(f"  recorded: `{short(r['turn']['commands'])}`  ")
            w(f"  A: `{short(r['v']['A']['cmds'])}`  ")
            w(f"  B: `{short(r['v']['B']['cmds'])}`  ")
            w(f"  E: `{short(r['v']['E']['cmds'])}`  ")
            w(f"  D: `{short(r['v']['D']['cmds'])}`")
        w("")
    Path(options.out).write_text("\n".join(out) + "\n")
    print(f"wrote {options.out} ({len(rows)} turns)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    s = sub.add_parser("sample")
    s.add_argument("runs", nargs="+")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out-dir", required=True)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--require-all", action="store_true")
    r.add_argument("--prompt-test", help="results of thinking_off_run.py --prompt-test")
    options = parser.parse_args()
    if options.step == "sample":
        sample(options)
    else:
        report(options)


if __name__ == "__main__":
    main()
