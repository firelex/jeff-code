"""Offline score of a trained Jeff variant: the time it would save and how often its cheaper choices are wrong,
predicted from held-out labelled Qwen turns instead of running the benchmarks.

Jeff makes three decisions in Jeff-pi, each with a probability threshold (the run-time rules, quoted from the code):
- step (chooser.ts JeffChooser): on each page of the scout's menu Jeff's most likely option is taken when its
  probability is at least the step threshold, otherwise the scout hands over ("hand_over" on a tool page, "none_of_these"
  on an argument page). The scout stops after STEP_CAP steps in a row (scout.ts).
- router (thinking.ts jeffRouter): before each request to Qwen3.8-27B, the most likely thinking level (off, low,
  medium, xhigh) is used when it is xhigh or its probability is at least the router threshold, otherwise xhigh.
- trim (output-trim-control.ts jeffTrimDecider): when a new output of Qwen's own command shows more than 40 lines, the
  most likely cut (all, last200, last40, first40, first20last20) is used when it is "all" or its probability is at
  least the trim threshold, otherwise the whole output stays.
Ties go to the first option in the order the run time sends them (menu order; off..xhigh; all..first20last20), as the
run-time code's strict ">" scan does.

Subcommands:
- build: the held-out bundle. From the routing labels (routing_labels.py run), the trimming labels (trim_labels.py
  run) and the trial folders: every labelled turn of a held-out task with its stage-3 rows (imitation/stage3.py's own
  conversion of the trial), the session's recorded times (ceiling.py read_trial), and Jeff's questions for the three
  decisions in run-time option order (questions-step.jsonl, questions-router.jsonl, questions-trim.jsonl, in the
  Example format jeff-dev's `jeff.evaluate` reads). Held-out = the tasks the exporters put in the development or
  temperature split (export_routing.split_of: splits.json for Terminal-Bench 2.0, the seeded stage-1 rule per hub
  task), or the scoring tasks (held-out side of task-sets.json, listed in the collection's scoring-tasks.txt).
- predict: every option's probability for every question of one decision, from a running jeff-serve (POST
  /v1/systemone, one request at a time; busy answers are retried). Writes predictions in jeff.evaluate's format (id,
  options, probabilities), so `python -m jeff.evaluate --local --checkpoint <variant> --data questions-X.jsonl --output
  X.json` (X.predictions.jsonl) is an equivalent second way for a checkpoint not loaded in a service.
- oracle: sanity predictions with probability 1 on one option: "labels" (the labels themselves) or "fixed" (router
  xhigh, step hand over, trim all).
- score: one variant's predictions -> per decision and threshold: time saved, wrong-choice rate, coverage; the three
  decisions together on the same turns; and the best thresholds at fixed wrong-choice rates. Markdown and JSON.
- compare: several variants' score JSON files -> one table at fixed wrong-choice rates.

How each decision is scored on a turn (rec_s = the recorded xhigh turn's model time, the qwen_request line's
timings_ms.model):
- router: wrong = a level cheaper than the label (the label is the cheapest level the labeller found good enough).
  Saved = rec_s minus the re-ask wall time of the chosen level on that turn (routing labeller, request total_s), only
  for choices that are not wrong; output tokens likewise. A level more expensive than the label but below xhigh was not
  asked by the labeller's cascade (off, then low, then medium, stopping at the first good one), except on calibration
  turns: its saving counts 0 (low and medium take about as long as xhigh on average, routing-stats.md).
- trim: wrong = a cut the labels mark not good. Saved = the prompt tokens the chosen cut removes from this turn's
  request (trimming labeller, prompt_saved) times the prefill time per prompt token measured on the servers that
  replayed the turn (build --prefill-metrics: vLLM's own counters, the seconds requests spent in prefill divided by
  the prompt tokens not read from the prefix cache, under the collection's load). The cut
  output stays in the history, so later requests of the session hold fewer tokens too; that repeat is reported in
  tokens only (later requests read it from the prefix cache, whose time cost was not measured). A cut the labeller did
  not ask (last200 when a 40-line cut was already good, or a cut that does not shorten the output) is not judged:
  saved 0, not wrong, counted.
- step: Jeff walks the turn's recorded decisions page by page. A pick below the threshold, or a confident hand_over /
  none_of_these, hands over (no saving, not wrong). A confident pick that is not the recorded choice is wrong. Only a
  turn whose every step Jeff takes (covered) saves its rec_s; a turn Jeff starts but hands over saves nothing (Qwen
  still writes the turn): ceiling.md's rule that a turn counts only if Jeff could take every command in it.
- every Jeff request costs --decision-ms (the measured median is 178 ms per decision on B200 GPU 5): each page asked,
  one router question per Qwen request, one trim question per long output. Net saved = gross saved minus that.
- combined, on the same turns: a turn Jeff covers saves the turn (no Qwen request, so no router question); otherwise
  the router's saving; plus the trim saving. The turn is wrong if any of its decisions is.
Percentages: of the recorded Qwen generation time (all assistant replies and compaction summaries of the bundle's
sessions, session timestamps as ceiling.py) and of generation plus tool time.

Usage (from results/imitation/scripts; needs aiohttp and tokenizers as routing_labels.py, node for the trim question):
    uv run --with aiohttp==3.12.15 --with tokenizers python offline_score.py build --routing-labels F [F ...] \\
        --trim-labels F [F ...] --trials DIR [DIR ...] --split development --split temperature --out BUNDLE ...
    uv run ... python offline_score.py predict --url http://192.168.3.12:8920 --model jeff-router \\
        --questions BUNDLE/questions-router.jsonl --out PRED/router.predictions.jsonl
    uv run ... python offline_score.py score --bundle BUNDLE --name NAME --step P --router P --trim P \\
        --decision-ms 178 --out OUT/NAME
"""

import argparse
import collections
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "tools" / "jeff-first"))

import ceiling
import routing_labels as rl
from imitation import stage3
from imitation.export_jeff import question_text, row_id
from imitation.export_routing import ROUTER_OPTIONS, ROUTER_QUESTION, split_of
from imitation.export_trim import trim_question_text
from task_source import is_hub_task, load_task_sets
from tokenizers import Tokenizer

LEVELS = ("off", "low", "medium", "xhigh")
CHEAP_LEVELS = LEVELS[:-1]
UNCUT = "all"
HAND_OVER_IDS = ("hand_over", "none_of_these")
DECISIONS = ("step", "router", "trim")
HELD_OUT_SPLITS = ("development", "temperature", "scoring")
TARGETS = (0.02, 0.05, 0.10)
DEFAULT_THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0)
SCOUT_TS = HERE.parents[2] / "packages" / "coding-agent" / "src" / "core" / "jeff-first" / "scout.ts"


def step_cap():
    """scout.ts STEP_CAP: the most scout steps in a row before it hands over without asking."""
    match = re.search(r"^export const STEP_CAP = (\d+);$", SCOUT_TS.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError(f"no 'export const STEP_CAP = N;' line in {SCOUT_TS}")
    return int(match.group(1))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ build


class ScoringSides:
    """task-sets.json as stage3.convert_trial sees it for the scoring sessions: convert_trial converts only
    training-side sessions (collection must never record held-out tasks), but the scoring sessions are held-out tasks
    recorded on purpose for this score. This view names exactly the listed scoring tasks "training"; every other task
    keeps its real side."""

    def __init__(self, task_sets, scoring_tasks):
        self.task_sets = task_sets
        self.scoring_tasks = frozenset(scoring_tasks)

    def side(self, task_id):
        return "training" if task_id in self.scoring_tasks else self.task_sets.side(task_id)


def held_out_split(task, splits, task_sets, scoring_tasks):
    """The split a task's rows belong to: train, development or temperature as the exporters place them, "scoring"
    for a listed scoring task, or "held-out" for another held-out task."""
    if is_hub_task(task) and task_sets.side(task) != "training":
        if task in scoring_tasks:
            if task_sets.side(task) != "held_out":
                raise ValueError(f"the scoring task {task} is {task_sets.side(task)} in task-sets.json, not held_out")
            return "scoring"
        return "held-out"
    if task in scoring_tasks:
        raise ValueError(f"the scoring task {task} is on the training side")
    return split_of(task, splits, task_sets)


def find_trials(roots):
    """Trial folder name (<task>__<id>) -> path, for every trial folder with a config.json under the roots."""
    found = {}
    for root in roots:
        for config in Path(root).rglob("config.json"):
            trial = config.parent
            if "__" not in trial.name:
                continue
            if trial.name in found and found[trial.name].resolve() != trial.resolve():
                raise ValueError(f"the trial {trial.name} is in two places: {found[trial.name]} and {trial}")
            found[trial.name] = trial
    return found


PREFILL_METRICS = ("vllm:request_prefill_time_seconds_sum", "vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total")


def read_prefill_metrics(specs):
    """--prefill-metrics MACHINE=FILE (repeatable; one file per server, the text of its /metrics page): per replay
    machine, the seconds requests spent in prefill divided by the prompt tokens not read from the prefix cache, summed
    over that machine's servers."""
    sums = collections.defaultdict(lambda: {"prefill_s": 0.0, "uncached_tokens": 0.0, "servers": 0})
    for spec in specs:
        machine, _, path = spec.partition("=")
        if not machine or not path:
            raise ValueError(f"--prefill-metrics {spec!r}: write MACHINE=FILE")
        values = {}
        for line in Path(path).read_text().splitlines():
            name = line.split("{", 1)[0]
            if name in PREFILL_METRICS:
                if name in values:
                    raise ValueError(f"{path}: {name} appears twice (one engine per file expected)")
                values[name] = float(line.rsplit(" ", 1)[1])
        missing = [name for name in PREFILL_METRICS if name not in values]
        if missing:
            raise ValueError(f"{path}: no {', '.join(missing)}")
        entry = sums[machine]
        entry["prefill_s"] += values["vllm:request_prefill_time_seconds_sum"]
        entry["uncached_tokens"] += values["vllm:prompt_tokens_total"] - values["vllm:prompt_tokens_cached_total"]
        entry["servers"] += 1
    rates = {}
    for machine, entry in sorted(sums.items()):
        if entry["uncached_tokens"] <= 0 or entry["prefill_s"] <= 0:
            raise ValueError(f"{machine}: no prefill measured ({entry})")
        rates[machine] = {**entry, "seconds_per_token": entry["prefill_s"] / entry["uncached_tokens"]}
    return rates


def paired_prefill_estimate(trim_turns):
    """For comparison only: per replay machine, the within-turn least-squares slope of time to first token on prompt
    tokens over the trimming labeller's requests (each labelled turn asks the uncut and cut requests at once on one
    server; values are centred per turn, so the turn's queue and cache state cancel). Concurrent requests that share a
    prefix make this noisy (it can come out negative), so the score uses read_prefill_metrics."""
    sums = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])
    for turn in trim_turns:
        points = [(a["prompt_tokens"], a["first_s"]) for a in turn["asks"]
                  if a["outcome"] == "ok" and a["prompt_tokens"] is not None and a["first_s"] is not None]
        if len(points) < 2:
            continue
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        entry = sums[turn["replay_machine"]]
        entry[0] += sum((x - mean_x) * (y - mean_y) for x, y in points)
        entry[1] += sum((x - mean_x) ** 2 for x, _ in points)
        entry[2] += len(points)
        entry[3] += 1
    return {machine: {"seconds_per_token": sxy / sxx if sxx > 0 else None, "requests": requests, "turns": turns}
            for machine, (sxy, sxx, requests, turns) in sorted(sums.items())}


def router_levels(turn):
    """Per cheap level: whether it was asked, its check, its wall time and output tokens (first sample)."""
    levels = {}
    for level in CHEAP_LEVELS:
        asks = [a for a in turn["asks"] if a["level"] == level and a["sample"] == 1]
        if len(asks) > 1:
            raise ValueError(f"{turn['id']}: {len(asks)} first-sample asks at {level}")
        if not asks:
            if level in turn["checks"]:
                raise ValueError(f"{turn['id']}: {level} was checked but not asked")
            levels[level] = None
            continue
        if level not in turn["checks"]:
            raise ValueError(f"{turn['id']}: {level} was asked but not checked")
        levels[level] = {"good": turn["checks"][level]["good"], "s": asks[0]["total_s"],
                         "out": rl._tokens(asks[0])}
    label = turn["label"]
    needed = CHEAP_LEVELS if label == "xhigh" else CHEAP_LEVELS[:LEVELS.index(label) + 1]
    for level in needed:
        if levels[level] is None:
            raise ValueError(f"{turn['id']}: level {level} (needed for the label {label}) was never asked")
        if levels[level]["good"] != (level == label):
            raise ValueError(f"{turn['id']}: level {level} is {'good' if levels[level]['good'] else 'not good'} "
                             f"but the label is {label}")
    return levels


def trim_record(line, choices):
    """A trimming "turn" line -> the cuts asked with their check and prompt tokens saved."""
    asked = {a["cut"]: a for a in line["asks"]}
    if UNCUT not in asked:
        raise ValueError(f"{line['id']}: no uncut request")
    cuts = {}
    for cut, check in line["checks"].items():
        if cut not in choices or cut == UNCUT:
            raise ValueError(f"{line['id']}: unknown cut {cut}")
        cuts[cut] = {"good": check["good"], "prompt_saved": asked[cut]["prompt_saved"]}
    if line["label"] != UNCUT and not cuts[line["label"]]["good"]:
        raise ValueError(f"{line['id']}: label {line['label']} is not a good cut")
    return {"total_lines": line["total_lines"], "shown_lines": line["shown_lines"], "label": line["label"],
            "cuts": cuts, "replay_machine": line["replay_machine"]}


def session_totals(trial_path, tokenizer):
    """Recorded Qwen generation time (replies and compaction summaries) and tool time of one trial (ceiling.py)."""
    session = ceiling.read_trial(trial_path, tokenizer)
    if session is None:
        raise ValueError(f"{trial_path}: no session file")
    turns = session["turns"]
    return {"gen_s": sum(t["gen_ms"] for t in turns) / 1000, "tool_s": sum(t["tool_ms"] for t in turns) / 1000,
            "requests": sum(1 for t in turns if t["kind"] == "assistant")}


def step_question(row):
    return {"id": row_id(row), "suite": row["task"], "family": row["task"], "state": row["state"],
            "question": {"type": "choice", "instructions": question_text(row),
                         "criteria": {option["id"]: option["description"] for option in row["options"]}},
            "label": row["label"], "target": row["label"],
            "source": {"dataset": row["source"], "task": row["task"], "session": row["session"], "turn": row["turn"],
                       "decision": row["decision"], "level": row["level"], "page": row["page"]}}


def turn_question(kind, turn, state, instructions, criteria, label):
    return {"id": f"{kind}:{turn['id']}", "suite": turn["task"], "family": turn["task"], "state": state,
            "question": {"type": "choice", "instructions": instructions, "criteria": criteria},
            "label": label, "target": label,
            "source": {"dataset": "own", "task": turn["task"], "session": turn["session_id"], "turn": turn["turn"]}}


def order_rows(rows):
    """One decision's rows in the order the scout asks them: tool pages, then argument pages."""
    return sorted(rows, key=lambda row: (row["level"] != "tool", row["page"]))


def build(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to write into {out}: it is not empty")

    splits = json.loads(Path(args.splits).read_text())
    task_sets = load_task_sets(Path(args.task_sets), None)
    scoring = frozenset(Path(args.scoring_tasks).read_text().split()) if args.scoring_tasks else frozenset()
    if "scoring" in args.split and not scoring:
        raise ValueError("--split scoring needs --scoring-tasks")
    trim_question = trim_question_text(args.node)
    choices = trim_question["choices"]

    routing_turns, routing_excluded = rl.read_rows(args.routing_labels)
    trim_lines = [line for path in args.trim_labels for line in read_jsonl(path)]
    trim_by_id = {}
    for line in trim_lines:
        if line["kind"] not in ("turn", "short", "excluded"):
            continue
        if line["id"] in trim_by_id:
            raise ValueError(f"the trimming labels hold the turn {line['id']} twice")
        trim_by_id[line["id"]] = line
    rates = read_prefill_metrics(args.prefill_metrics)
    paired = paired_prefill_estimate([line for line in trim_lines if line["kind"] == "turn"])

    # A trial's session totals are the denominators, so only trials the routing labeller has finished (its "trial"
    # line: every labellable turn written) are used.
    finished = {Path(line["trial_dir"]).name for path in args.routing_labels for line in read_jsonl(path)
                if line["kind"] == "trial"}
    counts = collections.Counter()
    wanted = []
    for turn in routing_turns:
        split = held_out_split(turn["task"], splits, task_sets, scoring)
        counts[f"routing turns in split {split}"] += 1
        if split not in args.split:
            continue
        if turn["trial"] not in finished:
            counts["turns left out: the routing labeller has not finished their trial"] += 1
            continue
        wanted.append({**turn, "split": split})
    by_trial = collections.defaultdict(list)
    for turn in wanted:
        by_trial[turn["trial"]].append(turn)
    trial_paths = find_trials(args.trials)
    missing = sorted(set(by_trial) - set(trial_paths))
    if missing:
        raise FileNotFoundError(f"{len(missing)} trials with held-out labelled turns are not under --trials, e.g. "
                                f"{missing[0]}")

    tasks = stage3.read_tasks(Path(args.training_tasks), Path(args.task_sets))
    if scoring:
        tasks = replace(tasks, hub=ScoringSides(tasks.hub, scoring))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    builds = frozenset(args.build)
    turns_out, sessions_out = [], []
    questions = {decision: [] for decision in DECISIONS}
    for trial in sorted(by_trial):
        path = trial_paths[trial]
        conversion = stage3.Conversion(builds=sorted(builds))
        stage3.convert_trial(path, tasks, builds, conversion)
        if not conversion.trials:
            for reason in conversion.skipped:
                counts[f"trials left out by stage 3: {reason}"] += 1
                counts[f"turns left out (stage 3 left their trial out: {reason})"] += len(by_trial[trial])
            continue
        decisions = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in conversion.rows:
            decisions[(row["session"], row["turn"])][row["decision"]].append(row)
        kept = []
        for turn in sorted(by_trial[trial], key=lambda t: t["turn"]):
            found = decisions.get((turn["session_id"], turn["turn"]))
            if found is None:
                counts["turns left out: no stage-3 row"] += 1
                continue
            trim_line = trim_by_id.get(turn["id"])
            if trim_line is None:
                counts["turns left out: not yet seen by the trimming labeller"] += 1
                continue
            if trim_line["kind"] == "excluded":
                counts[f"turns left out: trimming labeller left it out ({trim_line['reason']})"] += 1
                continue
            ordered = [order_rows(found[d]) for d in sorted(found)]
            first = ordered[0][0]
            if first["level"] != "tool" or first["page"] != 1:
                raise ValueError(f"{turn['id']}: the first decision does not start on tool page 1")
            trim = trim_record(trim_line, choices) if trim_line["kind"] == "turn" else None
            record = {
                "id": turn["id"], "task": turn["task"], "split": turn["split"], "trial": trial,
                "session": turn["session_id"], "turn": turn["turn"], "class": turn["class"],
                "machine": turn["recording_machine"], "replay_machine": turn["replay_machine"],
                "rec_s": turn["recorded"]["model_ms"] / 1000, "rec_out": turn["recorded"]["output_tokens"],
                "router": {"label": turn["label"], "calibration": turn["calibration"], "levels": router_levels(turn)},
                "trim": trim,
                "step": [[row_id(row) for row in rows] for rows in ordered],
            }
            kept.append(record)
            for rows in ordered:
                questions["step"].extend(step_question(row) for row in rows)
            questions["router"].append(turn_question("route", turn, first["state"], ROUTER_QUESTION,
                                                     dict(ROUTER_OPTIONS), turn["label"]))
            if trim is not None:
                questions["trim"].append(turn_question(
                    "trim", turn, first["state"], trim_question["template"].replace("{lines}", str(trim["total_lines"])),
                    {choice: trim_question["options"][choice] for choice in choices}, trim["label"]))
        if not kept:
            continue
        totals = session_totals(path, tokenizer)
        for record in kept:
            record["later_requests"] = totals["requests"] - record["turn"]
            if record["later_requests"] < 0:
                raise ValueError(f"{record['id']}: turn {record['turn']} is past the session's {totals['requests']} "
                                 "requests")
        turns_out.extend(kept)
        sessions_out.append({"trial": trial, "task": by_trial[trial][0]["task"],
                             "split": by_trial[trial][0]["split"], **totals})
    if not turns_out:
        raise ValueError("no held-out turn has stage-3 rows and labels; nothing to write")
    machines = {t["replay_machine"] for t in turns_out if t["trim"] is not None}
    unmeasured = sorted(machines - set(rates))
    if unmeasured:
        raise ValueError(f"no prefill measurement for {unmeasured}: give --prefill-metrics for their servers")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "turns.jsonl", turns_out)
    write_jsonl(out / "sessions.jsonl", sessions_out)
    for decision in DECISIONS:
        write_jsonl(out / f"questions-{decision}.jsonl", questions[decision])
    counts.update({"routing turns left out by the routing labeller": len(routing_excluded),
                   "turns kept": len(turns_out), "sessions": len(sessions_out),
                   "turns with a trim decision": sum(t["trim"] is not None for t in turns_out),
                   **{f"questions {d}": len(q) for d, q in questions.items()}})
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "splits": sorted(args.split),
            "inputs": {"routing_labels": [str(p) for p in args.routing_labels],
                       "trim_labels": [str(p) for p in args.trim_labels], "trials": [str(p) for p in args.trials],
                       "builds": sorted(builds), "scoring_tasks": str(args.scoring_tasks) if args.scoring_tasks else None},
            "step_cap": step_cap(), "prefill": rates, "prefill_paired_estimate": paired, "counts": dict(sorted(counts.items())),
            "questions_sha256": {d: sha256_file(out / f"questions-{d}.jsonl") for d in DECISIONS},
            "turns_sha256": sha256_file(out / "turns.jsonl")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta["counts"], indent=1))


# ------------------------------------------------------------------------------------------------ predictions


OVER_LENGTH = re.compile(r"exceeds the \d+-token limit")


class OverLength(Exception):
    """jeff-serve refused the question because its prompt is longer than the model's limit (status 422). The run time
    gets the same refusal for the same state and question, and ends the turn with a JeffFirst error."""


def post_question(url, model, question, timeout_s, busy_give_up_s):
    """One question to jeff-serve; returns (probabilities by option id, served-by name, busy waits). Raises
    OverLength for a prompt over the model's limit."""
    body = json.dumps({"model": model, "state": question["state"],
                       "questions": {"q": question["question"]}}).encode()
    busy = 0
    started = time.monotonic()
    while True:
        request = urllib.request.Request(f"{url.rstrip('/')}/v1/systemone", data=body,
                                         headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                reply = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            if error.code != 529:
                detail = error.read()[:500].decode(errors="replace")
                if error.code == 422 and OVER_LENGTH.search(detail):
                    raise OverLength(detail) from error
                raise RuntimeError(f"{url} answered {error.code} for {question['id']}: {detail}") from error
            busy += 1
            if time.monotonic() - started > busy_give_up_s:
                raise RuntimeError(f"{url} stayed busy for {busy_give_up_s} s ({busy} tries) on {question['id']}")
            time.sleep(0.02)
    answer = reply["answers"]["q"]
    probabilities = answer["probabilities"]
    if set(probabilities) != set(question["question"]["criteria"]):
        raise ValueError(f"{question['id']}: probabilities for {sorted(probabilities)}, not for the options sent")
    return probabilities, reply["model"], busy


def predict(args):
    questions = read_jsonl(args.questions)
    out = Path(args.out)
    done = {row["id"] for row in read_jsonl(out)} if out.exists() else set()
    if done - {q["id"] for q in questions}:
        raise ValueError(f"{out} holds predictions for questions not in {args.questions}; use another --out")
    todo = [q for q in questions if q["id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    busy_total = 0
    over_length = 0
    started = time.monotonic()
    with out.open("a", encoding="utf-8") as stream:
        for index, question in enumerate(todo, start=1):
            asked = time.monotonic()
            options = list(question["question"]["criteria"])
            try:
                probabilities, served, busy = post_question(args.url, args.model, question, args.timeout,
                                                            args.busy_give_up)
                latency = (time.monotonic() - asked) * 1000
                latencies.append(latency)
                busy_total += busy
                row = {"id": question["id"], "options": options, "probabilities": [probabilities[o] for o in options],
                       "model": served, "latency_ms": round(latency, 1), "busy_waits": busy}
            except OverLength as refusal:
                over_length += 1
                row = {"id": question["id"], "options": options, "over_length": True, "detail": str(refusal),
                       "model": args.model}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if index % args.progress == 0 or index == len(todo):
                elapsed = time.monotonic() - started
                ordered = sorted(latencies) or [float("nan")]
                print(f"{index}/{len(todo)} ({len(done)} done before): {index / elapsed:.2f} questions/s, median "
                      f"{ordered[len(ordered) // 2]:.0f} ms, p90 {ordered[math.ceil(0.9 * len(ordered)) - 1]:.0f} ms, "
                      f"busy waits {busy_total}, over the length limit {over_length}", flush=True)
            if args.pause_ms:
                time.sleep(args.pause_ms / 1000)


def oracle(args):
    """Predictions with probability 1 on one option for every question of one decision that is not over the length
    limit (bundle over-length.json)."""
    questions = read_jsonl(Path(args.bundle) / f"questions-{args.decision}.jsonl")
    refused = bundle_over_length(args.bundle)[args.decision]
    fixed = {"router": "xhigh", "trim": UNCUT}
    rows = []
    for question in questions:
        options = list(question["question"]["criteria"])
        if question["id"] in refused:
            rows.append({"id": question["id"], "options": options, "over_length": True, "model": f"oracle:{args.kind}"})
            continue
        if args.kind == "labels":
            pick = question["label"]
        elif args.decision == "step":
            pick = next(o for o in options if o in HAND_OVER_IDS)
        else:
            pick = fixed[args.decision]
        if pick not in options:
            raise ValueError(f"{question['id']}: {pick} is not an option")
        rows.append({"id": question["id"], "options": options,
                     "probabilities": [1.0 if o == pick else 0.0 for o in options], "model": f"oracle:{args.kind}"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)


def read_predictions(path, questions, over_length):
    """id -> (options in the question's order, probabilities), or None for a question in the bundle's over-length list
    (over-length.json: jeff-serve refused it as longer than the model's limit). Every other question needs exactly one
    prediction with the question's options (jeff.evaluate's predictions and this script's have the criteria order; any
    order is read)."""
    by_id = {}
    for row in read_jsonl(path):
        if row["id"] in by_id:
            raise ValueError(f"{path}: two predictions for {row['id']}")
        by_id[row["id"]] = row
    result = {}
    for question in questions:
        row = by_id.get(question["id"])
        if question["id"] in over_length:
            if row is not None and not row.get("over_length"):
                raise ValueError(f"{path}: a prediction for {question['id']}, which the bundle lists as over the "
                                 "length limit")
            result[question["id"]] = None
            continue
        if row is None:
            raise ValueError(f"{path}: no prediction for {question['id']}")
        if row.get("over_length"):
            raise ValueError(f"{path}: {question['id']} was refused as over the length limit but is not in the "
                             "bundle's over-length.json; run the over-length command on a complete prediction run")
        options = list(question["question"]["criteria"])
        if sorted(row["options"]) != sorted(options):
            raise ValueError(f"{path}: {question['id']} predicted options {row['options']}, asked {options}")
        given = dict(zip(row["options"], row["probabilities"]))
        result[question["id"]] = (options, [float(given[o]) for o in options])
    extra = set(by_id) - set(result)
    if extra:
        raise ValueError(f"{path}: {len(extra)} predictions for questions not in the bundle, e.g. {min(extra)}")
    return result


def over_length_command(args):
    """Record which questions jeff-serve refused as over the length limit (from complete `predict` runs of any model:
    the limit and the prompt are the model's, the same for every variant of one base), and write
    questions-<decision>.fit.jsonl without them for `jeff.evaluate --local`, which stops on such a row."""
    folder = Path(args.bundle)
    _, _, _, questions = load_bundle(folder)
    ids = {}
    for decision in DECISIONS:
        path = getattr(args, decision)
        rows = {row["id"]: row for row in read_jsonl(path)}
        missing = [q["id"] for q in questions[decision] if q["id"] not in rows]
        if missing:
            raise ValueError(f"{path}: {len(missing)} questions not predicted yet, e.g. {missing[0]}")
        ids[decision] = sorted(row["id"] for row in rows.values() if row.get("over_length"))
    record = folder / "over-length.json"
    if record.exists() and json.loads(record.read_text())["ids"] != ids:
        raise ValueError(f"{record} already lists other questions; build a new bundle instead of changing it")
    record.write_text(json.dumps({"from": {d: str(Path(getattr(args, d)).resolve()) for d in DECISIONS}, "ids": ids},
                                 indent=1) + "\n")
    for decision in DECISIONS:
        refused = set(ids[decision])
        write_jsonl(folder / f"questions-{decision}.fit.jsonl",
                    [q for q in questions[decision] if q["id"] not in refused])
    print(json.dumps({d: len(v) for d, v in ids.items()}))


def most_likely(prediction):
    """The first option with the highest probability, in the run-time order (strict '>' scan from the first)."""
    options, probabilities = prediction
    best = 0
    for index in range(1, len(options)):
        if probabilities[index] > probabilities[best]:
            best = index
    return options[best], probabilities[best]


# ------------------------------------------------------------------------------------------------ scoring


def step_outcome(turn, predictions, labels, threshold, cap):
    """(outcome, pages asked, steps taken): outcome "covered", "handover", "wrong" (see the module docstring) or
    "error" (a page over Jeff's length limit: the run time ends the turn with an error)."""
    pages = 0
    for index, rows in enumerate(turn["step"]):
        if index >= cap:
            return "handover", pages, index
        for rid in rows:
            pages += 1
            if predictions[rid] is None:
                return "error", pages, index
            pick, probability = most_likely(predictions[rid])
            if probability < threshold or pick in HAND_OVER_IDS:
                return "handover", pages, index
            if pick != labels[rid]:
                return "wrong", pages, index
    return "covered", pages, len(turn["step"])


def router_outcome(turn, prediction, threshold):
    """(level, wrong, seconds saved as measured, seconds saved load-matched, output tokens saved, unmeasured).
    Measured = recorded seconds minus the re-ask's wall seconds (the brief's definition). Load-matched = the chosen
    level's output tokens at the recorded reply's own seconds per output token: the re-asks ran on servers busier
    than during the recording (about 40 against 74 tokens per second on the B200), which the measured difference
    charges to the cheaper level. Level None: the question is over Jeff's length limit (an error at run time)."""
    if prediction is None:
        return None, False, 0.0, 0.0, 0, False
    best, probability = most_likely(prediction)
    level = best if best == "xhigh" or probability >= threshold else "xhigh"
    label = turn["router"]["label"]
    if level == "xhigh":
        return level, False, 0.0, 0.0, 0, False
    if LEVELS.index(level) < LEVELS.index(label):
        return level, True, 0.0, 0.0, 0, False
    measured = turn["router"]["levels"][level]
    if measured is None:
        return level, False, 0.0, 0.0, 0, True
    if turn["rec_out"] <= 0:
        raise ValueError(f"{turn['id']}: the recorded reply has {turn['rec_out']} output tokens")
    out = turn["rec_out"] - measured["out"]
    return level, False, turn["rec_s"] - measured["s"], turn["rec_s"] * out / turn["rec_out"], out, False


def trim_outcome(turn, prediction, threshold, rates):
    """(cut, wrong, prompt tokens saved, seconds saved, unjudged); cut None: over Jeff's length limit."""
    if prediction is None:
        return None, False, 0, 0.0, False
    best, probability = most_likely(prediction)
    cut = best if best == UNCUT or probability >= threshold else UNCUT
    if cut == UNCUT:
        return cut, False, 0, 0.0, False
    asked = turn["trim"]["cuts"].get(cut)
    if asked is None:
        return cut, False, 0, 0.0, True
    if not asked["good"]:
        return cut, True, 0, 0.0, False
    tokens = asked["prompt_saved"]
    return cut, False, tokens, tokens * rates[turn["trim"]["replay_machine"]]["seconds_per_token"], False


def load_bundle(folder):
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    turns = read_jsonl(folder / "turns.jsonl")
    sessions = read_jsonl(folder / "sessions.jsonl")
    questions = {d: read_jsonl(folder / f"questions-{d}.jsonl") for d in DECISIONS}
    for decision in DECISIONS:
        if sha256_file(folder / f"questions-{decision}.jsonl") != meta["questions_sha256"][decision]:
            raise ValueError(f"{folder}: questions-{decision}.jsonl changed since the build")
    return meta, turns, sessions, questions


def bundle_over_length(folder):
    """The bundle's over-length question ids per decision (over-length.json), empty when none was recorded."""
    record = Path(folder) / "over-length.json"
    ids = json.loads(record.read_text())["ids"] if record.exists() else {}
    return {d: frozenset(ids.get(d, ())) for d in DECISIONS}


class Scorer:
    def __init__(self, meta, turns, sessions, questions, predictions, decision_s):
        self.meta = meta
        self.turns = turns
        self.labels = {q["id"]: q["label"] for d in DECISIONS for q in questions[d]}
        self.predictions = predictions
        self.decision_s = decision_s
        self.rates = meta["prefill"]
        self.cap = meta["step_cap"]
        self.gen_s = sum(s["gen_s"] for s in sessions)
        self.tool_s = sum(s["tool_s"] for s in sessions)
        self.rec_out = sum(t["rec_out"] for t in turns)
        self.trim_turns = [t for t in turns if t["trim"] is not None]

    def per_turn(self, decision, threshold):
        """Per turn, for one decision alone, a dict: wrong, saved (seconds, measured), matched (seconds, load-matched;
        equal to saved except for the router), jeff (seconds of Jeff requests) and details; None for a turn without
        that decision (trim on a turn without a long output)."""
        results = []
        for turn in self.turns:
            if decision == "step":
                outcome, pages, steps = step_outcome(turn, self.predictions["step"], self.labels, threshold, self.cap)
                saved = turn["rec_s"] if outcome == "covered" else 0.0
                results.append({"wrong": outcome == "wrong", "error": outcome == "error", "saved": saved,
                                "matched": saved, "jeff": pages * self.decision_s, "outcome": outcome, "steps": steps})
            elif decision == "router":
                level, wrong, saved, matched, out, unmeasured = router_outcome(
                    turn, self.predictions["router"][f"route:{turn['id']}"], threshold)
                results.append({"wrong": wrong, "error": level is None, "saved": saved, "matched": matched,
                                "jeff": self.decision_s, "level": level, "out": out, "unmeasured": unmeasured})
            elif turn["trim"] is None:
                results.append(None)
            else:
                cut, wrong, tokens, saved, unjudged = trim_outcome(
                    turn, self.predictions["trim"][f"trim:{turn['id']}"], threshold, self.rates)
                results.append({"wrong": wrong, "error": cut is None, "saved": saved, "matched": saved,
                                "jeff": self.decision_s, "cut": cut, "tokens": tokens, "unjudged": unjudged,
                                "repeat_tokens": tokens * turn["later_requests"]})
        return results

    def totals(self, wrong, decisions, gross, matched, jeff):
        return {"decisions": decisions, "wrong": int(wrong), "wrong_rate": wrong / decisions if decisions else 0.0,
                "gross_saved_s": gross, "jeff_s": jeff, "net_saved_s": gross - jeff,
                "net_saved_share_gen": (gross - jeff) / self.gen_s,
                "net_saved_share_gen_tool": (gross - jeff) / (self.gen_s + self.tool_s),
                "matched_net_saved_s": matched - jeff, "matched_net_saved_share_gen": (matched - jeff) / self.gen_s,
                "matched_net_saved_share_gen_tool": (matched - jeff) / (self.gen_s + self.tool_s)}

    def summary(self, decision, threshold, results):
        scored = [r for r in results if r is not None]
        n = len(scored)
        row = {"threshold": threshold, **self.totals(sum(r["wrong"] for r in scored), n, sum(r["saved"] for r in scored),
                                                     sum(r["matched"] for r in scored), sum(r["jeff"] for r in scored)),
               "errors": sum(r["error"] for r in scored)}
        if decision == "step":
            outcomes = collections.Counter(r["outcome"] for r in scored)
            row.update({"covered": outcomes["covered"], "coverage": outcomes["covered"] / n,
                        "handover": outcomes["handover"], "steps_taken": sum(r["steps"] for r in scored)})
        elif decision == "router":
            levels = collections.Counter(r["level"] for r in scored if r["level"] is not None)
            cheaper = sum(levels[level] for level in CHEAP_LEVELS)
            out = sum(r["out"] for r in scored)
            row.update({"levels": dict(levels), "cheaper": cheaper, "coverage": cheaper / n,
                        "wrong_of_cheaper": row["wrong"] / cheaper if cheaper else 0.0,
                        "output_tokens_saved": out, "output_tokens_saved_share": out / self.rec_out,
                        "unmeasured": sum(r["unmeasured"] for r in scored)})
        else:
            cuts = collections.Counter(r["cut"] for r in scored if r["cut"] is not None)
            cut = sum(count for choice, count in cuts.items() if choice != UNCUT)
            row.update({"cuts": dict(cuts), "cut": cut, "coverage": cut / n if n else 0.0,
                        "wrong_of_cut": row["wrong"] / cut if cut else 0.0,
                        "prompt_tokens_saved": sum(r["tokens"] for r in scored),
                        "prompt_tokens_saved_with_repeats": sum(r["tokens"] + r["repeat_tokens"] for r in scored),
                        "unjudged": sum(r["unjudged"] for r in scored)})
        return row

    def combined(self, step, router, trim):
        """The three decisions on the same turns, from per-turn results at one threshold each: a covered turn saves
        the turn (no Qwen request, so no router question); otherwise the router's saving; plus the trim saving."""
        wrong = errors = 0
        gross = matched = jeff = 0.0
        covered = 0
        for s, r, t in zip(step, router, trim, strict=True):
            turn_wrong = s["wrong"]
            turn_error = s["error"]
            jeff += s["jeff"]
            if s["outcome"] == "covered":
                covered += 1
                gross += s["saved"]
                matched += s["saved"]
            elif not s["wrong"] and not s["error"]:
                turn_wrong = r["wrong"]
                turn_error = r["error"]
                gross += r["saved"]
                matched += r["matched"]
                jeff += r["jeff"]
            if t is not None:
                turn_wrong = turn_wrong or t["wrong"]
                turn_error = turn_error or t["error"]
                gross += t["saved"]
                matched += t["saved"]
                jeff += t["jeff"]
            wrong += turn_wrong
            errors += turn_error
        return {"turns": len(step), "covered": covered, "errors": errors,
                **self.totals(wrong, len(step), gross, matched, jeff)}


# Two ways to count the router's seconds (router_outcome): the brief's measured wall-time difference, and the
# load-matched one. Each names the net-saving fields of a result row.
TIME_MODES = {"measured": ("net_saved_s", "net_saved_share_gen", "net_saved_share_gen_tool"),
              "load-matched": ("matched_net_saved_s", "matched_net_saved_share_gen", "matched_net_saved_share_gen_tool")}


def accuracy(predictions, questions):
    """Share of the answered questions whose most likely option is the label (None without any)."""
    answered = [q for q in questions if predictions[q["id"]] is not None]
    if not answered:
        return None
    return sum(most_likely(predictions[q["id"]])[0] == q["label"] for q in answered) / len(answered)


def best_at(rows, target, key="net_saved_s"):
    """The row with the largest saving among those whose wrong-choice rate is at most `target` (None if none); ties
    go to the higher threshold (fewer risky choices for the same saving)."""
    allowed = [row for row in rows if row["wrong_rate"] <= target]
    return max(allowed, key=lambda row: (row[key], row["threshold"])) if allowed else None


def score(args):
    meta, turns, sessions, questions = load_bundle(args.bundle)
    over_length = bundle_over_length(args.bundle)
    predictions = {d: read_predictions(getattr(args, d), questions[d], over_length[d]) for d in DECISIONS}
    scorer = Scorer(meta, turns, sessions, questions, predictions, args.decision_ms / 1000)
    thresholds = sorted(set(args.thresholds))
    per_turn = {d: {t: scorer.per_turn(d, t) for t in thresholds} for d in DECISIONS}
    tables = {d: [scorer.summary(d, t, per_turn[d][t]) for t in thresholds] for d in DECISIONS}
    common = [{"threshold": t, **scorer.combined(per_turn["step"][t], per_turn["router"][t], per_turn["trim"][t])}
              for t in thresholds]
    grid = []
    for ts in thresholds:
        for tr in thresholds:
            for tt in thresholds:
                grid.append({"thresholds": {"step": ts, "router": tr, "trim": tt},
                             **scorer.combined(per_turn["step"][ts], per_turn["router"][tr], per_turn["trim"][tt])})
    at_targets = {}
    for mode, (key, _, _) in TIME_MODES.items():
        at_targets[mode] = {}
        for target in TARGETS:
            entry = {d: best_at(tables[d], target, key) for d in DECISIONS}
            allowed = [row for row in grid if row["wrong_rate"] <= target]
            entry["combined"] = max(allowed, key=lambda row: (row[key], sum(row["thresholds"].values()))) \
                if allowed else None
            at_targets[mode][f"{target:.2f}"] = entry
    labels_router = [t["router"]["label"] for t in turns]
    routed = sum(t["rec_out"] if t["router"]["label"] == "xhigh" else t["router"]["levels"][t["router"]["label"]]["out"]
                 for t in turns)
    result = {
        "variant": args.name, "bundle": str(Path(args.bundle).resolve()), "bundle_turns_sha256": meta["turns_sha256"],
        "predictions": {d: {"path": str(Path(getattr(args, d)).resolve()), "sha256": sha256_file(getattr(args, d))}
                        for d in DECISIONS},
        "decision_ms": args.decision_ms, "thresholds": thresholds,
        "totals": {"turns": len(turns), "sessions": len(sessions), "trim_turns": len(scorer.trim_turns),
                   "gen_s": scorer.gen_s, "tool_s": scorer.tool_s, "scored_turns_rec_s": sum(t["rec_s"] for t in turns),
                   "rec_out": scorer.rec_out, "router_labels": dict(collections.Counter(labels_router)),
                   "label_routed_output_saved_share": 1 - routed / scorer.rec_out},
        "accuracy": {d: accuracy(predictions[d], questions[d]) for d in DECISIONS},
        "over_length": {d: len(over_length[d]) for d in DECISIONS},
        "tables": tables, "combined_common": common, "at_targets": at_targets,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    out.with_suffix(".md").write_text(markdown(result, meta))
    print(out.with_suffix(".md").read_text())


# ------------------------------------------------------------------------------------------------ reports


def pct(x):
    return f"{100 * x:.1f}%"


CAVEATS = (
    "Read these numbers as a ranking aid, not as a benchmark result:\n\n"
    "- Per turn only. Each turn is scored on its own, as if every earlier turn had gone as recorded. Knock-on effects "
    "(a wrong step or a too-cheap thinking level sending the session elsewhere, a turn Jeff takes moving Qwen's "
    "planning into the next turn, a shorter context delaying compaction) show only in the benchmark.\n"
    "- Strict single-reference labels. A choice counts as wrong when it differs from the one recorded Qwen action "
    "(step) or when the labeller found that level or cut not good enough against that one action (router, trim). "
    "Qwen at xhigh agrees with itself only about 40-50% of the time, so many 'wrong' choices would have been fine; "
    "the wrong-choice rates are upper bounds and the savings lower bounds of what the labels allow.\n"
    "- Tool time is not changed. Jeff runs the same commands; only Qwen's generation time (and prompt reading for "
    "trimming) is saved. The '% of generation + tool' column divides the same saving by the larger total.\n")


TIME_NOTE = ("Router seconds two ways. 'measured' (the brief's rule): recorded xhigh seconds minus the re-ask's wall "
             "seconds; the re-asks ran on busier servers (B200: about 40 output tokens per second against 74 while "
             "recording), so this charges the load difference to the cheaper level and can come out negative. "
             "'load-matched': the chosen level's output tokens at the recorded reply's own seconds per output token. "
             "Step and trim seconds are the same in both.\n")


def table(rows, decision):
    head = ("| threshold | decisions | wrong | wrong rate | over length | coverage | gross saved s | Jeff s | net saved s | "
            "% of gen | % of gen + tool |")
    extra = {"step": " steps taken |",
             "router": " net saved, load-matched (% of gen) | cheaper choices wrong | output tokens saved | unmeasured |",
             "trim": " cuts wrong | prompt tokens saved (once / with repeats) | unjudged |"}[decision]
    lines = [head + extra, "|" + "---:|" * (head.count("|") - 1 + extra.count("|"))]
    for row in rows:
        cells = [f"{row['threshold']:.2f}", str(row["decisions"]), str(row["wrong"]), pct(row["wrong_rate"]),
                 str(row["errors"]), pct(row["coverage"]), f"{row['gross_saved_s']:.0f}", f"{row['jeff_s']:.0f}",
                 f"{row['net_saved_s']:.0f}", pct(row["net_saved_share_gen"]), pct(row["net_saved_share_gen_tool"])]
        if decision == "step":
            cells.append(str(row["steps_taken"]))
        elif decision == "router":
            cells += [f"{row['matched_net_saved_s']:.0f} ({pct(row['matched_net_saved_share_gen'])})",
                      pct(row["wrong_of_cheaper"]), pct(row["output_tokens_saved_share"]), str(row["unmeasured"])]
        else:
            cells += [pct(row["wrong_of_cut"]),
                      f"{row['prompt_tokens_saved']} / {row['prompt_tokens_saved_with_repeats']}", str(row["unjudged"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def target_cells(entry, mode, with_wrong=True):
    _, share, share_tool = TIME_MODES[mode]
    cells = []
    for decision in DECISIONS:
        row = entry[decision]
        cells.append("-" if row is None else f"{pct(row[share])} @ {row['threshold']:.2f}"
                     + (f" (wrong {pct(row['wrong_rate'])})" if with_wrong else ""))
    row = entry["combined"]
    if row is None:
        return cells + ["-", "-"]
    th = row["thresholds"]
    return cells + [f"{pct(row[share])} @ {th['step']:.2f}/{th['router']:.2f}/{th['trim']:.2f}"
                    + (f" (wrong {pct(row['wrong_rate'])})" if with_wrong else ""), pct(row[share_tool])]


def markdown(result, meta):
    totals = result["totals"]
    out = [f"# Offline score: {result['variant']}\n", CAVEATS]
    out.append(f"Bundle: {result['bundle']} (splits {', '.join(meta['splits'])}): {totals['turns']} labelled turns of "
               f"{totals['sessions']} sessions ({totals['trim_turns']} with a long output). Recorded Qwen generation "
               f"time of these sessions {totals['gen_s'] / 3600:.2f} h, tool time {totals['tool_s'] / 3600:.2f} h; the "
               f"scored turns hold {totals['scored_turns_rec_s'] / 3600:.2f} h of generation. Router labels "
               f"{totals['router_labels']}; a perfect router saves {pct(totals['label_routed_output_saved_share'])} of "
               f"output tokens on these turns (routing-stats.md's 'routed saved'). Jeff time per decision "
               f"{result['decision_ms']} ms. Prefill: " + ", ".join(
                   f"{m} {1000 * r['seconds_per_token']:.3f} s per 1,000 uncached prompt tokens ({r['servers']} "
                   "servers, vLLM prefill counters)" for m, r in meta["prefill"].items()) + ".\n")
    out.append("Top-choice accuracy against the labels: " + ", ".join(
        f"{d} {'-' if a is None else pct(a)}" for d, a in result["accuracy"].items()) + ".\n")
    out.append("Questions over Jeff's length limit (jeff-serve refuses them; at run time the turn ends with a JeffFirst "
               "error; scored as no saving and counted under 'over length', not as wrong): " + ", ".join(
                   f"{d} {n}" for d, n in result["over_length"].items()) + ".\n")
    out.append(TIME_NOTE)
    for mode in TIME_MODES:
        out.append(f"## Best thresholds at fixed wrong-choice rates ({mode} router seconds)\n")
        out.append("Net time saved as a share of recorded Qwen generation time @ threshold; wrong-choice rate = "
                   "turns with a wrong choice / turns where that decision is made.\n")
        out.append("| wrong-choice rate at most | step | router | trim | all three (step/router/trim) | all three, % of "
                   "gen + tool |")
        out.append("|---|---|---|---|---|---|")
        for target, entry in result["at_targets"][mode].items():
            out.append(f"| {pct(float(target))} | " + " | ".join(target_cells(entry, mode)) + " |")
        out.append("")
    out.append("## All three decisions, one common threshold\n")
    out.append("| threshold | wrong turns | wrong rate | turns over length | covered by Jeff | gross saved s | Jeff s | "
               "net saved s | % of gen | % of gen + tool | net saved, load-matched (% of gen) |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["combined_common"]:
        out.append(f"| {row['threshold']:.2f} | {row['wrong']} | {pct(row['wrong_rate'])} | {row['errors']} | "
                   f"{row['covered']} | "
                   f"{row['gross_saved_s']:.0f} | {row['jeff_s']:.0f} | {row['net_saved_s']:.0f} | "
                   f"{pct(row['net_saved_share_gen'])} | {pct(row['net_saved_share_gen_tool'])} | "
                   f"{row['matched_net_saved_s']:.0f} ({pct(row['matched_net_saved_share_gen'])}) |")
    for decision in DECISIONS:
        out.append(f"\n## {decision}\n")
        out.append(table(result["tables"][decision], decision))
    return "\n".join(out) + "\n"


def compare(args):
    results = [json.loads(Path(p).read_text()) for p in args.scores]
    bundles = {r["bundle_turns_sha256"] for r in results}
    if len(bundles) != 1:
        raise ValueError("the scores come from different bundles; compare variants on the same held-out turns")
    out = ["# Offline scores: variants compared\n", CAVEATS,
           (f"Same held-out turns for all ({results[0]['totals']['turns']} turns, bundle {results[0]['bundle']}). Net "
            "time saved as a share of recorded Qwen generation time @ the best threshold whose wrong-choice rate is at "
            "most the target; the last column divides the all-three saving by generation + tool time. Ranked by the "
            "all-three saving.\n"), TIME_NOTE]
    for mode, (key, _, _) in TIME_MODES.items():
        for target in (f"{t:.2f}" for t in TARGETS):
            out.append(f"\n## Wrong-choice rate at most {pct(float(target))} ({mode} router seconds)\n")
            out.append("| variant | step | router | trim | all three (step/router/trim) | all three, % of gen + tool |")
            out.append("|---|---|---|---|---|---|")
            ranked = sorted(results, key=lambda r: -(r["at_targets"][mode][target]["combined"] or {key: -1e18})[key])
            for r in ranked:
                out.append(f"| {r['variant']} | " + " | ".join(target_cells(r["at_targets"][mode][target], mode, False))
                           + " |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))


# ------------------------------------------------------------------------------------------------ main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build", help="the held-out bundle: turns, sessions, Jeff's questions")
    b.add_argument("--routing-labels", nargs="+", required=True)
    b.add_argument("--trim-labels", nargs="+", required=True)
    b.add_argument("--trials", nargs="+", required=True, help="folders holding the trial folders (searched)")
    b.add_argument("--split", action="append", required=True, choices=HELD_OUT_SPLITS)
    b.add_argument("--scoring-tasks", help="the collection's hub/scoring-tasks.txt (for --split scoring)")
    b.add_argument("--splits", required=True, help="results/imitation/splits.json")
    b.add_argument("--task-sets", required=True, help="results/imitation/task-sets.json")
    b.add_argument("--training-tasks", required=True, help="results/imitation/training-tasks.json")
    b.add_argument("--build", action="append", required=True, help="scout tarball names whose trials are converted")
    b.add_argument("--prefill-metrics", action="append", required=True,
                   help="MACHINE=FILE: a saved /metrics page of one Qwen server of that replay machine (repeatable)")
    b.add_argument("--tokenizer", required=True, help="Qwen tokenizer.json (ceiling.py's session reader)")
    b.add_argument("--node", default="node")
    b.add_argument("--out", required=True)
    p = commands.add_parser("predict", help="probabilities from a running jeff-serve")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True, help="the adapter name, or jeff for the base")
    p.add_argument("--questions", required=True)
    p.add_argument("--out", required=True, help="predictions JSONL (appended to; a rerun continues)")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--busy-give-up", type=float, default=300)
    p.add_argument("--pause-ms", type=float, default=0, help="wait after each question (to leave the GPU to others)")
    p.add_argument("--limit", type=int, help="at most this many questions in this run")
    p.add_argument("--progress", type=int, default=100)
    v = commands.add_parser("over-length", help="record the questions jeff-serve refused as too long; write .fit files")
    v.add_argument("--bundle", required=True)
    for decision in DECISIONS:
        v.add_argument(f"--{decision}", required=True, help=f"complete predictions for questions-{decision}.jsonl")
    o = commands.add_parser("oracle", help="sanity predictions: the labels, or the fixed safe choice")
    o.add_argument("--kind", choices=("labels", "fixed"), required=True)
    o.add_argument("--decision", choices=DECISIONS, required=True)
    o.add_argument("--bundle", required=True)
    o.add_argument("--out", required=True)
    s = commands.add_parser("score", help="one variant's predictions -> tables")
    s.add_argument("--bundle", required=True)
    s.add_argument("--name", required=True)
    for decision in DECISIONS:
        s.add_argument(f"--{decision}", required=True, help=f"predictions for questions-{decision}.jsonl")
    s.add_argument("--decision-ms", type=float, required=True, help="Jeff's time per decision (178 measured)")
    s.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    s.add_argument("--out", required=True, help="output path without suffix (.md and .json are written)")
    c = commands.add_parser("compare", help="several variants' score JSON -> one table")
    c.add_argument("--scores", nargs="+", required=True)
    c.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"build": build, "predict": predict, "over-length": over_length_command, "oracle": oracle, "score": score,
     "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
