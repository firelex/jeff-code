"""Tonight's Jeff training files (2026-10-04) from the cut exports, in the Example format jeff-dev's train.py reads.

Commands:
  merge-rows --out ROWS IN...   one stage 3 rows file from the hosts' conversions (no session may appear in two inputs)
  build ...                     the training files below, each a folder with train.jsonl, development.jsonl and
                                temperature.jsonl, plus manifest.json (rows per file and stage, label shares)

Inputs of build (folders written by jeff_prompt.py fit-examples, each with train/development/temperature.jsonl):
  --stage1, --stage2, --stage3 (step rows), --router, --trim. Example ids are "<source>:<stage>:...": the second field
  is the data stage train.py's --keep-stage-order keeps in file order (1, 2, 3; router rows "route", trim rows "trim").

Training files:
  step-curriculum  stage 1 sampled down to --stage1-fraction x stage 3's training rows (whole task groups, taken in
                   the order of sha256("<seed>:<family>"), a group is skipped when it would overshoot), then all of
                   stage 2, then stage 3 shown --stage3-times times (copy k gets ":copy<k>" added to its id; all
                   copies are in the one stage 3 block, which train.py shuffles). Stages in that order in the file.
  step-stage3      stage 3 shown --stage3-times times, only.
  router           the router rows only.  trim: the trim rows only. (No replay rows: owner decision.)
  full             step-curriculum's rows, with the router and trim rows (once each) added to the stage 3 block; their
                   ids get the stage field "3" ("own:route:..." becomes "own:3:route:...").
Development and temperature rows: stage 3's for the two step files, the router's and trim's own, and all three for
full; each decision's are capped at --eval-rows rows (whole task families, seeded), because train.py scores all of them
at every evaluation. No row of a held-out task or of a scoring task (--scoring-tasks) may be in any file: that is an error.

Run (stdlib only): uv run --no-project python tonight_training_files.py build --stage1 ... --out DIR
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

SPLITS = ("train", "development", "temperature")
ROUTER_LABELS = ("off", "low", "medium", "xhigh")
TRIM_LABELS = ("all", "last200", "last40", "first40", "first20last20")


def stage_of(identifier: str) -> str:
    fields = identifier.split(":")
    if len(fields) < 3 or not fields[1]:
        raise ValueError(f"row id {identifier!r} has no stage field (source:stage:...)")
    return fields[1]


def read_lines(path: Path) -> Iterator[tuple[str, dict]]:
    """Each line of an Example file with its parsed row (lines end only at "\\n")."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield (line if line.endswith("\n") else line + "\n"), json.loads(line)


def merge_rows(out: Path, inputs: list[Path]) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    owner: dict[str, Path] = {}
    counts: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as target:
        for path in inputs:
            counts[str(path)] = 0
            seen_here: set[str] = set()
            for line, row in read_lines(path):
                session = row["session"]
                if session not in seen_here:
                    if session in owner:
                        raise ValueError(f"session {session} is in both {owner[session]} and {path}")
                    owner[session] = path
                    seen_here.add(session)
                target.write(line)
                counts[str(path)] += 1
    return {"rows": counts, "sessions": len(owner)}


def held_out_tasks(task_sets_path: Path, scoring_path: Path) -> tuple[set[str], set[str]]:
    """The hub task ids of every held-out task in task-sets.json, and the scoring task ids (all must be held out)."""
    data = json.loads(task_sets_path.read_text())
    held: set[str] = set()
    for entry in data["datasets"].values():
        hub = entry.get("hub")
        if hub is None:
            continue
        held.update(f"{hub['name']}:{task}" for task in entry["held_out"])
    scoring = {line.strip() for line in scoring_path.read_text().splitlines() if line.strip()}
    if not scoring:
        raise ValueError(f"{scoring_path} lists no scoring task")
    return held, scoring


def check_tasks(rows_by_file: dict[str, list[dict]], held: set[str], scoring: set[str], training_tb2: set[str]) -> None:
    for name, rows in rows_by_file.items():
        for row in rows:
            task = row["source"]["task"]
            if task in scoring:
                raise ValueError(f"{name}: row {row['id']} is on the scoring task {task}")
            if task in held:
                raise ValueError(f"{name}: row {row['id']} is on the held-out task {task}")
            if row["source"]["stage"] in (2, 3) and ":" not in task and task not in training_tb2:
                raise ValueError(f"{name}: row {row['id']} is on {task}, not a Terminal-Bench 2.0 training task")


def sample_groups(path: Path, target: int, seed: int) -> tuple[set[int], dict]:
    """Line numbers of the stage 1 rows kept: whole families in the order of sha256("<seed>:<family>"), skipping a
    family that would overshoot the target, until the target is reached."""
    by_family: dict[str, list[int]] = {}
    for number, (_, row) in enumerate(read_lines(path)):
        by_family.setdefault(row["family"], []).append(number)
    order = sorted(by_family, key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).hexdigest())
    kept: set[int] = set()
    families = 0
    for family in order:
        if len(kept) == target:
            break
        lines = by_family[family]
        if len(kept) + len(lines) > target:
            continue
        kept.update(lines)
        families += 1
    if len(kept) != target:
        raise ValueError(f"stage 1 sampling reached {len(kept)} rows, not the target {target}")
    return kept, {"target": target, "rows": len(kept), "families": families, "families_available": len(by_family),
                  "rows_available": sum(len(lines) for lines in by_family.values())}


def cap_by_family(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """At most `limit` rows, whole families in the order of sha256("<seed>:<family>"), a family skipped when it would
    overshoot; the rows keep their file order. Every evaluation scores all development and temperature rows, so their
    number sets the evaluation time."""
    by_family: dict[str, int] = Counter(row["family"] for row in rows)
    chosen: set[str] = set()
    total = 0
    for family in sorted(by_family, key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).hexdigest()):
        if total + by_family[family] <= limit:
            chosen.add(family)
            total += by_family[family]
    if not chosen:
        raise ValueError(f"no family fits in {limit} evaluation rows")
    return [row for row in rows if row["family"] in chosen]


def restaged(row: dict, stage: str) -> dict:
    fields = row["id"].split(":")
    return {**row, "id": ":".join([fields[0], stage, *fields[1:]])}


def shown(rows: list[dict], times: int) -> list[dict]:
    """The rows `times` times: copy k (k >= 2) gets ":copy<k>" added to its id (train.py refuses repeated ids)."""
    if times < 1:
        raise ValueError(f"stage 3 must be shown at least once, not {times} times")
    return rows + [{**row, "id": f"{row['id']}:copy{k}"} for k in range(2, times + 1) for row in rows]


def label_summary(rows: list[dict]) -> dict:
    """Rows per stage, and the label shares: step tool rows (act vs hand over), step argument rows (an option vs
    none_of_these), router and trim rows (each label)."""
    out: dict = {"rows": len(rows), "rows_by_stage": dict(Counter(stage_of(row["id"]) for row in rows))}
    groups: dict[str, Counter] = {}
    for row in rows:
        label = row["label"]
        if "level" in row["source"]:
            level = row["source"]["level"]
            if label not in ("hand_over", "none_of_these", "show_more"):
                label = "act" if level == "tool" else "option"
            kind = f"step-{level}"
        elif label in ROUTER_LABELS:
            kind = "route"
        elif label in TRIM_LABELS:
            kind = "trim"
        else:
            raise ValueError(f"row {row['id']}: label {label!r} is neither a step, router nor trim label")
        groups.setdefault(kind, Counter())[label] += 1
    out["labels"] = {
        kind: {label: {"rows": count, "share": round(count / sum(counter.values()), 4)} for label, count in counter.most_common()}
        for kind, counter in sorted(groups.items())
    }
    return out


def write_folder(folder: Path, splits: dict[str, list[dict]]) -> dict:
    folder.mkdir(parents=True)
    out = {}
    for name in SPLITS:
        rows = splits[name]
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{folder}/{name}.jsonl would repeat a row id")
        if not rows:
            raise ValueError(f"{folder}/{name}.jsonl would be empty")
        path = folder / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        out[name] = {**label_summary(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # train.py's --keep-stage-order: each stage one contiguous block.
    order: list[str] = []
    for row in splits["train"]:
        stage = stage_of(row["id"])
        if not order or order[-1] != stage:
            if stage in order:
                raise ValueError(f"{folder}: stage {stage} appears in two blocks")
            order.append(stage)
    families = {name: {(str(row["source"].get("dataset", row["suite"])), row["family"]) for row in splits[name]} for name in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and families[a] & families[b]:
                raise ValueError(f"{folder}: a task family is in both {a} and {b}: {sorted(families[a] & families[b])[:3]}")
    out["stage_order"] = order
    steps = -(-len(splits["train"]) // 64)
    out["optimizer_steps_at_64"] = steps
    out["eval_every_for_20"] = max(4, steps // 20)
    return out


def load(folder: Path, name: str) -> list[dict]:
    return [row for _, row in read_lines(folder / f"{name}.jsonl")]


def build(args: argparse.Namespace) -> dict:
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    stage2 = {name: load(args.stage2, name) for name in SPLITS}
    stage3 = {name: load(args.stage3, name) for name in SPLITS}
    router = {name: load(args.router, name) for name in SPLITS}
    trim = {name: load(args.trim, name) for name in SPLITS}
    for rows, stage in ((stage2, "2"), (stage3, "3"), (router, "route"), (trim, "trim")):
        for name in SPLITS:
            for row in rows[name]:
                if stage_of(row["id"]) != stage:
                    raise ValueError(f"row {row['id']} is in the stage {stage} input")
    if not 0 < args.stage1_fraction <= 1:
        raise ValueError(f"--stage1-fraction must be in (0, 1], not {args.stage1_fraction}")
    target = round(args.stage1_fraction * len(stage3["train"]))
    kept, sampling = sample_groups(args.stage1 / "train.jsonl", target, args.seed)
    stage1_train = [row for number, (_, row) in enumerate(read_lines(args.stage1 / "train.jsonl")) if number in kept]
    if any(stage_of(row["id"]) != "1" for row in stage1_train):
        raise ValueError("a row of the stage 1 input is not of stage 1")
    held, scoring = held_out_tasks(args.task_sets, args.scoring_tasks)
    training_tb2 = set(json.loads(args.training_tasks.read_text())["training"])
    check_tasks({"stage2": [r for n in SPLITS for r in stage2[n]], "stage3": [r for n in SPLITS for r in stage3[n]],
                 "router": [r for n in SPLITS for r in router[n]], "trim": [r for n in SPLITS for r in trim[n]]},
                held, scoring, training_tb2)
    stage3_shown = shown(stage3["train"], args.stage3_times)
    curriculum = stage1_train + stage2["train"] + stage3_shown
    full_train = curriculum + [restaged(row, "3") for row in router["train"] + trim["train"]]
    evaluation = ("development", "temperature")
    full_eval = {name: {"stage3": len(stage3[name]), "router": len(router[name]), "trim": len(trim[name])} for name in evaluation}
    for rows in (stage3, router, trim):
        for name in evaluation:
            rows[name] = cap_by_family(rows[name], args.eval_rows, args.seed)
    step_eval = {name: stage3[name] for name in evaluation}
    files = {
        "step-curriculum": {"train": curriculum, **step_eval},
        "step-stage3": {"train": stage3_shown, **step_eval},
        "router": router,
        "trim": trim,
        "full": {"train": full_train,
                 **{name: stage3[name] + router[name] + trim[name] for name in ("development", "temperature")}},
    }
    manifest = {"seed": args.seed, "stage3_times": args.stage3_times, "stage1_fraction": args.stage1_fraction,
                "stage1_sampling": sampling, "eval_rows_cap": args.eval_rows,
                "evaluation_rows_before_cap": full_eval,
                "inputs": {key: str(getattr(args, key)) for key in ("stage1", "stage2", "stage3", "router", "trim")},
                "files": {}}
    args.out.mkdir(parents=True)
    for name, splits in files.items():
        manifest["files"][name] = write_folder(args.out / name, splits)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge-rows")
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("inputs", type=Path, nargs="+")
    make = commands.add_parser("build")
    for name in ("stage1", "stage2", "stage3", "router", "trim"):
        make.add_argument(f"--{name}", type=Path, required=True)
    make.add_argument("--task-sets", type=Path, required=True)
    make.add_argument("--training-tasks", type=Path, required=True, help="results/imitation/training-tasks.json")
    make.add_argument("--scoring-tasks", type=Path, required=True)
    make.add_argument("--seed", type=int, required=True)
    make.add_argument("--stage3-times", type=int, required=True, help="how many times each stage 3 step row is shown")
    make.add_argument("--stage1-fraction", type=float, required=True,
                      help="stage 1 rows sampled, as a fraction of stage 3's train rows (shown once)")
    make.add_argument("--eval-rows", type=int, required=True,
                      help="at most this many development and this many temperature rows per decision (whole task families)")
    make.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "merge-rows":
        print(json.dumps(merge_rows(args.out, args.inputs)))
        return
    manifest = build(args)
    print(json.dumps({name: {"train": info["train"]["rows_by_stage"], "development": info["development"]["rows"],
                             "temperature": info["temperature"]["rows"]} for name, info in manifest["files"].items()}))


if __name__ == "__main__":
    main()
