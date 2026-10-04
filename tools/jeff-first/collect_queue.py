"""Shared work queue for the hub-dataset collection: every training task of the Harbor hub datasets in
results/imitation/task-sets.json (Terminal-Bench 2.0 is finished and not queued) is handed to one stream at a time,
and no task runs again while another has run fewer times.

Hosts cannot reach each other (B200, casdgx01 and datigator have no common network or shared disk), so the tasks are
split between hosts once, and each host keeps its own queue that its streams claim from under a file lock:
- `plan` gives every task to exactly one host, in proportion to the host's streams (a host marked "small" gets only
  tasks with at most 4 CPUs, 8 GB memory and no sidecar services). The split depends only on task-sets.json,
  the inventory, the host list and the seed, never on session counts, so every host computes the same split.
  The host's queue is ordered: tasks without a finished session first (in seeded random order, so datasets mix), then
  tasks with finished sessions, fewest first.
- `claim` hands a stream the task with the fewest runs (finished before the queue started + claims since) among
  tasks no other stream is running, earliest in the queue on a tie. When every task is running it says so (exit 3).
  A claim held by a stream process that has died is freed, so a restarted stream does not block its task.
- `release` ends a stream's claim after its session.
- `count` counts finished sessions per task id in collection folders (finished = result.json and a non-empty
  jeff-first-trace.jsonl), for `plan --counts`.

Usage:
    python3 collect_queue.py count OUT.json FOLDER...
    python3 collect_queue.py plan TASK_SETS INVENTORY STATE_DIR HOST --host NAME:STREAMS[:small] ... \\
        [--counts COUNTS.json ...] [--seed N]
    python3 collect_queue.py claim STATE_DIR STREAM PID        prints "<task id> <run number>"; exit 3: all running
    python3 collect_queue.py release STATE_DIR STREAM TASK
"""

import argparse
import fcntl
import json
import os
import random
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from task_source import load_task_sets

SEED = 20261004
ALL_RUNNING = None
SMALL_CPUS, SMALL_MEMORY_MB = 4, 8192


def _small(row: dict) -> bool:
    """At most 4 CPUs and 8 GB, no sidecars; a task that states no CPUs or memory is not small (unknown needs)."""
    if row["cpus"] is None or row["memory_mb"] is None:
        return False
    return row["cpus"] <= SMALL_CPUS and row["memory_mb"] <= SMALL_MEMORY_MB and not row["compose_services"]


def plan(task_sets: Path, inventory: Path, counts: dict[str, int], hosts: list[tuple[str, int, bool]], seed: int = SEED) -> dict[str, list[str]]:
    """Each host's queue (see the module docstring). hosts: (name, streams, small only)."""
    if len({name for name, _, _ in hosts}) != len(hosts) or any(streams <= 0 for _, streams, _ in hosts):
        raise ValueError(f"hosts need distinct names and a positive number of streams: {hosts}")
    sets = load_task_sets(task_sets, inventory)
    tasks = sets.training_ids()
    rows = {}
    for task in tasks:
        resolved = sets.resolve(task)
        folder = sets.datasets[resolved.dataset_spec.split("@")[0]]["folder"]
        rows[task] = sets.inventory[(folder, task.split(":")[1])]
    order = sorted(tasks)
    random.Random(seed).shuffle(order)
    eligible = {task: [name for name, _, small in hosts if not small or _small(rows[task])] for task in order}
    assigned: dict[str, list[str]] = {name: [] for name, _, _ in hosts}
    streams = {name: n for name, n, _ in hosts}
    # Tasks with fewer possible hosts first, so the small hosts' share is filled from the tasks they can take.
    for task in sorted(order, key=lambda t: len(eligible[t])):
        if not eligible[task]:
            raise ValueError(f"no host can run {task} (only small hosts given)")
        host = min(eligible[task], key=lambda name: (len(assigned[name]) + 1) / streams[name])
        assigned[host].append(task)
    position = {task: i for i, task in enumerate(order)}
    return {name: sorted(queue, key=lambda t: (counts.get(t, 0), position[t])) for name, queue in assigned.items()}


@contextmanager
def _locked(state: Path):
    with open(state / "lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        claims_path = state / "claims.json"
        claims = json.loads(claims_path.read_text()) if claims_path.exists() else {"claims": {}, "running": {}}
        yield claims
        tmp = state / "claims.json.tmp"
        tmp.write_text(json.dumps(claims, indent=1))
        os.replace(tmp, claims_path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(state: Path, stream: str, pid: int) -> tuple[str, int] | None:
    queue = json.loads((state / "queue.json").read_text())
    with _locked(state) as claims:
        running = claims["running"]
        for name in [name for name, held in running.items() if not _alive(held["pid"])]:
            del running[name]
        if stream in running:
            raise ValueError(f"stream {stream} already runs {running[stream]['task']}")
        busy = {held["task"] for held in running.values()}
        runs = {task: queue["finished_before"].get(task, 0) + claims["claims"].get(task, 0) for task in queue["order"]}
        free = [task for task in queue["order"] if task not in busy]
        if not free:
            return ALL_RUNNING
        task = min(free, key=lambda t: runs[t])  # min keeps the earliest in queue order on a tie
        claims["claims"][task] = claims["claims"].get(task, 0) + 1
        running[stream] = {"task": task, "pid": pid}
        return task, runs[task] + 1


def release(state: Path, stream: str, task: str) -> None:
    with _locked(state) as claims:
        held = claims["running"].get(stream)
        if held is None or held["task"] != task:
            raise ValueError(f"stream {stream} does not run {task} (it holds {held})")
        del claims["running"][stream]


def _task_id(config: dict) -> str:
    task = config["task"]
    if task.get("path") is not None:
        return task["path"]
    return f"{task['source']}:{task['name'].split('/', 1)[1]}"


def count_finished(folders: list[Path]) -> dict[str, int]:
    counts: Counter = Counter()
    for folder in folders:
        for result in folder.glob("*/round*/*/*/result.json"):
            trial = result.parent
            trace = trial / "agent" / "jeff-first-trace.jsonl"
            if trace.exists() and trace.stat().st_size > 0:
                counts[_task_id(json.loads((trial / "config.json").read_text()))] += 1
    return dict(sorted(counts.items()))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("count")
    c.add_argument("out", type=Path)
    c.add_argument("folders", type=Path, nargs="*")
    p = sub.add_parser("plan")
    p.add_argument("task_sets", type=Path)
    p.add_argument("inventory", type=Path)
    p.add_argument("state", type=Path)
    p.add_argument("this_host")
    p.add_argument("--host", action="append", required=True, help="NAME:STREAMS or NAME:STREAMS:small")
    p.add_argument("--counts", type=Path, action="append", default=[])
    p.add_argument("--seed", type=int, default=SEED)
    k = sub.add_parser("claim")
    k.add_argument("state", type=Path)
    k.add_argument("stream")
    k.add_argument("pid", type=int)
    r = sub.add_parser("release")
    r.add_argument("state", type=Path)
    r.add_argument("stream")
    r.add_argument("task")
    args = parser.parse_args(argv)
    if args.command == "count":
        args.out.write_text(json.dumps(count_finished(args.folders), indent=1) + "\n")
    elif args.command == "plan":
        if (args.state / "queue.json").exists():
            raise SystemExit(f"{args.state} already holds a queue; planning again would forget its claims")
        hosts = []
        for spec in args.host:
            parts = spec.split(":")
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "small"):
                raise SystemExit(f"--host {spec!r}: expected NAME:STREAMS or NAME:STREAMS:small")
            hosts.append((parts[0], int(parts[1]), len(parts) == 3))
        counts: Counter = Counter()
        for path in args.counts:
            counts.update(json.loads(path.read_text()))
        queues = plan(args.task_sets, args.inventory, dict(counts), hosts, args.seed)
        if args.this_host not in queues:
            raise SystemExit(f"host {args.this_host} is not among --host {args.host}")
        order = queues[args.this_host]
        args.state.mkdir(parents=True, exist_ok=True)
        (args.state / "queue.json").write_text(json.dumps({"host": args.this_host, "hosts": args.host, "seed": args.seed, "order": order, "finished_before": {t: counts[t] for t in order if counts.get(t)}}, indent=1) + "\n")
        print(f"{args.this_host}: {len(order)} tasks of {sum(len(q) for q in queues.values())}; " + ", ".join(f"{h} {len(q)}" for h, q in queues.items()))
    elif args.command == "claim":
        got = claim(args.state, args.stream, args.pid)
        if got is ALL_RUNNING:
            sys.exit(3)
        print(f"{got[0]} {got[1]}")
    else:
        release(args.state, args.stream, args.task)


if __name__ == "__main__":
    main(sys.argv[1:])
