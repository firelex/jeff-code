"""The work queue of tonight's four-arm evaluation (2026-10-04): every (task, attempt) is a block of four sessions, one per
arm, and a whole block runs on one Qwen server, so every arm gets the same share of every server; blocks are taken in
a fixed shuffled order, so every arm also gets the same share of every hour.

plan    writes one queue per host from the task list: attempt 1 of every task first (in an order shuffled with the
        seed), then attempt 2, then attempt 3; the arm order inside a block rotates with the block number. Tasks whose
        longest possible session (agent time limit x multiplier + verifier time limit + 20 minutes for image and setup)
        does not fit before the deadline of a host with one (--deadline-host) go to the other host; the remaining blocks
        are split in proportion to the hosts' stream counts.
claim   gives a stream on one server its next session: the next arm of a block already open on that server, else the
        first block not yet started (it opens on that server). With --deadline, a session starts only if its longest
        possible duration ends before the deadline; a block whose task does not fit is skipped and stays unstarted.
        Prints "BLOCK ARM TASK ATTEMPT BENCHMARK" (TB2 blocks: terminal-bench-2). Exit 3: nothing to start now; exit 4: nothing left that can ever start here.
followon writes follow-on queues (QUEUE_NAME-HOST.json): for each benchmark in the order given, its task ids shuffled
        with the seed, one block per task for the given attempt number, one block per task (block ids PREFIX + number), arm order rotating as in plan;
        blocks dealt to the hosts in proportion to their stream counts in queue order. Agent time per task as
        task_source.py gives it (at most 15 minutes x the multiplier). --first-block-to HOST puts the first block of
        the benchmark named by --first-block-of into a one-block queue (queue-check-HOST.json) for a check run.
finish  records a session's end (its run_phase0.sh exit status); prints "complete" when it was the block's last session.
rerun   (see eval_rerun.py) blocks may hold fewer arms and carry server_pin: such a block opens only on that server.
close   no session of the queue starts at or after TIME (claims then exit 4).
cut     marks the queue's still-running sessions as cut (status LABEL, cut.txt in the folder) and prints their folders.
upcoming prints the tasks of the next N blocks not yet started, one per line (for image prefetching).
take-unfit removes the unstarted blocks whose longest possible session no longer ends by --deadline and prints them as
        JSON (to append them to the other host's queue).
append  adds the blocks of a take-unfit JSON file to a queue (ids kept, queue kept in block id order).
move    moves unstarted blocks from one host's queue to another's (both files local), for rebalancing by hand.
status  prints the queue's counts.

The queue file is changed only under an exclusive lock (QUEUE.lock).
"""

import argparse
import datetime as dt
import fcntl
import json
import random
import sys
from pathlib import Path

ARMS = ("a1-baseline", "a2-off-guard", "a3-jeff07", "a4-jeff06")
SETUP_MARGIN_S = 1200


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def longest_s(task: dict, multiplier: float) -> float:
    return task["agent_timeout_s"] * multiplier + task["verifier_timeout_s"] + SETUP_MARGIN_S


def plan(args: argparse.Namespace) -> None:
    tasks = json.loads(Path(args.tasks).read_text())  # {name: {agent_timeout_s, verifier_timeout_s}}
    names = sorted(tasks)
    if len(names) != args.expect_tasks:
        raise SystemExit(f"{args.tasks} holds {len(names)} tasks, expected {args.expect_tasks}")
    hosts = dict(h.split(":") for h in args.host)
    streams = {h: int(n) for h, n in hosts.items()}
    rng = random.Random(args.seed)
    blocks = []
    for attempt in range(1, args.attempts + 1):
        order = names[:]
        rng.shuffle(order)
        for task in order:
            k = len(blocks)
            arms = list(ARMS[k % 4 :] + ARMS[: k % 4])
            blocks.append({"block": f"b{k:03d}", "task": task, "attempt": attempt, "arms": arms})
    deadline_host, deadline = None, None
    if args.deadline_host:
        deadline_host, stamp = args.deadline_host.split("=", 1)
        deadline = dt.datetime.fromisoformat(stamp)
        start = dt.datetime.fromisoformat(args.start)
    other = [h for h in streams if h != deadline_host]
    queues: dict[str, list] = {h: [] for h in streams}
    total = sum(streams.values())
    free = []
    for b in blocks:
        task = tasks[b["task"]]
        if deadline_host and start + dt.timedelta(seconds=longest_s(task, args.multiplier)) > deadline:
            if len(other) != 1:
                raise SystemExit("a task that cannot fit the deadline host needs exactly one other host")
            queues[other[0]].append(b)
        else:
            free.append(b)
    # Fill each host up to its share of all blocks, taking the free blocks in queue order (weighted round robin).
    target = {h: len(blocks) * streams[h] / total for h in streams}
    for b in free:
        host = min(streams, key=lambda h: (len(queues[h]) + 1) / target[h])
        queues[host].append(b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for host, qblocks in queues.items():
        qblocks.sort(key=lambda b: b["block"])
        for b in qblocks:
            b.update(server=None, opened=None, units={})
        state = {
            "host": host,
            "multiplier": args.multiplier,
            "seed": args.seed,
            "tasks": tasks,
            "blocks": qblocks,
        }
        path = out / f"queue-{host}.json"
        if path.exists():
            raise SystemExit(f"{path} exists; delete it by hand to plan again")
        path.write_text(json.dumps(state, indent=1))
        print(f"{host}: {len(qblocks)} blocks, {4 * len(qblocks)} sessions -> {path}")


class Locked:
    def __init__(self, path: str):
        self.path = Path(path)
        self.lock = open(str(self.path) + ".lock", "w")

    def __enter__(self) -> dict:
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.state = json.loads(self.path.read_text())
        return self.state

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.path)

    def __exit__(self, *exc) -> None:
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def followon(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task_source import TB2_DEFAULT_AGENT_SEC, load_task_sets

    sets = load_task_sets(Path(args.task_sets), Path(args.inventory))
    hosts = {h: int(n) for h, n in (x.split(":") for x in args.host)}
    total_streams = sum(hosts.values())
    tasks: dict[str, dict] = {}
    blocks = []
    counter = 0
    for spec in args.benchmark:
        name, prefix = spec.split("=")
        entry = next(e for e in sets.datasets.values() if e["folder"] == name)
        hub = entry["hub"]["name"]
        ids = sorted(f"{hub}:{t}" for t in entry["held_out"])
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        for k, task_id in enumerate(ids):
            task = sets.resolve(task_id)  # raises on a task that cannot run (excluded, GPU, MCP, unknown)
            row = sets.inventory[(entry["folder"], task_id.split(":", 1)[1])]
            tasks[task_id] = {
                "agent_timeout_s": min(task.agent_timeout_sec, TB2_DEFAULT_AGENT_SEC),
                "verifier_timeout_s": float(row["verifier_timeout_sec"]),
                "image": task.image,
                "benchmark": name,
            }
            chosen = tuple(args.arms.split(",")) if args.arms else ARMS
            if not set(chosen) <= set(ARMS):
                raise SystemExit(f"--arms must be from {ARMS}, got {chosen}")
            arms = list(chosen[counter % len(chosen) :] + chosen[: counter % len(chosen)])
            blocks.append({"block": f"{prefix}{k:03d}", "benchmark": name, "task": task_id, "attempt": args.attempt, "arms": arms})
            counter += 1
    queues: dict[str, list] = {h: [] for h in hosts}
    check = None
    if args.first_block_to:
        check = next(b for b in blocks if b["benchmark"] == args.first_block_of)
        blocks.remove(check)
    target = {h: len(blocks) * n / total_streams for h, n in hosts.items()}
    for b in blocks:
        host = min(hosts, key=lambda h: (len(queues[h]) + 1) / target[h])
        queues[host].append(b)
    out = Path(args.out)
    files = {f"{args.queue_name}-{h}.json": (h, q) for h, q in queues.items()}
    if check:
        files[f"queue-check-{args.first_block_to}.json"] = (args.first_block_to, [check])
    for fname, (host, qblocks) in files.items():
        path = out / fname
        if path.exists():
            raise SystemExit(f"{path} exists; delete it by hand to plan again")
        for b in qblocks:
            b.update(server=None, opened=None, units={})
        state = {"host": host, "multiplier": args.multiplier, "seed": args.seed, "tasks": tasks, "blocks": qblocks}
        path.write_text(json.dumps(state, indent=1))
        counts = {}
        for b in qblocks:
            counts[b["benchmark"]] = counts.get(b["benchmark"], 0) + 1
        print(f"{path}: {len(qblocks)} blocks {counts}")


def close(args: argparse.Namespace) -> None:
    queue = Locked(args.queue)
    with queue as state:
        dt.datetime.fromisoformat(args.time)  # must parse
        state["no_starts_after"] = args.time
        queue.save()
        print(f"{args.queue}: no session starts at or after {args.time}")


def cut(args: argparse.Namespace) -> None:
    """Marks every session of the queue that is still running (no finish recorded) as cut: status LABEL in the queue
    and cut.txt in its folder (EVAL_DIR/runs/ARM/TASK/attemptK), so the summary leaves it out. Prints the folders."""
    queue = Locked(args.queue)
    with queue as state:
        for b in state["blocks"]:
            for arm, unit in b["units"].items():
                if unit["finished"] is not None:
                    continue
                folder = Path(args.eval_dir) / "runs" / arm / b["task"].replace("/", ".").replace(":", ".") / f"attempt{b['attempt']}"
                if not folder.is_dir():
                    raise SystemExit(f"{folder} missing for running {b['block']} {arm}")
                (folder / "cut.txt").write_text(f"{args.label}: still running when the B200 runs were stopped\n")
                unit.update(finished=now().isoformat(), status=args.label)
                print(folder)
        queue.save()


def upcoming(args: argparse.Namespace) -> None:
    state = json.loads(Path(args.queue).read_text())
    for b in [b for b in state["blocks"] if b["server"] is None][: args.n]:
        print(b["task"])


def take_unfit(args: argparse.Namespace) -> None:
    queue = Locked(args.queue)
    with queue as state:
        deadline = dt.datetime.fromisoformat(args.deadline)
        t = now()
        unfit = [
            b
            for b in state["blocks"]
            if b["server"] is None
            and t + dt.timedelta(seconds=longest_s(state["tasks"][b["task"]], state["multiplier"])) > deadline
        ]
        state["blocks"] = [b for b in state["blocks"] if b not in unfit]
        tasks = {b["task"]: state["tasks"][b["task"]] for b in unfit}
        Path(args.out).write_text(json.dumps({"tasks": tasks, "blocks": unfit}))
        queue.save()
        print(len(unfit))


def append(args: argparse.Namespace) -> None:
    moving = json.loads(Path(args.file).read_text())
    queue = Locked(args.queue)
    with queue as state:
        have = {b["block"] for b in state["blocks"]}
        clash = have & {b["block"] for b in moving["blocks"]}
        if clash:
            raise SystemExit(f"{args.queue} already holds blocks {sorted(clash)}")
        state["tasks"].update(moving["tasks"])
        state["blocks"] = sorted(state["blocks"] + moving["blocks"], key=lambda b: b["block"])
        queue.save()
        print(f"appended {len(moving['blocks'])} blocks")


def claim(args: argparse.Namespace) -> int:
    queue = Locked(args.queue)
    with queue as state:
        deadline = dt.datetime.fromisoformat(args.deadline) if args.deadline else None
        t = now()
        if state.get("no_starts_after") and t >= dt.datetime.fromisoformat(state["no_starts_after"]):
            return 4  # the queue is closed for new sessions (eval_queue.py close)
        cap_file = Path(args.queue).parent / "streams-per-server"
        if cap_file.exists():
            # A cap on concurrent sessions per Qwen server: stream ...-sK with K above it takes nothing (it ends after
            # its current session), so the host runs at most cap x servers sessions.
            k = int(args.stream.rsplit("-s", 1)[1])
            if k > int(cap_file.read_text().strip()):
                return 4

        def fits(task: str) -> bool:
            if deadline is None:
                return True
            return t + dt.timedelta(seconds=longest_s(state["tasks"][task], state["multiplier"])) <= deadline

        def take(b: dict) -> int:
            arm = next(a for a in b["arms"] if a not in b["units"])
            b["units"][arm] = {"stream": args.stream, "started": t.isoformat(), "finished": None, "status": None}
            queue.save()
            print(b["block"], arm, b["task"], b["attempt"], b.get("benchmark", "terminal-bench-2"))
            return 0

        def mine(b: dict) -> bool:  # a block pinned to a server (a rerun) opens only there
            return b.get("server_pin") in (None, args.server)

        for b in state["blocks"]:
            if b["server"] == args.server and len(b["units"]) < len(b["arms"]) and fits(b["task"]):
                return take(b)
        for b in state["blocks"]:
            if b["server"] is None and mine(b) and fits(b["task"]):
                b["server"], b["opened"] = args.server, t.isoformat()
                return take(b)
        open_here = [b for b in state["blocks"] if b["server"] == args.server and len(b["units"]) < len(b["arms"])]
        unstarted = [b for b in state["blocks"] if b["server"] is None and mine(b)]
        if deadline is not None and not any(fits(b["task"]) for b in open_here + unstarted):
            return 4
        return 4 if not open_here and not unstarted else 3


def finish(args: argparse.Namespace) -> None:
    queue = Locked(args.queue)
    with queue as state:
        b = next(b for b in state["blocks"] if b["block"] == args.block)
        unit = b["units"][args.arm]
        if unit["stream"] != args.stream or unit["finished"] is not None:
            raise SystemExit(f"{args.block} {args.arm} is not running on stream {args.stream}: {unit}")
        unit.update(finished=now().isoformat(), status=args.status)
        queue.save()
        if len(b["units"]) == len(b["arms"]) and all(u["finished"] is not None for u in b["units"].values()):
            print("complete")


def move(args: argparse.Namespace) -> None:
    src, dst = Locked(args.source), Locked(args.target)
    with src as s, dst as d:
        moving = [b for b in s["blocks"] if b["block"] in set(args.blocks)]
        if len(moving) != len(args.blocks) or any(b["server"] is not None for b in moving):
            raise SystemExit("every block to move must exist in the source queue and be unstarted")
        s["blocks"] = [b for b in s["blocks"] if b["block"] not in set(args.blocks)]
        d["blocks"] = sorted(d["blocks"] + moving, key=lambda b: b["block"])
        src.save()
        dst.save()
        print(f"moved {len(moving)} blocks")


def status(args: argparse.Namespace) -> None:
    state = json.loads(Path(args.queue).read_text())
    units = [u for b in state["blocks"] for u in b["units"].values()]
    print(
        json.dumps(
            {
                "host": state["host"],
                "blocks": len(state["blocks"]),
                "unstarted_blocks": sum(b["server"] is None for b in state["blocks"]),
                "sessions": sum(len(b["arms"]) for b in state["blocks"]),
                "running": sum(u["finished"] is None for u in units),
                "finished": sum(u["finished"] is not None for u in units),
                "nonzero_exit": sum(u["status"] not in (None, 0, "0") for u in units),
            }
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("tasks")
    a.add_argument("out")
    a.add_argument("--host", action="append", required=True, help="NAME:STREAMS")
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--attempts", type=int, required=True)
    a.add_argument("--multiplier", type=float, required=True)
    a.add_argument("--expect-tasks", type=int, required=True)
    a.add_argument("--deadline-host", help="NAME=ISO time")
    a.add_argument("--start", help="ISO time the run starts (with --deadline-host)")
    c = sub.add_parser("claim")
    c.add_argument("queue")
    c.add_argument("stream")
    c.add_argument("server")
    c.add_argument("--deadline")
    f = sub.add_parser("finish")
    for name in ("queue", "stream", "block", "arm", "status"):
        f.add_argument(name)
    o = sub.add_parser("followon")
    o.add_argument("task_sets")
    o.add_argument("inventory")
    o.add_argument("out")
    o.add_argument("--benchmark", action="append", required=True, help="task-sets.json folder=block id prefix, in order")
    o.add_argument("--host", action="append", required=True, help="NAME:STREAMS")
    o.add_argument("--seed", type=int, required=True)
    o.add_argument("--multiplier", type=float, required=True)
    o.add_argument("--attempt", type=int, required=True)
    o.add_argument("--arms", help="comma-separated subset of the arms (default all four); the order rotates per block")
    o.add_argument("--queue-name", required=True, help="output files QUEUE_NAME-HOST.json, e.g. queue2")
    o.add_argument("--first-block-to")
    o.add_argument("--first-block-of")
    cl = sub.add_parser("close")
    cl.add_argument("queue")
    cl.add_argument("time")
    cu = sub.add_parser("cut")
    cu.add_argument("queue")
    cu.add_argument("eval_dir")
    cu.add_argument("label")
    u = sub.add_parser("upcoming")
    u.add_argument("queue")
    u.add_argument("n", type=int)
    t = sub.add_parser("take-unfit")
    t.add_argument("queue")
    t.add_argument("out")
    t.add_argument("--deadline", required=True)
    ap = sub.add_parser("append")
    ap.add_argument("queue")
    ap.add_argument("file")
    m = sub.add_parser("move")
    m.add_argument("source")
    m.add_argument("target")
    m.add_argument("blocks", nargs="+")
    s = sub.add_parser("status")
    s.add_argument("queue")
    args = p.parse_args()
    if args.command == "plan":
        if bool(args.deadline_host) != bool(args.start):
            raise SystemExit("--deadline-host and --start go together")
        plan(args)
    elif args.command == "claim":
        sys.exit(claim(args))
    elif args.command == "followon":
        followon(args)
    elif args.command == "close":
        close(args)
    elif args.command == "cut":
        cut(args)
    elif args.command == "upcoming":
        upcoming(args)
    elif args.command == "take-unfit":
        take_unfit(args)
    elif args.command == "append":
        append(args)
    elif args.command == "finish":
        finish(args)
    elif args.command == "move":
        move(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
