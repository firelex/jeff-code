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
        Prints "BLOCK ARM TASK ATTEMPT". Exit 3: nothing to start now; exit 4: nothing left that can ever start here.
finish  records a session's end (its run_phase0.sh exit status).
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


def claim(args: argparse.Namespace) -> int:
    queue = Locked(args.queue)
    with queue as state:
        deadline = dt.datetime.fromisoformat(args.deadline) if args.deadline else None
        t = now()

        def fits(task: str) -> bool:
            if deadline is None:
                return True
            return t + dt.timedelta(seconds=longest_s(state["tasks"][task], state["multiplier"])) <= deadline

        def take(b: dict) -> int:
            arm = next(a for a in b["arms"] if a not in b["units"])
            b["units"][arm] = {"stream": args.stream, "started": t.isoformat(), "finished": None, "status": None}
            queue.save()
            print(b["block"], arm, b["task"], b["attempt"])
            return 0

        for b in state["blocks"]:
            if b["server"] == args.server and len(b["units"]) < 4 and fits(b["task"]):
                return take(b)
        for b in state["blocks"]:
            if b["server"] is None and fits(b["task"]):
                b["server"], b["opened"] = args.server, t.isoformat()
                return take(b)
        open_here = [b for b in state["blocks"] if b["server"] == args.server and len(b["units"]) < 4]
        unstarted = [b for b in state["blocks"] if b["server"] is None]
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
        unit.update(finished=now().isoformat(), status=int(args.status))
        queue.save()


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
                "sessions": 4 * len(state["blocks"]),
                "running": sum(u["finished"] is None for u in units),
                "finished": sum(u["finished"] is not None for u in units),
                "nonzero_exit": sum(u["status"] not in (None, 0) for u in units),
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
    elif args.command == "finish":
        finish(args)
    elif args.command == "move":
        move(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
