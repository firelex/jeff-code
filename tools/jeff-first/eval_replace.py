"""Replacement blocks for tonight's evaluation: a block whose run is unusable (a session killed by the system, recorded
as "Command failed (exit 137)"; or a block cut or left incomplete when the B200 stopped) is run again whole, as a new
block with the same task and arms, its id and attempt suffixed (k = killed, t = cut by time), side by side on one
casdgx01 Qwen server. The replaced block's sessions (and those of its Jeff-arm rerun block "<id>r") are left out of
the summary through superseded.txt, which names the replacement.

    python3 eval_replace.py killed SUFFIX OUT_QUEUE OUT_MARKS --queues Q... --units U...
        every block with a session whose error says "exit 137"
    python3 eval_replace.py blocks SUFFIX OUT_QUEUE OUT_MARKS --queues Q... --units U... --block ID...
        the given blocks
OUT_QUEUE is in eval_queue.py append format; OUT_MARKS lists "host<TAB>folder<TAB>reason" for mark_superseded.py.
A block already replaced (its replacement id exists in a queue) raises.
"""

import argparse
import json
from pathlib import Path


def base_id(block: str) -> str:
    return block.removesuffix("r")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("killed", "blocks"))
    p.add_argument("suffix", choices=("k", "t"))
    p.add_argument("out_queue")
    p.add_argument("out_marks")
    p.add_argument("--queues", nargs="+", required=True)
    p.add_argument("--units", nargs="+", required=True)
    p.add_argument("--block", nargs="*", default=[])
    a = p.parse_args()
    for out in (a.out_queue, a.out_marks):
        if Path(out).exists():
            raise SystemExit(f"{out} exists")
    blocks, tasks = {}, {}
    for q in a.queues:
        state = json.loads(Path(q).read_text())
        tasks.update(state["tasks"])
        for b in state["blocks"]:
            if b["block"] in blocks:
                raise SystemExit(f"block {b['block']} is in two queues")
            blocks[b["block"]] = b
    rows = []
    for u in a.units:
        host = "b200" if "b200" in Path(u).name else "casdgx01"
        rows += [dict(json.loads(line), _host=host) for line in open(u) if line.strip()]
    if a.mode == "killed":
        targets = sorted({base_id(r["block"]) for r in rows if r.get("block") and r.get("error") and "exit 137" in r["error"]})
    else:
        targets = sorted(set(a.block))
    new, marks = [], []
    for t in targets:
        if t not in blocks:
            raise SystemExit(f"block {t} not found in the queues")
        if t + a.suffix in blocks:
            raise SystemExit(f"block {t} already has a replacement {t + a.suffix}")
        o = blocks[t]
        new.append(
            {
                "block": t + a.suffix,
                "benchmark": o.get("benchmark", "terminal-bench-2"),
                "task": o["task"],
                "attempt": f"{o['attempt']}{a.suffix}",
                "arms": list(o["arms"]),
                "server": None,
                "opened": None,
                "units": {},
            }
        )
        why = "killed by the system (exit 137)" if a.suffix == "k" else "cut or left incomplete when the B200 stopped at 15:30"
        for r in rows:
            if r.get("block") in (t, t + "r") or (r.get("block") is None and r["folder"].endswith(f"/attempt{o['attempt']}") and r["task"] == o["task"].replace("/", ".").replace(":", ".")):
                marks.append(f"{r['_host']}\t{r['folder']}\tblock {t}: {why}; replaced by block {t + a.suffix}")
    Path(a.out_queue).write_text(json.dumps({"tasks": {b["task"]: tasks[b["task"]] for b in new}, "blocks": new}, indent=1))
    Path(a.out_marks).write_text("".join(m + "\n" for m in marks))
    by_bench = {}
    for b in new:
        by_bench[b["benchmark"]] = by_bench.get(b["benchmark"], 0) + 1
    print(f"{len(new)} replacement blocks {by_bench}; {len(marks)} sessions to mark superseded")


if __name__ == "__main__":
    main()
