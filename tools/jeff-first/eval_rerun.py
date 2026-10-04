"""Re-queue Jeff-arm sessions that started before the Jeff capacity fix (2026-10-04 ~23:00: one jeff-serve per host
could not keep up, so Jeff-arm sessions waited in its queue or ended after 300 s of busy answers).

For every block of QUEUE whose a3-jeff07 / a4-jeff06 session started before FIX_TIME (from the queue's own start
times, so sessions still running count too), this:
- writes superseded.txt (the reason) into that session's folder EVAL_DIR/runs/ARM/TASK/attemptK/, so the summary
  leaves it out of the Jeff-arm results (the block's a1/a2 sessions keep their results);
- adds a rerun block to OUT (a queue file): the same task and attempt with an "r" suffix (block id + "r", attempt
  "Kr"), only the affected Jeff arms, pinned to the original block's server (server_pin), so the rerun is paired with
  the original a1/a2 sessions on the same Qwen server.
Usage: python3 eval_rerun.py EVAL_DIR QUEUE OUT FIX_TIME   (FIX_TIME ISO with offset)
"""

import datetime as dt
import json
import sys
from pathlib import Path

JEFF_ARMS = ("a3-jeff07", "a4-jeff06")


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    evald, queue_path, out, fix = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), dt.datetime.fromisoformat(sys.argv[4])
    if out.exists():
        raise SystemExit(f"{out} exists")
    state = json.loads(queue_path.read_text())
    blocks, marked = [], 0
    for b in state["blocks"]:
        arms = [a for a in JEFF_ARMS if a in b["units"] and dt.datetime.fromisoformat(b["units"][a]["started"]) < fix]
        if not arms:
            continue
        for arm in arms:
            folder = evald / "runs" / arm / b["task"].replace("/", ".").replace(":", ".") / f"attempt{b['attempt']}"
            if not folder.is_dir():
                raise SystemExit(f"{folder} missing for {b['block']} {arm}")
            (folder / "superseded.txt").write_text(
                f"started {b['units'][arm]['started']} before the Jeff capacity fix {fix.isoformat()}; rerun as block {b['block']}r\n"
            )
            marked += 1
        blocks.append(
            {
                "block": b["block"] + "r",
                "benchmark": b.get("benchmark", "terminal-bench-2"),
                "task": b["task"],
                "attempt": f"{b['attempt']}r",
                "arms": arms,
                "server_pin": b["server"],
                "server": None,
                "opened": None,
                "units": {},
            }
        )
    out.write_text(json.dumps({"host": state["host"], "multiplier": state["multiplier"], "seed": state["seed"], "tasks": state["tasks"], "blocks": blocks}, indent=1))
    print(f"{queue_path.name}: {marked} Jeff-arm sessions superseded, {len(blocks)} rerun blocks -> {out}")


if __name__ == "__main__":
    main()
