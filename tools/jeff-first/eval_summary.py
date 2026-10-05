"""Summary of tonight's four-arm evaluation (2026-10-04) from eval_units.py lines of every host.

Per arm: sessions finished (of planned) and running, pass rate (reward 1 / finished), median and mean wall time per
session (agent execution seconds), Qwen generation seconds per session, Qwen turns per session, share of turns at off,
guard hits (loop and runaway re-asks), forced-xhigh turns, thinking-limit cuts, Jeff steps taken, trim cuts (of trim
questions), errors, and per host.
Paired per task: for every task where the arm and the baseline both have a solved session, the ratio of their mean
wall times of solved sessions (arm / baseline); median and geometric mean over those tasks. Also paired per block
(same task, attempt and server) where both solved.
One section per benchmark and attempt (BENCHMARK for attempt 1, BENCHMARK#2 for attempt 2; pairs within a section) (Terminal-Bench 2.0 first, then the follow-on benchmarks in the order planned).
Usage: python3 eval_summary.py PLANNED UNITS.jsonl...   (PLANNED: benchmark=sessions,benchmark=sessions,...)
"""

import json
import math
import statistics
import sys
from collections import defaultdict

ARMS = ("a1-baseline", "a2-off-guard", "a3-jeff07", "a4-jeff06")


def fmt(x: float | None, digits: int = 0) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def ratio_stats(ratios: list[float]) -> str:
    if not ratios:
        return "n=0"
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return f"n={len(ratios)} median {statistics.median(ratios):.2f} geomean {geo:.2f}"


def main() -> None:
    planned = dict(item.split("=") for item in sys.argv[1].split(","))
    rows = [json.loads(line) for path in sys.argv[2:] for line in open(path) if line.strip()]
    # A follow-on benchmark's attempt 2 is its own section (key BENCHMARK#2); TB2 keeps its 3 attempts in one section,
    # as planned. Reruns ("1r") belong to their attempt.
    for r in rows:
        attempt = str(r["attempt"]).removesuffix("r")
        r["section"] = r["benchmark"] if r["benchmark"] == "terminal-bench-2" or attempt == "1" else f"{r['benchmark']}#{attempt}"
    unknown = {r["section"] for r in rows} - set(planned)
    if unknown:
        raise SystemExit(f"sessions of benchmarks not in PLANNED: {sorted(unknown)}")
    for benchmark, sessions in planned.items():
        subset = [r for r in rows if r["section"] == benchmark]
        print(f"## {benchmark}")
        print()
        section(int(sessions), subset)
        print()


def pct(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def section(planned: int, all_rows: list[dict]) -> None:
    superseded = [r for r in all_rows if r.get("superseded")]
    rows = [r for r in all_rows if not r.get("superseded")]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    print(
        f"planned sessions: {planned} ({planned // 4} per arm); started {len(all_rows)}; "
        f"left out as before the Jeff capacity fix (rerun): {len(superseded)}"
    )
    header = (
        "arm | finished | running | passed | pass rate | wall s median/mean | Qwen gen s/session | turns/session | "
        "turns at off | guard loop/runaway | forced xhigh | limit cuts | Jeff steps | trim cuts/questions | errors"
    )
    print(header)
    print("|".join("---" for _ in header.split("|")))
    for arm in ARMS:
        units = by_arm.get(arm, [])
        done = [u for u in units if u["state"] == "finished"]
        running = len(units) - len(done)
        scored = [u for u in done if u.get("reward") is not None]
        passed = sum(u["reward"] == 1 for u in scored)
        errors = [u for u in done if u.get("error")]
        walls = [u["agent_s"] for u in done if u.get("agent_s") is not None]
        traced = [u for u in done if "turns" in u]
        n = len(traced)
        turns = sum(u["turns"] for u in traced)

        def total(key: str) -> int:
            return sum(u[key] for u in traced)

        print(
            " | ".join(
                [
                    arm,
                    str(len(done)),
                    str(running),
                    str(passed),
                    f"{100 * passed / len(done):.1f}% ({passed}/{len(done)})" if done else "-",
                    f"{fmt(statistics.median(walls)) if walls else '-'}/{fmt(statistics.mean(walls)) if walls else '-'}",
                    fmt(total("qwen_s") / n) if n else "-",
                    fmt(turns / n, 1) if n else "-",
                    f"{100 * total('turns_off') / turns:.1f}%" if turns else "-",
                    f"{total('guard_loop')}/{total('guard_runaway')}" if n else "-",
                    str(total("forced_xhigh")) if n else "-",
                    str(total("limit_cuts")) if n else "-",
                    str(total("jeff_steps")) if n else "-",
                    f"{total('trim_cuts')}/{total('trim_questions')}" if n else "-",
                    str(len(errors)),
                ]
            )
        )
    print()
    print("Per host (finished, passed, wall s median):")
    for arm in ARMS:
        parts = []
        for host in sorted({u.get("host") for u in by_arm.get(arm, []) if u.get("host")}):
            done = [u for u in by_arm[arm] if u.get("host") == host and u["state"] == "finished"]
            walls = [u["agent_s"] for u in done if u.get("agent_s") is not None]
            passed = sum(u.get("reward") == 1 for u in done)
            parts.append(f"{host} {len(done)}/{passed}/{fmt(statistics.median(walls)) if walls else '-'}")
        print(f"  {arm}: " + "; ".join(parts))
    print()
    print("Paired with the baseline (wall-time ratio arm/baseline where both solved):")
    solved = defaultdict(list)  # (arm, task) -> wall seconds of solved sessions
    block_solved = {}
    for r in rows:
        if r["state"] == "finished" and r.get("reward") == 1 and r.get("agent_s"):
            solved[(r["arm"], r["task"])].append(r["agent_s"])
            block_solved[(r["arm"], r["pair_block"])] = r["agent_s"]
    for arm in ARMS[1:]:
        per_task = [
            statistics.mean(solved[(arm, t)]) / statistics.mean(solved[("a1-baseline", t)])
            for (a, t) in list(solved)
            if a == arm and ("a1-baseline", t) in solved
        ]
        per_block = [
            block_solved[(arm, b)] / block_solved[("a1-baseline", b)]
            for (a, b) in list(block_solved)
            if a == arm and ("a1-baseline", b) in block_solved
        ]
        print(f"  {arm}: per task {ratio_stats(per_task)}; per block {ratio_stats(per_block)}")
    print()
    print("Jeff latency per question, ms (median / p90 / n) in the sessions counted above:")
    for arm in ARMS[2:]:
        parts = []
        for key, name in (("step_ms_per_question", "step"), ("router_ms", "router"), ("trim_ms", "trim")):
            values = [v for u in by_arm.get(arm, []) for v in u.get(key, [])]
            parts.append(f"{name} {pct(values, 0.5)} / {pct(values, 0.9)} / {len(values)}" if values else f"{name} -")
        print(f"  {arm}: " + "; ".join(parts))
    print()
    errs = [r for r in rows if r["state"] == "finished" and r.get("error")]
    if errs:
        print(f"Errors ({len(errs)}):")
        for r in errs:
            print(f"  {r['arm']} {r['task']} attempt {r['attempt']} {r.get('server')}: {r['error'][:200]}")


if __name__ == "__main__":
    main()
