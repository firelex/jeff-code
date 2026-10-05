"""Final report of tonight's evaluation (2026-10-04/05) from eval_units.py lines of every host.

Arms: a1-baseline (Qwen at xhigh thinking, Jeff off), a2-off-guard (thinking off with safeguards), a3-jeff07 and
a4-jeff06 (Jeff routes thinking at P(xhigh) >= 0.7 / 0.6, takes steps, may shorten tool output). A block is one task
attempt run under the arms of that benchmark on one Qwen server (swe-bench-verified ran a1 and a4 only); a Jeff-arm
rerun (block id ending in "r") pairs with the original block's a1/a2 sessions. Sessions marked superseded (Jeff arms
started before the Jeff capacity fix) or cut (still running when the B200 stopped) are left out and counted.

Sections: one per benchmark and round (Terminal-Bench 2.0 keeps its three attempts in one section; a follow-on
benchmark's attempt 2 is the section BENCHMARK#2). The pooled scope takes every round of the benchmarks with a
meaningful baseline pass rate (all but terminal-bench and terminal-bench-science).

Speed measures (owner's final decision), all paired by block, ratio = arm / baseline, in this order:
- "on average per task": the geometric mean of per-block wall-time ratios over all paired blocks (the verdict's speed
  half), then the median per-block ratio;
- total wall time over all paired blocks (all outcomes), with why it can be higher (a few long sessions dominate
  the sum; the top outliers are named), and total Qwen generation time;
- time on tasks the baseline solved (blocks where a1 passed, the arm's session counted whatever its outcome), both
  solved, and the median wall time per session.
Per section and arm, the 10 pairs with the largest wall-time difference either way are listed (outliers).
Pass rate per arm (Wilson 95%) and the paired difference (task-bootstrap 95%, exact McNemar on discordant blocks).
Also: behaviour per arm, a verdict per arm against the owner's target (25% faster at the same pass rate), caveats,
and every session or block left out with the reason.

Owner's ruling: every block in which a session was killed by the system (command exit 137, SIGKILL, e.g. out of
memory, read from the recorded exit code) is left out of the pass rates and every time measure, in all arms; the
blocks are listed.

Intervals resample tasks (all rounds, attempts and arms of a task together). Wall time is the agent's execution time.
A pass is reward 1 (partial rewards such as 0.667 count as not passed).

Usage: python3 eval_report.py OUT.md UNITS.jsonl...
"""

import datetime as dt
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BASE = "a1-baseline"
ARMS = (BASE, "a2-off-guard", "a3-jeff07", "a4-jeff06")
JEFF_ARMS = ("a3-jeff07", "a4-jeff06")
# Section order: Terminal-Bench 2.0, the follow-on benchmarks (round 1, then round 2), SWE-bench Verified, then the two
# benchmarks moved to the end.
SECTIONS = (
    "terminal-bench-2",
    "harbor-index-1.0",
    "harbor-index-1.0#2",
    "skillsbench",
    "skillsbench#2",
    "terminal-bench-pro",
    "terminal-bench-pro#2",
    "swe-rebench-leaderboard",
    "swe-rebench-leaderboard#2",
    "swe-bench-verified",
    "terminal-bench",
    "terminal-bench-science",
)
NOT_POOLED = {
    "terminal-bench": "baseline pass rate near zero, so it cannot show a pass-rate difference",
    "terminal-bench-science": "baseline pass rate near zero, so it cannot show a pass-rate difference",
}
POOLED = tuple(s for s in SECTIONS if s.split("#")[0] not in NOT_POOLED)
EXCLUDED_TASKS = {
    ("terminal-bench-2", "pytorch-model-recovery"): (
        "harness bug: the instruction starts with '- ', so pi stopped with 'Unknown option' before doing anything in "
        "every a1/a2 session (and the a3/a4 sessions started before the fix); the a3/a4 reruns ran with the fix, so "
        "the arms did not get the same task"
    ),
}
TARGET_RATIO = 0.75  # 25% faster
SAME_PASS_MARGIN = 0.05  # "same pass rate": the paired difference's interval stays above -5 points
THIN_PAIRS = 30  # fewer paired blocks than this, or an interval wider than THIN_WIDTH, is called too thin
THIN_WIDTH = 0.20
RESAMPLES = 4000
SEED = 20261005


KILL_EXIT = "137"
_EXIT = re.compile(r"Command failed \(exit (\d+)\)")


def exit_code(row: dict) -> str | None:
    """The agent command's exit code as recorded in the session's error, None when the trial had no exception (or only
    the agent time limit), "not recorded" when it had another exception whose message holds no exit code."""
    match = _EXIT.search(row.get("error") or "")
    if match:
        return match.group(1)
    if row.get("exception") in (None, "AgentTimeoutError"):
        return None
    return "not recorded"


def section_of(row: dict) -> str:
    attempt = str(row["attempt"]).removesuffix("r")
    if row["benchmark"] == "terminal-bench-2" or attempt == "1":
        return row["benchmark"]
    return f"{row['benchmark']}#{attempt}"


# ---------------------------------------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion k/n."""
    if n <= 0:
        raise ValueError("wilson interval of zero sessions")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p: binomial test of b baseline-only vs c arm-only passes at probability 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geometric mean of no values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def nearest_rank(values: list[float], q: float) -> float:
    """The q-quantile by nearest rank (the value at rank ceil(q * n))."""
    if not values:
        raise ValueError("quantile of no values")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def sum_fields(items: list[dict]) -> dict:
    """Adds number fields and concatenates list fields."""
    total: dict = {}
    for item in items:
        for key, value in item.items():
            total[key] = total.get(key, [] if isinstance(value, list) else 0) + value
    return total


@dataclass
class Interval:
    lo: float | None
    hi: float | None
    undefined: int  # resamples in which the statistic was not defined (e.g. no pass); any such -> no interval


def _quantile(ordered: list[float], q: float) -> float:
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_intervals(
    clusters: list[dict], statistics_: dict[str, Callable[[dict], float | None]], resamples: int, seed: int
) -> dict[str, Interval]:
    """Percentile 95% intervals by resampling whole clusters (tasks) with replacement. Each cluster is a dict of fields
    (numbers are summed over a resample, lists concatenated); each statistic is computed from those totals."""
    if not clusters:
        raise ValueError("bootstrap of no clusters")
    keys = sorted(clusters[0])
    if any(sorted(c) != keys for c in clusters):
        raise ValueError("bootstrap clusters with different fields")
    list_keys = [k for k in keys if isinstance(clusters[0][k], list)]
    number_keys = [k for k in keys if k not in list_keys]
    numbers = [tuple(c[k] for k in number_keys) for c in clusters]
    rng = random.Random(seed)
    values: dict[str, list[float]] = {name: [] for name in statistics_}
    undefined = dict.fromkeys(statistics_, 0)
    for _ in range(resamples):
        picks = rng.choices(range(len(clusters)), k=len(clusters))
        totals = dict(zip(number_keys, (sum(column) for column in zip(*(numbers[i] for i in picks)))))
        for k in list_keys:
            totals[k] = [x for i in picks for x in clusters[i][k]]
        for name, statistic in statistics_.items():
            value = statistic(totals)
            if value is None:
                undefined[name] += 1
            else:
                values[name].append(value)
    result = {}
    for name in statistics_:
        if undefined[name]:
            result[name] = Interval(None, None, undefined[name])
        else:
            ordered = sorted(values[name])
            result[name] = Interval(_quantile(ordered, 0.025), _quantile(ordered, 0.975), 0)
    return result


def bootstrap_interval(
    clusters: list[dict], statistic: Callable[[dict], float | None], resamples: int, seed: int
) -> Interval:
    return bootstrap_intervals(clusters, {"x": statistic}, resamples, seed)["x"]


# ------------------------------------------------------------------------------------------------------------- pairing


@dataclass
class Pair:
    cluster: tuple[str, str]  # (benchmark, task)
    block: str
    rerun: bool  # the arm's session is a rerun (ran later than its baseline)
    fields: dict
    qwen_zero_note: str | None  # why this block is out of the per-block Qwen ratios, if it is
    task: str
    base_wall: float
    arm_wall: float
    base_pass: bool
    arm_pass: bool
    arm_turns: int | None  # the arm session's trace counts (None without a trace)
    arm_turns_off: int | None
    arm_guard: int | None
    arm_forced: int | None


def passed(row: dict) -> bool:
    return row["reward"] == 1


def pair_up(rows: list[dict], arm: str) -> tuple[list[Pair], list[tuple[tuple, str]]]:
    """Pairs every block's finished baseline and arm sessions. Blocks where either side has no finished session or no
    reward are left out and returned with the reason (running and cut sessions are counted elsewhere)."""
    sides: dict[str, dict[str, dict]] = {BASE: {}, arm: {}}
    for r in rows:
        if r.get("superseded") or r.get("cut"):
            raise ValueError(f"superseded or cut session passed to pair_up: {r.get('folder')}")
        if r["arm"] in sides and r["state"] == "finished":
            if r["pair_block"] in sides[r["arm"]]:
                raise ValueError(f"two finished {r['arm']} sessions in block {r['pair_block']}")
            sides[r["arm"]][r["pair_block"]] = r
    pairs, left_out = [], []
    for block in sorted(set(sides[BASE]) | set(sides[arm])):
        base, other = sides[BASE].get(block), sides[arm].get(block)
        some = base or other
        key = (some["benchmark"], some["task"], block)
        if base is None or other is None:
            left_out.append((key, f"no finished {BASE if base is None else arm} session"))
            continue
        if (base["task"], base["benchmark"], base["server"]) != (other["task"], other["benchmark"], other["server"]):
            raise ValueError(
                f"block {block}: {BASE} and {arm} differ in task/benchmark/server: "
                f"{base['task']} {base['benchmark']} {base['server']} vs {other['task']} {other['benchmark']} "
                f"{other['server']}"
            )
        missing = [name for name, r in ((BASE, base), (arm, other)) if r.get("reward") is None]
        if missing:
            who = "both sessions" if len(missing) == 2 else f"{missing[0]} session"
            left_out.append((key, f"{who} finished without a reward"))
            continue
        for name, r in ((BASE, base), (arm, other)):
            if r.get("agent_s") is None or r["agent_s"] <= 0:
                raise ValueError(f"block {block}: {name} session has a reward but no agent time: {r.get('folder')}")
        bw, aw = base["agent_s"], other["agent_s"]
        bp, ap = passed(base), passed(other)
        has_qwen = "qwen_s" in base and "qwen_s" in other
        bq, aq = (base["qwen_s"], other["qwen_s"]) if has_qwen else (0.0, 0.0)
        qwen_zero = has_qwen and (bq <= 0 or aq <= 0)
        fields = {
            "n": 1,
            "base_pass": int(bp),
            "arm_pass": int(ap),
            "base_only": int(bp and not ap),
            "arm_only": int(ap and not bp),
            # 1. time on tasks the baseline solved (the arm counted whatever its outcome)
            "bs_n": int(bp),
            "bs_base_wall": bw if bp else 0.0,
            "bs_arm_wall": aw if bp else 0.0,
            "bs_logw": [math.log(aw / bw)] if bp else [],
            "bs_qwen_n": int(bp and has_qwen),
            "bs_base_qwen": bq if bp else 0.0,
            "bs_arm_qwen": aq if bp else 0.0,
            "bs_logq": [math.log(aq / bq)] if bp and has_qwen and not qwen_zero else [],
            # 2. both solved
            "solved_both": int(bp and ap),
            "both_logw": [math.log(aw / bw)] if bp and ap else [],
            # 3. totals over all paired blocks
            "base_wall": bw,
            "arm_wall": aw,
            "qwen_n": int(has_qwen),
            "base_qwen": bq,
            "arm_qwen": aq,
            "qwen_zero": int(qwen_zero),
            # per-block wall ratios on all paired blocks
            "all_logw": [math.log(aw / bw)],
            # 4. per-session wall times
            "base_walls": [bw],
            "arm_walls": [aw],
        }
        pairs.append(
            Pair(
                cluster=(base["benchmark"], base["task"]),
                block=block,
                rerun=other["block"].endswith("r"),
                fields=fields,
                qwen_zero_note=f"{key[1]} {block}: Qwen time {bq:.0f} s / {aq:.0f} s" if qwen_zero else None,
                task=base["task"],
                base_wall=bw,
                arm_wall=aw,
                base_pass=bp,
                arm_pass=ap,
                arm_turns=other.get("turns"),
                arm_turns_off=other.get("turns_off"),
                arm_guard=None if "guard_loop" not in other else other["guard_loop"] + other["guard_runaway"],
                arm_forced=other.get("forced_xhigh"),
            )
        )
    return pairs, left_out


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def _exp_mean(logs: list[float]) -> float | None:
    return math.exp(sum(logs) / len(logs)) if logs else None


def _exp_median(logs: list[float]) -> float | None:
    return math.exp(statistics.median(logs)) if logs else None


PAIR_STATISTICS: dict[str, Callable[[dict], float | None]] = {
    "diff": lambda t: (t["arm_pass"] - t["base_pass"]) / t["n"],
    "bs_geo": lambda t: _exp_mean(t["bs_logw"]),
    "bs_median": lambda t: _exp_median(t["bs_logw"]),
    "bs_total": lambda t: _ratio(t["bs_arm_wall"], t["bs_base_wall"]),
    "bs_qwen_geo": lambda t: _exp_mean(t["bs_logq"]),
    "bs_qwen_median": lambda t: _exp_median(t["bs_logq"]),
    "bs_qwen_total": lambda t: _ratio(t["bs_arm_qwen"], t["bs_base_qwen"]),
    "both_geo": lambda t: _exp_mean(t["both_logw"]),
    "both_median": lambda t: _exp_median(t["both_logw"]),
    "all_geo": lambda t: _exp_mean(t["all_logw"]),
    "all_median": lambda t: _exp_median(t["all_logw"]),
    "total_wall": lambda t: _ratio(t["arm_wall"], t["base_wall"]),
    "total_qwen": lambda t: _ratio(t["arm_qwen"], t["base_qwen"]),
    "session_median": lambda t: statistics.median(t["arm_walls"]) / statistics.median(t["base_walls"]),
    "tts": lambda t: (
        (t["arm_wall"] / t["arm_pass"]) / (t["base_wall"] / t["base_pass"]) if t["arm_pass"] and t["base_pass"] else None
    ),
}


def largest_differences(pairs: list[Pair], k: int) -> list[Pair]:
    """The k pairs with the largest wall-time difference between arm and baseline, either way."""
    return sorted(pairs, key=lambda p: (-abs(p.arm_wall - p.base_wall), p.block))[:k]


def statistics_of(totals: dict) -> dict[str, float | None]:
    return {name: f(totals) for name, f in PAIR_STATISTICS.items()}


@dataclass
class Comparison:
    arm: str
    pairs: list[Pair]
    totals: dict
    point: dict[str, float | None]
    intervals: dict[str, Interval]


def compare(pairs: list[Pair], arm: str, seed: int) -> Comparison | None:
    if not pairs:
        return None
    by_cluster: dict[tuple, list[dict]] = defaultdict(list)
    for p in pairs:
        by_cluster[p.cluster].append(p.fields)
    clusters = [sum_fields(v) for _, v in sorted(by_cluster.items())]
    totals = sum_fields([p.fields for p in pairs])
    return Comparison(
        arm=arm,
        pairs=pairs,
        totals=totals,
        point=statistics_of(totals),
        intervals=bootstrap_intervals(clusters, PAIR_STATISTICS, RESAMPLES, seed),
    )


# ---------------------------------------------------------------------------------------------------- session counts

ERROR_KINDS = (
    ("JeffFirst: ", "JeffFirst error"),
    ("pi output unreadable: ", "no pi output"),
    ("image pull failed", "image pull failed"),
)


def error_kind(error: str) -> str:
    for prefix, kind in ERROR_KINDS:
        if error.startswith(prefix):
            return kind
    if error.endswith("result.json files"):
        return "no Harbor result"
    head = error.split(":", 1)[0]
    if head.endswith("Error") and " " not in head:
        return head
    raise ValueError(f"unknown kind of session error: {error[:200]}")


def arm_profile(sessions: list[dict]) -> dict:
    """Counts over an arm's finished sessions."""
    traced = [s for s in sessions if "turns" in s]
    turns = sum(s["turns"] for s in traced)

    def total(key: str) -> int:
        return sum(s[key] for s in traced)

    errors = Counter(error_kind(s["error"]) for s in sessions if s.get("error"))
    latency = {
        name: [v for s in traced for v in s[key]]
        for name, key in (("router", "router_ms"), ("step", "step_ms_per_question"), ("trim", "trim_ms"))
    }
    return {
        "sessions": len(sessions),
        "traced": len(traced),
        "turns": turns,
        "turns_off": total("turns_off"),
        "guard_loop": total("guard_loop"),
        "guard_runaway": total("guard_runaway"),
        "forced_xhigh": total("forced_xhigh"),
        "limit_cuts": total("limit_cuts"),
        "jeff_steps": total("jeff_steps"),
        "trim_cuts": total("trim_cuts"),
        "trim_questions": total("trim_questions"),
        "time_limit": sum(s.get("exception") == "AgentTimeoutError" for s in sessions),
        "errors": errors,
        "latency": latency,
    }


# ------------------------------------------------------------------------------------------------------------ loading


@dataclass
class Data:
    active: list[dict]  # not superseded, not cut, not of an excluded task, not in a block with a killed session
    superseded: list[dict]
    cut: list[dict]
    killed_blocks: list[dict]  # every finished session of a block in which a session was killed (exit 137)
    unrecorded_exit: list[dict]  # sessions with an exception but no recorded exit code (kept in)
    excluded: list[dict]
    inputs: list[tuple[str, int, str]]  # path, lines, modified time


def load(paths: list[str]) -> Data:
    rows, inputs = [], []
    for path in paths:
        lines = [line for line in Path(path).read_text().split("\n") if line.strip()]
        if not lines:
            raise ValueError(f"{path} holds no session lines")
        rows.extend(json.loads(line) for line in lines)
        modified = dt.datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        inputs.append((path, len(lines), modified))
    for r in rows:
        if r["arm"] not in ARMS:
            raise ValueError(f"unknown arm {r['arm']!r} in {r.get('folder')}")
        r["section"] = section_of(r)
        if r["section"] not in SECTIONS:
            raise ValueError(f"unknown benchmark section {r['section']!r} in {r.get('folder')}")
        if r["state"] not in ("finished", "running", "cut"):
            raise ValueError(f"unknown state {r['state']!r} in {r.get('folder')}")
    folders = Counter(r["folder"] for r in rows)
    twice = [f for f, n in folders.items() if n > 1]
    if twice:
        raise ValueError(f"sessions listed twice (same input given twice?): {twice[:5]}")
    blocks: dict[str, tuple[str, str]] = {}
    for r in rows:
        if r["state"] == "finished":
            seen = blocks.setdefault(r["pair_block"], (r["section"], r["task"]))
            if seen != (r["section"], r["task"]):
                raise ValueError(f"block {r['pair_block']} holds two tasks: {seen} and {(r['section'], r['task'])}")
    superseded = [r for r in rows if r.get("superseded")]
    cut = [r for r in rows if not r.get("superseded") and r.get("cut")]
    rest = [r for r in rows if not r.get("superseded") and not r.get("cut")]
    excluded = [r for r in rest if (r["benchmark"], r["task"]) in EXCLUDED_TASKS]
    kept = [r for r in rest if (r["benchmark"], r["task"]) not in EXCLUDED_TASKS]
    finished = [r for r in kept if r["state"] == "finished"]
    killed = {r["pair_block"] for r in finished if exit_code(r) == KILL_EXIT}
    killed_blocks = [r for r in finished if r["pair_block"] in killed]
    active = [r for r in kept if not (r["state"] == "finished" and r["pair_block"] in killed)]
    unrecorded = [r for r in active if r["state"] == "finished" and exit_code(r) == "not recorded"]
    return Data(
        active=active,
        superseded=superseded,
        cut=cut,
        excluded=excluded,
        inputs=inputs,
        killed_blocks=killed_blocks,
        unrecorded_exit=unrecorded,
    )


# ------------------------------------------------------------------------------------------------------------- format


def pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def interval_text(iv: Interval, fmt: Callable[[float], str]) -> str:
    if iv.undefined:
        return f"n/a, {iv.undefined}/{RESAMPLES} resamples undefined"
    return f"{fmt(iv.lo)} to {fmt(iv.hi)}"


def r2(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def pass_rate_cell(sessions: list[dict]) -> str:
    scored = [s for s in sessions if s.get("reward") is not None]
    if not scored:
        return "-"
    k = sum(passed(s) for s in scored)
    lo, hi = wilson(k, len(scored))
    return f"{pct(k / len(scored))} ({k}/{len(scored)}; {pct(lo, 0)} to {pct(hi, 0)})"


def diff_cell(c: Comparison) -> str:
    t = c.totals
    return (
        f"{pts(c.point['diff'])} pts ({interval_text(c.intervals['diff'], pts)}); "
        f"{t['base_only']} vs {t['arm_only']}, p={mcnemar_p(t['base_only'], t['arm_only']):.2f}"
    )


def ratio_cell(c: Comparison, name: str) -> str:
    return f"{r2(c.point[name])} ({interval_text(c.intervals[name], r2)})"


def verdict(c: Comparison | None, arm: str) -> str:
    if c is None:
        return f"{arm}: no paired blocks."
    t = c.totals
    diff_iv, speed_iv = c.intervals["diff"], c.intervals["all_geo"]
    speed_value = c.point["all_geo"]
    if speed_value is None or speed_iv.undefined:
        speed = "speed not defined"
    elif speed_iv.hi <= TARGET_RATIO:
        speed = "speed target met (whole interval at or below 0.75)"
    elif speed_value <= TARGET_RATIO and speed_iv.hi < 1:
        speed = "speed target met on the estimate, but the interval reaches above 0.75"
    elif speed_iv.hi < 1:
        speed = "faster than baseline, but short of the 25% target"
    elif speed_iv.lo > 1:
        speed = "slower than baseline"
    else:
        speed = "no clear speed difference (interval spans 1)"
    if diff_iv.undefined:
        same = "pass-rate interval undefined"
    elif diff_iv.lo >= -SAME_PASS_MARGIN:
        same = "same pass rate shown (interval stays above -5 points)"
    elif diff_iv.hi < 0:
        same = "pass rate lower than baseline"
    else:
        same = "a pass-rate loss of more than 5 points is not ruled out"
    met = (
        speed_value is not None
        and not speed_iv.undefined
        and not diff_iv.undefined
        and speed_value <= TARGET_RATIO
        and speed_iv.hi < 1
        and diff_iv.lo >= -SAME_PASS_MARGIN
    )
    thin = t["n"] < THIN_PAIRS or (not diff_iv.undefined and diff_iv.hi - diff_iv.lo > THIN_WIDTH)
    overall = "MEETS the target" if met else "does NOT meet the target"
    if thin:
        overall += f" -- data too thin to rely on ({t['n']} paired blocks; pass-rate interval width " + (
            "undefined)" if diff_iv.undefined else f"{100 * (diff_iv.hi - diff_iv.lo):.0f} pts)"
        )
    return (
        f"**{arm}: {overall}.** On average per task (geometric mean of per-block wall ratios) {r2(speed_value)} x "
        f"baseline ({interval_text(speed_iv, r2)}; {t['n']} blocks): {speed}. Median per block "
        f"{r2(c.point['all_median'])}; total wall {ratio_cell(c, 'total_wall')}. "
        f"Pass rate {pts(c.point['diff'])} pts ({interval_text(diff_iv, pts)}): {same}."
    )


def headline_summary(c: Comparison) -> str:
    """The owner's headline for one arm: pass difference, per-task time ratio, median, total and why they differ."""
    t = c.totals
    text = (
        f"**{c.arm}**: pass rate {pts(c.point['diff'])} pts vs baseline ({interval_text(c.intervals['diff'], pts)}; "
        f"{t['base_only']} blocks only the baseline passed, {t['arm_only']} only the arm). On average per task "
        f"(geometric mean of per-block wall-time ratios) **{r2(c.point['all_geo'])} x** baseline "
        f"({interval_text(c.intervals['all_geo'], r2)}; {t['n']} paired blocks); median per block "
        f"{r2(c.point['all_median'])}. Total wall time over all paired blocks {ratio_cell(c, 'total_wall')}."
    )
    if c.point["total_wall"] > c.point["all_geo"]:
        top = largest_differences([p for p in c.pairs if p.arm_wall > p.base_wall], 3)
        text += (
            " The total is higher than the per-task mean because a few long sessions dominate the sum: "
            + "; ".join(
                f"{p.task} ({p.base_wall / 60:.0f} -> {p.arm_wall / 60:.0f} min, "
                f"{'pass' if p.base_pass else 'fail'} -> {'pass' if p.arm_pass else 'fail'})"
                for p in top
            )
            + "."
        )
    return text


# ------------------------------------------------------------------------------------------------------------- report


@dataclass
class Scope:
    name: str
    sections: tuple[str, ...]
    comparisons: dict[str, Comparison | None]
    left_out: dict[str, list[tuple[tuple, str]]]
    sessions: dict[str, list[dict]]  # arm -> finished sessions
    running: Counter


def build_scope(name: str, sections: tuple[str, ...], data: Data, seed: int) -> Scope:
    rows = [r for r in data.active if r["section"] in sections]
    comparisons, left_out = {}, {}
    for i, arm in enumerate(ARMS[1:]):
        pairs, out = pair_up(rows, arm)
        comparisons[arm] = compare(pairs, arm, seed + i)
        left_out[arm] = out
    return Scope(
        name=name,
        sections=sections,
        comparisons=comparisons,
        left_out=left_out,
        sessions={arm: [r for r in rows if r["arm"] == arm and r["state"] == "finished"] for arm in ARMS},
        running=Counter(r["arm"] for r in rows if r["state"] == "running"),
    )


HEADLINE_HEADER = [
    "scope",
    "arm",
    "pass rate (95%)",
    "paired blocks",
    "pass diff vs baseline, pts (95%); baseline-only vs arm-only",
    "on average per task: geomean of per-block wall ratios (95%)",
    "median per-block ratio",
    "total wall, all paired blocks (95%)",
    "total Qwen gen, all paired (95%)",
    "baseline-solved tasks: total (95%) / geomean, blocks",
    "both solved: geomean, blocks",
    "median wall per session",
]


def headline_rows(scope: Scope) -> list[list[str]]:
    out = [[scope.name, BASE, pass_rate_cell(scope.sessions[BASE])] + ["-"] * 8]
    for arm in ARMS[1:]:
        c = scope.comparisons[arm]
        if c is None:
            if scope.sessions[arm]:
                out.append([scope.name, arm, pass_rate_cell(scope.sessions[arm]), "0"] + ["-"] * 7)
            continue
        t = c.totals
        out.append(
            [
                scope.name,
                arm,
                pass_rate_cell(scope.sessions[arm]),
                str(t["n"]),
                diff_cell(c),
                f"**{ratio_cell(c, 'all_geo')}**",
                r2(c.point["all_median"]),
                ratio_cell(c, "total_wall"),
                ratio_cell(c, "total_qwen"),
                f"{ratio_cell(c, 'bs_total')} / {r2(c.point['bs_geo'])}, {t['bs_n']}",
                f"{r2(c.point['both_geo'])}, {t['solved_both']}",
                r2(c.point["session_median"]),
            ]
        )
    return out


def scope_section(scope: Scope) -> list[str]:
    out = []
    finished = {arm: len(scope.sessions[arm]) for arm in ARMS}
    out.append(
        "Sessions finished per arm: "
        + ", ".join(f"{arm} {finished[arm]}" for arm in ARMS)
        + "; still running: "
        + (", ".join(f"{arm} {scope.running[arm]}" for arm in ARMS if scope.running[arm]) or "none")
        + "."
    )
    out.append("")
    arms = [arm for arm in ARMS if scope.sessions[arm] or scope.running[arm]]
    compared = [arm for arm in arms[1:] if arm != BASE]
    out.append("**Pass rate**")
    out.append("")
    rows = []
    for arm in arms:
        c = scope.comparisons.get(arm)
        rows.append(
            [
                arm,
                pass_rate_cell(scope.sessions[arm]),
                "-" if arm == BASE else (str(c.totals["n"]) if c else "0"),
                "-"
                if arm == BASE or c is None
                else f"{pct(c.totals['base_pass'] / c.totals['n'])} -> {pct(c.totals['arm_pass'] / c.totals['n'])}",
                "-" if arm == BASE or c is None else diff_cell(c),
            ]
        )
    out += table(
        [
            "arm",
            "pass rate, all scored sessions (95%)",
            "paired blocks",
            "baseline -> arm on paired blocks",
            "difference (95%); baseline-only vs arm-only; McNemar p",
        ],
        rows,
    )
    out.append("")
    out.append("**Speed, paired by block (ratio = arm / baseline; below 1 is faster)**")
    out.append("")
    rows = []
    for arm in compared:
        c = scope.comparisons[arm]
        if c is None:
            rows.append([arm] + ["-"] * 7)
            continue
        t = c.totals
        rows.append(
            [
                arm,
                f"{t['n']}: geomean **{ratio_cell(c, 'all_geo')}**; median {ratio_cell(c, 'all_median')}",
                f"wall {hours(t['base_wall'])} -> {hours(t['arm_wall'])}, ratio {ratio_cell(c, 'total_wall')}; "
                f"Qwen gen {hours(t['base_qwen'])} -> {hours(t['arm_qwen'])} (n={t['qwen_n']}), ratio "
                f"{ratio_cell(c, 'total_qwen')}",
                f"{t['bs_n']}: total {hours(t['bs_base_wall'])} -> {hours(t['bs_arm_wall'])}, "
                f"ratio {ratio_cell(c, 'bs_total')}; geomean {ratio_cell(c, 'bs_geo')}; "
                f"median {ratio_cell(c, 'bs_median')}",
                f"{t['bs_qwen_n']}: total {hours(t['bs_base_qwen'])} -> {hours(t['bs_arm_qwen'])}, "
                f"ratio {ratio_cell(c, 'bs_qwen_total')}; geomean {ratio_cell(c, 'bs_qwen_geo')}; "
                f"median {ratio_cell(c, 'bs_qwen_median')}",
                f"{t['solved_both']}: geomean {ratio_cell(c, 'both_geo')}; median {ratio_cell(c, 'both_median')}",
                f"{statistics.median(t['base_walls']) / 60:.1f} -> {statistics.median(t['arm_walls']) / 60:.1f} min, "
                f"ratio {ratio_cell(c, 'session_median')}",
                f"{t['base_wall'] / t['base_pass'] / 60:.1f} -> {t['arm_wall'] / t['arm_pass'] / 60:.1f} min, "
                f"ratio {ratio_cell(c, 'tts')}"
                if t["base_pass"] and t["arm_pass"]
                else "-",
            ]
        )
    out += table(
        [
            "arm",
            "on average per task: per-block wall ratio, all paired blocks (blocks)",
            "total over all paired blocks",
            "time on tasks the baseline solved, wall (blocks)",
            "same, Qwen generation time (blocks)",
            "both solved, wall (blocks)",
            "median wall per session",
            "extra: wall per pass (time to solve)",
        ],
        rows,
    )
    out.append("")
    for arm in compared:
        c = scope.comparisons[arm]
        if c is None:
            continue
        out.append(f"Largest wall-time differences, {arm} vs baseline (10 pairs, either way):")
        out.append("")
        rows = []
        for p in largest_differences(c.pairs, 10):
            rows.append(
                [
                    p.task,
                    p.block,
                    f"{p.base_wall / 60:.1f} / {p.arm_wall / 60:.1f}",
                    f"{p.arm_wall - p.base_wall:+.0f} s",
                    f"{'pass' if p.base_pass else 'fail'} / {'pass' if p.arm_pass else 'fail'}",
                    "-" if not p.arm_turns else f"{p.arm_turns_off}/{p.arm_turns}",
                    "-" if p.arm_guard is None else str(p.arm_guard),
                    "-" if p.arm_forced is None else str(p.arm_forced),
                ]
            )
        out += table(
            [
                "task",
                "block",
                "wall min, baseline / arm",
                "arm - baseline",
                "outcome, baseline / arm",
                "arm turns at off",
                "arm guard re-asks",
                "arm forced xhigh",
            ],
            rows,
        )
        out.append("")
    out.append("**Behaviour per arm (all finished sessions of the scope)**")
    out.append("")
    rows = []
    for arm in arms:
        p = arm_profile(scope.sessions[arm])
        lat = []
        for name in ("router", "step", "trim"):
            values = p["latency"][name]
            if values:
                lat.append(
                    f"{name} {statistics.median(values):.0f}/{nearest_rank(values, 0.9):.0f} ms (n={len(values)})"
                )
        rows.append(
            [
                arm,
                f"{p['sessions']} ({p['traced']} traced)",
                f"{pct(p['turns_off'] / p['turns']) if p['turns'] else '-'} of {p['turns']}",
                f"{p['guard_loop']} / {p['guard_runaway']}",
                str(p["forced_xhigh"]),
                str(p["limit_cuts"]),
                str(p["jeff_steps"]),
                f"{p['trim_cuts']}/{p['trim_questions']}",
                "; ".join(lat) or "-",
                str(p["time_limit"]),
                ", ".join(f"{k} {v}" for k, v in sorted(p["errors"].items())) or "none",
            ]
        )
    out += table(
        [
            "arm",
            "sessions",
            "turns at thinking off",
            "loop-guard / runaway re-asks",
            "forced xhigh",
            "thinking-limit cuts",
            "Jeff steps",
            "trim cuts/questions",
            "Jeff latency median/p90",
            "killed at time limit",
            "errors by kind",
        ],
        rows,
    )
    return out


def rerun_split(scope: Scope) -> list[list[str]]:
    rows = []
    for arm in JEFF_ARMS:
        c = scope.comparisons[arm]
        if c is None:
            continue
        for label, subset in (
            ("side by side", [p for p in c.pairs if not p.rerun]),
            ("rerun", [p for p in c.pairs if p.rerun]),
        ):
            if not subset:
                rows.append([arm, label, "0", "-", "-", "-"])
                continue
            s = statistics_of(sum_fields([p.fields for p in subset]))
            rows.append(
                [arm, label, str(len(subset)), f"{pts(s['diff'])} pts", r2(s["all_geo"]), r2(s["total_wall"])]
            )
    return rows


def report(data: Data) -> str:
    pooled = build_scope("pooled", POOLED, data, SEED)
    scopes = {s: build_scope(s, (s,), data, SEED + 100 * (i + 1)) for i, s in enumerate(SECTIONS)}
    tb2 = scopes["terminal-bench-2"]
    out = ["# Tonight's evaluation (2026-10-04/05): final report", ""]
    out.append(
        f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `tools/jeff-first/eval_report.py` from "
        + "; ".join(f"`{p}` ({n} sessions, collected {m})" for p, n, m in data.inputs)
        + "."
    )
    out.append("")
    still = sum(r["state"] == "running" for r in data.active)
    out.append(
        f"**Status: INTERIM, {still} sessions were still running when the lines were collected.**"
        if still
        else "**Status: final, no session was running when the lines were collected.**"
    )
    out.append("")
    out.append(
        "Arms: **a1-baseline** Qwen3.8-27B at xhigh thinking on every turn, Jeff off. **a2-off-guard** thinking off on "
        "every turn plus safeguards (loop guard, stuck output or 2 failed commands -> xhigh, 8,000-token thinking "
        "limit). **a3-jeff07 / a4-jeff06** Jeff routes each turn: off unless Jeff's P(xhigh) >= 0.7 / 0.6; Jeff may "
        "take steps itself; same safeguards. A block is one task attempt run under the arms on one Qwen server; every "
        "comparison pairs an arm's session with the baseline session of the same block. swe-bench-verified ran a1 "
        "and a4 only, so the pooled a2 and a3 lines hold no swe-bench-verified blocks."
    )
    out.append("")
    out.append(
        f"Owner's target: **25% faster at the same pass rate**. Read here as: on average per task (geometric mean of "
        f"per-block wall-time ratios over all paired blocks) the arm takes at most {TARGET_RATIO} x the baseline's "
        "time, and "
        f"the paired pass-rate difference's 95% interval stays above -{100 * SAME_PASS_MARGIN:.0f} points (the "
        "5-point margin is this report's choice; the owner did not set one)."
    )
    out.append("")
    out.append(
        "Speed measures (owner's choice), all paired by block, ratio = arm / baseline, below 1 = faster: **on "
        "average per task** = geometric mean of per-block wall-time ratios over all paired blocks, then the median "
        "per-block ratio; then **total wall time over all paired blocks** (all outcomes; a few long sessions can "
        "dominate it) and total Qwen generation time; then time on tasks the baseline solved (blocks where a1 "
        "passed, the arm's session counted whatever its outcome), both solved, and the median wall time per "
        "session. Each section lists the 10 pairs with the largest time difference per arm."
    )
    out.append("")
    out.append(
        f"Pooled = every round of {', '.join(sorted({s.split('#')[0] for s in POOLED}))}. Not pooled (shown below): "
        + "; ".join(f"{b} ({why})" for b, why in NOT_POOLED.items())
        + ". Sections named BENCHMARK#2 are the second round (attempt 2) of that benchmark; Terminal-Bench 2.0 keeps "
        "its three attempts in one section."
    )
    out.append("")
    out.append("## Headline")
    out.append("")
    for label, scope in (("All pooled benchmarks", pooled), ("Terminal-Bench 2.0", tb2)):
        out.append(f"{label}:")
        out.append("")
        for arm in ARMS[1:]:
            c = scope.comparisons[arm]
            out.append(f"- {headline_summary(c)}" if c else f"- **{arm}**: no paired blocks.")
        out.append("")
    out.append("## Headline tables: pooled and Terminal-Bench 2.0")
    out.append("")
    out += table(HEADLINE_HEADER, headline_rows(pooled) + headline_rows(tb2))
    out.append("")
    out.append("## Verdict per arm")
    out.append("")
    for arm in ARMS[1:]:
        out.append(f"- Pooled. {verdict(pooled.comparisons[arm], arm)}")
    for arm in ARMS[1:]:
        out.append(f"- Terminal-Bench 2.0. {verdict(tb2.comparisons[arm], arm)}")
    out.append("")
    out.append("Per section (the pooled lines above are the ones to act on):")
    out.append("")
    for s in SECTIONS[1:]:
        for arm in ARMS[1:]:
            if scopes[s].comparisons[arm] is not None:
                out.append(f"- {s}. {verdict(scopes[s].comparisons[arm], arm)}")
    out.append("")
    out.append("## Pooled")
    out.append("")
    out += scope_section(pooled)
    out.append("")
    for s in SECTIONS:
        out.append(f"## {s}" + (" (not pooled)" if s not in POOLED else ""))
        out.append("")
        out += scope_section(scopes[s])
        out.append("")
    out += caveats(data, pooled)
    out += left_out_section(data, pooled, scopes)
    out.append("## Method")
    out.append("")
    out.append(
        f"- Pass rate interval: Wilson 95%. Paired intervals: percentile bootstrap, {RESAMPLES} resamples of whole "
        f"tasks (seed {SEED}), so the attempts and rounds of a task move together. McNemar p: exact binomial test on "
        "the blocks only one side passed. An interval marked 'n/a' had resamples where the statistic is undefined "
        "(no block in the set, or no pass)."
    )
    out.append(
        "- Wall time = Harbor's agent execution time (agent start to end, Jeff time included); Qwen generation time = "
        "the sum of the model time of every Qwen request in the trace. A pass is reward 1. Median of per-block "
        "ratios: with an even count, the two middle ratios are averaged geometrically."
    )
    out.append(
        "- Behaviour counts: turns = first-attempt Qwen requests; turns at off = those sent at thinking off; "
        "loop-guard / runaway re-asks = guard triggers; forced xhigh = turns the stuck/failed-command rule raised "
        "to xhigh; thinking-limit cuts = replies cut at 8,000 thinking tokens; Jeff latency = time per Jeff question "
        "(router: thinking level; step: per candidate level; trim: output shortening)."
    )
    out.append(
        "- Errors: trial exceptions other than the agent time limit, JeffFirst errors, missing pi output or result. "
        "Sessions with an error but a reward stay in (scored as their reward); sessions without a reward are left "
        "out with their block and listed below."
    )
    out.append("")
    return "\n".join(out)


def caveats(data: Data, pooled: Scope) -> list[str]:
    out = ["## Caveats", ""]
    reruns = {
        arm: sum(p.rerun for p in (pooled.comparisons[arm].pairs if pooled.comparisons[arm] else []))
        for arm in JEFF_ARMS
    }
    out.append(
        "- **Jeff reruns ran later than their paired a1/a2 sessions.** Jeff-arm sessions that started before the Jeff "
        f"capacity fix (B200 22:59, casdgx01 23:01) are left out ({len(data.superseded)} sessions) and their task "
        "attempts were rerun on the same Qwen server, but hours later, so server load differs within those pairs. "
        f"Pooled paired blocks that are reruns: {', '.join(f'{a} {n}' for a, n in reruns.items())}. Split "
        "(per-task geometric mean; total wall over the paired blocks):"
    )
    out.append("")
    out += [
        "  " + line
        for line in table(["arm", "blocks", "n", "pass diff", "per-task geomean", "total wall"], rerun_split(pooled))
    ]
    out.append("")
    for (benchmark, task), why in EXCLUDED_TASKS.items():
        sessions = [r for r in data.excluded if (r["benchmark"], r["task"]) == (benchmark, task)]
        outcome = ", ".join(
            f"{arm} {sum(r.get('reward') == 1 for r in sessions if r['arm'] == arm and r['state'] == 'finished')}"
            f"/{sum(r['arm'] == arm and r['state'] == 'finished' for r in sessions)} passed"
            for arm in ARMS
        )
        out.append(f"- **{benchmark} {task} is left out of every comparison**: {why}. Outcomes: {outcome}.")
    killed_blocks = len({r["pair_block"] for r in data.killed_blocks})
    out.append(
        f"- **Blocks with a system kill are left out (owner's ruling).** {killed_blocks} blocks in which a session was "
        "killed by the system (agent command exit 137, SIGKILL, e.g. out of memory) are left out of every pass rate "
        "and time measure in all arms, identified from the recorded exit code, not the outcome. A kill in a Jeff "
        "rerun takes out the original block's a1/a2 sessions too. Sessions whose exception message holds no exit "
        f"code ({len(data.unrecorded_exit)}) cannot be checked and stay in; they are listed below."
    )
    limit = {arm: arm_profile(pooled.sessions[arm])["time_limit"] for arm in ARMS}
    out.append(
        "- **Sessions killed at the agent time limit** count as their verifier reward (usually 0) and their full wall "
        "time. Pooled per arm: " + ", ".join(f"{a} {n}" for a, n in limit.items()) + ". A killed session whose "
        "verifier gave no reward is left out with its block (listed below)."
    )
    running = sum(r["state"] == "running" for r in data.active)
    out.append(
        f"- **B200 sessions cut and unfinished sessions.** {len(data.cut)} sessions were cut (still running when the "
        f"B200 evaluation stopped) and {running} were still running when these lines were collected; both are left "
        "out and their blocks are unpaired. Which tasks finished in a partly run round is not random (short tasks "
        "finish first), and a cut removes the longest sessions of the arms that were still running."
    )
    out.append(
        "- Hosts differ (B200 NVFP4 vLLM 0.29, casdgx01 H100 FP8 vLLM 0.30) and so do their loads; pairing by block "
        "keeps both sessions of a pair on the same Qwen server."
    )
    out.append(
        "- The trim adapter never shortened any output (its top choice was 'keep all' on every question), so trim "
        "questions only cost Jeff time in a3/a4."
    )
    out.append("")
    return out


def left_out_section(data: Data, pooled: Scope, scopes: dict[str, Scope]) -> list[str]:
    out = ["## Left out (every session or block not in the comparisons)", ""]
    for label, rows in (
        ("Superseded (Jeff arm before the capacity fix, rerun instead)", data.superseded),
        ("Cut (still running when the B200 evaluation stopped)", data.cut),
    ):
        counts = Counter((r["section"], r["arm"]) for r in rows)
        out.append(
            f"- {label}: {len(rows)} sessions"
            + (": " + ", ".join(f"{s} {a} {n}" for (s, a), n in sorted(counts.items())) if rows else "")
            + "."
        )
    killed = [r for r in data.killed_blocks if exit_code(r) == KILL_EXIT]
    blocks_by_section = Counter(s for s, _ in {(r["section"], r["pair_block"]) for r in data.killed_blocks})
    out.append(
        f"- Blocks with a session killed by the system (exit 137), left out in all arms: "
        f"{sum(blocks_by_section.values())} blocks, {len(data.killed_blocks)} sessions; per section: "
        + (", ".join(f"{s} {n}" for s, n in sorted(blocks_by_section.items())) or "none")
        + ". Killed sessions (section, task, block, arm, server, start, exception):"
    )
    for r in sorted(killed, key=lambda r: (r["section"], r["task"], r["pair_block"], r["arm"])):
        out.append(
            f"  - {r['section']} {r['task']} {r['pair_block']} {r['arm']} {r['server']} {r['started']} {r['exception']}"
        )
    out.append(
        f"- Sessions with an exception but no recorded exit code (kept in; mostly failures before pi started): "
        f"{len(data.unrecorded_exit)}: "
        + ", ".join(
            f"{r['section']} {r['task']} {r['arm']} ({r['exception']})"
            for r in sorted(data.unrecorded_exit, key=lambda r: (r["section"], r["task"], r["arm"]))
        )
        + "."
    )
    out.append(
        f"- Excluded task (see caveats): {len(data.excluded)} sessions: "
        + ", ".join(sorted(f"{r['arm']} {r['task']} attempt {r['attempt']}" for r in data.excluded))
        + "."
    )
    running = sorted(
        f"{r['section']} {r['arm']} {r['task']} attempt {r['attempt']}" for r in data.active if r["state"] == "running"
    )
    out.append(
        f"- Still running when collected: {len(running)} sessions" + (": " + ", ".join(running) if running else "") + "."
    )
    unscored = [r for r in data.active if r["state"] == "finished" and r.get("reward") is None]
    out.append(f"- Finished without a reward: {len(unscored)} sessions:")
    for r in sorted(unscored, key=lambda r: (r["section"], r["task"], r["arm"])):
        wall = "-" if r.get("agent_s") is None else f"{r['agent_s']:.0f} s"
        out.append(
            f"  - {r['section']} {r['task']} attempt {r['attempt']} {r['arm']} ({r['server']}, wall {wall}): "
            f"{r.get('exception') or 'no exception'}; {(r.get('error') or 'no error').splitlines()[0][:140]}"
        )
    out.append("- Unpaired or unscored blocks per arm comparison (task, block: reason):")
    for s in SECTIONS:
        for arm in ARMS[1:]:
            items = scopes[s].left_out[arm]
            if items:
                out.append(
                    f"  - {s} {arm} ({len(items)}): " + "; ".join(f"{task} {block}: {why}" for (_, task, block), why in items)
                )
    zero = [
        f"{arm} {p.qwen_zero_note}"
        for arm in ARMS[1:]
        if pooled.comparisons[arm]
        for p in pooled.comparisons[arm].pairs
        if p.qwen_zero_note
    ]
    out.append(
        f"- Paired blocks out of the per-block Qwen ratios (a side with zero Qwen time; still in the totals): {len(zero)}"
        + (": " + "; ".join(zero) if zero else "")
        + "."
    )
    out.append(
        "- Pooled totals left out: " + ", ".join(f"{a} {len(pooled.left_out[a])} blocks" for a in ARMS[1:]) + "."
    )
    out.append("")
    return out


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: eval_report.py OUT.md UNITS.jsonl...")
    data = load(sys.argv[2:])
    Path(sys.argv[1]).write_text(report(data))
    print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
