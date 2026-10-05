"""Final report of tonight's four-arm evaluation (2026-10-04/05) from eval_units.py lines of every host.

Arms: a1-baseline (Qwen at xhigh thinking, Jeff off), a2-off-guard (thinking off with safeguards), a3-jeff07 and
a4-jeff06 (Jeff routes thinking at P(xhigh) >= 0.7 / 0.6, takes steps, may shorten tool output). A block is one task
attempt run under every arm on one Qwen server; a Jeff-arm rerun (block id ending in "r") pairs with the original
block's a1/a2 sessions. Sessions marked superseded (Jeff arms started before the Jeff capacity fix) are left out.

Written per benchmark and pooled over the benchmarks with a meaningful baseline pass rate (POOLED):
1. pass rate per arm (Wilson 95% interval) and the paired difference to the baseline over blocks where both sessions
   have a reward (cluster bootstrap 95% interval; baseline-only vs arm-only passes; exact McNemar p);
2. speed, paired by block: wall-time ratio on blocks both solved (geometric mean, median); total wall time and total
   Qwen generation time over the same paired blocks, all outcomes; time to solve (total wall time / passes);
3. per arm: turns at thinking off, guard hits, forced xhigh, thinking-limit cuts, Jeff steps, Jeff question latency,
   time-limit kills and errors by kind;
4. a verdict per arm against the owner's target (25% faster at the same pass rate);
5. caveats, and every session or block left out with the reason.

Intervals resample tasks (all attempts and arms of a task together), so the three TB2 attempts of a task are not
treated as independent. Wall time is the agent's execution time (agent start to agent end), as in eval_summary.py.
A pass is reward 1 (partial rewards such as 0.667 count as not passed).

Usage: python3 eval_report.py OUT.md UNITS.jsonl...
"""

import datetime as dt
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BASE = "a1-baseline"
ARMS = (BASE, "a2-off-guard", "a3-jeff07", "a4-jeff06")
JEFF_ARMS = ("a3-jeff07", "a4-jeff06")
# Order of the sections: Terminal-Bench 2.0, the follow-on benchmarks in run order, then the two moved to the end.
BENCHMARKS = (
    "terminal-bench-2",
    "harbor-index-1.0",
    "skillsbench",
    "terminal-bench-pro",
    "swe-rebench-leaderboard",
    "terminal-bench",
    "terminal-bench-science",
)
NOT_POOLED = {
    "terminal-bench": "baseline pass rate near zero, so it cannot show a pass-rate difference",
    "terminal-bench-science": "baseline pass rate near zero, so it cannot show a pass-rate difference",
}
POOLED = tuple(b for b in BENCHMARKS if b not in NOT_POOLED)
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
    total: dict = {}
    for item in items:
        for key, value in item.items():
            total[key] = total.get(key, 0) + value
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
    """Percentile 95% intervals by resampling whole clusters (tasks) with replacement; each cluster is a dict of summed
    fields and each statistic is computed from the fields summed over a resample."""
    if not clusters:
        raise ValueError("bootstrap of no clusters")
    keys = sorted(clusters[0])
    if any(sorted(c) != keys for c in clusters):
        raise ValueError("bootstrap clusters with different fields")
    rows = [tuple(c[k] for k in keys) for c in clusters]
    rng = random.Random(seed)
    values: dict[str, list[float]] = {name: [] for name in statistics_}
    undefined = dict.fromkeys(statistics_, 0)
    for _ in range(resamples):
        sample = rng.choices(rows, k=len(rows))
        totals = dict(zip(keys, (sum(column) for column in zip(*sample))))
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
    base_wall: float
    arm_wall: float


def passed(row: dict) -> bool:
    return row["reward"] == 1


def pair_up(rows: list[dict], arm: str) -> tuple[list[Pair], list[tuple[tuple, str]]]:
    """Pairs every block's finished baseline and arm sessions. Blocks where either side has no finished session or no
    reward are left out and returned with the reason (running sessions have no block yet and are counted elsewhere)."""
    sides: dict[str, dict[str, dict]] = {BASE: {}, arm: {}}
    for r in rows:
        if r.get("superseded"):
            raise ValueError(f"superseded session passed to pair_up: {r.get('folder')}")
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
            if r.get("agent_s") is None:
                raise ValueError(f"block {block}: {name} session has a reward but no agent time: {r.get('folder')}")
        bp, ap = passed(base), passed(other)
        both = bp and ap
        has_qwen = "qwen_s" in base and "qwen_s" in other
        fields = {
            "n": 1,
            "base_pass": int(bp),
            "arm_pass": int(ap),
            "base_only": int(bp and not ap),
            "arm_only": int(ap and not bp),
            "solved_both": int(both),
            "log_ratio": math.log(other["agent_s"] / base["agent_s"]) if both else 0.0,
            "base_wall": base["agent_s"],
            "arm_wall": other["agent_s"],
            "qwen_n": int(has_qwen),
            "base_qwen": base["qwen_s"] if has_qwen else 0.0,
            "arm_qwen": other["qwen_s"] if has_qwen else 0.0,
        }
        pairs.append(
            Pair(
                cluster=(base["benchmark"], base["task"]),
                block=block,
                rerun=other["block"].endswith("r"),
                fields=fields,
                base_wall=base["agent_s"],
                arm_wall=other["agent_s"],
            )
        )
    return pairs, left_out


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


PAIR_STATISTICS: dict[str, Callable[[dict], float | None]] = {
    "diff": lambda t: (t["arm_pass"] - t["base_pass"]) / t["n"],
    "geo": lambda t: math.exp(t["log_ratio"] / t["solved_both"]) if t["solved_both"] else None,
    "wall": lambda t: _ratio(t["arm_wall"], t["base_wall"]),
    "qwen": lambda t: _ratio(t["arm_qwen"], t["base_qwen"]),
    "tts": lambda t: (
        (t["arm_wall"] / t["arm_pass"]) / (t["base_wall"] / t["base_pass"]) if t["arm_pass"] and t["base_pass"] else None
    ),
}


@dataclass
class Comparison:
    arm: str
    pairs: list[Pair]
    totals: dict
    point: dict[str, float | None]
    intervals: dict[str, Interval]
    solved_ratios: list[float]


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
        point={name: f(totals) for name, f in PAIR_STATISTICS.items()},
        intervals=bootstrap_intervals(clusters, PAIR_STATISTICS, RESAMPLES, seed),
        solved_ratios=[p.arm_wall / p.base_wall for p in pairs if p.fields["solved_both"]],
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
    active: list[dict]  # not superseded, not of an excluded task
    superseded: list[dict]
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
        if r["benchmark"] not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {r['benchmark']!r} in {r.get('folder')}")
        if r["state"] not in ("finished", "running"):
            raise ValueError(f"unknown state {r['state']!r} in {r.get('folder')}")
    folders = Counter(r["folder"] for r in rows)
    twice = [f for f, n in folders.items() if n > 1]
    if twice:
        raise ValueError(f"sessions listed twice (same input given twice?): {twice[:5]}")
    blocks: dict[str, tuple[str, str]] = {}
    for r in rows:
        if r["state"] == "finished":
            seen = blocks.setdefault(r["pair_block"], (r["benchmark"], r["task"]))
            if seen != (r["benchmark"], r["task"]):
                raise ValueError(f"block {r['pair_block']} holds two tasks: {seen} and {(r['benchmark'], r['task'])}")
    superseded = [r for r in rows if r.get("superseded")]
    rest = [r for r in rows if not r.get("superseded")]
    excluded = [r for r in rest if (r["benchmark"], r["task"]) in EXCLUDED_TASKS]
    active = [r for r in rest if (r["benchmark"], r["task"]) not in EXCLUDED_TASKS]
    return Data(active=active, superseded=superseded, excluded=excluded, inputs=inputs)


# ------------------------------------------------------------------------------------------------------------- format


def pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def interval_text(iv: Interval, fmt: Callable[[float], str]) -> str:
    if iv.undefined:
        return f"n/a ({iv.undefined}/{RESAMPLES} resamples undefined)"
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
    diff_iv, wall_iv, tts_iv = c.intervals["diff"], c.intervals["wall"], c.intervals["tts"]
    wall = c.point["wall"]
    parts = []
    if wall_iv.undefined:
        speed = "speed interval undefined"
    elif wall_iv.hi <= TARGET_RATIO:
        speed = "speed target met (whole interval at or below 0.75)"
    elif wall <= TARGET_RATIO and wall_iv.hi < 1:
        speed = "speed target met on the estimate, but the interval reaches above 0.75"
    elif wall_iv.hi < 1:
        speed = "faster than baseline, but short of the 25% target"
    elif wall_iv.lo > 1:
        speed = "slower than baseline"
    else:
        speed = "no clear speed difference"
    parts.append(
        f"total wall time {r2(wall)} x baseline ({interval_text(wall_iv, r2)}); "
        f"time per solve {r2(c.point['tts'])} x ({interval_text(tts_iv, r2)}): {speed}."
    )
    if diff_iv.undefined:
        same = "pass-rate interval undefined"
    elif diff_iv.lo >= -SAME_PASS_MARGIN:
        same = "same pass rate shown (interval stays above -5 points)"
    elif diff_iv.hi < 0:
        same = "pass rate lower than baseline"
    else:
        same = "a pass-rate loss of more than 5 points is not ruled out"
    parts.append(f"Pass rate {pts(c.point['diff'])} pts ({interval_text(diff_iv, pts)}): {same}.")
    met = (
        not wall_iv.undefined
        and not diff_iv.undefined
        and wall <= TARGET_RATIO
        and wall_iv.hi < 1
        and diff_iv.lo >= -SAME_PASS_MARGIN
    )
    thin = t["n"] < THIN_PAIRS or (not diff_iv.undefined and diff_iv.hi - diff_iv.lo > THIN_WIDTH)
    overall = "MEETS the target" if met else "does NOT meet the target"
    if thin:
        overall += f" -- data too thin to rely on ({t['n']} paired blocks; pass-rate interval width " + (
            "undefined)" if diff_iv.undefined else f"{100 * (diff_iv.hi - diff_iv.lo):.0f} pts)"
        )
    return f"**{arm}: {overall}.** " + " ".join(parts)


# ------------------------------------------------------------------------------------------------------------- report


@dataclass
class Scope:
    name: str
    benchmarks: tuple[str, ...]
    comparisons: dict[str, Comparison | None]
    left_out: dict[str, list[tuple[tuple, str]]]
    sessions: dict[str, list[dict]]  # arm -> finished sessions
    running: Counter


def build_scope(name: str, benchmarks: tuple[str, ...], data: Data, seed: int) -> Scope:
    rows = [r for r in data.active if r["benchmark"] in benchmarks]
    comparisons, left_out = {}, {}
    for i, arm in enumerate(ARMS[1:]):
        pairs, out = pair_up(rows, arm)
        comparisons[arm] = compare(pairs, arm, seed + i)
        left_out[arm] = out
    return Scope(
        name=name,
        benchmarks=benchmarks,
        comparisons=comparisons,
        left_out=left_out,
        sessions={arm: [r for r in rows if r["arm"] == arm and r["state"] == "finished"] for arm in ARMS},
        running=Counter(r["arm"] for r in rows if r["state"] == "running"),
    )


def headline_rows(scope: Scope) -> list[list[str]]:
    out = [[scope.name, BASE, pass_rate_cell(scope.sessions[BASE]), "-", "-", "-", "-", "-", "-"]]
    for arm in ARMS[1:]:
        c = scope.comparisons[arm]
        if c is None:
            out.append([scope.name, arm, pass_rate_cell(scope.sessions[arm]), "0", "-", "-", "-", "-", "-"])
            continue
        out.append(
            [
                scope.name,
                arm,
                pass_rate_cell(scope.sessions[arm]),
                str(c.totals["n"]),
                diff_cell(c),
                f"{ratio_cell(c, 'geo')} n={c.totals['solved_both']}",
                ratio_cell(c, "wall"),
                ratio_cell(c, "qwen"),
                ratio_cell(c, "tts"),
            ]
        )
    return out


HEADLINE_HEADER = [
    "scope",
    "arm",
    "pass rate (95% interval)",
    "paired blocks",
    "pass diff vs baseline (95%); baseline-only vs arm-only passes",
    "wall ratio, both solved (geomean)",
    "total wall ratio",
    "total Qwen gen ratio",
    "time-per-solve ratio",
]


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
    out.append("**1. Pass rate**")
    out.append("")
    rows = []
    for arm in ARMS:
        c = scope.comparisons.get(arm)
        rows.append(
            [
                arm,
                pass_rate_cell(scope.sessions[arm]),
                "-" if arm == BASE else (str(c.totals["n"]) if c else "0"),
                "-" if arm == BASE or c is None else f"{pct(c.totals['base_pass'] / c.totals['n'])} -> "
                f"{pct(c.totals['arm_pass'] / c.totals['n'])}",
                "-" if arm == BASE or c is None else diff_cell(c),
            ]
        )
    out += table(
        [
            "arm",
            "pass rate, all scored sessions (95% interval)",
            "paired blocks",
            "baseline -> arm on paired blocks",
            "difference (95%); baseline-only vs arm-only; McNemar p",
        ],
        rows,
    )
    out.append("")
    out.append("**2. Speed (paired by block; ratio = arm / baseline, below 1 is faster)**")
    out.append("")
    rows = []
    for arm in ARMS[1:]:
        c = scope.comparisons[arm]
        if c is None:
            rows.append([arm] + ["-"] * 7)
            continue
        t = c.totals
        rows.append(
            [
                arm,
                f"{t['solved_both']}: geomean {ratio_cell(c, 'geo')}, median "
                f"{r2(statistics.median(c.solved_ratios)) if c.solved_ratios else '-'}",
                f"{hours(t['base_wall'])} -> {hours(t['arm_wall'])}",
                ratio_cell(c, "wall"),
                f"{hours(t['base_qwen'])} -> {hours(t['arm_qwen'])} (n={t['qwen_n']})",
                ratio_cell(c, "qwen"),
                f"{t['base_wall'] / t['base_pass'] / 60:.1f} -> {t['arm_wall'] / t['arm_pass'] / 60:.1f} min"
                if t["base_pass"] and t["arm_pass"]
                else "-",
                ratio_cell(c, "tts"),
            ]
        )
    out += table(
        [
            "arm",
            "(a) both solved: blocks, wall ratio (95%), median",
            "(b) total wall, baseline -> arm",
            "total wall ratio (95%)",
            "total Qwen gen, baseline -> arm",
            "Qwen gen ratio (95%)",
            "(c) wall per pass, baseline -> arm",
            "time-per-solve ratio (95%)",
        ],
        rows,
    )
    out.append("")
    out.append("**3. Behaviour per arm (all finished sessions of the scope)**")
    out.append("")
    rows = []
    for arm in ARMS:
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
        for label, subset in (("side by side", [p for p in c.pairs if not p.rerun]), ("rerun", [p for p in c.pairs if p.rerun])):
            if not subset:
                rows.append([arm, label, "0", "-", "-", "-"])
                continue
            t = sum_fields([p.fields for p in subset])
            solved = [p.arm_wall / p.base_wall for p in subset if p.fields["solved_both"]]
            rows.append(
                [
                    arm,
                    label,
                    str(t["n"]),
                    f"{pts((t['arm_pass'] - t['base_pass']) / t['n'])} pts",
                    r2(t["arm_wall"] / t["base_wall"]),
                    f"{r2(geomean(solved))} (n={len(solved)})" if solved else "-",
                ]
            )
    return rows


def report(data: Data) -> str:
    pooled = build_scope("pooled", POOLED, data, SEED)
    scopes = {b: build_scope(b, (b,), data, SEED + 100 * (i + 1)) for i, b in enumerate(BENCHMARKS)}
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
        "take steps itself; same safeguards. A block is one task attempt run under all four arms on one Qwen server; "
        "every comparison below pairs an arm's session with the baseline session of the same block."
    )
    out.append("")
    out.append(
        f"Owner's target: **25% faster (time ratio <= {TARGET_RATIO}) at the same pass rate**. Read here as: total "
        "wall time over the paired blocks (all outcomes) at most 0.75 x baseline, and the paired pass-rate "
        f"difference's 95% interval staying above -{100 * SAME_PASS_MARGIN:.0f} points (the 5-point margin is this "
        "report's choice; the owner did not set one)."
    )
    out.append("")
    out.append(
        f"Pooled = {', '.join(POOLED)}. Not pooled (shown below): "
        + "; ".join(f"{b} ({why})" for b, why in NOT_POOLED.items())
        + "."
    )
    out.append("")
    out.append("## Headline: pooled and Terminal-Bench 2.0")
    out.append("")
    out += table(HEADLINE_HEADER, headline_rows(pooled) + headline_rows(tb2))
    out.append("")
    out.append(
        "Ratios are arm / baseline (below 1 = faster) with 95% intervals; the pass difference is in percentage "
        "points. 'Both solved' compares only blocks both arms passed (geometric mean of per-block ratios). 'Total "
        "wall' and 'total Qwen gen' sum every paired block whatever the outcome (the throughput view). "
        "'Time per solve' = total wall time / passes, arm over baseline."
    )
    out.append("")
    out.append("## Verdict per arm")
    out.append("")
    for arm in ARMS[1:]:
        out.append(f"- Pooled. {verdict(pooled.comparisons[arm], arm)}")
    for arm in ARMS[1:]:
        out.append(f"- Terminal-Bench 2.0. {verdict(tb2.comparisons[arm], arm)}")
    out.append("")
    out.append("Per benchmark (one line per arm; the pooled lines above are the ones to act on):")
    out.append("")
    for b in BENCHMARKS[1:]:
        for arm in ARMS[1:]:
            out.append(f"- {b}. {verdict(scopes[b].comparisons[arm], arm)}")
    out.append("")
    out.append("## Pooled")
    out.append("")
    out += scope_section(pooled)
    out.append("")
    for b in BENCHMARKS:
        out.append(f"## {b}" + (" (not pooled)" if b in NOT_POOLED else ""))
        out.append("")
        out += scope_section(scopes[b])
        out.append("")
    out += caveats(data, pooled, tb2)
    out += left_out_section(data, pooled, scopes)
    out.append("## Method")
    out.append("")
    out.append(
        f"- Pass rate interval: Wilson 95%. Paired intervals: percentile bootstrap, {RESAMPLES} resamples of whole "
        f"tasks (seed {SEED}), so the three Terminal-Bench 2.0 attempts of a task move together. McNemar p: exact "
        "binomial test on the blocks only one side passed. An interval marked 'n/a' had resamples where the "
        "statistic is undefined (no block both solved, or no pass)."
    )
    out.append(
        "- Wall time = Harbor's agent execution time (agent start to end, Jeff time included); Qwen generation time = "
        "the sum of the model time of every Qwen request in the trace. A pass is reward 1."
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


def caveats(data: Data, pooled: Scope, tb2: Scope) -> list[str]:
    out = ["## Caveats", ""]
    reruns = {arm: sum(p.rerun for p in (pooled.comparisons[arm].pairs if pooled.comparisons[arm] else [])) for arm in JEFF_ARMS}
    out.append(
        "- **Jeff reruns ran later than their paired a1/a2 sessions.** Jeff-arm sessions that started before the Jeff "
        f"capacity fix (B200 22:59, casdgx01 23:01) are left out ({len(data.superseded)} sessions) and their task "
        "attempts were rerun on the same Qwen server, but hours later, so server load differs within those pairs. "
        f"Pooled paired blocks that are reruns: {', '.join(f'{a} {n}' for a, n in reruns.items())}. Split:"
    )
    out.append("")
    out += ["  " + line for line in table(
        ["arm", "blocks", "n", "pass diff", "total wall ratio", "both-solved geomean"], rerun_split(pooled)
    )]
    out.append("")
    excluded = Counter((r["benchmark"], r["task"], r["arm"], r.get("reward")) for r in data.excluded if r["state"] == "finished")
    for (benchmark, task), why in EXCLUDED_TASKS.items():
        outcome = ", ".join(
            f"{arm} {sum(n for (b, t, a, rw), n in excluded.items() if (b, t, a) == (benchmark, task, arm) and rw == 1)}"
            f"/{sum(n for (b, t, a, _), n in excluded.items() if (b, t, a) == (benchmark, task, arm))} passed"
            for arm in ARMS
        )
        out.append(f"- **{benchmark} {task} is left out of every comparison**: {why}. Outcomes: {outcome}.")
    limit = {arm: arm_profile(pooled.sessions[arm])["time_limit"] for arm in ARMS}
    out.append(
        "- **Sessions killed at the agent time limit** count as their verifier reward (usually 0) and their full wall "
        "time. Pooled per arm: " + ", ".join(f"{a} {n}" for a, n in limit.items()) + ". A killed session whose "
        "verifier gave no reward is left out with its block (listed below)."
    )
    running = Counter((r.get("benchmark"), r["arm"]) for r in data.active if r["state"] == "running")
    out.append(
        "- **B200 sessions cut at 08:15 and unfinished sessions.** The B200 started a session only if its longest "
        "possible duration ended by 08:15 BST; blocks that no longer fit moved to casdgx01. Sessions still running "
        f"when these lines were collected: {sum(running.values())} (left out; their blocks are unpaired). The "
        "follow-on benchmarks are partly finished, swe-rebench-leaderboard most of all, and which of their tasks "
        "finished is not random (short tasks finish first)."
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
    sup = Counter((r["benchmark"], r["arm"]) for r in data.superseded)
    out.append(
        f"- Superseded (Jeff arm before the capacity fix, rerun instead): {len(data.superseded)} sessions: "
        + ", ".join(f"{b} {a} {n}" for (b, a), n in sorted(sup.items()))
        + "."
    )
    out.append(
        f"- Excluded task (see caveats): {len(data.excluded)} sessions: "
        + ", ".join(sorted(f"{r['arm']} {r['task']} attempt {r['attempt']}" for r in data.excluded))
        + "."
    )
    running = sorted(f"{r['benchmark']} {r['arm']} {r['task']} attempt {r['attempt']}" for r in data.active if r["state"] == "running")
    out.append(f"- Still running when collected: {len(running)} sessions" + (": " + ", ".join(running) if running else "") + ".")
    unscored = [r for r in data.active if r["state"] == "finished" and r.get("reward") is None]
    out.append(f"- Finished without a reward: {len(unscored)} sessions:")
    for r in sorted(unscored, key=lambda r: (r["benchmark"], r["task"], r["arm"])):
        wall = "-" if r.get("agent_s") is None else f"{r['agent_s']:.0f} s"
        out.append(
            f"  - {r['benchmark']} {r['task']} attempt {r['attempt']} {r['arm']} ({r['server']}, wall {wall}): "
            f"{r.get('exception') or 'no exception'}; {(r.get('error') or 'no error').splitlines()[0][:140]}"
        )
    out.append("- Unpaired or unscored blocks per arm comparison (benchmark, task, block: reason):")
    for b in BENCHMARKS:
        for arm in ARMS[1:]:
            items = scopes[b].left_out[arm]
            if items:
                out.append(
                    f"  - {b} {arm} ({len(items)}): "
                    + "; ".join(f"{task} {block}: {why}" for (_, task, block), why in items)
                )
    pooled_counts = {arm: len(pooled.left_out[arm]) for arm in ARMS[1:]}
    out.append("- Pooled totals left out: " + ", ".join(f"{a} {n} blocks" for a, n in pooled_counts.items()) + ".")
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
