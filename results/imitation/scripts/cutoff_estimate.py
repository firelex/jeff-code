"""Offline estimate of a "stop this session now" rule for Jeff-Code, from tonight's evaluation sessions (no new runs).

Question: if a session that is still running at a checkpoint (10, 15, 20, 30, 45 or 60 minutes after the agent
started) looks like it will fail, stopping it there saves time. How much total time does such a rule save for
a4-jeff06 against a1-baseline, and how many passes does it lose?

Inputs: per-session lines of eval_units.py (eval_summary.sh) and per-session checkpoint features of
cutoff_features.py, both from every host; results/imitation/task-sets-inventory.json for each task's agent time limit.

Sessions used: finished, not superseded, not cut, with a reward, in the pooled benchmarks (all but terminal-bench and
terminal-bench-science; the task excluded by eval_report.py stays excluded). Models train on sessions of all four
arms. A row is one session at one checkpoint it was still running at; the label is whether the session finally
passed (reward 1).

Splits (seed 20261005): tasks (benchmark + task name; all attempts, rounds and arms of a task together) are shuffled;
the first 20% are held out before anything is fitted or chosen; the rest are dealt into 5 folds. Every fitted model
(feature selection included) sees only training folds; the "cross-validated" numbers use each row's prediction from
the model that did not see its task. Policies (checkpoints and threshold) are chosen on the cross-validated numbers,
then checked once on the held-out tasks with models fitted on all non-held-out tasks.

Models, each giving P(pass) for a row:
- turn table: the training pass rate of the row's cell (checkpoint x number of Qwen turns so far), smoothed;
- decision tree: two levels of yes/no conditions on the features, leaves give the training pass rate;
- logistic regression: L2-regularised, at most 8 features chosen by forward selection on cross-validated log-loss.

Policy: at the checked checkpoints (one checkpoint, or every checkpoint from a start), stop the session at the first
one where P(pass) < threshold. A stopped session counts as failed and costs its setup time plus the checkpoint time
instead of its full trial time (setup + agent + verifier). Time ratio = total a4 time / total a1 time over paired
blocks (one a1 and one a4 session of the same task attempt on the same server), all outcomes. Pass loss = passes the
rule throws away, in points of the paired a4 sessions.

Usage: python3 cutoff_estimate.py OUT.md --units U.jsonl... --features F.jsonl...
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20261005
CHECKPOINTS = (10, 15, 20, 30, 45, 60)
BASE, JEFF = "a1-baseline", "a4-jeff06"
ARMS = (BASE, "a2-off-guard", "a3-jeff07", JEFF)
NOT_POOLED = ("terminal-bench", "terminal-bench-science")
EXCLUDED_TASKS = {("terminal-bench-2", "pytorch-model-recovery")}  # as in tools/jeff-first/eval_report.py
HOLDOUT_SHARE = 0.2
FOLDS = 5
MAX_FEATURES = 8
L2 = 1.0
THRESHOLDS = tuple(round(0.01 * i, 2) for i in range(1, 61))
BUDGETS = (0.0, 2.0)  # pass loss in points
TURN_BINS = (0, 5, 10, 20, 30, 45, 70, 100, 150)
INVENTORY = Path(__file__).resolve().parent.parent / "task-sets-inventory.json"

LOG_FEATURES = (
    "turns", "turns_last5min", "jeff_steps", "guard_hits", "forced_xhigh", "limit_cuts", "thinking_chars",
    "output_tokens", "context_tokens", "compactions", "commands", "failed_cmds", "tests", "tests_recent",
    "file_changes", "file_changes_recent", "written_paths", "write_chars", "tracebacks", "compile_errors", "timeouts",
    "not_found", "last_output_chars", "approach_phrases", "approach_recent", "done_phrases",
)
RAW_FEATURES = (
    "elapsed_min", "turns_per_min", "off_share", "fail_share_recent", "repeat_share", "near_repeat_share",
    "last_test_failed", "test_counts_seen", "last_test_pass_frac", "test_frac_trend", "since_change_min",
    "error_share_recent", "limit_min", "share_of_limit",
)
GROUP_FEATURES = ("benchmark", "arm")
CANDIDATES = RAW_FEATURES + LOG_FEATURES + GROUP_FEATURES


# ------------------------------------------------------------------------------------------------------------ loading


def section_of(row: dict) -> str:
    attempt = str(row["attempt"]).removesuffix("r")
    if row["benchmark"] == "terminal-bench-2" or attempt == "1":
        return row["benchmark"]
    return f"{row['benchmark']}#{attempt}"


def short_task(task: str) -> str:
    return task.split(":")[-1]


def read_lines(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        lines = [x for x in Path(path).read_text().split("\n") if x.strip()]
        if not lines:
            raise ValueError(f"{path} holds no lines")
        rows.extend(json.loads(x) for x in lines)
    return rows


def time_limits() -> dict[tuple[str, str], float]:
    limits = {}
    for item in json.loads(INVENTORY.read_text()):
        key = (item["dataset"], item["task"])
        if key in limits and limits[key] != float(item["agent_timeout_sec"]):
            raise ValueError(f"two time limits for {key} in {INVENTORY}")
        limits[key] = float(item["agent_timeout_sec"])
    return limits


def load(unit_paths: list[str], feature_paths: list[str]) -> tuple[list[dict], list[tuple[str, str]], Counter]:
    """Sessions used, with their features; the sessions left out (folder, reason); counts of rows not considered."""
    units = read_lines(unit_paths)
    features = {}
    for r in read_lines(feature_paths):
        if r["folder"] in features:
            raise ValueError(f"features listed twice for {r['folder']}")
        features[r["folder"]] = r
    limits = time_limits()
    skipped: Counter = Counter()
    left_out: list[tuple[str, str]] = []
    sessions = []
    seen = set()
    for u in units:
        if u["folder"] in seen:
            raise ValueError(f"session listed twice: {u['folder']}")
        seen.add(u["folder"])
        if u["arm"] not in ARMS:
            raise ValueError(f"unknown arm {u['arm']} in {u['folder']}")
        if u.get("superseded"):
            skipped["superseded"] += 1
            continue
        if u.get("cut"):
            skipped["cut"] += 1
            continue
        if u["state"] != "finished":
            skipped[f"state {u['state']}"] += 1
            continue
        if u["benchmark"] in NOT_POOLED:
            skipped[f"benchmark {u['benchmark']} (not pooled)"] += 1
            continue
        task = short_task(u["task"])
        if (u["benchmark"], task) in EXCLUDED_TASKS:
            skipped["excluded task"] += 1
            continue
        if u.get("reward") is None:
            left_out.append((u["folder"], f"no reward ({u.get('error') or u.get('exception') or 'no error given'})"[:200]))
            continue
        f = features.get(u["folder"])
        if f is None:
            left_out.append((u["folder"], "no feature line (host not read?)"))
            continue
        if f.get("left_out"):
            left_out.append((u["folder"], f"features: {f['left_out']}"))
            continue
        if abs(f["agent_s"] - u["agent_s"]) > 1:
            raise ValueError(f"{u['folder']}: agent time {u['agent_s']} in units vs {f['agent_s']} in features")
        limit_key = (u["benchmark"], task)
        if limit_key not in limits:
            raise ValueError(f"no agent time limit in {INVENTORY.name} for {limit_key}")
        limit_min = limits[limit_key] * f["timeout_multiplier"] / 60
        checkpoints = {}
        for t, x in f["checkpoints"].items():
            x = dict(x)
            x["limit_min"] = limit_min
            x["share_of_limit"] = int(t) / limit_min
            checkpoints[int(t)] = x
        sessions.append(
            {
                "folder": u["folder"],
                "arm": u["arm"],
                "benchmark": u["benchmark"],
                "section": section_of(u),
                "task": task,
                "group": (u["benchmark"], task),
                "block": u["pair_block"],
                "server": u["server"],
                "passed": u["reward"] == 1,
                "trial_s": u["trial_s"],
                "agent_s": u["agent_s"],
                "setup_s": f["setup_s"],
                "checkpoints": checkpoints,
            }
        )
    return sessions, left_out, skipped


def pair_up(sessions: list[dict]) -> tuple[list[tuple[dict, dict]], list[str]]:
    sides: dict[str, dict[str, dict]] = {BASE: {}, JEFF: {}}
    for s in sessions:
        if s["arm"] in sides:
            if s["block"] in sides[s["arm"]]:
                raise ValueError(f"two {s['arm']} sessions in block {s['block']}")
            sides[s["arm"]][s["block"]] = s
    pairs, unpaired = [], []
    for block in sorted(set(sides[BASE]) | set(sides[JEFF])):
        b, j = sides[BASE].get(block), sides[JEFF].get(block)
        if b is None or j is None:
            unpaired.append(f"{block}: no usable {BASE if b is None else JEFF} session")
            continue
        if (b["group"], b["server"]) != (j["group"], j["server"]):
            raise ValueError(f"block {block}: a1 and a4 differ in task or server")
        pairs.append((b, j))
    return pairs, unpaired


def split_tasks(groups: list[tuple[str, str]], seed: int = SEED) -> tuple[set, dict]:
    """Held-out tasks (the first 20% after a seeded shuffle) and the fold of every other task."""
    ordered = sorted(set(groups))
    random.Random(seed).shuffle(ordered)
    n_hold = round(HOLDOUT_SHARE * len(ordered))
    holdout = set(ordered[:n_hold])
    folds = {g: i % FOLDS for i, g in enumerate(ordered[n_hold:])}
    return holdout, folds


# ------------------------------------------------------------------------------------------------------------- models


def rows_of(sessions: list[dict]) -> list[dict]:
    rows = []
    for s in sessions:
        for t, x in sorted(s["checkpoints"].items()):
            rows.append({"session": s, "t": t, "x": x, "y": int(s["passed"])})
    return rows


class Design:
    """Turns feature names into a standardised matrix; levels and scales come from the training rows only."""

    def __init__(self, names: list[str], train: list[dict]) -> None:
        self.names = names
        self.levels = {g: sorted({r["session"][g] for r in train}) for g in names if g in GROUP_FEATURES}
        raw = self._raw(train)
        self.mean = raw.mean(axis=0)
        sd = raw.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.columns = [c for n in names for c in self._column_names(n)]

    def _column_names(self, name: str) -> list[str]:
        if name in GROUP_FEATURES:
            return [f"{name}={level}" for level in self.levels[name]]
        return [f"log(1+{name})" if name in LOG_FEATURES else name]

    def _raw(self, rows: list[dict]) -> np.ndarray:
        cols = []
        for name in self.names:
            if name in GROUP_FEATURES:
                for level in self.levels[name]:
                    cols.append([float(r["session"][name] == level) for r in rows])
            elif name in LOG_FEATURES:
                cols.append([math.log1p(r["x"][name]) for r in rows])
            else:
                cols.append([float(r["x"][name]) for r in rows])
        return np.array(cols, dtype=float).T.reshape(len(rows), -1)

    def matrix(self, rows: list[dict]) -> np.ndarray:
        return (self._raw(rows) - self.mean) / self.sd


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = L2) -> np.ndarray:
    """Newton's method for L2-regularised logistic regression; weight 0 is the unpenalised intercept."""
    xa = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(xa.shape[1])
    penalty = np.full(xa.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(xa @ w)))
        grad = xa.T @ (p - y) + penalty * w
        hess = (xa * (p * (1 - p))[:, None]).T @ xa + np.diag(penalty) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            return w
    raise ValueError("logistic regression did not converge in 50 Newton steps")


def predict_logistic(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(w[0] + x @ w[1:])))


class Logistic:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def fit(self, train: list[dict]) -> "Logistic":
        self.design = Design(self.names, train)
        self.w = fit_logistic(self.design.matrix(train), np.array([r["y"] for r in train], dtype=float))
        return self

    def predict(self, rows: list[dict]) -> np.ndarray:
        return predict_logistic(self.w, self.design.matrix(rows))


def turn_bin(turns: int) -> int:
    return max(i for i, edge in enumerate(TURN_BINS) if turns >= edge)


def bin_label(i: int) -> str:
    return f"{TURN_BINS[i]}+" if i == len(TURN_BINS) - 1 else f"{TURN_BINS[i]}-{TURN_BINS[i + 1] - 1}"


class TurnTable:
    """P(pass) = training pass rate of the row's (checkpoint, turns bin) cell, pulled toward the checkpoint's pass rate
    with the weight of 5 rows."""

    def fit(self, train: list[dict]) -> "TurnTable":
        cells: dict = defaultdict(lambda: [0, 0])
        by_t: dict = defaultdict(lambda: [0, 0])
        for r in train:
            c = cells[(r["t"], turn_bin(r["x"]["turns"]))]
            c[0] += r["y"]
            c[1] += 1
            by_t[r["t"]][0] += r["y"]
            by_t[r["t"]][1] += 1
        self.prior = {t: k / n for t, (k, n) in by_t.items()}
        self.cells = {key: (k + 5 * self.prior[key[0]]) / (n + 5) for key, (k, n) in cells.items()}
        return self

    def predict(self, rows: list[dict]) -> np.ndarray:
        return np.array([self.cells.get((r["t"], turn_bin(r["x"]["turns"])), self.prior[r["t"]]) for r in rows])


TREE_FEATURES = RAW_FEATURES + LOG_FEATURES


class Tree:
    """Two levels of yes/no conditions (feature >= cut), chosen by the Gini impurity of pass/fail, at least 30 rows per
    leaf; a leaf's P(pass) is its training pass rate (plus one pass and one fail)."""

    MIN_LEAF = 30

    def fit(self, train: list[dict]) -> "Tree":
        self.root = self._grow(train, 2)
        return self

    def _value(self, r: dict, name: str) -> float:
        if name.startswith("benchmark="):
            return float(r["session"]["benchmark"] == name.split("=", 1)[1])
        return float(r["x"][name])

    def _grow(self, rows: list[dict], depth: int) -> dict:
        k = sum(r["y"] for r in rows)
        leaf = {"p": (k + 1) / (len(rows) + 2), "n": len(rows), "passes": k}
        if depth == 0 or len(rows) < 2 * self.MIN_LEAF:
            return leaf
        y = np.array([r["y"] for r in rows], dtype=float)
        names = list(TREE_FEATURES) + [f"benchmark={b}" for b in sorted({r["session"]["benchmark"] for r in rows})]
        best = None
        for name in names:
            v = np.array([self._value(r, name) for r in rows])
            for cut in np.unique(np.quantile(v, np.linspace(0.05, 0.95, 19))):
                right = v >= cut
                n_r = int(right.sum())
                if n_r < self.MIN_LEAF or len(rows) - n_r < self.MIN_LEAF:
                    continue
                score = 0.0
                for side in (right, ~right):
                    p = y[side].mean()
                    score += side.sum() * p * (1 - p)
                if best is None or score < best[0] - 1e-12:
                    best = (score, name, float(cut), right)
        if best is None:
            return leaf
        _, name, cut, right = best
        yes = [r for r, s in zip(rows, right) if s]
        no = [r for r, s in zip(rows, right) if not s]
        return {**leaf, "name": name, "cut": cut, "yes": self._grow(yes, depth - 1), "no": self._grow(no, depth - 1)}

    def _leaf(self, r: dict) -> dict:
        node = self.root
        while "name" in node:
            node = node["yes"] if self._value(r, node["name"]) >= node["cut"] else node["no"]
        return node

    def predict(self, rows: list[dict]) -> np.ndarray:
        return np.array([self._leaf(r)["p"] for r in rows])

    def describe(self) -> list[str]:
        out = []

        def walk(node: dict, conds: list[str]) -> None:
            if "name" not in node:
                out.append(f"{' and '.join(conds) or 'always'}: pass rate {node['passes']}/{node['n']}")
                return
            if node["name"].startswith("benchmark="):
                level = node["name"].split("=", 1)[1]
                yes, no = f"benchmark is {level}", f"benchmark is not {level}"
            else:
                yes, no = f"{node['name']} >= {node['cut']:.3g}", f"{node['name']} < {node['cut']:.3g}"
            walk(node["yes"], conds + [yes])
            walk(node["no"], conds + [no])

        walk(self.root, [])
        return out


def cross_validate(make, rows: list[dict], fold_of: dict) -> np.ndarray:
    """Out-of-fold P(pass) for every row: each from a model fitted on the other folds' tasks."""
    out = np.full(len(rows), np.nan)
    folds = np.array([fold_of[r["session"]["group"]] for r in rows])
    for k in range(FOLDS):
        test = folds == k
        model = make().fit([r for r, t in zip(rows, test) if not t])
        out[test] = model.predict([r for r, t in zip(rows, test) if t])
    return out


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Area under the ROC curve (ties count half); None without both classes."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    values = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    rank_of = np.empty(len(values))
    rank_of[order] = ranks
    return float((rank_of[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def forward_select(rows: list[dict], fold_of: dict) -> tuple[list[str], list[tuple[str, float, float]]]:
    """Adds, one at a time, the feature that most lowers cross-validated log-loss, starting from elapsed time; stops at
    MAX_FEATURES or when no feature lowers it by 0.0005."""
    y = np.array([r["y"] for r in rows], dtype=float)
    chosen = ["elapsed_min"]
    p = cross_validate(lambda: Logistic(chosen), rows, fold_of)
    steps = [("elapsed_min", log_loss(p, y), auc(p, y))]
    while len(chosen) < MAX_FEATURES:
        best = None
        for name in CANDIDATES:
            if name in chosen:
                continue
            p = cross_validate(lambda: Logistic(chosen + [name]), rows, fold_of)
            loss = log_loss(p, y)
            if best is None or loss < best[1]:
                best = (name, loss, auc(p, y))
        if best is None or steps[-1][1] - best[1] < 0.0005:
            break
        chosen.append(best[0])
        steps.append(best)
    return chosen, steps


def single_feature_value(rows: list[dict], fold_of: dict) -> list[dict]:
    """Each candidate's cross-validated gain over elapsed time alone, and its direction in a fit on all rows."""
    y = np.array([r["y"] for r in rows], dtype=float)
    p0 = cross_validate(lambda: Logistic(["elapsed_min"]), rows, fold_of)
    base_loss, base_auc = log_loss(p0, y), auc(p0, y)
    out = []
    for name in CANDIDATES:
        if name == "elapsed_min":
            continue
        p = cross_validate(lambda: Logistic(["elapsed_min", name]), rows, fold_of)
        if name in GROUP_FEATURES:
            direction = "by level"
        else:
            model = Logistic(["elapsed_min", name]).fit(rows)
            direction = "more = more likely to pass" if model.w[2] > 0 else "more = more likely to fail"
        out.append({"name": name, "loss_gain": base_loss - log_loss(p, y), "auc": auc(p, y), "base_auc": base_auc,
                    "direction": direction})
    return sorted(out, key=lambda d: -d["loss_gain"])


# ----------------------------------------------------------------------------------------------------------- policies


def stop_time(session: dict, scores: dict, checks: tuple[int, ...], threshold: float) -> int | None:
    """The first checked checkpoint (minutes) at which the session was still running and P(pass) < threshold."""
    for t in checks:
        if t in session["checkpoints"] and scores[(session["folder"], t)] < threshold:
            return t
    return None


def session_outcome(session: dict, stop: int | None) -> tuple[float, float, int]:
    """(trial seconds charged, agent seconds charged, pass) under a stop at minute `stop` (None: not stopped)."""
    if stop is None:
        return session["trial_s"], session["agent_s"], int(session["passed"])
    if stop * 60 >= session["agent_s"]:
        raise ValueError(f"{session['folder']} stopped at {stop} min but its agent ran {session['agent_s']:.0f} s")
    return session["setup_s"] + stop * 60, stop * 60.0, 0


def evaluate(pairs: list, stop_jeff, stop_base=None) -> dict:
    """Totals per section and pooled for a stop rule on a4 (and optionally on a1); stop_* map a session to a stop
    minute or None."""
    out: dict = defaultdict(lambda: Counter())
    for b, j in pairs:
        bt, ba, bp = session_outcome(b, stop_base(b) if stop_base else None)
        jt, ja, jp = session_outcome(j, stop_jeff(j))
        for key in (b["section"], "pooled"):
            c = out[key]
            c["n"] += 1
            c["base_time"] += bt
            c["base_agent"] += ba
            c["base_pass"] += bp
            c["jeff_time"] += jt
            c["jeff_agent"] += ja
            c["jeff_pass"] += jp
            c["jeff_pass_uncut"] += int(j["passed"])
            c["base_pass_uncut"] += int(b["passed"])
            c["jeff_stopped"] += stop_jeff(j) is not None
            c["jeff_lost"] += int(j["passed"]) - jp
            c["base_stopped"] += bool(stop_base) and stop_base(b) is not None
    return dict(out)


def summary(c: Counter) -> dict:
    n = c["n"]
    return {
        "n": n,
        "ratio": c["jeff_time"] / c["base_time"],
        "agent_ratio": c["jeff_agent"] / c["base_agent"],
        "diff": 100 * (c["jeff_pass"] - c["base_pass"]) / n,
        "loss": 100 * c["jeff_lost"] / n,
        "jeff_rate": 100 * c["jeff_pass"] / n,
        "base_rate": 100 * c["base_pass"] / n,
        "stopped": c["jeff_stopped"],
        "lost": c["jeff_lost"],
        "base_stopped": c["base_stopped"],
    }


def base_effect(pairs: list, stop) -> tuple[float, float]:
    """What a policy does to a1 by itself: (a1 time stopped / a1 time not stopped, a1 passes lost in points)."""
    cut = [session_outcome(b, stop(b)) for b, _ in pairs]
    full = sum(b["trial_s"] for b, _ in pairs)
    lost = sum(int(b["passed"]) for b, _ in pairs) - sum(o[2] for o in cut)
    return sum(o[0] for o in cut) / full, 100 * lost / len(pairs)


MODES = tuple(("at", (t,)) for t in CHECKPOINTS) + tuple(
    ("from", tuple(u for u in CHECKPOINTS if u >= t)) for t in CHECKPOINTS[:-1]
)


def mode_text(mode: tuple) -> str:
    kind, checks = mode
    if kind == "at":
        return f"check once at {checks[0]} min"
    return f"check at {', '.join(str(t) for t in checks)} min"


def best_policies(pairs: list, scores: dict) -> dict:
    """For each pass-loss budget: the (mode, threshold) with the lowest pooled time ratio whose pooled pass loss stays
    within it (ties: smaller loss, then the earlier mode)."""
    best: dict = {}
    for mode in MODES:
        for th in THRESHOLDS:
            s = summary(evaluate(pairs, lambda x, m=mode, th=th: stop_time(x, scores, m[1], th))["pooled"])
            for budget in BUDGETS:
                if s["loss"] <= budget + 1e-9:
                    key = (s["ratio"], s["loss"])
                    if budget not in best or key < best[budget][0]:
                        best[budget] = (key, mode, th, s)
    return {b: {"mode": v[1], "threshold": v[2], "cv": v[3]} for b, v in best.items()}


# ------------------------------------------------------------------------------------------------------------- report


def fmt_ratio(x: float) -> str:
    return f"{x:.2f}x"


def fmt_pts(x: float) -> str:
    return f"{x:+.1f}"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def section_rows(result: dict) -> list[list]:
    out = []
    for key in sorted(k for k in result if k != "pooled") + ["pooled"]:
        s = summary(result[key])
        out.append([key, s["n"], f"{s['base_rate']:.1f}%", f"{s['jeff_rate']:.1f}%", fmt_pts(s["diff"]),
                    fmt_pts(-s["loss"]), fmt_ratio(s["ratio"]), fmt_ratio(s["agent_ratio"]), s["stopped"]])
    return out


SECTION_HEADER = ["benchmark", "pairs", "a1 pass", "a4 pass (rule)", "a4 - a1 (pts)", "lost by rule (pts)",
                  "time ratio", "agent-time ratio", "a4 stopped"]


def run(unit_paths: list[str], feature_paths: list[str], out: Path, notes: list[str]) -> dict:
    sessions, left_out, skipped = load(unit_paths, feature_paths)
    hosts = Counter(s["folder"].split("/runs/")[0] for s in sessions)
    notes = [f"- Inputs: {', '.join(unit_paths + feature_paths)}."] + [f"- {n}" for n in notes] + [
        f"- Sessions used by evaluation folder: {', '.join(f'{h} {n}' for h, n in sorted(hosts.items()))}."]
    pairs, unpaired = pair_up(sessions)
    holdout, fold_of = split_tasks([s["group"] for s in sessions])
    cv_sessions = [s for s in sessions if s["group"] not in holdout]
    ho_sessions = [s for s in sessions if s["group"] in holdout]
    cv_pairs = [(b, j) for b, j in pairs if b["group"] not in holdout]
    ho_pairs = [(b, j) for b, j in pairs if b["group"] in holdout]
    cv_rows, ho_rows = rows_of(cv_sessions), rows_of(ho_sessions)
    y_cv = np.array([r["y"] for r in cv_rows], dtype=float)
    y_ho = np.array([r["y"] for r in ho_rows], dtype=float)

    ranking = single_feature_value(cv_rows, fold_of)
    chosen, steps = forward_select(cv_rows, fold_of)
    makers = {"turn table": TurnTable, "decision tree": Tree, "logistic regression": lambda: Logistic(chosen)}
    oof, held, final = {}, {}, {}
    for name, make in makers.items():
        oof[name] = cross_validate(make, cv_rows, fold_of)
        final[name] = make().fit(cv_rows)
        held[name] = final[name].predict(ho_rows)

    def score_map(rows: list[dict], p: np.ndarray) -> dict:
        return {(r["session"]["folder"], r["t"]): float(v) for r, v in zip(rows, p)}

    chosen_policies = {}
    for name in makers:
        cv_scores, ho_scores = score_map(cv_rows, oof[name]), score_map(ho_rows, held[name])
        for budget, pol in best_policies(cv_pairs, cv_scores).items():
            checks, th = pol["mode"][1], pol["threshold"]

            def rule(scores, checks=checks, th=th):
                return lambda x: stop_time(x, scores, checks, th)

            pol["cv_sections"] = evaluate(cv_pairs, rule(cv_scores))
            pol["ho_sections"] = evaluate(ho_pairs, rule(ho_scores))
            pol["ho"] = summary(pol["ho_sections"]["pooled"])
            pol["cv_both"] = summary(evaluate(cv_pairs, rule(cv_scores), rule(cv_scores))["pooled"])
            pol["ho_both"] = summary(evaluate(ho_pairs, rule(ho_scores), rule(ho_scores))["pooled"])
            pol["cv_base"] = base_effect(cv_pairs, rule(cv_scores))
            pol["ho_base"] = base_effect(ho_pairs, rule(ho_scores))
            chosen_policies[(name, budget)] = pol

    lines = report(locals())
    out.write_text("\n".join(lines) + "\n")
    return {"policies": chosen_policies, "chosen": chosen, "y": (y_cv, y_ho)}


def references(pairs: list) -> list[list]:
    rows = []
    s = summary(evaluate(pairs, lambda x: None)["pooled"])
    rows.append(["never stop", fmt_ratio(s["ratio"]), fmt_pts(s["diff"]), fmt_pts(-s["loss"])])
    for t in (20, 30, 45, 60):
        s = summary(evaluate(pairs, lambda x, t=t: t if x["agent_s"] > t * 60 else None)["pooled"])
        rows.append([f"stop every session at {t} min", fmt_ratio(s["ratio"]), fmt_pts(s["diff"]), fmt_pts(-s["loss"])])
    for t in (20, 30, 45, 60):
        s = summary(evaluate(pairs, lambda x, t=t: t if x["agent_s"] > t * 60 and not x["passed"] else None)["pooled"])
        rows.append([f"perfect rule at {t} min (stops only sessions that fail anyway)", fmt_ratio(s["ratio"]),
                     fmt_pts(s["diff"]), fmt_pts(-s["loss"])])
    return rows


def report(v: dict) -> list[str]:
    sessions, pairs, cv_pairs, ho_pairs = v["sessions"], v["pairs"], v["cv_pairs"], v["ho_pairs"]
    cv_rows, ho_rows, y_cv, y_ho = v["cv_rows"], v["ho_rows"], v["y_cv"], v["y_ho"]
    pol = v["chosen_policies"]
    L = [
        "# Stopping hopeless sessions early: offline estimate",
        "",
        "Generated by `results/imitation/scripts/cutoff_estimate.py` (features: `cutoff_features.py`). This is an "
        "offline estimate on tonight's evaluation sessions (2026-10-04/05): no session was stopped for real. Every "
        "number below the model sections comes from tasks the model did not see: cross-validated (5 folds grouped by "
        "task) or the 20% of tasks set aside before any fitting (seed 20261005).",
        "",
        "## What was measured",
        "",
        "- A **checkpoint** is a time after the agent started: 10, 15, 20, 30, 45 or 60 minutes. A session still "
        "running at a checkpoint gives one row: what its pi session and Jeff trace show up to then, and whether it "
        "finally passed (reward 1).",
        "- A **policy** checks P(pass) at one checkpoint, or at every checkpoint from some start, and stops the "
        "session at the first check where P(pass) is below a threshold. A stopped session counts as failed and costs "
        "its setup time plus the checkpoint time instead of its full trial time.",
        "- **Time ratio**: total a4-jeff06 trial time / total a1-baseline trial time over paired blocks (same task "
        "attempt, same Qwen server), all outcomes. **a4 - a1**: pass-rate difference in points. **Lost by rule**: "
        "passes the rule throws away (sessions that would have passed but were stopped), in points.",
        "- Models train on rows of all four arms (a1-a4); the policy is evaluated on a4 against an unstopped a1, and, "
        "as the plain-pi variant, with the same policy also stopping a1 sessions.",
        "- Policies are picked on the cross-validated numbers (lowest time ratio with at most 0 or 2 points lost), then "
        "checked once on the held-out tasks.",
        "",
    ]
    top = ["## Result", "",
          "Chosen on cross-validation tasks (lowest pooled time ratio with the pass loss caused by the rule at most 0 or "
          "2 points), then applied unchanged to the held-out tasks. Time ratio and a4 - a1 are against an a1 that is "
          "never stopped; 'both stopped' applies the same policy to a1 too (a plain-pi user could) and compares a4 "
          "stopped with a1 stopped; 'a1 alone' is what the policy does to a1 by itself (a1 stopped / a1 not "
          "stopped, and the a1 passes it throws away).", ""]
    prow = []
    for (name, budget), p in pol.items():
        prow.append([name, f"<= {budget:.0f} pts", f"{mode_text(p['mode'])}, stop if P(pass) < {p['threshold']:.2f}",
                     f"{fmt_ratio(p['cv']['ratio'])} / {fmt_pts(p['cv']['diff'])} / {fmt_pts(-p['cv']['loss'])}",
                     f"{fmt_ratio(p['ho']['ratio'])} / {fmt_pts(p['ho']['diff'])} / {fmt_pts(-p['ho']['loss'])}",
                     f"{p['cv']['stopped']} / {p['ho']['stopped']}",
                     f"{fmt_ratio(p['cv_both']['ratio'])} / {fmt_pts(p['cv_both']['diff'])}",
                     f"{fmt_ratio(p['ho_both']['ratio'])} / {fmt_pts(p['ho_both']['diff'])}",
                     f"{p['cv_base'][0]:.2f}x / {fmt_pts(-p['cv_base'][1])}", f"{p['ho_base'][0]:.2f}x / {fmt_pts(-p['ho_base'][1])}"])
    top += md_table(["model", "loss budget", "policy", "CV: ratio / a4-a1 / lost", "held out: ratio / a4-a1 / lost",
                   "a4 stopped (CV / held out)", "both stopped, CV: ratio / a4-a1",
                   "both stopped, held out: ratio / a4-a1", "a1 alone, CV: time / lost", "a1 alone, held out: time / lost"],
                     prow)
    lg = v["final"]["logistic regression"]
    top += [f"- Logistic regression features (forward selection on cross-validation tasks): "
            f"{', '.join(v['chosen'])}. The arm was a candidate feature: "
            f"{'chosen' if 'arm' in v['chosen'] else 'not chosen (it did not lower cross-validated log-loss enough)'}.",
            "- Reference: never stopping gives a4 a time ratio of "
            f"{summary(evaluate(cv_pairs, lambda x: None)['pooled'])['ratio']:.2f}x on cross-validation tasks and "
            f"{summary(evaluate(ho_pairs, lambda x: None)['pooled'])['ratio']:.2f}x on held-out tasks; the held-out tasks "
            "happen to be harder for a4 than for a1 even without any stopping (see the reference table).", ""]
    L += top + ["## Data used", ""]
    L += v["notes"]
    arms = Counter(s["arm"] for s in sessions)
    L += [f"- Sessions used: {len(sessions)} ({', '.join(f'{a} {arms[a]}' for a in ARMS)}); tasks "
          f"{len({s['group'] for s in sessions})}, of which {len(v['holdout'])} held out.",
          f"- Paired a1/a4 blocks: {len(pairs)} ({len(cv_pairs)} on cross-validation tasks, {len(ho_pairs)} on held-out "
          f"tasks).",
          f"- Not considered: {', '.join(f'{k} {n}' for k, n in sorted(v['skipped'].items()))}.",
          f"- Left out (finished but unusable): {len(v['left_out'])}, listed at the end. a1/a4 blocks without both "
          f"sessions usable: {len(v['unpaired'])}.",
          ""]
    L += ["Rows (sessions still running) per checkpoint, cross-validation tasks / held-out tasks:", ""]
    trs = []
    for t in CHECKPOINTS:
        def cnt(rows, arm=None, t=t):
            sel = [r for r in rows if r["t"] == t and (arm is None or r["session"]["arm"] == arm)]
            return f"{len(sel)} ({sum(r['y'] for r in sel)} pass)"
        trs.append([f"{t} min", cnt(cv_rows), cnt(cv_rows, JEFF), cnt(cv_rows, BASE), cnt(ho_rows), cnt(ho_rows, JEFF)])
    L += md_table(["checkpoint", "all arms (CV)", "a4 (CV)", "a1 (CV)", "all arms (held out)", "a4 (held out)"], trs)

    L += ["## Reference points (no model)", "", "Cross-validation tasks / held-out tasks, pooled:", ""]
    ref_cv, ref_ho = references(cv_pairs), references(ho_pairs)
    L += md_table(["policy", "time ratio (CV tasks)", "a4 - a1 (CV)", "lost (CV)", "time ratio (held out)",
                   "a4 - a1 (held out)", "lost (held out)"],
                  [a + b[1:] for a, b in zip(ref_cv, ref_ho)])

    L += ["## Which signals predict a pass", "",
          "Each feature added alone to elapsed time, logistic regression, cross-validated by task on all arms' rows. "
          "Gain = drop in log-loss (higher is better); AUC = how well P(pass) ranks passing above failing rows "
          f"(elapsed time alone: {v['ranking'][0]['base_auc']:.3f}). Counts enter as log(1 + count).", ""]
    L += md_table(["feature", "log-loss gain", "AUC with elapsed", "direction"],
                  [[d["name"], f"{d['loss_gain']:.4f}", f"{d['auc']:.3f}", d["direction"]] for d in v["ranking"]])
    L += ["Forward selection (cross-validated log-loss, at most 8 features, held-out tasks never used):", ""]
    L += md_table(["step", "feature added", "CV log-loss", "CV AUC"],
                  [[i + 1, n, f"{loss:.4f}", f"{a:.3f}"] for i, (n, loss, a) in enumerate(v["steps"])])
    lg = v["final"]["logistic regression"]
    L += ["Final logistic regression (fitted on all cross-validation tasks; weights on standardised features, "
          "positive = more likely to pass):", ""]
    L += md_table(["column", "weight"], [["intercept", f"{lg.w[0]:+.3f}"]] +
                  [[c, f"{w:+.3f}"] for c, w in zip(lg.design.columns, lg.w[1:])])
    L += ["Decision tree (fitted on all cross-validation tasks; rows of all arms and checkpoints):", ""]
    L += [f"- {d}" for d in v["final"]["decision tree"].describe()] + [""]
    L += ["Turn table: pass rate of sessions still running, by checkpoint and Qwen turns so far (cross-validation "
          "tasks, all arms; a4 in brackets):", ""]
    trows = []
    for t in CHECKPOINTS:
        cells = []
        for i in range(len(TURN_BINS)):
            sel = [r for r in cv_rows if r["t"] == t and turn_bin(r["x"]["turns"]) == i]
            sel4 = [r for r in sel if r["session"]["arm"] == JEFF]
            cells.append(f"{sum(r['y'] for r in sel)}/{len(sel)} ({sum(r['y'] for r in sel4)}/{len(sel4)})"
                         if sel else "-")
        trows.append([f"{t} min"] + cells)
    L += md_table(["checkpoint"] + [f"{bin_label(i)} turns" for i in range(len(TURN_BINS))], trows)

    L += ["## How well each model ranks sessions (AUC)", "",
          "AUC per checkpoint: cross-validated on CV tasks / on held-out tasks (model fitted on all CV tasks). "
          "All arms, and a4 only.", ""]
    arows = []
    for name in v["makers"]:
        for t in CHECKPOINTS + (None,):
            def a(rows, p, y, arm=None, t=t):
                sel = np.array([(t is None or r["t"] == t) and (arm is None or r["session"]["arm"] == arm)
                                for r in rows])
                val = auc(p[sel], y[sel]) if sel.any() else None
                return "-" if val is None else f"{val:.3f}"
            arows.append([name, "all" if t is None else f"{t} min", a(cv_rows, v["oof"][name], y_cv),
                          a(ho_rows, v["held"][name], y_ho), a(cv_rows, v["oof"][name], y_cv, JEFF),
                          a(ho_rows, v["held"][name], y_ho, JEFF)])
    L += md_table(["model", "checkpoint", "CV all arms", "held out all arms", "CV a4", "held out a4"], arows)

    L += ["## Each chosen policy per benchmark", ""]
    for (name, budget), p in pol.items():
        L += [f"### {name}, loss budget {budget:.0f} points: {mode_text(p['mode'])}, stop if P(pass) < "
              f"{p['threshold']:.2f}", "", "Cross-validation tasks:", ""]
        L += md_table(SECTION_HEADER, section_rows(p["cv_sections"]))
        L += ["Held-out tasks:", ""]
        L += md_table(SECTION_HEADER, section_rows(p["ho_sections"]))

    L += ["## Caveats", "",
          "- One night of sessions; a task's attempts and arms are correlated, which is why splits are by task. The "
          "held-out check is on about a fifth of the tasks, so its numbers are noisy (one pass is "
          f"{100 / max(1, len(ho_pairs)):.2f} points there).",
          "- Policies are chosen among many (6 single checkpoints, 5 every-checkpoint starts, 60 thresholds, per model) "
          "on the cross-validated numbers; the held-out row is the honest check of that choice.",
          "- Tests, file changes and error types are read from command text and output with simple patterns "
          "(cutoff_features.py); they are approximate.",
          "- Stopping a session also frees its Qwen server earlier; this estimate counts only the session's own time.",
          ""]
    L += ["## Sessions left out", ""]
    L += [f"- {f}: {why}" for f, why in v["left_out"]] or ["- none"]
    L += [f"- block {u}" for u in v["unpaired"]]
    return L


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--units", nargs="+", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--note", action="append", default=[], help="a line added to the report's data section")
    args = parser.parse_args()
    result = run(args.units, args.features, Path(args.out), args.note)
    for (name, budget), p in result["policies"].items():
        print(f"{name} <= {budget:.0f} pts: {mode_text(p['mode'])} P<{p['threshold']:.2f}: "
              f"CV {p['cv']['ratio']:.3f}x {p['cv']['diff']:+.1f} (lost {p['cv']['loss']:.1f}); "
              f"held out {p['ho']['ratio']:.3f}x {p['ho']['diff']:+.1f} (lost {p['ho']['loss']:.1f})")


if __name__ == "__main__":
    main()
