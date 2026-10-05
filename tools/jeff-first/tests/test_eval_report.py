import json
import math

import pytest

from eval_report import (
    bootstrap_interval,
    error_kind,
    exit_code,
    geomean,
    largest_differences,
    load,
    mcnemar_p,
    nearest_rank,
    pair_up,
    section_of,
    statistics_of,
    sum_fields,
    wilson,
)


def test_wilson_matches_the_textbook_values():
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_wilson_at_zero_and_all_passes_stays_inside_0_1():
    assert wilson(0, 20)[0] == 0.0
    assert wilson(20, 20)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_mcnemar_exact_two_sided():
    # 1 vs 9 discordant pairs: two-sided binomial p = 2 * P(X <= 1 | n=10, 0.5) = 2 * 11/1024
    assert mcnemar_p(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_p(9, 1) == pytest.approx(22 / 1024)
    assert mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(0, 0) == 1.0


def test_geomean_and_nearest_rank():
    assert geomean([0.5, 2.0]) == pytest.approx(1.0)
    assert geomean([1.0, 4.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        geomean([])
    assert nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9
    assert nearest_rank([5], 0.9) == 5
    assert nearest_rank([3, 1, 2], 0.5) == 2


def test_sum_fields_adds_every_key():
    assert sum_fields([{"a": 1, "b": 2.5}, {"a": 2, "b": 0.5}]) == {"a": 3, "b": 3.0}


def test_bootstrap_interval_of_a_constant_statistic_is_a_point():
    clusters = [{"x": 1.0, "n": 1}, {"x": 1.0, "n": 1}, {"x": 1.0, "n": 1}]
    interval = bootstrap_interval(clusters, lambda t: t["x"] / t["n"], resamples=200, seed=1)
    assert interval.lo == pytest.approx(1.0)
    assert interval.hi == pytest.approx(1.0)
    assert interval.undefined == 0


def test_bootstrap_interval_covers_the_mean_and_is_reproducible():
    clusters = [{"x": float(i), "n": 1} for i in range(20)]
    a = bootstrap_interval(clusters, lambda t: t["x"] / t["n"], resamples=500, seed=7)
    b = bootstrap_interval(clusters, lambda t: t["x"] / t["n"], resamples=500, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert a.lo < 9.5 < a.hi
    assert 6 < a.lo and a.hi < 13


def test_bootstrap_interval_counts_undefined_resamples_and_gives_no_interval():
    clusters = [{"p": 1, "n": 1}] + [{"p": 0, "n": 1}] * 9
    interval = bootstrap_interval(clusters, lambda t: 1 / t["p"] if t["p"] else None, resamples=300, seed=3)
    assert interval.undefined > 0
    assert interval.lo is None and interval.hi is None


def row(arm, block, reward, wall, qwen=10.0, task="t1", benchmark="terminal-bench-2", server="s0"):
    r = {
        "arm": arm,
        "pair_block": block,
        "block": block,
        "task": task,
        "benchmark": benchmark,
        "server": server,
        "state": "finished",
        "superseded": None,
        "reward": reward,
        "agent_s": wall,
        "error": None,
    }
    if qwen is not None:
        r["qwen_s"] = qwen
    return r


def test_pair_up_pairs_by_block_and_counts_discordant_passes():
    rows = [
        row("a1-baseline", "b1", 1.0, 100.0),
        row("a2-off-guard", "b1", 1.0, 50.0),
        row("a1-baseline", "b2", 1.0, 100.0, task="t2"),
        row("a2-off-guard", "b2", 0.0, 300.0, task="t2"),
        row("a1-baseline", "b3", 0.0, 100.0, task="t3"),
        row("a2-off-guard", "b3", 1.0, 80.0, task="t3"),
    ]
    pairs, left_out = pair_up(rows, "a2-off-guard")
    assert left_out == []
    total = sum_fields([p.fields for p in pairs])
    assert total["n"] == 3
    assert total["base_only"] == 1 and total["arm_only"] == 1
    assert total["solved_both"] == 1
    assert total["both_logw"] == pytest.approx([math.log(0.5)])
    assert total["base_wall"] == 300.0 and total["arm_wall"] == 430.0
    assert {p.cluster for p in pairs} == {("terminal-bench-2", "t1"), ("terminal-bench-2", "t2"), ("terminal-bench-2", "t3")}


def test_time_on_tasks_the_baseline_solved_counts_the_arm_whatever_its_outcome():
    rows = [
        row("a1-baseline", "b1", 1.0, 100.0, qwen=40.0),
        row("a2-off-guard", "b1", 1.0, 50.0, qwen=10.0),
        row("a1-baseline", "b2", 1.0, 100.0, qwen=40.0, task="t2"),
        row("a2-off-guard", "b2", 0.0, 300.0, qwen=160.0, task="t2"),
        row("a1-baseline", "b3", 0.0, 100.0, task="t3"),
        row("a2-off-guard", "b3", 1.0, 80.0, task="t3"),
    ]
    pairs, _ = pair_up(rows, "a2-off-guard")
    total = sum_fields([p.fields for p in pairs])
    assert total["bs_n"] == 2
    assert (total["bs_base_wall"], total["bs_arm_wall"]) == (200.0, 350.0)
    assert (total["bs_base_qwen"], total["bs_arm_qwen"]) == (80.0, 170.0)
    s = statistics_of(total)
    assert s["bs_total"] == pytest.approx(350 / 200)
    assert s["bs_geo"] == pytest.approx(math.sqrt(0.5 * 3.0))
    assert s["bs_median"] == pytest.approx(math.sqrt(0.5 * 3.0))  # two blocks: middle pair averaged geometrically
    assert s["bs_qwen_total"] == pytest.approx(170 / 80)
    assert s["bs_qwen_geo"] == pytest.approx(math.sqrt(0.25 * 4.0))
    assert s["both_geo"] == pytest.approx(0.5)
    assert s["total_wall"] == pytest.approx(430 / 300)
    # per-session medians: baseline 100, arm median of 50, 300, 80 = 80
    assert s["session_median"] == pytest.approx(0.8)
    # per-block ratios on all paired blocks: 0.5, 3.0, 0.8
    assert s["all_geo"] == pytest.approx((0.5 * 3.0 * 0.8) ** (1 / 3))
    assert s["all_median"] == pytest.approx(0.8)


def test_largest_differences_lists_the_pairs_with_the_biggest_time_gap_either_way():
    rows = [
        row("a1-baseline", "b1", 1.0, 100.0),
        row("a2-off-guard", "b1", 1.0, 50.0),
        row("a1-baseline", "b2", 1.0, 100.0, task="t2"),
        row("a2-off-guard", "b2", 0.0, 900.0, task="t2"),
        row("a1-baseline", "b3", 0.0, 400.0, task="t3"),
        row("a2-off-guard", "b3", 1.0, 80.0, task="t3"),
    ]
    rows[3].update({"turns": 10, "turns_off": 7, "guard_loop": 2, "guard_runaway": 0, "forced_xhigh": 1})
    pairs, _ = pair_up(rows, "a2-off-guard")
    top = largest_differences(pairs, 2)
    assert [p.block for p in top] == ["b2", "b3"]
    assert (top[0].arm_turns_off, top[0].arm_turns, top[0].arm_guard, top[0].arm_forced) == (7, 10, 2, 1)
    assert (top[0].base_pass, top[0].arm_pass) == (True, False)


def test_a_paired_session_with_zero_qwen_time_is_counted_apart_from_the_qwen_ratios():
    rows = [row("a1-baseline", "b1", 1.0, 100.0, qwen=40.0), row("a2-off-guard", "b1", 0.0, 30.0, qwen=0.0)]
    pairs, _ = pair_up(rows, "a2-off-guard")
    f = pairs[0].fields
    assert f["qwen_zero"] == 1 and f["bs_logq"] == [] and f["bs_logw"] == pytest.approx([math.log(0.3)])


def test_section_of_keeps_tb2_whole_and_splits_later_rounds():
    assert section_of({"benchmark": "terminal-bench-2", "attempt": "3r"}) == "terminal-bench-2"
    assert section_of({"benchmark": "skillsbench", "attempt": "1"}) == "skillsbench"
    assert section_of({"benchmark": "skillsbench", "attempt": "2"}) == "skillsbench#2"
    assert section_of({"benchmark": "skillsbench", "attempt": "1r"}) == "skillsbench"


def test_bootstrap_concatenates_list_fields():
    clusters = [{"v": [1.0, 3.0], "n": 2}, {"v": [2.0], "n": 1}]
    interval = bootstrap_interval(clusters, lambda t: len(t["v"]) / t["n"], resamples=100, seed=2)
    assert (interval.lo, interval.hi) == (1.0, 1.0)


def test_pair_up_leaves_out_and_lists_unscored_and_unpaired_sessions():
    rows = [
        row("a1-baseline", "b1", 1.0, 100.0),
        row("a2-off-guard", "b1", None, 50.0),
        row("a1-baseline", "b2", 1.0, 100.0),
        {"arm": "a2-off-guard", "task": "t1", "benchmark": "terminal-bench-2", "state": "running", "superseded": None},
    ]
    pairs, left_out = pair_up(rows, "a2-off-guard")
    assert pairs == []
    reasons = sorted(reason for _, reason in left_out)
    assert reasons == ["a2-off-guard session finished without a reward", "no finished a2-off-guard session"]


def test_pair_up_counts_a_pair_without_qwen_time_apart():
    rows = [row("a1-baseline", "b1", 0.0, 1.0, qwen=None), row("a2-off-guard", "b1", 0.0, 1.0)]
    pairs, _ = pair_up(rows, "a2-off-guard")
    assert pairs[0].fields["qwen_n"] == 0 and pairs[0].fields["base_qwen"] == 0.0 and pairs[0].fields["bs_qwen_n"] == 0


def test_pair_up_refuses_a_pair_on_two_servers():
    rows = [row("a1-baseline", "b1", 1.0, 100.0), row("a2-off-guard", "b1", 1.0, 50.0, server="s1")]
    with pytest.raises(ValueError, match="server"):
        pair_up(rows, "a2-off-guard")


def test_error_kind_names_known_errors_and_refuses_unknown_ones():
    assert error_kind("NonZeroAgentExitCodeError: Command failed (exit 137)") == "NonZeroAgentExitCodeError"
    assert error_kind("pi output unreadable: x has no agent/pi.txt") == "no pi output"
    assert error_kind("JeffFirst: JeffFirst: turn 660 ran away") == "JeffFirst error"
    assert error_kind("0 result.json files") == "no Harbor result"
    assert error_kind("image pull failed (pull-failed 1)") == "image pull failed"
    with pytest.raises(ValueError):
        error_kind("something new")


def test_exit_code_reads_the_recorded_command_exit_and_flags_unrecorded_ones():
    killed = {"exception": "ApiRateLimitError", "error": "ApiRateLimitError: Command failed (exit 137): . ~/.nvm"}
    assert exit_code(killed) == "137"
    assert exit_code({"exception": "NetworkConnectionError", "error": "NetworkConnectionError: Command failed (exit 143): x"}) == "143"
    assert exit_code({"exception": None, "error": None}) is None
    assert exit_code({"exception": "AgentTimeoutError", "error": None}) is None
    assert exit_code({"exception": "RuntimeError", "error": "pi output unreadable: x has no agent/pi.txt"}) == "not recorded"


def test_load_takes_every_session_of_a_block_with_a_killed_session_out(tmp_path):
    base = {"task": "t1", "attempt": "1", "benchmark": "terminal-bench-2", "server": "s0", "host": "b200",
            "state": "finished", "superseded": None, "cut": None, "reward": 0.0, "agent_s": 10.0, "error": None,
            "exception": None, "started": "2026-10-05T01:00:00+01:00"}
    rows = [
        dict(base, arm="a1-baseline", block="b1", pair_block="b1", folder="f1"),
        dict(base, arm="a3-jeff07", block="b1r", pair_block="b1", folder="f2", attempt="1r", exception="NonZeroAgentExitCodeError",
             error="NonZeroAgentExitCodeError: Command failed (exit 137): pi"),
        dict(base, arm="a1-baseline", block="b2", pair_block="b2", folder="f3", task="t2"),
    ]
    path = tmp_path / "units.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    data = load([str(path)])
    assert [r["folder"] for r in data.active] == ["f3"]
    assert sorted(r["folder"] for r in data.killed_blocks) == ["f1", "f2"]
