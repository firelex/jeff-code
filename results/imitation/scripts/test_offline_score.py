"""Tests of offline_score.py. Run: uv run --with pytest --with aiohttp==3.12.15 --with tokenizers python -m pytest
test_offline_score.py -q (from results/imitation/scripts)."""

import json

import offline_score as os_
import pytest
import routing_labels as rl

RATES = {"b200-nvfp4": {"seconds_per_token": 0.001, "prefill_s": 1.0, "uncached_tokens": 1000.0, "servers": 1}}


def level(good, s, out):
    return {"good": good, "s": s, "out": out}


def make_turn(tid, label, levels, rec_s=10.0, rec_out=1000, step=None, trim=None, later=2):
    return {"id": tid, "task": "t", "split": "development", "trial": "t__x", "session": "s", "turn": 1,
            "class": "information", "machine": "m", "replay_machine": "b200-nvfp4", "rec_s": rec_s,
            "rec_out": rec_out, "router": {"label": label, "calibration": False, "levels": levels},
            "trim": trim, "step": step or [[f"{tid}:d0:tool"]], "later_requests": later}


def question(qid, options, label):
    return {"id": qid, "suite": "t", "family": "t", "state": "S",
            "question": {"type": "choice", "instructions": "Q", "criteria": {o: o for o in options}},
            "label": label, "target": label, "source": {"dataset": "own"}}


def pred(options, probabilities):
    return (list(options), list(probabilities))


# ------------------------------------------------------------------ run-time rules


def test_most_likely_takes_the_first_of_tied_options_in_run_time_order():
    assert os_.most_likely(pred(["off", "low", "medium", "xhigh"], [0.25, 0.25, 0.25, 0.25])) == ("off", 0.25)
    assert os_.most_likely(pred(["a", "b"], [0.4, 0.6])) == ("b", 0.6)


def test_router_rule_uses_xhigh_below_the_threshold_and_scores_cheaper_than_label_as_wrong():
    turn = make_turn("x", "low", {"off": level(False, 2.0, 100), "low": level(True, 6.0, 300), "medium": None})
    options = ["off", "low", "medium", "xhigh"]
    # off at 0.7: below 0.8 -> xhigh, nothing saved
    assert os_.router_outcome(turn, pred(options, [0.7, 0.1, 0.1, 0.1]), 0.8) == ("xhigh", False, 0.0, 0.0, 0, False)
    # off at 0.7 >= 0.5: cheaper than the label low -> wrong, no saving counted
    assert os_.router_outcome(turn, pred(options, [0.7, 0.1, 0.1, 0.1]), 0.5) == ("off", True, 0.0, 0.0, 0, False)
    # low = the label: recorded 10 s minus the measured 6 s; 1000 - 300 tokens; load-matched 10 s x 700 / 1000
    assert os_.router_outcome(turn, pred(options, [0.1, 0.7, 0.1, 0.1]), 0.5) == ("low", False, 4.0, 7.0, 700, False)
    # medium: above the label, never asked -> not wrong, saving 0, counted as unmeasured
    assert os_.router_outcome(turn, pred(options, [0.1, 0.1, 0.7, 0.1]), 0.5) == ("medium", False, 0.0, 0.0, 0, True)
    # xhigh is taken whatever its probability
    assert os_.router_outcome(turn, pred(options, [0.2, 0.2, 0.2, 0.4]), 0.99)[0] == "xhigh"


def test_trim_rule_wrong_saved_and_unjudged():
    trim = {"total_lines": 300, "shown_lines": 300, "label": "first40", "replay_machine": "b200-nvfp4",
            "cuts": {"last40": {"good": False, "prompt_saved": 900}, "first40": {"good": True, "prompt_saved": 800},
                     "first20last20": {"good": False, "prompt_saved": 850}}}
    turn = make_turn("x", "off", {}, trim=trim)
    options = ["all", "last200", "last40", "first40", "first20last20"]
    assert os_.trim_outcome(turn, pred(options, [0.1, 0.0, 0.1, 0.8, 0.0]), 0.5, RATES) == ("first40", False, 800, 0.8, False)
    assert os_.trim_outcome(turn, pred(options, [0.1, 0.0, 0.8, 0.1, 0.0]), 0.5, RATES) == ("last40", True, 0, 0.0, False)
    assert os_.trim_outcome(turn, pred(options, [0.1, 0.0, 0.8, 0.1, 0.0]), 0.9, RATES) == ("all", False, 0, 0.0, False)
    # last200 was not asked on this turn: not judged
    assert os_.trim_outcome(turn, pred(options, [0.1, 0.8, 0.1, 0.0, 0.0]), 0.5, RATES) == ("last200", False, 0, 0.0, True)


def step_fixture():
    # Two decisions: read on page 1 (tool + argument), then list reached through "show more" (page 2).
    turn = make_turn("x", "xhigh", {}, step=[["d0:tool:1", "d0:arg:1"], ["d1:tool:1", "d1:tool:2", "d1:arg:2"]])
    labels = {"d0:tool:1": "read", "d0:arg:1": "read-2", "d1:tool:1": "show_more", "d1:tool:2": "list",
              "d1:arg:2": "list-1"}
    tool = ["read", "list", "show_more", "hand_over"]
    sure = {"d0:tool:1": pred(tool, [0.9, 0.05, 0.0, 0.05]),
            "d0:arg:1": pred(["read-1", "read-2", "none_of_these"], [0.05, 0.9, 0.05]),
            "d1:tool:1": pred(tool, [0.0, 0.05, 0.9, 0.05]),
            "d1:tool:2": pred(["find", "list", "hand_over"], [0.05, 0.9, 0.05]),
            "d1:arg:2": pred(["list-1", "none_of_these"], [0.9, 0.1])}
    return turn, labels, sure


def test_step_covers_a_turn_only_when_every_page_is_confident_and_right():
    turn, labels, sure = step_fixture()
    assert os_.step_outcome(turn, sure, labels, 0.8, 8) == ("covered", 5, 2)
    # below the threshold on the first page: hand over at once
    assert os_.step_outcome(turn, sure, labels, 0.95, 8) == ("handover", 1, 0)
    unsure = dict(sure, **{"d1:tool:2": pred(["find", "list", "hand_over"], [0.3, 0.6, 0.1])})
    assert os_.step_outcome(turn, unsure, labels, 0.8, 8) == ("handover", 4, 1)
    wrong = dict(sure, **{"d0:arg:1": pred(["read-1", "read-2", "none_of_these"], [0.9, 0.05, 0.05])})
    assert os_.step_outcome(turn, wrong, labels, 0.8, 8) == ("wrong", 2, 0)
    # a confident hand over where the recorded step was on the menu: not wrong
    hand = dict(sure, **{"d0:tool:1": pred(["read", "list", "show_more", "hand_over"], [0.05, 0.0, 0.0, 0.95])})
    assert os_.step_outcome(turn, hand, labels, 0.8, 8) == ("handover", 1, 0)
    # the stint cap: a turn with more decisions than the cap is never covered
    assert os_.step_outcome(turn, sure, labels, 0.8, 1) == ("handover", 2, 1)


def test_step_hand_over_label_agreed_is_a_hand_over():
    turn = make_turn("x", "xhigh", {}, step=[["d0:tool:1"]])
    labels = {"d0:tool:1": "hand_over"}
    preds = {"d0:tool:1": pred(["read", "hand_over"], [0.1, 0.9])}
    assert os_.step_outcome(turn, preds, labels, 0.5, 8) == ("handover", 1, 0)
    preds = {"d0:tool:1": pred(["read", "hand_over"], [0.9, 0.1])}
    assert os_.step_outcome(turn, preds, labels, 0.5, 8) == ("wrong", 1, 0)


# ------------------------------------------------------------------ build helpers


def routing_turn(label, checks, asks, calibration=False):
    return {"id": "t__x:e1", "label": label, "checks": checks, "calibration": calibration,
            "asks": [{"level": lv, "sample": 1, "total_s": s, "completion_tokens": out, "chunks": 0}
                     for lv, s, out in asks]}


def test_router_levels_reads_first_samples_and_checks_the_cascade():
    turn = routing_turn("low", {"off": {"good": False}, "low": {"good": True}}, [("off", 2.0, 50), ("low", 5.0, 400)])
    levels = os_.router_levels(turn)
    assert levels == {"off": level(False, 2.0, 50), "low": level(True, 5.0, 400), "medium": None}
    with pytest.raises(ValueError):
        os_.router_levels(routing_turn("low", {"off": {"good": False}}, [("off", 2.0, 50)]))
    with pytest.raises(ValueError):  # a level below the label marked good contradicts the label
        os_.router_levels(routing_turn("low", {"off": {"good": True}, "low": {"good": True}},
                                       [("off", 2.0, 50), ("low", 5.0, 400)]))


def test_prefill_rate_from_vllm_counters_summed_over_servers(tmp_path):
    def page(name, prefill, prompt, cached):
        path = tmp_path / name
        path.write_text(f'vllm:prompt_tokens_total{{engine="0"}} {prompt}\n'
                        f'vllm:prompt_tokens_cached_total{{engine="0"}} {cached}\n'
                        f'vllm:request_prefill_time_seconds_count{{engine="0"}} 7.0\n'
                        f'vllm:request_prefill_time_seconds_sum{{engine="0"}} {prefill}\n')
        return str(path)
    rates = os_.read_prefill_metrics([f"m={page('a', 10.0, 3000.0, 1000.0)}", f"m={page('b', 20.0, 9000.0, 1000.0)}"])
    assert rates["m"]["seconds_per_token"] == pytest.approx(30.0 / 10000.0)
    assert rates["m"]["servers"] == 2
    bad = tmp_path / "bad"
    bad.write_text("vllm:prompt_tokens_total 5\n")
    with pytest.raises(ValueError):
        os_.read_prefill_metrics([f"m={bad}"])


def test_paired_prefill_estimate_is_the_within_turn_slope():
    def ask(tokens, first):
        return {"outcome": "ok", "prompt_tokens": tokens, "first_s": first}
    turns = [{"replay_machine": "m", "asks": [ask(1000, 1.0), ask(500, 0.5)]},
             # a slower turn (queue): same slope, other offset
             {"replay_machine": "m", "asks": [ask(3000, 9.0), ask(1000, 7.0), ask(2000, 8.0)]}]
    rates = os_.paired_prefill_estimate(turns)
    assert rates["m"]["seconds_per_token"] == pytest.approx(0.001)
    assert rates["m"]["requests"] == 5 and rates["m"]["turns"] == 2


class FakeTaskSets:
    def __init__(self, sides):
        self.sides = sides

    def side(self, task):
        return self.sides[task]


def test_held_out_split_and_scoring_view():
    splits = json.loads((os_.HERE.parent / "splits.json").read_text())
    hub_scoring = "terminal-bench-pro/terminal-bench-pro:x"
    sets = FakeTaskSets({hub_scoring: "held_out", "o/d:other": "held_out"})
    assert os_.held_out_split("gcode-to-text", splits, sets, frozenset()) == "development"
    assert os_.held_out_split("build-pmars", splits, sets, frozenset()) == "temperature"
    assert os_.held_out_split(hub_scoring, splits, sets, frozenset({hub_scoring})) == "scoring"
    assert os_.held_out_split("o/d:other", splits, sets, frozenset({hub_scoring})) == "held-out"
    view = os_.ScoringSides(sets, {hub_scoring})
    assert view.side(hub_scoring) == "training" and view.side("o/d:other") == "held_out"


# ------------------------------------------------------------------ whole score on a small bundle


def bundle(tmp_path):
    """Three turns: router labels off / medium / xhigh; one long output; step rows on each turn."""
    turns = [
        make_turn("a", "off", {"off": level(True, 2.0, 100), "low": None, "medium": None}, rec_s=10.0, rec_out=1000,
                  step=[["a:tool"]],
                  trim={"total_lines": 100, "shown_lines": 100, "label": "last40", "replay_machine": "b200-nvfp4",
                        "cuts": {"last40": {"good": True, "prompt_saved": 500},
                                 "first40": {"good": False, "prompt_saved": 500},
                                 "first20last20": {"good": True, "prompt_saved": 480}}}),
        make_turn("b", "medium", {"off": level(False, 3.0, 50), "low": level(False, 9.0, 700),
                                  "medium": level(True, 7.0, 600)}, rec_s=20.0, rec_out=2000,
                  step=[["b:tool", "b:arg"]]),
        make_turn("c", "xhigh", {"off": level(False, 1.0, 10), "low": level(False, 1.0, 10),
                                 "medium": level(False, 1.0, 10)}, rec_s=30.0, rec_out=3000, step=[["c:tool"]]),
    ]
    levels = ["off", "low", "medium", "xhigh"]
    cuts = ["all", "last200", "last40", "first40", "first20last20"]
    questions = {
        "step": [question("a:tool", ["read", "hand_over"], "hand_over"), question("b:tool", ["read", "hand_over"], "read"),
                 question("b:arg", ["read-1", "none_of_these"], "read-1"),
                 question("c:tool", ["read", "hand_over"], "hand_over")],
        "router": [question(f"route:{t['id']}", levels, t["router"]["label"]) for t in turns],
        "trim": [question("trim:a", cuts, "last40")],
    }
    folder = tmp_path / "bundle"
    folder.mkdir()
    os_.write_jsonl(folder / "turns.jsonl", turns)
    os_.write_jsonl(folder / "sessions.jsonl", [{"trial": "t__x", "task": "t", "split": "development",
                                                  "gen_s": 100.0, "tool_s": 100.0, "requests": 3}])
    for decision, rows in questions.items():
        os_.write_jsonl(folder / f"questions-{decision}.jsonl", rows)
    meta = {"splits": ["development"], "step_cap": 8, "prefill": RATES,
            "questions_sha256": {d: os_.sha256_file(folder / f"questions-{d}.jsonl") for d in os_.DECISIONS},
            "turns_sha256": os_.sha256_file(folder / "turns.jsonl")}
    (folder / "meta.json").write_text(json.dumps(meta))
    write_fit_report(folder, {})
    return folder, turns


def write_fit_report(folder, cut_ids):
    """BUNDLE/cut/fit-report.json as jeff_prompt.py fit-examples writes it, for the bundle's question files."""
    files = {}
    for decision in os_.DECISIONS:
        path = folder / f"questions-{decision}.jsonl"
        ids = cut_ids.get(decision, [])
        files[path.name] = {"input": str(path), "input_sha256": os_.sha256_file(path), "rows": len(os_.read_jsonl(path)),
                            "rows_cut": len(ids), "longest_before": 9000, "longest_after": 8100,
                            "cut": [{"id": i, "tokens_before": 9000, "tokens_after": 8100, "lines_left_out": 40}
                                    for i in ids]}
    (folder / "cut").mkdir(exist_ok=True)
    (folder / "cut" / "fit-report.json").write_text(json.dumps({"limit": 8192, "files": files}))


def run_score(tmp_path, folder, kind, decision_ms=0.0):
    preds = {}
    for decision in os_.DECISIONS:
        out = tmp_path / kind / f"{decision}.jsonl"
        os_.main(["oracle", "--kind", kind, "--decision", decision, "--bundle", str(folder), "--out", str(out)])
        preds[decision] = str(out)
    out = tmp_path / "scores" / kind
    os_.main(["score", "--bundle", str(folder), "--name", kind, "--step", preds["step"], "--router", preds["router"],
              "--trim", preds["trim"], "--decision-ms", str(decision_ms), "--out", str(out)])
    return json.loads(out.with_suffix(".json").read_text())


def test_fixed_oracle_saves_nothing_and_is_never_wrong(tmp_path):
    folder, _ = bundle(tmp_path)
    result = run_score(tmp_path, folder, "fixed")
    for decision in os_.DECISIONS:
        for row in result["tables"][decision]:
            assert row["net_saved_s"] == 0 and row["wrong"] == 0
    for row in result["combined_common"]:
        assert row["net_saved_s"] == 0 and row["wrong"] == 0 and row["covered"] == 0


def test_label_oracle_reproduces_the_routing_stats_saving_and_combines_without_double_counting(tmp_path):
    folder, _ = bundle(tmp_path)
    result = run_score(tmp_path, folder, "labels")
    router = result["tables"]["router"][0]
    # routing-stats.md "routed saved" on the same turns: 1 - (100 + 600 + 3000) / 6000
    assert router["output_tokens_saved_share"] == pytest.approx(1 - 3700 / 6000)
    assert result["totals"]["label_routed_output_saved_share"] == pytest.approx(router["output_tokens_saved_share"])
    assert router["wrong"] == 0 and router["gross_saved_s"] == pytest.approx((10 - 2) + (20 - 7))
    step = result["tables"]["step"][0]
    assert step["covered"] == 1 and step["gross_saved_s"] == pytest.approx(20.0) and step["wrong"] == 0
    trim = result["tables"]["trim"][0]
    assert trim["prompt_tokens_saved"] == 500 and trim["gross_saved_s"] == pytest.approx(0.5)
    assert trim["prompt_tokens_saved_with_repeats"] == 500 * 3
    combined = result["combined_common"][0]
    # turn b is taken by Jeff (20 s), so its router saving (13 s) is not added; turn a: router 8 s + trim 0.5 s
    assert combined["gross_saved_s"] == pytest.approx(20.0 + 8.0 + 0.5)
    assert combined["wrong"] == 0
    assert combined["net_saved_share_gen"] == pytest.approx(28.5 / 100)
    assert combined["net_saved_share_gen_tool"] == pytest.approx(28.5 / 200)
    # load-matched: turn a's router saving is 10 s x (1000 - 100) / 1000 = 9 s instead of the measured 8 s
    assert combined["matched_net_saved_s"] == pytest.approx(20.0 + 9.0 + 0.5)


def test_jeff_time_is_charged_per_request(tmp_path):
    folder, _ = bundle(tmp_path)
    result = run_score(tmp_path, folder, "labels", decision_ms=100)
    # step pages asked: a 1, b 2, c 1; router asked on a and c (b is covered); trim on a
    assert result["tables"]["step"][0]["jeff_s"] == pytest.approx(0.4)
    assert result["combined_common"][0]["jeff_s"] == pytest.approx(0.4 + 0.2 + 0.1)


def test_label_share_matches_routing_labels_stats_formula():
    labels = []
    for i, (label, out) in enumerate([("off", 100), ("xhigh", None), ("low", 300)]):
        asks = [] if out is None else [{"level": label, "sample": 1, "completion_tokens": out, "chunks": 0}]
        labels.append({"label": label, "recorded": {"output_tokens": 1000 * (i + 1)}, "asks": asks})
    routed = sum(t["recorded"]["output_tokens"] if t["label"] == "xhigh" else
                 rl._tokens(next(a for a in t["asks"] if a["level"] == t["label"] and a["sample"] == 1)) for t in labels)
    assert 1 - routed / 6000 == pytest.approx(1 - (100 + 2000 + 300) / 6000)


def test_predictions_must_cover_every_question_with_its_options(tmp_path):
    q = [question("q1", ["a", "b"], "a")]
    path = tmp_path / "p.jsonl"
    os_.write_jsonl(path, [{"id": "q1", "options": ["b", "a"], "probabilities": [0.3, 0.7]}])
    assert os_.read_predictions(path, q) == {"q1": (["a", "b"], [0.7, 0.3])}
    os_.write_jsonl(path, [{"id": "q1", "options": ["a", "c"], "probabilities": [0.3, 0.7]}])
    with pytest.raises(ValueError):
        os_.read_predictions(path, q)
    os_.write_jsonl(path, [])
    with pytest.raises(ValueError):
        os_.read_predictions(path, q)


def test_best_at_picks_the_largest_saving_within_the_wrong_rate():
    rows = [{"threshold": 0.5, "wrong_rate": 0.12, "net_saved_s": 50},
            {"threshold": 0.7, "wrong_rate": 0.04, "net_saved_s": 30},
            {"threshold": 0.9, "wrong_rate": 0.01, "net_saved_s": 10}]
    assert os_.best_at(rows, 0.05)["threshold"] == 0.7
    assert os_.best_at(rows, 0.02)["threshold"] == 0.9
    assert os_.best_at(rows, 0.001) is None


def test_compare_ranks_variants_on_the_same_bundle(tmp_path):
    folder, _ = bundle(tmp_path)
    run_score(tmp_path, folder, "labels")
    run_score(tmp_path, folder, "fixed")
    out = tmp_path / "comparison.md"
    os_.main(["compare", "--scores", str(tmp_path / "scores" / "fixed.json"), str(tmp_path / "scores" / "labels.json"),
              "--out", str(out)])
    text = out.read_text()
    section = text.split("## Wrong-choice rate at most 2.0% (measured router seconds)")[1]
    assert section.index("| labels |") < section.index("| fixed |")
    other = json.loads((tmp_path / "scores" / "fixed.json").read_text())
    other["bundle_turns_sha256"] = "different"
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(ValueError):
        os_.main(["compare", "--scores", str(tmp_path / "other.json"), str(tmp_path / "scores" / "labels.json"),
                  "--out", str(out)])


def test_cut_questions_are_scored_like_the_others_and_counted(tmp_path):
    folder, _ = bundle(tmp_path)
    write_fit_report(folder, {"step": ["b:arg"]})
    result = run_score(tmp_path, folder, "labels")
    assert result["cut_to_fit"] == {"step": 1, "router": 0, "trim": 0}
    # b's argument page was cut, and Jeff still covers turn b as without the cut
    assert result["tables"]["step"][0]["covered"] == 1
    assert "Questions cut to fit Jeff's 8,192-token limit (as the run time cuts them; scored like the others): step 1, " \
        "router 0, trim 0." in (tmp_path / "scores" / "labels.md").read_text()


def test_score_needs_the_bundle_cut_to_fit(tmp_path):
    folder, _ = bundle(tmp_path)
    (folder / "cut" / "fit-report.json").unlink()
    with pytest.raises(ValueError, match="cut the bundle's questions to fit first"):
        run_score(tmp_path, folder, "labels")
    write_fit_report(folder, {})
    report = json.loads((folder / "cut" / "fit-report.json").read_text())
    report["files"]["questions-trim.jsonl"]["input_sha256"] = "other"  # a report made from other question files
    (folder / "cut" / "fit-report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cut other questions-trim.jsonl than this bundle's"):
        run_score(tmp_path, folder, "fixed")
