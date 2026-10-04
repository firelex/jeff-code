"""Tests of tonight_training_files.py. Run: uv run --no-project --with pytest python -m pytest
test_tonight_training_files.py -q (from results/imitation/scripts)."""

import json
from pathlib import Path

import pytest

import tonight_training_files as ttf


def step(source, stage, task, n, label="hand_over", level="tool", family=None, dataset=None):
    return {"id": f"{source}:{stage}:{task}:s{n}:0:{level}:1", "suite": task, "family": family or task, "state": "S",
            "question": {"type": "choice", "instructions": "Q", "criteria": {label: "x", "hand_over": "h"}},
            "label": label, "target": label,
            "source": {"dataset": dataset or source, "stage": stage, "task": task, "level": level}}


def decision(kind, task, n, label):
    return {"id": f"own:{kind}:{task}:s{n}:{n}", "suite": task, "family": task, "state": "S",
            "question": {"type": "choice", "instructions": "Q", "criteria": {label: "x"}}, "label": label,
            "target": label, "source": {"dataset": "own", "stage": 3, "task": task}}


def write(folder: Path, splits: dict) -> Path:
    folder.mkdir(parents=True)
    for name in ttf.SPLITS:
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in splits[name]))
    return folder


HUB = "terminal-bench-pro/terminal-bench-pro"


def inputs(tmp_path: Path, stage3_train_task="tb-train") -> dict:
    stage1_train = [step("nl2bash", 1, f"g{g}-t{t}", g * 10 + t, family=f"g{g}") for g in range(30) for t in range(1 + g % 3)]
    paths = {
        "stage1": write(tmp_path / "s1", {"train": stage1_train, "development": [step("nl2bash", 1, "d", 1, family="gd")],
                                          "temperature": [step("nl2bash", 1, "e", 1, family="ge")]}),
        "stage2": write(tmp_path / "s2", {"train": [step("openguardrails", 2, "tb-train", 1, label="read")],
                                          "development": [step("openguardrails", 2, "tb-dev", 2)],
                                          "temperature": [step("openguardrails", 2, "tb-temp", 3)]}),
        "stage3": write(tmp_path / "s3", {
            "train": [step("own", 3, stage3_train_task, n, label="read" if n % 4 == 0 else "hand_over") for n in range(10)]
            + [step("own", 3, f"{HUB}:fix-a", 99, label="read-1", level="argument")],
            "development": [step("own", 3, "tb-dev", 1)], "temperature": [step("own", 3, "tb-temp", 1)]}),
        "router": write(tmp_path / "r", {"train": [decision("route", "tb-train", n, "off") for n in range(4)],
                                         "development": [decision("route", "tb-dev", 1, "xhigh")],
                                         "temperature": [decision("route", "tb-temp", 1, "low")]}),
        "trim": write(tmp_path / "t", {"train": [decision("trim", "tb-train", n, "last40") for n in range(3)],
                                       "development": [decision("trim", "tb-dev", 1, "all")],
                                       "temperature": [decision("trim", "tb-temp", 1, "all")]}),
    }
    sets = {"datasets": {"terminal-bench-pro": {"hub": {"name": HUB}, "training": ["fix-a"], "held_out": ["held-c"], "excluded": {}}}}
    (tmp_path / "sets.json").write_text(json.dumps(sets))
    (tmp_path / "scoring.txt").write_text(f"{HUB}:held-c\n")
    (tmp_path / "tasks.json").write_text(json.dumps({"training": ["tb-train", "tb-dev", "tb-temp"]}))
    return paths


def run_build(tmp_path: Path, paths: dict, out="out") -> dict:
    argv = ["build"] + [f"--{key}={value}" for key, value in paths.items()] + [
        f"--task-sets={tmp_path / 'sets.json'}", f"--scoring-tasks={tmp_path / 'scoring.txt'}",
        f"--training-tasks={tmp_path / 'tasks.json'}", "--seed=7", f"--out={tmp_path / out}"]
    ttf.main(argv)
    return json.loads((tmp_path / out / "manifest.json").read_text())


def ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


def test_the_curriculum_samples_stage_1_to_stage_3_size_in_whole_groups_then_stage_2_then_stage_3_twice(tmp_path):
    manifest = run_build(tmp_path, inputs(tmp_path))
    train = ids(tmp_path / "out" / "step-curriculum" / "train.jsonl")
    stages = [ttf.stage_of(i) for i in train]
    assert stages == ["1"] * 11 + ["2"] + ["3"] * 22
    stage3 = [i for i in train if ttf.stage_of(i) == "3"]
    originals = sorted(i for i in stage3 if not i.endswith(":copy2"))
    assert len(originals) == 11 and sorted(i.removesuffix(":copy2") for i in stage3 if i.endswith(":copy2")) == originals
    families = {i.split(":")[2].split("-t")[0] for i in train if ttf.stage_of(i) == "1"}
    rows1 = [json.loads(line) for line in (tmp_path / "s1" / "train.jsonl").read_text().splitlines()]
    # whole groups: every row of a chosen family is in
    assert sum(1 for row in rows1 if row["family"] in families) == 11
    assert manifest["stage1_sampling"]["rows"] == 11
    assert manifest["files"]["step-curriculum"]["train"]["rows_by_stage"] == {"1": 11, "2": 1, "3": 22}
    assert manifest["files"]["step-curriculum"]["stage_order"] == ["1", "2", "3"]
    assert ids(tmp_path / "out" / "step-curriculum" / "development.jsonl") == ["own:3:tb-dev:s1:0:tool:1"]


def test_step_stage3_router_trim_and_full(tmp_path):
    manifest = run_build(tmp_path, inputs(tmp_path))
    assert len(ids(tmp_path / "out" / "step-stage3" / "train.jsonl")) == 22
    assert ids(tmp_path / "out" / "router" / "train.jsonl")[0].startswith("own:route:")
    full = ids(tmp_path / "out" / "full" / "train.jsonl")
    assert [ttf.stage_of(i) for i in full] == ["1"] * 11 + ["2"] + ["3"] * 29
    assert "own:3:route:tb-train:s0:0" in full and "own:3:trim:tb-train:s2:2" in full
    assert len(ids(tmp_path / "out" / "full" / "development.jsonl")) == 3
    labels = manifest["files"]["full"]["train"]["labels"]
    assert labels["route"]["off"]["rows"] == 4 and labels["trim"]["last40"]["rows"] == 3
    assert labels["step-tool"]["act"]["rows"] == 2 * 3 + 1  # stage 3 reads (n = 0, 4, 8) twice, stage 2 once
    assert labels["step-argument"]["option"]["rows"] == 2


def test_a_scoring_or_held_out_task_anywhere_is_an_error(tmp_path):
    paths = inputs(tmp_path, stage3_train_task=f"{HUB}:held-c")
    with pytest.raises(ValueError, match="scoring task"):
        run_build(tmp_path, paths)


def test_merge_rows_refuses_a_session_on_two_hosts(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"session": "x", "n": 1}) + "\n" + json.dumps({"session": "y"}) + "\n")
    b.write_text(json.dumps({"session": "z"}) + "\n")
    assert ttf.merge_rows(tmp_path / "m.jsonl", [a, b])["sessions"] == 3
    b.write_text(json.dumps({"session": "y"}) + "\n")
    with pytest.raises(ValueError, match="both"):
        ttf.merge_rows(tmp_path / "m2.jsonl", [a, b])
