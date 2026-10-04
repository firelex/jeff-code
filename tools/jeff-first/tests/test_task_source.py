import json
import subprocess
import sys
from pathlib import Path

import pytest

from task_source import (
    TB2_DEFAULT_AGENT_SEC,
    agent_timeout_multiplier,
    hub_harbor_args,
    is_hub_task,
    load_task_sets,
    safe_name,
)

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE.parent.parent / "results" / "imitation"
PRO_REF = "sha256:" + "a" * 64


def write_task_sets(tmp_path: Path) -> tuple[Path, Path]:
    task_sets = {
        "datasets": {
            "terminal-bench-pro": {
                "hub": {"name": "terminal-bench-pro/terminal-bench-pro", "ref": PRO_REF, "revision": 1},
                "training": ["fix-a", "slow-b"],
                "held_out": ["held-c"],
                "excluded": {"leak-d": "near evaluation task x"},
            },
            "terminal-bench-2": {
                "hub": {"name": "terminal-bench/terminal-bench-2", "ref": "sha256:" + "b" * 64, "revision": 1},
                "training": ["dna-insert"],
                "held_out": [],
                "excluded": {},
            },
            "unpinned": {"training": ["t"], "held_out": [], "excluded": {}},
        }
    }
    inventory = [
        {"dataset": "terminal-bench-pro", "task": "fix-a", "task_name": "terminal-bench-pro/fix-a", "agent_timeout_sec": 600.0, "gpus": 0, "mcp_servers": 0},
        {"dataset": "terminal-bench-pro", "task": "slow-b", "task_name": "terminal-bench-pro/slow-b", "agent_timeout_sec": 28800.0, "gpus": 0, "mcp_servers": 0},
        {"dataset": "terminal-bench-pro", "task": "held-c", "task_name": "terminal-bench-pro/held-c", "agent_timeout_sec": 3600.0, "gpus": 0, "mcp_servers": 0},
        {"dataset": "terminal-bench-pro", "task": "leak-d", "task_name": "terminal-bench-pro/leak-d", "agent_timeout_sec": 3600.0, "gpus": 0, "mcp_servers": 0},
    ]
    sets_path, inventory_path = tmp_path / "task-sets.json", tmp_path / "task-sets-inventory.json"
    sets_path.write_text(json.dumps(task_sets))
    inventory_path.write_text(json.dumps(inventory))
    return sets_path, inventory_path


def test_hub_task_ids_have_a_dataset_and_bare_names_are_terminal_bench_2():
    assert is_hub_task("terminal-bench-pro/terminal-bench-pro:fix-a")
    assert not is_hub_task("adaptive-rejection-sampler")
    with pytest.raises(ValueError, match="dataset"):
        is_hub_task("terminal-bench-pro:fix-a")
    with pytest.raises(ValueError, match="dataset"):
        is_hub_task("terminal-bench-pro/fix-a")


def test_a_training_task_resolves_to_the_pinned_dataset_and_its_package_name(tmp_path):
    sets = load_task_sets(*write_task_sets(tmp_path))
    task = sets.resolve("terminal-bench-pro/terminal-bench-pro:fix-a")
    assert task.dataset_spec == f"terminal-bench-pro/terminal-bench-pro@{PRO_REF}"
    assert task.include == "terminal-bench-pro/fix-a"
    assert task.side == "training"
    assert task.agent_timeout_sec == 600.0
    assert sets.side("terminal-bench-pro/terminal-bench-pro:held-c") == "held_out"
    assert sets.side("terminal-bench-pro/terminal-bench-pro:leak-d") == "excluded"


@pytest.mark.parametrize(
    ("task_id", "message"),
    [
        ("nobody/nothing:fix-a", "unknown dataset nobody/nothing"),
        ("terminal-bench-pro/terminal-bench-pro:no-such-task", "no task no-such-task"),
        ("terminal-bench-pro/terminal-bench-pro:leak-d", "excluded: near evaluation task x"),
        ("terminal-bench/terminal-bench-2:dna-insert", "bare task name"),
    ],
)
def test_unknown_excluded_and_terminal_bench_2_hub_tasks_raise(tmp_path, task_id, message):
    sets = load_task_sets(*write_task_sets(tmp_path))
    with pytest.raises(ValueError, match=message):
        sets.resolve(task_id)


@pytest.mark.parametrize(("field", "message"), [("gpus", "needs a GPU"), ("mcp_servers", "needs MCP tools")])
def test_a_task_needing_a_gpu_or_mcp_tools_raises_even_if_listed(tmp_path, field, message):
    sets_path, inventory_path = write_task_sets(tmp_path)
    inventory = json.loads(inventory_path.read_text())
    inventory[0][field] = 1
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match=message):
        load_task_sets(sets_path, inventory_path).resolve("terminal-bench-pro/terminal-bench-pro:fix-a")


def test_a_dataset_without_a_pinned_version_raises(tmp_path):
    sets_path, inventory_path = write_task_sets(tmp_path)
    data = json.loads(sets_path.read_text())
    data["datasets"]["unpinned"]["hub"] = {"name": "x/unpinned", "ref": "latest", "revision": 1}
    sets_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pinned"):
        load_task_sets(sets_path, inventory_path)


def test_the_agent_time_is_capped_at_the_terminal_bench_default_times_the_multiplier():
    # Terminal-Bench 2.0's default agent limit is 15 minutes; times 6 gives the 90-minute cap.
    assert TB2_DEFAULT_AGENT_SEC == 900
    assert agent_timeout_multiplier(600, 6) == 6  # shorter tasks keep their limit times the multiplier: 60 minutes
    assert agent_timeout_multiplier(900, 6) == 6
    assert agent_timeout_multiplier(1800, 6) == 3  # 90 minutes
    assert agent_timeout_multiplier(3600, 6) == 1.5
    assert agent_timeout_multiplier(28800, 6) == 0.1875
    assert agent_timeout_multiplier(28800, 2.5) * 28800 == 900 * 2.5
    with pytest.raises(ValueError, match="positive"):
        agent_timeout_multiplier(0, 6)


def test_harbor_args_and_safe_names(tmp_path):
    sets = load_task_sets(*write_task_sets(tmp_path))
    assert hub_harbor_args(sets, "terminal-bench-pro/terminal-bench-pro:slow-b", 6) == [
        f"terminal-bench-pro/terminal-bench-pro@{PRO_REF}",
        "terminal-bench-pro/slow-b",
        "0.1875",
        "terminal-bench-pro.terminal-bench-pro.slow-b",
        "online",
    ]
    assert safe_name("terminal-bench-pro/terminal-bench-pro:slow-b") == "terminal-bench-pro.terminal-bench-pro.slow-b"


def test_the_cli_prints_the_harbor_args_and_fails_loudly(tmp_path):
    sets_path, inventory_path = write_task_sets(tmp_path)
    run = [sys.executable, str(HERE / "task_source.py"), "harbor-args", str(sets_path), str(inventory_path), "6"]
    out = subprocess.run([*run, "terminal-bench-pro/terminal-bench-pro:fix-a"], capture_output=True, text=True, check=True)
    assert out.stdout.split() == [f"terminal-bench-pro/terminal-bench-pro@{PRO_REF}", "terminal-bench-pro/fix-a", "6", "terminal-bench-pro.terminal-bench-pro.fix-a", "online"]
    bad = subprocess.run([*run, "terminal-bench-pro/terminal-bench-pro:held-x"], capture_output=True, text=True)
    assert bad.returncode != 0 and "no task held-x" in bad.stderr
    check = [sys.executable, str(HERE / "task_source.py"), "check", str(sets_path), str(inventory_path)]
    assert subprocess.run([*check, "adaptive-rejection-sampler", "terminal-bench-pro/terminal-bench-pro:fix-a"], capture_output=True).returncode == 0
    bad = subprocess.run([*check, "terminal-bench-pro/terminal-bench-pro:leak-d"], capture_output=True, text=True)
    assert bad.returncode != 0 and "excluded" in bad.stderr


def test_the_real_task_sets_pin_every_training_dataset_and_resolve_all_training_tasks():
    sets = load_task_sets(RESULTS / "task-sets.json", RESULTS / "task-sets-inventory.json")
    ids = sets.training_ids()
    by_dataset = {}
    for task_id in ids:
        by_dataset[task_id.split(":")[0]] = by_dataset.get(task_id.split(":")[0], 0) + 1
    # The first survey's 182 (188 less 6 that need the model to see an image), then the second survey's SkillsBench
    # (43 less drone-planning-control, which passes an API key into its container) and SWE-rebench.
    assert by_dataset["terminal-bench-pro/terminal-bench-pro"] + by_dataset["terminal-bench/terminal-bench"] + by_dataset[
        "terminal-bench-science/terminal-bench-science"] + by_dataset["harbor-index/harbor-index-1.0"] == 182
    assert by_dataset["benchflow/skillsbench"] == 42
    assert "benchflow/skillsbench:drone-planning-control" not in ids
    assert "swe-rebench/swe-rebench-leaderboard" in by_dataset
    for task_id in ids:
        task = sets.resolve(task_id)
        assert task.side == "training"
        assert task.agent_timeout_sec * float(agent_timeout_multiplier(task.agent_timeout_sec, 6)) <= 5400


def add_swe_rebench(sets_path: Path, inventory_path: Path) -> str:
    data = json.loads(sets_path.read_text())
    data["datasets"]["swe-rebench-leaderboard"] = {
        "hub": {"name": "swe-rebench/swe-rebench-leaderboard", "ref": "sha256:" + "e" * 64, "revision": 2},
        "training": ["ASPP__pelita-863"],
        "held_out": [],
        "excluded": {},
    }
    sets_path.write_text(json.dumps(data))
    rows = json.loads(inventory_path.read_text())
    rows.append({"dataset": "swe-rebench-leaderboard", "task": "ASPP__pelita-863", "task_name": "swe-rebench/ASPP__pelita-863", "agent_timeout_sec": 3000, "gpus": 0, "mcp_servers": 0, "dockerfile_base": "swerebench/sweb.eval.x86_64.aspp_1776_pelita-863:latest"})
    inventory_path.write_text(json.dumps(rows))
    return "swe-rebench/swe-rebench-leaderboard:ASPP__pelita-863"


def test_swe_rebench_runs_offline_from_a_rotated_image(tmp_path):
    sets_path, inventory_path = write_task_sets(tmp_path)
    task_id = add_swe_rebench(sets_path, inventory_path)
    sets = load_task_sets(sets_path, inventory_path)
    assert hub_harbor_args(sets, task_id, 6)[-1] == "offline"
    assert sets.resolve(task_id).image == "swerebench/sweb.eval.x86_64.aspp_1776_pelita-863:latest"
    assert sets.resolve("terminal-bench-pro/terminal-bench-pro:fix-a").image is None
    run = [sys.executable, str(HERE / "task_source.py"), "image", str(sets_path), str(inventory_path)]
    assert subprocess.run([*run, task_id], capture_output=True, text=True, check=True).stdout.strip() == "swerebench/sweb.eval.x86_64.aspp_1776_pelita-863:latest"
    assert subprocess.run([*run, "terminal-bench-pro/terminal-bench-pro:fix-a"], capture_output=True, text=True, check=True).stdout.strip() == "-"
    assert subprocess.run([*run, "adaptive-rejection-sampler"], capture_output=True, text=True, check=True).stdout.strip() == "-"
