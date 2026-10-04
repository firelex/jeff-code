import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from collect_queue import ALL_RUNNING, claim, count_finished, plan, release
from tests.test_task_source import write_task_sets

HERE = Path(__file__).resolve().parent.parent
DEAD_PID = 2**22 + 12345  # above Linux's and macOS's pid limits, so never alive


def task_sets(tmp_path: Path, n: int = 12) -> tuple[Path, Path]:
    """A hub dataset with n training tasks; the last two are big (16 CPUs or a sidecar service)."""
    sets_path, inventory_path = write_task_sets(tmp_path)
    data = json.loads(sets_path.read_text())
    tasks = [f"t{i:02d}" for i in range(n)]
    data["datasets"]["terminal-bench-pro"]["training"] = tasks
    sets_path.write_text(json.dumps(data))
    inventory = [
        {"dataset": "terminal-bench-pro", "task": t, "task_name": f"terminal-bench-pro/{t}", "agent_timeout_sec": 3600.0, "gpus": 0, "mcp_servers": 0, "cpus": 2, "memory_mb": 4096, "compose_services": []}
        for t in tasks
    ]
    inventory[-1]["cpus"] = 16
    inventory[-2]["compose_services"] = ["db"]
    inventory_path.write_text(json.dumps(inventory))
    return sets_path, inventory_path


def tid(t: str) -> str:
    return f"terminal-bench-pro/terminal-bench-pro:{t}"


def test_plan_gives_every_task_to_exactly_one_host_by_stream_share_and_big_tasks_to_big_hosts(tmp_path):
    sets, inventory = task_sets(tmp_path)
    queues = plan(sets, inventory, {}, [("b200", 6, False), ("rtx", 3, True)], seed=1)
    every = queues["b200"] + queues["rtx"]
    assert sorted(every) == sorted(tid(f"t{i:02d}") for i in range(12))
    assert len(set(every)) == 12
    assert len(queues["b200"]) == 8 and len(queues["rtx"]) == 4
    assert tid("t11") in queues["b200"] and tid("t10") in queues["b200"]  # 16 CPUs, sidecar: not on the small host


def test_plan_puts_tasks_with_finished_sessions_last_fewest_first_and_is_deterministic(tmp_path):
    sets, inventory = task_sets(tmp_path)
    counts = {tid("t00"): 3, tid("t01"): 1, "adaptive-rejection-sampler": 6}
    queues = plan(sets, inventory, counts, [("b200", 1, False)], seed=1)
    order = queues["b200"]
    assert order[-2:] == [tid("t01"), tid("t00")]
    assert "adaptive-rejection-sampler" not in order  # Terminal-Bench 2.0 is not queued
    assert plan(sets, inventory, counts, [("b200", 1, False)], seed=1) == queues
    # The partition does not depend on the counts, so hosts that see different counts still agree on it.
    two = [("b200", 2, False), ("cas", 1, False)]
    assert {h: set(q) for h, q in plan(sets, inventory, counts, two, seed=1).items()} == {h: set(q) for h, q in plan(sets, inventory, {}, two, seed=1).items()}


def make_queue(tmp_path: Path, tasks: list[str], finished: dict | None = None) -> Path:
    state = tmp_path / "queue"
    state.mkdir()
    (state / "queue.json").write_text(json.dumps({"order": tasks, "finished_before": finished or {}}))
    return state


def test_claims_go_least_run_first_and_never_hand_a_running_task_to_a_second_stream(tmp_path):
    state = make_queue(tmp_path, ["a", "b", "c"], {"a": 1})
    me = os.getpid()
    assert claim(state, "s1", me) == ("b", 1)
    assert claim(state, "s2", me) == ("c", 1)
    assert claim(state, "s3", me) == ("a", 2)  # a ran once before this queue started
    assert claim(state, "s4", me) == ALL_RUNNING
    release(state, "s1", "b")
    assert claim(state, "s4", me) == ("b", 2)
    release(state, "s2", "c")
    release(state, "s3", "a")
    # c has run once, a twice, b twice: c first.
    assert claim(state, "s2", me) == ("c", 2)


def test_a_claim_of_a_stream_that_died_is_freed(tmp_path):
    state = make_queue(tmp_path, ["a"])
    assert claim(state, "s1", DEAD_PID) == ("a", 1)
    assert claim(state, "s2", os.getpid()) == ("a", 2)


def test_a_stream_holds_one_task_and_release_checks_the_task(tmp_path):
    state = make_queue(tmp_path, ["a", "b"])
    assert claim(state, "s1", os.getpid()) == ("a", 1)
    with pytest.raises(ValueError, match="already runs a"):
        claim(state, "s1", os.getpid())
    with pytest.raises(ValueError, match="does not run b"):
        release(state, "s1", "b")


def finished_trial(root: Path, stream: str, task_id: str, n: int, *, trace: str = "x\n", result: bool = True) -> None:
    trial = root / stream / "round1" / f"job{n}" / f"t__{n}"
    (trial / "agent").mkdir(parents=True)
    if ":" in task_id:
        hub, name = task_id.split(":")
        task = {"name": f"{hub.split('/')[0]}/{name}", "ref": "sha256:" + "c" * 64, "source": hub}
    else:
        task = {"path": task_id}
    (trial / "config.json").write_text(json.dumps({"task": task}))
    (trial / "agent" / "jeff-first-trace.jsonl").write_text(trace)
    if result:
        (trial / "result.json").write_text("{}")


def test_count_finished_needs_a_result_and_a_non_empty_trace(tmp_path):
    root = tmp_path / "runs-collect-xhigh2"
    finished_trial(root, "s1", tid("t00"), 1)
    finished_trial(root, "s1", tid("t00"), 2)
    finished_trial(root, "s2", "dna-assembly", 3)
    finished_trial(root, "s2", tid("t01"), 4, trace="")
    finished_trial(root, "s2", tid("t02"), 5, result=False)
    assert count_finished([root]) == {tid("t00"): 2, "dna-assembly": 1}


def test_the_cli_plans_claims_and_releases(tmp_path):
    sets, inventory = task_sets(tmp_path, n=3)
    root = tmp_path / "runs-collect-xhigh-hub"
    finished_trial(root, "s1", tid("t00"), 1)
    cli = [sys.executable, str(HERE / "collect_queue.py")]
    counts = tmp_path / "counts.json"
    subprocess.run([*cli, "count", str(counts), str(root)], check=True)
    state = tmp_path / "queue-b200"
    subprocess.run([*cli, "plan", str(sets), str(inventory), str(state), "b200", "--host", "b200:2", "--host", "rtx:1:small", "--counts", str(counts), "--seed", "1"], check=True)
    order = json.loads((state / "queue.json").read_text())["order"]
    assert order[-1] == tid("t00") or tid("t00") not in order
    first = subprocess.run([*cli, "claim", str(state), "s1", str(os.getpid())], capture_output=True, text=True, check=True).stdout.split()
    assert first == [order[0], "1"]
    subprocess.run([*cli, "release", str(state), "s1", order[0]], check=True)
    # Planning again over an existing queue would forget its claims: refused.
    again = subprocess.run([*cli, "plan", str(sets), str(inventory), str(state), "b200", "--host", "b200:2"], capture_output=True, text=True)
    assert again.returncode != 0 and "already" in again.stderr


def test_hub_launch_takes_the_streams_urls_and_drivers_of_the_old_launcher(tmp_path):
    collect = tmp_path / "collect"
    (collect / "hub" / "tools").mkdir(parents=True)
    (collect / "hub" / "tools" / "jeff-first").symlink_to(HERE)
    sets, inventory = task_sets(tmp_path, n=4)
    (collect / "hub" / "task-sets.json").write_text(sets.read_text())
    (collect / "hub" / "task-sets-inventory.json").write_text(inventory.read_text())
    old = collect / "collect_launch_b200.sh"
    old.write_text(
        '#!/usr/bin/env bash\n'
        'tmux -L jeffcollect new-session -d -s collect-b200-gpu0-1 "cd /c && bash collect_rounds.sh /c http://192.168.3.12:8885 runs-collect-xhigh2/b200-gpu0-s1 qwen3.8-27b-nvfp4@b200-gpu0-vllm0.29 0 10"\n'
        'tmux -L jeffcollect new-session -d -s collect-b200-gpu7-6 "cd /c && bash collect_rounds.sh /c http://192.168.3.12:8892 runs-collect-xhigh2/b200-gpu7-s6 qwen3.8-27b-nvfp4@b200-gpu7-vllm0.29 329 10"\n'
    )
    result = subprocess.run(["bash", str(HERE / "hub_launch.sh"), "--dry-run", str(collect), "b200", "collect_launch_b200.sh", "--host", "b200:2", "--host", "rtx:1:small"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("hub-")]
    c = collect.resolve()
    assert lines == [
        f"hub-b200-gpu0-s1: bash {c}/hub/tools/jeff-first/hub_stream.sh {c} {c}/hub/queue http://192.168.3.12:8885 runs-collect-xhigh-hub/b200-gpu0-s1 qwen3.8-27b-nvfp4@b200-gpu0-vllm0.29 b200-gpu0-s1",
        f"hub-b200-gpu7-s6: bash {c}/hub/tools/jeff-first/hub_stream.sh {c} {c}/hub/queue http://192.168.3.12:8892 runs-collect-xhigh-hub/b200-gpu7-s6 qwen3.8-27b-nvfp4@b200-gpu7-vllm0.29 b200-gpu7-s6",
    ]
    assert "b200: 3 tasks of 4" in result.stdout or "b200: 2 tasks of 4" in result.stdout
    assert not (collect / "hub" / "queue").exists()  # a dry run plans into a temporary folder
