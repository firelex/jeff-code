import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from collect_queue import ALL_RUNNING, append, claim, count_finished, plan, release
from tests.test_task_source import write_task_sets

HERE = Path(__file__).resolve().parent.parent
DEAD_PID = 2**22 + 12345
PRO = "terminal-bench-pro/terminal-bench-pro"  # above Linux's and macOS's pid limits, so never alive


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
    queues = plan(sets, inventory, {}, [("b200", 6, False), ("rtx", 3, True)], [PRO], seed=1)
    every = queues["b200"] + queues["rtx"]
    assert sorted(every) == sorted(tid(f"t{i:02d}") for i in range(12))
    assert len(set(every)) == 12
    assert len(queues["b200"]) == 8 and len(queues["rtx"]) == 4
    assert tid("t11") in queues["b200"] and tid("t10") in queues["b200"]  # 16 CPUs, sidecar: not on the small host


def test_plan_puts_tasks_with_finished_sessions_last_fewest_first_and_is_deterministic(tmp_path):
    sets, inventory = task_sets(tmp_path)
    counts = {tid("t00"): 3, tid("t01"): 1, "adaptive-rejection-sampler": 6}
    queues = plan(sets, inventory, counts, [("b200", 1, False)], [PRO], seed=1)
    order = queues["b200"]
    assert order[-2:] == [tid("t01"), tid("t00")]
    assert "adaptive-rejection-sampler" not in order  # Terminal-Bench 2.0 is not queued
    assert plan(sets, inventory, counts, [("b200", 1, False)], [PRO], seed=1) == queues
    # The partition does not depend on the counts, so hosts that see different counts still agree on it.
    two = [("b200", 2, False), ("cas", 1, False)]
    assert {h: set(q) for h, q in plan(sets, inventory, counts, two, [PRO], seed=1).items()} == {h: set(q) for h, q in plan(sets, inventory, {}, two, [PRO], seed=1).items()}


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
    subprocess.run([*cli, "plan", str(sets), str(inventory), str(state), "b200", "--host", "b200:2", "--host", "rtx:1:small", "--counts", str(counts), "--seed", "1", "--dataset", PRO], check=True)
    order = json.loads((state / "queue.json").read_text())["order"]
    assert order[-1] == tid("t00") or tid("t00") not in order
    # A stand-in docker whose root folder is tmp_path (plenty of free space on the test machine).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(f"#!/bin/sh\necho {tmp_path}\n")
    (bin_dir / "docker").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    first = subprocess.run([*cli, "claim", str(state), "s1", str(os.getpid())], capture_output=True, text=True, check=True, env=env).stdout.split()
    assert first == [order[0], "1"]
    subprocess.run([*cli, "release", str(state), "s1", order[0]], check=True)
    # Planning again over an existing queue would forget its claims: refused.
    again = subprocess.run([*cli, "plan", str(sets), str(inventory), str(state), "b200", "--host", "b200:2", "--dataset", PRO], capture_output=True, text=True)
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
    result = subprocess.run(["bash", str(HERE / "hub_launch.sh"), "--dry-run", str(collect), "b200", "collect_launch_b200.sh", "--host", "b200:2", "--host", "rtx:1:small", "--dataset", PRO], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("hub-")]
    c = collect.resolve()
    assert lines == [
        f"hub-b200-gpu0-s1: bash {c}/hub/tools/jeff-first/hub_stream.sh {c} {c}/hub/queue http://192.168.3.12:8885 runs-collect-xhigh-hub/b200-gpu0-s1 qwen3.8-27b-nvfp4@b200-gpu0-vllm0.29 b200-gpu0-s1",
        f"hub-b200-gpu7-s6: bash {c}/hub/tools/jeff-first/hub_stream.sh {c} {c}/hub/queue http://192.168.3.12:8892 runs-collect-xhigh-hub/b200-gpu7-s6 qwen3.8-27b-nvfp4@b200-gpu7-vllm0.29 b200-gpu7-s6",
    ]
    assert "b200: 3 tasks of 4" in result.stdout or "b200: 2 tasks of 4" in result.stdout
    assert "b200: 2 streams; dry run" in result.stdout
    assert not (collect / "hub" / "queue").exists()  # a dry run plans into a temporary folder


def add_dataset(sets: Path, inventory: Path, folder: str, hub: str, tasks: list[str], excluded: dict | None = None) -> None:
    data = json.loads(sets.read_text())
    data["datasets"][folder] = {"hub": {"name": hub, "ref": "sha256:" + "d" * 64, "revision": 1}, "training": tasks, "held_out": [], "excluded": excluded or {}}
    sets.write_text(json.dumps(data))
    rows = json.loads(inventory.read_text())
    org = hub.split("/")[0]
    rows += [{"dataset": folder, "task": t, "task_name": f"{org}/{t}", "agent_timeout_sec": 900.0, "gpus": 0, "mcp_servers": 0, "cpus": 1, "memory_mb": 2048, "compose_services": []} for t in tasks + list((excluded or {}))]
    inventory.write_text(json.dumps(rows))


def test_append_adds_a_new_datasets_tasks_after_the_queue_disjointly_across_hosts_without_touching_claims(tmp_path):
    sets, inventory = task_sets(tmp_path, n=4)
    hosts = [("b200", 2, False), ("rtx", 1, True)]
    states = {}
    for host, order in plan(sets, inventory, {}, hosts, [PRO], seed=1).items():
        states[host] = tmp_path / f"queue-{host}"
        states[host].mkdir()
        (states[host] / "queue.json").write_text(json.dumps({"order": order, "finished_before": {}}))
    me = os.getpid()
    first = claim(states["b200"], "s1", me)
    before = json.loads((states["b200"] / "queue.json").read_text())["order"]
    # Read at append time: the dataset is added (and one task excluded) after the queues were planned.
    add_dataset(sets, inventory, "skillsbench", "benchflow/skillsbench", [f"k{i}" for i in range(6)], {"drone": "passes a key"})
    skills = "benchflow/skillsbench"
    counts = {f"{skills}:k0": 1}
    added = {host: append(sets, inventory, state, host, counts, hosts, [skills], seed=1) for host, state in states.items()}
    assert sorted(added["b200"] + added["rtx"]) == sorted(f"{skills}:k{i}" for i in range(6))
    assert not set(added["b200"]) & set(added["rtx"])
    assert len(added["b200"]) == 4 and len(added["rtx"]) == 2
    queue = json.loads((states["b200"] / "queue.json").read_text())
    assert queue["order"] == before + added["b200"]
    assert json.loads((states["b200"] / "claims.json").read_text())["running"]["s1"]["task"] == first[0]
    with pytest.raises(ValueError, match="already queued"):
        append(sets, inventory, states["b200"], "b200", counts, hosts, [skills], seed=1)
    with pytest.raises(ValueError, match="not pinned hub datasets"):
        append(sets, inventory, states["b200"], "b200", counts, hosts, ["swe-rebench/swe-rebench-leaderboard"], seed=1)


def test_appended_tasks_come_after_unrun_tasks_and_before_repeats(tmp_path):
    state = make_queue(tmp_path, ["a", "b"], {"a": 1})
    sets, inventory = task_sets(tmp_path, n=2)
    add_dataset(sets, inventory, "skillsbench", "benchflow/skillsbench", ["k0"])
    append(sets, inventory, state, "b200", {}, [("b200", 1, False)], ["benchflow/skillsbench"], seed=1)
    me = os.getpid()
    assert [claim(state, f"s{i}", me)[0] for i in range(3)] == ["b", "benchflow/skillsbench:k0", "a"]


def test_no_claims_while_the_disk_is_below_the_floor(tmp_path):
    from collect_queue import LOW_DISK, MIN_FREE_GB

    state = make_queue(tmp_path, ["a"])
    assert MIN_FREE_GB == 150
    assert claim(state, "s1", os.getpid(), free_gb=149.9) is LOW_DISK
    assert not (state / "claims.json").exists()
    assert claim(state, "s1", os.getpid(), free_gb=151) == ("a", 1)


def test_a_session_that_never_started_is_released_without_counting_as_a_run(tmp_path):
    state = make_queue(tmp_path, ["a", "b"])
    me = os.getpid()
    assert claim(state, "s1", me) == ("a", 1)
    release(state, "s1", "a", ran=False)  # e.g. Docker Hub's pull limit was hit before the session
    assert claim(state, "s1", me) == ("a", 1)


def test_old_streams_are_drained_when_the_queue_needs_a_newer_stream(tmp_path):
    from collect_queue import OLD_STREAM, require_stream_version

    state = make_queue(tmp_path, ["a", "b"])
    me = os.getpid()
    assert claim(state, "s1", me) == ("a", 1)
    require_stream_version(state, 2)
    release(state, "s1", "a")
    assert claim(state, "s1", me) is OLD_STREAM  # the old stream's loop then ends
    assert claim(state, "s1", me, stream_version=2) == ("b", 1)


def test_scoring_tasks_are_seeded_held_out_tasks_spread_over_datasets_after_the_queue(tmp_path):
    from collect_queue import append_scoring

    sets, inventory = task_sets(tmp_path, n=2)
    data = json.loads(sets.read_text())
    data["datasets"]["terminal-bench-pro"]["held_out"] = [f"h{i}" for i in range(10)]
    sets.write_text(json.dumps(data))
    rows = json.loads(inventory.read_text())
    rows += [{"dataset": "terminal-bench-pro", "task": f"h{i}", "task_name": f"terminal-bench-pro/h{i}", "agent_timeout_sec": 900.0, "gpus": 0, "mcp_servers": 0, "cpus": 1, "memory_mb": 2048, "compose_services": []} for i in range(10)]
    inventory.write_text(json.dumps(rows))
    add_dataset(sets, inventory, "skillsbench", "benchflow/skillsbench", ["k0"])
    data = json.loads(sets.read_text())
    data["datasets"]["skillsbench"]["held_out"] = ["s0", "s1", "s2"]
    sets.write_text(json.dumps(data))
    rows = json.loads(inventory.read_text())
    rows += [{"dataset": "skillsbench", "task": f"s{i}", "task_name": f"benchflow/s{i}", "agent_timeout_sec": 900.0, "gpus": 0, "mcp_servers": 0, "cpus": 1, "memory_mb": 2048, "compose_services": []} for i in range(3)]
    rows[-1]["gpus"] = 1  # not runnable: never picked
    inventory.write_text(json.dumps(rows))
    state = make_queue(tmp_path, ["a"])
    picked = append_scoring(sets, inventory, state, ["terminal-bench-pro/terminal-bench-pro", "benchflow/skillsbench"], 6, seed=1, skip=[tid("h3")])
    assert len(picked) == 6
    assert sum(t.startswith("benchflow/") for t in picked) == 2  # spread: both runnable SkillsBench tasks, then Pro
    assert tid("h3") not in picked and "benchflow/skillsbench:s2" not in picked
    queue = json.loads((state / "queue.json").read_text())
    assert queue["order"] == ["a", *picked] and queue["scoring"] == picked
    with pytest.raises(ValueError, match="already queued"):
        append_scoring(sets, inventory, state, ["terminal-bench-pro/terminal-bench-pro"], 2, seed=1, skip=[])
