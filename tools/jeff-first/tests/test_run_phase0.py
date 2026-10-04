"""run_phase0.sh --dry-run: Terminal-Bench 2.0 commands stay byte-for-byte as before; hub tasks get their pinned
dataset, package name, capped agent time and the full task id in the trace."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.test_task_source import PRO_REF, write_task_sets

HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "run_phase0.sh"
# The run script as committed before hub datasets were added (f4c8e8686 forwards the thinking router; unchanged since).
BEFORE = "f4c8e8686"


def run(tmp_path: Path, script: Path, tasks: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    tarball = tmp_path / "jeff-pi-scout-8db5381f3.tgz"
    tarball.write_text("x")
    tasks_json = tmp_path / "tasks.json"
    tasks_json.write_text(json.dumps({"phase0": ["adaptive-rejection-sampler"]}))
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    full_env = {
        **os.environ,
        "JEFF_RUN_API_KEY": "unused",
        "JEFF_RUN_THINKING_FORMAT": "qwen-chat-template",
        "JEFF_RUN_MAX_OUTPUT_TOKENS": "32768",
        "JEFF_FIRST_RUN_APPROVAL": "all",
        "JEFF_FIRST_DRIVER_BUILD": "qwen3.8-27b-nvfp4@b200-gpu0-vllm0.29",
        "JEFF_FIRST_THINKING_ROUTER": "fixed:xhigh",
        **(env or {}),
    }
    args = [str(tasks_json), str(tarball), "http://192.168.3.12:8885", str(jobs), "1", "high", "bash", "qwen3.8-27b", "record", "6"]
    return subprocess.run(["bash", str(script), "--dry-run", *args, *tasks], capture_output=True, text=True, env=full_env)


def without_time(text: str) -> str:
    return re.sub(r"-\d{8}-\d{6}", "-<time>", text)


def test_terminal_bench_2_commands_are_unchanged(tmp_path):
    before = tmp_path / "run_phase0_before.sh"
    before.write_text(subprocess.run(["git", "show", f"{BEFORE}:tools/jeff-first/run_phase0.sh"], cwd=HERE, capture_output=True, text=True, check=True).stdout)
    tasks = ["adaptive-rejection-sampler", "dna-assembly"]
    old, new = run(tmp_path, before, tasks), run(tmp_path, SCRIPT, tasks)
    assert old.returncode == 0 and new.returncode == 0, new.stderr
    # The old copy lives in tmp_path, so its uv project folder differs; compare with that folder normalised.
    normalise = lambda text: re.sub(r"--project \S+", "--project <here>", without_time(text))
    assert sorted(normalise(new.stdout).splitlines()) == sorted(normalise(old.stdout).splitlines())
    assert "--dataset terminal-bench@2.0 -i adaptive-rejection-sampler" in new.stdout


def test_a_hub_task_runs_from_its_pinned_dataset_with_the_capped_agent_time(tmp_path):
    sets, inventory = write_task_sets(tmp_path)
    env = {"JEFF_RUN_TASK_SETS": str(sets), "JEFF_RUN_TASK_INVENTORY": str(inventory)}
    result = run(tmp_path, SCRIPT, ["terminal-bench-pro/terminal-bench-pro:slow-b", "adaptive-rejection-sampler"], env)
    assert result.returncode == 0, result.stderr
    hub = [line for line in result.stdout.splitlines() if "slow-b" in line]
    assert len(hub) == 1
    line = hub[0]
    assert f"--dataset terminal-bench-pro/terminal-bench-pro@{PRO_REF} -i terminal-bench-pro/slow-b" in line
    assert "--agent-timeout-multiplier 0.1875" in line  # 8 hours x 0.1875 = 90 minutes
    assert "JEFF_FIRST_TASK_ID=terminal-bench-pro/terminal-bench-pro:slow-b" in line
    assert re.search(r"--job-name terminal-bench-pro\.terminal-bench-pro\.slow-b-\d{8}-\d{6}", line)
    assert "terminal-bench@2.0" not in line
    tb2 = [line for line in result.stdout.splitlines() if "adaptive-rejection-sampler" in line]
    assert "--agent-timeout-multiplier 6 " in tb2[0]


def test_a_hub_task_needs_the_task_sets(tmp_path):
    result = run(tmp_path, SCRIPT, ["terminal-bench-pro/terminal-bench-pro:slow-b"])
    assert result.returncode == 2 and "JEFF_RUN_TASK_SETS" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ("terminal-bench-pro/terminal-bench-pro:no-such", "no task no-such"),
        ("nobody/nothing:x", "unknown dataset"),
        ("terminal-bench-pro/terminal-bench-pro:leak-d", "excluded"),
        ("terminal-bench-pro:x", "neither"),
    ],
)
def test_unknown_or_excluded_tasks_stop_the_run_before_anything_starts(tmp_path, task, message):
    sets, inventory = write_task_sets(tmp_path)
    env = {"JEFF_RUN_TASK_SETS": str(sets), "JEFF_RUN_TASK_INVENTORY": str(inventory)}
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler", task], env)
    assert result.returncode != 0 and message in result.stderr
    assert result.stdout == ""
