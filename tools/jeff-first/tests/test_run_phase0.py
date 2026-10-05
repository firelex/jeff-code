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


def run(tmp_path: Path, script: Path, tasks: list[str], env: dict | None = None, mode: str = "record") -> subprocess.CompletedProcess:
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
        "JEFF_FIRST_OUTPUT_TRIM": "off",
        "JEFF_FIRST_THINKING_LIMIT": "off",
        **(env or {}),
    }
    args = [str(tasks_json), str(tarball), "http://192.168.3.12:8885", str(jobs), "1", "high", "bash", "qwen3.8-27b", mode, "6"]
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
    # The output trimming and thinking limit settings (added later) are the only new arguments; the Harbor agent was
    # renamed with the project (JeffPi in jeff_pi.py, now JeffCode in jeff_code.py).
    new_lines = [line.replace("--ae JEFF_FIRST_OUTPUT_TRIM=off --ae JEFF_FIRST_THINKING_LIMIT=off ", "")
                 .replace("-a harbor_agent.jeff_code:JeffCode ", "-a harbor_agent.jeff_pi:JeffPi ")
                 for line in normalise(new.stdout).splitlines()]
    assert all("JEFF_FIRST_OUTPUT_TRIM=off" in line for line in new.stdout.splitlines())
    assert sorted(new_lines) == sorted(normalise(old.stdout).splitlines())
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


def test_an_offline_task_runs_in_the_egress_controlled_environment_with_only_the_model_host_allowed(tmp_path):
    from tests.test_task_source import add_swe_rebench

    sets, inventory = write_task_sets(tmp_path)
    task_id = add_swe_rebench(sets, inventory)
    env = {"JEFF_RUN_TASK_SETS": str(sets), "JEFF_RUN_TASK_INVENTORY": str(inventory)}
    result = run(tmp_path, SCRIPT, [task_id, "terminal-bench-pro/terminal-bench-pro:fix-a"], env)
    assert result.returncode == 0, result.stderr
    offline = next(line for line in result.stdout.splitlines() if "pelita" in line)
    assert "--env harbor_agent.offline:EgressDocker" in offline and "--ak allowed_hosts=192.168.3.12" in offline
    assert "--agent-timeout-multiplier 1.8 " in offline  # 50 minutes x 1.8 = 90 minutes
    online = next(line for line in result.stdout.splitlines() if "fix-a" in line)
    assert "EgressDocker" not in online and "allowed_hosts" not in online


JEFF_ENV = {
    "JEFF_FIRST_JEFF_URL": "http://192.168.2.10:8920",
    "JEFF_FIRST_JEFF_STEP_ADAPTER": "jeff-step",
    "JEFF_FIRST_JEFF_STEP_THRESHOLD": "0.55",
    "JEFF_FIRST_THINKING_ROUTER": "jeff:jeff-router",
    "JEFF_FIRST_JEFF_ROUTER_THRESHOLD": "0.4",
}


def test_jeff_mode_forwards_the_service_the_adapters_and_the_thresholds(tmp_path):
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], JEFF_ENV, mode="jeff")
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip() + " "
    for setting in [
        "JEFF_FIRST_MODE=jeff",
        "JEFF_FIRST_JEFF_URL=http://192.168.2.10:8920",
        "JEFF_FIRST_JEFF_STEP_ADAPTER=jeff-step",
        "JEFF_FIRST_JEFF_STEP_THRESHOLD=0.55",
        "JEFF_FIRST_THINKING_ROUTER=jeff:jeff-router",
        "JEFF_FIRST_JEFF_ROUTER_THRESHOLD=0.4",
        "JEFF_FIRST_RUN_APPROVAL=all",
        "JEFF_FIRST_DRIVER_BUILD=qwen3.8-27b-nvfp4@b200-gpu0-vllm0.29",
    ]:
        assert f"--ae {setting} " in line, setting
    assert "TEACHER" not in line


def test_jeff_mode_with_a_fixed_router_needs_no_router_threshold(tmp_path):
    env = {**JEFF_ENV, "JEFF_FIRST_THINKING_ROUTER": "fixed:xhigh", "JEFF_FIRST_JEFF_ROUTER_THRESHOLD": ""}
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], env, mode="jeff")
    assert result.returncode == 0, result.stderr
    assert "--ae JEFF_FIRST_THINKING_ROUTER=fixed:xhigh " in result.stdout
    assert "JEFF_FIRST_JEFF_ROUTER_THRESHOLD" not in result.stdout


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("JEFF_FIRST_JEFF_URL", "JEFF_FIRST_JEFF_URL"),
        ("JEFF_FIRST_JEFF_STEP_ADAPTER", "JEFF_FIRST_JEFF_STEP_ADAPTER"),
        ("JEFF_FIRST_JEFF_STEP_THRESHOLD", "JEFF_FIRST_JEFF_STEP_THRESHOLD"),
        ("JEFF_FIRST_JEFF_ROUTER_THRESHOLD", "JEFF_FIRST_JEFF_ROUTER_THRESHOLD"),
    ],
)
def test_jeff_mode_stops_without_a_required_setting(tmp_path, missing, message):
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {**JEFF_ENV, missing: ""}, mode="jeff")
    assert result.returncode == 2 and message in result.stderr
    assert result.stdout == ""


def test_jeff_mode_with_the_flipped_router_forwards_it_without_a_separate_threshold(tmp_path):
    env = {**JEFF_ENV, "JEFF_FIRST_THINKING_ROUTER": "jeff-off-unless:jeff-router:0.6", "JEFF_FIRST_JEFF_ROUTER_THRESHOLD": ""}
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], env, mode="jeff")
    assert result.returncode == 0, result.stderr
    assert "--ae JEFF_FIRST_THINKING_ROUTER=jeff-off-unless:jeff-router:0.6 " in result.stdout
    assert result.stdout.count("JEFF_FIRST_JEFF_URL=") == 1
    assert "JEFF_FIRST_JEFF_ROUTER_THRESHOLD" not in result.stdout


@pytest.mark.parametrize(
    ("router", "threshold", "message"),
    [
        ("jeff-off-unless:jeff-router:1.5", "", "must be a number from 0 to 1"),
        ("jeff-off-unless:jeff-router:", "", "must be a number from 0 to 1"),
        ("jeff-off-unless:jeff-router", "", "needs JEFF_FIRST_THINKING_ROUTER"),
        ("jeff-off-unless:jeff-router:0.6", "0.4", "holds its own threshold"),
    ],
)
def test_the_flipped_router_is_checked(tmp_path, router, threshold, message):
    env = {**JEFF_ENV, "JEFF_FIRST_THINKING_ROUTER": router, "JEFF_FIRST_JEFF_ROUTER_THRESHOLD": threshold}
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], env, mode="jeff")
    assert result.returncode == 2 and message in result.stderr, result.stderr


def test_the_flipped_router_in_record_mode_passes_the_service(tmp_path):
    env = {"JEFF_FIRST_THINKING_ROUTER": "jeff-off-unless:r:0.7", "JEFF_FIRST_JEFF_URL": "http://192.168.2.10:8920"}
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], env)
    assert result.returncode == 0, result.stderr
    assert "--ae JEFF_FIRST_JEFF_URL=http://192.168.2.10:8920 " in result.stdout


def test_the_jeff_router_in_record_mode_needs_the_service(tmp_path):
    env = {"JEFF_FIRST_THINKING_ROUTER": "jeff:jeff-router", "JEFF_FIRST_JEFF_ROUTER_THRESHOLD": "0.4"}
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], env)
    assert result.returncode == 2 and "JEFF_FIRST_JEFF_URL" in result.stderr
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {**env, "JEFF_FIRST_JEFF_URL": "http://192.168.2.10:8920"})
    assert result.returncode == 0, result.stderr
    assert "--ae JEFF_FIRST_JEFF_URL=http://192.168.2.10:8920 " in result.stdout


def test_an_offline_task_in_jeff_mode_also_allows_the_jeff_service_host(tmp_path):
    from tests.test_task_source import add_swe_rebench

    sets, inventory = write_task_sets(tmp_path)
    task_id = add_swe_rebench(sets, inventory)
    env = {"JEFF_RUN_TASK_SETS": str(sets), "JEFF_RUN_TASK_INVENTORY": str(inventory), **JEFF_ENV}
    result = run(tmp_path, SCRIPT, [task_id], env, mode="jeff")
    assert result.returncode == 0, result.stderr
    # The dry run prints each argument shell-quoted (printf %q), which escapes the comma.
    assert "--ak allowed_hosts=192.168.3.12\\,192.168.2.10 " in result.stdout


def test_the_output_trimming_setting_is_required_and_forwarded(tmp_path):
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {"JEFF_FIRST_OUTPUT_TRIM": ""})
    assert result.returncode == 2 and "JEFF_FIRST_OUTPUT_TRIM" in result.stderr
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {"JEFF_FIRST_OUTPUT_TRIM": "fixed:200"})
    assert result.returncode == 2 and "fixed:first20last20" in result.stderr
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {"JEFF_FIRST_OUTPUT_TRIM": "fixed:last40"})
    assert result.returncode == 0, result.stderr
    assert "--ae JEFF_FIRST_OUTPUT_TRIM=fixed:last40 " in result.stdout
    assert "JEFF_FIRST_JEFF_TRIM_THRESHOLD" not in result.stdout


def test_the_thinking_limit_is_required_checked_and_forwarded(tmp_path):
    for value in ["", "8k", "0", "-5", "OFF"]:
        result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {"JEFF_FIRST_THINKING_LIMIT": value})
        assert result.returncode == 2 and "JEFF_FIRST_THINKING_LIMIT: off or a whole number" in result.stderr, value
    for value in ["off", "8000"]:
        result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {"JEFF_FIRST_THINKING_LIMIT": value})
        assert result.returncode == 0, result.stderr
        assert f"--ae JEFF_FIRST_THINKING_LIMIT={value} " in result.stdout


def test_jeff_trimming_needs_the_service_and_its_threshold(tmp_path):
    env = {"JEFF_FIRST_OUTPUT_TRIM": "jeff:jeff-trim", "JEFF_FIRST_JEFF_TRIM_THRESHOLD": "0.6"}
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], env)
    assert result.returncode == 2 and "JEFF_FIRST_JEFF_URL" in result.stderr
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {**env, "JEFF_FIRST_JEFF_URL": "http://192.168.2.10:8920",
                                                                    "JEFF_FIRST_JEFF_TRIM_THRESHOLD": ""})
    assert result.returncode == 2 and "JEFF_FIRST_JEFF_TRIM_THRESHOLD" in result.stderr
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {**env, "JEFF_FIRST_JEFF_URL": "http://192.168.2.10:8920"})
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip() + " "
    for setting in ["JEFF_FIRST_OUTPUT_TRIM=jeff:jeff-trim", "JEFF_FIRST_JEFF_URL=http://192.168.2.10:8920",
                    "JEFF_FIRST_JEFF_TRIM_THRESHOLD=0.6"]:
        assert f"--ae {setting} " in line, setting
    # In jeff mode with a Jeff router the service address is passed once.
    result = run(tmp_path, SCRIPT, ["adaptive-rejection-sampler"], {**JEFF_ENV, **env}, mode="jeff")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("JEFF_FIRST_JEFF_URL=") == 1
    assert "--ae JEFF_FIRST_JEFF_TRIM_THRESHOLD=0.6 " in result.stdout
