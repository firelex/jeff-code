import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
IMAGE = "swerebench/sweb.eval.x86_64.aspp_1776_pelita-863:latest"


def fake_docker(tmp_path: Path, *, present: bool = False, pull_output: str = "Status: Downloaded", pull_status: int = 0, rm_output: str = "", rm_status: int = 0) -> dict:
    """A stand-in docker that logs its calls to calls.txt."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = f"""#!/bin/sh
echo "$@" >> {tmp_path}/calls.txt
case "$1 $2" in
  "image inspect") exit {0 if present else 1} ;;
  "image rm") echo "{rm_output}" >&2; exit {rm_status} ;;
  "pull "*) echo "{pull_output}" >&2; exit {pull_status} ;;
esac
exit 9
"""
    (bin_dir / "docker").write_text(script)
    (bin_dir / "docker").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def run(tmp_path: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HERE / "image_pull.py"), *args], capture_output=True, text=True, env=env)


def calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def test_a_present_image_is_not_pulled(tmp_path):
    env = fake_docker(tmp_path, present=True)
    assert run(tmp_path, env, "pull", IMAGE, str(tmp_path / "lock"), "0").returncode == 0
    assert calls(tmp_path) == [f"image inspect {IMAGE}"]


def test_pulls_are_spaced_by_the_minimum_interval(tmp_path):
    env = fake_docker(tmp_path)
    lock = tmp_path / "lock"
    assert run(tmp_path, env, "pull", IMAGE, str(lock), "2").returncode == 0
    start = time.monotonic()
    assert run(tmp_path, env, "pull", IMAGE, str(lock), "2").returncode == 0
    assert time.monotonic() - start >= 1.5
    assert [c for c in calls(tmp_path) if c.startswith("pull")] == [f"pull {IMAGE}", f"pull {IMAGE}"]


def test_a_rate_limit_error_exits_75(tmp_path):
    env = fake_docker(tmp_path, pull_output="Error response from daemon: toomanyrequests: You have reached your pull rate limit.", pull_status=1)
    result = run(tmp_path, env, "pull", IMAGE, str(tmp_path / "lock"), "0")
    assert result.returncode == 75 and "rate limit" in result.stderr


def test_another_pull_error_fails(tmp_path):
    env = fake_docker(tmp_path, pull_output="manifest unknown", pull_status=1)
    result = run(tmp_path, env, "pull", IMAGE, str(tmp_path / "lock"), "0")
    assert result.returncode == 1 and "manifest unknown" in result.stderr


def test_remove_deletes_the_image_and_keeps_one_still_in_use(tmp_path):
    env = fake_docker(tmp_path)
    assert run(tmp_path, env, "remove", IMAGE).returncode == 0
    assert calls(tmp_path) == [f"image rm {IMAGE}"]
    env = fake_docker(tmp_path, rm_output=f"Error response from daemon: conflict: unable to remove repository reference {IMAGE} (must force) - container 1a2b is using its referenced image", rm_status=1)
    result = run(tmp_path, env, "remove", IMAGE)
    assert result.returncode == 0 and "in use" in result.stdout
    env = fake_docker(tmp_path, rm_output="Error: something else", rm_status=1)
    assert run(tmp_path, env, "remove", IMAGE).returncode == 1


def test_a_slow_pull_does_not_hold_back_the_next_one(tmp_path):
    env = fake_docker(tmp_path)
    slow = Path(env["PATH"].split(":")[0]) / "docker"
    slow.write_text(slow.read_text().replace('"pull "*) echo', '"pull "*) sleep 3; echo'))
    lock = tmp_path / "lock"
    start = time.monotonic()
    first = subprocess.Popen([sys.executable, str(HERE / "image_pull.py"), "pull", IMAGE, str(lock), "0.5"], env=env)
    second = subprocess.Popen([sys.executable, str(HERE / "image_pull.py"), "pull", IMAGE, str(lock), "0.5"], env=env)
    assert first.wait() == 0 and second.wait() == 0
    assert time.monotonic() - start < 5.5  # side by side: about 3.5 s, not 6
