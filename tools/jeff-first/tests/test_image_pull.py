import json
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


def write_settings(lock: Path, max_concurrent: int = 4, per_hour: int = 100) -> Path:
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "settings.json").write_text(json.dumps({"max_concurrent": max_concurrent, "per_hour": per_hour}))
    return lock


def run(tmp_path: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    if len(args) >= 3 and args[0] == "pull" and not (Path(args[2]) / "settings.json").exists():
        write_settings(Path(args[2]))
    return subprocess.run([sys.executable, str(HERE / "image_pull.py"), *args], capture_output=True, text=True, env=env)


def calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def test_a_present_image_is_not_pulled(tmp_path):
    env = fake_docker(tmp_path, present=True)
    assert run(tmp_path, env, "pull", IMAGE, str(tmp_path / "lock"), "0").returncode == 0
    assert calls(tmp_path) == [f"image inspect {IMAGE}"]


def test_pulls_beyond_the_hourly_budget_wait(tmp_path):
    env = fake_docker(tmp_path)
    lock = write_settings(tmp_path / "lock", per_hour=2)
    (lock / "starts.json").write_text(json.dumps([time.time() - 3600 + 2, time.time()]))
    start = time.monotonic()
    assert run(tmp_path, env, "pull", IMAGE, str(lock)).returncode == 0
    assert time.monotonic() - start >= 1.5  # waited until the oldest start left the hour
    assert len(json.loads((lock / "starts.json").read_text())) == 2


def test_settings_are_required(tmp_path):
    env = fake_docker(tmp_path)
    (tmp_path / "lock").mkdir()
    result = subprocess.run([sys.executable, str(HERE / "image_pull.py"), "pull", IMAGE, str(tmp_path / "lock")], capture_output=True, text=True, env=env)
    assert result.returncode != 0 and "settings.json" in result.stderr


def test_concurrent_pulls_are_capped_by_the_slots(tmp_path):
    env = fake_docker(tmp_path)
    slow = Path(env["PATH"].split(":")[0]) / "docker"
    slow.write_text(slow.read_text().replace('"pull "*) echo', '"pull "*) sleep 2; echo'))
    lock = write_settings(tmp_path / "lock", max_concurrent=1)
    start = time.monotonic()
    procs = [subprocess.Popen([sys.executable, str(HERE / "image_pull.py"), "pull", IMAGE, str(lock), "40"], env=env) for _ in range(2)]
    assert all(p.wait() == 0 for p in procs)
    assert time.monotonic() - start >= 3.5  # one slot: one after the other


def test_a_rate_limit_error_exits_75(tmp_path):
    env = fake_docker(tmp_path, pull_output="Error response from daemon: toomanyrequests: You have reached your pull rate limit.", pull_status=1)
    result = run(tmp_path, env, "pull", IMAGE, str(tmp_path / "lock"), "0")
    assert result.returncode == 75 and "rate limit" in result.stderr


def test_only_a_missing_image_counts_as_a_failed_run(tmp_path):
    env = fake_docker(tmp_path, pull_output="Error response from daemon: manifest unknown", pull_status=1)
    result = run(tmp_path, env, "pull", IMAGE, str(tmp_path / "lock"), "0")
    assert result.returncode == 1 and "no such image" in result.stderr
    env = fake_docker(tmp_path, pull_output="net/http: TLS handshake timeout", pull_status=1)
    result = run(tmp_path, env, "pull", IMAGE, str(tmp_path / "lock"), "0")
    assert result.returncode == 75 and "not counted" in result.stderr


def test_a_pull_stopped_by_a_signal_is_not_counted(tmp_path):
    env = fake_docker(tmp_path)
    slow = Path(env["PATH"].split(":")[0]) / "docker"
    slow.write_text(slow.read_text().replace('"pull "*) echo', '"pull "*) sleep 30; echo'))
    lock = write_settings(tmp_path / "lock")
    proc = subprocess.Popen([sys.executable, str(HERE / "image_pull.py"), "pull", IMAGE, str(lock)], env=env)
    time.sleep(1.5)
    proc.terminate()
    assert proc.wait(timeout=10) == 75


def test_remove_deletes_the_image_and_keeps_one_still_in_use(tmp_path):
    env = fake_docker(tmp_path)
    assert run(tmp_path, env, "remove", IMAGE).returncode == 0
    assert calls(tmp_path) == [f"image rm {IMAGE}"]
    env = fake_docker(tmp_path, rm_output=f"Error response from daemon: conflict: unable to remove repository reference {IMAGE} (must force) - container 1a2b is using its referenced image", rm_status=1)
    result = run(tmp_path, env, "remove", IMAGE)
    assert result.returncode == 0 and "in use" in result.stdout
    env = fake_docker(tmp_path, rm_output="Error: something else", rm_status=1)
    assert run(tmp_path, env, "remove", IMAGE).returncode == 1


def test_two_slots_let_slow_pulls_run_side_by_side(tmp_path):
    env = fake_docker(tmp_path)
    slow = Path(env["PATH"].split(":")[0]) / "docker"
    slow.write_text(slow.read_text().replace('"pull "*) echo', '"pull "*) sleep 3; echo'))
    lock = write_settings(tmp_path / "lock", max_concurrent=2)
    start = time.monotonic()
    procs = [subprocess.Popen([sys.executable, str(HERE / "image_pull.py"), "pull", IMAGE, str(lock)], env=env) for _ in range(2)]
    assert all(p.wait() == 0 for p in procs)
    assert time.monotonic() - start < 5.5
