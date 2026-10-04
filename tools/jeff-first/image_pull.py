"""Docker Hub images for rotated datasets (task_source.ROTATED_DATASETS, e.g. SWE-rebench): pulled before a session
(ahead of time by the prefetcher, or by the stream itself), within the host's share of Docker Hub's pull limit, and
removed after the session.

Per-host settings, required, in LOCK_DIR/settings.json: {"max_concurrent": 6, "per_hour": 120}. At most
max_concurrent pulls run at once (one file lock per slot) and at most per_hour pulls start in any 60 minutes (start
times kept in LOCK_DIR/starts.json under a lock). The hosts sharing one Docker Hub account must keep the sum of
their per_hour values under the account's limit (read it with the rate-limit headers, see the report).

    python3 image_pull.py pull IMAGE LOCK_DIR [IGNORED]
        An image already present is not pulled (a pull of a present image still counts against Docker Hub's limit).
        Exit 1 only when the registry says the image does not exist (the caller counts the claim as a run). Exit 75
        on everything else that stops a pull: Docker Hub's rate limit, network errors, or a signal (SIGTERM, SIGINT,
        SIGHUP: the process ends with 75), so the caller releases the claim without counting a run and waits. The third argument (the old fixed spacing, still passed by hub_stream.sh) is
        not used: the spacing comes from settings.json.
    python3 image_pull.py remove IMAGE
        Removes the image; one still used by a container is kept (said on stdout); any other error fails.
    python3 image_pull.py prefetch HUB_DIR QUEUE_DIR LOCK_DIR AHEAD MIN_FREE_GB
        Runs until stopped: every 30 s, the images of the next AHEAD tasks streams will claim (queue order, least-run
        first, not running) that are rotated are pulled if missing, while Docker's disk keeps MIN_FREE_GB free.
        A rate-limit or other passing error pauses prefetching for 15 minutes; an image the registry says does not exist
        is logged (MISSING IMAGE) and skipped.
"""

import fcntl
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from collect_queue import upcoming
from task_source import load_task_sets

RATE_LIMITED = 75  # also: any failure that says nothing about the task (stopped by a signal, network trouble)
RATE_LIMIT_MARKERS = ("toomanyrequests", "pull rate limit", "rate limit exceeded")
# Definite answers from the registry that the image cannot be had: only these count the claim as a run.
MISSING_MARKERS = ("manifest unknown", "not found", "pull access denied", "repository does not exist")
WINDOW_SEC = 3600


def settings(lock_dir: Path) -> dict:
    path = lock_dir / "settings.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: write {{\"max_concurrent\": N, \"per_hour\": M}} for this host")
    data = json.loads(path.read_text())
    if not (isinstance(data.get("max_concurrent"), int) and data["max_concurrent"] > 0 and isinstance(data.get("per_hour"), int) and data["per_hour"] > 0):
        raise SystemExit(f"{path}: max_concurrent and per_hour must be positive whole numbers, got {data}")
    return data


def present(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def _take_slot(lock_dir: Path, slots: int):
    """Holds one of `slots` slot locks (returned open file); waits while all are taken."""
    while True:
        for k in range(slots):
            handle = open(lock_dir / f"slot-{k}.lock", "a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                handle.close()
        time.sleep(2)


def _wait_for_budget(lock_dir: Path, per_hour: int) -> None:
    """Records a pull start once fewer than `per_hour` starts lie in the last hour."""
    while True:
        with open(lock_dir / "rate.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = lock_dir / "starts.json"
            now = time.time()
            starts = [t for t in (json.loads(path.read_text()) if path.exists() else []) if t > now - WINDOW_SEC]
            if len(starts) < per_hour:
                path.write_text(json.dumps(starts + [now]))
                return
            wait = starts[0] + WINDOW_SEC - now
        time.sleep(min(max(wait, 1), 60))


def pull(image: str, lock_dir: Path) -> int:
    if present(image):
        print(f"{image} is present")
        return 0
    config = settings(lock_dir)
    slot = _take_slot(lock_dir, config["max_concurrent"])
    try:
        if present(image):  # pulled by someone else (e.g. the prefetcher) while waiting for a slot
            print(f"{image} is present")
            return 0
        _wait_for_budget(lock_dir, config["per_hour"])
        result = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
    finally:
        slot.close()
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        print(f"pulled {image}")
        return 0
    if any(marker in output.lower() for marker in RATE_LIMIT_MARKERS):
        print(f"Docker Hub rate limit while pulling {image}: {output[-500:]}", file=sys.stderr)
        return RATE_LIMITED
    if any(marker in output.lower() for marker in MISSING_MARKERS):
        print(f"docker pull {image}: the registry has no such image (exit {result.returncode}): {output[-1000:]}", file=sys.stderr)
        return 1
    print(f"docker pull {image} failed without saying the image is missing (exit {result.returncode}); not counted as a run: {output[-1000:]}", file=sys.stderr)
    return RATE_LIMITED


def remove(image: str) -> int:
    result = subprocess.run(["docker", "image", "rm", image], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        print(f"removed {image}")
        return 0
    if "is using its referenced image" in output or "image is being used" in output:
        print(f"kept {image}: in use by a container ({output[-300:]})")
        return 0
    print(f"docker image rm {image} failed (exit {result.returncode}): {output[-1000:]}", file=sys.stderr)
    return 1


def prefetch(hub: Path, queue_dir: Path, lock_dir: Path, ahead: int, min_free_gb: float) -> None:
    sets = load_task_sets(hub / "task-sets.json", hub / "task-sets-inventory.json")
    root = subprocess.run(["docker", "info", "-f", "{{.DockerRootDir}}"], capture_output=True, text=True, check=True).stdout.strip()
    in_flight: dict[str, threading.Thread] = {}
    results: dict[str, int] = {}
    missing: set[str] = set()
    paused_until = 0.0
    while True:
        for image, code in list(results.items()):
            del results[image]
            if code == RATE_LIMITED:
                print(f"{time.strftime('%FT%T')} rate limit: prefetching paused for 15 minutes", flush=True)
                paused_until = time.time() + 900
            elif code != 0:
                # The registry says the image does not exist: the stream that claims the task gets the same answer
                # and counts the run; prefetching goes on with the other tasks.
                print(f"{time.strftime('%FT%T')} MISSING IMAGE {image}: not prefetched again", flush=True)
                missing.add(image)
        if time.time() >= paused_until:
            images = []
            for task in upcoming(queue_dir):
                image = sets.resolve(task).image if ":" in task else None
                if image:
                    images.append(image)
                if len(images) >= ahead:
                    break
            for image in images:
                if image in in_flight or image in missing or present(image):
                    continue
                free_gb = shutil.disk_usage(root).free / 1e9
                if free_gb < min_free_gb:
                    print(f"{time.strftime('%FT%T')} {root} has {free_gb:.0f} GB free (< {min_free_gb}): not prefetching", flush=True)
                    break

                def run(image: str = image) -> None:
                    results[image] = pull(image, lock_dir)

                print(f"{time.strftime('%FT%T')} prefetching {image}", flush=True)
                in_flight[image] = threading.Thread(target=run, daemon=True)
                in_flight[image].start()
        for image in [i for i, t in in_flight.items() if not t.is_alive()]:
            del in_flight[image]
        time.sleep(30)


def _stopped(signum, frame) -> None:
    print(f"pull stopped by signal {signum}; not counted as a run", file=sys.stderr)
    sys.exit(RATE_LIMITED)


if __name__ == "__main__":
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _stopped)
    args = sys.argv[1:]
    if len(args) in (3, 4) and args[0] == "pull":
        sys.exit(pull(args[1], Path(args[2])))
    if len(args) == 2 and args[0] == "remove":
        sys.exit(remove(args[1]))
    if len(args) == 6 and args[0] == "prefetch":
        prefetch(Path(args[1]), Path(args[2]), Path(args[3]), int(args[4]), float(args[5]))
    raise SystemExit(__doc__)
