"""Docker Hub images for rotated datasets (task_source.ROTATED_DATASETS, e.g. SWE-rebench): pulled one at a time per
host before a session, spaced so a host stays under Docker Hub's pull limits, and removed after it.

    python3 image_pull.py pull IMAGE LOCK_DIR MIN_INTERVAL_SEC
        An image already present is not pulled (a pull of a present image still counts against Docker Hub's limit).
        Pulls on one host start at least MIN_INTERVAL_SEC apart (file lock in LOCK_DIR) and then run side by side.
        Exit 75 on Docker Hub's rate-limit error (the caller releases its claim without counting a run and waits);
        exit 1 on any other pull error.
    python3 image_pull.py remove IMAGE
        Removes the image; one still used by a container is kept (said on stdout); any other error fails.
"""

import fcntl
import subprocess
import sys
import time
from pathlib import Path

RATE_LIMITED = 75
RATE_LIMIT_MARKERS = ("toomanyrequests", "pull rate limit", "rate limit exceeded")


def pull(image: str, lock_dir: Path, min_interval: float) -> int:
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        print(f"{image} is present")
        return 0
    lock_dir.mkdir(parents=True, exist_ok=True)
    # The lock only spaces the starts of pulls; the pulls themselves run side by side (a 1-2 GB pull takes longer
    # than MIN_INTERVAL_SEC, and holding the lock through it made pulls, not the interval, the limit).
    with open(lock_dir / "pull.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        last_path = lock_dir / "last-pull"
        last = float(last_path.read_text()) if last_path.exists() else 0.0
        wait = last + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        last_path.write_text(str(time.time()))
    result = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        print(f"pulled {image}")
        return 0
    if any(marker in output.lower() for marker in RATE_LIMIT_MARKERS):
        print(f"Docker Hub rate limit while pulling {image}: {output[-500:]}", file=sys.stderr)
        return RATE_LIMITED
    print(f"docker pull {image} failed (exit {result.returncode}): {output[-1000:]}", file=sys.stderr)
    return 1


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


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 4 and args[0] == "pull":
        sys.exit(pull(args[1], Path(args[2]), float(args[3])))
    if len(args) == 2 and args[0] == "remove":
        sys.exit(remove(args[1]))
    raise SystemExit(__doc__)
