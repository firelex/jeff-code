"""Inputs of the stage 1 replay (tools/jeff-first/imitation/stage1_replay.py), which runs on a Docker host without
pyarrow: the nl2bash sessions of the stage 1 conversion, their task folders, and the two image build folders.

- sessions.jsonl: one line per nl2bash session that the stage 1 conversion kept (stage1-sessions.jsonl: the leak check
  passed), with its conversation (the latest episode's row of ukisai.parquet).
- tasks.jsonl: each needed task folder of DCAgent/nl2bash-verified (train-00000-of-00001.parquet: path, task_binary);
  every one must have the shared Dockerfile.
- images/<harness>/: the Dockerfile of each harness's image (stage1_replay.image_dockerfile from the tasks' shared
  Dockerfile) and, for the asciinema harness, Harbor's get-asciinema-timestamp.sh (from the harbor package).

Usage (from tools/jeff-first):
  uv run --with pyarrow==21.0.0 python ../../results/imitation/scripts/prepare_stage1_replay.py \\
      DATA_FOLDER NL2BASH_VERIFIED_PARQUET OUT_FOLDER
DATA_FOLDER holds ukisai.parquet and stage1-sessions.jsonl.
"""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "jeff-first"))

import pyarrow.parquet as pq
from harbor.agents.terminus_2.tmux_session import TmuxSession

from imitation.stage1_replay import ASCIINEMA, ASCIINEMA_SCRIPT, TASK_DOCKERFILE_SHA256, TMUX_SOCKET, image_dockerfile, session_line, task_folder, task_line
from imitation.terminus import latest_episodes


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare_stage1_replay.py DATA_FOLDER NL2BASH_VERIFIED_PARQUET OUT_FOLDER")
    data, verified, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=False)
    wanted = {}
    for line in (data / "stage1-sessions.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["task"].startswith("nl2bash-"):
            wanted[record["session"]] = record["task"]
    rows = [row for row in latest_episodes(pq.read_table(data / "ukisai.parquet").to_pylist()) if row["trial_name"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError(f"{len(wanted) - len(rows)} sessions of stage1-sessions.jsonl are not in ukisai.parquet")
    with open(out / "sessions.jsonl", "w") as handle:
        for row in rows:
            if row["task"] != wanted[row["trial_name"]]:
                raise ValueError(f"session {row['trial_name']}: task {row['task']} here, {wanted[row['trial_name']]} in the conversion")
            handle.write(session_line(row) + "\n")
    needed = {task_folder(task) for task in wanted.values()}
    archives = {row["path"]: row["task_binary"] for row in pq.read_table(verified).to_pylist() if row["path"] in needed}
    missing = sorted(needed - set(archives))
    if missing:
        raise ValueError(f"{len(missing)} task folders are not in {verified}, e.g. {missing[:5]}")
    with open(out / "tasks.jsonl", "w") as handle:
        for task in sorted(archives):
            with tarfile.open(fileobj=io.BytesIO(archives[task])) as archive:
                dockerfile = archive.extractfile("environment/Dockerfile").read().decode()
            if hashlib.sha256(dockerfile.encode()).hexdigest() != TASK_DOCKERFILE_SHA256:
                raise ValueError(f"task {task}: its Dockerfile is not the shared nl2bash one, so the shared image does not fit it")
            handle.write(task_line(task, archives[task]) + "\n")
    for harness in (TMUX_SOCKET, ASCIINEMA):
        folder = out / "images" / harness.name
        folder.mkdir(parents=True)
        (folder / "Dockerfile").write_text(image_dockerfile(dockerfile, harness))
        if harness is ASCIINEMA:
            (folder / ASCIINEMA_SCRIPT).write_bytes(TmuxSession._GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH.read_bytes())
    print(f"{len(rows)} sessions, {len(archives)} task folders, images for {TMUX_SOCKET.name} and {ASCIINEMA.name} in {out}")


if __name__ == "__main__":
    main()
