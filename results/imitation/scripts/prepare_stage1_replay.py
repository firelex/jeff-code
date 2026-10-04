"""Inputs of the stage 1 replay (tools/jeff-first/imitation/stage1_replay.py), which runs on a Docker host without
pyarrow: the stage 1 sessions, their task folders, and the image build folders.

- sessions.jsonl: one line per session of the stage 1 conversion (stage1-sessions.jsonl: the leak check passed), nl2bash
  and InferredBugs, plus the InferredBugs sessions fix wave 4 dropped for an "empty /app"
  (stage1-dropped-empty-app.json): /app is empty by design in InferredBugs tasks, so they are replayed too. Each line
  holds the conversation (the latest episode's row of ukisai.parquet).
- tasks.jsonl: each needed task folder, from DCAgent/nl2bash-verified and mlfoundations-dev/inferredbugs-sandboxes
  (parquet files with path and task_binary); every one must have its family's shared Dockerfile.
- images/<family>-<harness>/: the Dockerfile of each image (stage1_replay.image_dockerfile) and, for the asciinema
  harness, Harbor's get-asciinema-timestamp.sh (from the harbor package).

Usage (from tools/jeff-first):
  uv run --with pyarrow==21.0.0 python ../../results/imitation/scripts/prepare_stage1_replay.py \\
      DATA_FOLDER NL2BASH_VERIFIED_PARQUET INFERREDBUGS_PARQUET OUT_FOLDER
DATA_FOLDER holds ukisai.parquet, stage1-sessions.jsonl and stage1-dropped-empty-app.json.
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

from imitation.stage1_replay import (
    ASCIINEMA,
    ASCIINEMA_SCRIPT,
    INFERREDBUGS,
    NL2BASH,
    TMUX_SOCKET,
    family_of,
    image_dockerfile,
    session_line,
    task_folder,
    task_line,
)
from imitation.terminus import latest_episodes


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: prepare_stage1_replay.py DATA_FOLDER NL2BASH_VERIFIED_PARQUET INFERREDBUGS_PARQUET OUT_FOLDER")
    data, out = Path(sys.argv[1]), Path(sys.argv[4])
    parquets = {NL2BASH.name: Path(sys.argv[2]), INFERREDBUGS.name: Path(sys.argv[3])}
    out.mkdir(parents=True, exist_ok=False)
    wanted = {}
    for line in (data / "stage1-sessions.jsonl").read_text().splitlines():
        record = json.loads(line)
        wanted[record["session"]] = record["task"]
    dropped = json.loads((data / "stage1-dropped-empty-app.json").read_text())["sessions"]
    for session, record in dropped.items():
        if session in wanted or not record["task"].startswith(INFERREDBUGS.prefix):
            raise ValueError(f"session {session}: a dropped session must be an InferredBugs one not in the conversion")
        wanted[session] = record["task"]
    rows = [row for row in latest_episodes(pq.read_table(data / "ukisai.parquet").to_pylist()) if row["trial_name"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError(f"{len(wanted) - len(rows)} wanted sessions are not in ukisai.parquet")
    with open(out / "sessions.jsonl", "w") as handle:
        for row in rows:
            if row["task"] != wanted[row["trial_name"]]:
                raise ValueError(f"session {row['trial_name']}: task {row['task']} here, {wanted[row['trial_name']]} in the conversion")
            handle.write(session_line(row) + "\n")
    needed = {task_folder(task) for task in wanted.values()}
    dockerfiles = {}
    with open(out / "tasks.jsonl", "w") as handle:
        for family in (NL2BASH, INFERREDBUGS):
            names = {task for task in needed if family_of(task) is family}
            archives = {row["path"]: row["task_binary"] for row in pq.read_table(parquets[family.name]).to_pylist() if row["path"] in names}
            missing = sorted(names - set(archives))
            if missing:
                raise ValueError(f"{len(missing)} {family.name} task folders are not in {parquets[family.name]}, e.g. {missing[:5]}")
            for task in sorted(archives):
                with tarfile.open(fileobj=io.BytesIO(archives[task])) as archive:
                    dockerfile = archive.extractfile("environment/Dockerfile").read().decode()
                if hashlib.sha256(dockerfile.encode()).hexdigest() != family.dockerfile_sha256:
                    raise ValueError(f"task {task}: its Dockerfile is not the shared {family.name} one, so the shared image does not fit it")
                handle.write(task_line(task, archives[task]) + "\n")
            dockerfiles[family.name] = dockerfile
    for family in (NL2BASH, INFERREDBUGS):
        for harness in (TMUX_SOCKET, ASCIINEMA):
            folder = out / "images" / f"{family.name}-{harness.name}"
            folder.mkdir(parents=True)
            (folder / "Dockerfile").write_text(image_dockerfile(dockerfiles[family.name], family, harness))
            if harness is ASCIINEMA:
                (folder / ASCIINEMA_SCRIPT).write_bytes(TmuxSession._GET_ASCIINEMA_TIMESTAMP_SCRIPT_HOST_PATH.read_bytes())
    print(f"{len(rows)} sessions, {len(needed)} task folders, images in {out / 'images'}")


if __name__ == "__main__":
    main()
