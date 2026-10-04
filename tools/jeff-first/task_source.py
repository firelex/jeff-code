"""Where a collection task comes from: Terminal-Bench 2.0 by bare name, or one of the Harbor hub datasets of
results/imitation/task-sets.json by "<org>/<dataset>:<task>".

Task ids:
- A bare name (e.g. `adaptive-rejection-sampler`) is a Terminal-Bench 2.0 task; run_phase0.sh runs it exactly as before
  (`--dataset terminal-bench@2.0 -i <name>`). This module does not handle it.
- `<org>/<dataset>:<task>` (e.g. `terminal-bench-pro/terminal-bench-pro:analyze-fen-with-stockfish`) is a task of a
  Harbor hub dataset. The dataset must be one of task-sets.json's datasets with a pinned hub version (`hub.ref`, a
  content digest), and the task must be on its training or held-out side. Excluded tasks, unknown datasets and tasks,
  tasks that need a GPU or MCP tools (task-sets-inventory.json), and Terminal-Bench 2.0 / 2.1 tasks named by hub id
  raise. The full id is the trace's task id (JEFF_FIRST_TASK_ID), so converted rows tell datasets apart.

Agent time: Harbor multiplies each task's own agent limit by --agent-timeout-multiplier. The new datasets have limits
up to 8 hours, so for them the multiplier is lowered so that no task gets more than Terminal-Bench 2.0's default limit
(15 minutes) times the run's multiplier, the time a default Terminal-Bench task gets (6 x 15 = 90 minutes):
    multiplier = min(run multiplier, 900 * run multiplier / task limit)
A task with a shorter limit keeps limit x run multiplier, as Terminal-Bench 2.0 tasks do (a 10-minute task gets 60).

Usage (run_phase0.sh calls these):
    python3 task_source.py check TASK_SETS INVENTORY TASK...        exits non-zero naming the first bad task id
    python3 task_source.py harbor-args TASK_SETS INVENTORY MULTIPLIER TASK
        prints: <dataset@ref> <task package name for -i> <agent timeout multiplier> <file-safe name> <offline|online>
    python3 task_source.py image TASK_SETS INVENTORY TASK     the image to pull before the session, or "-"
    python3 task_source.py side TASK_SETS INVENTORY TASK      training, held_out or excluded (bare TB 2.0 names: training)
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TB2_DEFAULT_AGENT_SEC = 900
# Terminal-Bench 2.0 and 2.1 tasks are run by bare name (terminal-bench@2.0, the frozen split); a hub id for them would
# be a second name for the same task.
TB2_FOLDERS = {"terminal-bench-2", "terminal-bench-2-1"}
HUB_TASK = re.compile(r"^([a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*):([A-Za-z0-9][A-Za-z0-9._-]*)$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
# Run without internet (harbor_agent/offline.py; only the model's host stays reachable): a SWE task's agent could
# otherwise fetch the upstream fix from GitHub. SWE-rebench's tests pass offline (checked 2026-10-04, see the report).
OFFLINE_DATASETS = {"swe-rebench/swe-rebench-leaderboard"}
# One large Docker Hub image per task (SWE-rebench: 671 images, ~1.3 TB): pulled before the session and removed after
# it (hub_stream.sh), so the disk holds only the running tasks' images.
ROTATED_DATASETS = {"swe-rebench/swe-rebench-leaderboard"}


def is_hub_task(task_id: str) -> bool:
    """True for "<org>/<dataset>:<task>", False for a bare Terminal-Bench 2.0 name; anything else raises."""
    if HUB_TASK.match(task_id):
        return True
    if ":" in task_id or "/" in task_id:
        raise ValueError(f"task id {task_id!r} is neither a bare Terminal-Bench 2.0 name nor '<org>/<dataset>:<task>'")
    return False


@dataclass(frozen=True)
class HubTask:
    task_id: str
    dataset_spec: str  # "<org>/<dataset>@sha256:..." for harbor run --dataset
    include: str  # the task's package name "<org>/<task>" for harbor run -i
    side: str
    agent_timeout_sec: float
    offline: bool
    image: str | None  # the Docker Hub image pulled before and removed after the session (ROTATED_DATASETS)


@dataclass(frozen=True)
class TaskSets:
    # hub name ("org/dataset") -> task-sets.json dataset entry (with "folder" added)
    datasets: dict[str, dict]
    # (folder, task) -> inventory row
    inventory: dict[tuple[str, str], dict]

    def side(self, task_id: str) -> str:
        """training, held_out or excluded; unknown datasets and tasks raise."""
        if not is_hub_task(task_id):
            raise ValueError(f"{task_id!r} is a bare Terminal-Bench 2.0 name, not a hub task id")
        hub, task = HUB_TASK.match(task_id).groups()
        entry = self.datasets.get(hub)
        if entry is None:
            raise ValueError(f"{task_id}: unknown dataset {hub} (task-sets.json has {', '.join(sorted(self.datasets))})")
        if entry["folder"] in TB2_FOLDERS:
            raise ValueError(f"{task_id}: Terminal-Bench 2.0 / 2.1 tasks are run by their bare task name (terminal-bench@2.0)")
        if task in entry["training"]:
            return "training"
        if task in entry["held_out"]:
            return "held_out"
        if task in entry["excluded"]:
            return "excluded"
        raise ValueError(f"{task_id}: dataset {hub} has no task {task} in task-sets.json")

    def resolve(self, task_id: str) -> HubTask:
        """A runnable task: training or held-out, no GPU, no MCP tools."""
        side = self.side(task_id)
        hub, task = HUB_TASK.match(task_id).groups()
        entry = self.datasets[hub]
        if side == "excluded":
            raise ValueError(f"{task_id} is excluded: {entry['excluded'][task]}")
        row = self.inventory.get((entry["folder"], task))
        if row is None:
            raise ValueError(f"{task_id}: not in task-sets-inventory.json")
        if row["gpus"]:
            raise ValueError(f"{task_id} needs a GPU ({row['gpus']}); collection runs without GPUs")
        if row["mcp_servers"]:
            raise ValueError(f"{task_id} needs MCP tools; pi gives the model only bash")
        image = None
        if hub in ROTATED_DATASETS:
            image = row["dockerfile_base"]
            if not image:
                raise ValueError(f"{task_id}: its dataset's images are rotated, but the inventory names no base image")
        return HubTask(task_id, f"{hub}@{entry['hub']['ref']}", row["task_name"], side, float(row["agent_timeout_sec"]), hub in OFFLINE_DATASETS, image)

    def training_ids(self) -> list[str]:
        """Every hub training task id (Terminal-Bench 2.0 / 2.1 left out: those run by bare name)."""
        return sorted(f"{hub}:{task}" for hub, entry in self.datasets.items() if entry["folder"] not in TB2_FOLDERS for task in entry["training"])


def load_task_sets(task_sets_path: Path, inventory_path: Path | None) -> TaskSets:
    """task-sets.json (sides and pinned hub versions) and, for running, task-sets-inventory.json (limits, GPUs, MCP)."""
    datasets: dict[str, dict] = {}
    for folder, entry in json.loads(Path(task_sets_path).read_text())["datasets"].items():
        hub = entry.get("hub")
        if hub is None:
            continue
        if not DIGEST.match(hub["ref"]):
            raise ValueError(f"{task_sets_path}: dataset {folder} is not pinned to a content digest (hub.ref {hub['ref']!r})")
        datasets[hub["name"]] = {**entry, "folder": folder}
    if not datasets:
        raise ValueError(f"{task_sets_path}: no dataset has a pinned hub version")
    inventory: dict[tuple[str, str], dict] = {}
    if inventory_path is not None:
        inventory = {(row["dataset"], row["task"]): row for row in json.loads(Path(inventory_path).read_text())}
    return TaskSets(datasets, inventory)


def agent_timeout_multiplier(task_limit_sec: float, run_multiplier: float) -> float:
    """The multiplier giving min(limit x run multiplier, 15 minutes x run multiplier); see the module docstring."""
    if task_limit_sec <= 0 or run_multiplier <= 0:
        raise ValueError(f"the task limit and the multiplier must be positive, got {task_limit_sec} and {run_multiplier}")
    return min(run_multiplier, TB2_DEFAULT_AGENT_SEC * run_multiplier / task_limit_sec)


def safe_name(task_id: str) -> str:
    """The task id as one file or folder name (job names and log files)."""
    return task_id.replace("/", ".").replace(":", ".")


def hub_harbor_args(sets: TaskSets, task_id: str, run_multiplier: float) -> list[str]:
    task = sets.resolve(task_id)
    multiplier = agent_timeout_multiplier(task.agent_timeout_sec, run_multiplier)
    return [task.dataset_spec, task.include, format(multiplier, ".17g"), safe_name(task_id), "offline" if task.offline else "online"]


def main(argv: list[str]) -> None:
    if len(argv) >= 4 and argv[0] == "check":
        sets = load_task_sets(Path(argv[1]), Path(argv[2]))
        for task_id in argv[3:]:
            if is_hub_task(task_id):
                sets.resolve(task_id)
        return
    if len(argv) == 5 and argv[0] == "harbor-args":
        sets = load_task_sets(Path(argv[1]), Path(argv[2]))
        print(" ".join(hub_harbor_args(sets, argv[4], float(argv[3]))))
        return
    if len(argv) == 4 and argv[0] == "side":
        print(load_task_sets(Path(argv[1]), Path(argv[2])).side(argv[3]) if is_hub_task(argv[3]) else "training")
        return
    if len(argv) == 4 and argv[0] == "image":
        image = load_task_sets(Path(argv[1]), Path(argv[2])).resolve(argv[3]).image if is_hub_task(argv[3]) else None
        print(image or "-")
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
