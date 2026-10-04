"""Inventory of Harbor task sets: one record per task with image, resources, limits and network settings.

Input: a directory laid out as `harbor download <org/dataset> --export -o DIR` writes it,
i.e. DIR/<dataset>/<task>/{task.toml, instruction.md, environment/...}.
Only task.toml, instruction.md, environment/Dockerfile (+ docker-compose files) and tests/Dockerfile are read.

Output: JSON list of task records (stdout or --out).

Usage: python3 task_sets_inventory.py DIR --out inventory.json
"""

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

# Words in a Dockerfile that signal a large install (used only to flag image-size estimates).
HEAVY_PATTERNS = {
    "torch": r"\btorch\b|pytorch",
    "cuda": r"\bcuda\b|nvidia",
    "tensorflow/jax": r"tensorflow|\bjax\b",
    "conda": r"conda",
    "texlive": r"texlive",
    "hf-model-download": r"huggingface|snapshot_download|hf_hub_download",
    "rust/go/java toolchain": r"rustup|golang|openjdk|\bjdk\b",
}


def final_from(dockerfile: str) -> tuple[str | None, list[str]]:
    """Return the base image of the last build stage and the list of all FROM images."""
    froms = []
    stage_names = set()
    for line in dockerfile.splitlines():
        # Whole-line match, so a Python "from x import y" line inside a RUN heredoc is not taken for a FROM.
        m = re.match(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?\s*$", line, re.IGNORECASE)
        if m:
            image = m.group(1)
            froms.append(image)
            if m.group(2):
                stage_names.add(m.group(2).lower())
    if not froms:
        return None, []
    last = froms[-1]
    # A final stage built FROM an earlier stage name is not a registry image; walk back to a real one.
    i = len(froms) - 1
    while i >= 0 and froms[i].lower() in stage_names:
        i -= 1
    real_froms = [f for f in froms if f.lower() not in stage_names]
    return (froms[i] if i >= 0 else last), real_froms


def install_prefix_hash(dockerfile: str) -> str | None:
    """Hash of the Dockerfile up to its first COPY/ADD (comments and blank lines dropped).

    Two tasks with the same hash build the same layers before copying task files in, so when Harbor builds them on
    one machine the Docker build cache stores those layers once.
    """
    lines = []
    for line in dockerfile.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if re.match(r"(COPY|ADD)\b", t, re.IGNORECASE):
            break
        lines.append(t)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16] if lines else None


def gpu_from_compose(text: str) -> bool:
    return bool(re.search(r"driver:\s*nvidia|capabilities:\s*\[?\s*gpu|runtime:\s*nvidia", text))


def record(dataset: str, task_dir: Path) -> dict:
    cfg = tomllib.loads((task_dir / "task.toml").read_text())
    env = cfg.get("environment", {})
    agent = cfg.get("agent", {})
    verifier = cfg.get("verifier", {})
    ver_env = verifier.get("environment") or {}
    env_dir = task_dir / "environment"
    dockerfile_path = env_dir / "Dockerfile"
    dockerfile = dockerfile_path.read_text(errors="replace") if dockerfile_path.exists() else ""
    base, all_froms = final_from(dockerfile)
    compose_files = sorted(p.name for p in env_dir.glob("docker-compose*.y*ml")) if env_dir.exists() else []
    compose_text = "".join((env_dir / c).read_text(errors="replace") for c in compose_files)
    compose_services = sorted(set(re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", compose_text, re.MULTILINE)))
    # Sidecar images pulled by docker compose (the main service is built from the task's own image).
    compose_images = sorted(
        {i.strip("'\"") for i in re.findall(r"^\s+image:\s*(\S+)", compose_text, re.MULTILINE) if "$" not in i}
    )
    # A separate verifier without a prebuilt image is built from tests/Dockerfile.
    ver_dockerfile = task_dir / "tests" / "Dockerfile"
    ver_base = None
    if verifier.get("environment_mode") == "separate" and not ver_env.get("docker_image") and ver_dockerfile.exists():
        ver_base, _ = final_from(ver_dockerfile.read_text(errors="replace"))
    instruction_path = task_dir / "instruction.md"
    instruction = instruction_path.read_text(errors="replace") if instruction_path.exists() else ""
    if "network_mode" in env:
        network = env["network_mode"]
    elif env.get("allow_internet") is False:
        network = "no-network"
    elif env.get("allow_internet") is True:
        network = "public"
    else:
        network = "public (Harbor default)"
    heavy = [k for k, pat in HEAVY_PATTERNS.items() if re.search(pat, dockerfile, re.IGNORECASE)]
    return {
        "dataset": dataset,
        "task": task_dir.name,
        "task_name": cfg.get("task", {}).get("name"),
        "category": cfg.get("metadata", {}).get("category"),
        "difficulty": cfg.get("metadata", {}).get("difficulty"),
        "prebuilt_image": env.get("docker_image"),
        "dockerfile_base": base,
        "dockerfile_froms": all_froms,
        "dockerfile_lines": len(dockerfile.splitlines()),
        "dockerfile_prefix_hash": install_prefix_hash(dockerfile),
        "heavy_installs": heavy,
        "verifier_image": ver_env.get("docker_image"),
        "verifier_separate": verifier.get("environment_mode") == "separate",
        "verifier_dockerfile_base": ver_base,
        "compose_files": compose_files,
        "compose_services": compose_services,
        "compose_images": compose_images,
        "cpus": env.get("cpus"),
        "memory_mb": env.get("memory_mb"),
        "storage_mb": env.get("storage_mb"),
        "gpus": env.get("gpus") or 0,
        "gpu_types": env.get("gpu_types"),
        "gpu_in_compose": gpu_from_compose(compose_text),
        "network": network,
        "mcp_servers": len(env.get("mcp_servers") or []),
        "agent_timeout_sec": agent.get("timeout_sec"),
        "verifier_timeout_sec": verifier.get("timeout_sec"),
        "build_timeout_sec": env.get("build_timeout_sec"),
        "instruction_chars": len(instruction),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = []
    for ds in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in ds.iterdir() if p.is_dir()):
            if not (task_dir / "task.toml").exists():
                raise FileNotFoundError(f"{task_dir} has no task.toml")
            rows.append(record(ds.name, task_dir))
    text = json.dumps(rows, indent=1)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    print(f"{len(rows)} tasks", file=sys.stderr)


if __name__ == "__main__":
    main()
