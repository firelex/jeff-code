"""Per-dataset resource and image-size tables for results/imitation/task-sets.md, and the casdgx01 disk plan.

Image size (compressed, linux/amd64, from the registry, see task_sets_image_sizes.py):
  - prebuilt task image: exact registry size.
  - Dockerfile-only task image: base image size + build-context size + an average for what the Dockerfile installs.
    The average is calibrated on the tasks that have both a Dockerfile and a prebuilt image (Terminal-Bench 2.0 / 2.1,
    terminal-bench, harbor-index): prebuilt size minus base size, mean over tasks, separately for Dockerfiles that
    install something large (torch, CUDA, conda, TeX Live, model downloads, JVM/Go/Rust toolchains) and the rest.
    - separate verifier images and docker-compose sidecar images are counted too (exact when prebuilt, otherwise
    estimated the same way from tests/Dockerfile and the tests/ folder).
Shared storage is counted once: registry layers by digest (an agent image and its verifier image often share almost
all layers), and the install layers of locally built images by (base image, Dockerfile text before the first
COPY/ADD), since the Docker build cache reuses them.
On-disk size = compressed size x DISK_RATIO, measured on casdgx01 for the 45 Terminal-Bench 2.0 training images
(docker image inspect .Size against the registry's compressed size).

Usage: python3 task_sets_summary.py inventory.json image_sizes.json context_sizes.txt task-sets.json \
           casdgx_local_sizes.txt similarity.json [--inventory-out task-sets-inventory.json] > tables.md
The inventory written by --inventory-out adds per task: side (training / held_out / excluded), agent_image_gb (exact,
prebuilt images), agent_base_image_gb (Dockerfile-only tasks: size of the FROM image), task_total_gb_estimate
(task image + verifier + sidecars, compressed; Dockerfile-only parts estimated as described above).
context_sizes.txt: lines "<dataset>/<task> <environment bytes> <tests bytes>" (du -sb on the downloaded tasks).
casdgx_local_sizes.txt: lines "<image> <docker image inspect .Size>" from casdgx01.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GB = 1e9


def fmt(x: float) -> str:
    return f"{x:,.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory", type=Path)
    ap.add_argument("sizes", type=Path)
    ap.add_argument("contexts", type=Path)
    ap.add_argument("task_sets", type=Path)
    ap.add_argument("local_sizes", type=Path)
    ap.add_argument("similarity", type=Path)
    ap.add_argument("--free-gb", type=float, default=870.0)
    ap.add_argument("--inventory-out", type=Path, help="write the inventory with per-task sizes and split side")
    args = ap.parse_args()

    inv = json.loads(args.inventory.read_text())
    sizes = json.loads(args.sizes.read_text())
    splits = json.loads(args.task_sets.read_text())["datasets"]
    ctx = {}
    for line in args.contexts.read_text().splitlines():
        k, e, t = line.split()
        ctx[k] = (int(e), int(t))

    def layers(img: str) -> dict[str, int] | None:
        v = sizes.get(img)
        if v is None:
            raise KeyError(f"no size entry for {img}; run task_sets_image_sizes.py first")
        if "error" in v:
            return None
        return {"layer:" + d: b for d, b in v["layers"].items()}

    # Calibration of what a Dockerfile adds on top of its base image: layers of the prebuilt image that are not
    # layers of the base image.
    extra = {"heavy": [], "plain": []}
    for r in inv:
        if r["prebuilt_image"] and r["dockerfile_base"] and r["dockerfile_base"] in sizes:
            pl, bl = layers(r["prebuilt_image"]), layers(r["dockerfile_base"])
            if pl is not None and bl is not None:
                extra["heavy" if r["heavy_installs"] else "plain"].append(sum(b for d, b in pl.items() if d not in bl))
    extra_mean = {k: statistics.mean(v) for k, v in extra.items()}

    local = []
    for line in args.local_sizes.read_text().splitlines():
        name, disk = line.split()
        if sizes.get(name, {}).get("bytes"):
            local.append((int(disk), sizes[name]["bytes"]))
    if not local:
        raise ValueError("no casdgx01 image matched a registry size; cannot compute the disk ratio")
    disk_ratio = sum(d for d, _ in local) / sum(c for _, c in local)
    disk_ratio_median = statistics.median(d / c for d, c in local)

    unknown: set = set()

    def add_image(u: dict, img: str) -> None:
        lay = layers(img)
        if lay is None:
            unknown.add(img)
        else:
            u.update(lay)

    def units(r: dict) -> dict[str, float]:
        """Disk units a task needs, keyed so that shared ones collapse: registry layers by digest, the estimated
        install layers of a locally built image by (base, Dockerfile text before the first COPY/ADD), and the copied
        task files by task."""
        u: dict[str, float] = {}
        key = f"{r['dataset']}/{r['task']}"
        if r["prebuilt_image"]:
            add_image(u, r["prebuilt_image"])
        else:
            add_image(u, r["dockerfile_base"])
            kind = "heavy" if r["heavy_installs"] else "plain"
            u[f"install:{r['dockerfile_base']}:{r['dockerfile_prefix_hash']}"] = extra_mean[kind]
            u["context:" + key] = ctx[key][0]
        if r["verifier_image"]:
            add_image(u, r["verifier_image"])
        elif r["verifier_dockerfile_base"]:
            add_image(u, r["verifier_dockerfile_base"])
            u["verifier-build:" + key] = ctx[key][1] + extra_mean["plain"]
        for img in r["compose_images"]:
            add_image(u, img)
        return u

    by_ds = defaultdict(list)
    for r in inv:
        by_ds[r["dataset"]].append(r)

    if args.inventory_out:
        rows = []
        for r in inv:
            sp = splits[r["dataset"]]
            side = (
                "training"
                if r["task"] in sp["training"]
                else "held_out" if r["task"] in sp["held_out"] else "excluded"
            )
            agent = sizes.get(r["prebuilt_image"] or r["dockerfile_base"], {}).get("bytes")
            row = {k: v for k, v in r.items() if k != "dockerfile_prefix_hash"}
            row["side"] = side
            row["agent_image_gb"] = round(agent / GB, 3) if agent is not None and r["prebuilt_image"] else None
            row["agent_base_image_gb"] = round(agent / GB, 3) if agent is not None and not r["prebuilt_image"] else None
            row["task_total_gb_estimate"] = round(sum(units(r).values()) / GB, 3)
            rows.append(row)
        args.inventory_out.write_text(json.dumps(rows, indent=1) + "\n")

    out = sys.stdout
    out.write("### Resources per dataset\n\n")
    out.write(
        "| Dataset | Tasks | Image source | GPU tasks | Network | Sidecars / MCP | Agent time limit (s) min / median / max "
        "| CPUs median / max | Memory GB median / max | Storage GB max |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for ds, rows in sorted(by_ds.items()):
        pre = sum(1 for r in rows if r["prebuilt_image"])
        src = f"{pre} prebuilt" if pre == len(rows) else (f"{len(rows) - pre} Dockerfile" if pre == 0 else f"{pre} prebuilt, {len(rows) - pre} Dockerfile")
        gpus = sum(1 for r in rows if r["gpus"])
        nets = defaultdict(int)
        for r in rows:
            nets[r["network"].split(" ")[0]] += 1
        net = ", ".join(f"{k} {v}" for k, v in sorted(nets.items()))
        side = sum(1 for r in rows if len([s for s in r["compose_services"] if s != "main"]) > 0)
        mcp = sum(1 for r in rows if r["mcp_servers"])
        at = sorted(float(r["agent_timeout_sec"]) for r in rows)
        cp = [r["cpus"] for r in rows if r["cpus"]]
        mem = [r["memory_mb"] / 1024 for r in rows if r["memory_mb"]]
        st = [r["storage_mb"] / 1024 for r in rows if r["storage_mb"]]
        out.write(
            f"| {ds} | {len(rows)} | {src} | {gpus} | {net} | {side} / {mcp} | {at[0]:.0f} / {statistics.median(at):.0f} / {at[-1]:.0f} "
            f"| {statistics.median(cp):g} / {max(cp)} | {statistics.median(mem):g} / {max(mem):g} | {max(st):g} |\n"
        )

    out.write("\n### Image size per dataset (compressed GB, shared layers counted once)\n\n")
    out.write(
        "| Dataset | All tasks | Median per task | Largest task | Training side | Held-out side | Training on disk (x"
        f"{disk_ratio:.2f}) |\n|---|---|---|---|---|---|---|\n"
    )
    totals = {"training": {}, "held_out": {}, "all": {}}
    for ds, rows in sorted(by_ds.items()):
        sp = splits[ds]
        train = set(sp["training"])
        held = set(sp["held_out"])
        all_u: dict = {}
        tr_u: dict = {}
        ho_u: dict = {}
        per_task = []
        for r in rows:
            u = units(r)
            per_task.append((sum(u.values()), r["task"]))
            all_u.update(u)
            if r["task"] in train:
                tr_u.update(u)
            if r["task"] in held:
                ho_u.update(u)
        if ds not in ("terminal-bench-2-1",):
            totals["all"].update(all_u)
            totals["training"].update(tr_u)
            totals["held_out"].update(ho_u)
        per_task.sort()
        out.write(
            f"| {ds} | {fmt(sum(all_u.values()) / GB)} | {statistics.median(p for p, _ in per_task) / GB:.2f} "
            f"| {per_task[-1][1]} ({per_task[-1][0] / GB:.1f}) | {fmt(sum(tr_u.values()) / GB)} | {fmt(sum(ho_u.values()) / GB)} "
            f"| {fmt(sum(tr_u.values()) * disk_ratio / GB)} |\n"
        )
    tr = sum(totals["training"].values())
    ho = sum(totals["held_out"].values())
    al = sum(totals["all"].values())
    out.write(
        f"| **all (2.1 not added; it repeats 2.0)** | {fmt(al / GB)} | | | {fmt(tr / GB)} | {fmt(ho / GB)} | {fmt(tr * disk_ratio / GB)} |\n"
    )
    out.write("\n### Calibration and disk numbers\n\n")
    out.write(
        f"- Dockerfile additions (prebuilt layers not in the base, mean): plain {extra_mean['plain'] / GB:.2f} GB over {len(extra['plain'])} tasks, "
        f"large installs {extra_mean['heavy'] / GB:.2f} GB over {len(extra['heavy'])} tasks.\n"
    )
    out.write(
        f"- Disk / compressed ratio on casdgx01: {disk_ratio:.2f} summed over {len(local)} images (median per image {disk_ratio_median:.2f}).\n"
    )
    out.write(
        f"- Training side, all datasets: {fmt(tr / GB)} GB compressed, about {fmt(tr * disk_ratio / GB)} GB on disk "
        f"({fmt(tr * disk_ratio_median / GB)} GB at the median ratio); free on casdgx01 /: {args.free_gb:.0f} GB.\n"
    )
    out.write(
        f"- Held-out side, all datasets: {fmt(ho / GB)} GB compressed, about {fmt(ho * disk_ratio / GB)} GB on disk "
        f"({fmt(ho * disk_ratio_median / GB)} GB at the median ratio).\n"
    )
    if unknown:
        out.write(f"- Sizes not readable from a registry: {', '.join(sorted(unknown))}\n")

    sim = json.loads(args.similarity.read_text())
    by_sim = defaultdict(list)
    for e in sim["eval_max"]:
        by_sim[e["dataset"]].append(e)
    out.write(
        "\n### Text similarity against the 44 frozen evaluation tasks and twins (per task: its highest score)\n\n"
        "| Dataset | Tasks | Median Jaccard | Highest Jaccard (task vs evaluation task) | Median difflib "
        "| Highest difflib (task vs evaluation task) | At or over 0.5 / 0.8 |\n|---|---|---|---|---|---|---|\n"
    )
    for ds, rs in sorted(by_sim.items()):
        if ds in ("terminal-bench-2", "terminal-bench-2-1"):
            continue
        j = max(rs, key=lambda e: e["max_jaccard5"])
        d = max(rs, key=lambda e: e["max_difflib2000"])
        over = sum(1 for e in rs if e["max_jaccard5"] >= 0.5 or e["max_difflib2000"] >= 0.8)
        out.write(
            f"| {ds} | {len(rs)} | {statistics.median(e['max_jaccard5'] for e in rs):.3f} "
            f"| {j['max_jaccard5']:.3f} ({j['task']} vs {j['max_jaccard5_vs']}) "
            f"| {statistics.median(e['max_difflib2000'] for e in rs):.3f} "
            f"| {d['max_difflib2000']:.3f} ({d['task']} vs {d['max_difflib2000_vs']}) | {over} |\n"
        )

    # Largest training images, for the rotation plan.
    big = []
    for ds, rows in by_ds.items():
        if ds == "terminal-bench-2-1":
            continue
        for r in rows:
            if r["task"] in set(splits[ds]["training"]):
                big.append((sum(units(r).values()), ds, r["task"]))
    big.sort(reverse=True)
    out.write("\n### Largest training-side tasks (compressed GB, task image + verifier + sidecars)\n\n| Task | GB |\n|---|---|\n")
    for s, ds, t in big[:12]:
        out.write(f"| {ds}/{t} | {s / GB:.1f} |\n")


if __name__ == "__main__":
    main()
