"""Add Harbor datasets to an existing task-set split (results/imitation/task-sets.json) without changing it.

The first survey (task_sets_split.py) split nine datasets. This script adds more datasets on top: every task of an
added dataset goes to training, held-out or excluded, and the entries already in task-sets.json are kept as they are.

Inputs:
  task-sets.json        the existing split (read; written back with the added datasets)
  inventory.json        task_sets_inventory.py over the added datasets
  similarity.json       task_sets_similarity.py over all datasets (old + added) with --new <added datasets>
  topic.json            task_sets_topic_check.py over all datasets with --new <trainable added datasets>
  image_sizes.json      task_sets_image_sizes.py over the added inventory
  context_sizes.txt     "<dataset>/<task> <environment bytes> <tests bytes>" (du -sb on the downloaded tasks)
  DL                    the downloaded added datasets (tests/Dockerfile is read to size verifier builds)

Links (connected components = groups; every member of a group gets the same side):
  - text near-duplicate (5-gram Jaccard >= 0.5 or difflib >= 0.8) with any task, old or added
  - equal normalised task name across datasets
  - word TF-IDF cosine >= 0.45, except between software-engineering tasks of two different datasets that both name
    a repository: their vocabulary is the repository's, so these pairs were read by hand instead (all different
    issues; listed in task-sets.md)
  - HELD_OUT_NEAR below (hand-checked)
  - SWE-rebench: all tasks of one repository (the owner asked for a repository-level split)
Sides:
  1. A group with a training task of the first survey (or a Terminal-Bench 2.0 training task) trains. A member from a
     whole-held-out dataset is then excluded.
  2. A group with a held-out, evaluation or excluded task of the first survey is held out (its added tasks may be
     evaluated, never trained on).
  3. A group with a task of a whole-held-out added dataset is held out.
  4. Other groups are drawn with the seed: SWE-rebench groups until HELD_OUT_REPO_FRACTION of its repositories are
     held out; SkillsBench groups until HELD_OUT_FRACTION of its tasks are held out.
  5. Excluded on both sides: tasks that need MCP tools (pi gives the model only bash). Excluded on the training side:
     tasks that need a GPU.
Sizes: as task_sets_summary.py (compressed, shared registry layers counted once, Dockerfile installs estimated from
the first survey's calibration), with one change: a Dockerfile that only installs uv or makes directories (SWE-rebench)
or has no RUN step (ORCA-bench) adds LIGHT_INSTALL_GB, not the 0.32 GB average.

Usage:
  python3 task_sets_add.py task-sets.json inventory.json similarity.json topic.json image_sizes.json \
      context_sizes.txt DL training-tasks.json --inventory task-sets-inventory.json > tables.md
(task-sets.json and task-sets-inventory.json are updated in place.)
"""

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SEED = 20261004
TOPIC_LINK = 0.45
HELD_OUT_FRACTION = 0.5
HELD_OUT_REPO_FRACTION = 0.15
GB = 1e9
# Calibration from the first survey (task-sets.md, "Image sizes"): mean size a Dockerfile adds on top of its base.
INSTALL_GB = {"plain": 0.32, "heavy": 1.34}
LIGHT_INSTALL_GB = 0.05

SOURCES = {
    "swe-rebench-leaderboard": "swe-rebench/swe-rebench-leaderboard@1.0.1",
    "deep-swe-1-1": "datacurve/deep-swe-1-1@1.0.0 (DeepSWE 1.1, the latest; datacurve/deep-swe is 1.0)",
    "swe-atlas-qna": "scale-ai/swe-atlas-qna@1.0.0",
    "swe-atlas-rf": "scale-ai/swe-atlas-rf@1.0.0",
    "swe-atlas-tw": "scale-ai/swe-atlas-tw@1.1.0",
    "skillsbench": "benchflow/skillsbench@2.0.1 (SkillsBench v1.1)",
    "orca-bench": "orca-bench/orca-bench@2.0.0 (public split)",
    "slopcodebench": "gabeorlanski/slopcodebench@4.0.0",
}

WHOLE_HELD_OUT = {
    "deep-swe-1-1": "DeepSWE is a test benchmark for frontier coding agents with a public leaderboard and no training "
    "split; kept whole so its numbers stay comparable. SWE-rebench already supplies SWE training tasks",
    "swe-atlas-qna": "SWE-Atlas (Scale AI leaderboard) test benchmark; graded by a language-model judge with rubrics "
    "(default Claude Opus 4.5), so every training rollout would cost a judge call",
    "swe-atlas-rf": "SWE-Atlas (Scale AI leaderboard) test benchmark; graded by a language-model judge with rubrics",
    "swe-atlas-tw": "SWE-Atlas (Scale AI leaderboard) test benchmark; graded by a language-model judge with rubrics",
    "slopcodebench": "SlopCodeBench is a leaderboard benchmark (scbench.ai) of 36 multi-step tasks (196 checkpoints, "
    "2 h agent limit per checkpoint); our runner has not been tried on Harbor multi-step tasks",
    "orca-bench": "ORCA-bench test benchmark (root cause analysis). Its tasks share telemetry snapshots: grouped by "
    "snapshot and incident events they form 17 groups, and the 40 tasks of its reported verified subset sit in 15 of "
    "them, so no split keeps that subset held out. It also needs a patched Harbor, privileged containers with the "
    "host Docker socket, and a language-model judge",
}
SPLIT_POLICY = {
    "swe-rebench-leaderboard": "repository-level seeded split: about 15% of repositories (all their tasks) held out as "
    "our own validation benchmark; the rest train (owner decision)",
    "skillsbench": "seeded group split, 50% held out (Terminal-Bench-style tasks; numbers are not comparable with the "
    "published ones anyway, because pi does not load the benchmark's skill folders)",
}

# Hand-checked: added task -> (task of the first survey, why). Same shape as a held-out task.
HELD_OUT_NEAR = {
    ("skillsbench", "jpg-ocr-stat"): (
        ("harbor-index-1.0", "skillsbench-ocr-receipts-to-excel"),
        "the same task (Harbor Index adapted it from SkillsBench; Jaccard 0.94)",
    ),
    ("skillsbench", "shock-analysis-supply"): (
        ("harbor-index-1.0", "skillsbench-model-investment-shock-gdp"),
        "the same task reworded (Georgia investment shock, Cobb-Douglas, Excel; cosine 0.77)",
    ),
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def install_class(dockerfile: str) -> str:
    """'none' (no RUN), 'light' (RUN steps only install uv / make directories / chmod), else ''."""
    runs = [l.strip() for l in dockerfile.splitlines() if re.match(r"\s*RUN\b", l, re.IGNORECASE)]
    if not runs:
        return "none"
    if all(re.search(r"astral\.sh/uv|UV_URL|^RUN mkdir|^RUN chmod", r) for r in runs):
        return "light"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("task_sets", "inventory", "similarity", "topic", "sizes", "contexts", "dl", "training_tasks"):
        ap.add_argument(name, type=Path)
    ap.add_argument("--inventory-out", type=Path, required=True, help="task-sets-inventory.json (rows are appended)")
    args = ap.parse_args()

    split = json.loads(args.task_sets.read_text())
    inv = json.loads(args.inventory.read_text())
    sim = json.loads(args.similarity.read_text())
    topic = json.loads(args.topic.read_text())
    sizes = json.loads(args.sizes.read_text())
    frozen = json.loads(args.training_tasks.read_text())
    added = sorted({r["dataset"] for r in inv})
    clash = set(added) & set(split["datasets"])
    if clash:
        raise ValueError(f"datasets already in {args.task_sets}: {sorted(clash)}")
    unknown = set(added) - set(WHOLE_HELD_OUT) - set(SPLIT_POLICY)
    if unknown:
        raise ValueError(f"no policy for {sorted(unknown)}")
    by_key = {(r["dataset"], r["task"]): r for r in inv}

    # Side of every task of the first survey; the frozen evaluation tasks count as held out.
    old_side: dict = {}
    for ds, e in split["datasets"].items():
        for s in ("training", "held_out"):
            for t in e[s]:
                old_side[(ds, t)] = s
        for t in e["excluded"]:
            old_side[(ds, t)] = "excluded"
    for t in frozen["excluded_evaluation"] + list(frozen["excluded_leak_twins"]):
        old_side[("tb2-eval", t)] = "held_out"

    uf = UnionFind()
    for k in by_key:
        uf.find(k)
    link_why: dict = defaultdict(list)

    def link(a, b, why: str) -> None:
        a, b = tuple(a), tuple(b)
        for x in (a, b):
            if x not in by_key and x not in old_side:
                raise KeyError(f"unknown task {x} in a link ({why})")
        uf.union(a, b)
        for x, y in ((a, b), (b, a)):
            if x in by_key:
                link_why[x].append(f"{why} with {y[0]}/{y[1]}")

    for p in sim["pairs"]:
        if p["near_duplicate"]:
            link(p["a"], p["b"], f"text near-duplicate (Jaccard {p['jaccard5']}, difflib {p['difflib2000']})")
    for m in sim["name_matches"]:
        if m["exact"]:
            link(m["a"], m["b"], "same task name")
    repo_of = {k: r["repository"] for k, r in by_key.items()}
    for p in topic["pair_hits"]:
        if p["cosine"] < TOPIC_LINK:
            continue
        a, b = tuple(p["a"]), tuple(p["b"])
        both_repo = all(
            (x in by_key and repo_of[x]) or (x not in by_key and re.fullmatch(r"[\w.-]+__[\w.-]+-\d+", x[1]))
            for x in (a, b)
        )
        if both_repo and a[0] != b[0]:
            continue
        link(a, b, f"same topic (word TF-IDF cosine {p['cosine']})")
    for k, (other, why) in HELD_OUT_NEAR.items():
        if old_side.get(other) != "held_out":
            raise ValueError(f"HELD_OUT_NEAR target {other} is not held out in {args.task_sets}")
        link(k, other, why)
    repo_node = {}
    for k, r in by_key.items():
        if k[0] == "swe-rebench-leaderboard":
            if not r["repository"]:
                raise ValueError(f"no repository for {k}")
            repo_node[k] = ("repository", r["repository"])
            uf.union(k, repo_node[k])

    groups: dict = defaultdict(set)
    for node in list(uf.parent):
        groups[uf.find(node)].add(node)

    side: dict = {}
    forced: dict = {}
    roots = sorted({uf.find(k) for k in by_key}, key=str)
    for root in roots:
        g = groups[root]
        olds = {n: old_side[n] for n in g if n in old_side}
        if any(s == "training" for s in olds.values()):
            side[root] = "training"
            forced[root] = "same group as training task(s) " + ", ".join(
                sorted(f"{n[0]}/{n[1]}" for n, s in olds.items() if s == "training")
            )
        elif olds:
            side[root] = "held_out"
            forced[root] = "same group as held-out task(s) " + ", ".join(sorted(f"{n[0]}/{n[1]}" for n in olds))
        elif any(n in by_key and n[0] in WHOLE_HELD_OUT for n in g):
            side[root] = "held_out"
            if any(n in by_key and n[0] not in WHOLE_HELD_OUT for n in g):
                forced[root] = "same group as a task of a dataset kept whole as held-out"

    # Seeded draw of the free groups.
    def units(root) -> dict:
        """What a group counts towards its dataset's target: repositories for SWE-rebench, tasks otherwise."""
        c: dict = defaultdict(set)
        for n in groups[root]:
            if n in by_key:
                c[n[0]].add(repo_node[n] if n[0] == "swe-rebench-leaderboard" else n)
        return {d: len(v) for d, v in c.items()}

    total: dict = defaultdict(int)
    for root in roots:
        for d, c in units(root).items():
            total[d] += c
    target = {
        d: round((HELD_OUT_REPO_FRACTION if d == "swe-rebench-leaderboard" else HELD_OUT_FRACTION) * total[d])
        for d in SPLIT_POLICY
    }
    held: dict = defaultdict(int)
    for root, s in side.items():
        if s == "held_out":
            for d, c in units(root).items():
                held[d] += c
    free = [r for r in roots if r not in side]
    rng = random.Random(SEED)
    rng.shuffle(free)
    for root in free:
        u = units(root)
        fits = all(held[d] + c <= target[d] for d, c in u.items())
        side[root] = "held_out" if fits else "training"
        if fits:
            for d, c in u.items():
                held[d] += c

    out_entries: dict = {}
    for ds in added:
        entry = {
            "source": "harbor download " + SOURCES[ds],
            "policy": WHOLE_HELD_OUT.get(ds) or SPLIT_POLICY[ds],
            "training": [],
            "held_out": [],
            "excluded": {},
            "forced_side": {},
        }
        for k in sorted(k for k in by_key if k[0] == ds):
            r = by_key[k]
            root = uf.find(k)
            s = side[root]
            reason = None
            if r["mcp_servers"]:
                reason = "needs MCP tools (pi gives the model only bash)"
            elif s == "training" and ds in WHOLE_HELD_OUT:
                reason = forced[root] + "; its dataset is kept whole as held-out, so it is not trained on either"
            elif s == "training" and (r["gpus"] or r["gpu_in_compose"]):
                reason = f"needs {r['gpus'] or 'a'} GPU"
            if reason:
                entry["excluded"][k[1]] = reason
                continue
            entry[s].append(k[1])
            if root in forced:
                entry["forced_side"][k[1]] = f"{s}: {forced[root]}"
        out_entries[ds] = entry

    # SWE-rebench repositories: held-out list and overlaps with other sets' repositories.
    sr = out_entries["swe-rebench-leaderboard"]
    repo_side: dict = defaultdict(set)
    for t in sr["training"]:
        repo_side[by_key[("swe-rebench-leaderboard", t)]["repository"]].add("training")
    for t in sr["held_out"]:
        repo_side[by_key[("swe-rebench-leaderboard", t)]["repository"]].add("held_out")
    mixed = sorted(r for r, s in repo_side.items() if len(s) > 1)
    if mixed:
        raise ValueError(f"repositories on both sides: {mixed}")
    sr["held_out_repositories"] = sorted(r for r, s in repo_side.items() if s == {"held_out"})
    sr["training_repositories"] = len([r for r, s in repo_side.items() if s == {"training"}])
    swe_verified_repos = {
        re.sub(r"-\d+$", "", t).replace("__", "/").lower() for t in split["datasets"]["swe-bench-verified"]["held_out"]
    }
    other_repos = {"swe-bench-verified": swe_verified_repos}
    for ds in added:
        if ds != "swe-rebench-leaderboard":
            other_repos[ds] = {by_key[(ds, t)]["repository"] for t in out_entries[ds]["held_out"]} - {None}
    shared = {}
    for ds, repos in other_repos.items():
        common = sorted(set(repo_side) & repos)
        if common:
            shared[ds] = {
                r: {
                    "swe_rebench_side": sorted(repo_side[r])[0],
                    "swe_rebench_tasks": sum(
                        1 for k in by_key if k[0] == "swe-rebench-leaderboard" and by_key[k]["repository"] == r
                    ),
                }
                for r in common
            }
    sr["repositories_shared_with_held_out_sets"] = shared

    for ds in added:
        split["datasets"][ds] = out_entries[ds]
    split["datasets"] = dict(sorted(split["datasets"].items()))
    split["added_datasets"] = {
        "date": "2026-10-04",
        "script": "results/imitation/scripts/task_sets_add.py",
        "seed": SEED,
        "datasets": added,
        "swe_rebench_held_out_repository_fraction": HELD_OUT_REPO_FRACTION,
        "skillsbench_held_out_fraction": HELD_OUT_FRACTION,
    }
    cross = []
    for root in roots:
        g = groups[root]
        real = sorted(n for n in g if n in by_key or n in old_side)
        if len({n[0] for n in real}) > 1:
            cross.append(sorted(f"{n[0]}/{n[1]}" for n in real))
    split["cross_dataset_groups"] = split["cross_dataset_groups"] + sorted(cross)
    args.task_sets.write_text(json.dumps(split, indent=1) + "\n")

    # Sizes.
    ctx = {}
    for line in args.contexts.read_text().splitlines():
        k, e, t = line.split()
        ctx[k] = (int(e), int(t))
    unknown_images: set = set()

    def add_image(u: dict, img: str) -> None:
        v = sizes.get(img)
        if v is None:
            raise KeyError(f"no size entry for {img}; run task_sets_image_sizes.py first")
        if "error" in v:
            unknown_images.add(img)
        else:
            u.update({"layer:" + d: b for d, b in v["layers"].items()})

    def task_units(r: dict) -> dict:
        u: dict = {}
        key = f"{r['dataset']}/{r['task']}"
        tdir = args.dl / r["dataset"] / r["task"]
        if r["prebuilt_image"]:
            add_image(u, r["prebuilt_image"])
        else:
            add_image(u, r["dockerfile_base"])
            cls = install_class((tdir / "environment" / "Dockerfile").read_text(errors="replace"))
            gb = 0.0 if cls == "none" else LIGHT_INSTALL_GB if cls == "light" else INSTALL_GB["heavy" if r["heavy_installs"] else "plain"]
            u[f"install:{r['dockerfile_base']}:{r['dockerfile_prefix_hash']}"] = gb * GB
            u["context:" + key] = ctx[key][0]
        if r["verifier_image"]:
            add_image(u, r["verifier_image"])
        elif r["verifier_dockerfile_base"]:
            add_image(u, r["verifier_dockerfile_base"])
            cls = install_class((tdir / "tests" / "Dockerfile").read_text(errors="replace"))
            gb = 0.0 if cls in ("none", "light") else INSTALL_GB["plain"]
            u["verifier-build:" + key] = ctx[key][1] + gb * GB
        for img in r["compose_images"]:
            add_image(u, img)
        return u

    rows = json.loads(args.inventory_out.read_text())
    if any((x["dataset"], x["task"]) in by_key for x in rows):
        raise ValueError(f"{args.inventory_out} already has rows of the added datasets")
    out = sys.stdout
    out.write(
        "| Dataset | Tasks | Training | Held-out | Excluded | All tasks GB | Median per task GB | Largest task (GB) "
        "| Training GB | Held-out GB |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    tot_tr: dict = {}
    tot_ho: dict = {}
    for ds in added:
        e = out_entries[ds]
        tr, ho, al = {}, {}, {}
        per = []
        for k in sorted(k for k in by_key if k[0] == ds):
            r = by_key[k]
            u = task_units(r)
            al.update(u)
            s = "training" if k[1] in e["training"] else "held_out" if k[1] in e["held_out"] else "excluded"
            if s == "training":
                tr.update(u)
            elif s == "held_out":
                ho.update(u)
            per.append((sum(u.values()), k[1]))
            agent = sizes.get(r["prebuilt_image"] or r["dockerfile_base"], {}).get("bytes")
            row = {x: v for x, v in r.items() if x != "dockerfile_prefix_hash"}
            row["side"] = s
            row["agent_image_gb"] = round(agent / GB, 3) if agent is not None and r["prebuilt_image"] else None
            row["agent_base_image_gb"] = round(agent / GB, 3) if agent is not None and not r["prebuilt_image"] else None
            row["task_total_gb_estimate"] = round(sum(u.values()) / GB, 3)
            rows.append(row)
        tot_tr.update(tr)
        tot_ho.update(ho)
        per.sort()
        out.write(
            f"| {ds} | {len(per)} | {len(e['training'])} | {len(e['held_out'])} | {len(e['excluded'])} "
            f"| {sum(al.values()) / GB:,.1f} | {statistics.median(p for p, _ in per) / GB:.2f} "
            f"| {per[-1][1]} ({per[-1][0] / GB:.1f}) | {sum(tr.values()) / GB:,.1f} | {sum(ho.values()) / GB:,.1f} |\n"
        )
    out.write(
        f"| **all added** | | | | | | | | {sum(tot_tr.values()) / GB:,.1f} | {sum(tot_ho.values()) / GB:,.1f} |\n"
    )
    if unknown_images:
        out.write(f"\nSizes not readable from a registry: {', '.join(sorted(unknown_images))}\n")
    args.inventory_out.write_text(json.dumps(rows, indent=1) + "\n")
    for ds in added:
        e = out_entries[ds]
        print(f"{ds:26s} train {len(e['training']):4d}  held-out {len(e['held_out']):4d}  excluded {len(e['excluded']):3d}", file=sys.stderr)
    print(f"SWE-rebench held-out repositories: {len(sr['held_out_repositories'])} of {len(repo_side)}", file=sys.stderr)


if __name__ == "__main__":
    main()
