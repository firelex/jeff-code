"""Seeded split of the Harbor task sets into training, held-out evaluation, and excluded tasks.

Inputs (all produced by the other task_sets_*.py scripts, plus the frozen Terminal-Bench 2.0 split):
  inventory.json      task_sets_inventory.py
  similarity.json     task_sets_similarity.py (5-gram Jaccard / difflib pairs, task-name matches)
  topic.json          task_sets_topic_check.py (word TF-IDF cosine pairs)
  training-tasks.json results/imitation/training-tasks.json (45 training, 40 evaluation, 4 evaluation twins)

Rules, in order:
 1. Near-duplicate groups. Two tasks are linked when any of these holds; groups are the connected components.
    - text near-duplicate: 5-gram Jaccard >= 0.5 or difflib ratio >= 0.8 (same thresholds as leak-check-ukisai.md)
    - the same normalised task name in two datasets
    - word TF-IDF cosine >= 0.45 (same topic and vocabulary; read by hand, see task-sets.md)
    - aider-polyglot: the same Exercism exercise in different languages
    - harbor-index-1.0: the same source family (name prefix before the first '-', e.g. 'gso', 'hle'), except 'tb'
    - the hand-checked lists EVAL_NEAR and TRAIN_NEAR below link a task to a Terminal-Bench 2.0 task
 2. Terminal-Bench 2.0 and 2.1 keep the frozen split (2.1 is a revision of the same 89 tasks).
 3. A group that contains a Terminal-Bench 2.0 training task goes to the training side: the model has already been
    trained on its twin, so it cannot be held out.
 4. A group that contains a task of a whole-held-out dataset (WHOLE_HELD_OUT) goes to the held-out side (unless
    rule 3 applies; a whole-held-out task in a group with a Terminal-Bench 2.0 training task is then excluded).
 5. All other groups are drawn at random (seed SEED): groups are shuffled, and a group joins the held-out side while
    its dataset's held-out count stays at or below the target (HELD_OUT_FRACTION of the dataset); the rest train.
 6. Exclusions, applied after the draw:
    - a training-side task in a group with a frozen evaluation task (or its twin) is excluded ("near evaluation task")
    - a task that needs Model Context Protocol (MCP) tools is excluded on both sides: the Jeff-Code harness gives the model
      only a bash tool, so the task cannot be solved as intended
    - a training-side task that needs a GPU is excluded (the casdgx01 GPUs serve the models)
Output: task-sets.json.

Usage: python3 task_sets_split.py inventory.json similarity.json topic.json training-tasks.json --out task-sets.json
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 20261004
TOPIC_LINK = 0.45
HELD_OUT_FRACTION = 0.5

WHOLE_HELD_OUT = {
    "swe-bench-verified": "SWE-bench Verified is the curated test subset of SWE-bench; reported whole by everyone",
    "openthoughts-tblite": "OpenThoughts-TBLite is the evaluation set of OpenThoughts-Agent, whose training task pool "
    "is the source of the stage 1 data (ukisai); kept whole so it stays a clean held-out set",
    "aider-polyglot": "Aider Polyglot is a leaderboard benchmark without a training split; reported whole",
}
SPLIT_DATASETS = ["terminal-bench-pro", "terminal-bench", "terminal-bench-science", "harbor-index-1.0"]
TB2_SETS = ["terminal-bench-2", "terminal-bench-2-1"]

# Hand-checked: candidate task -> (frozen evaluation task or twin, why). Found by the text checks and by reading the
# top word TF-IDF neighbours of each evaluation task; only same-shape tasks (same inputs, outputs and procedure) are
# listed, not tasks that merely share a subject.
EVAL_NEAR = {
    ("terminal-bench", "html-js-filter"): ("filter-js-from-html", "text near-duplicate (Jaccard 0.63)"),
    ("harbor-index-1.0", "tb-train-fasttext"): ("train-fasttext", "copy of the evaluation task (Jaccard 0.58)"),
    ("harbor-index-1.0", "tb-make-doom-for-mips"): ("make-doom-for-mips", "copy of an evaluation twin (same name)"),
    ("openthoughts-tblite", "grpc-plant-position-server"): (
        "kv-store-grpc",
        "same template: grpc proto + server.py class Server on port 5328",
    ),
    ("openthoughts-tblite", "csv-json-jsonl-merger"): (
        "multi-source-data-merger",
        "merge user records from three files with different schemas",
    ),
    ("terminal-bench-pro", "sanitize-dclm-repo-secrets"): ("sanitize-git-repo", "same dclm repository secret scrub"),
    ("terminal-bench-pro", "merge-parser-branches"): (
        "merge-diff-arc-agi-task",
        "fetch /app/bundle1.bundle and bundle2.bundle into branches and merge",
    ),
    ("terminal-bench-pro", "merge-git-bundles-and-implement-transform"): (
        "merge-diff-arc-agi-task",
        "fetch git bundles into branches and merge, then implement a transform",
    ),
    ("terminal-bench-pro", "recover-corrupted-sqlite-data"): (
        "db-wal-recovery",
        "recover missing SQLite records to recovered.json, same output format",
    ),
    ("terminal-bench-pro", "schedule-multi-team-kickoff-meeting"): (
        "constraints-scheduling",
        "meeting slot for three parties from three ICS calendars",
    ),
    ("terminal-bench-pro", "prove-nat-mult-commutativity-in-coq"): (
        "prove-plus-comm",
        "complete a Coq commutativity proof on nat",
    ),
    ("terminal-bench-pro", "xrd-two-peak-fitting"): ("raman-fitting", "fit two spectral peaks, write parameters"),
    ("terminal-bench-pro", "lorentzian-fit-fluorescence-spectra"): (
        "raman-fitting",
        "fit Lorentzian peaks, write /app/results.json",
    ),
    ("terminal-bench-pro", "fluorescence-peak-fitting-pipeline"): (
        "raman-fitting",
        "peak fitting of spectra (topic twin of lorentzian-fit-fluorescence-spectra)",
    ),
    ("terminal-bench-pro", "linear-sem-causal-discovery-intervention"): (
        "bn-fit-modify",
        "recover a DAG from samples to /app/learned_dag.csv, then intervene",
    ),
    ("terminal-bench-pro", "extract-binary-symbol-table"): (
        "extract-elf",
        "extract data from a compiled binary to JSON, 75% coverage bar",
    ),
    ("terminal-bench-pro", "present-known-plaintext-key-recovery"): (
        "feal-linear-cryptanalysis",
        "known-plaintext key recovery on a block cipher, then decrypt ciphertexts",
    ),
    ("terminal-bench-pro", "implement-mitm-attack-for-24bit-double-cipher"): (
        "feal-linear-cryptanalysis",
        "known-plaintext key recovery from pairs.txt, then decrypt ciphertexts",
    ),
    ("terminal-bench-pro", "recover-stream-cipher-key"): (
        "feal-linear-cryptanalysis",
        "known-plaintext key recovery for a custom cipher",
    ),
    ("terminal-bench-pro", "implement-tensor-parallel-matmul"): (
        "torch-tensor-parallelism",
        "column-split tensor-parallel linear layer in PyTorch",
    ),
    ("terminal-bench-pro", "compile-postgresql-with-sanitizers"): (
        "sqlite-with-gcov",
        "compile a database from a vendored tarball with instrumentation, put it on PATH",
    ),
    ("terminal-bench-pro", "build-grpc-user-profile-service"): (
        "kv-store-grpc",
        "Python grpc server with grpcio 1.73.0 (larger variant of the same task)",
    ),
    ("terminal-bench-pro", "email-and-timestamp-regex"): (
        "regex-log",
        "one regex over log lines saved to /app/regex.txt",
    ),
    ("terminal-bench-pro", "regex-bitcoin-p2pkh-extraction"): ("regex-log", "one regex extracting items from logs"),
}

# Hand-checked: candidate task -> Terminal-Bench 2.0 training task it closely resembles (must not be held out).
TRAIN_NEAR = {
    ("harbor-index-1.0", "tb-dna-insert"): "dna-insert",
    ("openthoughts-tblite", "log-summary"): "log-summary-date-ranges",
    ("terminal-bench-pro", "summarize-api-log-status-metrics"): "log-summary-date-ranges",
    ("terminal-bench-pro", "select-best-english-embedding-model"): "mteb-leaderboard",
    ("terminal-bench-pro", "configure-apache-logging-and-rate-limit"): "nginx-request-logging",
    ("terminal-bench-pro", "sparql-asian-senior-researchers"): "sparql-university",
    ("terminal-bench-pro", "recover-git-history-secrets"): "git-leak-recovery",
}

# Training-side tasks a bash-only model cannot do (read by hand on 2026-10-04 from the pinned hub versions): the
# instruction asks the model to look at an image. Jeff-Code gives the model only a bash tool, which cannot show it one.
BASH_ONLY_UNABLE = {
    ("harbor-index-1.0", "hle-dirac-fermion-tunneling"): "needs to see an image (/app/image.png); Jeff-Code gives the model only bash",
    ("harbor-index-1.0", "hle-identify-city-from-photo"): "needs to see an image (/app/image.png); Jeff-Code gives the model only bash",
    ("harbor-index-1.0", "hle-identify-ingvar-runestone"): "needs to see an image (/app/image.png); Jeff-Code gives the model only bash",
    ("harbor-index-1.0", "hle-name-alkaloid-compound"): "needs to see an image (/app/image.gif); Jeff-Code gives the model only bash",
    ("harbor-index-1.0", "hle-vowel-marking-system"): "needs to see an image (/app/image.jpg); Jeff-Code gives the model only bash",
    ("terminal-bench", "cad-model"): "needs to see an image (the 2D schematic /app/schematic.png); Jeff-Code gives the model only bash",
}

# Harbor hub package of each dataset folder, pinned to the version the folders were downloaded from on 2026-10-04
# (the "latest" tag then; `harbor version list <org>/<name> --json`): (hub name, revision, content digest).
HUB = {
    "aider-polyglot": ("aider/aider-polyglot", 1, "sha256:01e28d85e46beae5b7e29a29f57cb49d882b5486583d52cec4ee5bf3540a1c84"),
    "harbor-index-1.0": ("harbor-index/harbor-index-1.0", 1, "sha256:9d4514cb93f6fafd9cf8ff352c784495ab675176c7f09671db523bd19b663584"),
    "openthoughts-tblite": ("openthoughts/openthoughts-tblite", 1, "sha256:4eb34ffc5540dff4f8a0ea00f12d46db88def7c99ec2b5ac79564625df7ecddb"),
    "swe-bench-verified": ("swe-bench/swe-bench-verified", 2, "sha256:b934b0cc3dc800fe945eaf9f1623329db97ee3133c706d20644524c7759fb341"),
    "terminal-bench": ("terminal-bench/terminal-bench", 4, "sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732"),
    "terminal-bench-2": ("terminal-bench/terminal-bench-2", 1, "sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b"),
    "terminal-bench-2-1": ("terminal-bench/terminal-bench-2-1", 6, "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"),
    "terminal-bench-pro": ("terminal-bench-pro/terminal-bench-pro", 1, "sha256:1e0df73902c63a708b75e5a4b633eaa21524583904fdadaf937203280f037461"),
    "terminal-bench-science": ("terminal-bench-science/terminal-bench-science", 10, "sha256:91531bf50016a7c64f6cc60794a17c64c6b2c14858a8ae0de39ca16f2abd611a"),
}

# The Terminal-Bench 2.0 hub copy names this folder with '-' instead of '.'.
TB2_HUB_NAME = {"install-windows-3.11": "install-windows-3-11"}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory", type=Path)
    ap.add_argument("similarity", type=Path)
    ap.add_argument("topic", type=Path)
    ap.add_argument("training_tasks", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inv = json.loads(args.inventory.read_text())
    sim = json.loads(args.similarity.read_text())
    topic = json.loads(args.topic.read_text())
    frozen = json.loads(args.training_tasks.read_text())
    by_key = {(r["dataset"], r["task"]): r for r in inv}

    tb2_train = set(frozen["training"])
    tb2_eval = set(frozen["excluded_evaluation"])
    tb2_twins = dict(frozen["excluded_leak_twins"])  # twin -> evaluation task
    tb2_eval_all = tb2_eval | set(tb2_twins)

    def tb2_node(name: str) -> tuple[str, str]:
        return ("tb2", name)

    uf = UnionFind()
    for key in by_key:
        uf.find(key)
    # The Terminal-Bench 2.0 tasks are one node per task name, shared by the eval copy, the hub 2.0 and 2.1 copies.
    for ds in TB2_SETS:
        for name in tb2_train | tb2_eval_all:
            uf.union((ds, TB2_HUB_NAME.get(name, name) if ds == "terminal-bench-2" else name), tb2_node(name))
    for name in tb2_eval_all:
        uf.union(("tb2-eval", name), tb2_node(name))

    reasons_link: dict = defaultdict(list)
    for p in sim["pairs"]:
        if p["near_duplicate"]:
            a, b = tuple(p["a"]), tuple(p["b"])
            uf.union(a, b)
    for m in sim["name_matches"]:
        if m["exact"]:
            uf.union(tuple(m["a"]), tuple(m["b"]))
    for p in topic["pair_hits"]:
        if p["cosine"] >= TOPIC_LINK:
            uf.union(tuple(p["a"]), tuple(p["b"]))
    for r in inv:
        if r["dataset"] == "aider-polyglot":
            exercise = re.sub(r"^polyglot_[a-z0-9]+_", "", r["task"])
            uf.union((r["dataset"], r["task"]), ("aider-exercise", exercise))
        if r["dataset"] == "harbor-index-1.0":
            family = r["task"].split("-", 1)[0]
            if family != "tb":
                uf.union((r["dataset"], r["task"]), ("harbor-index-family", family))
    for key, (ev, why) in EVAL_NEAR.items():
        if ev not in tb2_eval_all:
            raise ValueError(f"EVAL_NEAR names {ev}, which is not a frozen evaluation task or twin")
        uf.union(key, tb2_node(ev))
        reasons_link[key].append(f"near evaluation task {ev}: {why}")
    for key, tr in TRAIN_NEAR.items():
        if tr not in tb2_train:
            raise ValueError(f"TRAIN_NEAR names {tr}, which is not a Terminal-Bench 2.0 training task")
        uf.union(key, tb2_node(tr))

    groups: dict = defaultdict(set)
    for node in list(uf.parent):
        groups[uf.find(node)].add(node)

    def group_of(key) -> set:
        return groups[uf.find(key)]

    def eval_members(g: set) -> list[str]:
        return sorted(n[1] for n in g if n[0] == "tb2" and n[1] in tb2_eval_all)

    def train_members(g: set) -> list[str]:
        return sorted(n[1] for n in g if n[0] == "tb2" and n[1] in tb2_train)

    side: dict = {}
    forced_reason: dict = {}
    # Rule 2: Terminal-Bench 2.0 / 2.1 keep the frozen split.
    for ds in TB2_SETS:
        for r in inv:
            if r["dataset"] != ds:
                continue
            name = "install-windows-3.11" if r["task"] == "install-windows-3-11" else r["task"]
            if name in tb2_train:
                side[(ds, r["task"])] = "training"
            elif name in tb2_eval_all:
                side[(ds, r["task"])] = "held_out" if name in tb2_eval else "twin"
            else:
                raise ValueError(f"{ds}/{r['task']} is in neither list of training-tasks.json")

    candidates = [k for k in by_key if k[0] not in TB2_SETS]
    root_side: dict = {}
    for k in candidates:
        g = group_of(k)
        root = uf.find(k)
        has_train = bool(train_members(g))
        has_whole = any(n[0] in WHOLE_HELD_OUT for n in g if n in by_key)
        if has_train:
            # A whole-held-out task in such a group is excluded below (it cannot be held out, and its dataset is
            # not used for training).
            root_side[root] = "training"
            forced_reason[root] = "same group as Terminal-Bench 2.0 training task(s) " + ", ".join(train_members(g))
        elif has_whole:
            root_side[root] = "held_out"
            whole = sorted({n[0] for n in g if n in by_key and n[0] in WHOLE_HELD_OUT})
            if any(n[0] not in WHOLE_HELD_OUT for n in g if n in by_key):
                forced_reason[root] = "same group as a task of " + ", ".join(whole) + " (kept whole as held-out)"

    # Rule 5: seeded draw of the remaining groups, per dataset targets.
    target = {
        ds: round(HELD_OUT_FRACTION * sum(1 for k in candidates if k[0] == ds)) for ds in SPLIT_DATASETS
    }
    held = defaultdict(int)
    for root, s in root_side.items():
        if s == "held_out":
            for n in groups[root]:
                if n in by_key and n[0] in SPLIT_DATASETS:
                    held[n[0]] += 1
    free_roots = sorted({uf.find(k) for k in candidates if uf.find(k) not in root_side}, key=str)
    rng = random.Random(SEED)
    rng.shuffle(free_roots)
    for root in free_roots:
        members = [n for n in groups[root] if n in by_key and n[0] in SPLIT_DATASETS]
        counts = defaultdict(int)
        for n in members:
            counts[n[0]] += 1
        fits = all(held[ds] + c <= target[ds] for ds, c in counts.items())
        root_side[root] = "held_out" if fits else "training"
        if fits:
            for ds, c in counts.items():
                held[ds] += c
    for k in candidates:
        side[k] = root_side[uf.find(k)]

    # Rule 6: exclusions.
    excluded: dict = {}
    for k in candidates:
        r = by_key[k]
        g = group_of(k)
        if r["mcp_servers"]:
            excluded[k] = "needs MCP tools (Jeff-Code gives the model only bash)"
            continue
        if k[0] in WHOLE_HELD_OUT and train_members(g):
            excluded[k] = (
                "resembles Terminal-Bench 2.0 training task(s) " + ", ".join(train_members(g))
                + ", so it cannot be held out; its dataset is kept whole as held-out, so it is not trained on either"
            )
            continue
        if side[k] == "training":
            ev = eval_members(g)
            if ev:
                why = "; ".join(reasons_link.get(k, [])) or "same near-duplicate group as evaluation task(s) " + ", ".join(ev)
                excluded[k] = why
            elif r["gpus"]:
                excluded[k] = f"needs {r['gpus']} GPU ({', '.join(r['gpu_types'] or ['any'])})"
            elif k in BASH_ONLY_UNABLE:
                excluded[k] = BASH_ONLY_UNABLE[k]

    out: dict = {
        "seed": SEED,
        "held_out_fraction": HELD_OUT_FRACTION,
        "topic_link_cosine": TOPIC_LINK,
        "frozen_split": str(args.training_tasks),
        "datasets": {},
    }
    for ds in sorted({k[0] for k in by_key}):
        keys = sorted(k for k in by_key if k[0] == ds)
        entry: dict = {
            "source": f"harbor download (hub dataset folder {ds})",
            "hub": {"name": HUB[ds][0], "ref": HUB[ds][2], "revision": HUB[ds][1]},
            "policy": WHOLE_HELD_OUT.get(ds)
            or ("frozen Terminal-Bench 2.0 split" if ds in TB2_SETS else f"seeded group split, held-out target {target.get(ds)}"),
            "training": [],
            "held_out": [],
            "excluded": {},
            "forced_side": {},
        }
        for k in keys:
            t = k[1]
            if ds in TB2_SETS and side[k] == "twin":
                entry["excluded"][t] = "evaluation twin (training-tasks.json excluded_leak_twins)"
            elif k in excluded:
                entry["excluded"][t] = excluded[k]
                if side[k] == "held_out":
                    entry["excluded"][t] += " (was on the held-out side)"
            else:
                entry[side[k]].append(t)
            root = uf.find(k)
            if ds not in TB2_SETS and root in forced_reason and k not in excluded:
                entry["forced_side"][t] = f"{side[k]}: {forced_reason[root]}"
        out["datasets"][ds] = entry

    # Cross-dataset near-duplicate groups.
    cross = []
    for root, g in groups.items():
        real = sorted(n for n in g if n in by_key or n[0] == "tb2")
        sets = {n[0] for n in real}
        # Skip groups that are only one Terminal-Bench 2.0 task and its 2.0 / 2.1 hub copies.
        if len(sets) > 1 and (sets - {"tb2", *TB2_SETS} or len({n[1] for n in real if n[0] == "tb2"}) > 1):
            cross.append(sorted(f"{n[0]}/{n[1]}" for n in real))
    out["cross_dataset_groups"] = sorted(cross)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for ds, e in out["datasets"].items():
        print(f"{ds:24s} train {len(e['training']):4d}  held-out {len(e['held_out']):4d}  excluded {len(e['excluded']):3d}")


if __name__ == "__main__":
    main()
