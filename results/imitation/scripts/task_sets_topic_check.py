"""Topic-level similarity between tasks: word TF-IDF cosine similarity of the instruction texts.

The character 5-gram Jaccard and difflib checks (task_sets_similarity.py) catch copied or lightly edited text.
They miss a task that poses the same problem in new words; for example the known evaluation twin pair
feal-linear-cryptanalysis / feal-differential-cryptanalysis scores only 0.35 Jaccard. This script produces two
lists for a person to read:

1. For every reference task (the frozen evaluation tasks and their twins, and the Terminal-Bench 2.0 training tasks),
   the --top candidate tasks with the most similar vocabulary, if their cosine is >= --min.
2. Every pair of candidate tasks (across all candidate datasets, and within a dataset) with cosine >= --pairs-min,
   except pairs inside aider-polyglot (grouped by exercise name instead) and inside swe-bench-verified (kept whole
   as held-out, see task_sets_split.py).

TF-IDF: each word's count is weighted (1 + log count) * log(N / number of tasks containing the word), the vector is
normalised to length 1, and cosine similarity is the dot product.

Usage:
  python3 task_sets_topic_check.py DL --ref eval=EVAL_DIR --ref tb2-train=TRAIN_DIR \
      --top 4 --min 0.2 --pairs-min 0.45 --out topic.json
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-z][a-z0-9]+")
SKIP_DATASETS = {"terminal-bench-2", "terminal-bench-2-1"}
NO_INTERNAL_PAIRS = {"aider-polyglot", "swe-bench-verified"}


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 2]


def dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dl", type=Path)
    ap.add_argument("--ref", action="append", required=True, help="LABEL=DIR of <task>.md files")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--min", type=float, default=0.2)
    ap.add_argument("--pairs-min", type=float, default=0.45)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    docs: list[tuple[str, str, Counter]] = []
    for spec in args.ref:
        label, d = spec.split("=", 1)
        for f in sorted(Path(d).glob("*.md")):
            docs.append((label, f.stem, Counter(tokens(f.read_text(errors="replace")))))
    ref_labels = {s.split("=", 1)[0] for s in args.ref}
    for ds in sorted(p for p in args.dl.iterdir() if p.is_dir() and p.name not in SKIP_DATASETS):
        for t in sorted(p for p in ds.iterdir() if p.is_dir()):
            docs.append((ds.name, t.name, Counter(tokens((t / "instruction.md").read_text(errors="replace")))))
    df = Counter(w for _, _, c in docs for w in c)
    n = len(docs)
    vecs = []
    for _, _, c in docs:
        v = {w: (1 + math.log(k)) * math.log(n / df[w]) for w, k in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})

    ref_idx = [i for i, d in enumerate(docs) if d[0] in ref_labels]
    cand_idx = [i for i, d in enumerate(docs) if d[0] not in ref_labels]

    ref_hits = []
    for r in ref_idx:
        scored = sorted(((dot(vecs[r], vecs[c]), c) for c in cand_idx), reverse=True)
        for s, c in scored[: args.top]:
            if s >= args.min:
                ref_hits.append(
                    {
                        "ref_set": docs[r][0],
                        "ref_task": docs[r][1],
                        "dataset": docs[c][0],
                        "task": docs[c][1],
                        "cosine": round(s, 3),
                    }
                )

    pair_hits = []
    for x, i in enumerate(cand_idx):
        for j in cand_idx[x + 1 :]:
            if docs[i][0] == docs[j][0] and docs[i][0] in NO_INTERNAL_PAIRS:
                continue
            s = dot(vecs[i], vecs[j])
            if s >= args.pairs_min:
                pair_hits.append({"a": [docs[i][0], docs[i][1]], "b": [docs[j][0], docs[j][1]], "cosine": round(s, 3)})
    pair_hits.sort(key=lambda p: -p["cosine"])
    args.out.write_text(json.dumps({"ref_hits": ref_hits, "pair_hits": pair_hits}, indent=1))
    for h in ref_hits:
        print(f"REF  {h['cosine']:.3f}  {h['ref_set']}:{h['ref_task']:32s} {h['dataset']}:{h['task']}")
    for p in pair_hits:
        print(f"PAIR {p['cosine']:.3f}  {p['a'][0]}:{p['a'][1]:40s} {p['b'][0]}:{p['b'][1]}")


if __name__ == "__main__":
    main()
