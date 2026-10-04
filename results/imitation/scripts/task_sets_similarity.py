"""Pairwise task-text similarity across Harbor task sets, plus task-name matches.

Same two text metrics as results/imitation/leak-check-ukisai.md:
  - character 5-gram Jaccard similarity over the full instruction text (near-duplicate at >= 0.5)
  - difflib.SequenceMatcher.ratio() over the first 2000 characters of each (near-duplicate at >= 0.8)
Every pair with Jaccard >= 0.3 or ratio >= 0.6 is written out, so pairs just under the thresholds can be read by hand.
difflib is only run where its cheap upper bound (quick_ratio) reaches 0.6; below that the exact ratio cannot reach 0.6.
It is also skipped when the 5-gram Jaccard of the two 2000-character prefixes is below PREFIX_JACCARD_GATE (0.15):
quick_ratio passes for most pairs of English texts, and the full difflib on them made the second survey (2,135 added
tasks, 5 million pairs) take hours. A difflib ratio of 0.6 on the prefixes needs long shared runs of characters,
which give a prefix 5-gram Jaccard far above 0.15. The per-task maximum against the evaluation tasks (eval_max) is
still exact.

Task names: each task's folder name is normalised (lower case, '_' -> '-', known source prefixes such as 'tb-' or
'swebenchverified-fix-' removed). A name match is equality or containment in either direction (names of at least
6 characters only, so short names do not match everything).

Inputs:
  DL      directory of `harbor download --export` output (DL/<dataset>/<task>/instruction.md)
  EVAL    directory of the frozen evaluation instructions (EVAL/<task>.md), treated as dataset 'tb2-eval'
Output: JSON with "pairs" (text), "name_matches", and "eval_max" (per task, the highest Jaccard and difflib ratio
against any frozen evaluation task, computed exactly for every pair).

Usage: python3 task_sets_similarity.py DL EVAL --out similarity.json [--new DATASET ...]
With --new, only pairs that involve at least one task of the named datasets are scored (used when datasets are added
to an existing survey: pairs among the earlier datasets were scored before and are not repeated).
With --skip-internal, pairs inside each named dataset are not scored (datasets kept whole on one side whose tasks
share one long instruction template, e.g. ORCA-bench; scoring them costs hours and cannot change the split).
Multi-step tasks (no top-level instruction.md) use their step instructions joined (task_sets_inventory.read_instruction).
"""

import argparse
import difflib
import json
import re
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

from task_sets_inventory import read_instruction

JACCARD_NEAR = 0.5
RATIO_NEAR = 0.8
JACCARD_REVIEW = 0.3
RATIO_REVIEW = 0.6

NAME_PREFIXES = [
    "tb-",
    "swebenchverified-fix-",
    "swebenchverified-",
    "swebenchpro-fix-",
    "swebenchpro-",
    "polyglot-",
]

PREFIX_JACCARD_GATE = 0.15

TEXTS: list[str] = []
GRAMS: list[frozenset] = []
PREFIX_GRAMS: list[frozenset] = []


def norm_name(name: str) -> str:
    n = name.lower().replace("_", "-")
    for p in NAME_PREFIXES:
        if n.startswith(p):
            n = n[len(p) :]
            break
    return n


def grams(text: str) -> frozenset:
    return frozenset(text[i : i + 5] for i in range(max(0, len(text) - 4)))


def ratio_both(a: str, b: str) -> float:
    """difflib ratio on the first 2000 characters, the larger of both argument orders.

    SequenceMatcher is not symmetric (its junk heuristic depends on the second text), so both orders are tried.
    """
    a, b = a[:2000], b[:2000]
    return max(difflib.SequenceMatcher(None, a, b).ratio(), difflib.SequenceMatcher(None, b, a).ratio())


def pair_scores(ij: tuple[int, int]) -> tuple[int, int, float, float] | None:
    i, j = ij
    a, b = GRAMS[i], GRAMS[j]
    union = len(a | b)
    jac = len(a & b) / union if union else 0.0
    ratio = 0.0
    pa, pb = PREFIX_GRAMS[i], PREFIX_GRAMS[j]
    pu = len(pa | pb)
    if pu and len(pa & pb) / pu < PREFIX_JACCARD_GATE:
        return (i, j, jac, ratio) if jac >= JACCARD_REVIEW else None
    sm = difflib.SequenceMatcher(None, TEXTS[i][:2000], TEXTS[j][:2000])
    if sm.real_quick_ratio() >= RATIO_REVIEW and sm.quick_ratio() >= RATIO_REVIEW:
        ratio = ratio_both(TEXTS[i], TEXTS[j])
    if jac >= JACCARD_REVIEW or ratio >= RATIO_REVIEW:
        return i, j, jac, ratio
    return None


def full_scores(ij: tuple[int, int]) -> tuple[int, int, float, float]:
    """Exact Jaccard and difflib ratio, no shortcuts (used for the per-task maximum against the evaluation set)."""
    i, j = ij
    a, b = GRAMS[i], GRAMS[j]
    union = len(a | b)
    jac = len(a & b) / union if union else 0.0
    return i, j, jac, ratio_both(TEXTS[i], TEXTS[j])


def init(texts: list[str]) -> None:
    global TEXTS, GRAMS, PREFIX_GRAMS
    TEXTS = texts
    GRAMS = [grams(t) for t in texts]
    PREFIX_GRAMS = [grams(t[:2000]) for t in texts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dl", type=Path)
    ap.add_argument("eval", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--new", nargs="+", help="score only pairs involving these datasets")
    ap.add_argument("--skip-internal", nargs="+", default=[], help="do not score pairs inside these datasets")
    args = ap.parse_args()

    ids: list[tuple[str, str]] = []
    texts: list[str] = []
    for f in sorted(args.eval.glob("*.md")):
        ids.append(("tb2-eval", f.stem))
        texts.append(f.read_text(errors="replace").strip())
    for ds in sorted(p for p in args.dl.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in ds.iterdir() if p.is_dir()):
            ids.append((ds.name, task_dir.name))
            texts.append(read_instruction(task_dir).strip())
    print(f"{len(ids)} tasks, {len(ids) * (len(ids) - 1) // 2} pairs", file=sys.stderr)

    all_pairs = list(combinations(range(len(ids)), 2))
    if args.new:
        new = set(args.new)
        missing = new - {d for d, _ in ids}
        if missing:
            raise ValueError(f"--new names datasets not found under {args.dl}: {sorted(missing)}")
        all_pairs = [(i, j) for i, j in all_pairs if ids[i][0] in new or ids[j][0] in new]
    skip = set(args.skip_internal)
    all_pairs = [(i, j) for i, j in all_pairs if not (ids[i][0] == ids[j][0] and ids[i][0] in skip)]
    pairs = []
    with Pool(initializer=init, initargs=(texts,)) as pool:
        for n, res in enumerate(pool.imap_unordered(pair_scores, all_pairs, chunksize=2000)):
            if res:
                i, j, jac, ratio = res
                pairs.append(
                    {
                        "a": list(ids[i]),
                        "b": list(ids[j]),
                        "jaccard5": round(jac, 4),
                        "difflib2000": round(ratio, 4),
                        "near_duplicate": jac >= JACCARD_NEAR or ratio >= RATIO_NEAR,
                    }
                )
            if n % 200000 == 0:
                print(f"  {n}", file=sys.stderr)

    # Per task: the highest score against any frozen evaluation task (or twin), as in leak-check-ukisai.md.
    eval_idx = [i for i, d in enumerate(ids) if d[0] == "tb2-eval"]
    other_idx = [i for i, d in enumerate(ids) if d[0] != "tb2-eval" and (not args.new or d[0] in args.new)]
    best: dict = {}
    with Pool(initializer=init, initargs=(texts,)) as pool:
        work = [(o, e) for o in other_idx for e in eval_idx]
        for o, e, jac, ratio in pool.imap_unordered(full_scores, work, chunksize=500):
            b = best.setdefault(o, {"jaccard5": (0.0, ""), "difflib2000": (0.0, "")})
            if jac > b["jaccard5"][0]:
                b["jaccard5"] = (jac, ids[e][1])
            if ratio > b["difflib2000"][0]:
                b["difflib2000"] = (ratio, ids[e][1])
    eval_max = [
        {
            "dataset": ids[o][0],
            "task": ids[o][1],
            "max_jaccard5": round(best[o]["jaccard5"][0], 4),
            "max_jaccard5_vs": best[o]["jaccard5"][1],
            "max_difflib2000": round(best[o]["difflib2000"][0], 4),
            "max_difflib2000_vs": best[o]["difflib2000"][1],
        }
        for o in other_idx
    ]

    name_matches = []
    normed = [norm_name(t) for _, t in ids]
    for i, j in all_pairs:
        if ids[i][0] == ids[j][0]:
            continue
        a, b = normed[i], normed[j]
        if a == b or (min(len(a), len(b)) >= 6 and (a in b or b in a)):
            name_matches.append({"a": list(ids[i]), "b": list(ids[j]), "exact": a == b})

    args.out.write_text(json.dumps({"pairs": pairs, "name_matches": name_matches, "eval_max": eval_max}, indent=1))
    print(f"{len(pairs)} pairs above review band, {len(name_matches)} name matches", file=sys.stderr)


if __name__ == "__main__":
    main()
