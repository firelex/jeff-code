# Leak check: ukisai/Qwen3.8-27B-multi-turn-agent-sft against the 40 evaluation tasks

Stage 1 data source: `ukisai/Qwen3.8-27B-multi-turn-agent-sft` (parquet
`qwen38-agent-sft-combined-100pct.parquet`, 145.8 MB, downloaded 2026-10-03). The parquet has 15,209 rows, one per
episode; each trial's conversation grows across its episodes, so only the latest episode per `trial_name` is a full
session. That leaves **14,415 distinct sessions**.

Evaluation set: the 40 task names in `excluded_evaluation` of `results/imitation/training-tasks.json`, fetched from
`github.com/harbor-framework/terminal-bench-2` at commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`
(`instruction.md` of each task folder, via `gh api`). All 40 fetched without error (123 to 4,365 bytes each).

## Method

Two independent checks, either one drops a session:

1. **Task name**: the ukisai `task` field equals, or contains, or is contained in, one of the 40 evaluation task
   names (checked both directions as substrings).
2. **Task text similarity**: the task part of the session's first user message (between `Task Description:` and
   `Current terminal state:`) compared against each evaluation task's `instruction.md`:
   - character 5-gram Jaccard similarity (over the full task text) >= 0.5, or
   - `difflib.SequenceMatcher.ratio()` on the first 2000 characters of each >= 0.8.

   For each session the maximum of each metric across all 40 evaluation tasks was recorded (with which evaluation
   task produced it).

## Result: no leaks found

- **Task name check: 0 matches.** ukisai's `task` values are OpenThoughts-Agent identifiers (`inferredbugs-NNNN`,
  `nl2bash-NNNN-0000[_runK]`, optionally with a `__dsv4-trial-2` suffix) — a different naming scheme from
  Terminal-Bench 2.0's task slugs (`bn-fit-modify`, `crack-7z-hash`, ...). No substring containment either way.
- **Task text similarity check: 0 matches.** Across all 14,415 sessions x 40 evaluation tasks, the highest 5-gram
  Jaccard seen was **0.061** and the highest difflib ratio was **0.137** — both far below the 0.5 / 0.8 drop
  thresholds. This is expected: ukisai's tasks are InferredBugs bug-fix tasks (C#/Java/C++ source diffs) and
  NL2Bash command-synthesis tasks, topically unrelated to the Terminal-Bench 2.0 evaluation tasks (systems/research
  tasks such as `build-pmars`, `fix-ocaml-gc`, `qemu-startup`). The nonzero similarity that does appear comes from
  shared markdown boilerplate ("## Project Information", "### File-Level Changes", generic section headers), not
  from matching task content.

**Dropped sessions: 0.** No session's score reached either threshold, so no per-session drop list is needed; the
distribution below covers every session.

## Distribution of the max similarity per session (across all 40 evaluation tasks)

| Percentile | max 5-gram Jaccard | max difflib ratio (first 2000 chars) |
|---|---|---|
| min | 0.005 | 0.033 |
| p10 | 0.021 | 0.054 |
| p25 | 0.029 | 0.061 |
| p50 (median) | 0.041 | 0.072 |
| p75 | 0.044 | 0.084 |
| p90 | 0.046 | 0.088 |
| p95 | 0.047 | 0.089 |
| p99 | 0.050 | 0.093 |
| p99.9 | 0.054 | 0.111 |
| max | 0.061 | 0.137 |

Drop thresholds (0.5 Jaccard, 0.8 ratio) sit far above the observed max in both metrics, i.e. the whole distribution
is well clear of the drop boundary — not just the typical case.

Highest-scoring sessions by metric, for inspection (all far below threshold):

| Session | Max Jaccard | vs. evaluation task |
|---|---|---|
| inferredbugs-2596__4h6bTGb | 0.061 | fix-code-vulnerability |
| nl2bash-2212-0000__A8EaaY4 | 0.060 | merge-diff-arc-agi-task |
| inferredbugs-4824__AkXKt58 | 0.059 | fix-code-vulnerability |
| inferredbugs-0619__8FZc4Vs | 0.057 | fix-code-vulnerability |
| inferredbugs-2220__9oBxBUr | 0.056 | fix-code-vulnerability |

| Session | Max ratio | vs. evaluation task |
|---|---|---|
| inferredbugs-1487__ZqY7spk | 0.137 | fix-git |
| inferredbugs-0999__ywhSdC7 | 0.123 | fix-git |
| inferredbugs-2122__QTotkJr | 0.120 | fix-git |
| inferredbugs-1253__eZ6K68X | 0.119 | fix-git |
| inferredbugs-1146__ccpuzCD | 0.117 | fix-git |

## Conclusion

No session of `ukisai/Qwen3.8-27B-multi-turn-agent-sft` needs to be dropped for leakage against the 40 evaluation
tasks. All 14,415 sessions proceed to stage 1 conversion.
