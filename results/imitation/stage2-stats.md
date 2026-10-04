# Stage 2 imitation statistics (openguardrails Qwen3.8-27B Terminus-2, replayed in the task images)

Replay: `tools/jeff-first/imitation/replay.py` on casdgx01; menus built inside each task's
container with `scripts/jeff-first-menus.ts --live` (live facts: the container's disk and PATH), labels by the stage 1
walk (labels.py; stints followed, unmatched information commands dropped). Rows: `stage: 2`, `quality: "near-exact"`,
`source: "openguardrails"`, `machine`. Code at jeff-pi 399eb2a36.

## Sessions

| | count |
|---|---|
| Trajectories in the dataset | 89 |
| Skipped (task not in training-tasks.json) | 44 |
| Replayed | 45 |
| Failed | 0 |
| Batch stopped by the 5% failure rule | False |

End reasons of the replayed sessions: complete 43, interactive 2.
Replay time per session: median 190 s, longest 3569 s.

## Rows and decisions

| | count |
|---|---|
| Rows | 1,171 |
| Decisions | 906 |
| Hand over | 674 (74.4%) |
| Labelled information steps | 200 |
| Dropped information turns | 317 |
| Labelled share of information turns | 38.7% |

Rows by level and page: argument page 1: 210, argument page 2: 11, argument page 3: 11, tool page 1: 906, tool page 2: 22, tool page 3: 11.

### Decisions by label (tool kind)

| | count |
|---|---|
| hand_over | 674 (74.4%) |
| read | 119 (13.1%) |
| list | 42 (4.6%) |
| run | 27 (3.0%) |
| toolchain | 18 (2.0%) |
| peek | 18 (2.0%) |
| install | 5 (0.6%) |
| find | 2 (0.2%) |
| docs | 1 (0.1%) |

### Drops by reason (kind of the unmatched information command)

| | count |
|---|---|
| search | 79 (24.9%) |
| read | 55 (17.4%) |
| peek | 52 (16.4%) |
| toolchain | 48 (15.1%) |
| service | 34 (10.7%) |
| list | 27 (8.5%) |
| find | 14 (4.4%) |
| docs | 8 (2.5%) |

## Same sessions, stage 1 method (menus from facts rebuilt from the transcript)

| | stage 1 method | stage 2 (replay) |
|---|---|---|
| Rows | 998 | 1,171 |
| Decisions | 827 | 906 |
| Hand over | 667 | 674 |
| Labelled information steps | 128 | 200 |
| Dropped information turns | 360 | 317 |
| Labelled share of information turns | 26.2% | 38.7% |

Turns with the same sequence of labels in both (turns that have a decision in either): 738 of 807.
Stage 1 labels by tool: hand_over 667, read 59, list 41, run 27, toolchain 15, peek 9, install 5, find 3, docs 1.

## Replay fidelity

| | count |
|---|---|
| Commands replayed | 1,934 |
| ... with a transcript output to compare | 1,774 |
| ... of them differing beyond whitespace (any command) | 728 |
| Information commands replayed | 822 |
| ... with a transcript output to compare | 743 |
| ... replay_mismatch (differ beyond whitespace) | 286 (38.5%) |
| ... still differing when digits are ignored too | 274 (36.9%) |
| Commands interrupted at their time limit | 15 |

Information mismatches by kind (first that applies):

| | count |
|---|---|
| other | 84 (29.4%) |
| transcript holds more (screen text of a later command or prompt attached by the parser) | 55 (19.2%) |
| replay reports a missing program or file | 53 (18.5%) |
| ls listing (dates, sizes, entries) | 49 (17.1%) |
| replay printed nothing (file or program missing in the replay, or output only on a terminal) | 21 (7.3%) |
| digits only (times, sizes, ids, versions) | 12 (4.2%) |
| replay holds more (the screen showed part of the output) | 12 (4.2%) |

Commands interrupted at their limit, by session: caffe-cifar-10 8, circuit-fibsqrt 2, mteb-leaderboard 2, mailman 1, query-optimize 1, regex-chess 1.

## Failures

None.
