# Stage 3 conversion statistics (own record-mode sessions)

Build: `jeff-pi-scout-4abde3ece.tgz` only. Rows: stage 3, quality "exact", source "own", each with its `machine`.

## Trials

- Converted: 87 (cut: no result.json (the trial was stopped or is still running) 29).
- Skipped: no trace: the agent failed before its first turn (NonZeroAgentExitCodeError) 4, no trace yet: no result.json and no trace (still running, or stopped before its first turn) 2.
- Turns given no row: the trace has no record line for this turn 1.

## Rows and decisions

- Sessions with rows: 87; rows: 2691; decisions: 2528; hand-over share of decisions: 94.0%.
- Rows by level and page: argument page 1 144, argument page 2 8, argument page 3 1, tool page 1 2528, tool page 2 9, tool page 3 1.
- Argument rows by tool: read 92, list 38, toolchain 8, docs 6, run 4, install 3, peek 2.

Labels at the tool level (one per decision):

| Label | Decisions | Share |
|---|---|---|
| hand_over | 2375 | 93.9% |
| read | 92 | 3.6% |
| list | 38 | 1.5% |
| toolchain | 8 | 0.3% |
| docs | 6 | 0.2% |
| run | 4 | 0.2% |
| install | 3 | 0.1% |
| peek | 2 | 0.1% |

## By model format

| Format | Sessions | Rows | Decisions | Hand over | Labels |
|---|---|---|---|---|---|
| fp8 | 65 | 2127 | 1984 | 93.2% | hand_over 1850, read 86, list 30, toolchain 6, docs 6, run 3, install 2, peek 1 |
| nvfp4 | 22 | 564 | 544 | 96.5% | hand_over 525, list 8, read 6, toolchain 2, install 1, run 1, peek 1 |

## By machine

| Machine | Sessions | Rows | Decisions | Hand over | Labels |
|---|---|---|---|---|---|
| qwen3.8-27b-fp8@casdgx01-gpu0 | 7 | 225 | 216 | 95.8% | hand_over 207, read 6, list 2, toolchain 1 |
| qwen3.8-27b-fp8@casdgx01-gpu1 | 12 | 223 | 210 | 93.8% | hand_over 197, list 6, read 4, docs 1, peek 1, toolchain 1 |
| qwen3.8-27b-fp8@casdgx01-gpu2 | 8 | 263 | 252 | 95.6% | hand_over 241, read 6, list 3, install 1, toolchain 1 |
| qwen3.8-27b-fp8@casdgx01-gpu3 | 6 | 332 | 309 | 93.2% | hand_over 288, read 13, list 3, docs 3, run 2 |
| qwen3.8-27b-fp8@casdgx01-gpu4 | 9 | 210 | 203 | 96.5% | hand_over 196, list 3, read 2, run 1, toolchain 1 |
| qwen3.8-27b-fp8@casdgx01-gpu5 | 13 | 324 | 298 | 92.0% | hand_over 274, read 18, list 5, toolchain 1 |
| qwen3.8-27b-fp8@casdgx01-gpu6 | 8 | 234 | 211 | 89.6% | hand_over 189, read 12, list 6, docs 2, toolchain 1, install 1 |
| qwen3.8-27b-fp8@casdgx01-gpu7 | 2 | 316 | 285 | 90.5% | hand_over 258, read 25, list 2 |
| qwen3.8-27b-nvfp4@rtx-pro-6000 | 14 | 382 | 371 | 97.0% | hand_over 360, list 4, read 3, install 1, toolchain 1, run 1, peek 1 |
| qwen3.8-27b-nvfp4@spark-head | 4 | 69 | 65 | 95.4% | hand_over 62, read 2, toolchain 1 |
| qwen3.8-27b-nvfp4@spark-worker | 4 | 113 | 108 | 95.4% | hand_over 103, list 4, read 1 |

## Sessions by task

adaptive-rejection-sampler 2, build-pmars 2, caffe-cifar-10 2, cancel-async-tasks 2, chess-best-move 1, circuit-fibsqrt 1, cobol-modernization 2, configure-git-webserver 1, distribution-search 1, dna-assembly 1, dna-insert 1, financial-document-processor 2, fix-ocaml-gc 3, gcode-to-text 3, git-leak-recovery 3, gpt2-codegolf 2, headless-terminal 2, largest-eigenval 2, log-summary-date-ranges 2, mailman 2, model-extraction-relu-logits 1, modernize-scientific-stack 1, mteb-leaderboard 2, mteb-retrieve 2, nginx-request-logging 2, openssl-selfsigned-cert 2, overfull-hbox 2, path-tracing 2, path-tracing-reverse 2, polyglot-c-py 2, polyglot-rust-c 3, protein-assembly 3, pytorch-model-cli 3, query-optimize 2, regex-chess 3, reshard-c4-data 3, rstan-to-pystan 2, schemelike-metacircular-eval 2, sparql-university 2, sqlite-db-truncate 2, tune-mjcf 3, winning-avg-corewars 3, write-compressor 1

## Export (`/private/tmp/claude-501/imitation/export/stage3/`)

Copy of both run folders taken 2026-10-04 00:30 BST (collection still running; 29 trials cut). Task groups with a row
over 8,192 tokens are dropped whole (`export_jeff --over-length`): 1 row over the limit (largest-eigenval, 8,439
tokens), so its 42 rows are dropped (temperature split).

| Split | Rows | Tasks | Median tokens | Max tokens |
|---|---|---|---|---|
| train | 2,121 | 35 | 3,674 | 8,175 |
| development | 308 | 4 | 3,220 | 6,173 |
| temperature | 220 | 3 | 3,907 (before the drop) | 7,293 (after the drop) |

Every stage 3 prompt (2,691 rows) equals, byte for byte, the prompt teacherMessages builds from that turn's logged
state and lists (fix wave 4 check, after the record-mode attribution fix).
