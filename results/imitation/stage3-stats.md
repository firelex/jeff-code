# Stage 3 statistics (own record-mode sessions, replayed)

Build `jeff-pi-scout-4abde3ece.tgz` only. Each session was replayed in its task's Docker image without a model (tools/jeff-first/imitation/pi_replay.py): rows before a coding-model turn use the menu logged live; rows inside a turn (stints) use the menu built in the container after the previous step ran. Quality "exact-replayed".

## Sessions

- Replayed: 369; trials failed: 6.
- Excluded (before-turn menus differ from the logged ones in more than 10% of turns): 23 (by task: caffe-cifar-10 1, dna-insert 1, financial-document-processor 1, fix-ocaml-gc 1, largest-eigenval 1, mailman 1, mteb-leaderboard 3, nginx-request-logging 1, reshard-c4-data 3, rstan-to-pystan 6, sparql-university 4).

## Rows and decisions

- Rows: 14047; decisions: 13093; hand-over share of decisions: 93.3%.
- Stint decisions (inside a turn, after a scout step): 240; stint rows: 326.
- Argument rows by tool: read 593, list 155, install 31, toolchain 31, peek 29, run 20, docs 12, service 4, search 2.

Labels at the tool level (one per decision):

| Label | Decisions | Share | Of them inside a turn |
|---|---|---|---|
| hand_over | 12216 | 93.3% | 157 |
| read | 593 | 4.5% | 55 |
| list | 155 | 1.2% | 13 |
| install | 31 | 0.2% | 0 |
| toolchain | 31 | 0.2% | 5 |
| peek | 29 | 0.2% | 10 |
| run | 20 | 0.2% | 0 |
| docs | 12 | 0.1% | 0 |
| service | 4 | 0.0% | 0 |
| search | 2 | 0.0% | 0 |

## Fidelity

- Before-turn menus built in the container equal to the logged ones: 13784 of 14355 (96.0%); in the sessions kept: 12801 of 12853.
- Where they differ (menu parts, counted per differing menu): find 290, read 264, peek 261, tools 23, list 12, repeat 4.
- Commands replayed: 15664; with a recorded output to compare: 15664. Outputs that differ from pi's recorded output (whitespace ignored): 3040 (19.4%); still differing with digits ignored: 1514 (9.7%). Information commands: 665 of 5416 differ. Stopped at a time limit: 36.

Examples of differing before-turn menus (what the container's menu adds or lacks):

- largest-eigenval session 01a104e9 turn 124: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named generate_umath.py"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`
- largest-eigenval session 01a104e9 turn 125: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named generate_umath.py"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`
- largest-eigenval session 01a104e9 turn 126: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named generate_umath.py"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`
- largest-eigenval session 01a104e9 turn 127: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named nep-0015-merge-multiarray-umath.rst"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`
- largest-eigenval session 01a104e9 turn 128: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named _umath_linalg.c"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`
- largest-eigenval session 01a104e9 turn 129: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named umath_linalg.c"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`
- largest-eigenval session 01a104e9 turn 130: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named umath_linalg.c"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`
- largest-eigenval session 01a104e9 turn 131: `{"find": {"added": ["Find files named pi-bash-edd06d9d38c2040c.log"], "removed": ["Find files named umath_linalg.c"]}, "peek": {"added": [], "removed": ["Look at the data in /tmp/pi-bash-edd06d9d38c2040c.log"]}}`

## By machine and task

Rows by machine: qwen3.8-27b-nvfp4@rtx-pro-6000 1772, qwen3.8-27b-fp8@casdgx01-gpu2 1682, qwen3.8-27b-fp8@casdgx01-gpu1 1649, qwen3.8-27b-fp8@casdgx01-gpu5 1558, qwen3.8-27b-fp8@casdgx01-gpu7 1460, qwen3.8-27b-fp8@casdgx01-gpu3 1382, qwen3.8-27b-fp8@casdgx01-gpu4 1309, qwen3.8-27b-fp8@casdgx01-gpu0 1294, qwen3.8-27b-fp8@casdgx01-gpu6 1176, qwen3.8-27b-nvfp4@spark-worker 488, qwen3.8-27b-nvfp4@spark-head 277.

Sessions kept by task: adaptive-rejection-sampler 11, build-pmars 11, caffe-cifar-10 5, cancel-async-tasks 9, chess-best-move 9, circuit-fibsqrt 8, cobol-modernization 8, configure-git-webserver 8, distribution-search 8, dna-assembly 8, dna-insert 7, financial-document-processor 8, fix-ocaml-gc 9, gcode-to-text 9, git-leak-recovery 9, gpt2-codegolf 9, headless-terminal 9, largest-eigenval 8, log-summary-date-ranges 10, mailman 8, model-extraction-relu-logits 9, modernize-scientific-stack 9, mteb-leaderboard 6, mteb-retrieve 9, nginx-request-logging 9, openssl-selfsigned-cert 10, overfull-hbox 9, path-tracing 10, path-tracing-reverse 8, polyglot-c-py 6, polyglot-rust-c 7, protein-assembly 7, pytorch-model-cli 7, query-optimize 7, regex-chess 8, reshard-c4-data 5, rstan-to-pystan 2, schemelike-metacircular-eval 8, sparql-university 4, sqlite-db-truncate 8, tune-mjcf 9, winning-avg-corewars 9, write-compressor 9.

## Failed trials

- /home/mstrasser/stage3replay/runs/datigator/rtx/round-1/caffe-cifar-10-20261004-015910/caffe-cifar-10__ahw9MTh: trial /home/mstrasser/stage3replay/runs/datigator/rtx/round-1/caffe-cifar-10-20261004-015910/caffe-cifar-10__ahw9MTh: RuntimeError: docker exec -i -u did not return within 1800 s
- /home/mstrasser/stage3replay/runs/datigator/spark-head/round-1/caffe-cifar-10-20261003-234134/caffe-cifar-10__6XKk3oX: trial /home/mstrasser/stage3replay/runs/datigator/spark-head/round-1/caffe-cifar-10-20261003-234134/caffe-cifar-10__6XKk3oX: RuntimeError: docker exec -i -u did not return within 1800 s
- /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu2/round-1/caffe-cifar-10-20261004-044433/caffe-cifar-10__P8wL8iq: trial /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu2/round-1/caffe-cifar-10-20261004-044433/caffe-cifar-10__P8wL8iq: RuntimeError: docker exec -i -u did not return within 1800 s
- /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu5/round-1/overfull-hbox-20261004-050518/overfull-hbox__BXasL4S: trial /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu5/round-1/overfull-hbox-20261004-050518/overfull-hbox__BXasL4S: ValueError: the folder a part of the command 'd=/tmp/dbg3; cd $d && grep -n "cmr/m/n/10" main.log | head -5; echo ===; sed -n "$(grep -n \'hbox(11.04317\' main.log | he' runs in i
- /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu0/round-2/caffe-cifar-10-20261004-060954/caffe-cifar-10__9c7RzjB: trial /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu0/round-2/caffe-cifar-10-20261004-060954/caffe-cifar-10__9c7RzjB: RuntimeError: docker exec -i -u did not return within 1800 s
- /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu0/round-1/caffe-cifar-10-20261003-232830/caffe-cifar-10__6Rh4Z4i: trial /home/mstrasser/stage3replay/runs/casdgx01/h100-gpu0/round-1/caffe-cifar-10-20261003-232830/caffe-cifar-10__6Rh4Z4i: RuntimeError: docker exec -i -u did not return within 300 s
## How it was run

- 375 trials (17 without a trace skipped), casdgx01, 60 at a time. First pass: every trial. Second pass
  (`--think-waits`, later fidelity fixes): every trial with a differing before-turn menu or a failure; per trial the
  replay that matched more logged menus is kept. First pass alone: 39 sessions excluded, 94% of menus equal.
- Six trials stayed in the first pass because their second pass was stopped unfinished (3 mteb-leaderboard,
  path-tracing__MUNeQUs, fix-ocaml-gc__5FsJSeg, caffe-cifar-10__6Rh4Z4i).
- Failed trials: five caffe-cifar-10 (a menu build in the replay did not return; Qwen had started `make all -j224` in
  the 1-CPU, 2 GB container) and overfull-hbox__BXasL4S (labels.py cannot know the folder of
  `d=/tmp/dbg3; cd $d && ...`; the record-mode conversion stage3.py raises on it too).

## Compared with the record-mode conversion

The record-mode conversion of the same snapshot (stage3.py, which labels only each turn's first point): 374
sessions, 15,491 rows, 14,503 decisions, hand over 93.8% (overfull-hbox__BXasL4S left out). At the 9,991 before-turn
points both conversions share (first pass), the labels are identical.

## Export

`/private/tmp/claude-501/imitation/export/stage3/` (splits `results/imitation/splits.json`; lengths with
`imitation/token_lengths.py`, Qwen3.5-0.8B processor, layout live-last, max length 8,192):

| Split | Rows before the length filter | Rows kept | Tasks kept |
|---|---|---|---|
| train | 11,319 | 9,127 | 32 |
| development | 943 | 943 | 4 |
| temperature | 1,785 | 1,297 | 3 (build-pmars, mteb-retrieve, path-tracing) |

10 rows were over 8,192 tokens (median 3,603 in train, maximum 21,036), in 4 task groups (chess-best-move,
largest-eigenval, path-tracing-reverse, winning-avg-corewars); the whole-group filter dropped 2,680 rows of those
groups. Every file loads with `jeff.evaluate.read_rows`; no family or id is in two splits.
