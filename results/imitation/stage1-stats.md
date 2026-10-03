# Stage 1 conversion statistics (ukisai/Qwen3.8-27B-multi-turn-agent-sft)

Converter: jeff-pi `tools/jeff-first/imitation/` after fix wave 4 (labels: 57404bc06, 9cd19790e, b5a2f4a9a,
11d4a091e, 27831e7ff, a0a498d05; empty task folder: 9a6b680ae). Source: the dataset's parquet, keeping the latest episode of each trial. Leak check:
`leak-check-ukisai.md` (0 sessions dropped).
Rows: `/private/tmp/claude-501/imitation/stage1-rows.jsonl` (stage 1, quality "approximate": menus rebuilt
from the transcript by the TypeScript list code over a virtual file system).

## Sessions

- Episode rows in the dataset: 15,209; sessions (latest episode per trial): 14,415.
- Dropped before conversion, filter "empty task folder" (review finding 2): 5,695 InferredBugs sessions whose own
  commands show /app empty or missing (`terminus.empty_task_folder`: the first full listing of the working folder has no
  entries, or an error says it does not exist; by finding: empty 5,695). Ids and evidence:
  `/private/tmp/claude-501/imitation/stage1-dropped-empty-app.json`. Not applied to nl2bash, whose tasks start in an
  empty /workspace by design (they ask to create files): 283 such sessions are kept.
- Converted: 8,720 (none raised). Sessions with no row at all: 220.
- By task family: nl2bash 7,866 sessions / 66,155 rows, inferredbugs 854 sessions / 2,310 rows.

End reasons (where the conversion of a session stops):

| End reason | Sessions | Meaning |
|---|---|---|
| complete | 8,670 | the whole transcript was converted |
| interactive | 42 | a turn sends keystrokes to a running program (q, an editor, C-c other than an interrupt); the session ends before it |
| unparsed_final_reply | 7 | the transcript's last reply cannot be read in either format; nothing ran after it |
| folder_unknown | 1 | a command gathers information in a folder that cannot be known |

## Rows and decisions

- Rows: 68,465; decisions (decision points with a row): 42,663.
- Rows by level and page: argument page 1 25,771, argument page 2 14, argument page 3 1, tool page 1 42,663, tool page 2 15, tool page 3 1.
- Hand-over share of decisions: 39.6% (16,877).
- Interrupts (a reply starting with C-c while a command ran; the session went on): 236.

Labels at the tool level (one per decision; "show_more" rows are paging):

| Label | Tool rows | Share of decisions |
|---|---|---|
| hand_over | 16,877 | 39.6% |
| list | 13,796 | 32.3% |
| read | 5,972 | 14.0% |
| find | 5,335 | 12.5% |
| toolchain | 229 | 0.5% |
| peek | 215 | 0.5% |
| run | 182 | 0.4% |
| search | 28 | 0.1% |
| install | 19 | 0.0% |
| show_more | 16 | - |
| repeat | 8 | 0.0% |
| service | 1 | 0.0% |
| docs | 1 | 0.0% |

Argument rows by tool (the option chosen within the tool): list 13,796, read 5,972, find 5,335, toolchain 229, peek 215, run 182, search 28, install 19, repeat 8, service 1, docs 1.

Labelled steps by tool (every step labelled with an option, stints included): list 13,796, read 5,972, find 5,335, toolchain 229, peek 215, run 182, search 28, install 19, repeat 8, service 1, docs 1.

Notes on the labels:

- Find argument rows by option: "Find the files ..." 4,684, "Find files named ..." 629, "Find files matching ..." 22.
  "Find the files under FOLDER" matches only a find that starts in that folder and has no name or path filter
  (7623d50bd); a find for an exact name (`-name`/`-iname`, no wildcard) matches "Find files named NAME" from any
  start folder, and "Find files matching **/NAME" when NAME is a named path known to be missing.
- Check is never labelled: the rebuilt menus almost never offer it (no Check option in any of the 16,850 tool
  rows of a 4,000-session sample): check-commands.ts finds no test command named in the task and no revealed
  test configuration (Makefile, package.json, pytest settings, Cargo.toml, go.mod) in the working folder. In
  the same sample only 1 hand-over point started with a test command (`mvn test`).
- Run labels are scripts written in an earlier step and run now (mostly `bash run_*.sh` in /workspace); Install
  labels are mostly apt installs of git and Java and pip installs.

## Information turns that got no row

In approximate data a point whose next command only gathers information that no option on the rebuilt menu
matches is dropped (no row), instead of being labelled "hand over".

- Labelled information steps: 25,577; dropped information turns: 14,902.
- Labelled share of information turns: 63.2%.

Across the fix waves (wave 1: commit eaa00852a; wave 2b: wave 2 plus 7623d50bd, a find with a name or path
filter no longer matches "Find the files under"; wave 3: Find by name; wave 4: empty-/app sessions dropped,
Peek and Toolchain matched narrowly, finding 6 slips; waves 1-3 cover all 14,415 sessions, wave 4 the 8,720 kept):

| | Wave 1 | Wave 2b | Wave 3 | Wave 4 |
|---|---|---|---|---|
| Sessions converted | 14,415 | 14,415 | 14,415 | 8,720 |
| Rows | 100,589 | 117,761 | 137,442 | 68,465 |
| Decisions | 67,627 | 77,246 | 87,088 | 42,663 |
| Hand-over share of decisions | 51.3% | 47.6% | 42.4% | 39.6% |
| Labelled information steps | 32,940 | 40,193 | 49,861 | 25,577 |
| Dropped information turns | 37,026 | 34,215 | 29,485 | 14,902 |
| Labelled share of information turns | 47.1% | 54.0% | 62.8% | 63.2% |
| Sessions ending at interactive keystrokes | 576 | 155 | 155 | 42 |
| Labelled steps by tool | read 14,446, list 14,398, toolchain 2,413, peek 677, find 511, search 473, repeat 20, service 1, docs 1 | list 18,790, read 12,317, find 5,886, toolchain 2,227, peek 606, search 365, run 178, install 63, repeat 37, service 1, docs 1 | list 19,829, find 14,493, read 12,319, toolchain 2,238, peek 606, search 374, run 178, install 63, repeat 39, service 1, docs 1 | list 13,796, read 5,972, find 5,335, toolchain 229, peek 215, run 182, search 28, install 19, repeat 8, service 1, docs 1 |
| Drop reasons | find 41.6%, list 26.8%, search 12.6%, read 9.2%, peek 8.8%, service 0.5%, toolchain 0.4%, docs 0.1% | find 44.5%, list 17.3%, search 14.9%, read 11.8%, peek 10.2%, service 0.6%, toolchain 0.6%, docs 0.2% | find 33.0%, list 20.3%, search 17.9%, read 15.4%, peek 11.8%, service 0.7%, toolchain 0.7%, docs 0.2% | list 28.9%, find 23.1%, peek 22.2%, read 13.3%, search 5.7%, toolchain 5.4%, service 1.1%, docs 0.4% |

| Kind of the dropped command | Turns | Share |
|---|---|---|
| list | 4,310 | 28.9% |
| find | 3,437 | 23.1% |
| peek | 3,315 | 22.2% |
| read | 1,981 | 13.3% |
| search | 843 | 5.7% |
| toolchain | 806 | 5.4% |
| service | 157 | 1.1% |
| docs | 53 | 0.4% |

## Export (`/private/tmp/claude-501/imitation/export/stage1/`)

Task groups with a row over 8,192 tokens are dropped whole (`export_jeff --over-length`):
206 rows over the limit in 43 task groups; 215 rows dropped
(train 196, development 12, temperature 7).

| Split | Rows | Median tokens | p99 | Max | Over the limit |
|---|---|---|---|---|---|
| train | 61,573 | 928 | 4,801 | 8,118 | 0 |
| development | 3,332 | 925 | 4,478 | 8,075 | 0 |
| temperature | 3,345 | 928 | 4,933 | 8,131 | 0 |

## Checks

- Exact duplicate rows within a session (same level, page, state, options and label): 0 (must be 0).
- Labelled steps taken from a command that comes after an acting command of the same turn: 111 (allowed since wave 2: the acting command before them matched a Run or Install option; a check on 4,000 sessions found 0 labels after an unmatched acting command).
- Conversion time: 162 s.
