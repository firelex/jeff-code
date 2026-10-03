# Stage 1 conversion statistics (ukisai/Qwen3.8-27B-multi-turn-agent-sft)

Converter: jeff-pi `tools/jeff-first/imitation/` after fix wave 3 (menu code: f1b194894, b68716996, f69919341,
3b1043d36; converter: adad96753, 7623d50bd, 0ff595854). Source: the dataset's parquet, keeping the latest episode of each trial. Leak check:
`leak-check-ukisai.md` (0 sessions dropped).
Rows: `/private/tmp/claude-501/imitation/stage1-rows.jsonl` (stage 1, quality "approximate": menus rebuilt
from the transcript by the TypeScript list code over a virtual file system).

## Sessions

- Episode rows in the dataset: 15,209; sessions (latest episode per trial): 14,415.
- Converted: 14,415 (every session; none raised). Sessions with no row at all: 220.
- By task family: nl2bash 7,866 sessions / 72,449 rows, inferredbugs 6,549 sessions / 64,993 rows.

End reasons (where the conversion of a session stops):

| End reason | Sessions | Meaning |
|---|---|---|
| complete | 14,248 | the whole transcript was converted |
| interactive | 155 | a turn sends keystrokes to a running program (q, an editor, C-c other than an interrupt); the session ends before it |
| unparsed_final_reply | 9 | the transcript's last reply cannot be read in either format; nothing ran after it |
| folder_unknown | 3 | a command gathers information in a folder that cannot be known |

## Rows and decisions

- Rows: 137,442; decisions (decision points with a row): 87,088.
- Rows by level and page: argument page 1 49,945, argument page 2 179, argument page 3 17, tool page 1 87,088, tool page 2 196, tool page 3 17.
- Hand-over share of decisions: 42.4% (36,947).
- Interrupts (a reply starting with C-c while a command ran; the session went on): 453.

Labels at the tool level (one per decision; "show_more" rows are paging):

| Label | Tool rows | Share of decisions |
|---|---|---|
| hand_over | 36,947 | 42.4% |
| list | 19,829 | 22.8% |
| find | 14,493 | 16.6% |
| read | 12,319 | 14.1% |
| toolchain | 2,238 | 2.6% |
| peek | 606 | 0.7% |
| search | 374 | 0.4% |
| show_more | 213 | - |
| run | 178 | 0.2% |
| install | 63 | 0.1% |
| repeat | 39 | 0.0% |
| service | 1 | 0.0% |
| docs | 1 | 0.0% |

Argument rows by tool (the option chosen within the tool): list 19,829, find 14,493, read 12,319, toolchain 2,238, peek 606, search 374, run 178, install 63, repeat 39, service 1, docs 1.

Labelled steps by tool (every step labelled with an option, stints included): list 19,829, find 14,493, read 12,319, toolchain 2,238, peek 606, search 374, run 178, install 63, repeat 39, service 1, docs 1.

Notes on the labels:

- Find argument rows by option: "Find files matching ..." 7,924, "Find the files ..." 5,434, "Find files named ..." 1,135.
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

- Labelled information steps: 49,861; dropped information turns: 29,485.
- Labelled share of information turns: 62.8%.

Across the fix waves (wave 1: commit eaa00852a; wave 2b: wave 2 plus 7623d50bd, a find with a name or path
filter no longer matches "Find the files under"; wave 3: Find by name; same 14,415 sessions):

| | Wave 1 | Wave 2b | Wave 3 |
|---|---|---|---|
| Decisions | 67,627 | 77,246 | 87,088 |
| Hand-over share of decisions | 51.3% | 47.6% | 42.4% |
| Labelled information steps | 32,940 | 40,193 | 49,861 |
| Dropped information turns | 37,026 | 34,215 | 29,485 |
| Labelled share of information turns | 47.1% | 54.0% | 62.8% |
| Sessions ending at interactive keystrokes | 576 | 155 | 155 |
| Labelled steps by tool | read 14,446, list 14,398, toolchain 2,413, peek 677, find 511, search 473, repeat 20, service 1, docs 1 | list 18,790, read 12,317, find 5,886, toolchain 2,227, peek 606, search 365, run 178, install 63, repeat 37, service 1, docs 1 | list 19,829, find 14,493, read 12,319, toolchain 2,238, peek 606, search 374, run 178, install 63, repeat 39, service 1, docs 1 |
| Drop reasons | find 41.6%, list 26.8%, search 12.6%, read 9.2%, peek 8.8%, service 0.5%, toolchain 0.4%, docs 0.1% | find 44.5%, list 17.3%, search 14.9%, read 11.8%, peek 10.2%, service 0.6%, toolchain 0.6%, docs 0.2% | find 33.0%, list 20.3%, search 17.9%, read 15.4%, peek 11.8%, service 0.7%, toolchain 0.7%, docs 0.2% |

| Kind of the dropped command | Turns | Share |
|---|---|---|
| find | 9,736 | 33.0% |
| list | 5,975 | 20.3% |
| search | 5,290 | 17.9% |
| read | 4,546 | 15.4% |
| peek | 3,475 | 11.8% |
| service | 202 | 0.7% |
| toolchain | 200 | 0.7% |
| docs | 61 | 0.2% |

## Checks

- Exact duplicate rows within a session (same level, page, state, options and label): 0 (must be 0).
- Labelled steps taken from a command that comes after an acting command of the same turn: 121 (allowed since wave 2: the acting command before them matched a Run or Install option; a check on 4,000 sessions found 0 labels after an unmatched acting command).
- Conversion time: 487 s.
