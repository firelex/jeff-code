# Stage 1 conversion statistics (ukisai/Qwen3.8-27B-multi-turn-agent-sft)

Converter: jeff-pi `tools/jeff-first/imitation/` after fix wave 2 (commits f1b194894 and b68716996: menu code;
adad96753: converter). Source: the dataset's parquet, keeping the latest episode of each trial. Leak check:
`leak-check-ukisai.md` (0 sessions dropped).
Rows: `/private/tmp/claude-501/imitation/stage1-rows.jsonl` (stage 1, quality "approximate": menus rebuilt
from the transcript by the TypeScript list code over a virtual file system).

## Sessions

- Episode rows in the dataset: 15,209; sessions (latest episode per trial): 14,415.
- Converted: 14,415 (every session; none raised). Sessions with no row at all: 135.
- By task family: nl2bash 7,866 sessions / 73,407 rows, inferredbugs 6,549 sessions / 65,418 rows.

End reasons (where the conversion of a session stops):

| End reason | Sessions | Meaning |
|---|---|---|
| complete | 14,248 | the whole transcript was converted |
| interactive | 155 | a turn sends keystrokes to a running program (q, an editor, C-c other than an interrupt); the session ends before it |
| unparsed_final_reply | 9 | the transcript's last reply cannot be read in either format; nothing ran after it |
| folder_unknown | 3 | a command gathers information in a folder that cannot be known |

## Rows and decisions

- Rows: 138,825; decisions (decision points with a row): 87,872.
- Rows by level and page: argument page 1 50,856, argument page 2 41, argument page 3 5, tool page 1 87,872, tool page 2 46, tool page 3 5.
- Hand-over share of decisions: 42.1% (36,970).
- Interrupts (a reply starting with C-c while a command ran; the session went on): 453.

Labels at the tool level (one per decision; "show_more" rows are paging):

| Label | Tool rows | Share of decisions |
|---|---|---|
| hand_over | 36,970 | 42.1% |
| list | 20,156 | 22.9% |
| find | 14,903 | 17.0% |
| read | 12,337 | 14.0% |
| toolchain | 2,242 | 2.6% |
| peek | 607 | 0.7% |
| search | 376 | 0.4% |
| run | 178 | 0.2% |
| install | 63 | 0.1% |
| show_more | 51 | - |
| repeat | 38 | 0.0% |
| service | 1 | 0.0% |
| docs | 1 | 0.0% |

Argument rows by tool (the option chosen within the tool): list 20,156, find 14,903, read 12,337, toolchain 2,242, peek 607, search 376, run 178, install 63, repeat 38, service 1, docs 1.

Labelled steps by tool (every step labelled with an option, stints included): list 20,156, find 14,903, read 12,337, toolchain 2,242, peek 607, search 376, run 178, install 63, repeat 38, service 1, docs 1.

Notes on the wave 2 labels:

- Of the find labels, 14,398 are "Find the files under FOLDER" (a find of the coding model that starts in that
  folder, whatever its filters) and 505 "Find files matching NAME".
- Check is never labelled: the rebuilt menus almost never offer it (no Check option in any of the 16,850 tool
  rows of a 4,000-session sample): check-commands.ts finds no test command named in the task and no revealed
  test configuration (Makefile, package.json, pytest settings, Cargo.toml, go.mod) in the working folder. In
  the same sample only 1 hand-over point started with a test command (`mvn test`).
- Run labels are scripts written in an earlier step and run now (mostly `bash run_*.sh` in /workspace); Install
  labels are mostly apt installs of git and Java and pip installs.

## Information turns that got no row

In approximate data a point whose next command only gathers information that no option on the rebuilt menu
matches is dropped (no row), instead of being labelled "hand over".

- Labelled information steps: 50,623; dropped information turns: 28,504.
- Labelled share of information turns: 64.0%.

Before and after fix wave 2 (wave 1: commit eaa00852a, same 14,415 sessions):

| | Wave 1 | Wave 2 |
|---|---|---|
| Decisions | 67,627 | 87,872 |
| Hand-over share of decisions | 51.3% | 42.1% |
| Labelled information steps | 32,940 | 50,623 |
| Dropped information turns | 37,026 | 28,504 |
| Labelled share of information turns | 47.1% | 64.0% |
| Sessions ending at interactive keystrokes | 576 | 155 |
| Labelled steps by tool | read 14,446, list 14,398, toolchain 2,413, peek 677, find 511, search 473, repeat 20, service 1, docs 1 | list 20,156, find 14,903, read 12,337, toolchain 2,242, peek 607, search 376, run 178, install 63, repeat 38, service 1, docs 1 |
| Drop reasons | find 41.6%, list 26.8%, search 12.6%, read 9.2%, peek 8.8%, service 0.5%, toolchain 0.4%, docs 0.1% | find 29.6%, list 21.0%, search 18.8%, read 16.8%, peek 12.2%, service 0.7%, toolchain 0.7%, docs 0.2% |

| Kind of the dropped command | Turns | Share |
|---|---|---|
| find | 8,438 | 29.6% |
| list | 5,976 | 21.0% |
| search | 5,345 | 18.8% |
| read | 4,802 | 16.8% |
| peek | 3,482 | 12.2% |
| service | 202 | 0.7% |
| toolchain | 198 | 0.7% |
| docs | 61 | 0.2% |

## Checks

- Exact duplicate rows within a session (same level, page, state, options and label): 0 (must be 0).
- Labelled steps taken from a command that comes after an acting command of the same turn: 121 (allowed since wave 2: the acting command before them matched a Run or Install option; a check on 4,000 sessions found 0 labels after an unmatched acting command).
- Conversion time: 442 s.
