# Stage 1 conversion statistics (ukisai/Qwen3.8-27B-multi-turn-agent-sft)

Converter: jeff-pi `tools/jeff-first/imitation/` after fix wave 1 (commit eaa00852a). Source: the dataset's
parquet, keeping the latest episode of each trial. Leak check: `leak-check-ukisai.md` (0 sessions dropped).
Rows: `/private/tmp/claude-501/imitation/stage1-rows.jsonl` (stage 1, quality "approximate": menus rebuilt
from the transcript by the TypeScript list code over a virtual file system).

## Sessions

- Episode rows in the dataset: 15,209; sessions (latest episode per trial): 14,415.
- Converted: 14,415 (every session; none raised). Sessions with no row at all: 324.
- By task family: nl2bash 7,866 sessions / 49,663 rows, inferredbugs 6,549 sessions / 50,926 rows.

End reasons (where the conversion of a session stops):

| End reason | Sessions | Meaning |
|---|---|---|
| complete | 13,828 | the whole transcript was converted |
| interactive | 576 | a turn sends keystrokes to a running program (C-c, q, an editor); the session ends before it |
| unparsed_final_reply | 8 | the transcript's last reply cannot be read in either format; nothing ran after it |
| folder_unknown | 3 | a command gathers information in a folder that cannot be known |

## Rows and decisions

- Rows: 100,589; decisions (decision points with a row): 67,627.
- Rows by level and page: argument page 1 32,922, argument page 2 14, argument page 3 4, tool page 1 67,627, tool page 2 18, tool page 3 4.
- Hand-over share of decisions: 51.3% (34,687).

Labels at the tool level (one per decision; "show_more" rows are paging):

| Label | Tool rows | Share of decisions |
|---|---|---|
| hand_over | 34,687 | 51.3% |
| read | 14,446 | 21.4% |
| list | 14,398 | 21.3% |
| toolchain | 2,413 | 3.6% |
| peek | 677 | 1.0% |
| find | 511 | 0.8% |
| search | 473 | 0.7% |
| show_more | 22 | - |
| repeat | 20 | 0.0% |
| service | 1 | 0.0% |
| docs | 1 | 0.0% |

Argument rows by tool (the option chosen within the tool): read 14,446, list 14,398, toolchain 2,413, peek 677, find 511, search 473, repeat 20, service 1, docs 1.

## Information turns that got no row

In approximate data a point whose next command only gathers information that no option on the rebuilt menu
matches is dropped (no row), instead of being labelled "hand over".

- Labelled information steps: 32,940; dropped information turns: 37,026.
- Labelled share of information turns: 47.1%.

| Kind of the dropped command | Turns | Share |
|---|---|---|
| find | 15,408 | 41.6% |
| list | 9,924 | 26.8% |
| search | 4,663 | 12.6% |
| read | 3,415 | 9.2% |
| peek | 3,262 | 8.8% |
| service | 167 | 0.5% |
| toolchain | 139 | 0.4% |
| docs | 48 | 0.1% |

## Checks

- Exact duplicate rows within a session (same level, page, state, options and label): 0 (must be 0).
- Labelled steps taken from a command that comes after an acting command of the same turn: 0 (must be 0).
- Conversion time: 745 s.
## Notes on the dropped turns

- Measured on a fixed random sample of 1,000 sessions, the labelled share was 46.7% before fix wave 1 (958
  sessions that the old converter could read) and 47.7% after (same 958 sessions; 47.7% on all 1,000).
- Most remaining drops come from the menu rules in lists.ts, not from missing file facts: the Find option is offered
  only for a name known to be missing where it was named (Qwen often runs `find` first); List is offered only for the
  working folder and folders named as a whole token in recent outputs or the task (`/output` named only inside
  `/output/command_capture.txt` is not offered); Read needs a file known to be text (`.cs` is not in the text
  extension list, so a C# file needs its whole text seen) and named in recent outputs, the task or recent commands.
  The menu code was not changed in this wave.
- An acting command (writes, edits, runs, installs, compiles) is never a label and ends the labelled part of its turn
  (ruling 2026-10-03), so Run, Check, Install and Repeat options are practically never labels in this data.
