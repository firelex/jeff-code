"""Write results/imitation/stage1-stats.md from stage1-summary.json and stage1-sessions.jsonl.

Usage: python write_stats.py DATA_FOLDER   (the folder convert_stage1.py wrote to, with the earlier waves' summaries and
the export's length files)
"""
import collections
import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: write_stats.py DATA_FOLDER")
OUT = sys.argv[1]
s = json.load(open(f"{OUT}/stage1-summary.json"))
w1 = json.load(open(f"{OUT}/stage1-summary-wave1.json"))
w2 = json.load(open(f"{OUT}/stage1-summary-wave2b.json"))
w3 = json.load(open(f"{OUT}/stage1-summary-wave3.json"))
empty = json.load(open(f"{OUT}/stage1-dropped-empty-app.json"))
exported = json.load(open(f"{OUT}/export/stage1/dropped-over-length.json"))
lengths = json.load(open(f"{OUT}/export/stage1-lengths-after-filter.json"))
find_kinds = collections.Counter()
for line in open(f"{OUT}/stage1-rows.jsonl"):
    if '"level": "argument"' not in line:
        continue
    row = json.loads(line)
    if row["label"].startswith("find-"):
        description = next(o["description"] for o in row["options"] if o["id"] == row["label"])
        find_kinds[" ".join(description.split(" ")[:3])] += 1
sessions = [json.loads(line) for line in open(f"{OUT}/stage1-sessions.jsonl")]
labels = s["tool_row_labels"]
decisions = s["decisions"]
labelled = s["labelled_information_steps"]
dropped = s["dropped_information_turns"]
families = collections.Counter(x["task"].split("-")[0] for x in sessions)
rows_by_family = collections.Counter()
for x in sessions:
    rows_by_family[x["task"].split("-")[0]] += x["rows"]
no_rows = sum(1 for x in sessions if x["rows"] == 0)


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%"


lines = [
    "# Stage 1 conversion statistics (ukisai/Qwen3.8-27B-multi-turn-agent-sft)",
    "",
    "Converter: Jeff-Code `tools/jeff-first/imitation/` after fix wave 4 (labels: 57404bc06, 9cd19790e, b5a2f4a9a,",
    "11d4a091e, 27831e7ff, a0a498d05; empty task folder: 9a6b680ae). Source: the dataset's parquet, keeping the latest episode of each trial. Leak check:",
    "`leak-check-ukisai.md` (0 sessions dropped).",
    f"Rows: `{OUT}/stage1-rows.jsonl` (stage 1, quality \"approximate\": menus rebuilt",
    "from the transcript by the TypeScript list code over a virtual file system).",
    "",
    "## Sessions",
    "",
    f"- Episode rows in the dataset: {s['episode_rows']:,}; sessions (latest episode per trial): {s['sessions']:,}.",
    f"- Dropped before conversion, filter \"empty task folder\" (review finding 2): {s['dropped_empty_app']:,} InferredBugs sessions whose own",
    "  commands show /app empty or missing (`terminus.empty_task_folder`: the first full listing of the working folder has no",
    "  entries, or an error says it does not exist; by finding: " + ", ".join(f"{k} {v:,}" for k, v in empty["by_finding"].items()) + "). Ids and evidence:",
    f"  `{OUT}/stage1-dropped-empty-app.json`. Not applied to nl2bash, whose tasks start in an",
    f"  empty /workspace by design (they ask to create files): {s['kept_other_empty_task_folder']:,} such sessions are kept.",
    f"- Converted: {s['converted']:,} (none raised). Sessions with no row at all: {no_rows:,}.",
    f"- By task family: " + ", ".join(f"{k} {v:,} sessions / {rows_by_family[k]:,} rows" for k, v in families.most_common()) + ".",
    "",
    "End reasons (where the conversion of a session stops):",
    "",
    "| End reason | Sessions | Meaning |",
    "|---|---|---|",
]
meaning = {
    "complete": "the whole transcript was converted",
    "interactive": "a turn sends keystrokes to a running program (q, an editor, C-c other than an interrupt); the session ends before it",
    "unparsed_final_reply": "the transcript's last reply cannot be read in either format; nothing ran after it",
    "folder_unknown": "a command gathers information in a folder that cannot be known",
}
for reason, count in s["end_reasons"].items():
    lines.append(f"| {reason} | {count:,} | {meaning[reason]} |")
lines += [
    "",
    "## Rows and decisions",
    "",
    f"- Rows: {s['rows']:,}; decisions (decision points with a row): {decisions:,}.",
    f"- Rows by level and page: " + ", ".join(f"{k} {v:,}" for k, v in s["rows_by_level_and_page"].items()) + ".",
    f"- Hand-over share of decisions: {pct(s['hand_over_decisions'], decisions)} ({s['hand_over_decisions']:,}).",
    f"- Interrupts (a reply starting with C-c while a command ran; the session went on): {s['interrupts']:,}.",
    "",
    "Labels at the tool level (one per decision; \"show_more\" rows are paging):",
    "",
    "| Label | Tool rows | Share of decisions |",
    "|---|---|---|",
]
for label, count in labels.items():
    lines.append(f"| {label} | {count:,} | {pct(count, decisions) if label != 'show_more' else '-'} |")
lines += [
    "",
    "Argument rows by tool (the option chosen within the tool): " + ", ".join(f"{k} {v:,}" for k, v in s["argument_rows_by_tool"].items()) + ".",
    "",
    "Labelled steps by tool (every step labelled with an option, stints included): " + ", ".join(f"{k} {v:,}" for k, v in s["labelled_steps_by_tool"].items()) + ".",
    "",
    "Notes on the labels:",
    "",
    "- Find argument rows by option: " + ", ".join(f"\"{k} ...\" {v:,}" for k, v in find_kinds.most_common()) + ".",
    "  \"Find the files under FOLDER\" matches only a find that starts in that folder and has no name or path filter",
    "  (7623d50bd); a find for an exact name (`-name`/`-iname`, no wildcard) matches \"Find files named NAME\" from any",
    "  start folder, and \"Find files matching **/NAME\" when NAME is a named path known to be missing.",
    "- Check is never labelled: the rebuilt menus almost never offer it (no Check option in any of the 16,850 tool",
    "  rows of a 4,000-session sample): check-commands.ts finds no test command named in the task and no revealed",
    "  test configuration (Makefile, package.json, pytest settings, Cargo.toml, go.mod) in the working folder. In",
    "  the same sample only 1 hand-over point started with a test command (`mvn test`).",
    "- Run labels are scripts written in an earlier step and run now (mostly `bash run_*.sh` in /workspace); Install",
    "  labels are mostly apt installs of git and Java and pip installs.",
    "",
    "## Information turns that got no row",
    "",
    "In approximate data a point whose next command only gathers information that no option on the rebuilt menu",
    "matches is dropped (no row), instead of being labelled \"hand over\".",
    "",
    f"- Labelled information steps: {labelled:,}; dropped information turns: {dropped:,}.",
    f"- Labelled share of information turns: {pct(labelled, labelled + dropped)}.",
    "",
    "Across the fix waves (wave 1: commit eaa00852a; wave 2b: wave 2 plus 7623d50bd, a find with a name or path",
    "filter no longer matches \"Find the files under\"; wave 3: Find by name; wave 4: empty-/app sessions dropped,",
    "Peek and Toolchain matched narrowly, finding 6 slips; waves 1-3 cover all 14,415 sessions, wave 4 the 8,720 kept):",
    "",
    "| | Wave 1 | Wave 2b | Wave 3 | Wave 4 |",
    "|---|---|---|---|---|",
    f"| Sessions converted | {w1['converted']:,} | {w2['converted']:,} | {w3['converted']:,} | {s['converted']:,} |",
    f"| Rows | {w1['rows']:,} | {w2['rows']:,} | {w3['rows']:,} | {s['rows']:,} |",
    f"| Decisions | {w1['decisions']:,} | {w2['decisions']:,} | {w3['decisions']:,} | {decisions:,} |",
    f"| Hand-over share of decisions | {pct(w1['hand_over_decisions'], w1['decisions'])} | {pct(w2['hand_over_decisions'], w2['decisions'])} | {pct(w3['hand_over_decisions'], w3['decisions'])} | {pct(s['hand_over_decisions'], decisions)} |",
    f"| Labelled information steps | {w1['labelled_steps']:,} | {w2['labelled_information_steps']:,} | {w3['labelled_information_steps']:,} | {labelled:,} |",
    f"| Dropped information turns | {w1['dropped_information_turns']:,} | {w2['dropped_information_turns']:,} | {w3['dropped_information_turns']:,} | {dropped:,} |",
    f"| Labelled share of information turns | {pct(w1['labelled_steps'], w1['labelled_steps'] + w1['dropped_information_turns'])} | {pct(w2['labelled_information_steps'], w2['labelled_information_steps'] + w2['dropped_information_turns'])} | {pct(w3['labelled_information_steps'], w3['labelled_information_steps'] + w3['dropped_information_turns'])} | {pct(labelled, labelled + dropped)} |",
    f"| Sessions ending at interactive keystrokes | {w1['end_reasons']['interactive']:,} | {w2['end_reasons'].get('interactive', 0):,} | {w3['end_reasons'].get('interactive', 0):,} | {s['end_reasons'].get('interactive', 0):,} |",
    "| Labelled steps by tool | " + ", ".join(f"{k} {v:,}" for k, v in w1["argument_rows_by_tool"].items()) + " | " + ", ".join(f"{k} {v:,}" for k, v in w2["labelled_steps_by_tool"].items()) + " | " + ", ".join(f"{k} {v:,}" for k, v in w3["labelled_steps_by_tool"].items()) + " | " + ", ".join(f"{k} {v:,}" for k, v in s["labelled_steps_by_tool"].items()) + " |",
    "| Drop reasons | " + ", ".join(f"{k} {pct(v, w1['dropped_information_turns'])}" for k, v in w1["drop_reasons"].items()) + " | " + ", ".join(f"{k} {pct(v, w2['dropped_information_turns'])}" for k, v in w2["drop_reasons"].items()) + " | " + ", ".join(f"{k} {pct(v, w3['dropped_information_turns'])}" for k, v in w3["drop_reasons"].items()) + " | " + ", ".join(f"{k} {pct(v, dropped)}" for k, v in s["drop_reasons"].items()) + " |",
    "",
    "| Kind of the dropped command | Turns | Share |",
    "|---|---|---|",
]
for kind, count in s["drop_reasons"].items():
    lines.append(f"| {kind} | {count:,} | {pct(count, dropped)} |")
split_lengths = {path.rsplit("/", 1)[-1].removesuffix(".jsonl"): v for path, v in lengths.items()}
lines += [
    "",
    f"## Export (`{OUT}/export/stage1/`)",
    "",
    f"Task groups with a row over {exported['max_length']:,} tokens are dropped whole (`export_jeff --over-length`):",
    f"{exported['rows_over_max_length']:,} rows over the limit in {exported['group_count']:,} task groups; {exported['rows_dropped']:,} rows dropped",
    "(" + ", ".join(f"{k} {v:,}" for k, v in exported["rows_dropped_by_split"].items()) + ").",
    "",
    "| Split | Rows | Median tokens | p99 | Max | Over the limit |",
    "|---|---|---|---|---|---|",
]
for split in ("train", "development", "temperature"):
    v = split_lengths[split]
    lines.append(f"| {split} | {exported['rows_kept_by_split'][split]:,} | {v['median']:,} | {v['p99']:,} | {v['max']:,} | {v['over_max_length']:,} |")
lines += [
    "",
    "## Checks",
    "",
    f"- Exact duplicate rows within a session (same level, page, state, options and label): {s['duplicate_rows_within_a_session']:,} (must be 0).",
    f"- Labelled steps taken from a command that comes after an acting command of the same turn: {s['labelled_steps_after_an_acting_command']:,} (allowed since wave 2: the acting command before them matched a Run or Install option; a check on 4,000 sessions found 0 labels after an unmatched acting command).",
    f"- Conversion time: {s['seconds']:,} s.",
    "",
]
open(Path(__file__).resolve().parents[1] / "stage1-stats.md", "w").write("\n".join(lines))
print("\n".join(lines))
