"""Stage 1 conversion of the ukisai sessions (fix wave 4). Fails loud: any session that raises stops the run, and the
error names the session. Writes the rows, one record per session, and a summary with the counts for stage1-stats.md.

Filter "empty task folder" (fix wave 4, review finding 2): an InferredBugs session whose own commands show its task
folder (/app) empty or missing (terminus.empty_task_folder: the first full listing of the folder has no entries, or an
error says it does not exist) is dropped whole: the project the task is about was never copied in. Each dropped
session is written to stage1-dropped-empty-app.json with its evidence. nl2bash tasks start in an empty /workspace by
design (they ask to create files), so the filter does not apply to them; their count is reported.

Usage (from tools/jeff-first):
  uv run --with pyarrow==21.0.0 python ../../results/imitation/scripts/convert_stage1.py DATA_FOLDER [LIMIT]
DATA_FOLDER holds ukisai.parquet and leak_results.json (the leak check's output) and receives the outputs; LIMIT converts
only the first LIMIT sessions, for a trial run.
"""
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "jeff-first"))

import pyarrow.parquet as pq

from imitation import labels as L
from imitation.labels import NEUTRAL, command_acts, command_is_neutral, part_intent
from imitation.splitter import split_command
from imitation.terminus import convert_sessions, empty_task_folder, latest_episodes, session_from_dataset_row

SOURCE = "ukisai/Qwen3.8-27B-multi-turn-agent-sft"
CHUNK_SIZE = 500
JACCARD_THRESHOLD = 0.5
RATIO_THRESHOLD = 0.8

# Measurement hooks (this driver only): which command each labelled step came from, and what each dropped turn's
# command was. They wrap the labeler's methods without changing what they do.
taken: list[tuple[str, int, int, str]] = []  # (session, turn, command index, tool kind) of every labelled step
drops: list[tuple[str, int, str]] = []  # (session, turn, kind of the dropped information command)
current_session: dict[int, str] = {}
pending_names: list[str] = []
original_take = L.SessionLabeler._take
original_no_match = L.SessionLabeler._no_match
original_post_init = L.SessionLabeler.__post_init__


def _take(self, index, choice, menu):
    taken.append((current_session[id(self)], self._turn + 1, index, choice.kind))
    return original_take(self, index, choice, menu)


def _no_match(self, menu, next_commands):
    before = len(self.dropped)
    original_no_match(self, menu, next_commands)
    if len(self.dropped) > before:
        relevant = [text for text in next_commands if not command_is_neutral(text)]
        first = [result for result in (part_intent(part) for part in split_command(relevant[0])) if result is not NEUTRAL][0]
        drops.append((current_session[id(self)], self.dropped[-1], first.kind))


def _post_init(self):
    current_session[id(self)] = pending_names.pop(0)
    original_post_init(self)


L.SessionLabeler._take = _take
L.SessionLabeler._no_match = _no_match
L.SessionLabeler.__post_init__ = _post_init


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: convert_stage1.py DATA_FOLDER [LIMIT]")
    OUT = sys.argv[1]
    PARQUET = f"{OUT}/ukisai.parquet"
    LEAK_RESULTS = f"{OUT}/leak_results.json"
    limit = int(sys.argv[2]) if len(sys.argv) == 3 else None
    started = time.time()
    raw = pq.read_table(PARQUET).to_pylist()
    latest = latest_episodes(raw)
    leaks = json.load(open(LEAK_RESULTS))
    leaked = {trial for trial, r in leaks.items() if r["max_jaccard"] >= JACCARD_THRESHOLD or r["max_ratio"] >= RATIO_THRESHOLD}
    kept = [row for row in latest if row["trial_name"] not in leaked][:limit]
    print(f"{len(raw)} episode rows, {len(latest)} sessions, {len(leaked)} leaked, converting {len(kept)}", flush=True)
    parsed = [session_from_dataset_row(row, source=SOURCE, stage=1, quality="approximate") for row in kept]
    empty_app: dict[str, dict] = {}
    empty_workspace_kept = 0
    sessions = []
    for meta, session in parsed:
        found = empty_task_folder(session)
        if found is not None and meta.task.startswith("inferredbugs-"):
            empty_app[meta.session] = {"task": meta.task, "cwd": session.cwd, "turn": found.turn, "finding": found.finding, "command": found.command[:300]}
            continue
        empty_workspace_kept += found is not None
        sessions.append((meta, session))
    json.dump(
        {"filter": "InferredBugs session whose own listings or errors show its task folder empty or missing", "dropped": len(empty_app),
         "by_finding": dict(collections.Counter(v["finding"] for v in empty_app.values())), "sessions": empty_app},
        open(f"{OUT}/stage1-dropped-empty-app.json", "w"), indent=1,
    )
    print(f"dropped {len(empty_app)} InferredBugs sessions with an empty or missing /app; kept {empty_workspace_kept} other sessions with an empty task folder", flush=True)
    by_name = {meta.session: session for meta, session in sessions}
    print(f"parsed in {time.time() - started:.0f}s", flush=True)
    rows_out = open(f"{OUT}/stage1-rows.jsonl", "w")
    sessions_out = open(f"{OUT}/stage1-sessions.jsonl", "w")
    duplicates = 0
    total_rows = 0
    tool_labels = collections.Counter()
    argument_rows = collections.Counter()
    levels = collections.Counter()
    end_reasons = collections.Counter()
    decisions = 0
    hand_over = 0
    for start in range(0, len(sessions), CHUNK_SIZE):
        chunk = sessions[start : start + CHUNK_SIZE]
        pending_names.extend(meta.session for meta, _ in chunk)
        conversion = convert_sessions(chunk)
        seen: dict[str, set] = collections.defaultdict(set)
        for row in conversion.rows:
            key = (row.level, row.page, row.state, json.dumps(row.options), row.label)
            if key in seen[row.session]:
                duplicates += 1
            seen[row.session].add(key)
            levels[(row.level, row.page)] += 1
            if row.level == "tool":
                tool_labels[row.label] += 1
                hand_over += row.label == "hand_over"
            else:
                argument_rows[row.label.rsplit("-", 1)[0]] += 1
            rows_out.write(row.to_json() + "\n")
        total_rows += len(conversion.rows)
        for meta, _ in chunk:
            result = conversion.sessions[meta.session]
            end_reasons[result.end_reason] += 1
            decisions += result.decisions
            record = {
                "session": meta.session,
                "task": meta.task,
                "end_reason": result.end_reason,
                "end_detail": result.end_detail,
                "rows": result.rows,
                "decisions": result.decisions,
                "dropped_turns": conversion.dropped_turns[meta.session],
                "interrupts": sum(turn.interrupts for turn in by_name[meta.session].turns),
            }
            sessions_out.write(json.dumps(record) + "\n")
        print(f"{start + len(chunk)} sessions, {total_rows} rows, {time.time() - started:.0f}s", flush=True)
    rows_out.close()
    sessions_out.close()
    after_acting = []
    for session, turn, index, _ in taken:
        commands = by_name[session].turns[turn - 1].commands
        if any(command_acts(command.text) for command in commands[:index]):
            after_acting.append((session, turn, index))
    information = [entry for entry in taken if entry[3] in L.INFORMATION_KINDS]
    interrupts = sum(turn.interrupts for session in by_name.values() for turn in session.turns)
    summary = {
        "episode_rows": len(raw),
        "sessions": len(latest),
        "leaked": len(leaked),
        "dropped_empty_app": len(empty_app),
        "kept_other_empty_task_folder": empty_workspace_kept,
        "converted": len(sessions),
        "end_reasons": dict(end_reasons.most_common()),
        "rows": total_rows,
        "decisions": decisions,
        "hand_over_decisions": hand_over,
        "rows_by_level_and_page": {f"{level} page {page}": count for (level, page), count in sorted(levels.items())},
        "tool_row_labels": dict(tool_labels.most_common()),
        "argument_rows_by_tool": dict(argument_rows.most_common()),
        "labelled_steps": len(taken),
        "labelled_steps_after_an_acting_command": len(after_acting),
        "after_acting_examples": after_acting[:10],
        "labelled_information_steps": len(information),
        "labelled_share_of_information_turns": round(len(information) / (len(information) + len(drops)), 4),
        "labelled_steps_by_tool": dict(collections.Counter(kind for *_, kind in taken).most_common()),
        "interrupts": interrupts,
        "dropped_information_turns": len(drops),
        "drop_reasons": dict(collections.Counter(kind for _, _, kind in drops).most_common()),
        "duplicate_rows_within_a_session": duplicates,
        "seconds": round(time.time() - started),
    }
    json.dump(summary, open(f"{OUT}/stage1-summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
