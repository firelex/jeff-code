import { appendFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { JsonObject, StopReason } from "@earendil-works/pi-ai";
import type { Pick } from "./chooser.ts";
import type { RunApproval } from "./config.ts";
import type { ArgumentOption, ToolKind, ToolOption } from "./lists.ts";
import type { MenuMatch, MenuOption, MenuToolCall } from "./menu.ts";
import type { JeffState } from "./state.ts";
import type { ShownOption } from "./teacher-prompt.ts";

/** One line per large-model turn in phase 0 (shadow mode). Field names are snake_case for the Python tools that read the traces. */
export interface ShadowRecord {
	schema: "jeff-first-trace/1";
	task_id: string;
	session_id: string;
	turn: number;
	mode: "shadow";
	time: string;
	state: JeffState;
	menu: MenuOption[];
	check_command_notes: string[];
	/** Jeff's probabilities per option id; always null in shadow mode. */
	jeff: null;
	actor: "model";
	action: {
		stop_reason: StopReason;
		error_message: string | null;
		text_chars: number;
		tool_calls: Array<{ name: string; arguments: JsonObject; match: MenuMatch }>;
	};
	model_usage: { input: number; output: number; cache_read: number; cache_write: number };
	timings_ms: { menu: number; model: number; jeff: null };
}

/** One question asked in a decision: the options shown, who chose, each option's share, the teacher's picks, the winner. */
export interface LevelRecord {
	level: "tool" | "argument";
	page: number;
	/** The chosen tool, for argument pages; null on tool pages. */
	tool: string | null;
	options: Array<ShownOption | ArgumentOption>;
	chooser: string;
	shares: Record<string, number>;
	picks: Pick[];
	chosen: string;
}

/** One line per decision in teacher mode: Jeff's place, taken by the teacher. */
export interface DecisionRecord {
	schema: "jeff-first-trace/3";
	kind: "decision";
	task_id: string;
	session_id: string;
	decision: number;
	/** Scout steps already taken since the large model's last turn. */
	step_in_stint: number;
	mode: "teacher";
	/** The large model driving the session (its model id). */
	driver: string;
	driver_build: string;
	run_approval: RunApproval;
	time: string;
	state: JeffState;
	check_command_notes: string[];
	/** Every question asked in this decision, in order; empty when the step cap handed over without asking. */
	levels: LevelRecord[];
	action: { kind: "step"; tool_call: MenuToolCall } | { kind: "hand_over"; why: "chosen" | "none_of_these" | "cap" };
	timings_ms: { lists: number; chooser: number };
}

/** One line per large-model turn in teacher mode. */
export interface ModelTurnRecord {
	schema: "jeff-first-trace/3";
	kind: "model_turn";
	task_id: string;
	session_id: string;
	turn: number;
	mode: "teacher";
	driver: string;
	driver_build: string;
	time: string;
	action: {
		stop_reason: StopReason;
		error_message: string | null;
		text_chars: number;
		tool_calls: Array<{ name: string; arguments: JsonObject }>;
	};
	model_usage: { input: number; output: number; cache_read: number; cache_write: number };
	timings_ms: { model: number };
}

/**
 * The schema of record-mode traces. "jeff-first-trace/4": one "record" line before each of Qwen's turns.
 * "jeff-first-trace/5" adds a "record_step" line after each step of a turn that another call of the same turn follows
 * (the lists inside a stint, see record.ts); "record" lines are unchanged.
 */
export const TRACE_SCHEMA = "jeff-first-trace/5";

/**
 * One line per turn of plain Qwen in record mode: the scout's full option lists at that moment (built by code, no
 * scout step taken), so a converter can later label the point with Qwen's actual next action, or "hand over".
 */
export interface RecordRecord {
	schema: typeof TRACE_SCHEMA;
	kind: "record";
	task_id: string;
	session_id: string;
	turn: number;
	mode: "record";
	/** The large model driving the session (its model id). */
	driver: string;
	driver_build: string;
	run_approval: RunApproval;
	time: string;
	state: JeffState;
	check_command_notes: string[];
	lists: { tools: ToolOption[]; arguments_by_tool: Partial<Record<ToolKind, ArgumentOption[]>> };
	action: {
		stop_reason: StopReason;
		error_message: string | null;
		text_chars: number;
		tool_calls: Array<{ name: string; arguments: JsonObject }>;
	};
	model_usage: { input: number; output: number; cache_read: number; cache_write: number };
	timings_ms: { lists: number; model: number };
}

/**
 * One line in record mode after step `step` (1-based) of turn `turn`, when another of the turn's `calls_in_turn` calls
 * follows: the lists built from the disk right after that step ran, with the turn's steps so far (1 to `step`) credited
 * to the scout in `state` (byScout true), as a live stint shows the scout's own steps. `command` is the step's shell
 * command as `state` shows it. The point it describes is the one before call `step` + 1.
 */
export interface RecordStepRecord {
	schema: typeof TRACE_SCHEMA;
	kind: "record_step";
	task_id: string;
	session_id: string;
	turn: number;
	step: number;
	calls_in_turn: number;
	command: string;
	mode: "record";
	driver: string;
	driver_build: string;
	run_approval: RunApproval;
	time: string;
	state: JeffState;
	check_command_notes: string[];
	lists: { tools: ToolOption[]; arguments_by_tool: Partial<Record<ToolKind, ArgumentOption[]>> };
	timings_ms: { lists: number };
}

export type TraceRecord = ShadowRecord | DecisionRecord | ModelTurnRecord | RecordRecord | RecordStepRecord;

export class TraceWriter {
	private readonly path: string;

	constructor(path: string) {
		const folder = dirname(path);
		if (!existsSync(folder)) throw new Error(`JeffFirst: the trace folder ${folder} does not exist`);
		this.path = path;
	}

	append(record: TraceRecord): void {
		appendFileSync(this.path, `${JSON.stringify(record)}\n`);
	}
}
