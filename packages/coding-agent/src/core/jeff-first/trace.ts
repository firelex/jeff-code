import { appendFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { JsonObject, StopReason } from "@earendil-works/pi-ai";
import type { Pick } from "./chooser.ts";
import type { RunApproval } from "./config.ts";
import type { ArgumentOption } from "./lists.ts";
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

export type TraceRecord = ShadowRecord | DecisionRecord | ModelTurnRecord;

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
