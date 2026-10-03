import { appendFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { JsonObject, StopReason } from "@earendil-works/pi-ai";
import type { Pick } from "./chooser.ts";
import type { ArgumentOption, ToolOption } from "./lists.ts";
import type { MenuMatch, MenuOption, MenuToolCall } from "./menu.ts";
import type { JeffState } from "./state.ts";

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

/** One level of a decision: the options shown, who chose, each option's share, the teacher's picks, the winner. */
export interface LevelRecord {
	options: Array<ToolOption | ArgumentOption>;
	chooser: string;
	shares: Record<string, number>;
	picks: Pick[];
	chosen: string;
	/** Word joiners put into the teacher's text to get past the GLM endpoint's firewall (see chooser.ts). */
	word_joiners_inserted: number;
}

/** One line per decision in teacher mode: Jeff's place, taken by the teacher. */
export interface DecisionRecord {
	schema: "jeff-first-trace/2";
	kind: "decision";
	task_id: string;
	session_id: string;
	decision: number;
	/** Scout steps already taken since the large model's last turn. */
	step_in_stint: number;
	mode: "teacher";
	/** The large model driving the session (its model id). */
	driver: string;
	time: string;
	state: JeffState;
	check_command_notes: string[];
	/** null when the step cap handed over without asking. */
	tool_level: LevelRecord | null;
	argument_level: LevelRecord | null;
	action: { kind: "step"; tool_call: MenuToolCall } | { kind: "hand_over"; why: "chosen" | "cap" };
	timings_ms: { lists: number; chooser: number };
}

/** One line per large-model turn in teacher mode. */
export interface ModelTurnRecord {
	schema: "jeff-first-trace/2";
	kind: "model_turn";
	task_id: string;
	session_id: string;
	turn: number;
	mode: "teacher";
	driver: string;
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
