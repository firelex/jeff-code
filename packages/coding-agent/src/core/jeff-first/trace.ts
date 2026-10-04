import { appendFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { JsonObject, StopReason } from "@earendil-works/pi-ai";
import type { Pick } from "./chooser.ts";
import type { RunApproval } from "./config.ts";
import type { JeffCannotFit, JeffCut } from "./jeff-service.ts";
import type { ArgumentOption, ToolKind, ToolOption } from "./lists.ts";
import type { CallComparison, FailedCommands, StuckOutputs } from "./loop-guard.ts";
import type { MenuMatch, MenuOption, MenuToolCall } from "./menu.ts";
import type { JeffState } from "./state.ts";
import type { ShownOption } from "./teacher-prompt.ts";
import type { QwenThinkingLevel } from "./thinking.ts";

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
	/** Jeff only: how its question was cut to fit its token limit (jeff-service.ts); null when it fit as it was. */
	jeff_cut: JeffCut | null;
	/** Jeff only: set when Jeff abstained because the question cannot be cut to fit (the scout handed over). */
	jeff_abstained: JeffCannotFit | null;
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
	/** teacher: the teacher model chose; jeff: the trained Jeff chose. */
	mode: "teacher" | "jeff";
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
	mode: "teacher" | "jeff";
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

	append(record: TraceRecord | QwenRequestRecord): void {
		appendFileSync(this.path, `${JSON.stringify(record)}\n`);
	}
}

/** Why the thinking control discarded a reply and asked again (see loop-guard.ts and runaway.ts). */
export type GuardTrigger =
	| {
			trigger: "loop";
			/** The tool calls the reply repeated. */
			repeated_action: Array<{ name: string; arguments: JsonObject }>;
			/** 1 when the reply repeated Qwen's latest action, 2 for the one before, up to 6. */
			turns_back: number;
			/** The lowest Dice similarity over the call pairs (1 when every pair is equal). */
			similarity: number;
			/** Each call of the reply compared with the earlier action's call at the same position: which rule decided
			 * ("equal" or "dice" for a match), the Dice similarity and the two texts' lengths (the reply's first). */
			pairs: CallComparison[];
			/** Writes after the repeated action that did not count, because they wrote what the file already held. */
			unchanged_writes: Array<{ name: string; paths: string[]; turns_back: number }>;
	  }
	| {
			trigger: "runaway";
			/** Which part of the reply ran away; the generation was stopped there. */
			where: "thinking" | "text";
			rule: "repeated_piece" | "repeated_line";
			/** The piece or line that repeated, and how often. */
			repeated: string;
			count: number;
	  };

/**
 * Schema "jeff-first-trace/6" adds this line kind; the other kinds keep their schema. One line per request to the
 * coding model (Qwen) in teacher and record modes, written before that turn's model_turn or record line: the thinking
 * level the router (or the guard) chose, what pi sent for it, and what came back. A turn has a second request
 * (attempt 2, always at "xhigh") only when the first reply was discarded: it repeated one of Qwen's last 6 actions at
 * thinking "off" or "low" (guard.trigger "loop", see loop-guard.ts), or it ran away (guard.trigger "runaway"). A discarded reply never enters the
 * session. A guard on a kept attempt-1 line means the user aborted the request while the guard fired.
 */
export interface QwenRequestRecord {
	schema: "jeff-first-trace/6";
	kind: "qwen_request";
	task_id: string;
	session_id: string;
	/** Counts the session's Qwen turns from 1, as the model_turn and record lines do. */
	turn: number;
	attempt: 1 | 2;
	driver: string;
	/** The router's name, for example "fixed:medium" or "jeff:jeff-router". */
	router: string;
	/** The trained router's probability per level asked on attempt 1 (four levels for jeff:, off and xhigh for
	 * jeff-off-unless:); null for a fixed router and on attempt 2. */
	router_probabilities: Partial<Record<QwenThinkingLevel, number>> | null;
	/** How the trained router's question was cut to fit Jeff's token limit, on attempt 1; null when it fit, for a fixed
	 * router and on attempt 2. */
	router_cut: JeffCut | null;
	/** Set on attempt 1 when the trained router abstained because its question cannot be cut to fit (the level is then
	 * xhigh); else null. */
	router_abstained: JeffCannotFit | null;
	/** The level the router chose, on attempt 1 (it differs from thinking_level when forced_xhigh is set); null on
	 * attempt 2. */
	router_level: QwenThinkingLevel | null;
	/** Set on attempt 1 when the loop guard made this turn run at xhigh whatever the router chose: Qwen's last 3 tool
	 * outputs were near-identical (stuck_outputs, with each pair's similarity and lengths), or its last 2 shell
	 * commands failed (failed_commands); else null. */
	forced_xhigh: Array<StuckOutputs | FailedCommands> | null;
	thinking_level: QwenThinkingLevel;
	/** chat_template_kwargs as sent (format qwen-chat-template); null where the request had no such field. */
	sent: { enable_thinking: boolean | null; reasoning_effort: string | null };
	/** "kept": the reply went to the session; "discarded": asked again; "turn_ended": the re-ask ran away too. */
	outcome: "kept" | "discarded" | "turn_ended";
	guard: GuardTrigger | null;
	stop_reason: StopReason;
	error_message: string | null;
	/** Characters of thinking in the reply (up to the stop, for a runaway). */
	thinking_chars: number;
	/** thinking_tokens is null when the server reported no separate reasoning count; output includes them. */
	usage: { input: number; output: number; thinking_tokens: number | null; cache_read: number; cache_write: number };
	/** JEFF_FIRST_THINKING_LIMIT: thinking tokens per reply before it is cut; null for no limit. */
	thinking_limit: number | null;
	/** Set when this reply's thinking reached the limit and was cut and continued; else null. */
	limit_cut: LimitCut | null;
	/** model: both requests together when the reply was cut (see limit_cut for each); router: the time the router took
	 * to choose (attempt 1 only; null on attempt 2). */
	timings_ms: { model: number; router: number | null };
}

/**
 * A reply whose thinking reached the thinking limit: the server's output-token count and the characters of the
 * thinking at the cut, how the continuation request ended ("tool_call"; "no_tool_call" and "error" are passed on like
 * any reply without a tool call; "runaway" is re-asked), its output tokens, and the time of both requests.
 */
export interface LimitCut {
	thinking_tokens: number;
	thinking_chars: number;
	continuation: {
		outcome: "tool_call" | "no_tool_call" | "error" | "runaway";
		stop_reason: StopReason;
		error_message: string | null;
		output_tokens: number;
	};
	timings_ms: { thinking: number; continuation: number };
}
