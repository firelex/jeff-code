import { appendFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { AgentMessage } from "@jeffhub/jeff-code-agent-core";
import type { AssistantMessage, ImageContent, Message, TextContent, ToolResultMessage } from "@jeffhub/jeff-code-ai";
import {
	JEFF_SERVICE_POLICY,
	type JeffCannotFit,
	type JeffCut,
	type JeffQuestion,
	type JeffReply,
	JeffService,
} from "./jeff-service.ts";
import {
	availableCuts,
	parseToolOutput,
	TRIM_CHOICES,
	TRIM_MIN_LINES,
	TRIM_OPTIONS,
	type TrimChoice,
	type TrimCut,
	trimQuestion,
	trimToolOutput,
} from "./output-trim.ts";
import { JEFF_PROVIDER } from "./provider.ts";
import { type JeffState, trimState } from "./state.ts";
import { renderState } from "./teacher-prompt.ts";
import { readThreshold } from "./thinking.ts";
import { collectSteps, taskText } from "./transcript.ts";

/**
 * How JeffFirst shortens a new tool output of the coding model before it enters the session (JEFF_FIRST_OUTPUT_TRIM):
 * - off: never;
 * - fixed:<choice>: always the same choice (all, last200, last40, first40 or first20last20);
 * - jeff:<adapter>: Jeff's trimming adapter answers the trimming question (output-trim.ts); its most likely choice is
 *   taken when it is "all", or when its probability is at least the threshold; otherwise the output stays whole.
 * Outputs of TRIM_MIN_LINES lines or fewer are never shortened, and no question is asked for them.
 */
export type OutputTrimSpec =
	| { kind: "off" }
	| { kind: "fixed"; choice: TrimChoice }
	| { kind: "jeff"; url: string; adapter: string; threshold: number };

const VALUES = `off, ${TRIM_CHOICES.map((choice) => `fixed:${choice}`).join(", ")} or jeff:<trimming adapter>`;

/** Reads JEFF_FIRST_OUTPUT_TRIM, which teacher, record and jeff modes require. */
export function readOutputTrimSpec(env: NodeJS.ProcessEnv, mode: string): OutputTrimSpec {
	const value = env.JEFF_FIRST_OUTPUT_TRIM;
	if (value === undefined || value === "") {
		throw new Error(
			`JeffFirst: JEFF_FIRST_MODE=${mode} needs JEFF_FIRST_OUTPUT_TRIM, how much of each new long command output the coding model sees: ${VALUES}`,
		);
	}
	if (value === "off") return { kind: "off" };
	if (value.startsWith("jeff:")) {
		const adapter = value.slice("jeff:".length);
		if (adapter === "") throw new Error("JeffFirst: JEFF_FIRST_OUTPUT_TRIM=jeff: needs the trimming adapter's name");
		const url = env.JEFF_FIRST_JEFF_URL;
		if (!url) {
			throw new Error(
				`JeffFirst: JEFF_FIRST_OUTPUT_TRIM=${value} needs JEFF_FIRST_JEFF_URL, the Jeff service's address as the task containers reach it, for example http://192.168.2.10:8920`,
			);
		}
		const threshold = readThreshold(
			env,
			"JEFF_FIRST_JEFF_TRIM_THRESHOLD",
			"the probability from 0 to 1 the trimming adapter's most likely choice needs before an output is shortened (from the adapter's calibration)",
			mode,
		);
		return { kind: "jeff", url, adapter, threshold };
	}
	const choice = value.startsWith("fixed:") ? value.slice("fixed:".length) : undefined;
	if (choice === undefined || !TRIM_CHOICES.includes(choice as TrimChoice)) {
		throw new Error(`JeffFirst: JEFF_FIRST_OUTPUT_TRIM must be ${VALUES}, got "${value}"`);
	}
	return { kind: "fixed", choice: choice as TrimChoice };
}

/** The question Jeff's trimming adapter answers, exactly as the training rows ask it (export_trim.py). */
export function trimJeffQuestion(state: JeffState, totalLines: number): JeffQuestion {
	return {
		state: renderState(state),
		instructions: trimQuestion(totalLines),
		criteria: Object.fromEntries(TRIM_CHOICES.map((choice) => [choice, TRIM_OPTIONS[choice]])),
	};
}

/** The part of the Jeff service the trimmer uses (jeff-service.ts JeffService). */
export interface TrimJeff {
	ask(model: string, question: JeffQuestion): Promise<JeffReply>;
}

export interface TrimDecision {
	choice: TrimChoice;
	probabilities: Record<TrimChoice, number> | null;
	jeffMs: number | null;
	/** How Jeff's question was cut to fit its token limit; null when it fit, and for a fixed choice. */
	jeffCut: JeffCut | null;
	/** Set when Jeff abstained because its question cannot be cut to fit (the whole output is then kept). */
	jeffAbstained: JeffCannotFit | null;
}

export interface OutputTrimDecider {
	/** How the trace names it, for example "fixed:last40" or "jeff:jeff-trim". */
	readonly name: string;
	choose(state: JeffState, totalLines: number): Promise<TrimDecision>;
}

export function fixedTrimDecider(choice: TrimChoice): OutputTrimDecider {
	return {
		name: `fixed:${choice}`,
		choose: async () => ({ choice, probabilities: null, jeffMs: null, jeffCut: null, jeffAbstained: null }),
	};
}

export function jeffTrimDecider(jeff: TrimJeff, adapter: string, threshold: number): OutputTrimDecider {
	if (!(threshold >= 0 && threshold <= 1))
		throw new Error(`the trimming threshold must be from 0 to 1, got ${threshold}`);
	return {
		name: `jeff:${adapter}`,
		choose: async (state, totalLines) => {
			const answer = await jeff.ask(adapter, trimJeffQuestion(state, totalLines));
			if (answer.kind === "abstain") {
				return {
					choice: "all",
					probabilities: null,
					jeffMs: answer.ms,
					jeffCut: null,
					jeffAbstained: answer.cannotFit,
				};
			}
			const probabilities = Object.fromEntries(
				TRIM_CHOICES.map((choice) => [choice, answer.probabilities[choice]]),
			) as Record<TrimChoice, number>;
			let best: TrimChoice = "all";
			for (const choice of TRIM_CHOICES) if (probabilities[choice] > probabilities[best]) best = choice;
			const choice = best === "all" || probabilities[best] >= threshold ? best : "all";
			return { choice, probabilities, jeffMs: answer.ms, jeffCut: answer.cut, jeffAbstained: null };
		},
	};
}

/** The decider JEFF_FIRST_OUTPUT_TRIM names (not "off": then no trimmer is installed). */
export function createTrimDecider(spec: Exclude<OutputTrimSpec, { kind: "off" }>): OutputTrimDecider {
	if (spec.kind === "fixed") return fixedTrimDecider(spec.choice);
	return jeffTrimDecider(new JeffService(spec.url, JEFF_SERVICE_POLICY), spec.adapter, spec.threshold);
}

/**
 * Schema "jeff-first-trace/7" adds this line kind; the other kinds keep their schema. One line per new tool output of
 * the coding model longer than TRIM_MIN_LINES shown lines: the choice, Jeff's probabilities (null for a fixed choice),
 * and, when the output was shortened, the whole output as the command produced it (the session holds the shortened
 * text only).
 */
export interface OutputTrimRecord {
	schema: "jeff-first-trace/7";
	kind: "output_trim";
	task_id: string;
	session_id: string;
	tool_call_id: string;
	tool_name: string;
	trimmer: string;
	/** Lines of the output as Jeff-Code showed it, and of the command's whole output (more when Jeff-Code had cut it). */
	shown_lines: number;
	total_lines: number;
	/** The cuts that would shorten this output. */
	available: TrimCut[];
	choice: TrimChoice;
	probabilities: Record<TrimChoice, number> | null;
	/** Whether the text changed: false for "all" and for a cut that keeps every shown line. */
	shortened: boolean;
	chars: { before: number; after: number };
	/** The output as the command produced it, when shortened; else null. */
	full_output: string | null;
	/** How Jeff's question was cut to fit its token limit (jeff-service.ts); null when it fit, and for a fixed choice. */
	jeff_cut: JeffCut | null;
	/** Set when Jeff abstained because its question cannot be cut to fit (the whole output was kept); else null. */
	jeff_abstained: JeffCannotFit | null;
	timings_ms: { jeff: number | null };
}

/** What the session's tool-result hook passes for one finished tool call. */
export interface ToolResultInput {
	toolName: string;
	toolCallId: string;
	content: (TextContent | ImageContent)[];
	isError: boolean;
	/** The assistant message that made the call. */
	assistantMessage: AssistantMessage;
	/** The agent's messages at the time: the history up to and including that assistant message. */
	messages: AgentMessage[];
}

const LLM_ROLES = new Set(["user", "assistant", "toolResult", "system"]);

/**
 * The trimmer Jeff-Code's session calls for each finished tool call before the result enters the session (agent-session.ts
 * toolResultTransform). It shortens only the coding model's own calls (not the scout's), and only text results of more
 * than TRIM_MIN_LINES shown lines; it returns the new content, or undefined to keep the result as it is.
 */
export function createOutputTrimmer(options: {
	decider: OutputTrimDecider;
	taskId: string;
	traceFile: string;
	sessionId: () => string;
}): (input: ToolResultInput) => Promise<(TextContent | ImageContent)[] | undefined> {
	const folder = dirname(options.traceFile);
	if (!existsSync(folder)) throw new Error(`JeffFirst: the trace folder ${folder} does not exist`);
	// Results of the current batch: the agent adds a batch's results to its messages only after the whole batch.
	const batch = new Map<string, ToolResultMessage>();
	let batchOf: AssistantMessage | undefined;
	let task: string | undefined;
	return async (input) => {
		if (input.assistantMessage !== batchOf) {
			batch.clear();
			batchOf = input.assistantMessage;
		}
		const result: ToolResultMessage = {
			role: "toolResult",
			toolCallId: input.toolCallId,
			toolName: input.toolName,
			content: input.content,
			isError: input.isError,
			timestamp: Date.now(),
		};
		if (input.assistantMessage.provider === JEFF_PROVIDER) {
			batch.set(input.toolCallId, result);
			return undefined;
		}
		const part = input.content.length === 1 ? input.content[0] : undefined;
		const text = part?.type === "text" ? part.text : undefined;
		const parsed = text === undefined ? undefined : parseToolOutput(text);
		if (text === undefined || parsed === undefined || parsed.partialLine || parsed.lines.length <= TRIM_MIN_LINES) {
			batch.set(input.toolCallId, result);
			return undefined;
		}
		const messages = input.messages.filter((message): message is Message => LLM_ROLES.has(message.role));
		const known = new Set(messages.flatMap((m) => (m.role === "toolResult" ? [m.toolCallId] : [])));
		const earlier = [...batch.values()].filter((done) => !known.has(done.toolCallId));
		task ??= taskText(messages);
		const state = trimState(task, collectSteps([...messages, ...earlier, result]));
		const decision = await options.decider.choose(state, parsed.totalLines);
		const shortened = decision.choice === "all" ? undefined : trimToolOutput(text, decision.choice);
		const record: OutputTrimRecord = {
			schema: "jeff-first-trace/7",
			kind: "output_trim",
			task_id: options.taskId,
			session_id: options.sessionId(),
			tool_call_id: input.toolCallId,
			tool_name: input.toolName,
			trimmer: options.decider.name,
			shown_lines: parsed.lines.length,
			total_lines: parsed.totalLines,
			available: availableCuts(text),
			choice: decision.choice,
			probabilities: decision.probabilities,
			shortened: shortened !== undefined,
			chars: { before: text.length, after: (shortened ?? text).length },
			full_output: shortened === undefined ? null : text,
			jeff_cut: decision.jeffCut,
			jeff_abstained: decision.jeffAbstained,
			timings_ms: { jeff: decision.jeffMs },
		};
		appendFileSync(options.traceFile, `${JSON.stringify(record)}\n`);
		if (shortened === undefined) {
			batch.set(input.toolCallId, result);
			return undefined;
		}
		const content: TextContent[] = [{ type: "text", text: shortened }];
		batch.set(input.toolCallId, { ...result, content });
		return content;
	};
}
