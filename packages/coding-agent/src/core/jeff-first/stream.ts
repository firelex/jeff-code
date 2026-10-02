import { performance } from "node:perf_hooks";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	type Model,
} from "@earendil-works/pi-ai";
import { type CheckCommands, detectCheckCommands } from "./check-commands.ts";
import { buildMenu, type MenuOption, matchToolCall } from "./menu.ts";
import { type JeffState, trimState } from "./state.ts";
import type { ShadowRecord, TraceWriter } from "./trace.ts";
import { activeToolNames, collectSteps, taskText } from "./transcript.ts";

/** Every JeffFirst failure starts with this; agent-session.ts never retries such errors. */
export const JEFF_FIRST_ERROR_PREFIX = "JeffFirst:";

export const ZERO_USAGE: AssistantMessage["usage"] = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

export interface ShadowOptions {
	inner: StreamFn;
	cwd: string;
	taskId: string;
	trace: TraceWriter;
	/** True for the session's own agent turns; compaction and summaries pass no session id. */
	isSessionTurn: (sessionId: string | undefined) => boolean;
}

export function describeError(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function failedMessage(model: Model<Api>, errorMessage: string, usage?: AssistantMessage["usage"]): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: usage ?? ZERO_USAGE,
		stopReason: "error",
		errorMessage,
		timestamp: Date.now(),
	};
}

/** pi's stream functions must not throw; a failed turn is a stream ending in an error event. */
export function errorStream(model: Model<Api>, errorMessage: string): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	stream.push({ type: "error", reason: "error", error: failedMessage(model, errorMessage) });
	return stream;
}

/**
 * Pass the large model's events through, call `onFinal` with its final message (to log it), then end the stream.
 * If `onFinal` throws, the turn ends with a JeffFirst error instead of the model's message.
 */
export function forwardModelTurn(
	model: Model<Api>,
	inner: AssistantMessageEventStream,
	onFinal: (final: AssistantMessage) => void,
	label: string,
	wrapper: string,
): AssistantMessageEventStream {
	const outer = createAssistantMessageEventStream();
	const forward = async () => {
		let finalEvent: AssistantMessageEvent | undefined;
		for await (const event of inner) {
			if (event.type === "done" || event.type === "error") {
				finalEvent = event;
				break;
			}
			outer.push(event);
		}
		const final = await inner.result();
		try {
			onFinal(final);
		} catch (error) {
			const message = `${JEFF_FIRST_ERROR_PREFIX} could not write the trace line for ${label}: ${describeError(error)}`;
			outer.push({ type: "error", reason: "error", error: failedMessage(model, message, final.usage) });
			return;
		}
		if (finalEvent) outer.push(finalEvent);
		else outer.end(final);
	};
	forward().catch((error: unknown) => {
		const message = `${JEFF_FIRST_ERROR_PREFIX} the ${wrapper} failed on ${label}: ${describeError(error)}`;
		outer.push({ type: "error", reason: "error", error: failedMessage(model, message) });
	});
	return outer;
}

interface Prepared {
	state: JeffState;
	menu: MenuOption[];
	checks: CheckCommands;
	menuMs: number;
}

export function createShadowStreamFn(options: ShadowOptions): StreamFn {
	let checks: CheckCommands | undefined;
	// Taken on the first turn: compaction later replaces the first user message with a summary.
	let task: string | undefined;
	let turn = 0;

	return async (model, context, streamOptions) => {
		const sessionId = streamOptions?.sessionId;
		// Only agent turns, which offer tools, get a menu; summaries and bug reports pass straight through.
		if (!options.isSessionTurn(sessionId) || activeToolNames(context.messages).size === 0) {
			return options.inner(model, context, streamOptions);
		}
		turn++;
		const thisTurn = turn;

		let prepared: Prepared;
		try {
			const started = performance.now();
			task ??= taskText(context.messages);
			const steps = collectSteps(context.messages);
			checks ??= detectCheckCommands(options.cwd, task);
			const menu = buildMenu({
				cwd: options.cwd,
				task,
				steps,
				activeTools: activeToolNames(context.messages),
				checkCommands: checks.commands,
			});
			prepared = { state: trimState(task, steps), menu, checks, menuMs: performance.now() - started };
		} catch (error) {
			// Not a fallback: the turn ends here, as an error the agent loop shows and pi does not retry.
			return errorStream(
				model,
				`${JEFF_FIRST_ERROR_PREFIX} could not build the menu for turn ${thisTurn}: ${describeError(error)}`,
			);
		}

		const writeTrace = (final: AssistantMessage, modelMs: number) => {
			const toolCalls = final.content.filter((part) => part.type === "toolCall");
			const record: ShadowRecord = {
				schema: "jeff-first-trace/1",
				task_id: options.taskId,
				session_id: sessionId as string,
				turn: thisTurn,
				mode: "shadow",
				time: new Date().toISOString(),
				state: prepared.state,
				menu: prepared.menu,
				check_command_notes: prepared.checks.notes,
				jeff: null,
				actor: "model",
				action: {
					stop_reason: final.stopReason,
					error_message: final.errorMessage ?? null,
					text_chars: final.content.reduce((sum, part) => sum + (part.type === "text" ? part.text.length : 0), 0),
					tool_calls: toolCalls.map((call) => ({
						name: call.name,
						arguments: call.arguments,
						match: matchToolCall(prepared.menu, call, options.cwd),
					})),
				},
				model_usage: {
					input: final.usage.input,
					output: final.usage.output,
					cache_read: final.usage.cacheRead,
					cache_write: final.usage.cacheWrite,
				},
				timings_ms: { menu: prepared.menuMs, model: modelMs, jeff: null },
			};
			options.trace.append(record);
		};

		const modelStarted = performance.now();
		const inner = await options.inner(model, context, streamOptions);
		return forwardModelTurn(
			model,
			inner,
			(final) => writeTrace(final, performance.now() - modelStarted),
			`turn ${thisTurn}`,
			"shadow wrapper",
		);
	};
}
