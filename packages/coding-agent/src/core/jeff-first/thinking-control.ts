import { performance } from "node:perf_hooks";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	type Model,
	type SimpleStreamOptions,
	type ToolCall,
} from "@earendil-works/pi-ai";
import { repeatedAction } from "./loop-guard.ts";
import { findRunaway } from "./runaway.ts";
import { trimState } from "./state.ts";
import { describeError, errorStream, JEFF_FIRST_ERROR_PREFIX } from "./stream.ts";
import type { QwenThinkingLevel, ThinkingRouter } from "./thinking.ts";
import type { GuardTrigger, QwenRequestRecord, TraceWriter } from "./trace.ts";
import { activeToolNames, collectSteps, taskText } from "./transcript.ts";

/** The thinking level of a re-ask after a loop or a runaway (owner: one re-ask, for that turn only). */
export const REASK_LEVEL: QwenThinkingLevel = "xhigh";
/** Levels whose replies the loop guard checks. */
const GUARDED_LEVELS: readonly QwenThinkingLevel[] = ["off", "low"];
/** The runaway rules are checked each time a piece of thinking or text has grown by this many characters. */
const RUNAWAY_CHECK_CHARS = 64;

export interface ThinkingControlOptions {
	inner: StreamFn;
	taskId: string;
	trace: TraceWriter;
	router: ThinkingRouter;
	/** True for the session's own agent turns; compaction and summaries pass no session id. */
	isSessionTurn: (sessionId: string | undefined) => boolean;
}

/** One request to the coding model, read to its end before anything is passed on. */
interface Attempt {
	level: QwenThinkingLevel;
	events: AssistantMessageEvent[];
	final: AssistantMessage;
	runaway: Extract<GuardTrigger, { trigger: "runaway" }> | null;
	sent: QwenRequestRecord["sent"];
	ms: number;
}

/** Reads enable_thinking and reasoning_effort from the request body pi sent (chat_template_kwargs). */
function sentThinking(payload: unknown): QwenRequestRecord["sent"] {
	const kwargs =
		payload !== null && typeof payload === "object"
			? (payload as { chat_template_kwargs?: unknown }).chat_template_kwargs
			: undefined;
	if (kwargs === null || typeof kwargs !== "object") return { enable_thinking: null, reasoning_effort: null };
	const { enable_thinking, reasoning_effort } = kwargs as { enable_thinking?: unknown; reasoning_effort?: unknown };
	return {
		enable_thinking: typeof enable_thinking === "boolean" ? enable_thinking : null,
		reasoning_effort: typeof reasoning_effort === "string" ? reasoning_effort : null,
	};
}

/**
 * Sends one request at `level` and reads its events to the end, without passing them on: the reply may still be
 * discarded. Stops the generation when a piece of thinking or text runs away (see runaway.ts).
 */
async function runAttempt(
	inner: StreamFn,
	model: Model<Api>,
	context: Parameters<StreamFn>[1],
	streamOptions: SimpleStreamOptions | undefined,
	level: QwenThinkingLevel,
): Promise<Attempt> {
	const controller = new AbortController();
	const outerSignal = streamOptions?.signal;
	const abortFromOuter = () => controller.abort(outerSignal?.reason);
	if (outerSignal?.aborted) controller.abort(outerSignal.reason);
	else outerSignal?.addEventListener("abort", abortFromOuter, { once: true });

	let sent: QwenRequestRecord["sent"] = { enable_thinking: null, reasoning_effort: null };
	const outerOnPayload = streamOptions?.onPayload;
	const started = performance.now();
	try {
		const stream = await inner(model, context, {
			...streamOptions,
			reasoning: level === "off" ? undefined : level,
			signal: controller.signal,
			onPayload: async (payload, payloadModel) => {
				const replaced = await outerOnPayload?.(payload, payloadModel);
				sent = sentThinking(replaced ?? payload);
				return replaced;
			},
		});
		const events: AssistantMessageEvent[] = [];
		const pieces = new Map<number, { text: string; checkedAt: number }>();
		let runaway: Attempt["runaway"] = null;
		for await (const event of stream) {
			if (event.type === "done" || event.type === "error") break;
			events.push(event);
			if (runaway || (event.type !== "thinking_delta" && event.type !== "text_delta")) continue;
			const piece = pieces.get(event.contentIndex) ?? { text: "", checkedAt: 0 };
			piece.text += event.delta;
			pieces.set(event.contentIndex, piece);
			if (piece.text.length - piece.checkedAt < RUNAWAY_CHECK_CHARS) continue;
			piece.checkedAt = piece.text.length;
			const found = findRunaway(piece.text);
			if (!found) continue;
			runaway = {
				trigger: "runaway",
				where: event.type === "thinking_delta" ? "thinking" : "text",
				rule: found.rule,
				repeated: found.rule === "repeated_piece" ? found.piece : found.line,
				count: found.rule === "repeated_piece" ? found.repeats : found.count,
			};
			controller.abort(new Error("JeffFirst stopped a runaway generation"));
		}
		const final = await stream.result();
		return { level, events, final, runaway, sent, ms: performance.now() - started };
	} finally {
		outerSignal?.removeEventListener("abort", abortFromOuter);
	}
}

/** Passes an accepted reply on as if it had streamed now, marked with the thinking level it was generated at. */
function replay(attempt: Attempt): AssistantMessageEventStream {
	const final: AssistantMessage = { ...attempt.final, thinkingLevel: attempt.level };
	const stream = createAssistantMessageEventStream();
	for (const event of attempt.events) stream.push(event);
	if (final.stopReason === "error" || final.stopReason === "aborted") {
		stream.push({ type: "error", reason: final.stopReason, error: final });
	} else {
		stream.push({
			type: "done",
			reason: final.stopReason as "stop" | "length" | "toolUse" | "deferred",
			message: final,
		});
	}
	return stream;
}

/**
 * Thinking control for the coding model (Qwen) in teacher and record modes, wrapped around the model request:
 *
 * 1. The router chooses each request's thinking level (off, low, medium or xhigh), sent as pi's thinking level.
 * 2. Loop guard: a reply generated at "off" or "low" whose tool calls repeat one of Qwen's previous 2 actions, with
 *    no file written since, is discarded (it never enters the session) and the turn is asked again once at "xhigh".
 * 3. Runaway cut-off: a reply whose thinking or text keeps repeating itself is stopped and the turn is asked again once
 *    at "xhigh"; if that re-ask runs away too, the turn ends with a JeffFirst error (no further retries).
 *
 * Every request writes a "qwen_request" trace line (see QwenRequestRecord). Replies are read to their end before
 * they are passed on, since any reply may be discarded: the session shows a reply only once it is complete.
 */
export function createThinkingControlStreamFn(options: ThinkingControlOptions): StreamFn {
	// Taken on the first turn: compaction later replaces the first user message with a summary.
	let task: string | undefined;
	let turn = 0;

	return async (model, context, streamOptions) => {
		const sessionId = streamOptions?.sessionId;
		if (!options.isSessionTurn(sessionId) || activeToolNames(context.messages).size === 0) {
			return options.inner(model, context, streamOptions);
		}
		turn++;
		const thisTurn = turn;

		// The router's probabilities and time apply to the first request only; a re-ask always runs at REASK_LEVEL.
		let routed:
			| {
					probabilities: QwenRequestRecord["router_probabilities"];
					cut: QwenRequestRecord["router_cut"];
					abstained: QwenRequestRecord["router_abstained"];
					ms: number;
			  }
			| undefined;
		const writeLine = (
			attempt: Attempt,
			number: 1 | 2,
			outcome: QwenRequestRecord["outcome"],
			guard: GuardTrigger | null,
		) => {
			const { final } = attempt;
			options.trace.append({
				schema: "jeff-first-trace/6",
				kind: "qwen_request",
				task_id: options.taskId,
				session_id: sessionId as string,
				turn: thisTurn,
				attempt: number,
				driver: model.id,
				router: options.router.name,
				router_probabilities: number === 1 && routed ? routed.probabilities : null,
				router_cut: number === 1 && routed ? routed.cut : null,
				router_abstained: number === 1 && routed ? routed.abstained : null,
				thinking_level: attempt.level,
				sent: attempt.sent,
				outcome,
				guard,
				stop_reason: final.stopReason,
				error_message: final.errorMessage ?? null,
				thinking_chars: final.content.reduce(
					(sum, part) => sum + (part.type === "thinking" ? part.thinking.length : 0),
					0,
				),
				usage: {
					input: final.usage.input,
					output: final.usage.output,
					thinking_tokens: final.usage.reasoning ?? null,
					cache_read: final.usage.cacheRead,
					cache_write: final.usage.cacheWrite,
				},
				timings_ms: { model: attempt.ms, router: number === 1 && routed ? routed.ms : null },
			});
		};

		try {
			task ??= taskText(context.messages);
			const routerStarted = performance.now();
			const choice = await options.router.levelFor(trimState(task, collectSteps(context.messages)));
			routed = {
				probabilities: choice.probabilities,
				cut: choice.cut,
				abstained: choice.abstained,
				ms: performance.now() - routerStarted,
			};
			const level = choice.level;
			if (level !== "off" && !model.reasoning) {
				throw new Error(
					`the router ${options.router.name} chose thinking "${level}", but the model ${model.id} is not marked as able to think (model.reasoning), so pi would send no thinking`,
				);
			}

			const first = await runAttempt(options.inner, model, context, streamOptions, level);
			let trigger: GuardTrigger | null = first.runaway;
			// A request the user aborted is passed on as it is; only the guard's own stop is a runaway.
			if (!trigger && GUARDED_LEVELS.includes(level) && first.final.stopReason === "toolUse") {
				const calls = first.final.content.filter((part): part is ToolCall => part.type === "toolCall");
				const repeat = repeatedAction(context.messages, calls);
				if (repeat) {
					trigger = {
						trigger: "loop",
						repeated_action: calls.map((call) => ({ name: call.name, arguments: call.arguments })),
						turns_back: repeat.turnsBack,
					};
				}
			}
			if (!trigger || streamOptions?.signal?.aborted) {
				writeLine(first, 1, "kept", trigger);
				return replay(first);
			}
			writeLine(first, 1, "discarded", trigger);

			const second = await runAttempt(options.inner, model, context, streamOptions, REASK_LEVEL);
			if (second.runaway && !streamOptions?.signal?.aborted) {
				writeLine(second, 2, "turn_ended", second.runaway);
				return errorStream(
					model,
					`${JEFF_FIRST_ERROR_PREFIX} turn ${thisTurn} ran away again when asked again at thinking "${REASK_LEVEL}" (${second.runaway.where} repeated ${JSON.stringify(second.runaway.repeated)} ${second.runaway.count} times); the turn ends here`,
				);
			}
			writeLine(second, 2, "kept", null);
			return replay(second);
		} catch (error) {
			// Not a fallback: the turn ends here, as an error the agent loop shows and pi does not retry.
			return errorStream(
				model,
				`${JEFF_FIRST_ERROR_PREFIX} the thinking control failed on turn ${thisTurn}: ${describeError(error)}`,
			);
		}
	};
}
