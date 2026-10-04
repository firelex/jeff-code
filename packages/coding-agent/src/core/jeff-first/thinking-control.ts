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
	type ThinkingContent,
	type ToolCall,
	type Usage,
} from "@earendil-works/pi-ai";
import { repeatedAction } from "./loop-guard.ts";
import { findRunaway } from "./runaway.ts";
import { trimState } from "./state.ts";
import { describeError, errorStream, JEFF_FIRST_ERROR_PREFIX } from "./stream.ts";
import type { QwenThinkingLevel, ThinkingRouter } from "./thinking.ts";
import type { GuardTrigger, LimitCut, QwenRequestRecord, TraceWriter } from "./trace.ts";
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
	/** JEFF_FIRST_THINKING_LIMIT: the most thinking tokens one reply may use before it is cut; null for no limit. */
	thinkingLimit: number | null;
}

/** The thinking of a reply cut at the limit: the text so far and the server's output-token count at the cut. */
interface ThinkingCut {
	thinking: string;
	/** The field the server streamed the thinking in (pi's thinkingSignature), kept on the joined reply. */
	signature: string | undefined;
	tokens: number;
	/** The events of the reply up to the cut (the cut thinking's own events). */
	events: AssistantMessageEvent[];
	usage: Usage;
	ms: number;
}

/** One request to the coding model, read to its end before anything is passed on. */
interface Attempt {
	level: QwenThinkingLevel;
	events: AssistantMessageEvent[];
	final: AssistantMessage;
	runaway: Extract<GuardTrigger, { trigger: "runaway" }> | null;
	sent: QwenRequestRecord["sent"];
	ms: number;
	/** Set when the reply's thinking reached the limit and was cut (the reply is then the cut joined to its
	 * continuation, see joinCut). */
	limitCut: LimitCut | null;
}

/** Stops a reply's thinking at the limit (set) or continues a reply from its cut thinking (continueFrom). */
type LimitMode = { kind: "none" } | { kind: "limit"; tokens: number } | { kind: "continue"; thinking: string };

/**
 * The request body changes the thinking limit needs, on top of pi's own body:
 * - limit: every streamed chunk carries the output tokens so far (vLLM stream_options.continuous_usage_stats), so the
 *   thinking is counted as the server counts output tokens;
 * - continue: the request ends with the cut reply as an assistant message (its thinking in the "reasoning" field, no
 *   answer yet) and asks the server to continue that message (vLLM continue_final_message, no generation prompt).
 *   The prompt is then the chat template's rendering of a reply with that thinking, up to and including the closing
 *   "</think>" (the template follows it with "\n\n" and the answer, which the model writes); a test of the live
 *   server checked this token for token (thinking-limit section of the JeffFirst report).
 */
function limitPayload(payload: unknown, mode: LimitMode): unknown {
	if (mode.kind === "none") return payload;
	if (payload === null || typeof payload !== "object") {
		throw new Error("the thinking limit needs pi's request body as an object");
	}
	const body = payload as { stream_options?: object; messages?: unknown };
	const counted = {
		...body,
		stream_options: { ...body.stream_options, include_usage: true, continuous_usage_stats: true },
	};
	if (mode.kind === "limit") return counted;
	if (!Array.isArray(body.messages)) throw new Error("the thinking limit's continuation needs pi's request messages");
	return {
		...counted,
		messages: [...body.messages, { role: "assistant", content: "", reasoning: mode.thinking }],
		continue_final_message: true,
		add_generation_prompt: false,
	};
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
	limit: LimitMode,
): Promise<{ attempt: Attempt; cut: ThinkingCut | null }> {
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
				const next = limit.kind === "none" ? replaced : limitPayload(replaced ?? payload, limit);
				sent = sentThinking(next ?? payload);
				return next;
			},
		});
		const events: AssistantMessageEvent[] = [];
		const pieces = new Map<number, { text: string; checkedAt: number }>();
		let runaway: Attempt["runaway"] = null;
		let cut: ThinkingCut | null = null;
		for await (const event of stream) {
			if (event.type === "done" || event.type === "error") break;
			if (cut) continue;
			events.push(event);
			if (runaway || (event.type !== "thinking_delta" && event.type !== "text_delta")) continue;
			const piece = pieces.get(event.contentIndex) ?? { text: "", checkedAt: 0 };
			piece.text += event.delta;
			pieces.set(event.contentIndex, piece);
			if (piece.text.length - piece.checkedAt >= RUNAWAY_CHECK_CHARS) {
				piece.checkedAt = piece.text.length;
				const found = findRunaway(piece.text);
				if (found) {
					runaway = {
						trigger: "runaway",
						where: event.type === "thinking_delta" ? "thinking" : "text",
						rule: found.rule,
						repeated: found.rule === "repeated_piece" ? found.piece : found.line,
						count: found.rule === "repeated_piece" ? found.repeats : found.count,
					};
					controller.abort(new Error("JeffFirst stopped a runaway generation"));
					continue;
				}
			}
			// The limit: checked on each piece of thinking after the runaway rules, while the reply is still all
			// thinking. The count is the server's own output-token count so far (every chunk carries it).
			if (
				limit.kind === "limit" &&
				event.type === "thinking_delta" &&
				event.partial.usage.output >= limit.tokens &&
				event.partial.content.every((part) => part.type === "thinking")
			) {
				const block = event.partial.content[event.contentIndex] as ThinkingContent;
				cut = {
					thinking: block.thinking,
					signature: block.thinkingSignature,
					tokens: event.partial.usage.output,
					events: [...events],
					usage: { ...event.partial.usage },
					ms: performance.now() - started,
				};
				controller.abort(new Error("JeffFirst stopped the thinking at its limit"));
			}
		}
		const final = await stream.result();
		return { attempt: { level, events, final, runaway, sent, ms: performance.now() - started, limitCut: null }, cut };
	} finally {
		outerSignal?.removeEventListener("abort", abortFromOuter);
	}
}

function addUsage(a: Usage, b: Usage): Usage {
	return {
		input: a.input + b.input,
		output: a.output + b.output,
		cacheRead: a.cacheRead + b.cacheRead,
		cacheWrite: a.cacheWrite + b.cacheWrite,
		reasoning: (a.reasoning ?? 0) + (b.reasoning ?? 0),
		totalTokens: a.totalTokens + b.totalTokens,
		cost: {
			input: a.cost.input + b.cost.input,
			output: a.cost.output + b.cost.output,
			cacheRead: a.cost.cacheRead + b.cost.cacheRead,
			cacheWrite: a.cost.cacheWrite + b.cost.cacheWrite,
			total: a.cost.total + b.cost.total,
		},
	};
}

/**
 * One reply as the session sees it after a cut: the cut thinking followed by the continuation's answer, as if Qwen
 * had written both in one reply. The usage is the sum of both requests (the thinking counted once, at the cut).
 */
function joinCut(cut: ThinkingCut, continuation: Attempt, sent: QwenRequestRecord["sent"]): Attempt {
	const thinking: ThinkingContent = { type: "thinking", thinking: cut.thinking, thinkingSignature: cut.signature };
	const final: AssistantMessage = {
		...continuation.final,
		content: [thinking, ...continuation.final.content],
		usage: addUsage(cut.usage, continuation.final.usage),
	};
	const shifted = continuation.events
		.filter((event) => event.type !== "start")
		.map((event) =>
			"contentIndex" in event
				? { ...event, contentIndex: event.contentIndex + 1, partial: final }
				: { ...event, partial: final },
		) as AssistantMessageEvent[];
	const events: AssistantMessageEvent[] = [
		...cut.events,
		{ type: "thinking_end", contentIndex: 0, content: cut.thinking, partial: final },
		...shifted,
	];
	const calls = continuation.final.content.some((part) => part.type === "toolCall");
	const stop = continuation.final.stopReason;
	const outcome: LimitCut["continuation"]["outcome"] = continuation.runaway
		? "runaway"
		: stop === "error" || stop === "aborted"
			? "error"
			: calls
				? "tool_call"
				: "no_tool_call";
	return {
		level: continuation.level,
		events,
		final,
		runaway: continuation.runaway,
		sent,
		ms: cut.ms + continuation.ms,
		limitCut: {
			thinking_tokens: cut.tokens,
			thinking_chars: cut.thinking.length,
			continuation: {
				outcome,
				stop_reason: stop,
				error_message: continuation.final.errorMessage ?? null,
				output_tokens: continuation.final.usage.output,
			},
			timings_ms: { thinking: cut.ms, continuation: continuation.ms },
		},
	};
}

/**
 * One request at `level` with the thinking limit: when the thinking reaches the limit, the request is stopped and a
 * second request continues the same reply from the cut thinking (no added words), with the same settings; the two
 * are joined into one reply (joinCut). A continuation that fails or brings no tool call is passed on as it is, like
 * any reply without a tool call; the trace says so (limit_cut.continuation.outcome).
 */
async function runLimited(
	inner: StreamFn,
	model: Model<Api>,
	context: Parameters<StreamFn>[1],
	streamOptions: SimpleStreamOptions | undefined,
	level: QwenThinkingLevel,
	limit: number | null,
): Promise<Attempt> {
	const mode: LimitMode = limit === null ? { kind: "none" } : { kind: "limit", tokens: limit };
	const { attempt, cut } = await runAttempt(inner, model, context, streamOptions, level, mode);
	if (!cut || attempt.runaway || streamOptions?.signal?.aborted) return attempt;
	const { attempt: continuation } = await runAttempt(inner, model, context, streamOptions, level, {
		kind: "continue",
		thinking: cut.thinking,
	});
	return joinCut(cut, continuation, attempt.sent);
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
 * 4. Thinking limit (JEFF_FIRST_THINKING_LIMIT, at every level, also on a re-ask): checked on the same stream after
 *    the runaway rules; when a reply's thinking reaches the limit, it is cut and continued (runLimited). The joined
 *    reply then goes through the loop guard like any reply, and a continuation that runs away is a runaway (re-ask).
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
				thinking_limit: options.thinkingLimit,
				limit_cut: attempt.limitCut,
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

			const first = await runLimited(options.inner, model, context, streamOptions, level, options.thinkingLimit);
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

			const second = await runLimited(
				options.inner,
				model,
				context,
				streamOptions,
				REASK_LEVEL,
				options.thinkingLimit,
			);
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
