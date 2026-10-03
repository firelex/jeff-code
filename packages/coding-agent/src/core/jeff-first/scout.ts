import { randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	type Message,
	type Model,
} from "@earendil-works/pi-ai";
import { type CheckCommands, detectCheckCommands } from "./check-commands.ts";
import type { Choice, Chooser } from "./chooser.ts";
import type { RunApproval } from "./config.ts";
import { buildLists, type ToolKind } from "./lists.ts";
import type { MenuToolCall } from "./menu.ts";
import { JEFF_PROVIDER } from "./provider.ts";
import { trimState } from "./state.ts";
import { describeError, errorStream, forwardModelTurn, JEFF_FIRST_ERROR_PREFIX, ZERO_USAGE } from "./stream.ts";
import type { DecisionRecord, LevelRecord, TraceWriter } from "./trace.ts";
import { activeToolNames, collectSteps, taskText } from "./transcript.ts";

export const JEFF_MODEL_ID = "scout";
/** Scout steps allowed between two large-model turns. */
export const STEP_CAP = 8;

export interface ScoutOptions {
	inner: StreamFn;
	cwd: string;
	taskId: string;
	trace: TraceWriter;
	chooser: Chooser;
	isSessionTurn: (sessionId: string | undefined) => boolean;
	runApproval: RunApproval;
}

/** The scout's assistant messages since the large model's last message (or since the user's message). */
export function stepsSinceModel(messages: Message[]): number {
	let count = 0;
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message.role === "user") break;
		if (message.role !== "assistant") continue;
		if (message.provider !== JEFF_PROVIDER) break;
		count++;
	}
	return count;
}

function scoutMessage(model: Model<Api>, call: MenuToolCall): AssistantMessage {
	return {
		role: "assistant",
		content: [
			{
				type: "toolCall",
				id: `jeff_${randomUUID().replaceAll("-", "")}`,
				name: call.name,
				arguments: structuredClone(call.arguments),
			},
		],
		api: model.api,
		provider: JEFF_PROVIDER,
		model: JEFF_MODEL_ID,
		usage: ZERO_USAGE,
		stopReason: "toolUse",
		timestamp: Date.now(),
	};
}

function messageStream(message: AssistantMessage): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	stream.push({ type: "start", partial: message });
	stream.push({ type: "done", reason: "toolUse", message });
	return stream;
}

function levelRecord(options: LevelRecord["options"], choice: Choice, chooser: string): LevelRecord {
	return {
		options,
		chooser,
		shares: choice.shares,
		picks: choice.picks,
		chosen: choice.optionId,
		word_joiners_inserted: choice.wordJoinersInserted,
	};
}

export function createScoutStreamFn(options: ScoutOptions): StreamFn {
	let checks: CheckCommands | undefined;
	// Taken on the first turn: compaction later replaces the first user message with a summary.
	let task: string | undefined;
	let decision = 0;
	let turn = 0;

	return async (model, context, streamOptions) => {
		const sessionId = streamOptions?.sessionId;
		if (!options.isSessionTurn(sessionId) || activeToolNames(context.messages).size === 0) {
			return options.inner(model, context, streamOptions);
		}
		decision++;
		const thisDecision = decision;
		const stint = stepsSinceModel(context.messages);

		let call: MenuToolCall | null;
		try {
			const listsStarted = performance.now();
			task ??= taskText(context.messages);
			const steps = collectSteps(context.messages);
			checks ??= detectCheckCommands(options.cwd, task);
			const state = trimState(task, steps);
			const base = {
				schema: "jeff-first-trace/2",
				kind: "decision",
				task_id: options.taskId,
				session_id: sessionId as string,
				decision: thisDecision,
				step_in_stint: stint,
				mode: "teacher",
				driver: model.id,
				time: new Date().toISOString(),
				state,
				check_command_notes: checks.notes,
			} as const;

			let record: DecisionRecord;
			if (stint >= STEP_CAP) {
				record = {
					...base,
					tool_level: null,
					argument_level: null,
					action: { kind: "hand_over", why: "cap" },
					timings_ms: { lists: 0, chooser: 0 },
				};
				call = null;
			} else {
				const lists = buildLists({
					cwd: options.cwd,
					task,
					steps,
					activeTools: activeToolNames(context.messages),
					checkCommands: checks.commands,
					runApproval: options.runApproval,
				});
				const listsMs = performance.now() - listsStarted;
				const chooserStarted = performance.now();
				const toolChoice = await options.chooser.choose(state, { level: "tool", page: 1, options: lists.tools });
				const toolLevel = levelRecord(lists.tools, toolChoice, options.chooser.name);
				if (toolChoice.optionId === "hand_over") {
					record = {
						...base,
						tool_level: toolLevel,
						argument_level: null,
						action: { kind: "hand_over", why: "chosen" },
						timings_ms: { lists: listsMs, chooser: performance.now() - chooserStarted },
					};
					call = null;
				} else {
					const kind = toolChoice.optionId as ToolKind;
					const tool = lists.tools.find((option) => option.id === kind);
					const argumentOptions = lists.argumentsByTool[kind];
					if (!tool || !argumentOptions)
						throw new Error(`the chooser picked the tool ${kind}, which was not offered`);
					const argumentChoice = await options.chooser.choose(state, {
						level: "argument",
						page: 1,
						tool,
						options: argumentOptions,
					});
					const chosen = argumentOptions.find((option) => option.id === argumentChoice.optionId);
					if (!chosen)
						throw new Error(`the chooser picked ${argumentChoice.optionId}, which is not an argument option`);
					record = {
						...base,
						tool_level: toolLevel,
						argument_level: levelRecord(argumentOptions, argumentChoice, options.chooser.name),
						action: { kind: "step", tool_call: chosen.toolCall },
						timings_ms: { lists: listsMs, chooser: performance.now() - chooserStarted },
					};
					call = chosen.toolCall;
				}
			}
			options.trace.append(record);
		} catch (error) {
			// Not a fallback: the turn ends here, as an error the agent loop shows and pi does not retry.
			return errorStream(
				model,
				`${JEFF_FIRST_ERROR_PREFIX} the scout could not decide at decision ${thisDecision}: ${describeError(error)}`,
			);
		}

		if (call) return messageStream(scoutMessage(model, call));

		turn++;
		const thisTurn = turn;
		const modelStarted = performance.now();
		const inner = await options.inner(model, context, streamOptions);
		return forwardModelTurn(
			model,
			inner,
			(final) => {
				options.trace.append({
					schema: "jeff-first-trace/2",
					kind: "model_turn",
					task_id: options.taskId,
					session_id: sessionId as string,
					turn: thisTurn,
					mode: "teacher",
					driver: model.id,
					time: new Date().toISOString(),
					action: {
						stop_reason: final.stopReason,
						error_message: final.errorMessage ?? null,
						text_chars: final.content.reduce(
							(sum, part) => sum + (part.type === "text" ? part.text.length : 0),
							0,
						),
						tool_calls: final.content.flatMap((part) =>
							part.type === "toolCall" ? [{ name: part.name, arguments: part.arguments }] : [],
						),
					},
					model_usage: {
						input: final.usage.input,
						output: final.usage.output,
						cache_read: final.usage.cacheRead,
						cache_write: final.usage.cacheWrite,
					},
					timings_ms: { model: performance.now() - modelStarted },
				});
			},
			`turn ${thisTurn}`,
			"scout wrapper",
		);
	};
}
