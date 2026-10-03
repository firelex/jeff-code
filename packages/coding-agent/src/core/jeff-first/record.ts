import { performance } from "node:perf_hooks";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { type CheckCommands, detectCheckCommands } from "./check-commands.ts";
import type { RunApproval } from "./config.ts";
import { buildLists, type Lists } from "./lists.ts";
import { type JeffState, trimState } from "./state.ts";
import { describeError, errorStream, forwardModelTurn, JEFF_FIRST_ERROR_PREFIX } from "./stream.ts";
import type { RecordRecord, TraceWriter } from "./trace.ts";
import { activeToolNames, collectSteps, taskText } from "./transcript.ts";

export interface RecordOptions {
	inner: StreamFn;
	cwd: string;
	taskId: string;
	trace: TraceWriter;
	/** True for the session's own agent turns; compaction and summaries pass no session id. */
	isSessionTurn: (sessionId: string | undefined) => boolean;
	runApproval: RunApproval;
	driverBuild: string;
}

interface Prepared {
	state: JeffState;
	lists: Lists;
	checks: CheckCommands;
	listsMs: number;
}

/**
 * Record mode: plain Qwen works alone, exactly as plain pi. Before each of its turns, this builds and logs the
 * scout's full option lists (every tool, every argument option, up to the usual limit per tool) so that a converter
 * can later label each point with Qwen's actual next action, or "hand over". It never acts on Qwen's behalf: it
 * always calls the inner model and forwards its message unchanged.
 */
export function createRecordStreamFn(options: RecordOptions): StreamFn {
	let checks: CheckCommands | undefined;
	// Taken on the first turn: compaction later replaces the first user message with a summary.
	let task: string | undefined;
	let turn = 0;

	return async (model, context, streamOptions) => {
		const sessionId = streamOptions?.sessionId;
		// Only agent turns, which offer tools, get a logged list; summaries and bug reports pass straight through.
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
			const lists = buildLists({
				cwd: options.cwd,
				task,
				steps,
				activeTools: activeToolNames(context.messages),
				checkCommands: checks.commands,
				runApproval: options.runApproval,
			});
			prepared = { state: trimState(task, steps), lists, checks, listsMs: performance.now() - started };
		} catch (error) {
			// Not a fallback: the turn ends here, as an error the agent loop shows and pi does not retry.
			return errorStream(
				model,
				`${JEFF_FIRST_ERROR_PREFIX} could not build the lists for turn ${thisTurn}: ${describeError(error)}`,
			);
		}

		const writeTrace = (final: AssistantMessage, modelMs: number) => {
			const toolCalls = final.content.filter((part) => part.type === "toolCall");
			const record: RecordRecord = {
				schema: "jeff-first-trace/4",
				kind: "record",
				task_id: options.taskId,
				session_id: sessionId as string,
				turn: thisTurn,
				mode: "record",
				driver: model.id,
				driver_build: options.driverBuild,
				run_approval: options.runApproval,
				time: new Date().toISOString(),
				state: prepared.state,
				check_command_notes: prepared.checks.notes,
				lists: { tools: prepared.lists.tools, arguments_by_tool: prepared.lists.argumentsByTool },
				action: {
					stop_reason: final.stopReason,
					error_message: final.errorMessage ?? null,
					text_chars: final.content.reduce((sum, part) => sum + (part.type === "text" ? part.text.length : 0), 0),
					tool_calls: toolCalls.map((call) => ({ name: call.name, arguments: call.arguments })),
				},
				model_usage: {
					input: final.usage.input,
					output: final.usage.output,
					cache_read: final.usage.cacheRead,
					cache_write: final.usage.cacheWrite,
				},
				timings_ms: { lists: prepared.listsMs, model: modelMs },
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
			"record wrapper",
		);
	};
}
