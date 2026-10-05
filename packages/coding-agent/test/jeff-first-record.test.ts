import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentEvent, StreamFn } from "@jeffhub/jeff-code-agent-core";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	normalizeContext,
	type ToolResultMessage,
} from "@jeffhub/jeff-code-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRecorder } from "../src/core/jeff-first/record.ts";
import { TraceWriter } from "../src/core/jeff-first/trace.ts";

const model = { id: "qwen", api: "openai-completions", provider: "local" } as unknown as Model<Api>;

function reply(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "toolUse",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-completions",
		provider: "local",
		model: "qwen",
		usage: {
			input: 1200,
			output: 40,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 1240,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: 0,
	};
}

/** A fake large model that streams start, then the final event, and counts its calls. */
function fakeModel(message: AssistantMessage) {
	const calls = { count: 0 };
	const inner: StreamFn = () => {
		calls.count++;
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() => {
			stream.push({ type: "start", partial: message });
			if (message.stopReason === "error" || message.stopReason === "aborted") {
				stream.push({ type: "error", reason: message.stopReason, error: message });
			} else {
				stream.push({ type: "done", reason: message.stopReason as "toolUse" | "stop", message });
			}
		});
		return stream;
	};
	return { inner, calls };
}

async function drain(stream: Awaited<ReturnType<StreamFn>>): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

describe("createRecordStreamFn", () => {
	let dir: string;
	let tracePath: string;
	const messages: Message[] = [
		{
			role: "system",
			content: "pi",
			toolsAdded: [{ name: "bash", description: "", parameters: {} as never }],
			timestamp: 0,
		},
		{ role: "user", content: "Fix README.md", timestamp: 0 },
	];

	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-record-"));
		writeFileSync(join(dir, "README.md"), "x");
		tracePath = join(dir, "trace.jsonl");
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	const recorder = (inner: StreamFn) =>
		createRecorder({
			inner,
			cwd: dir,
			taskId: "task-1",
			trace: new TraceWriter(tracePath),
			isSessionTurn: (id) => id === "s1",
			runApproval: "all",
			driverBuild: "test-build",
		});
	const record = (inner: StreamFn) => recorder(inner).streamFn;
	const traceLines = () =>
		readFileSync(tracePath, "utf8")
			.trimEnd()
			.split("\n")
			.map((line) => JSON.parse(line));

	it("passes requests from outside the session straight through without logging", async () => {
		const { inner, calls } = fakeModel(reply([{ type: "text", text: "summary" }], "stop"));
		const stream = await record(inner)(model, normalizeContext({ messages }), { sessionId: undefined });
		await drain(stream);
		expect(calls.count).toBe(1);
		expect(() => readFileSync(tracePath)).toThrow();
	});

	it("passes requests whose session id does not match straight through without logging", async () => {
		const { inner, calls } = fakeModel(reply([{ type: "text", text: "summary" }], "stop"));
		const stream = await record(inner)(model, normalizeContext({ messages }), { sessionId: "other" });
		await drain(stream);
		expect(calls.count).toBe(1);
		expect(() => readFileSync(tracePath)).toThrow();
	});

	it("always calls the model, forwards its message, and logs the lists, the model's tool calls, driver_build and run_approval", async () => {
		const message = reply([{ type: "toolCall", id: "t1", name: "read", arguments: { path: "README.md" } }]);
		const { inner, calls } = fakeModel(message);
		const events = await drain(await record(inner)(model, normalizeContext({ messages }), { sessionId: "s1" }));
		expect(events.map((e) => e.type)).toEqual(["start", "done"]);
		expect(calls.count).toBe(1);
		const [line] = traceLines();
		expect(line).toMatchObject({
			schema: "jeff-first-trace/5",
			kind: "record",
			task_id: "task-1",
			session_id: "s1",
			turn: 1,
			mode: "record",
			driver: "qwen",
			driver_build: "test-build",
			run_approval: "all",
			action: { stop_reason: "toolUse", tool_calls: [{ name: "read", arguments: { path: "README.md" } }] },
			model_usage: { input: 1200, output: 40 },
		});
		expect(line.state.task).toBe("Fix README.md");
		const readOptions = line.lists.arguments_by_tool.read as Array<{
			id: string;
			description: string;
			toolCall: { name: string; arguments: { command: string } };
		}>;
		expect(readOptions).toBeDefined();
		const readme = readOptions.find((option) => option.toolCall.arguments.command.endsWith("README.md'"));
		expect(readme).toMatchObject({
			id: expect.any(String),
			description: expect.any(String),
			toolCall: { name: "bash", arguments: { command: `cat '${join(dir, "README.md")}'` } },
		});
	});

	it("writes the trace line before the final event reaches the agent loop", async () => {
		const { inner } = fakeModel(reply([{ type: "text", text: "done" }], "stop"));
		const stream = await record(inner)(model, normalizeContext({ messages }), { sessionId: "s1" });
		for await (const event of stream) {
			if (event.type === "done") expect(traceLines()).toHaveLength(1);
		}
	});

	it("passes a model error through unchanged and still logs the turn", async () => {
		const failed = { ...reply([], "error"), errorMessage: "503 from server" };
		const events = await drain(
			await record(fakeModel(failed).inner)(model, normalizeContext({ messages }), { sessionId: "s1" }),
		);
		expect(events.at(-1)).toEqual({ type: "error", reason: "error", error: failed });
		expect(traceLines()[0].action).toMatchObject({ stop_reason: "error", error_message: "503 from server" });
	});

	it("fails the turn without calling the model when the lists cannot be built", async () => {
		const { inner, calls } = fakeModel(reply([], "stop"));
		const stream = await record(inner)(model, normalizeContext({ messages: [messages[0]] }), { sessionId: "s1" });
		const final = await stream.result();
		expect(calls.count).toBe(0);
		expect(final.stopReason).toBe("error");
		expect(final.errorMessage).toMatch(/^JeffFirst: could not build the lists for turn 1: .*no user message/);
	});

	it("fails the turn when the trace line cannot be written", async () => {
		const fn = record(fakeModel(reply([{ type: "text", text: "ok" }], "stop")).inner);
		rmSync(dir, { recursive: true, force: true });
		const final = await (await fn(model, normalizeContext({ messages }), { sessionId: "s1" })).result();
		expect(final.stopReason).toBe("error");
		expect(final.errorMessage).toMatch(/^JeffFirst: could not write the trace line for turn 1: /);
	});

	describe("menus after each step within a turn", () => {
		const bash = (id: string, command: string) => ({
			type: "toolCall" as const,
			id,
			name: "bash",
			arguments: { command },
		});
		const result = (id: string, text: string): ToolResultMessage => ({
			role: "toolResult",
			toolCallId: id,
			toolName: "bash",
			content: [{ type: "text", text }],
			isError: false,
			timestamp: 0,
		});
		const start = (id: string): AgentEvent => ({
			type: "tool_execution_start",
			toolCallId: id,
			toolName: "bash",
			args: {},
		});
		const end = (id: string, text: string): AgentEvent => ({ type: "message_end", message: result(id, text) });

		async function turnWithCalls(calls: ReturnType<typeof bash>[]) {
			const rec = recorder(fakeModel(reply(calls)).inner);
			await drain(await rec.streamFn(model, normalizeContext({ messages }), { sessionId: "s1" }));
			return rec;
		}

		it("logs the lists after each step that another call of the turn follows, the steps credited to the scout", async () => {
			const rec = await turnWithCalls([bash("c1", "ls"), bash("c2", "cat notes.md"), bash("c3", "cat README.md")]);
			rec.onEvent(start("c1"));
			// The first step reveals notes.md, which exists only from now on: the lists are built after the step ran.
			writeFileSync(join(dir, "notes.md"), "n");
			rec.onEvent(end("c1", "README.md\nnotes.md"));
			rec.onEvent(start("c2"));
			rec.onEvent(end("c2", "n"));
			rec.onEvent(start("c3"));
			rec.onEvent(end("c3", "x"));
			const lines = traceLines();
			expect(lines.map((line) => [line.kind, line.turn, line.step ?? null])).toEqual([
				["record", 1, null],
				["record_step", 1, 1],
				["record_step", 1, 2],
			]);
			expect(lines[1]).toMatchObject({
				schema: "jeff-first-trace/5",
				task_id: "task-1",
				session_id: "s1",
				mode: "record",
				driver: "qwen",
				driver_build: "test-build",
				run_approval: "all",
				calls_in_turn: 3,
				command: "ls",
			});
			expect(lines[1].state.recentSteps).toEqual([
				{ command: "ls", output: "README.md\nnotes.md", isError: false, byScout: true },
			]);
			expect(
				lines[2].state.recentSteps.map((step: { command: string; byScout: boolean }) => [
					step.command,
					step.byScout,
				]),
			).toEqual([
				["ls", true],
				["cat notes.md", true],
			]);
			expect(lines[2].command).toBe("cat notes.md");
			const reads = (lines[1].lists.arguments_by_tool.read ?? []).map(
				(option: { toolCall: { arguments: { command: string } } }) => option.toolCall.arguments.command,
			);
			expect(reads).toContain(`cat '${join(dir, "notes.md")}'`);
			expect(
				lines[0].lists.arguments_by_tool.read.map(
					(option: { toolCall: { arguments: { command: string } } }) => option.toolCall.arguments.command,
				),
			).not.toContain(`cat '${join(dir, "notes.md")}'`);
		});

		it("logs nothing after a turn's only call", async () => {
			const rec = await turnWithCalls([bash("c1", "ls")]);
			rec.onEvent(start("c1"));
			rec.onEvent(end("c1", "README.md"));
			expect(traceLines().map((line) => line.kind)).toEqual(["record"]);
		});

		it("ignores tool results of calls it did not record", async () => {
			const rec = await turnWithCalls([bash("c1", "ls"), bash("c2", "ls")]);
			rec.onEvent(start("other"));
			rec.onEvent(end("other", "x"));
			expect(traceLines().map((line) => line.kind)).toEqual(["record"]);
		});

		it("refuses calls run in parallel: a call starts before the one before it has its result", async () => {
			const rec = await turnWithCalls([bash("c1", "ls"), bash("c2", "ls")]);
			rec.onEvent(start("c1"));
			expect(() => rec.onEvent(start("c2"))).toThrow(/^JeffFirst: .*one after another/);
		});

		it("fails loudly with the JeffFirst prefix when the lists after a step cannot be built", async () => {
			const rec = await turnWithCalls([bash("c1", "ls"), bash("c2", "ls")]);
			rec.onEvent(start("c1"));
			rmSync(dir, { recursive: true, force: true });
			expect(() => rec.onEvent(end("c1", "x"))).toThrow(
				/^JeffFirst: could not log the lists after step 1 of turn 1: /,
			);
		});
	});
});
