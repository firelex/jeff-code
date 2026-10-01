import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	normalizeContext,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createShadowStreamFn } from "../src/core/jeff-first/stream.ts";
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

describe("createShadowStreamFn", () => {
	let dir: string;
	let tracePath: string;
	const messages: Message[] = [
		{
			role: "system",
			content: "pi",
			toolsAdded: [{ name: "read", description: "", parameters: {} as never }],
			timestamp: 0,
		},
		{ role: "user", content: "Fix README.md", timestamp: 0 },
	];

	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-stream-"));
		writeFileSync(join(dir, "README.md"), "x");
		tracePath = join(dir, "trace.jsonl");
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	const shadow = (inner: StreamFn) =>
		createShadowStreamFn({
			inner,
			cwd: dir,
			taskId: "task-1",
			trace: new TraceWriter(tracePath),
			isSessionTurn: (id) => id === "s1",
		});
	const traceLines = () =>
		readFileSync(tracePath, "utf8")
			.trimEnd()
			.split("\n")
			.map((line) => JSON.parse(line));

	it("passes requests from outside the session straight through without logging", async () => {
		const { inner, calls } = fakeModel(reply([{ type: "text", text: "summary" }], "stop"));
		const stream = await shadow(inner)(model, normalizeContext({ messages }), { sessionId: undefined });
		await drain(stream);
		expect(calls.count).toBe(1);
		expect(() => readFileSync(tracePath)).toThrow();
	});

	it("forwards the model's events and logs the turn with its menu match", async () => {
		const message = reply([{ type: "toolCall", id: "t1", name: "read", arguments: { path: "README.md" } }]);
		const { inner } = fakeModel(message);
		const events = await drain(await shadow(inner)(model, normalizeContext({ messages }), { sessionId: "s1" }));
		expect(events.map((e) => e.type)).toEqual(["start", "done"]);
		const [line] = traceLines();
		expect(line).toMatchObject({
			schema: "jeff-first-trace/1",
			task_id: "task-1",
			session_id: "s1",
			turn: 1,
			jeff: null,
			actor: "model",
			action: { stop_reason: "toolUse", tool_calls: [{ name: "read", match: { kind: "exact" } }] },
			model_usage: { input: 1200, output: 40 },
		});
		expect(line.menu.at(-1).id).toBe("ask_model");
	});

	it("writes the trace line before the final event reaches the agent loop", async () => {
		const { inner } = fakeModel(reply([{ type: "text", text: "done" }], "stop"));
		const stream = await shadow(inner)(model, normalizeContext({ messages }), { sessionId: "s1" });
		for await (const event of stream) {
			if (event.type === "done") expect(traceLines()).toHaveLength(1);
		}
	});

	it("logs every tool call of a message, and text-only turns with none", async () => {
		const fn = shadow(
			fakeModel(
				reply([
					{ type: "toolCall", id: "a", name: "read", arguments: { path: "README.md" } },
					{ type: "toolCall", id: "b", name: "bash", arguments: { command: "make" } },
				]),
			).inner,
		);
		await drain(await fn(model, normalizeContext({ messages }), { sessionId: "s1" }));
		const textOnly = shadow(fakeModel(reply([{ type: "text", text: "All done." }], "stop")).inner);
		await drain(await textOnly(model, normalizeContext({ messages }), { sessionId: "s1" }));
		const [several, none] = traceLines();
		expect(several.action.tool_calls.map((c: { match: { kind: string } }) => c.match.kind)).toEqual([
			"exact",
			"none",
		]);
		expect(none.action).toMatchObject({ tool_calls: [], text_chars: 9 });
	});

	it("passes a model error through unchanged and still logs the turn", async () => {
		const failed = { ...reply([], "error"), errorMessage: "503 from server" };
		const events = await drain(
			await shadow(fakeModel(failed).inner)(model, normalizeContext({ messages }), { sessionId: "s1" }),
		);
		expect(events.at(-1)).toEqual({ type: "error", reason: "error", error: failed });
		expect(traceLines()[0].action).toMatchObject({ stop_reason: "error", error_message: "503 from server" });
	});

	it("fails the turn without calling the model when the menu cannot be built", async () => {
		const { inner, calls } = fakeModel(reply([], "stop"));
		const stream = await shadow(inner)(model, normalizeContext({ messages: [messages[0]] }), { sessionId: "s1" });
		const final = await stream.result();
		expect(calls.count).toBe(0);
		expect(final.stopReason).toBe("error");
		expect(final.errorMessage).toMatch(/^JeffFirst: could not build the menu for turn 1: .*no user message/);
	});

	it("fails the turn when the trace line cannot be written", async () => {
		const fn = shadow(fakeModel(reply([{ type: "text", text: "ok" }], "stop")).inner);
		rmSync(dir, { recursive: true, force: true });
		const final = await (await fn(model, normalizeContext({ messages }), { sessionId: "s1" })).result();
		expect(final.stopReason).toBe("error");
		expect(final.errorMessage).toMatch(/^JeffFirst: could not write the trace line for turn 1: /);
	});
	it("keeps the task from the first turn after compaction replaces the first user message", async () => {
		const fn = shadow(fakeModel(reply([{ type: "text", text: "ok" }], "stop")).inner);
		await drain(await fn(model, normalizeContext({ messages }), { sessionId: "s1" }));
		const compacted: Message[] = [
			messages[0],
			{ role: "user", content: "Summary of the earlier conversation: ...", timestamp: 0 },
		];
		await drain(await fn(model, normalizeContext({ messages: compacted }), { sessionId: "s1" }));
		expect(traceLines().map((line) => line.state.task)).toEqual(["Fix README.md", "Fix README.md"]);
	});

	it("passes session requests that declare no tools straight through without logging", async () => {
		const { inner, calls } = fakeModel(reply([{ type: "text", text: "bug summary" }], "stop"));
		const noTools: Message[] = [
			{ role: "user", content: "Summarise this conversation for a bug report", timestamp: 0 },
		];
		await drain(await shadow(inner)(model, normalizeContext({ messages: noTools }), { sessionId: "s1" }));
		expect(calls.count).toBe(1);
		expect(() => readFileSync(tracePath)).toThrow();
	});
});
