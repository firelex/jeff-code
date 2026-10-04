import { mkdtempSync, readFileSync, rmSync } from "node:fs";
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
	type SimpleStreamOptions,
	type ToolCall,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { fixedRouter, type QwenThinkingLevel, type ThinkingRouter } from "../src/core/jeff-first/thinking.ts";
import { createThinkingControlStreamFn } from "../src/core/jeff-first/thinking-control.ts";
import { TraceWriter } from "../src/core/jeff-first/trace.ts";

const model = { id: "qwen", api: "openai-completions", provider: "local", reasoning: true } as unknown as Model<Api>;

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
			cacheRead: 100,
			cacheWrite: 0,
			reasoning: 30,
			totalTokens: 1240,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: 0,
	};
}

const bash = (command: string, id = "q1"): ToolCall => ({ type: "toolCall", id, name: "bash", arguments: { command } });

/** What the fake model does for one request: answer with a message, or think in a loop until it is stopped. */
type Script = { answer: AssistantMessage; thinking?: string } | { loopThinking: string };

interface Request {
	reasoning: SimpleStreamOptions["reasoning"];
	aborted: () => boolean;
}

/**
 * A fake coding model: plays one script per request, in order. Sends a qwen-chat-template payload through onPayload
 * as pi's provider does, and stops a looping script when its signal aborts (as a real request would).
 */
function fakeModel(scripts: Script[]) {
	const requests: Request[] = [];
	const payloads: unknown[] = [];
	const inner: StreamFn = (_model, _context, options) => {
		const script = scripts[requests.length];
		if (!script) throw new Error(`no script for request ${requests.length + 1}`);
		const signal = options?.signal;
		requests.push({ reasoning: options?.reasoning, aborted: () => signal?.aborted === true });
		const stream = createAssistantMessageEventStream();
		const run = async () => {
			const effort =
				options?.reasoning === undefined ? undefined : options.reasoning === "high" ? "xhigh" : options.reasoning;
			const payload = {
				model: "qwen",
				chat_template_kwargs: {
					enable_thinking: effort !== undefined,
					preserve_thinking: true,
					...(effort ? { reasoning_effort: effort } : {}),
				},
			};
			payloads.push(payload);
			await options?.onPayload?.(payload, _model);
			if ("loopThinking" in script) {
				const partial = reply([{ type: "thinking", thinking: "" }], "aborted");
				stream.push({ type: "start", partial });
				for (let index = 0; index < 2000 && !signal?.aborted; index++) {
					(partial.content[0] as { thinking: string }).thinking += script.loopThinking;
					stream.push({ type: "thinking_delta", contentIndex: 0, delta: script.loopThinking, partial });
					await new Promise((resolve) => setImmediate(resolve));
				}
				if (!signal?.aborted) throw new Error("the loop was never stopped");
				partial.errorMessage = "Request was aborted";
				stream.push({ type: "error", reason: "aborted", error: partial });
				return;
			}
			const message = script.answer;
			stream.push({ type: "start", partial: message });
			if (script.thinking) {
				stream.push({ type: "thinking_delta", contentIndex: 0, delta: script.thinking, partial: message });
			}
			if (message.stopReason === "error" || message.stopReason === "aborted") {
				stream.push({ type: "error", reason: message.stopReason, error: message });
			} else {
				stream.push({ type: "done", reason: message.stopReason as "toolUse" | "stop", message });
			}
		};
		run().catch((error: unknown) => {
			stream.push({ type: "error", reason: "error", error: { ...reply([], "error"), errorMessage: String(error) } });
		});
		return stream;
	};
	return { inner, requests, payloads };
}

async function drain(stream: Awaited<ReturnType<StreamFn>>) {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return { events, final: await stream.result() };
}

describe("createThinkingControlStreamFn", () => {
	let dir: string;
	let tracePath: string;
	const system: Message = {
		role: "system",
		content: "pi",
		toolsAdded: [{ name: "bash", description: "", parameters: {} as never }],
		timestamp: 0,
	};
	const task: Message = { role: "user", content: "Make the tests pass", timestamp: 0 };
	/** A session where Qwen already ran `make test` once. */
	const afterMakeTest: Message[] = [
		system,
		task,
		reply([bash("make test", "p1")]),
		{
			role: "toolResult",
			toolCallId: "p1",
			toolName: "bash",
			content: [{ type: "text", text: "1 failed" }],
			isError: false,
			timestamp: 0,
		},
	];

	beforeEach(() => {
		dir = mkdtempSync(join(tmpdir(), "jeff-first-thinking-"));
		tracePath = join(dir, "trace.jsonl");
	});
	afterEach(() => {
		rmSync(dir, { recursive: true, force: true });
	});

	const control = (inner: StreamFn, router: ThinkingRouter) =>
		createThinkingControlStreamFn({
			inner,
			taskId: "task-1",
			trace: new TraceWriter(tracePath),
			router,
			isSessionTurn: (id) => id === "s1",
		});
	const traceLines = () =>
		readFileSync(tracePath, "utf8")
			.trimEnd()
			.split("\n")
			.map((line) => JSON.parse(line));

	it("passes requests from outside the session straight through, at the session's level, without logging", async () => {
		const fake = fakeModel([{ answer: reply([{ type: "text", text: "summary" }], "stop") }]);
		await drain(
			await control(fake.inner, fixedRouter("off"))(model, normalizeContext({ messages: [system, task] }), {
				sessionId: undefined,
				reasoning: "medium",
			}),
		);
		expect(fake.requests.map((request) => request.reasoning)).toEqual(["medium"]);
		expect(() => readFileSync(tracePath)).toThrow();
	});

	it.each([
		["off", undefined, { enable_thinking: false, reasoning_effort: null }],
		["low", "low", { enable_thinking: true, reasoning_effort: "low" }],
		["medium", "medium", { enable_thinking: true, reasoning_effort: "medium" }],
		["xhigh", "xhigh", { enable_thinking: true, reasoning_effort: "xhigh" }],
	] as const)(
		"sends the router's level %s as pi's thinking level and logs what was sent",
		async (level, reasoning, sent) => {
			const fake = fakeModel([{ answer: reply([bash("ls")]), thinking: "look around" }]);
			const payloadsSeen: unknown[] = [];
			const { events, final } = await drain(
				await control(fake.inner, fixedRouter(level as QwenThinkingLevel))(
					model,
					normalizeContext({ messages: [system, task] }),
					{
						sessionId: "s1",
						reasoning: "medium",
						onPayload: (payload) => {
							payloadsSeen.push(payload);
							return undefined;
						},
					},
				),
			);
			expect(fake.requests.map((request) => request.reasoning)).toEqual([reasoning]);
			expect(payloadsSeen).toEqual(fake.payloads);
			expect(events.map((event) => event.type)).toEqual(["start", "thinking_delta", "done"]);
			expect(final.thinkingLevel).toBe(level);
			expect(traceLines()).toEqual([
				expect.objectContaining({
					schema: "jeff-first-trace/6",
					kind: "qwen_request",
					task_id: "task-1",
					session_id: "s1",
					turn: 1,
					attempt: 1,
					driver: "qwen",
					router: `fixed:${level}`,
					thinking_level: level,
					sent,
					outcome: "kept",
					guard: null,
					stop_reason: "toolUse",
					usage: { input: 1200, output: 40, thinking_tokens: 30, cache_read: 100, cache_write: 0 },
				}),
			]);
			expect(traceLines()[0].timings_ms.model).toBeGreaterThanOrEqual(0);
		},
	);

	it("logs the payload as the session's own payload hook replaced it", async () => {
		const fake = fakeModel([{ answer: reply([bash("ls")]) }]);
		await drain(
			await control(fake.inner, fixedRouter("medium"))(model, normalizeContext({ messages: [system, task] }), {
				sessionId: "s1",
				onPayload: () => ({ chat_template_kwargs: { enable_thinking: true, reasoning_effort: "low" } }),
			}),
		);
		expect(traceLines()[0].sent).toEqual({ enable_thinking: true, reasoning_effort: "low" });
	});

	it("discards a thinking-off reply that repeats a recent action and asks again once at xhigh", async () => {
		const fake = fakeModel([
			{ answer: reply([bash("make test", "q1")]) },
			{ answer: reply([bash("cat Makefile", "q2")]), thinking: "the test needs a fixture" },
		]);
		const { events, final } = await drain(
			await control(fake.inner, fixedRouter("off"))(model, normalizeContext({ messages: afterMakeTest }), {
				sessionId: "s1",
			}),
		);
		expect(fake.requests.map((request) => request.reasoning)).toEqual([undefined, "xhigh"]);
		// Only the re-ask reaches the session.
		expect(events.map((event) => event.type)).toEqual(["start", "thinking_delta", "done"]);
		expect(final.content).toEqual([bash("cat Makefile", "q2")]);
		expect(final.thinkingLevel).toBe("xhigh");
		const lines = traceLines();
		expect(lines.map((line) => [line.attempt, line.thinking_level, line.outcome])).toEqual([
			[1, "off", "discarded"],
			[2, "xhigh", "kept"],
		]);
		expect(lines[0].guard).toEqual({
			trigger: "loop",
			repeated_action: [{ name: "bash", arguments: { command: "make test" } }],
			turns_back: 1,
		});
		expect(lines[1].guard).toBeNull();
	});

	it("keeps a repeated reply from the re-ask: one re-ask only", async () => {
		const fake = fakeModel([
			{ answer: reply([bash("make test", "q1")]) },
			{ answer: reply([bash("make test", "q2")]) },
		]);
		const { final } = await drain(
			await control(fake.inner, fixedRouter("low"))(model, normalizeContext({ messages: afterMakeTest }), {
				sessionId: "s1",
			}),
		);
		expect(fake.requests).toHaveLength(2);
		expect(final.content).toEqual([bash("make test", "q2")]);
	});

	it("does not guard replies generated at medium or xhigh", async () => {
		for (const level of ["medium", "xhigh"] as const) {
			const fake = fakeModel([{ answer: reply([bash("make test", "q1")]) }]);
			await drain(
				await control(fake.inner, fixedRouter(level))(model, normalizeContext({ messages: afterMakeTest }), {
					sessionId: "s1",
				}),
			);
			expect(fake.requests).toHaveLength(1);
		}
	});

	it("does not re-ask when a file was written since the repeated action", async () => {
		const messages: Message[] = [
			...afterMakeTest,
			reply([bash("sed -i 's/1/2/' fixture.txt", "p2")]),
			{
				role: "toolResult",
				toolCallId: "p2",
				toolName: "bash",
				content: [{ type: "text", text: "" }],
				isError: false,
				timestamp: 0,
			},
		];
		const fake = fakeModel([{ answer: reply([bash("make test", "q1")]) }]);
		await drain(
			await control(fake.inner, fixedRouter("off"))(model, normalizeContext({ messages }), { sessionId: "s1" }),
		);
		expect(fake.requests).toHaveLength(1);
		expect(traceLines()[0]).toMatchObject({ outcome: "kept", guard: null });
	});

	it("stops runaway thinking, discards it and asks again at xhigh", async () => {
		const fake = fakeModel([
			{ loopThinking: "Wait, let me re-check the indices. " },
			{ answer: reply([bash("python fix.py", "q2")]) },
		]);
		const { events, final } = await drain(
			await control(fake.inner, fixedRouter("medium"))(model, normalizeContext({ messages: [system, task] }), {
				sessionId: "s1",
			}),
		);
		expect(fake.requests.map((request) => request.reasoning)).toEqual(["medium", "xhigh"]);
		expect(fake.requests[0].aborted()).toBe(true);
		expect(events.map((event) => event.type)).toEqual(["start", "done"]);
		expect(final.content).toEqual([bash("python fix.py", "q2")]);
		const [first, second] = traceLines();
		expect(first).toMatchObject({
			attempt: 1,
			thinking_level: "medium",
			outcome: "discarded",
			stop_reason: "aborted",
			guard: {
				trigger: "runaway",
				where: "thinking",
				rule: "repeated_piece",
				repeated: "Wait, let me re-check the indices. ",
			},
		});
		// Stopped soon after the thinking passed RUNAWAY_MIN_TEXT_CHARS (20,000), not at the fake's 2,000-step limit.
		expect(first.thinking_chars).toBeGreaterThanOrEqual(20_000);
		expect(first.thinking_chars).toBeLessThan(21_000);
		expect(second).toMatchObject({ attempt: 2, thinking_level: "xhigh", outcome: "kept" });
	});

	it("ends the turn with a JeffFirst error when the re-ask runs away too", async () => {
		const fake = fakeModel([
			{ loopThinking: "Wait, let me re-check the indices. " },
			{ loopThinking: "Hmm, so the offset is 12. " },
		]);
		const { events, final } = await drain(
			await control(fake.inner, fixedRouter("off"))(model, normalizeContext({ messages: [system, task] }), {
				sessionId: "s1",
			}),
		);
		expect(fake.requests).toHaveLength(2);
		expect(events.map((event) => event.type)).toEqual(["error"]);
		expect(final.stopReason).toBe("error");
		expect(final.errorMessage).toMatch(/^JeffFirst: turn 1 ran away again when asked again at thinking "xhigh"/);
		expect(traceLines().map((line) => [line.attempt, line.outcome, line.guard.trigger])).toEqual([
			[1, "discarded", "runaway"],
			[2, "turn_ended", "runaway"],
		]);
	});

	it("passes a request the user aborted on as it is", async () => {
		const controller = new AbortController();
		const fake = fakeModel([{ answer: reply([bash("make test")], "aborted") }]);
		controller.abort();
		const { final } = await drain(
			await control(fake.inner, fixedRouter("off"))(model, normalizeContext({ messages: afterMakeTest }), {
				sessionId: "s1",
				signal: controller.signal,
			}),
		);
		expect(fake.requests).toHaveLength(1);
		expect(fake.requests[0].aborted()).toBe(true);
		expect(final.stopReason).toBe("aborted");
		expect(traceLines()[0]).toMatchObject({ outcome: "kept", guard: null });
	});

	it("counts turns and asks the router before every request", async () => {
		const levels: QwenThinkingLevel[] = ["xhigh", "off"];
		const seen: string[] = [];
		const router: ThinkingRouter = {
			name: "test",
			levelFor: (state) => {
				seen.push(state.task);
				return levels[seen.length - 1];
			},
		};
		const fake = fakeModel([{ answer: reply([bash("ls", "q1")]) }, { answer: reply([bash("ls src", "q2")]) }]);
		const stream = control(fake.inner, router);
		await drain(await stream(model, normalizeContext({ messages: [system, task] }), { sessionId: "s1" }));
		await drain(await stream(model, normalizeContext({ messages: afterMakeTest }), { sessionId: "s1" }));
		expect(seen).toEqual(["Make the tests pass", "Make the tests pass"]);
		expect(fake.requests.map((request) => request.reasoning)).toEqual(["xhigh", undefined]);
		expect(traceLines().map((line) => [line.turn, line.thinking_level])).toEqual([
			[1, "xhigh"],
			[2, "off"],
		]);
	});

	it("ends the turn with a JeffFirst error when the router asks for thinking from a model that cannot think", async () => {
		const fake = fakeModel([]);
		const { final } = await drain(
			await control(fake.inner, fixedRouter("low"))(
				{ ...model, reasoning: false },
				normalizeContext({ messages: [system, task] }),
				{ sessionId: "s1" },
			),
		);
		expect(fake.requests).toHaveLength(0);
		expect(final.errorMessage).toMatch(
			/^JeffFirst: the thinking control failed on turn 1: the router fixed:low chose thinking "low"/,
		);
	});
});
