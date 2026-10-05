import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamFn } from "@jeffhub/jeff-code-agent-core";
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
} from "@jeffhub/jeff-code-ai";
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

/**
 * What the fake model does for one request: answer with a message, think in a loop until it is stopped, or think one
 * token per piece (the server's output-token count rising by one per chunk) until stopped or `tokens` are written,
 * then answer.
 */
type Script =
	| { answer: AssistantMessage; thinking?: string }
	| { loopThinking: string }
	| { countedThinking: string; tokens: number; answer: AssistantMessage };

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
	/** The bodies as sent: after the onPayload hooks replaced them. */
	const sentPayloads: unknown[] = [];
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
				messages: [{ role: "user", content: "Make the tests pass" }],
				stream_options: { include_usage: true },
				chat_template_kwargs: {
					enable_thinking: effort !== undefined,
					preserve_thinking: true,
					...(effort ? { reasoning_effort: effort } : {}),
				},
			};
			payloads.push(payload);
			sentPayloads.push((await options?.onPayload?.(payload, _model)) ?? payload);
			if ("countedThinking" in script) {
				const partial = reply([{ type: "thinking", thinking: "", thinkingSignature: "reasoning" }], "aborted");
				partial.usage = { ...partial.usage, output: 0 };
				stream.push({ type: "start", partial });
				for (let index = 0; index < script.tokens && !signal?.aborted; index++) {
					(partial.content[0] as { thinking: string }).thinking += script.countedThinking;
					partial.usage = { ...partial.usage, output: partial.usage.output + 1 };
					stream.push({ type: "thinking_delta", contentIndex: 0, delta: script.countedThinking, partial });
					await new Promise((resolve) => setImmediate(resolve));
				}
				if (signal?.aborted) {
					partial.errorMessage = "Request was aborted";
					stream.push({ type: "error", reason: "aborted", error: partial });
					return;
				}
				const message = { ...script.answer, content: [...partial.content, ...script.answer.content] };
				stream.push({ type: "done", reason: message.stopReason as "toolUse" | "stop", message });
				return;
			}
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
	return { inner, requests, payloads, sentPayloads };
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

	const control = (inner: StreamFn, router: ThinkingRouter, thinkingLimit: number | null = null) =>
		createThinkingControlStreamFn({
			inner,
			taskId: "task-1",
			trace: new TraceWriter(tracePath),
			router,
			isSessionTurn: (id) => id === "s1",
			thinkingLimit,
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
			similarity: 1,
			pairs: [{ name: "bash", same: true, rule: "equal", similarity: 1, lengths: [9, 9] }],
			unchanged_writes: [],
		});
		expect(lines.map((line) => [line.router_level, line.forced_xhigh])).toEqual([
			["off", null],
			[null, null],
		]);
		expect(lines[1].guard).toBeNull();
	});

	it("logs the trained router's probabilities, cut and time on the first request only", async () => {
		const probabilities = { off: 0.8, low: 0.1, medium: 0.05, xhigh: 0.05 };
		const cut = { tokens_before: 9000, tokens_after: 8100, lines_left_out: 50, limit: 8192 };
		const router: ThinkingRouter = {
			name: "jeff:r",
			levelFor: async () => ({ level: "off", probabilities, cut, abstained: null }),
		};
		const fake = fakeModel([
			{ answer: reply([bash("make test", "q1")]) },
			{ answer: reply([bash("cat Makefile", "q2")]) },
		]);
		await drain(
			await control(fake.inner, router)(model, normalizeContext({ messages: afterMakeTest }), { sessionId: "s1" }),
		);
		const lines = traceLines();
		expect(
			lines.map((line) => [
				line.attempt,
				line.router,
				line.router_probabilities,
				line.router_cut,
				line.router_abstained,
			]),
		).toEqual([
			[1, "jeff:r", probabilities, cut, null],
			[2, "jeff:r", null, null, null],
		]);
		expect(lines[0].timings_ms.router).toBeGreaterThanOrEqual(0);
		expect(lines[1].timings_ms.router).toBeNull();
	});

	it("ends the turn with a JeffFirst error when the router fails", async () => {
		const router: ThinkingRouter = {
			name: "jeff:r",
			levelFor: async () => {
				throw new Error("the Jeff service at http://x answered 422: unknown model");
			},
		};
		const fake = fakeModel([]);
		const { final } = await drain(
			await control(fake.inner, router)(model, normalizeContext({ messages: [system, task] }), { sessionId: "s1" }),
		);
		expect(fake.requests).toHaveLength(0);
		expect(final.errorMessage).toBe(
			"JeffFirst: the thinking control failed on turn 1: the Jeff service at http://x answered 422: unknown model",
		);
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
			levelFor: async (state) => {
				seen.push(state.task);
				return { level: levels[seen.length - 1], probabilities: null, cut: null, abstained: null };
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

	describe("loop guard v2", () => {
		const result = (id: string, text: string, isError = false): Message => ({
			role: "toolResult",
			toolCallId: id,
			toolName: "bash",
			content: [{ type: "text", text }],
			isError,
			timestamp: 0,
		});
		/** A session where Qwen ran `commands`, each answered with the output and error flag given. */
		const session = (steps: Array<[string, string, boolean?]>): Message[] => [
			system,
			task,
			...steps.flatMap(([command, output, isError], index): Message[] => [
				reply([bash(command, `p${index}`)]),
				result(`p${index}`, output, isError),
			]),
		];
		const run = async (messages: Message[], router: ThinkingRouter, scripts: Script[]) => {
			const fake = fakeModel(scripts);
			const out = await drain(
				await control(fake.inner, router)(model, normalizeContext({ messages }), { sessionId: "s1" }),
			);
			return { fake, ...out };
		};

		it("discards a thinking-off reply near-identical to an earlier action and logs the similarity and lengths", async () => {
			const earlier = "cd /app && python -m pytest tests/test_parser.py -x -q";
			const messages = session([
				[earlier, "1 failed"],
				["cat src/parser.py", "def parse(): ..."],
			]);
			const near = earlier.replace("-q", "-v");
			const { fake } = await run(messages, fixedRouter("off"), [
				{ answer: reply([bash(near, "q1")]) },
				{ answer: reply([bash("cat tests/test_parser.py", "q2")]) },
			]);
			expect(fake.requests.map((request) => request.reasoning)).toEqual([undefined, "xhigh"]);
			const [first] = traceLines();
			expect(first).toMatchObject({
				outcome: "discarded",
				guard: {
					trigger: "loop",
					turns_back: 2,
					pairs: [{ name: "bash", same: true, rule: "dice", lengths: [near.length, earlier.length] }],
				},
			});
			expect(first.guard.similarity).toBeGreaterThanOrEqual(0.9);
		});

		it("re-asks a repeat when the only write since wrote what the file already held", async () => {
			const write = "cat > fix.py <<'EOF'\nprint(1)\nEOF";
			const messages = session([
				[write, ""],
				["python fix.py", "Traceback: boom"],
				[write, ""],
			]);
			await run(messages, fixedRouter("off"), [
				{ answer: reply([bash("python fix.py", "q1")]) },
				{ answer: reply([bash("cat fix.py", "q2")]) },
			]);
			expect(traceLines()[0].guard).toMatchObject({
				trigger: "loop",
				turns_back: 2,
				unchanged_writes: [{ name: "bash", paths: ["fix.py"], turns_back: 1 }],
			});
		});

		it("runs the turn at xhigh whatever the router says when Qwen's last 3 outputs are near-identical", async () => {
			const output = Array.from({ length: 40 }, (_, index) => `test_${index} FAILED AssertionError`).join("\n");
			const messages = session([
				["pytest -x", output],
				["pytest -x -q", output],
				["pytest -x -vv", output],
			]);
			const probabilities = { off: 0.9, low: 0.05, medium: 0.03, xhigh: 0.02 };
			const router: ThinkingRouter = {
				name: "jeff-off-unless:r:0.6",
				levelFor: async () => ({ level: "off", probabilities, cut: null, abstained: null }),
			};
			const { fake, final } = await run(messages, router, [{ answer: reply([bash("pytest -x", "q1")]) }]);
			// At xhigh the reply is not checked for repeats: it is kept.
			expect(fake.requests.map((request) => request.reasoning)).toEqual(["xhigh"]);
			expect(final.thinkingLevel).toBe("xhigh");
			const [line] = traceLines();
			expect(line).toMatchObject({
				router: "jeff-off-unless:r:0.6",
				router_probabilities: probabilities,
				router_level: "off",
				thinking_level: "xhigh",
				outcome: "kept",
				guard: null,
			});
			expect(line.forced_xhigh).toHaveLength(1);
			expect(line.forced_xhigh[0].rule).toBe("stuck_outputs");
			expect(line.forced_xhigh[0].pairs.map((pair: { rule: string }) => pair.rule)).toEqual([
				"equal",
				"equal",
				"equal",
			]);
		});

		it("runs the turn at xhigh after 2 failed commands in a row", async () => {
			const messages = session([
				["make", "error: missing header\n\nCommand exited with code 2", true],
				["make all", "error: no rule to make target\n\nCommand exited with code 2", true],
			]);
			const { fake } = await run(messages, fixedRouter("low"), [{ answer: reply([bash("ls", "q1")]) }]);
			expect(fake.requests.map((request) => request.reasoning)).toEqual(["xhigh"]);
			expect(traceLines()[0]).toMatchObject({
				router_level: "low",
				thinking_level: "xhigh",
				forced_xhigh: [
					{ rule: "failed_commands", last_lines: ["Command exited with code 2", "Command exited with code 2"] },
				],
			});
		});

		it("follows the router after one failed command, and the turn after a guard hit (no forced window)", async () => {
			const messages = session([
				["make", "ok"],
				["make test", "1 failed\n\nCommand exited with code 1", true],
			]);
			const fake = fakeModel([
				{ answer: reply([bash("make test", "q1")]) },
				{ answer: reply([bash("cat Makefile", "q2")]) },
				{ answer: reply([bash("ls", "q3")]) },
			]);
			const stream = control(fake.inner, fixedRouter("off"));
			await drain(await stream(model, normalizeContext({ messages }), { sessionId: "s1" }));
			const next: Message[] = [...messages, reply([bash("cat Makefile", "q2")]), result("q2", "all: build")];
			await drain(await stream(model, normalizeContext({ messages: next }), { sessionId: "s1" }));
			expect(fake.requests.map((request) => request.reasoning)).toEqual([undefined, "xhigh", undefined]);
			expect(traceLines().map((line) => [line.turn, line.attempt, line.thinking_level, line.forced_xhigh])).toEqual([
				[1, 1, "off", null],
				[1, 2, "xhigh", null],
				[2, 1, "off", null],
			]);
		});
	});

	describe("thinking limit", () => {
		const answer = reply([{ type: "text", text: "Listing." }, bash("ls")]);

		it("cuts the thinking at the limit, continues the same reply from it and joins the two", async () => {
			const fake = fakeModel([{ countedThinking: "hmm ", tokens: 100, answer }, { answer }]);
			const { events, final } = await drain(
				await control(fake.inner, fixedRouter("xhigh"), 10)(model, normalizeContext({ messages: [system, task] }), {
					sessionId: "s1",
				}),
			);
			expect(fake.requests.map((request) => request.reasoning)).toEqual(["xhigh", "xhigh"]);
			const cut = "hmm ".repeat(10);
			expect(fake.sentPayloads[0]).toMatchObject({
				stream_options: { include_usage: true, continuous_usage_stats: true },
			});
			expect(fake.sentPayloads[0]).not.toHaveProperty("continue_final_message");
			expect(fake.sentPayloads[1]).toEqual({
				...(fake.payloads[1] as object),
				stream_options: { include_usage: true, continuous_usage_stats: true },
				messages: [
					{ role: "user", content: "Make the tests pass" },
					{ role: "assistant", content: "", reasoning: cut },
				],
				continue_final_message: true,
				add_generation_prompt: false,
			});
			expect(final.content).toEqual([
				{ type: "thinking", thinking: cut, thinkingSignature: "reasoning" },
				{ type: "text", text: "Listing." },
				bash("ls"),
			]);
			expect(final.stopReason).toBe("toolUse");
			expect(final.usage.output).toBe(10 + 40);
			expect(events.filter((event) => event.type === "thinking_delta")).toHaveLength(10);
			expect(events.find((event) => event.type === "thinking_end")).toMatchObject({ contentIndex: 0, content: cut });
			const [line] = traceLines();
			expect(line).toMatchObject({
				outcome: "kept",
				thinking_limit: 10,
				limit_cut: {
					thinking_tokens: 10,
					thinking_chars: cut.length,
					continuation: { outcome: "tool_call", stop_reason: "toolUse", error_message: null, output_tokens: 40 },
				},
			});
			expect(line.limit_cut.timings_ms.thinking).toBeGreaterThanOrEqual(0);
			expect(line.limit_cut.timings_ms.continuation).toBeGreaterThanOrEqual(0);
		});

		it("leaves a reply whose thinking stays below the limit alone", async () => {
			const fake = fakeModel([{ countedThinking: "hmm ", tokens: 5, answer }]);
			const { final } = await drain(
				await control(fake.inner, fixedRouter("low"), 10)(model, normalizeContext({ messages: [system, task] }), {
					sessionId: "s1",
				}),
			);
			expect(fake.requests).toHaveLength(1);
			expect(final.content).toHaveLength(3);
			expect(traceLines()[0]).toMatchObject({ thinking_limit: 10, limit_cut: null });
		});

		it("does not count or cut without a limit", async () => {
			const fake = fakeModel([{ countedThinking: "hmm ", tokens: 50, answer }]);
			await drain(
				await control(fake.inner, fixedRouter("xhigh"), null)(
					model,
					normalizeContext({ messages: [system, task] }),
					{
						sessionId: "s1",
					},
				),
			);
			expect(fake.requests).toHaveLength(1);
			expect(fake.sentPayloads[0]).toEqual(fake.payloads[0]);
			expect(traceLines()[0]).toMatchObject({ thinking_limit: null, limit_cut: null });
		});

		it("passes a continuation without a tool call on as it is, and says so", async () => {
			const fake = fakeModel([
				{ countedThinking: "hmm ", tokens: 100, answer },
				{ answer: reply([{ type: "text", text: "I give up." }], "stop") },
			]);
			const { final } = await drain(
				await control(fake.inner, fixedRouter("xhigh"), 10)(model, normalizeContext({ messages: [system, task] }), {
					sessionId: "s1",
				}),
			);
			expect(final.stopReason).toBe("stop");
			expect(final.content.map((part) => part.type)).toEqual(["thinking", "text"]);
			expect(traceLines()[0].limit_cut.continuation).toMatchObject({ outcome: "no_tool_call", stop_reason: "stop" });
		});

		it("passes a failed continuation on as the error it is, and says so", async () => {
			const fake = fakeModel([
				{ countedThinking: "hmm ", tokens: 100, answer },
				{ answer: { ...reply([], "error"), errorMessage: "server error" } },
			]);
			const { events, final } = await drain(
				await control(fake.inner, fixedRouter("xhigh"), 10)(model, normalizeContext({ messages: [system, task] }), {
					sessionId: "s1",
				}),
			);
			expect(final.stopReason).toBe("error");
			expect(events.at(-1)?.type).toBe("error");
			expect(traceLines()[0].limit_cut.continuation).toEqual({
				outcome: "error",
				stop_reason: "error",
				error_message: "server error",
				output_tokens: 40,
			});
		});

		it("cuts at every level, also on the loop guard's re-ask, and the joined reply goes through the guard", async () => {
			const repeat = reply([bash("make test", "q1")]);
			const fake = fakeModel([
				{ countedThinking: "hmm ", tokens: 100, answer },
				{ answer: repeat },
				{ countedThinking: "think ", tokens: 100, answer },
				{ answer: reply([bash("cat Makefile", "q2")]) },
			]);
			const { final } = await drain(
				await control(fake.inner, fixedRouter("low"), 10)(model, normalizeContext({ messages: afterMakeTest }), {
					sessionId: "s1",
				}),
			);
			expect(fake.requests.map((request) => request.reasoning)).toEqual(["low", "low", "xhigh", "xhigh"]);
			expect(final.content.at(-1)).toEqual(bash("cat Makefile", "q2"));
			expect(
				traceLines().map((line) => [
					line.attempt,
					line.outcome,
					line.guard?.trigger ?? null,
					line.limit_cut?.thinking_tokens,
				]),
			).toEqual([
				[1, "discarded", "loop", 10],
				[2, "kept", null, 10],
			]);
		});
	});
});
