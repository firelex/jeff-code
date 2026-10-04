import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type AssistantMessage,
	type Context,
	createAssistantMessageEventStream,
	type Model,
	type ToolResultMessage,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { trimQuestion } from "../src/core/jeff-first/output-trim.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { answer as jeffAnswer, startFakeJeff } from "./jeff-first-fake-jeff.ts";
import { answerFor, type FakeTeacher, startFakeTeacher } from "./jeff-first-fake-teacher.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

const model: Model<Api> = {
	id: "fake-qwen",
	name: "Fake Qwen",
	api: "openai-completions",
	provider: "fake-local",
	baseUrl: "https://fake.invalid/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 4096,
};

function answer(content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"]): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 100,
			output: 10,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 110,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: Date.now(),
	};
}

describe("JeffFirst through createAgentSession", () => {
	let tempDir: string;
	let cwd: string;
	let agentDir: string;
	let traceDir: string;

	beforeEach(() => {
		tempDir = mkdtempSync(join(tmpdir(), "jeff-first-sdk-"));
		cwd = join(tempDir, "project");
		agentDir = join(tempDir, "agent");
		traceDir = join(tempDir, "timeout-traces"); // "timeout" makes a write error look transient to pi's retry rules
		for (const folder of [cwd, agentDir, traceDir]) mkdirSync(folder, { recursive: true });
		writeFileSync(join(cwd, "README.md"), "The bug is in main.py.");
	});
	afterEach(() => {
		rmSync(tempDir, { recursive: true, force: true });
	});

	async function startSession(
		firstAnswer: AssistantMessage["content"] = [
			{ type: "toolCall", id: "r1", name: "read", arguments: { path: "README.md" } },
		],
	) {
		const settingsManager = SettingsManager.inMemory({});
		settingsManager.applyOverrides({
			retry: { enabled: true, maxRetries: 3, baseDelayMs: 1, maxAgentDelayMs: 1000 },
		});
		const resourceLoader = new DefaultResourceLoader({ cwd, agentDir, settingsManager, extensionFactories: [] });
		await resourceLoader.reload();
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		await authStorage.modify(model.provider, async () => ({ type: "api_key", key: "test" }));
		const registry = await createModelRegistry(authStorage, join(agentDir, "models.json"));
		const provider = { calls: 0, contexts: [] as Context[] };
		registry.registerProvider(model.provider, {
			api: model.api,
			streamSimple: (_model, context) => {
				provider.calls++;
				provider.contexts.push(structuredClone(context));
				const stream = createAssistantMessageEventStream();
				stream.end(
					provider.calls === 1
						? answer(firstAnswer, "toolUse")
						: answer([{ type: "text", text: "Done." }], "stop"),
				);
				return stream;
			},
		});
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model,
			modelRuntime: getModelRuntime(registry),
			settingsManager,
			sessionManager: SessionManager.inMemory(cwd),
			resourceLoader,
		});
		return { session, provider, registry };
	}

	it("writes nothing when JEFF_FIRST_MODE is unset", async () => {
		const { session, provider } = await startSession();
		await session.prompt("Read README.md and fix the bug.");
		session.dispose();
		expect(provider.calls).toBe(2);
		expect(existsSync(join(traceDir, "trace.jsonl"))).toBe(false);
	});

	it("logs one line per model turn in shadow mode, with the read on the menu", async () => {
		vi.stubEnv("JEFF_FIRST_MODE", "shadow");
		vi.stubEnv("JEFF_FIRST_TRACE_FILE", join(traceDir, "trace.jsonl"));
		vi.stubEnv("JEFF_FIRST_TASK_ID", "sdk-test");
		const { session } = await startSession();
		await session.prompt("Read README.md and fix the bug.");
		session.dispose();
		const lines = readFileSync(join(traceDir, "trace.jsonl"), "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l));
		expect(lines.map((l) => l.turn)).toEqual([1, 2]);
		expect(lines[0].action.tool_calls[0]).toMatchObject({ name: "read", match: { kind: "exact" } });
		expect(lines[1].state.recentSteps[0].output).toContain("The bug is in main.py.");
	});

	it("logs one line per model turn in record mode, with the lists built and the model's tool calls", async () => {
		vi.stubEnv("JEFF_FIRST_MODE", "record");
		vi.stubEnv("JEFF_FIRST_TRACE_FILE", join(traceDir, "trace.jsonl"));
		vi.stubEnv("JEFF_FIRST_TASK_ID", "sdk-test");
		vi.stubEnv("JEFF_FIRST_RUN_APPROVAL", "all");
		vi.stubEnv("JEFF_FIRST_DRIVER_BUILD", "test-build");
		vi.stubEnv("JEFF_FIRST_THINKING_ROUTER", "fixed:off");
		vi.stubEnv("JEFF_FIRST_OUTPUT_TRIM", "off");
		const { session } = await startSession();
		await session.prompt("Read README.md and fix the bug.");
		session.dispose();
		const all = readFileSync(join(traceDir, "trace.jsonl"), "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l));
		// The thinking control logs each Qwen request before the turn's record line.
		expect(all.map((l) => `${l.kind}:${l.turn}`)).toEqual([
			"qwen_request:1",
			"record:1",
			"qwen_request:2",
			"record:2",
		]);
		expect(all[0]).toMatchObject({
			schema: "jeff-first-trace/6",
			router: "fixed:off",
			thinking_level: "off",
			attempt: 1,
			outcome: "kept",
		});
		const answers = session.messages.filter((message) => message.role === "assistant");
		expect(answers.map((message) => message.thinkingLevel)).toEqual(["off", "off"]);
		const lines = all.filter((l) => l.kind !== "qwen_request");
		expect(lines.map((l) => l.turn)).toEqual([1, 2]);
		expect(lines[0]).toMatchObject({
			schema: "jeff-first-trace/5",
			kind: "record",
			mode: "record",
			driver_build: "test-build",
			run_approval: "all",
		});
		expect(lines[0].action.tool_calls[0]).toMatchObject({ name: "read" });
		expect(lines[0].lists.arguments_by_tool.read).toBeDefined();
	});

	it("in record mode runs a turn's calls one after another and logs the lists after each step another call follows", async () => {
		vi.stubEnv("JEFF_FIRST_MODE", "record");
		vi.stubEnv("JEFF_FIRST_TRACE_FILE", join(traceDir, "trace.jsonl"));
		vi.stubEnv("JEFF_FIRST_TASK_ID", "sdk-test");
		vi.stubEnv("JEFF_FIRST_RUN_APPROVAL", "all");
		vi.stubEnv("JEFF_FIRST_DRIVER_BUILD", "test-build");
		vi.stubEnv("JEFF_FIRST_THINKING_ROUTER", "fixed:off");
		vi.stubEnv("JEFF_FIRST_OUTPUT_TRIM", "off");
		const { session } = await startSession([
			{ type: "toolCall", id: "b1", name: "bash", arguments: { command: "printf 'see notes.md' > notes.md && ls" } },
			{ type: "toolCall", id: "b2", name: "bash", arguments: { command: "cat notes.md" } },
		]);
		await session.prompt("Read README.md and fix the bug.");
		session.dispose();
		const lines = readFileSync(join(traceDir, "trace.jsonl"), "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l))
			.filter((l) => l.kind !== "qwen_request");
		expect(lines.map((l) => `${l.kind}:${l.turn}:${l.step ?? "-"}`)).toEqual([
			"record:1:-",
			"record_step:1:1",
			"record:2:-",
		]);
		expect(lines[1]).toMatchObject({ command: "printf 'see notes.md' > notes.md && ls", calls_in_turn: 2 });
		expect(lines[1].state.recentSteps).toHaveLength(1);
		expect(lines[1].state.recentSteps[0]).toMatchObject({ byScout: true });
		expect(lines[1].state.recentSteps[0].output).toContain("notes.md");
		expect(lines[2].state.recentSteps.map((step: { byScout: boolean }) => step.byScout)).toEqual([false, false]);
	});

	it("refuses to create a session with an unknown mode", async () => {
		vi.stubEnv("JEFF_FIRST_MODE", "routing");
		await expect(startSession()).rejects.toThrow(/must be off, shadow, teacher, record or jeff/);
	});

	it("in teacher mode runs the teacher's steps, then hands over, and logs both kinds of line", async () => {
		// The teacher reads README.md first, then hands over at every later decision.
		const teacher: FakeTeacher = await startFakeTeacher((options, prompt) => {
			if (prompt.includes("Which one exactly?")) return answerFor(options, "Read the file");
			if (prompt.includes("No steps have been taken yet.")) return answerFor(options, "Read part or all");
			return answerFor(options, "Hand over");
		});
		vi.stubEnv("JEFF_FIRST_MODE", "teacher");
		vi.stubEnv("JEFF_FIRST_TRACE_FILE", join(traceDir, "trace.jsonl"));
		vi.stubEnv("JEFF_FIRST_TASK_ID", "sdk-test");
		vi.stubEnv("JEFF_FIRST_TEACHER_URL", teacher.url);
		vi.stubEnv("JEFF_FIRST_TEACHER_MODEL", "glm-test");
		vi.stubEnv("JEFF_FIRST_RUN_APPROVAL", "all");
		vi.stubEnv("JEFF_FIRST_DRIVER_BUILD", "test-build");
		vi.stubEnv("JEFF_FIRST_THINKING_ROUTER", "fixed:off");
		vi.stubEnv("JEFF_FIRST_OUTPUT_TRIM", "off");
		const { session, provider } = await startSession();
		await session.prompt("Read README.md and fix the bug.");
		session.dispose();
		await teacher.close();
		const lines = readFileSync(join(traceDir, "trace.jsonl"), "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l));
		expect(lines.map((l) => `${l.kind}:${l.action?.kind ?? l.action?.stop_reason ?? l.thinking_level}`)).toEqual([
			"decision:step",
			"decision:hand_over",
			"qwen_request:off",
			"model_turn:toolUse",
			"decision:hand_over",
			"qwen_request:off",
			"model_turn:stop",
		]);
		expect(lines[0].schema).toBe("jeff-first-trace/3");
		expect(lines[1].state.recentSteps[0].output).toContain("The bug is in main.py.");
		expect(provider.calls).toBe(2);
		expect(teacher.requests).toHaveLength(4 * 5);
	});

	it("in jeff mode asks the Jeff service for the steps and the thinking level, and logs Jeff's probabilities", async () => {
		// Jeff reads README.md first, then hands over; the router answers "off" (the fake model cannot think).
		const jeff = await startFakeJeff((sent) => {
			const ids = Object.keys(sent.questions.q.criteria);
			const instructions = sent.questions.q.instructions;
			const wanted = instructions.includes("how much should it think")
				? "off"
				: instructions.includes("Which one exactly?")
					? ids[0]
					: sent.state.includes("No steps have been taken yet.")
						? "read"
						: "hand_over";
			return jeffAnswer(
				sent.model,
				Object.fromEntries(ids.map((id) => [id, id === wanted ? 0.9 : 0.1 / ids.length])),
			);
		});
		vi.stubEnv("JEFF_FIRST_MODE", "jeff");
		vi.stubEnv("JEFF_FIRST_TRACE_FILE", join(traceDir, "trace.jsonl"));
		vi.stubEnv("JEFF_FIRST_TASK_ID", "sdk-test");
		vi.stubEnv("JEFF_FIRST_JEFF_URL", jeff.url);
		vi.stubEnv("JEFF_FIRST_JEFF_STEP_ADAPTER", "jeff-step");
		vi.stubEnv("JEFF_FIRST_JEFF_STEP_THRESHOLD", "0.5");
		vi.stubEnv("JEFF_FIRST_RUN_APPROVAL", "all");
		vi.stubEnv("JEFF_FIRST_DRIVER_BUILD", "test-build");
		vi.stubEnv("JEFF_FIRST_THINKING_ROUTER", "jeff:jeff-router");
		vi.stubEnv("JEFF_FIRST_OUTPUT_TRIM", "off");
		vi.stubEnv("JEFF_FIRST_JEFF_ROUTER_THRESHOLD", "0.5");
		const { session, provider } = await startSession();
		await session.prompt("Read README.md and fix the bug.");
		session.dispose();
		await jeff.close();
		const lines = readFileSync(join(traceDir, "trace.jsonl"), "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l));
		expect(lines.map((l) => `${l.kind}:${l.action?.kind ?? l.action?.stop_reason ?? l.thinking_level}`)).toEqual([
			"decision:step",
			"decision:hand_over",
			"qwen_request:off",
			"model_turn:toolUse",
			"decision:hand_over",
			"qwen_request:off",
			"model_turn:stop",
		]);
		expect(lines[0]).toMatchObject({ mode: "jeff", levels: [{ chooser: "jeff:jeff-step", chosen: "read" }, {}] });
		expect(lines[0].levels[0].shares.read).toBe(0.9);
		expect(lines[2]).toMatchObject({
			router: "jeff:jeff-router",
			router_probabilities: { off: 0.9, low: 0.025, medium: 0.025, xhigh: 0.025 },
		});
		expect(lines[2].timings_ms.router).toBeGreaterThanOrEqual(0);
		expect(provider.calls).toBe(2);
		expect(jeff.requests.map((request) => request.model)).toEqual([
			"jeff-step",
			"jeff-step",
			"jeff-step",
			"jeff-router",
			"jeff-step",
			"jeff-router",
		]);
	});

	function recordModeWithTrim(value: string) {
		vi.stubEnv("JEFF_FIRST_MODE", "record");
		vi.stubEnv("JEFF_FIRST_TRACE_FILE", join(traceDir, "trace.jsonl"));
		vi.stubEnv("JEFF_FIRST_TASK_ID", "sdk-test");
		vi.stubEnv("JEFF_FIRST_RUN_APPROVAL", "all");
		vi.stubEnv("JEFF_FIRST_DRIVER_BUILD", "test-build");
		vi.stubEnv("JEFF_FIRST_THINKING_ROUTER", "fixed:off");
		vi.stubEnv("JEFF_FIRST_OUTPUT_TRIM", value);
	}
	const seq100: AssistantMessage["content"] = [
		{ type: "toolCall", id: "b1", name: "bash", arguments: { command: "seq 1 100" } },
	];
	const lines100 = Array.from({ length: 100 }, (_, index) => String(index + 1));

	function traceLines() {
		return readFileSync(join(traceDir, "trace.jsonl"), "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l));
	}

	it("shortens a long new output before it enters the session, and keeps the whole output in the trace", async () => {
		recordModeWithTrim("fixed:last40");
		const { session, provider } = await startSession(seq100);
		await session.prompt("Count to 100.");
		session.dispose();
		const shortened = `${lines100.slice(60).join("\n")}\n\n[Showing lines 61-100 of 100. 60 earlier lines not shown.]`;
		const result = session.messages.find((m) => m.role === "toolResult") as ToolResultMessage;
		expect(result.content).toEqual([{ type: "text", text: shortened }]);
		const sent = provider.contexts[1].messages.find((m) => m.role === "toolResult") as ToolResultMessage;
		expect(sent.content).toEqual([{ type: "text", text: shortened }]);
		const trim = traceLines().filter((l) => l.kind === "output_trim");
		expect(trim).toHaveLength(1);
		expect(trim[0]).toMatchObject({
			schema: "jeff-first-trace/7",
			tool_call_id: "b1",
			trimmer: "fixed:last40",
			shown_lines: 100,
			total_lines: 100,
			available: ["last40", "first40", "first20last20"],
			choice: "last40",
			probabilities: null,
			shortened: true,
			full_output: `${lines100.join("\n")}\n`,
		});
	});

	it("leaves outputs of at most 40 lines alone, without a trace line", async () => {
		recordModeWithTrim("fixed:last40");
		const { session } = await startSession([
			{ type: "toolCall", id: "b1", name: "bash", arguments: { command: "seq 1 40" } },
		]);
		await session.prompt("Count to 40.");
		session.dispose();
		const result = session.messages.find((m) => m.role === "toolResult") as ToolResultMessage;
		expect(result.content).toEqual([{ type: "text", text: `${lines100.slice(0, 40).join("\n")}\n` }]);
		expect(traceLines().filter((l) => l.kind === "output_trim")).toEqual([]);
	});

	it("asks Jeff's trimming adapter the trimming question and follows a confident answer", async () => {
		const jeff = await startFakeJeff((sent) =>
			jeffAnswer(sent.model, { all: 0.05, last200: 0.05, last40: 0.05, first40: 0.05, first20last20: 0.8 }),
		);
		recordModeWithTrim("jeff:jeff-trim");
		vi.stubEnv("JEFF_FIRST_JEFF_URL", jeff.url);
		vi.stubEnv("JEFF_FIRST_JEFF_TRIM_THRESHOLD", "0.7");
		const { session } = await startSession(seq100);
		await session.prompt("Count to 100.");
		session.dispose();
		await jeff.close();
		expect(jeff.requests).toHaveLength(1);
		const question = jeff.requests[0];
		expect(question.model).toBe("jeff-trim");
		expect(question.questions.q.instructions).toBe(trimQuestion(100));
		expect(Object.keys(question.questions.q.criteria)).toEqual([
			"all",
			"last200",
			"last40",
			"first40",
			"first20last20",
		]);
		// Jeff sees the whole new output's end, as the state shows any step.
		expect(question.state).toContain("$ seq 1 100\n[61 earlier lines not shown]\n62\n");
		const result = session.messages.find((m) => m.role === "toolResult") as ToolResultMessage;
		expect(result.content).toEqual([
			{
				type: "text",
				text: `${lines100.slice(0, 20).join("\n")}\n\n[Showing lines 1-20 and 81-100 of 100. 60 lines in between not shown.]\n\n${lines100.slice(80).join("\n")}`,
			},
		]);
		const trim = traceLines().find((l) => l.kind === "output_trim");
		expect(trim).toMatchObject({ trimmer: "jeff:jeff-trim", choice: "first20last20", shortened: true });
		expect(trim.probabilities.first20last20).toBe(0.8);
	});

	it("keeps the whole output when Jeff's most likely cut is below the threshold", async () => {
		const jeff = await startFakeJeff((sent) =>
			jeffAnswer(sent.model, { all: 0.1, last200: 0.1, last40: 0.6, first40: 0.1, first20last20: 0.1 }),
		);
		recordModeWithTrim("jeff:jeff-trim");
		vi.stubEnv("JEFF_FIRST_JEFF_URL", jeff.url);
		vi.stubEnv("JEFF_FIRST_JEFF_TRIM_THRESHOLD", "0.7");
		const { session } = await startSession(seq100);
		await session.prompt("Count to 100.");
		session.dispose();
		await jeff.close();
		const result = session.messages.find((m) => m.role === "toolResult") as ToolResultMessage;
		expect(result.content).toEqual([{ type: "text", text: `${lines100.join("\n")}\n` }]);
		expect(traceLines().find((l) => l.kind === "output_trim")).toMatchObject({
			choice: "all",
			shortened: false,
			full_output: null,
		});
	});

	it("does not retry a JeffFirst failure even when its text looks transient", async () => {
		vi.stubEnv("JEFF_FIRST_MODE", "shadow");
		vi.stubEnv("JEFF_FIRST_TRACE_FILE", join(traceDir, "trace.jsonl"));
		vi.stubEnv("JEFF_FIRST_TASK_ID", "sdk-test");
		const { session, provider } = await startSession();
		rmSync(traceDir, { recursive: true, force: true });
		await session.prompt("Read README.md and fix the bug.");
		const last = session.messages.at(-1) as AssistantMessage;
		session.dispose();
		expect(last.errorMessage).toMatch(/^JeffFirst: could not write the trace line for turn 1: .*timeout-traces/);
		expect(provider.calls).toBe(1);
	});
});
