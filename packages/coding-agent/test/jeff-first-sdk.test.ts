import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Api, type AssistantMessage, createAssistantMessageEventStream, type Model } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
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

	async function startSession() {
		const settingsManager = SettingsManager.inMemory({});
		settingsManager.applyOverrides({
			retry: { enabled: true, maxRetries: 3, baseDelayMs: 1, maxAgentDelayMs: 1000 },
		});
		const resourceLoader = new DefaultResourceLoader({ cwd, agentDir, settingsManager, extensionFactories: [] });
		await resourceLoader.reload();
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		await authStorage.modify(model.provider, async () => ({ type: "api_key", key: "test" }));
		const registry = await createModelRegistry(authStorage, join(agentDir, "models.json"));
		const provider = { calls: 0 };
		registry.registerProvider(model.provider, {
			api: model.api,
			streamSimple: () => {
				provider.calls++;
				const stream = createAssistantMessageEventStream();
				stream.end(
					provider.calls === 1
						? answer([{ type: "toolCall", id: "r1", name: "read", arguments: { path: "README.md" } }], "toolUse")
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

	it("refuses to create a session with an unknown mode", async () => {
		vi.stubEnv("JEFF_FIRST_MODE", "routing");
		await expect(startSession()).rejects.toThrow(/must be off, shadow or teacher/);
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
		const { session, provider } = await startSession();
		await session.prompt("Read README.md and fix the bug.");
		session.dispose();
		await teacher.close();
		const lines = readFileSync(join(traceDir, "trace.jsonl"), "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l));
		expect(lines.map((l) => `${l.kind}:${l.action.kind ?? l.action.stop_reason}`)).toEqual([
			"decision:step",
			"decision:hand_over",
			"model_turn:toolUse",
			"decision:hand_over",
			"model_turn:stop",
		]);
		expect(lines[0].schema).toBe("jeff-first-trace/3");
		expect(lines[1].state.recentSteps[0].output).toContain("The bug is in main.py.");
		expect(provider.calls).toBe(2);
		expect(teacher.requests).toHaveLength(4 * 5);
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
