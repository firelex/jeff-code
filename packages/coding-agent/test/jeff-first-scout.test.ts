import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	type Api,
	type AssistantMessage,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	normalizeContext,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Choice, Chooser } from "../src/core/jeff-first/chooser.ts";
import { JEFF_PROVIDER } from "../src/core/jeff-first/provider.ts";
import {
	BEFORE_FIRST_MODEL_TURN_NOTICE,
	createScoutStreamFn,
	STEP_CAP,
	stepsSinceModel,
} from "../src/core/jeff-first/scout.ts";
import type { Level } from "../src/core/jeff-first/teacher-prompt.ts";
import { TraceWriter } from "../src/core/jeff-first/trace.ts";

const model = { id: "qwen", api: "openai-completions", provider: "local" } as unknown as Model<Api>;
const ZERO = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistant(
	provider: string,
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "toolUse",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-completions",
		provider,
		model: "m",
		usage: ZERO,
		stopReason,
		timestamp: 0,
	};
}

function system(): Message {
	return {
		role: "system",
		content: "",
		toolsAdded: ["read", "bash", "ls", "grep", "find"].map((name) => ({ name, description: name, parameters: {} })),
	} as unknown as Message;
}

function fakeModel(message: AssistantMessage) {
	const calls = { count: 0 };
	const inner: StreamFn = () => {
		calls.count++;
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() => {
			stream.push({ type: "start", partial: message });
			stream.push({ type: "done", reason: "toolUse", message });
		});
		return stream;
	};
	return { inner, calls };
}

/** A chooser that answers each question in turn from a list of option ids, and records what it was asked. */
function scripted(answers: string[] | Error): Chooser & { asked: Level[] } {
	const asked: Level[] = [];
	const queue = answers instanceof Error ? [] : [...answers];
	return {
		name: "teacher:fake",
		asked,
		async choose(_state, level): Promise<Choice> {
			asked.push(level);
			if (answers instanceof Error) throw answers;
			const optionId = queue.shift() ?? level.options[0].id;
			return {
				optionId,
				shares: { [optionId]: 1 },
				picks: [{ optionId, reason: "scripted", failedAttempts: [] }],
			};
		},
	};
}

describe("stepsSinceModel", () => {
	it("counts the scout's messages after the large model's last message", () => {
		const messages = [
			{ role: "user", content: "task", timestamp: 0 },
			assistant("local", []),
			assistant(JEFF_PROVIDER, []),
			{ role: "toolResult", toolCallId: "x", toolName: "read", content: [], isError: false, timestamp: 0 },
			assistant(JEFF_PROVIDER, []),
		] as Message[];
		expect(stepsSinceModel(messages)).toBe(2);
	});

	it("counts from the user's message at the start of a session", () => {
		expect(stepsSinceModel([{ role: "user", content: "task", timestamp: 0 }] as Message[])).toBe(0);
	});
});

describe("createScoutStreamFn", () => {
	let cwd: string;
	let tracePath: string;

	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "jeff-first-scout-"));
		writeFileSync(join(cwd, "README.md"), "The bug is in main.py.");
		tracePath = join(cwd, "trace.jsonl");
	});
	afterEach(() => {
		rmSync(cwd, { recursive: true, force: true });
	});

	function lines() {
		return readFileSync(tracePath, "utf8")
			.trimEnd()
			.split("\n")
			.map((l) => JSON.parse(l));
	}

	function scout(chooser: Chooser, inner: StreamFn) {
		return createScoutStreamFn({
			inner,
			cwd,
			taskId: "t1",
			trace: new TraceWriter(tracePath),
			chooser,
			mode: "teacher",
			isSessionTurn: (id) => id === "s1",
			runApproval: "all",
			driverBuild: "test-build",
		});
	}

	const context = (messages: Message[]) => normalizeContext({ messages: [system(), ...messages] });
	const start = [{ role: "user", content: "Read README.md and fix the bug.", timestamp: 0 }] as Message[];

	it("returns the chosen step as an assistant message from the scout, without calling the model", async () => {
		const { inner, calls } = fakeModel(assistant("local", [{ type: "text", text: "hi" }], "stop"));
		const stream = await scout(scripted(["read"]), inner)(model, context(start), { sessionId: "s1" });
		const message = await stream.result();
		expect(calls.count).toBe(0);
		expect(message.provider).toBe(JEFF_PROVIDER);
		expect(message.stopReason).toBe("toolUse");
		const toolCall = message.content.find((part) => part.type === "toolCall");
		expect(toolCall).toMatchObject({
			type: "toolCall",
			name: "bash",
			arguments: { command: `cat '${join(cwd, "README.md")}'`, timeout: 60 },
		});
		expect(lines()[0]).toMatchObject({
			schema: "jeff-first-trace/3",
			kind: "decision",
			decision: 1,
			step_in_stint: 0,
			driver: "qwen",
			driver_build: "test-build",
			run_approval: "all",
			action: { kind: "step" },
			levels: [
				{ level: "tool", page: 1 },
				{ level: "argument", page: 1, tool: "read" },
			],
		});
	});

	it("starts the step's message with a notice when the coding model has not taken a turn yet", async () => {
		const { inner } = fakeModel(assistant("local", [{ type: "text", text: "hi" }], "stop"));
		const stream = await scout(scripted(["read"]), inner)(model, context(start), { sessionId: "s1" });
		const message = await stream.result();
		expect(message.content).toEqual([
			{ type: "text", text: BEFORE_FIRST_MODEL_TURN_NOTICE },
			{
				type: "toolCall",
				id: expect.any(String),
				name: "bash",
				arguments: { command: `cat '${join(cwd, "README.md")}'`, timeout: 60 },
			},
		]);
	});

	it("omits the notice once the coding model has already taken a turn", async () => {
		const { inner } = fakeModel(assistant("local", [{ type: "text", text: "hi" }], "stop"));
		const messages = [...start, assistant("local", [{ type: "text", text: "first turn" }], "stop")];
		const stream = await scout(scripted(["read"]), inner)(model, context(messages), { sessionId: "s1" });
		const message = await stream.result();
		expect(message.content).toEqual([
			{
				type: "toolCall",
				id: expect.any(String),
				name: "bash",
				arguments: { command: `cat '${join(cwd, "README.md")}'`, timeout: 60 },
			},
		]);
	});

	it("hands over when the teacher chooses to, and logs the model's turn", async () => {
		const reply = assistant("local", [{ type: "toolCall", id: "r1", name: "bash", arguments: { command: "ls" } }]);
		const { inner, calls } = fakeModel(reply);
		const stream = await scout(scripted(["hand_over"]), inner)(model, context(start), { sessionId: "s1" });
		expect((await stream.result()).content).toEqual(reply.content);
		expect(calls.count).toBe(1);
		expect(lines().map((l) => l.kind)).toEqual(["decision", "model_turn"]);
		expect(lines()[0].action).toEqual({ kind: "hand_over", why: "chosen" });
		expect(lines()[1]).toMatchObject({ turn: 1, action: { tool_calls: [{ name: "bash" }] } });
	});

	it(`hands over without asking after ${STEP_CAP} scout steps`, async () => {
		const chooser = scripted(["read"]);
		const { inner, calls } = fakeModel(assistant("local", [{ type: "text", text: "done" }], "stop"));
		const messages = [...start];
		for (let i = 0; i < STEP_CAP; i++) messages.push(assistant(JEFF_PROVIDER, []));
		await (await scout(chooser, inner)(model, context(messages), { sessionId: "s1" })).result();
		expect(chooser.asked).toHaveLength(0);
		expect(calls.count).toBe(1);
		expect(lines()[0]).toMatchObject({
			step_in_stint: STEP_CAP,
			levels: [],
			action: { kind: "hand_over", why: "cap" },
		});
	});

	it("fails the turn with a JeffFirst error when the teacher fails, and never calls the model", async () => {
		const { inner, calls } = fakeModel(assistant("local", [], "stop"));
		const stream = await scout(scripted(new Error("the teacher model at http://x answered 500: busy")), inner)(
			model,
			context(start),
			{ sessionId: "s1" },
		);
		const final = await stream.result();
		expect(final.stopReason).toBe("error");
		expect(final.errorMessage).toMatch(
			/^JeffFirst: the scout could not decide at decision 1: the teacher model at http:\/\/x answered 500/,
		);
		expect(calls.count).toBe(0);
	});

	it("passes requests that are not agent turns straight to the model", async () => {
		const chooser = scripted(["read"]);
		const { inner, calls } = fakeModel(assistant("local", [{ type: "text", text: "summary" }], "stop"));
		await (await scout(chooser, inner)(model, context(start), { sessionId: undefined })).result();
		expect(calls.count).toBe(1);
		expect(chooser.asked).toHaveLength(0);
	});

	it("hands over when the teacher picks None of these, and records why", async () => {
		const { inner, calls } = fakeModel(assistant("local", [{ type: "text", text: "ok" }], "stop"));
		await (
			await scout(scripted(["read", "none_of_these"]), inner)(model, context(start), { sessionId: "s1" })
		).result();
		expect(calls.count).toBe(1);
		expect(lines()[0].action).toEqual({ kind: "hand_over", why: "none_of_these" });
		expect(lines()[0].levels.map((l: { level: string }) => l.level)).toEqual(["tool", "argument"]);
	});

	it("asks the next page after Show more options, and opens the argument step on the page the tool was picked", async () => {
		for (let i = 0; i < 14; i++) writeFileSync(join(cwd, `f${i}.txt`), "x");
		const task = [
			{ role: "user", content: Array.from({ length: 14 }, (_, i) => `f${i}.txt`).join(" "), timestamp: 0 },
		] as Message[];
		const chooser = scripted(["show_more", "read", "read-11"]);
		const { inner } = fakeModel(assistant("local", [], "stop"));
		const message = await (await scout(chooser, inner)(model, context(task), { sessionId: "s1" })).result();
		expect(chooser.asked.map((l) => `${l.level}:${l.page}`)).toEqual(["tool:1", "tool:2", "argument:2"]);
		expect(chooser.asked[2].options.map((o) => o.id)).toContain("read-11");
		expect(chooser.asked[2].options.map((o) => o.id)).not.toContain("read-1");
		expect(message.content.find((part) => part.type === "toolCall")).toMatchObject({
			type: "toolCall",
			name: "bash",
			arguments: { command: `cat '${join(cwd, "f10.txt")}'` },
		});
		expect(lines()[0].levels.map((l: { page: number }) => l.page)).toEqual([1, 2, 2]);
	});
});
