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
import { createScoutStreamFn, JEFF_PROVIDER, STEP_CAP, stepsSinceModel } from "../src/core/jeff-first/scout.ts";
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

/** A chooser that answers each level from a script and records what it was asked. */
function scripted(answers: { tool: string; argument?: string } | Error): Chooser & { asked: Level[] } {
	const asked: Level[] = [];
	return {
		name: "teacher:fake",
		asked,
		async choose(_state, level): Promise<Choice> {
			asked.push(level);
			if (answers instanceof Error) throw answers;
			const optionId = level.level === "tool" ? answers.tool : (answers.argument ?? level.options[0].id);
			return { optionId, shares: { [optionId]: 1 }, picks: [{ optionId, reason: "scripted" }] };
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
			isSessionTurn: (id) => id === "s1",
		});
	}

	const context = (messages: Message[]) => normalizeContext({ messages: [system(), ...messages] });
	const start = [{ role: "user", content: "Read README.md and fix the bug.", timestamp: 0 }] as Message[];

	it("returns the chosen step as an assistant message from the scout, without calling the model", async () => {
		const { inner, calls } = fakeModel(assistant("local", [{ type: "text", text: "hi" }], "stop"));
		const stream = await scout(scripted({ tool: "read" }), inner)(model, context(start), { sessionId: "s1" });
		const message = await stream.result();
		expect(calls.count).toBe(0);
		expect(message.provider).toBe(JEFF_PROVIDER);
		expect(message.stopReason).toBe("toolUse");
		expect(message.content[0]).toMatchObject({
			type: "toolCall",
			name: "read",
			arguments: { path: join(cwd, "README.md") },
		});
		expect(lines()[0]).toMatchObject({
			kind: "decision",
			decision: 1,
			step_in_stint: 0,
			driver: "qwen",
			action: { kind: "step" },
		});
	});

	it("hands over when the teacher chooses to, and logs the model's turn", async () => {
		const reply = assistant("local", [{ type: "toolCall", id: "r1", name: "bash", arguments: { command: "ls" } }]);
		const { inner, calls } = fakeModel(reply);
		const stream = await scout(scripted({ tool: "hand_over" }), inner)(model, context(start), { sessionId: "s1" });
		expect((await stream.result()).content).toEqual(reply.content);
		expect(calls.count).toBe(1);
		expect(lines().map((l) => l.kind)).toEqual(["decision", "model_turn"]);
		expect(lines()[0].action).toEqual({ kind: "hand_over", why: "chosen" });
		expect(lines()[1]).toMatchObject({ turn: 1, action: { tool_calls: [{ name: "bash" }] } });
	});

	it(`hands over without asking after ${STEP_CAP} scout steps`, async () => {
		const chooser = scripted({ tool: "read" });
		const { inner, calls } = fakeModel(assistant("local", [{ type: "text", text: "done" }], "stop"));
		const messages = [...start];
		for (let i = 0; i < STEP_CAP; i++) messages.push(assistant(JEFF_PROVIDER, []));
		await (await scout(chooser, inner)(model, context(messages), { sessionId: "s1" })).result();
		expect(chooser.asked).toHaveLength(0);
		expect(calls.count).toBe(1);
		expect(lines()[0]).toMatchObject({
			step_in_stint: STEP_CAP,
			tool_level: null,
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
		const chooser = scripted({ tool: "read" });
		const { inner, calls } = fakeModel(assistant("local", [{ type: "text", text: "summary" }], "stop"));
		await (await scout(chooser, inner)(model, context(start), { sessionId: undefined })).result();
		expect(calls.count).toBe(1);
		expect(chooser.asked).toHaveLength(0);
	});
});
