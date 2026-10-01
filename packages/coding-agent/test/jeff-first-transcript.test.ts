import type { AssistantMessage, Message, ToolResultMessage } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { activeToolNames, collectSteps, taskText } from "../src/core/jeff-first/transcript.ts";

const usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistantCalls(
	...calls: Array<{ id: string; name: string; arguments: Record<string, string> }>
): AssistantMessage {
	return {
		role: "assistant",
		content: calls.map((c) => ({ type: "toolCall" as const, ...c })),
		api: "openai-completions",
		provider: "test",
		model: "test",
		usage,
		stopReason: "toolUse",
		timestamp: 0,
	};
}

function result(toolCallId: string, text: string, isError = false): ToolResultMessage {
	return { role: "toolResult", toolCallId, toolName: "bash", content: [{ type: "text", text }], isError, timestamp: 0 };
}

const tool = (name: string) => ({ name, description: name, parameters: { type: "object" } as never });

describe("taskText", () => {
	it("returns the first user message as text", () => {
		const messages: Message[] = [
			{ role: "system", content: "You are pi.", timestamp: 0 },
			{ role: "user", content: [{ type: "text", text: "Fix the tests." }], timestamp: 0 },
			{ role: "user", content: "later message", timestamp: 0 },
		];
		expect(taskText(messages)).toBe("Fix the tests.");
	});

	it("throws when there is no user message", () => {
		expect(() => taskText([{ role: "system", content: "x", timestamp: 0 }])).toThrow(/no user message/);
	});
});

describe("activeToolNames", () => {
	it("replays tools added and removed by system messages", () => {
		const messages: Message[] = [
			{ role: "system", content: "x", toolsAdded: [tool("read"), tool("bash"), tool("ls")], timestamp: 0 },
			{ role: "system", content: "", toolsRemoved: [{ name: "ls" }], timestamp: 0 },
		];
		expect([...activeToolNames(messages)].sort()).toEqual(["bash", "read"]);
	});
});

describe("collectSteps", () => {
	it("pairs each tool call with its result, oldest first", () => {
		const messages: Message[] = [
			{ role: "user", content: "task", timestamp: 0 },
			assistantCalls({ id: "a", name: "bash", arguments: { command: "ls" } }),
			result("a", "main.py"),
			assistantCalls(
				{ id: "b", name: "read", arguments: { path: "main.py" } },
				{ id: "c", name: "bash", arguments: { command: "pytest" } },
			),
			result("b", "print(1)"),
			result("c", "1 failed", true),
		];
		expect(collectSteps(messages)).toEqual([
			{
				call: { type: "toolCall", id: "a", name: "bash", arguments: { command: "ls" } },
				output: "main.py",
				isError: false,
			},
			{
				call: { type: "toolCall", id: "b", name: "read", arguments: { path: "main.py" } },
				output: "print(1)",
				isError: false,
			},
			{
				call: { type: "toolCall", id: "c", name: "bash", arguments: { command: "pytest" } },
				output: "1 failed",
				isError: true,
			},
		]);
	});

	it("marks a call without a recorded result with output null", () => {
		const messages: Message[] = [assistantCalls({ id: "a", name: "bash", arguments: { command: "sleep 99" } })];
		expect(collectSteps(messages)[0].output).toBeNull();
	});
});
